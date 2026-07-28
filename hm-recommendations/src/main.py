"""
main.py — H&M two-stage recommender, end-to-end run.
Uses actual function names from candidates.py and features.py.
Matches the notebook's logic exactly.
"""
import gc
import time
import numpy as np
import pandas as pd

from src.config import cfg
from src.data_loader import load_data
from src.candidates import (
    generate_bestsellers,
    generate_seasonal_bestsellers,
    generate_repurchases,
    create_i2i_map,
    generate_i2i_candidates,
    generate_index_group_bestsellers,
    generate_cf_candidates,
    merge_candidates,
)
from src.features import build_features, FEATURE_COLS, CAT_COLS
from src.model import fit_ranker, rank_and_score, candidate_recall

t_start = time.time()
rng = np.random.default_rng(cfg.RANDOM_STATE)

SOURCES = ['s1_recent', 's2_seasonal', 's3_repurchase',
           's4_i2i', 's5_idx', 's6_cf8', 's6_cf52']

GRID = [
    dict(n_estimators=100, learning_rate=0.10, num_leaves=31,  min_child_samples=20),
    dict(n_estimators=300, learning_rate=0.05, num_leaves=63,  min_child_samples=50),
    dict(n_estimators=500, learning_rate=0.05, num_leaves=127, min_child_samples=100),
    dict(n_estimators=500, learning_rate=0.03, num_leaves=63,  min_child_samples=50,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8),
]


# ── Labels ────────────────────────────────────────────────────────────────────
def make_labels(cand, tx, target_week):
    truth = (tx[tx['week'] == target_week]
             [['customer_id', 'article_id']]
             .drop_duplicates())
    truth['label'] = np.int8(1)
    out = cand.merge(truth, on=['customer_id', 'article_id'], how='left')
    out['label'] = out['label'].fillna(0).astype(np.int8)
    return out


# ── Build one week's full dataset ─────────────────────────────────────────────
def build_week(tx, customers, articles, target_week,
               sample_customers=None, tag=""):
    """
    For a given target_week:
      1. hist = tx strictly before target_week  (temporal leakage guard)
      2. buyers = customers who purchased during target_week
      3. generate all 6 candidate sources from hist
      4. merge candidates into one DataFrame
      5. attach ground-truth labels
      6. build all features from hist
    Returns a single DataFrame ready for fit_ranker or rank_and_score.
    """
    print(f"\n=== {tag} week {target_week} ===")
    hist = tx[tx['week'] < target_week].copy()
    buyers = tx.loc[tx['week'] == target_week, 'customer_id'].unique()

    if sample_customers is not None and len(buyers) > sample_customers:
        buyers = rng.choice(buyers, size=sample_customers, replace=False)
    print(f"  hist rows {len(hist):,} | customers scored {len(buyers):,}")

    week_number = int(target_week.split('-')[1])

    # ── generate all candidate sources ──
    t = time.time()
    sources = {}
    sources['s1_recent'] = generate_bestsellers(hist, list(buyers))
    sources['s2_seasonal'] = generate_seasonal_bestsellers(hist, list(buyers), week_number)
    sources['s3_repurchase'] = generate_repurchases(hist)
    i2i_map = create_i2i_map(hist)
    sources['s4_i2i'] = generate_i2i_candidates(hist, i2i_map)
    sources['s5_idx'] = generate_index_group_bestsellers(hist, articles)
    for w in cfg.CF_WINDOWS:
        key = f's6_cf{w}'
        try:
            sources[key] = generate_cf_candidates(hist, n_weeks=w)
            print(f"  CF {w}wk: {len(sources[key]):,} customers")
        except Exception as ex:
            print(f"  !! CF {w}wk FAILED: {type(ex).__name__}: {ex}")
            sources[key] = {}
    print(f"  sources built in {time.time()-t:.0f}s")

    # ── filter to buyers only and merge ──
    buyer_set = set(buyers)
    filtered_sources = {}
    for name, d in sources.items():
        filtered_sources[name] = {c: v for c, v in d.items() if c in buyer_set}

    cand = merge_candidates(filtered_sources)
    print(f"  candidates {len(cand):,} "
          f"({len(cand)/max(len(buyers), 1):.1f} per customer)")

    # ── attach labels ──
    cand = make_labels(cand, tx, target_week)
    pos = int(cand['label'].sum())
    print(f"  positives {pos:,} ({100*pos/len(cand):.3f}%)")

    # ── candidate recall ──
    recall = candidate_recall(cand, tx, target_week, buyers)
    print(f"  candidate recall {recall:.4f}")

    # ── build features ──
    cand = build_features(cand, hist, customers, articles)

    return cand


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("loading...")
    tx, customers, articles = load_data()
    # add week column (data_loader.load_data doesn't add it)
    import pandas as pd
    tx['t_dat'] = pd.to_datetime(tx['t_dat'])
    tx['week'] = tx['t_dat'].dt.strftime('%Y-%U')
    print(f"  tx {tx.shape} | customers {customers.shape} | "
          f"articles {articles.shape}")

    # ── build train and valid datasets ──
    train_df = build_week(
        tx, customers, articles,
        target_week=cfg.TRAIN_END_WEEK,
        sample_customers=None,
        tag="TRAIN"
    )
    valid_df = build_week(
        tx, customers, articles,
        target_week=cfg.VALIDATION_WEEK,
        tag="VALID"
    )

    # ── hyperparameter tuning on valid week ──
    print("\n=== tuning on VALID week ===")
    results = []
    for i, p in enumerate(GRID):
        model = fit_ranker(train_df, p)
        score, _ = rank_and_score(model, valid_df, tx, cfg.VALIDATION_WEEK)
        results.append((score, p))
        print(f"  [{i}] MAP@{cfg.FINAL_K} = {score:.5f}  {p}")

    best_score, best_params = max(results, key=lambda x: x[0])
    print(f"\nbest on valid: {best_score:.5f} -> {best_params}")

    # ── feature importance ──
    print("\nfeature importance (best config, trained on TRAIN week):")
    best_model = fit_ranker(train_df, best_params)
    imp = (pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
           .sort_values(ascending=False))
    print(imp.to_string())

    # ── final: retrain on valid, evaluate on test ONCE ──
    print("\n=== FINAL: retrain on VALID week, evaluate on TEST week ===")
    final_model = fit_ranker(valid_df, best_params)
    test_df = build_week(
        tx, customers, articles,
        target_week=cfg.TEST_WEEK,
        tag="TEST"
    )
    test_map, predictions = rank_and_score(
        final_model, test_df, tx, cfg.TEST_WEEK
    )
    print(f"\nTEST MAP@{cfg.FINAL_K} ({cfg.TEST_WEEK}) = {test_map:.5f}")
    print(f"total runtime {(time.time()-t_start)/60:.1f} min")