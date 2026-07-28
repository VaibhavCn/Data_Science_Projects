import numpy as np
import pandas as pd
import lightgbm as lgb
from src.config import cfg
from src.features import FEATURE_COLS, CAT_COLS


PARAM_GRID = [
    {'n_estimators': 100,  'learning_rate': 0.1,  'num_leaves': 31,  'min_child_samples': 20},
    {'n_estimators': 300,  'learning_rate': 0.05, 'num_leaves': 63,  'min_child_samples': 50},
    {'n_estimators': 500,  'learning_rate': 0.05, 'num_leaves': 127, 'min_child_samples': 100},
    {'n_estimators': 500,  'learning_rate': 0.03, 'num_leaves': 63,  'min_child_samples': 50,
     'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.8},
]


# ── MAP@K (Kaggle-standard) ───────────────────────────────────────────────────
def apk(actual, predicted, k=cfg.FINAL_K):
    """
    Average precision at k for one customer.
    actual    : set of article_ids actually purchased that week (from full tx)
    predicted : list of up to k article_ids, ranked best-first
    FIX: denominator is min(len(actual), k) — matches Kaggle competition metric.
    Previous version used labels.sum() which is wrong for customers who bought
    multiple items but only had some in the candidate pool.
    """
    predicted = predicted[:k]
    hits, score = 0, 0.0
    seen = set()
    for i, p in enumerate(predicted):
        if p in actual and p not in seen:
            hits += 1
            score += hits / (i + 1)
        seen.add(p)
    return score / min(len(actual), k)


def map_at_k(pred_dict, truth_dict, k=cfg.FINAL_K):
    """
    pred_dict  : {customer_id: [top-k article_ids]}   from rank_and_score
    truth_dict : {customer_id: set of article_ids}    from full tx for target week
    Returns mean AP@k across all customers in truth_dict.
    """
    return float(np.mean([
        apk(truth_dict[c], pred_dict.get(c, []), k)
        for c in truth_dict
    ]))


# ── Fit one LGBMRanker ────────────────────────────────────────────────────────
def fit_ranker(train_df, params):
    """
    train_df: must have customer_id, all FEATURE_COLS, and label column.
    FIX: sorts by customer_id first — groups array must align with row order.
    Without this sort, groupby.size() would count the right numbers but
    the rows themselves would be in wrong order, misaligning every group.
    """
    d = train_df.sort_values('customer_id').reset_index(drop=True)
    groups = d.groupby('customer_id', sort=False).size().values

    model = lgb.LGBMRanker(
        objective='lambdarank',
        metric='map',
        random_state=cfg.RANDOM_STATE,
        n_jobs=cfg.N_JOBS,
        verbose=-1,
        **params
    )
    model.fit(
        d[FEATURE_COLS],
        d['label'],
        group=groups,
        categorical_feature=CAT_COLS,
    )
    return model


# ── Score + rank + compute MAP@12 ─────────────────────────────────────────────
def rank_and_score(model, df, tx, target_week):
    """
    Scores df, ranks candidates per customer, computes MAP@12.
    Truth comes from FULL tx (not just candidate pool labels) so the
    denominator in apk reflects total actual purchases, not just recalled ones.
    Returns (map_score, pred_dict) where pred_dict = {customer_id: [top12]}.
    """
    df = df.copy()
    df['score'] = model.predict(df[FEATURE_COLS])
    df = df.sort_values(['customer_id', 'score'], ascending=[True, False])

    pred = (df.groupby('customer_id', sort=False)['article_id']
              .apply(lambda s: s.head(cfg.FINAL_K).tolist())
              .to_dict())

    truth = (tx[tx['week'] == target_week]
             .groupby('customer_id')['article_id']
             .apply(set)
             .to_dict())
    # only score customers we actually predicted for
    truth = {c: v for c, v in truth.items() if c in pred}

    return map_at_k(pred, truth), pred


# ── Candidate recall diagnostic ───────────────────────────────────────────────
def candidate_recall(cand, tx, target_week, buyers):
    """
    What fraction of actual purchases made it into the candidate pool at all?
    FIX: previous version divided positives/positives = always 1.0.
    Correct version: for each buyer, intersect their candidates with their
    actual purchases, divide by number of actual purchases.
    """
    truth = (tx[tx['week'] == target_week]
             .groupby('customer_id')['article_id']
             .apply(set))
    truth = truth[truth.index.isin(set(buyers))]
    cand_sets = cand.groupby('customer_id')['article_id'].apply(set)
    recalls = []
    for c in truth.index:
        if c in cand_sets.index:
            recalls.append(len(truth[c] & cand_sets[c]) / len(truth[c]))
    return float(np.mean(recalls)) if recalls else 0.0