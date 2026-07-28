import pandas as pd
import numpy as np
from src.config import cfg


# Columns fed to LightGBM as native categoricals.
CAT_COLS = ['product_type_name', 'department_name', 'garment_group_name',
            'colour_group_name', 'index_group_name',
            'club_member_status', 'fashion_news_frequency']


def build_features(cand, hist, df_customers, df_articles, k=cfg.RECENT_WEEKS):
    """
    cand          : merge_candidates output (customer_id, article_id, source flags, n_sources)
    hist          : transactions STRICTLY before the target week
    df_customers  : customers.csv
    df_articles   : articles.csv
    returns       : cand plus feature columns, same row count
    """
    print('building features')
    n_in = len(cand)

    cutoff = hist['t_dat'].max()
    recent_weeks = sorted(hist['week'].unique())[-k:]
    recent = hist[hist['week'].isin(recent_weeks)]

    art_meta = df_articles[['article_id', 'colour_group_name', 'index_group_name',
                            'product_type_name', 'department_name',
                            'garment_group_name']]

    # ---------------- article level ----------------
    art = pd.DataFrame(index=hist['article_id'].unique())
    art.index.name = 'article_id'
    art['art_pop_all'] = hist['article_id'].value_counts()
    art['art_pop_recent'] = recent['article_id'].value_counts()
    art['art_mean_price'] = recent.groupby('article_id')['price'].mean()
    art['art_n_buyers_recent'] = recent.groupby('article_id')['customer_id'].nunique()
    art['art_last_sale_days'] = (cutoff - hist.groupby('article_id')['t_dat'].max()).dt.days
    art = art.fillna(0).reset_index()

    # ---------------- customer level ----------------
    cus = pd.DataFrame(index=hist['customer_id'].unique())
    cus.index.name = 'customer_id'
    cus['cust_n_tx'] = hist['customer_id'].value_counts()
    cus['cust_n_distinct_art'] = hist.groupby('customer_id')['article_id'].nunique()
    cus['cust_mean_price'] = hist.groupby('customer_id')['price'].mean()
    cus['cust_max_price'] = hist.groupby('customer_id')['price'].max()
    cus['cust_last_tx_days'] = (cutoff - hist.groupby('customer_id')['t_dat'].max()).dt.days
    cus = cus.reset_index()

    # ---------------- customer x article level ----------------
    pair = (hist.groupby(['customer_id', 'article_id'])
            .agg(pair_buy_count=('t_dat', 'size'),
                 pair_last=('t_dat', 'max'))
            .reset_index())
    pair['pair_last_days'] = (cutoff - pair['pair_last']).dt.days
    pair = pair.drop(columns='pair_last')

    # ---------------- colour / index affinity ----------------
    h = hist.merge(art_meta[['article_id', 'colour_group_name', 'index_group_name']],
                   on='article_id', how='left')
    tot = h.groupby('customer_id').size().rename('tot')

    col = (h.groupby(['customer_id', 'colour_group_name']).size()
           .rename('cnt').reset_index().merge(tot, on='customer_id'))
    col['colour_share'] = col['cnt'] / col['tot']
    col = col[['customer_id', 'colour_group_name', 'colour_share']]

    idx = (h.groupby(['customer_id', 'index_group_name']).size()
           .rename('cnt').reset_index().merge(tot, on='customer_id'))
    idx['index_share'] = idx['cnt'] / idx['tot']
    idx = idx[['customer_id', 'index_group_name', 'index_share']]

    del h

    # ---------------- merge onto candidates ----------------
    out = cand.merge(art_meta, on='article_id', how='left')
    out = out.merge(art, on='article_id', how='left')
    out = out.merge(cus, on='customer_id', how='left')
    out = out.merge(pair, on=['customer_id', 'article_id'], how='left')
    out = out.merge(col, on=['customer_id', 'colour_group_name'], how='left')
    out = out.merge(idx, on=['customer_id', 'index_group_name'], how='left')
    out = out.merge(df_customers[['customer_id', 'age', 'club_member_status',
                                  'fashion_news_frequency', 'Active', 'FN']],
                    on='customer_id', how='left')

    # ---------------- derived ----------------
    out['price_diff'] = out['art_mean_price'] - out['cust_mean_price']
    out['price_ratio'] = out['art_mean_price'] / out['cust_mean_price'].replace(0, np.nan)

    # ---------------- fills ----------------
    out['pair_buy_count'] = out['pair_buy_count'].fillna(0)
    out['colour_share'] = out['colour_share'].fillna(0)
    out['index_share'] = out['index_share'].fillna(0)
    out['Active'] = out['Active'].fillna(0)
    out['FN'] = out['FN'].fillna(0)
    # 999 not 0: 0 means TODAY, opposite of "never bought this article"
    out['pair_last_days'] = out['pair_last_days'].fillna(999)

    # FIX: bought_before was in the notebook but missing from this module
    # binary flag: 1 if this customer has ever bought this article before
    out['bought_before'] = (out['pair_buy_count'] > 0).astype(np.int8)

    for c in CAT_COLS:
        out[c] = out[c].astype('category')

    if len(out) != n_in:
        raise ValueError(
            f"row count changed in build_features: {n_in:,} -> {len(out):,}. "
            "A lookup table has duplicate keys and fanned out the merge."
        )

    print(f'  features built on {len(out):,} rows')
    return out


FEATURE_COLS = [
    's1_recent', 's2_seasonal', 's3_repurchase', 's4_i2i', 's5_idx',
] + [f's6_cf{w}' for w in cfg.CF_WINDOWS] + ['n_sources',
    'art_pop_all', 'art_pop_recent', 'art_mean_price',
    'art_n_buyers_recent', 'art_last_sale_days',
    'cust_n_tx', 'cust_n_distinct_art', 'cust_mean_price',
    'cust_max_price', 'cust_last_tx_days',
    'pair_buy_count', 'pair_last_days', 'bought_before',
    'colour_share', 'index_share',
    'price_diff', 'price_ratio',
    'age', 'Active', 'FN',
] + CAT_COLS