import pandas as pd
from src.config import cfg
import numpy as np
from scipy.sparse import csr_matrix
import implicit


## Generating top 30 bestsellers
def generate_bestsellers(df, cust_list, k=cfg.RECENT_WEEKS, r=cfg.TOPK_RECENT):
    print('generating bestsellers')
    recent_weeks = sorted(df['week'].unique())[-k:]
    df1 = df[df['week'].isin(recent_weeks)]
    top_30 = df1['article_id'].value_counts().head(r).index.to_list()
    # FIX: was df_customers.customer_id.unique() — generates for ALL customers ever.
    # Now takes cust_list (buyers for this target week only) to match notebook.
    src_recent = {c: list(top_30) for c in cust_list}
    return src_recent


## Seasonal bestsellers
def generate_seasonal_bestsellers(df, cust_list, week_number, span=2, k=cfg.TOPK_SEASONAL):
    print('generating seasonal bestsellers')
    current_year = df.t_dat.dt.year.max()
    seasonal_weeks = [f"{yr}-{w:02d}" for yr in [current_year - 1, current_year - 2]
                      for w in range(week_number - span, week_number + span + 1)]
    available = set(df['week'].unique())
    seasonal_weeks = [w for w in seasonal_weeks if w in available]
    seasonal = df[df['week'].isin(seasonal_weeks)]
    top_30_seasonal = seasonal.article_id.value_counts().head(k).index.to_list()
    # FIX: same as generate_bestsellers — takes cust_list not df_customers
    src_seasonal = {c: list(top_30_seasonal) for c in cust_list}
    return src_seasonal


## Repurchases
def generate_repurchases(df, k=cfg.REPURCHASE_N):
    print('starting repurchase list generation..')
    history = df.sort_values(by='t_dat', ascending=False)

    def recent_purchases(s):
        seen = list(dict.fromkeys(s.to_list()))
        return seen[:k]

    src_repurchase = history.groupby('customer_id')['article_id'].apply(recent_purchases).to_dict()
    return src_repurchase


## item2item co-occurrence map
def create_i2i_map(df, k=cfg.I2I_WEEKS, min_sales=cfg.I2I_MIN_SALES, topk=cfg.I2I_TOPK):
    weeks = sorted(df['week'].unique())[-k:]
    data = df[df['week'].isin(weeks)]

    vc = data.article_id.value_counts()
    keep = set(vc[vc >= min_sales].index)
    d = data[data['article_id'].isin(keep)]

    if len(d) == 0:
        return {}

    customers, _ = pd.factorize(d['customer_id'])
    articles, articles_uniq = pd.factorize(d['article_id'])
    M = csr_matrix((np.ones(len(d), dtype=np.float32), (customers, articles)),
                   shape=(customers.max() + 1, len(articles_uniq)))
    M.data[:] = 1.0
    print(f"Matrix: {M.shape[0]:,} customers x {M.shape[1]:,} articles, {M.nnz:,} nonzeros")
    C = (M.T @ M).tocsr()
    mem_mb = (C.data.nbytes + C.indices.nbytes + C.indptr.nbytes) / 1e6
    print(f"Co-occurrence matrix: {C.nnz:,} nonzero pairs, ~{mem_mb:,.0f} MB")

    i2i_map = {}
    for i in range(C.shape[0]):
        start, end = C.indptr[i], C.indptr[i + 1]
        cols = C.indices[start:end]
        vals = C.data[start:end]
        mask = cols != i
        cols, vals = cols[mask], vals[mask]
        if len(cols) == 0:
            continue
        if len(cols) > topk:
            top_idx = np.argpartition(-vals, topk)[:topk]
            cols, vals = cols[top_idx], vals[top_idx]
        order = np.argsort(-vals)
        i2i_map[articles_uniq[i]] = [articles_uniq[c] for c in cols[order]]
    print(f"Built i2i map for {len(i2i_map):,} articles")
    return i2i_map


## item2item candidates (uses the map from create_i2i_map)
def generate_i2i_candidates(df, i2i_map, seeds=cfg.I2I_SEEDS):
    print('generating i2i candidates')
    history = df.sort_values(by='t_dat', ascending=False)
    seed_arts = (history.groupby('customer_id')['article_id']
                 .apply(lambda s: list(dict.fromkeys(s.to_list()))[:seeds])
                 .to_dict())
    src_i2i = {}
    for cust, arts in seed_arts.items():
        pool = []
        for a in arts:
            pool.extend(i2i_map.get(a, []))
        if pool:
            src_i2i[cust] = list(dict.fromkeys(pool))
    return src_i2i


## index-group bestsellers
def generate_index_group_bestsellers(df, df_articles, k=cfg.RECENT_WEEKS, topn=cfg.IDX_TOP_N):
    print('generating index group bestsellers')
    d = df.merge(df_articles[['article_id', 'index_group_name']], on='article_id', how='left')
    recent_weeks = sorted(d['week'].unique())[-k:]
    recent = d[d['week'].isin(recent_weeks)]

    idx_best = (recent.groupby('index_group_name')['article_id']
                .apply(lambda s: s.value_counts().head(topn).index.to_list())
                .to_dict())

    seg = (d.groupby(['customer_id', 'index_group_name']).size()
           .reset_index(name='cnt')
           .sort_values(['customer_id', 'cnt'], ascending=[True, False])
           .drop_duplicates(subset=['customer_id'], keep='first'))
    fav_idx = dict(zip(seg['customer_id'], seg['index_group_name']))

    src_idx = {c: list(idx_best[g]) for c, g in fav_idx.items() if g in idx_best}
    return src_idx


## collaborative filtering (ALS)
def generate_cf_candidates(df, n_weeks, topn=30, factors=cfg.CF_FACTORS,
                           iters=cfg.CF_ITERS, reg=cfg.CF_REG):
    print(f'generating CF candidates ({n_weeks}wk window)')
    weeks = sorted(df['week'].unique())[-n_weeks:]
    d = df[df['week'].isin(weeks)]

    customers, customers_uniq = pd.factorize(d['customer_id'])
    articles, articles_uniq = pd.factorize(d['article_id'])
    user_items = csr_matrix((np.ones(len(d), dtype=np.float32), (customers, articles)),
                            shape=(len(customers_uniq), len(articles_uniq)))
    user_items.data[:] = 1.0

    model = implicit.als.AlternatingLeastSquares(
        factors=factors, regularization=reg, iterations=iters, random_state=cfg.RANDOM_STATE)
    model.fit(user_items)

    # Manual matrix multiply approach — more reliable across implicit versions
    # than model.recommend() which changed API between versions
    uf = np.asarray(model.user_factors, dtype=np.float32)
    itf = np.asarray(model.item_factors, dtype=np.float32)

    out = {}
    batch = 1000
    for i in range(0, len(customers_uniq), batch):
        chunk_users = customers_uniq[i:i + batch]
        rows = np.arange(i, min(i + batch, len(customers_uniq)))
        scores = uf[rows] @ itf.T
        actual_topn = min(topn, scores.shape[1] - 1)
        top_idx = np.argpartition(-scores, actual_topn, axis=1)[:, :actual_topn]
        for j, c in enumerate(chunk_users):
            cols = top_idx[j]
            out[c] = [articles_uniq[x] for x in cols[np.argsort(-scores[j, cols])]]
        del scores

    return out


## merge all candidate sources into one DataFrame
def merge_candidates(sources):
    print('merging candidates')
    data = []
    for name, d in sources.items():
        if not d:
            print(f'{name} skipped')
            continue
        f = pd.DataFrame({'customer_id': list(d.keys()), 'article_id': list(d.values())})
        f = f.explode('article_id')
        f[name] = np.int8(1)
        data.append(f)
    if not data:
        return pd.DataFrame(columns=['customer_id', 'article_id'])

    cand = data[0]
    for df in data[1:]:
        cand = pd.merge(cand, df, on=['customer_id', 'article_id'], how='outer')

    for name in sources:
        if name not in cand.columns:
            cand[name] = 0
        # FIX: was .astpe(np.int8) — typo that crashes at runtime
        cand[name] = cand[name].fillna(0).astype(np.int8)
    cand['n_sources'] = cand[list(sources)].sum(axis=1).astype(np.int8)
    print(f'  total: {len(cand):,} candidate pairs')
    return cand.reset_index(drop=True)