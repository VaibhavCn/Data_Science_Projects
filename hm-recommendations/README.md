# H&M Personalised Fashion Recommendations — Two-Stage Recommender System

> **MAP@12 = 0.03157** (offline backtest on held-out week 2020-37)

A two-stage recommender system built on the [H&M Personalised Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) dataset. Stage one generates a candidate pool of ~200 articles per customer from six complementary sources; stage two re-ranks that pool with a LightGBM Learning-to-Rank model trained on ~30 hand-crafted features. The evaluation is an offline backtest against a held-out week — not submitted to the Kaggle leaderboard — and the score is comparable to publicly documented silver-medal-tier solutions from the original competition.

## Pipeline

```mermaid
graph LR
    A[Transactions CSV] --> B[Candidate Generation\n6 sources · ~200 per customer]
    B --> C[Feature Engineering\n~30 features]
    C --> D[LGBMRanker\nlambdarank objective]
    D --> E[Top-12 Recommendations]
```

## Results

| Split | Week | Candidate recall | MAP@12 |
|-------|------|-----------------|--------|
| Validation | 2020-36 | 14.49% | 0.02790 |
| Test | 2020-37 | 15.81% | 0.03157 |

## Architecture

### Stage 1 — Candidate Generation

Six sources are generated from transaction history strictly before the target week (no leakage), then merged into a single DataFrame with one-hot source flags:

| Key | Source | Logic |
|-----|--------|-------|
| `s1_recent` | Recent bestsellers | Top-30 most-sold articles in the last 8 weeks, given to every buyer |
| `s2_seasonal` | Seasonal bestsellers | Top-30 from the same calendar weeks in the prior two years |
| `s3_repurchase` | Repurchase history | Each customer's 30 most-recently-bought distinct articles |
| `s4_i2i` | Item-to-item co-occurrence | Co-purchase graph (sparse matrix multiply); top-20 neighbours per article, seeded by each customer's 5 most-recent articles |
| `s5_idx` | Index-group bestsellers | Top-30 per `index_group_name` (Ladieswear, Menswear, …), mapped to each customer's favourite group |
| `s6_cf8` / `s6_cf52` | Collaborative filtering (ALS) | `implicit` ALS at 64 factors, trained on 8-week and 52-week windows; top-30 per customer via matrix multiply |

Sources are filtered to active buyers of the target week, then unioned. Duplicate `(customer_id, article_id)` pairs are collapsed; `n_sources` counts how many sources nominated each pair.

### Stage 2 — LGBMRanker

The candidate DataFrame is enriched with ~30 features and passed to `lgb.LGBMRanker` with `objective='lambdarank'`. Groups are the per-customer candidate counts (rows must be sorted by `customer_id` before fitting).

**Feature groups:**

- **Source flags**: `s1_recent` … `s6_cf52`, `n_sources`
- **Article-level**: `art_pop_all`, `art_pop_recent`, `art_mean_price`, `art_n_buyers_recent`, `art_last_sale_days`
- **Customer-level**: `cust_n_tx`, `cust_n_distinct_art`, `cust_mean_price`, `cust_max_price`, `cust_last_tx_days`
- **Customer × article**: `pair_buy_count`, `pair_last_days`, `bought_before`
- **Affinity**: `colour_share`, `index_share`, `price_diff`, `price_ratio`
- **Demographics**: `age`, `Active`, `FN`
- **Categoricals** (native LightGBM): `product_type_name`, `department_name`, `garment_group_name`, `colour_group_name`, `index_group_name`, `club_member_status`, `fashion_news_frequency`

**Top features by LightGBM split importance:**

| Feature | Importance |
|---------|-----------|
| `department_name` | 4790 |
| `colour_group_name` | 2960 |
| `product_type_name` | 2253 |
| `cust_last_tx_days` | 1703 |
| `age` | 1695 |

Hyperparameters are tuned over a 4-point grid on the validation week; the best config is retrained on the validation week before final test evaluation (single hold-out, no leakage).

## Project Structure

```
hm-recommendations/
├── data/
│   ├── articles.csv
│   ├── customers.csv
│   └── transactions_train.csv
├── notebooks/
│   ├── eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── config.py        # paths and hyperparameter constants
│   ├── data_loader.py   # CSV loading and train/val/test split
│   ├── candidates.py    # all 6 candidate sources + merge
│   ├── features.py      # feature engineering + FEATURE_COLS
│   ├── model.py         # LGBMRanker, MAP@12, candidate recall
│   └── main.py          # end-to-end pipeline
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
PYTHONPATH=. python src/main.py
```

The script trains on week 2020-35, tunes on 2020-36, then evaluates once on 2020-37. Expected runtime is 20–40 minutes on a modern laptop (the 52-week ALS window dominates).
