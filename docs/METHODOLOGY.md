# Methodology

## Data Source

**JKP Global Factors** dataset from WRDS (`contrib.global_factor`), filtered to US common stocks (`common=1, excntry='USA'`). Sample period: 1970–2025.

### Variables Used

| Theme | Variable | Description | Expected Sign |
|-------|----------|-------------|---------------|
| Accruals | `oaccruals_at` | Operating accruals / total assets | − |
| Debt issuance | `dbnetis_at` | Net debt issuance / total assets | − |
| Investment | `at_gr1` | 1-year asset growth | − |
| Leverage | `debt_at` | Total debt / total assets | − |
| Low risk | `beta_60m` | 60-month CAPM beta | − |
| Momentum | `ret_6_1` | Cumulative return months t−6 to t−1 | + |
| Profit growth | `niq_at_chg1` | 1-year change in quarterly NI/assets | + |
| Profitability | `ni_be` | Net income / book equity (ROE) | + |
| Quality | `qmj` | Quality-minus-Junk composite score | + |
| Short-term reversal | `ret_1_0` | Prior 1-month return | − |
| Size | `market_equity` | Market capitalization (USD) | − |
| Value | `be_me` | Book-to-market ratio | + |

## Preprocessing

1. **Winsorization**: monthly cross-sectional clipping at 5th/95th percentiles.
2. **Rank standardization**: for ML models, map each feature to [−1, 1] via percentile ranks within each month.
3. **Missing values**: filled with cross-sectional median (per month), then remaining NaN filled with 0.

## Single-Factor Strategies (Notebook 02)

For each characteristic, form long/short portfolios:
- **Long**: stocks above the upper percentile cutoff
- **Short**: stocks below the lower percentile cutoff
- Weight schemes: equal, value, characteristic-rank, characteristic-minmax
- Optional weight cap per leg (prevents concentration)

Grid: 13 chars × 4 percentile pairs × 4 weight schemes × 4 weight caps = 832 configs.

## Combined Signal Methods (Notebook 03)

### Method 1: Consensus Voting
Each of S signals casts a +1 (long) or −1 (short) vote based on its percentile position. A stock enters the long (short) leg when the net vote count ≥ n (≤ −n). Tested with n ∈ {2, 4, 7}.

### Method 2: Equal-Weight Composite
1. Scale each characteristic to [0, 1] within each month (rank or minmax).
2. Map to signed signal: s = 2x − 1 (flipped for negative-sign chars).
3. Average all S signals → composite score.
4. Sort stocks on composite; long top / short bottom percentile.

### Method 3: IC-Weighted Composite
Same as Method 2, but weight each signal by its **rolling out-of-sample Information Coefficient** (Spearman correlation between signal and next-month return over the prior 18 months). This allows the model to upweight signals with recent predictive power.

When `auto_direction_from_ic=True`, the IC sign itself determines the long/short direction, ignoring the pre-specified `long_high` parameter.

## ML Models (Notebook 04)

All models predict `ret_exc_lead1m_w` (winsorized next-month excess return).

### Train/Test Split
- **Train**: ≤ 2015-12-31
- **Test**: ≥ 2016-01-01
- No look-ahead: all cross-validation uses `TimeSeriesSplit`.

### Models

| # | Model | HP Tuning | Search Space |
|---|-------|-----------|--------------|
| 1 | **Linear Regression** | — | Baseline OLS, no tuning. |
| 2 | **RBF Kernel Ridge** | Optuna (80 trials) | `alpha` [1e-2, 1000] log; `gamma` [1e-4, 10] log; `n_components` [100, 500]. Nystroem approximation avoids O(N³) kernel computation. |
| 3 | **Random Forest** | Optuna (100 trials) | `n_estimators` [50, 500] log; `max_depth` [2, 12]; `min_samples_leaf` [5, 200] log; `max_features` [0.1, 1.0]. |
| 4 | **Deep Neural Network** | Optuna (50 trials) | `l1_reg` [1e-6, 1e-1] log; `lr` [1e-5, 1e-2] log; `dropout` [0, 0.5]; `units_1` [32, 128]; `units_2` [16, 64]. 2-hidden-layer NN with BatchNorm + Dropout. |
| 5 | **AdaBoost** | Optuna (100 trials) | `max_depth` [1, 6]; `n_estimators` [50, 1000] log; `learning_rate` [1e-4, 2.0] log. |
| 6 | **MSRR** | — | Custom PyTorch loss maximizing in-sample portfolio Sharpe ratio. |
| 7 | **IPCA** | — | 3 latent factors (Kelly, Pruitt, Su 2019). |

### Hyperparameter Tuning

All tunable models (RBF Kernel Ridge, Random Forest, DNN, AdaBoost) use **Optuna** — a Bayesian optimization framework using the TPE (Tree-structured Parzen Estimator) sampler (Akiba et al., 2019). This replaces the original manual coarse-to-fine grid search and is the modern best practice because:

- **Sample-efficient**: TPE builds a probabilistic model of which HP regions produce good scores, and focuses subsequent trials on the most promising regions — the same concept as iterative coarse-to-fine refinement, but automatic and principled.
- **Mixed parameter types**: log-scale floats, integers, and categoricals are handled natively with correct spacing.
- **Reproducible**: seeded TPE sampler gives deterministic trial sequences.

Cross-validation for all HP tuning uses `TimeSeriesSplit(n_splits=5)` to prevent look-ahead bias. For the DNN, the final model is trained with a **temporal validation split** (first 80% of training data for fitting, last 20% for early stopping) — never a random split.

> **Reference**: Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019).
> *Optuna: A Next-generation Hyperparameter Optimization Framework*.
> Proceedings of the 25th ACM SIGKDD International Conference on Knowledge
> Discovery & Data Mining (KDD), 2623–2631.
> [https://doi.org/10.1145/3292500.3330701](https://doi.org/10.1145/3292500.3330701)

### Portfolio Evaluation
Each model's predictions are used to sort stocks into long/short portfolios. Evaluated across multiple percentile/weight-scheme configurations.
