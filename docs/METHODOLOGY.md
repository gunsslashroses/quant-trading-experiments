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

1. **Linear Regression** — baseline OLS.
2. **RBF Kernel Ridge** — Nystroem approximation (300 components) + Ridge regression; captures nonlinearities without O(N³) kernel computation.
3. **Random Forest** — 100 trees, max depth 6.
4. **Deep Neural Network** — 2-hidden-layer NN with BatchNorm, Dropout, and L1 regularization. Hyperparameters (L1 strength, learning rate, dropout rate, layer widths) tuned via Optuna with 5-fold temporal CV.
5. **AdaBoost** — boosted decision trees. Hyperparameters (max_depth, n_estimators, learning_rate) tuned via Optuna with 5-fold temporal CV.
6. **Max Sharpe Ratio Regression (MSRR)** — linear model trained with custom PyTorch loss that maximizes in-sample portfolio Sharpe ratio.
7. **IPCA** — Instrumented PCA (Kelly, Pruitt, Su 2019); latent factor model where factor loadings are linear functions of observable characteristics.

### Hyperparameter Tuning

All tunable models (DNN, AdaBoost) use **Optuna** (Bayesian optimization with the TPE sampler) instead of manual grid search. This is the modern best practice because:
- The TPE sampler builds a probabilistic model of which HP regions produce good scores, and focuses subsequent trials on the most promising regions — the same concept as iterative coarse-to-fine refinement, but automatic and principled.
- Mixed parameter types (log-scale floats, integers, categoricals) are handled natively.
- The search is reproducible via a seeded sampler.

Cross-validation for all HP tuning uses `TimeSeriesSplit(n_splits=5)` to prevent look-ahead bias. The final model is trained with a **temporal validation split** (first 80% of training data for fitting, last 20% for early stopping) — never a random split.

### Portfolio Evaluation
Each model's predictions are used to sort stocks into long/short portfolios. Evaluated across multiple percentile/weight-scheme configurations.
