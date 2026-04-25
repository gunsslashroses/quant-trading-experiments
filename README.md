# Quant Trading Experiments

Systematic exploration of cross-sectional equity return predictability using factor models, signal combination methods, and machine learning.

## Project Structure

```
├── src/quant_trading/        # Reusable library modules
│   ├── data.py               # Data loading, cleaning, winsorization, feature prep
│   ├── factors.py            # FF factor download & comparison utilities
│   ├── signals.py            # Signal generation, scaling, IC weights
│   ├── portfolio.py          # Portfolio return engines (classic + generalized)
│   ├── strategies.py         # High-level strategy runners (M1/M2/M3)
│   ├── tuning.py             # Optuna-based Bayesian HP tuning (DNN, AdaBoost, etc.)
│   ├── evaluation.py         # Performance metrics, OOS R², decile analysis
│   └── plotting.py           # Visualization helpers
│
├── notebooks/                # Exploratory analysis
│   ├── 02_single_factor_strategies.ipynb # 676-config grid search across 13 chars
│   ├── 03_combined_signal_methods.ipynb  # Consensus / Composite / IC-weighted
│   └── 04_ml_return_prediction.ipynb     # 7 ML models for return prediction
│
├── tests/                    # Unit tests for library modules
├── docs/                     # Extended documentation
│   ├── METHODOLOGY.md        # Detailed methodology notes
│   ├── CODE_REVIEW.md        # Code review notes and known issues
│   └── BLOGPOST_DATA_AND_SIGNALS.md  # Substack-style writeup
└── pyproject.toml            # Project config & dependencies
```

## Data

**JKP Global Factors** dataset from WRDS (`contrib.global_factor`), filtered to US common stocks. ~5M stock-month observations, 1970–2025.

## Setup

```bash
git clone https://github.com/gunsslashroses/quant-trading-experiments.git
cd quant-trading-experiments
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# For ML notebook (04): install ML extras
pip install -e ".[dev,ml]"

# Place JKP data
cp /path/to/jkp_data.csv .
```

## Notebooks

### 02 — Single-Factor Strategies

Grid search across **676 configurations**: 13 characteristics × 4 percentile pairs × 4 weight schemes (with conditional weight caps). Micro-cap stocks dropped. Top strategies ranked by Sharpe ratio.

**Key result**: Size (`market_equity`) dominates with Sharpe ~2.26 (5/95 percentile, char_rank_weighted). Short-term reversal (`ret_1_0`) follows at Sharpe ~2.15.

### 03 — Combined Signal Methods

Three ways to combine all 13 signals:
- **M1 (Consensus Voting)**: n-out-of-13 threshold
- **M2 (Equal-Weight Composite)**: rank-scaled signals
- **M3 (IC-Weighted Composite)**: rolling-IC weights

### 04 — ML Return Prediction

Seven models trained on pre-2016 data, tested 2016+. Tunable models use **[Optuna](https://doi.org/10.1145/3292500.3330701)** (Bayesian optimization with TPE sampler) for HP tuning with temporal CV. Data saved as parquet splits to manage memory.

| Model | HP Tuning | OOS R² | Sign Acc |
|-------|-----------|--------|----------|
| Linear Regression | — | 0.001 | 50.1% |
| RBF Kernel Ridge | Optuna (50 trials) | 0.003 | 50.9% |
| Random Forest | Optuna (40 trials) | 0.050 | 51.1% |
| DNN v2 (L2 + Huber) | Optuna (20 trials) | 0.004 | 52.4% |
| AdaBoost v2 | Optuna | — | — |
| MSRR | — | -1.45 | 51.1% |
| IPCA | — | 0.001 | 50.2% |

## Running Tests

```bash
pytest tests/ -v
ruff check src/ tests/
```
