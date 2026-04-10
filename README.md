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
│   ├── evaluation.py         # Performance metrics, OOS R², decile analysis
│   └── plotting.py           # Visualization helpers
│
├── notebooks/                # Exploratory analysis (run in order)
│   ├── 02_single_factor_strategies.ipynb # 832-config grid search across 13 chars
│   ├── 03_combined_signal_methods.ipynb  # Consensus / Composite / IC-weighted
│   └── 04_ml_return_prediction.ipynb     # 7 ML models for return prediction
│
├── tests/                    # Unit tests for library modules
├── data/                     # Data files (not tracked in git)
├── docs/                     # Extended documentation
│   ├── METHODOLOGY.md        # Detailed methodology notes
│   └── CODE_REVIEW.md        # Code review notes and known issues
└── pyproject.toml            # Project config & dependencies
```

## Data

This project uses the **JKP Global Factors** dataset from WRDS. The data file (`jkp_data.csv`) is not included in the repository due to licensing. To use:

1. Download from WRDS using the SQL query in `quant_trading.data.JKP_SQL_TEMPLATE`
2. Save as `data/jkp_data.csv`

## Setup

```bash
# Create virtual environment and install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Or with uv (faster)
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

For ML notebooks (04), you'll also need:
```bash
pip install tensorflow torch ipca
```

## Notebooks Overview

### 02 — Single-Factor Strategies
Grid search across **832 configurations**: 13 characteristics × 4 percentile pairs × 4 weight schemes × 4 max-weight caps. Top strategies ranked by Sharpe ratio. Key finding: short-term reversal (`ret_1_0`) dominates with Sharpe ~3.4.

### 03 — Combined Signal Methods
Three ways to combine all 13 signals:
- **M1 (Consensus Voting)**: n-out-of-13 threshold → best Sharpe ~0.96
- **M2 (Equal-Weight Composite)**: rank-scaled signals → best Sharpe ~1.91
- **M3 (IC-Weighted Composite)**: rolling-IC weights → best Sharpe ~1.49

### 04 — ML Return Prediction
Seven models trained on 1970–2015, tested 2016+:

| Model | OOS R² | Best Portfolio Sharpe |
|-------|--------|-----------------------|
| Linear Regression | 0.004 | 3.39 |
| RBF Kernel Ridge | 0.005 | 5.20 |
| Random Forest | 0.007 | 6.25 |
| Deep Neural Network | — | — |
| AdaBoost | — | — |
| Max Sharpe Regression | — | — |
| IPCA | 0.004 | 3.33 |

## Running Tests

```bash
pytest tests/ -v
```

## Linting

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```
