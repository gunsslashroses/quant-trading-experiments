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

### Local

```bash
git clone https://github.com/gunsslashroses/quant-trading-experiments.git
cd quant-trading-experiments

# Create virtual environment and install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# For ML notebook (04): install ML extras
pip install -e ".[dev,ml]"

# Place your JKP data file
cp /path/to/jkp_data.csv data/jkp_data.csv

# Start Jupyter
pip install jupyterlab
jupyter lab
```

### Google Colab

Each notebook includes a setup cell that automatically:
1. Clones the repo and installs the `quant_trading` package
2. Mounts Google Drive (for the JKP data file)

Just open any notebook in Colab and run the first cell. Update the `JKP_CSV_PATH` variable in the setup cell to point to your data file on Drive (e.g. `/content/drive/MyDrive/jkp_data.csv`).

## Notebooks Overview

### 02 — Single-Factor Strategies
Grid search across **832 configurations**: 13 characteristics × 4 percentile pairs × 4 weight schemes × 4 max-weight caps. Top strategies ranked by Sharpe ratio. Key finding: short-term reversal (`ret_1_0`) dominates with Sharpe ~3.4.

### 03 — Combined Signal Methods
Three ways to combine all 13 signals:
- **M1 (Consensus Voting)**: n-out-of-13 threshold → best Sharpe ~0.96
- **M2 (Equal-Weight Composite)**: rank-scaled signals → best Sharpe ~1.91
- **M3 (IC-Weighted Composite)**: rolling-IC weights → best Sharpe ~1.49

### 04 — ML Return Prediction
Seven models trained on 1970–2015, tested 2016+. Four tunable models use **[Optuna](https://doi.org/10.1145/3292500.3330701)** (Bayesian optimization with TPE sampler) for hyperparameter tuning with 5-fold temporal cross-validation. See `quant_trading.tuning` for the reusable tuning functions.

The notebook includes a **detailed background section** explaining the RBF kernel, Nystrom approximation, Ridge regression pipeline, and how Optuna compares to grid search — with intuition and mental models for each concept.

| Model | HP Tuning | OOS R² | Best Portfolio Sharpe |
|-------|-----------|--------|-----------------------|
| Linear Regression | — | 0.004 | 3.39 |
| RBF Kernel Ridge | Optuna (80 trials) | 0.005 | 5.20 |
| Random Forest | Optuna (100 trials) | 0.007 | 6.25 |
| Deep Neural Network | Optuna (50 trials) | — | — |
| AdaBoost | Optuna (100 trials) | — | — |
| Max Sharpe Regression | — | — | — |
| IPCA | — | 0.004 | 3.33 |

## Running Tests

```bash
pytest tests/ -v
```

## Linting

```bash
ruff check src/ tests/
ruff format --check src/ tests/
```
