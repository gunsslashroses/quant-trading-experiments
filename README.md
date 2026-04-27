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

## Running Tests

```bash
pytest tests/ -v
ruff check src/ tests/
```
