# Fama-French Portfolio Replication

Public, explanation-first replication of the classic Fama-French 2x3 portfolio
sorts that produce **SMB (Small Minus Big)** and **HML (High Minus Low)**.

The repository is built to answer two questions clearly:

1. **How did Fama and French actually form their portfolios?**
2. **How close can we get using a modern WRDS stock-level dataset rather than
   hand-built CRSP/Compustat merges?**

The main walkthrough lives in `notebooks/05_ff_factor_replication.ipynb`. The
reusable logic lives in `src/quant_trading/factors.py`.

## What this repo does

The replication follows the canonical Fama-French design:

- keep US common stocks on NYSE, AMEX, and NASDAQ,
- compute **NYSE-only** June breakpoints,
- split stocks into **2 size buckets** and **3 book-to-market buckets**,
- value-weight the six portfolios,
- form
  - `SMB = (SH + SM + SL) / 3 - (BH + BM + BL) / 3`
  - `HML = (SH + BH) / 2 - (SL + BL) / 2`
- compare the reconstructed factors with the official Ken French data library.

The notebook is intentionally teaching-oriented: every major code block is tied
back to the economic logic in Fama and French rather than presented as a black
box.

## Dataset

This project uses the **Jensen, Kelly, and Pedersen (2023) global factor
dataset** on WRDS (`contrib.global_factor`), filtered to US stocks.

Why this dataset:

- it is **stock-level**, so we can recreate the underlying sorts rather than
  only download finished factor returns;
- it already contains key firm characteristics such as `market_equity` and
  `be_me`;
- it is standardized and documented, which makes the repo easier to share and
  easier for reviewers to understand.

Why it is still only an approximation:

- the official Fama-French factors are built from CRSP and Compustat with
  specific accounting conventions and share-class aggregation rules;
- JKP gives us harmonized proxies for those inputs, not the exact original
  pipeline;
- the repo therefore aims for a **credible replication**, not a byte-for-byte
  clone.

The raw CSV is not committed because WRDS data is licensed. Put your extract at
`data/jkp_data.csv`.

## Why these columns

The replication only needs a small subset of the JKP table:

- `id`: stock identifier used to track a security through time,
- `month_date` / `eom`: month-end observation date,
- `primary_sec`: keeps the primary security for each firm,
- `exch_main`: exchange code so we can isolate NYSE for breakpoints,
- `source_crsp`: keeps the CRSP-backed observations,
- `market_equity`: size signal and value-weighting variable,
- `be_me`: book-to-market signal for value sorts,
- `ret_exc_lead1m`: next-month excess return used to measure portfolio returns.

That is deliberate: a public repo is easier to audit when the data inputs are
minimal and explicit.

## Repo map

```text
src/quant_trading/factors.py              FF replication pipeline
notebooks/05_ff_factor_replication.ipynb  Explanation-first walkthrough
tests/test_factors.py                     Unit tests for timing and formulas
data/README.md                            Data instructions
```

## Methodology choices

### 1. Universe filters

The notebook keeps primary securities, major exchanges, positive market equity,
and positive book-to-market. This mirrors the spirit of the original filters:
remove obvious non-common-share noise before sorting.

### 2. NYSE breakpoints

This is one of the most important design choices in the whole repo. If you use
all exchanges to set breakpoints, the large mass of tiny NASDAQ firms can move
the cutoffs materially. Fama and French use **NYSE-only** breakpoints to keep
the size and value partitions anchored to a stable reference market.

### 3. June rebalancing

Fama and French rebalance once per year in June. The reason is accounting-data
timing: book equity comes from annual statements, so the sorts should not peek
into information that would not have been available at the portfolio formation
date.

### 4. Lead-return timing

The JKP return column in this repo is `ret_exc_lead1m`, meaning the return from
month `t` to `t+1` is stored on the row dated `t`. That is convenient, but it
creates an easy off-by-one trap. The implementation in `factors.py` explicitly
maps **June month-dated rows to the current sort year**, because those rows
already contain the July realized return.

### 5. Expected differences vs. the official factors

Do not expect a perfect match. The most important differences are:

- book equity is proxied through JKP's `be_me`,
- market equity aggregation is not the exact Ken French share-class procedure,
- delisting handling differs,
- within-year weight updating differs from the original CRSP implementation.

Good replications should still show strong co-movement, especially for SMB.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
uv pip install -e ".[dev]"
cp /path/to/jkp_data.csv data/jkp_data.csv
```

## Run the notebook

```bash
source .venv/bin/activate
jupyter lab notebooks/05_ff_factor_replication.ipynb
```

The notebook automatically downloads the official Fama-French monthly factors
through `pandas_datareader` for the comparison step.

## Run tests

```bash
source .venv/bin/activate
pytest tests/test_factors.py -v
ruff check src/quant_trading/factors.py tests/test_factors.py
```

## References

- Fama, Eugene F., and Kenneth R. French (1992). *The Cross-Section of Expected
  Stock Returns*.
- Fama, Eugene F., and Kenneth R. French (1993). *Common Risk Factors in the
  Returns on Stocks and Bonds*.
- Jensen, Theis I., Bryan T. Kelly, and Lasse H. Pedersen (2023). *Is There a
  Replication Crisis in Finance?*
- Ken French Data Library: `F-F_Research_Data_5_Factors_2x3`.
