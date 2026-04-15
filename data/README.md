## Data instructions

This repository expects a WRDS extract saved as `data/jkp_data.csv`.

### Source

- WRDS table: `contrib.global_factor`
- Dataset: Jensen, Kelly, and Pedersen global factor data
- Geography used here: US common stocks

### Minimum columns for the Fama-French replication notebook

- `id`
- `eom` or `month_date`
- `primary_sec`
- `exch_main`
- `source_crsp`
- `market_equity`
- `be_me`
- `ret_exc_lead1m`

### Why the file is not committed

The WRDS extract is licensed and should not be redistributed publicly. The repo
therefore contains code and documentation only, not the underlying stock-level
data.
