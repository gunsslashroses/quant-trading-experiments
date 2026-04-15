# Fama-French SMB & HML Factor Replication

Constructs the **SMB** (Small Minus Big) and **HML** (High Minus Low) factors from scratch using CRSP and Compustat data via WRDS, following the exact methodology from Fama & French (1993). Compares the replicated factors against the official values published on [Ken French's data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

## Methodology

Every June, all eligible US common stocks are sorted into **6 value-weighted portfolios** using a 2×3 independent sort on size (market equity) and value (book-to-market):

```
                  Low B/M       Neutral B/M       High B/M
                (Growth)                          (Value)
Small ME    │    SL       │      SM         │      SH       │
Big ME      │    BL       │      BM         │      BH       │
```

**Breakpoints** (computed using NYSE stocks only to prevent micro-cap distortion):
- Size: NYSE median of market equity
- Value: NYSE 30th and 70th percentiles of book-to-market

**Factor definitions:**

$$\text{SMB} = \frac{SL + SM + SH}{3} - \frac{BL + BM + BH}{3}$$

$$\text{HML} = \frac{SH + BH}{2} - \frac{SL + BL}{2}$$

Portfolios are rebalanced annually in June and held from July through the following June. Returns are value-weighted using buy-and-hold adjusted market equity.

### Why this specific procedure?

| Design choice | Rationale |
|--------------|-----------|
| **June sort** | Ensures fiscal year-end accounting data (typically December) is publicly available before sorting — avoids look-ahead bias |
| **NYSE-only breakpoints** | NASDAQ/AMEX have many more micro-caps; using all-exchange breakpoints would put most stocks in the "Small" bucket |
| **December ME for B/M** | Matches the fiscal year-end timing of book equity in the denominator |
| **2-year Compustat requirement** | Compustat historically backfilled data for newly added firms — requiring 2+ years of history mitigates this survivorship bias |
| **Value-weighted** | Reflects investable capacity; equal-weight would be dominated by illiquid micro-caps |

## Data

All data is sourced from **WRDS** (Wharton Research Data Services):

| Source | Table | Purpose |
|--------|-------|---------|
| Compustat | `comp.funda` | Book equity components: stockholders' equity, deferred taxes, preferred stock |
| CRSP | `crsp.msf` + `crsp.msenames` | Monthly returns, prices, shares outstanding, exchange and share type codes |
| CRSP | `crsp.msedelist` | Delisting returns (removes survivorship bias) |
| CCM | `crsp.ccmxpf_linktable` | Links Compustat `gvkey` to CRSP `permno` |

The notebook can either download live from WRDS (set `DATA_MODE = "wrds"`) or load from pre-saved CSVs (set `DATA_MODE = "local"`).

## Setup

```bash
git clone https://github.com/gunsslashroses/fama-french-factor-replication.git
cd fama-french-factor-replication
pip install -r requirements.txt
```

### WRDS access

If downloading live data, you'll need a [WRDS account](https://wrds-www.wharton.upenn.edu/). The `wrds` Python package will prompt for your username and password on first run.

### Using pre-saved data

If you (or someone else) have already downloaded the data:
1. Place the CSV files in the `data/` directory
2. Set `DATA_MODE = "local"` in the notebook

The notebook saves CSVs automatically during a WRDS run, so you only need to download once.

## Notebook structure

| Section | What it does |
|---------|-------------|
| **0. Setup** | Configuration: start date, data mode |
| **1. Compustat** | Download accounting data, construct book equity (BE) |
| **2. CRSP** | Download returns/prices, adjust for delistings, aggregate ME across share classes |
| **3. Weights** | Compute FF's buy-and-hold adjusted portfolio weights |
| **4. CCM merge** | Link Compustat to CRSP via the link table, compute B/M ratio |
| **5. Sorting** | NYSE breakpoints, 2×3 portfolio assignment in June |
| **6. Returns** | Value-weighted monthly portfolio returns, SMB and HML construction |
| **7. Comparison** | Download official FF factors, run correlation, regression, cointegration tests |
| **8. Visualization** | Cumulative returns, scatter plots, rolling correlation |
| **9. Summary** | Performance statistics and tracking error |

## References

- Fama, E. F., & French, K. R. (1993). *Common risk factors in the returns on stocks and bonds.* Journal of Financial Economics, 33(1), 3–56. [doi:10.1016/0304-405X(93)90023-5](https://doi.org/10.1016/0304-405X(93)90023-5)

- Fama, E. F., & French, K. R. (2015). *A five-factor asset pricing model.* Journal of Financial Economics, 116(1), 1–22. [doi:10.1016/j.jfineco.2014.10.010](https://doi.org/10.1016/j.jfineco.2014.10.010)

- Kenneth French's Data Library: [https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)

## License

MIT
