# The Dataset Behind the Signals: A Map of 13 Equity Factors

*A practical guide to the data and portfolio engineering in this project.*

---

## The dataset

Everything here is built on one table: the **Jensen, Kelly, and Pedersen (2023) global factor dataset**, hosted on WRDS as `contrib.global_factor`. It's the academic community's most comprehensive standardized set of firm characteristics — over 300 variables, harmonized across global markets, updated monthly.

I filtered it down to **US common stocks** (`common=1, excntry='USA'`), spanning **January 1970 to February 2025**. That gives roughly **8.9 million stock-month observations** — every US common equity that appeared on NYSE, AMEX, or NASDAQ over 55 years, with its characteristics measured at the end of each month and its realized excess return over the following month.

> **Reference**: Jensen, T. I., Kelly, B. T., & Pedersen, L. H. (2023). *Is There a Replication Crisis in Finance?* Journal of Finance, 78(5), 2465–2518.

## Why this dataset

Most quant research starts with one of three sources: CRSP/Compustat merged files (build everything yourself), the Fama-French factor library (pre-built factors, no stock-level data), or some vendor's proprietary dataset. JKP sits in a sweet spot: it gives you **stock-level panel data** with pre-computed characteristics that follow the exact definitions from the original academic papers. No ambiguity about how book equity was calculated, or which share class was used, or how delisting returns were handled.

The prediction target throughout is `ret_exc_lead1m` — the stock's excess return over the risk-free rate in the **next** month. For strategy work I winsorize this at the 5th/95th percentiles cross-sectionally each month to limit the influence of extreme observations. That winsorized version is called `ret_exc_lead1m_w`.

## The 13 signals

The JKP dataset has hundreds of characteristics. I picked **one intuitive representative variable** for each of 13 well-known factor themes. The goal isn't to be exhaustive — it's to have a clean, interpretable set of signals to build portfolios around.

| # | Theme | Variable | What it measures | Expected direction |
|---|-------|----------|-----------------|-------------------|
| 1 | **Size** | `market_equity` | Market cap in USD | Small > Big |
| 2 | **Value** | `be_me` | Book-to-market ratio | High B/M > Low B/M |
| 3 | **Profitability** | `op_at` | Operating profitability / total assets | High > Low |
| 4 | **ROE** | `ni_be` | Net income / book equity | High > Low |
| 5 | **Quality** | `qmj` | Quality-minus-Junk composite | High > Low |
| 6 | **Investment** | `at_gr1` | 1-year asset growth | Low growth > High growth |
| 7 | **Accruals** | `oaccruals_at` | Operating accruals / total assets | Low > High |
| 8 | **Debt issuance** | `dbnetis_at` | Net debt issuance / total assets | Low > High |
| 9 | **Leverage** | `debt_at` | Total debt / total assets | Low > High |
| 10 | **Profit growth** | `niq_at_chg1` | 1-year change in quarterly NI/assets | High > Low |
| 11 | **Momentum** | `ret_6_1` | Cumulative return, months t−6 to t−1 | High > Low |
| 12 | **Short-term reversal** | `ret_1_0` | Prior 1-month return | Low > High |
| 13 | **Low risk** | `beta_60m` | 60-month CAPM beta | Low beta > High beta |

The "expected direction" column is the sign of the return premium that decades of academic research says should exist. Small stocks should outperform large. Cheap stocks should beat expensive. High-quality firms should earn more than junky ones. Past losers should bounce back over the next month (reversal), but past winners over 6 months should keep winning (momentum).

These aren't controversial claims — they're the bread and butter of empirical asset pricing. The interesting question is what happens when you actually build portfolios around them.

## How the signals become portfolios

This is where the engineering matters. The same 13 signals can produce wildly different performance depending on choices that seem like implementation details but are actually first-order decisions.

### Universe and breakpoints

Every month, I rank all stocks on each characteristic. But "all stocks" is a loaded term — NASDAQ is full of tiny firms that are expensive to trade and can distort percentile cutoffs. Following the Fama-French convention, I compute breakpoints using **NYSE stocks only**, then apply those breakpoints to the full universe (NYSE + AMEX + NASDAQ). This prevents micro-caps from overwhelming the portfolio assignments.

### Percentile cutoffs

How extreme do you go? I test four configurations:
- **30/70** — roughly the top and bottom third. Broad portfolios, hundreds of stocks per leg.
- **10/90** — top and bottom deciles. More concentrated.
- **5/95** — top and bottom 5%. Getting aggressive.
- **1/99** — top and bottom 1%. Very concentrated, very high signal strength.

Tighter cutoffs mean stronger signal — you're only trading the stocks where the characteristic value is most extreme. But you also get smaller portfolios, higher turnover, and more exposure to idiosyncratic risk.

### Weighting

Once you've decided which stocks go in the long and short legs, you need to decide how to weight them:

- **Equal weight** — every stock gets the same dollar allocation. Tilts toward small caps.
- **Value weight** — weight by market cap. Tilts toward large caps, more capacity.
- **Characteristic-rank weight** — stocks with more extreme signal values get larger weights. The strongest signals get the most capital.
- **Characteristic-minmax weight** — similar idea, but using min-max scaling of the raw characteristic value instead of ranks. More sensitive to the actual magnitude of the signal.

Each weighting scheme embeds a different belief about where the signal's alpha comes from. Equal weight says "every stock in the extreme tail contributes equally." Value weight says "I want this to be tradeable at scale." Characteristic weight says "I trust the signal linearly — more extreme = more alpha."

### Weight caps

To prevent any single stock from dominating a leg, I optionally cap the maximum weight per name at 5%, 10%, or 20% of the leg, then renormalize. This is a real-world risk management constraint — no PM wants 40% of their short book in one name.

### The grid

Crossing all these choices gives **832 strategy configurations per signal**: 4 percentile pairs × 4 weighting schemes × 4 weight caps × 13 characteristics. Each one produces a monthly long/short return series that I evaluate on annualized return, Sharpe ratio, max drawdown, and hit rate.

## Combining signals

Individual signals are interesting, but the real question is whether combining them produces something better. I test three approaches:

**Consensus voting**: each signal casts a +1 or −1 vote for each stock. A stock enters the long leg when it accumulates enough "buy" votes across the 13 signals (e.g., at least 7 out of 13 say "long"). Simple, interpretable, no model risk.

**Equal-weight composite**: scale each signal to [−1, 1], average them into a single composite score per stock, then sort on that composite. This is the "just average everything" baseline that's surprisingly hard to beat.

**IC-weighted composite**: same as above, but weight each signal by its rolling out-of-sample Information Coefficient — the rank correlation between the signal and next-month return over the prior 18 months. This lets the model lean into signals that have been working recently and fade those that haven't.

## What's next

The notebooks extend this further into ML territory — using the same 13 characteristics (plus realized volatility and industry dummies) as features for return prediction with linear regression, random forests, neural networks, AdaBoost, IPCA, and a custom max-Sharpe-ratio model. But the portfolio engineering described here is the foundation that all of it sits on.

The code for everything above lives in `src/quant_trading/` — signals, portfolios, strategies, evaluation. The notebooks in `notebooks/` are the exploratory analysis.

---

*Built with the [JKP global factor dataset](https://jkpfactors.com/) via WRDS.*
