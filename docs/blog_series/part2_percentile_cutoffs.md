# Building Optimal Systematic Portfolios (Part 2/n): How Extreme Should Your Bets Be?

*Documenting my learnings in building the best systematic portfolios with what I know*

---

In Part 1, I argued that I am less interested in discovering factor number 401 (at the moment) and more interested in using existing signals better. This post is where that philosophy starts to touch code.

When I think about building a long-short factor portfolio, I find myself facing three structural choices that determine most of the outcome. I think of them as tuning forks that shift the frequency of what I am listening to in the data.

1. **Percentile constriction**: how extreme are my long and short bets?
2. **Weight scheme**: once I have selected the universe, how do I weight my positions? Equal weight, value weight, or something proportional to the characteristic itself?
3. **Maximum weight cap**: do I let a single stock dominate the portfolio, or do I impose a ceiling?

We are going to do something deliberately simple. Take all 13 characteristics. Form long-short portfolios. Only touch one knob: how extreme the tails are. All results here are monthly rebalancing, equal weighted, with micro-cap stocks dropped from the universe.

## Testing Percentile Constrictions

I tested four portfolio construction thresholds across all 13 characteristics: 1/99, 5/95, 10/90, and 30/70. The 1/99 specification is the most extreme — going long the top 1% of stocks and short the bottom 1% — while 30/70 is the least extreme.

Conceptually this is a trade-off between signal strength and diversification. Moving to more extreme cutoffs increases exposure to the characteristic, so if the factor is real, we should see higher Sharpe ratios at tighter constrictions. At the same time, tighter constriction reduces the number of names, raises idiosyncratic noise, and weakens diversification. The question is whether there is a common optimum across signals, or whether each characteristic has its own sweet spot.

## The Sharpe Heatmap

*[Insert: heatmap_sharpe_by_cutoff.png]*

| Characteristic | 1/99 | 5/95 | 10/90 | 30/70 |
|---------------|------|------|-------|-------|
| **Size (Market Equity)** | 1.97 | **2.17** | 1.43 | 0.78 |
| **Short-Term Reversal** | **2.15** | 1.59 | 1.18 | 0.82 |
| **Net Debt Issuance** | 0.57 | 1.25 | **1.35** | 1.20 |
| **Investment (Asset Growth)** | 0.97 | **1.12** | 1.10 | 0.86 |
| **Profit Growth** | 0.40 | 0.64 | 0.85 | **1.04** |
| **Accruals** | 0.62 | 0.94 | 1.00 | **1.03** |
| **Value (Book-to-Market)** | **0.90** | 0.75 | 0.63 | 0.57 |
| **Profitability (Op. Prof.)** | 0.21 | 0.41 | 0.45 | **0.64** |
| **Momentum (6-1)** | -0.27 | 0.17 | 0.31 | **0.33** |
| **Leverage (Debt/Assets)** | -0.41 | 0.13 | **0.29** | 0.22 |
| **Quality Minus Junk** | 0.09 | 0.20 | **0.28** | 0.22 |
| **Beta (60m)** | **0.25** | -0.17 | -0.17 | -0.16 |
| **ROE (NI/Book Equity)** | 0.01 | **0.07** | 0.02 | 0.03 |

The heatmap reveals a striking pattern: **10 out of 13 signals get worse when you push from 5/95 to the extreme 1/99.** The average Sharpe drop is 0.23 — that is not a rounding error. The diversification cost of holding too few names overwhelms whatever extra signal strength the extremes offer.

Only 3 signals out of 13 improve at 1/99. And each of the three exceptions has a specific reason for why it works differently.

This is the central result of this post. Let me walk through it.

## The general pattern: 5/95 is the sweet spot, 1/99 destroys value

For the majority of signals — investment, debt issuance, accruals, profit growth, profitability, momentum, leverage, quality, size, ROE — the Sharpe ratio peaks somewhere in the 5/95 to 30/70 range and then **declines** at 1/99.

Some examples:

- **Momentum** (6-month return): Sharpe of 0.33 at 30/70, gradually improving, then flips to **-0.27 at 1/99**. The momentum crash of 2009 is concentrated in extreme winners and losers. At 30/70 you diversify through it. At 1/99 it wipes you out.
- **Profit growth**: 1.04 at 30/70, drops to 0.40 at 1/99. The smooth accounting-based relationship gets swamped by idiosyncratic noise when you only hold 50 stocks per leg.
- **Leverage**: 0.22 at 30/70, turns **negative** (-0.41) at 1/99 with a -82.6% max drawdown. The most extreme leverage stocks are distressed companies — high debt not because of a capital structure choice, but because they are spiraling.
- **Even size** peaks at 5/95 (Sharpe 2.17) and *drops* to 1.97 at 1/99.

Why does this happen? At 1/99, you are holding roughly 50 stocks in each leg. One bad earnings surprise, one lawsuit, one delisting — and it is a meaningful percentage of your portfolio. The signal might be real, but the noise from holding so few names dominates. At 5/95 you have ~250 stocks per leg, and the law of large numbers helps.

## Exception 1: Short-Term Reversal — the attention trade

**Reversal** is the strongest exception. Sharpe improves monotonically from 0.82 at 30/70 to 1.59 at 5/95 to **2.15 at 1/99**. Even the max drawdown improves: -41.9% at 30/70, -58.9% at 5/95, then tightening to **-39.7% at 1/99**. Both Sharpe and risk improve together as you go more extreme. No other signal does this.

I think this has a clear behavioral explanation. Stocks that experienced the most extreme recent moves are precisely the ones that attract disproportionate attention. Investors chase the top movers and flee the bottom movers, pushing prices further from fundamentals than any gradual drift would justify. The further I go into the tails, the more my portfolio is composed purely of stocks where this attention-driven overshoot was strongest — and where the subsequent reversal is most predictable.

Widening to 5/95 or 30/70 dilutes the portfolio with stocks where last month's move was moderate. A 3% drop does not trigger the same panic selling and subsequent bounce-back that a 20% drop does. The reversal signal is not linear — it is convex in the extremes.

## Exception 2: Value — the deepest discount wins

**Value** (book-to-market) also improves all the way to 1/99: Sharpe of 0.57 at 30/70 rising to **0.90 at 1/99**. The very cheapest stocks — trading at deep discounts to book value — really do outperform the merely cheap.

This is consistent with the classic value story. Extreme-value stocks are often companies that the market has given up on. They might be shrinking, out of favor, or in declining industries. The market prices in a bleak future, and when reality turns out to be less bad than expected, the stock reprices sharply. This "expectations gap" is largest for the most extreme value stocks, which is why tighter constrictions keep working.

## Exception 3: Beta — a quirk, not a strategy

**Beta** (60-month market beta) shows Sharpe of 0.25 at 1/99 but is **negative** at every other cutoff (-0.17 at 5/95). This looks like an exception but I would not read too much into it. A Sharpe of 0.25 from a low-beta strategy at 1/99 is not economically compelling, especially with a -96.7% max drawdown. The "low beta anomaly" likely requires more careful construction (industry-neutral, volatility-adjusted) than a simple sort can deliver.

## Why exceptions are exceptions

The three signals that improve at 1/99 share a common trait: **the characteristic-return relationship is convex, not linear.** For reversal, the overshoot-and-bounce mechanism is strongest at the extremes. For value, the expectations gap is widest for the cheapest stocks. For most other signals — profitability, investment, quality — the relationship is roughly linear. Going from the 5th percentile to the 1st percentile of asset growth does not give you meaningfully more information about future returns, but it does halve the number of stocks in your portfolio.

## The Drawdown Heatmap Tells the Other Half of the Story

*[Insert: heatmap_drawdown_by_cutoff.png]*

| Characteristic | 1/99 | 5/95 | 10/90 | 30/70 |
|---------------|------|------|-------|-------|
| Size (Market Equity) | -27.7% | -52.3% | -66.3% | -51.9% |
| Short-Term Reversal | -39.7% | -58.9% | -59.5% | -41.9% |
| Net Debt Issuance | -52.5% | -16.8% | -19.7% | -13.4% |
| Investment (Asset Growth) | -66.4% | -46.2% | -36.0% | -26.5% |
| Value (Book-to-Market) | -78.2% | -69.1% | -69.6% | -60.3% |
| Profitability (Op. Prof.) | -93.7% | -81.2% | -73.4% | -35.1% |
| Momentum (6-1) | -100.0% | -95.8% | -73.7% | -50.2% |
| Beta (60m) | -96.7% | -99.8% | -99.4% | -95.3% |

Look at the last column. At 30/70, no signal has worse than -95% drawdown. But at 1/99, several signals produce **complete wipeouts** — momentum (`ret_6_1`) and quality (`qmj`) both hit -100% max drawdown. These are not theoretical risks. The 2009 momentum crash and the COVID liquidity squeeze are in this sample.

The drawdown heatmap tells you something the Sharpe heatmap hides. A strategy can have a positive Sharpe ratio and still be uninvestable. Accruals at 1/99 has a Sharpe of 0.62 (not bad!) but a max drawdown of -82.3%. Would you hold through that? I would not.

## The Return Heatmap: Where the Raw Numbers Get Wild

*[Insert: heatmap_return_by_cutoff.png]*

At 1/99, size shows an annualized return of 392.5% and reversal shows 186.8%. These are not typos. At extreme constrictions, the long-short spread is huge because you are holding a tiny number of the most extreme stocks. The returns are real in-sample, but they come with massive volatility and would be difficult to implement with any real capital — the stocks in the 1st percentile are micro-caps with negligible liquidity.

At 5/95, the numbers are more reasonable: size at 77% annualized, reversal at 48%. Still very high, reflecting that this is a long-short spread without transaction costs, on a universe that includes very small stocks.

## Key Takeaway

**10 out of 13 signals get worse when you push from 5/95 to 1/99.** The average Sharpe drop is 0.23. Extreme constriction is not free — it buys you signal strength but costs you diversification, and for most signals the cost wins.

The three exceptions — reversal, value, and beta — are not random. They are signals where the characteristic-return relationship is **convex**: the effect accelerates at the extremes. Reversal has a behavioral explanation (attention-driven overshooting), value has a fundamental one (deepest discounts = widest expectations gaps). Understanding *why* a signal is convex is what separates a principled decision from a data-mined one.

For the rest of this series, I use **5/95 as the baseline**. It is either the optimal or near-optimal cutoff for the majority of signals, and even for the exceptions (reversal, value), 5/95 still produces strong performance. It is the "safe default" — you leave a little on the table for reversal but avoid catastrophe for momentum, leverage, and profitability.

In Part 3, we turn to the second tuning fork — weight scheme. Equal weighting, value weighting, characteristic-rank weighting, and characteristic-minmax weighting each impose a different structure on the portfolio. We will see that the choice of weight scheme can matter even more than the cutoff.

---

*All results use the JKP global factor dataset, US common equities, 1970–2025. 676 strategy configurations tested. Micro-cap stocks dropped. Code and data pipeline on [GitHub](https://github.com/gunsslashroses/quant-trading-experiments).*
