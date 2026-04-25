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
| **market_equity** | 1.97 | **2.17** | 1.43 | 0.78 |
| **ret_1_0** | **2.15** | 1.59 | 1.18 | 0.82 |
| **dbnetis_at** | 0.57 | 1.25 | **1.35** | 1.20 |
| **at_gr1** | 0.97 | **1.12** | 1.10 | 0.86 |
| **niq_at_chg1** | 0.40 | 0.64 | 0.85 | **1.04** |
| **oaccruals_at** | 0.62 | 0.94 | 1.00 | **1.03** |
| **be_me** | **0.90** | 0.75 | 0.63 | 0.57 |
| **op_at** | 0.21 | 0.41 | 0.45 | **0.64** |
| **ret_6_1** | -0.27 | 0.17 | 0.31 | **0.33** |
| **debt_at** | -0.41 | 0.13 | **0.29** | 0.22 |
| **qmj** | 0.09 | 0.20 | **0.28** | 0.22 |
| **beta_60m** | **0.25** | -0.17 | -0.17 | -0.16 |
| **ni_be** | 0.01 | **0.07** | 0.02 | 0.03 |

The heatmap confirms our hypothesis but with an important nuance. The trade-off between signal strength and diversification does not resolve the same way across signals. There is no common optimum. Each characteristic has its own sweet spot, and the drawdown column tells us just as much as the Sharpe ratio does.

I see three distinct groups.

## Group 1: Signals that reward extreme bets

**Size** (`market_equity`) peaks at 5/95 with a Sharpe of 2.17 — the highest single number in the entire heatmap. Going wider to 10/90 drops it to 1.43, and 30/70 falls to 0.78. The size premium concentrates in the smallest stocks. Diluting the portfolio with mid-caps weakens it.

**Value** (`be_me`) is similar: 0.90 at 1/99, declining monotonically to 0.57 at 30/70. The cheapest stocks really do outperform, and the relationship gets weaker as you include less extreme values.

## Group 2: Short-term reversal is the one exception

**Short-term reversal** (`ret_1_0`) is the only signal where both Sharpe and drawdown improve together as we tighten all the way to 1/99. Sharpe reaches 2.15 and max drawdown is -39.7% — better than its 5/95 drawdown of -58.9%.

I think this pattern has a clear behavioral explanation. Stocks that experience the most extreme recent moves are precisely the ones that attract disproportionate attention. Investors chase the top movers and flee the bottom movers, pushing prices further from fundamentals than any gradual drift would justify. This is why the strategy improves monotonically as I move toward the tails. The further I go into the extremes, the more my portfolio is composed purely of stocks where the attention effect was strongest and where the subsequent reversal is most predictable. Widening to 5/95 or 30/70 dilutes the portfolio with stocks where the move was too small to trigger the same feedback, and the signal weakens accordingly.

## Group 3: Signals destroyed by tightening

A third group is actively destroyed by extreme cutoffs.

**Profit growth** (`niq_at_chg1`) goes from Sharpe 1.04 at 30/70 down to 0.40 at 1/99. **Profitability** (`op_at`) drops from 0.64 to 0.21. **Momentum** (`ret_6_1`) actually flips negative at 1/99 (-0.27) despite being positive at every other cutoff.

Why? These signals are smoother. The relationship between the characteristic and future returns is roughly monotonic, but it does not steepen at the extremes the way size or reversal does. At 1/99, you have so few stocks that idiosyncratic noise dominates the signal. A single bad earnings surprise in your 50-stock long leg can wreck the month.

**Leverage** (`debt_at`) is the worst case: Sharpe of -0.41 at 1/99, with a -82.6% max drawdown. The most extreme leverage deciles are dominated by financial distress cases where the characteristic value is high not because of a deliberate capital structure choice, but because the company is in trouble. Going extreme here puts you long firms about to blow up.

## The Drawdown Heatmap Tells the Other Half of the Story

*[Insert: heatmap_drawdown_by_cutoff.png]*

| Characteristic | 1/99 | 5/95 | 10/90 | 30/70 |
|---------------|------|------|-------|-------|
| market_equity | -27.7% | -52.3% | -66.3% | -51.9% |
| ret_1_0 | -39.7% | -58.9% | -59.5% | -41.9% |
| dbnetis_at | -52.5% | -16.8% | -19.7% | -13.4% |
| at_gr1 | -66.4% | -46.2% | -36.0% | -26.5% |
| be_me | -78.2% | -69.1% | -69.6% | -60.3% |
| op_at | -93.7% | -81.2% | -73.4% | -35.1% |
| ret_6_1 | -100.0% | -95.8% | -73.7% | -50.2% |
| beta_60m | -96.7% | -99.8% | -99.4% | -95.3% |

Look at the last column. At 30/70, no signal has worse than -95% drawdown. But at 1/99, several signals produce **complete wipeouts** — momentum (`ret_6_1`) and quality (`qmj`) both hit -100% max drawdown. These are not theoretical risks. The 2009 momentum crash and the COVID liquidity squeeze are in this sample.

The drawdown heatmap tells you something the Sharpe heatmap hides. A strategy can have a positive Sharpe ratio and still be uninvestable. Accruals at 1/99 has a Sharpe of 0.62 (not bad!) but a max drawdown of -82.3%. Would you hold through that? I would not.

## The Return Heatmap: Where the Raw Numbers Get Wild

*[Insert: heatmap_return_by_cutoff.png]*

At 1/99, size shows an annualized return of 392.5% and reversal shows 186.8%. These are not typos. At extreme constrictions, the long-short spread is huge because you are holding a tiny number of the most extreme stocks. The returns are real in-sample, but they come with massive volatility and would be difficult to implement with any real capital — the stocks in the 1st percentile are micro-caps with negligible liquidity.

At 5/95, the numbers are more reasonable: size at 77% annualized, reversal at 48%. Still very high, reflecting that this is a long-short spread without transaction costs, on a universe that includes very small stocks.

## Key Takeaway

There is no single constriction level that works across all signals. The right answer is signal-specific, and ignoring that is leaving performance on the table at best and destroying it at worst.

However, for the rest of this series, as we play around with other portfolio settings, I find it reasonable to **use 5/95 as a baseline** since most signals either peak there or are not far off. The main exception is short-term reversal, where 1/99 dominates — but even for reversal, 5/95 gives a Sharpe of 1.59, which is plenty to work with.

In Part 3, we turn to the second tuning fork — weight scheme. Equal weighting, value weighting, characteristic-rank weighting, and characteristic-minmax weighting each impose a different structure on the portfolio. We will see that the choice of weight scheme can matter even more than the cutoff.

---

*All results use the JKP global factor dataset, US common equities, 1970–2025. 676 strategy configurations tested. Micro-cap stocks dropped. Code and data pipeline on [GitHub](https://github.com/gunsslashroses/quant-trading-experiments).*
