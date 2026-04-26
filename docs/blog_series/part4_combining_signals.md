# Building Optimal Systematic Portfolios (Part 4/n): Combining Signals

*Documenting my learnings in building the best systematic portfolios with what I know*

---

In Parts 2 and 3 we explored what happens when you build a long-short portfolio around a single characteristic — varying percentile cutoffs, weighting schemes, and weight caps. The best single-signal strategy was size with char-rank weighting at 5/95, Sharpe 2.26.

But in practice, nobody builds a portfolio around one characteristic. You have 13 signals, each carrying different information about the cross-section of expected returns. The natural question is: can we combine them into something better?

This is a bridge post. We are going to try three increasingly sophisticated ways to combine signals — starting with something a first-year analyst could implement in a spreadsheet and ending with something that touches the boundary of where machine learning begins. The results will set the stage for Part 5, where we cross that boundary.

## Why combine at all?

Two motivations.

First, **diversification**. A size-only portfolio crashed 50% during the dotcom bubble. A reversal-only portfolio had months down 58%. If these drawdowns do not happen at the same time across all 13 signals, then combining should smooth the ride.

Second, **information aggregation**. Each characteristic captures a different dimension of mispricing. Value says "this stock is cheap relative to fundamentals." Momentum says "this stock has been winning and will probably keep winning." Quality says "this company is well-run." A stock that scores well on all three dimensions is more likely to outperform than one that only scores well on size.

The question is how to do the combining — and whether more sophistication actually helps.

## Method 1: Consensus Voting

The simplest possible approach. Each of the 13 signals casts a +1 or -1 vote for each stock based on whether it falls in the top or bottom 5% of the cross-sectional distribution. Sum the votes. A stock enters the long leg when its net vote count exceeds a threshold *n*, and the short leg when it falls below *-n*.

I tested three thresholds: n≥2 (loose — only need 2 signals to agree), n≥4 (moderate), and n≥7 (strict — majority of signals must agree).

| Strategy | Sharpe | Ann Return (%) | Ann Vol (%) | Max DD (%) | Hit Rate (%) |
|----------|--------|---------------|-------------|------------|-------------|
| **Consensus n≥2** | **1.89** | 108.5 | 57.3 | **-10.3** | **89.4** |
| Consensus n≥4 | 1.39 | 159.2 | 114.9 | -26.6 | 83.2 |
| Consensus n≥7 | 0.43 | 181.9 | 424.0 | -93.1 | 62.1 |

The progression here is telling. As we demand more consensus, two things happen simultaneously: the raw return goes up (more extreme positions), but the volatility explodes faster than the return. Sharpe drops from 1.89 to 0.43.

The n≥2 threshold is the sweet spot — not because 2 is a magic number, but because at n≥2 you still have a large, diversified portfolio. Many stocks get at least 2 votes in the same direction, so the long and short legs each contain hundreds of names. At n≥7, only a handful of stocks get 7+ signals agreeing, and the portfolio becomes extremely concentrated.

**The n≥2 consensus strategy has the best max drawdown of anything in this entire series: -10.3%.** Better than any single factor. Better than any cutoff or weight scheme we tested. This is the diversification payoff — when you require at least 2 signals to agree, you are naturally filtering out stocks that are extreme on one dimension but average on everything else. The positions that survive are more robust.

## Method 2: Equal-Weight Composite

Instead of voting, we can be more precise. Scale each characteristic to [-1, 1] within each cross-section using percentile ranks. For signals where "low is good" (like reversal or accruals), flip the sign so that higher always means more attractive. Average all 13 scaled signals into a single composite score per stock. Sort on that composite; go long the top, short the bottom.

This is more quantitative than voting because it uses the full granularity of each signal rather than reducing it to a binary vote. And the result matters:

| Strategy | Sharpe | Ann Return (%) | Max DD (%) | Hit Rate (%) |
|----------|--------|---------------|------------|-------------|
| EqWt Composite · rank · signal_rank | **2.39** | 33.5 | -24.3 | 79.1 |
| EqWt Composite · rank · signal_minmax | 2.38 | 34.5 | -26.3 | 79.2 |
| EqWt Composite · rank · equal | 2.23 | 25.3 | -17.8 | 76.8 |

**The equal-weight composite with rank scaling achieves a Sharpe of 2.39** — the highest Sharpe we have seen in this entire series, including the single-factor grid search.

This is a meaningful result. A simple average of 13 signals, with no optimization, no lookback windows, no model fitting, beats the best single-signal portfolio (size at 2.26) and comes close to the best consensus strategy on drawdown.

One important caveat: this only works with **rank-scaled** signals. When I tried minmax scaling instead of rank scaling, every method collapsed to negative Sharpe with -100% drawdowns. Minmax scaling is sensitive to outliers in the raw characteristic values — one extreme observation in the composite score can dominate the sort. Rank scaling is robust because it only cares about ordering. This is the same lesson we learned with weight schemes in Part 3, but applied at the signal combination level.

## Method 3: IC-Weighted Composite

The natural objection to Method 2 is: why give every signal equal weight? Some signals are stronger than others, and their relative strength changes over time. Momentum was strong in the 2000s, weak after 2009. Value was strong in the early 2000s, weak for a decade afterward.

IC-weighting tries to adapt. Each month, we measure the rolling Information Coefficient for each signal — the rank correlation between the signal and realized returns over the prior 18 months. Signals with higher recent IC get more weight in the composite. Signals that have been failing get downweighted or zeroed out.

| Strategy | Sharpe | Ann Return (%) | Max DD (%) | Hit Rate (%) |
|----------|--------|---------------|------------|-------------|
| IC-Weighted · rank · equal | 0.41 | 10.1 | -88.5 | 65.4 |
| IC-Weighted · rank · signal_minmax | 0.40 | 12.7 | -98.9 | 66.0 |
| IC-Weighted · rank · signal_rank | 0.38 | 11.4 | -98.5 | 65.3 |

This is the most sobering result in this post. **IC-weighting dramatically underperforms both equal-weight compositing and consensus voting.** Sharpe drops from 2.39 to 0.41. Drawdowns explode to near -100%.

What went wrong? Two things.

First, **IC is noisy**. The rank correlation between a signal and returns, measured over 18 months, has massive sampling error. A signal might show high IC simply because a few lucky months happened to align. When you weight your composite based on this noisy estimate, you are essentially trend-following the noise.

Second, **IC-weighting is implicitly a momentum bet on signal performance**. You are overweighting signals that have been working recently and underweighting those that have not. But signal performance tends to mean-revert, not persist. A signal that had high IC last year is not necessarily going to have high IC next year — and by overweighting it, you are loading up on the dimension that is about to mean-revert.

The equal-weight composite avoids both problems by refusing to have an opinion about which signal is "hot." It just averages everything, every month, with the same weights. Less exciting, more robust.

## The Big Picture

*[Insert: bar_multisignal_sharpe.png]*

*[Insert: heatmap_multisignal_rank.png]*

Here is what I take away from these results:

**1. Simple combination works.** The equal-weight composite at Sharpe 2.39 beats everything else in this series. No optimization, no lookback, no model. Just average 13 rank-scaled signals and sort.

**2. Consensus voting trades Sharpe for drawdown protection.** At n≥2 it gives up about 0.5 Sharpe relative to the composite but delivers the best drawdown in the entire series (-10.3%). If you manage real capital and care about path dependency, this matters.

**3. Sophistication backfired.** IC-weighting is the most "quant" method of the three and it is by far the worst. The noise in rolling IC estimates overwhelms any benefit from adaptive weighting.

**4. Rank scaling is non-negotiable.** Every method that used minmax scaling collapsed. This is not a small implementation detail — it is the difference between a 2.39 Sharpe and a -0.78 Sharpe.

## The natural next question

Everything in Parts 2–4 has been a sort-based strategy. We pick a signal (or combination), rank stocks, go long the top, short the bottom. The portfolio construction is clever, but the "model" is always just a ranking.

Machine learning asks a different question entirely. Instead of ranking on one signal or an average of 13 signals, can we learn a function that maps all 13 characteristics simultaneously to an expected return? Can nonlinear feature interactions, decision trees, or neural networks find patterns that simple averaging misses?

That is Part 5.

---

*All results use the JKP global factor dataset, US common equities, 1970–2025. 13 signals, 5/95 percentile cutoffs, equal-weighted portfolios. Code on [GitHub](https://github.com/gunsslashroses/quant-trading-experiments).*
