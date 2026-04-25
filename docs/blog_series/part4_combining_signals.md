# Building Optimal Systematic Portfolios (Part 4/n): Combining Signals Without a PhD

*Documenting my learnings in building the best systematic portfolios with what I know*

---

In Parts 2 and 3, we exhausted the single-signal playbook. We learned that percentile cutoffs, weight schemes, and weight caps all matter — but in a signal-specific way. Size and reversal love extreme tails. Most other signals prefer 5/95 with equal weighting. Value weighting destroys most anomalies because the alpha lives in small caps.

Now the natural question: **can we do better by combining all 13 signals?**

The answer is yes, but the *how* matters a lot, and the simplest method is surprisingly hard to beat.

## Three Approaches

I tested three ways to go from 13 individual signals to a single portfolio:

### Method 1: Consensus Voting

Each signal casts a vote for each stock. If a stock is in the top 5% on momentum, it gets a +1 from momentum. If it's in the bottom 5% on accruals, it gets a +1 from accruals (since low accruals is good). Sum the votes across all 13 signals. A stock enters the long leg when its net vote count crosses a threshold — say, at least 4 out of 13 signals agree.

This is the most interpretable approach. There's no model. There's no weighting. A stock is long because a bunch of independent signals independently say it should be.

I tested thresholds of 2, 4, and 7 net votes.

### Method 2: Equal-Weight Composite

Scale each characteristic to [-1, 1] within each month (using cross-sectional rank). For negative-direction signals like accruals, flip the sign so that "good" always points up. Average all 13 scaled signals into a single composite score per stock. Sort on that composite; long the top, short the bottom.

This is the "just average everything" approach. It's what most practitioners start with, and it's a surprisingly high bar.

### Method 3: IC-Weighted Composite

Same as Method 2, but instead of averaging with equal weights, weight each signal by its **rolling out-of-sample Information Coefficient** — the rank correlation between the signal and next-month returns over the prior 18 months.

This lets the model lean into signals that have been working recently and fade those that haven't. It's the first step toward adaptivity. But it also introduces a parameter (the lookback window) and the risk of overfitting to recent history.

## What Matters More: The Signal, or the Engineering?

Here's what struck me looking at the single-factor results. The top strategy across all 676 configurations is size (market_equity) with char_rank_weighted at 5/95, Sharpe 2.26. The second is reversal at 1/99, Sharpe 2.15.

These numbers come from **single signals** with carefully chosen portfolio construction. The question is whether combining all 13 signals can beat a well-constructed single-signal portfolio.

The answer, from my experiments with Methods 1–3, is nuanced. Combination methods do a good job of *stabilizing* performance — lower drawdowns, smoother equity curves. But they rarely beat the best single signal in terms of raw Sharpe. The composite signal diversifies across themes, which reduces volatility but also dilutes the strongest signal.

This is actually an important finding. In the factor investing literature, there's a presumption that combining factors is always better. In practice, if one of your signals is genuinely strong (like size or reversal in this data), an equal-weight average across 13 signals will water it down with 12 weaker ones.

## When Combination Wins

Where the composite approaches shine is in the **worst months**. Single-signal portfolios can have catastrophic months — momentum crashed in 2009, value crashed during COVID. The composite signal tends to avoid these because when one signal is having a terrible month, the other 12 are typically not all failing simultaneously.

If your objective function includes drawdown constraints or you're managing real capital where drawdowns trigger redemptions, the composite approach may be the right choice even if the headline Sharpe is lower.

## The IC-Weighting Question

Method 3 (IC-weighted) is more complex than Method 2 (equal-weighted). Is it better?

In my experiments, it depends on the signal scaling method. Using rank-scaled signals with IC weights sometimes outperforms equal weights, but the improvement is modest and comes with additional complexity and parameter sensitivity. The rolling IC can be noisy with 18-month lookback, and the weight on each signal bounces around.

My honest assessment: for most practitioners, Method 2 (equal-weight composite) is the right starting point. Add IC-weighting only if you have a reason to believe that signal quality varies substantially over time and you have enough history to estimate IC reliably.

## Key Takeaway

1. **Combining signals stabilizes performance** — lower drawdowns, fewer catastrophic months.
2. **Simple averaging is surprisingly hard to beat** — IC-weighting adds complexity for modest gain.
3. **The best single signal, well-engineered, can outperform a naive combination** — size at 5/95 with char_rank_weighted beats most composite methods in raw Sharpe.
4. **The right choice depends on your objective** — if you optimize for Sharpe, stick with the best single signal. If you optimize for drawdown-adjusted returns, combine.

In Part 5, we leave the factor world entirely and ask: can machine learning do better?

---

*All results use the JKP global factor dataset, US common equities, 1970–2025. Code and data pipeline on [GitHub](https://github.com/gunsslashroses/quant-trading-experiments).*
