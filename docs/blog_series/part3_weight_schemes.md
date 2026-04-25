# Building Optimal Systematic Portfolios (Part 3/n): The Weight Scheme Changes Everything

*Documenting my learnings in building the best systematic portfolios with what I know*

---

In Part 2, we fixed the weight scheme at equal weighting and only turned one knob: how extreme the percentile cutoffs were. We found that there is no universal optimum — each signal has its own sweet spot, and short-term reversal is the one where going more extreme keeps paying off.

Now we turn to the second tuning fork: **how we weight the stocks once we've selected them.**

This turns out to matter more than I expected.

## The Four Weight Schemes

Using our 5/95 baseline from Part 2 (the cutoff that works reasonably across most signals), I tested four weighting approaches:

**Equal weight** — every selected stock gets the same dollar allocation. Simple. Gives the smallest stocks the same voice as the largest.

**Value weight** — weight by market capitalization. The biggest stocks dominate. This is what the academic literature defaults to, and it's the most realistic for large-scale implementation. You can't really put $100M into a $50M market cap stock.

**Characteristic-rank weight** — within each leg, stocks with more extreme characteristic values get more weight. If you're in the long leg because your book-to-market is high, the *highest* B/M stock gets the biggest position. This says "I trust the signal linearly — more extreme = more alpha."

**Characteristic-minmax weight** — similar idea, but using min-max scaling of the raw characteristic value instead of ranks. More sensitive to the actual magnitude differences.

## The Results: Equal Weight Usually Wins

Here's the Sharpe ratio for each characteristic at 5/95, averaged across weight caps:

| Characteristic | Equal | Char Rank | Char MinMax | Value |
|---------------|-------|-----------|-------------|-------|
| **market_equity** | 2.17 | **2.26** | 2.12 | 1.69 |
| **ret_1_0** | 1.59 | **1.72** | 1.65 | 0.28 |
| **dbnetis_at** | **1.25** | 1.21 | 1.24 | 0.42 |
| **at_gr1** | **1.12** | 1.11 | 0.92 | 0.45 |
| **oaccruals_at** | **0.94** | 0.93 | 0.61 | 0.25 |
| **be_me** | 0.75 | **0.75** | 0.47 | 0.15 |
| **niq_at_chg1** | **0.64** | 0.63 | 0.54 | 0.18 |
| **op_at** | 0.41 | 0.41 | 0.37 | **0.51** |
| **ret_6_1** | 0.17 | 0.17 | 0.05 | **0.57** |
| **qmj** | 0.20 | 0.20 | **0.20** | 0.18 |
| **debt_at** | **0.13** | 0.12 | -0.07 | -0.18 |
| **ni_be** | **0.07** | 0.09 | 0.08 | 0.01 |
| **beta_60m** | -0.17 | **-0.16** | -0.17 | -0.19 |

Three patterns jump out.

## Pattern 1: Value weighting destroys most signals

For 11 out of 13 characteristics, equal weight beats value weight. For some the gap is enormous:

- **Short-term reversal**: equal weight Sharpe 1.59, value weight 0.28. That's not a small difference — value weighting *destroys* the reversal signal.
- **Accruals**: 0.94 vs 0.25.
- **Debt issuance**: 1.25 vs 0.42.

Why? Because the return premium in most factors comes disproportionately from **small stocks**. When you value-weight, you're letting the biggest companies — the Apples and Microsofts — dominate the portfolio. But big companies don't exhibit strong accrual anomalies or reversal effects. Those are small-cap phenomena. Value weighting hands the microphone to the stocks with the weakest signal.

This is a well-known result in the academic literature (McLean & Pontiff 2016, Hou Xue Zhang 2020), but it's one thing to read about it and another to see it show up so consistently in your own grid search.

## Pattern 2: Two signals actually *prefer* value weight

Profitability (`op_at`) and momentum (`ret_6_1`) are the exceptions. Value weight gives Sharpe of 0.51 and 0.57, versus 0.41 and 0.17 equal-weighted.

I think this makes sense. Profitability is a stable, slow-moving characteristic — large firms have persistent profitability advantages, and the signal works precisely because markets undervalue those advantages for big boring companies. Momentum, similarly, requires capacity. You need the trend to persist long enough to trade it, and large-cap momentum is more investable and less prone to microstructure noise.

## Pattern 3: Characteristic weighting helps the two strongest signals

For size and short-term reversal — the two top-performing signals overall — characteristic-rank weighting edges out equal weighting. Size goes from 2.17 to 2.26, reversal from 1.59 to 1.72.

This is the signal saying "the relationship between the characteristic and future returns is not just monotonic — it's steeper at the extremes." If you're going to bet on small stocks outperforming, the *smallest* stocks outperform even more, and char-rank weighting lets you express that.

For most other signals, the difference between equal and rank-weighted is negligible.

## Minmax Weighting: Mostly Worse

Characteristic-minmax weighting underperforms rank weighting for almost every signal, and sometimes dramatically:

- Accruals: 0.61 vs 0.94 (equal) — minmax is terrible here.
- Investment: 0.92 vs 1.12 (equal).
- Leverage: -0.07 vs 0.13 (equal).

Minmax is sensitive to outliers in the raw characteristic values. One stock with an extreme accrual ratio can eat up most of the weight in a leg. Rank weighting is more robust because it only cares about ordering, not magnitude.

## What About Weight Caps?

Here's a surprising non-result: **weight caps barely matter.**

At 5/95 with char_rank_weighted, the Sharpe ratio for every characteristic is nearly identical whether the cap is 5%, 10%, 20%, or uncapped (100%):

| Characteristic | 5% cap | 10% cap | 20% cap | Uncapped |
|---------------|--------|---------|---------|----------|
| market_equity | 2.259 | 2.259 | 2.259 | 2.259 |
| ret_1_0 | 1.715 | 1.715 | 1.715 | 1.715 |
| dbnetis_at | 1.214 | 1.214 | 1.214 | 1.214 |

Why? Because at 5/95, you have enough stocks in each leg (~250+) that no single name naturally dominates. The cap only binds when you have very few stocks — which happens at 1/99 but not at 5/95. This is useful to know: if you pick a reasonable constriction level, you probably don't need to worry about weight caps. They're a second-order detail.

## Key Takeaway

Weight scheme is not a cosmetic choice. It can turn a Sharpe of 1.59 into 0.28 (reversal under value weight) or a 0.17 into 0.57 (momentum under value weight). The right answer depends on *where the alpha comes from* for each signal:

- Signals driven by small stocks → equal weight or char-rank weight.
- Signals that work in large caps → value weight is fine and more realistic.
- When in doubt, equal weight with a moderate constriction is a defensible default.

In Part 4, we bring it all together: what happens when we combine these 13 signals into a single portfolio.

---

*All results use the JKP global factor dataset, US common equities, 1970–2025. 676 strategy configurations tested. Code and data pipeline on [GitHub](https://github.com/gunsslashroses/quant-trading-experiments).*
