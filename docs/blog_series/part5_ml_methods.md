# Building Optimal Systematic Portfolios (Part 5/n): Can Machine Learning Beat Factor Sorts?

*Documenting my learnings in building the best systematic portfolios with what I know*

---

In Parts 2–4, we exhausted the sort-based playbook. We found that a simple equal-weight composite of 13 rank-scaled signals achieves a Sharpe of 2.39 with a -24% max drawdown, and that consensus voting at n≥2 delivers the best drawdown in the series at -10.3%.

Now the natural question: can machine learning do better?

The answer, spoiler alert, is yes — but not for the reasons you might expect. The improvement is not about finding hidden nonlinear patterns in the data. It is about the drawdown column.

## The Setup

I trained five models on all data up to December 2015 and tested them out-of-sample from January 2016 onward. Same 13 characteristics as features, plus realized volatility and Fama-French 49-industry dummies (64 features total). Same long-short portfolio evaluation framework from Parts 2–4. The only thing that changed is how the expected return prediction is generated.

| Model | What it does |
|-------|-------------|
| **Linear Regression** | Baseline OLS. Just a linear combination of features. |
| **RBF Kernel Ridge** | Captures nonlinear feature interactions via a kernel approximation with Ridge regularization. |
| **Random Forest** | Ensemble of shallow decision trees, each seeing a random subset of features. |
| **Deep Neural Network** | Two-layer NN with L2 regularization and Huber loss. |
| **IPCA** | Instrumented PCA — learns latent factors whose loadings depend on observable characteristics. |

For Random Forest and RBF Kernel Ridge, hyperparameters were tuned using Optuna (Bayesian optimization) with temporal cross-validation. The DNN used Optuna as well, searching over regularization strength, learning rate, layer widths, and loss function. All HP tuning was done on a subsample to keep compute feasible, then the best model was refit on the full training set.

## The Results

*[Insert: ml_sharpe_drawdown_combined.png]*

| Model | Sharpe | Max DD (%) |
|-------|--------|-----------|
| **Deep Neural Network** | **2.53** | **-1.9** |
| **Random Forest** | **2.35** | -7.6 |
| **RBF Kernel Ridge** | 2.31 | -1.6 |
| Linear Regression | 2.24 | -11.9 |
| IPCA | 2.20 | -11.3 |

*(Equal weight, best cutoff per model)*

Every ML model produces a Sharpe above 2.0. That is the first observation. Even the baseline linear regression, which is just a linear combination of the same 13 signals we used in Parts 2–4, achieves 2.24. The sort-based composite from Part 4 achieved 2.39. So ML is in the same ballpark — the headline Sharpe numbers are not dramatically different.

The difference is in the drawdowns.

## The Drawdown Story

The DNN achieves a max drawdown of **-1.9%** at the 30/70 cutoff. RBF Kernel Ridge hits **-1.6%** at 5/95. Compare this to the best sort-based results:

- Best single factor (size, char-rank, 5/95): Sharpe 2.26, max drawdown **-50.6%**
- Best composite (equal-weight, rank): Sharpe 2.39, max drawdown **-24.3%**
- Best consensus (n≥2): Sharpe 1.89, max drawdown **-10.3%**
- **DNN at 30/70**: Sharpe **2.53**, max drawdown **-1.9%**

The DNN does not just match the best Sharpe from Parts 2–4. It delivers it with almost no drawdown. A max drawdown of -1.9% over a 9-year out-of-sample period means the strategy essentially never had a bad quarter.

*[Insert: ml_sharpe_by_cutoff.png]*

## Why ML Helps: Ranking, Not Levels

How can a model with an OOS R² of less than 1% produce a Sharpe of 2.53? Because cross-sectional investing only requires getting the **ranking** approximately right. You do not need to predict that stock A will return 2.3% next month. You only need to predict that stock A will outperform stock B. Even tiny improvements in ranking accuracy translate to meaningful portfolio improvements when you sort 8,000+ stocks into long and short legs.

The DNN and Random Forest achieve this by capturing nonlinear interactions between features that a simple sort or linear average misses. A stock that has both low momentum and high accruals might be much worse than either signal alone would suggest. A linear model treats these as additive; a tree or neural net can learn that the combination is particularly toxic.

## The Cutoff Pattern Is Inverted for DNN

Something interesting in the heatmap: the DNN achieves its **best Sharpe at 30/70** (2.53), and it **declines** as you go to tighter cutoffs — 2.49 at 10/90, 2.26 at 5/95, 2.11 at 1/99.

This is the opposite of most single-factor strategies, where tighter cutoffs improved performance up to a point. The DNN's predictions are already a composite of all 13 signals plus their interactions. Going to extreme cutoffs does not add much signal — the model has already identified which stocks are the best and worst. But it does reduce the number of stocks, which hurts diversification.

In other words, the DNN has already done the signal concentration work internally. You do not need to do it again at the portfolio construction level.

Random Forest and RBF Kernel Ridge show the more familiar pattern — peaking around 5/95 to 10/90 — suggesting their predictions still benefit from focusing on the extremes.

*[Insert: ml_drawdown_by_cutoff.png]*

## IPCA and Linear Regression: The Academic Baselines

IPCA (Instrumented PCA) and Linear Regression perform similarly — Sharpe around 2.05–2.24, drawdowns around -10% to -17%. These are respectable numbers. They confirm that even a linear model applied to all 13 features simultaneously outperforms any single-factor sort from Part 2.

But they do not get the drawdown improvements that the nonlinear models achieve. The gap between linear regression (max DD -11.9%) and the DNN (max DD -1.9%) is not about Sharpe — it is about path. Both produce similar returns per unit of risk. The DNN just spreads the returns more evenly across months.

## Key Takeaways

**1. ML improves drawdowns more than Sharpe.** The headline Sharpe ratios (2.1–2.5) are in the same range as the best sort-based strategies (2.3–2.4). The real gain is in reducing max drawdowns from -10% to -50% down to -2%.

**2. The DNN prefers wide cutoffs.** Unlike single-factor sorts, the DNN's best performance is at 30/70 — the widest cutoff. It has already concentrated the signal internally. Tight cutoffs just reduce diversification without adding information.

**3. Random Forest is the most consistent.** It delivers Sharpe above 2.3 across all cutoffs and weight schemes, making it the most robust model to portfolio construction choices.

**4. Even linear regression beats single-factor sorts.** Using all 13 features simultaneously in a simple linear model (Sharpe 2.24) outperforms the best single-factor sort (Sharpe 2.26 for size, but with -50.6% drawdown vs -11.9%).

**5. The Sharpe numbers are high.** All models produce Sharpe ratios above 2.0 out of sample. These are long-short portfolios without transaction costs on a universe that includes small-cap stocks. Real-world implementation would face slippage, capacity constraints, and borrowing costs that would compress these numbers. But the relative ranking of models should be stable.

## Robustness Check: Are Nano-Caps Driving Everything?

At this point I need to be honest about a concern. The Sharpe ratios above — 2.1 to 2.5 across all models — are high. Suspiciously high. The full JKP universe includes thousands of nano-cap and micro-cap stocks with market caps below $50M. These stocks can have monthly returns of +500% or -90%. They are barely tradeable in practice. If our ML models are just learning to pick up on nano-cap return explosions, the results are real in-sample but meaningless for any practical purpose.

So I reran the entire analysis after **dropping all nano-cap and micro-cap stocks** from the universe. Same models, same features, same train/test split, same portfolio construction. The only change is the stock universe.

*[Insert: ml_robustness_bar.png]*

| Model | Sharpe (Full) | Sharpe (No Nano/Micro) | Drop |
|-------|:---:|:---:|:---:|
| Deep Neural Network | 2.53 | 1.96 | -0.57 |
| Random Forest | 2.35 | 1.58 | -0.77 |
| RBF Kernel Ridge | 2.31 | 1.24 | -1.07 |
| Linear Regression | 2.24 | 1.24 | -1.00 |
| IPCA | 2.20 | 1.05 | -1.15 |

The Sharpe ratios drop significantly — roughly cut in half for most models. This confirms that a large fraction of the performance was coming from the smallest, least tradeable stocks. The nano-cap universe is where the cross-sectional signal is strongest, because those stocks are the least efficiently priced. Remove them and you remove much of the alpha.

But here is the important part: **the ranking of models is preserved, and the DNN still leads.** At 1.96, the DNN's Sharpe without nano-caps is still a strong strategy. It also retains the best drawdown profile among all models (-5.9% vs -10% to -34% for the rest).

*[Insert: ml_robustness_4panel.png]*

The drawdown story also changes. Without nano-caps, the near-zero drawdowns from the full universe (DNN at -1.9%) become more realistic numbers (-5.9% for DNN, -10.8% to -16.7% for the linear models). The full-universe drawdowns were artificially suppressed because nano-cap returns are so volatile that the long-short spread was always positive — driven by outlier returns on the long side.

**What does this mean?**

1. **The headline Sharpe numbers from the full universe are inflated.** They are real in-sample but not achievable at scale. If you are managing real capital, the no-nano-micro numbers are the relevant benchmark.

2. **The relative model rankings survive.** DNN > Random Forest > the rest. This ordering is robust to the universe filter, which means the models are learning something real about the feature interactions — not just picking up on nano-cap noise.

3. **A Sharpe of 1.5–2.0 without nano-caps is still strong.** For context, most published factor strategies achieve Sharpe ratios of 0.5–1.0 after realistic filters. The DNN at 1.96 without nano-caps is a genuinely good result.

4. **This is a reminder to always question your backtest.** The first result looked too good. The robustness check shows why, and the answer is honest: small stocks contributed disproportionately. Reporting both results — and explaining the gap — is more useful than reporting only the flattering one.

---

*All results out-of-sample: train ≤ 2015, test 2016–2025. JKP global factor dataset, US common equities. Hyperparameter tuning via Optuna (Akiba et al., KDD 2019). Code on [GitHub](https://github.com/gunsslashroses/quant-trading-experiments).*
