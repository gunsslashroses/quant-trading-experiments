# Building Optimal Systematic Portfolios (Part 5/n): Can Machine Learning Beat Factor Models?

*Documenting my learnings in building the best systematic portfolios with what I know*

---

Up to this point, everything in this series has been a **sort-based strategy**. Take a characteristic, rank stocks, go long the top, short the bottom. The portfolio engineering we discussed in Parts 2–4 (cutoffs, weights, combinations) is sophisticated, but the underlying approach is simple: the signal is the raw characteristic value, and the portfolio is a mechanical sort on that value.

Machine learning asks a different question. Instead of sorting on one characteristic at a time (or a naive average of 13), can we learn a **function** that maps all characteristics simultaneously to an expected return?

I trained seven models on all data up to 2015 and tested them out-of-sample from 2016 onward. Same 13 characteristics as features, plus realized volatility and industry dummies. Same long-short portfolio evaluation. The only thing that changed is *how* the prediction is generated.

## The Models

| # | Model | What it does |
|---|-------|-------------|
| 1 | **Linear Regression** | Baseline OLS. Learns a linear combination of features. |
| 2 | **RBF Kernel Ridge** | Captures nonlinear feature interactions via an RBF kernel approximation (Nyström method) with Ridge regularization. |
| 3 | **Random Forest** | Ensemble of shallow decision trees. Each tree sees a random subset of features, reducing overfitting. |
| 4 | **Deep Neural Network** | Two-layer NN with L2 regularization and Huber loss (robust to outliers). |
| 5 | **AdaBoost** | Boosted decision trees — iteratively focuses on the hardest-to-predict stocks. |
| 6 | **Max Sharpe Regression (MSRR)** | A linear model trained not to minimize prediction error, but to directly maximize the portfolio's Sharpe ratio. Custom PyTorch loss function. |
| 7 | **IPCA** | Instrumented PCA — learns latent factors whose loadings are linear functions of observable characteristics. |

## Hyperparameter Tuning: Why Grid Search is Dead

For models 2–5, I used **Optuna** — a Bayesian optimization framework that replaces grid search. Instead of evaluating every point on a pre-defined grid, Optuna models the objective function surface and focuses trials on the most promising HP regions. It's the same idea as the coarse-to-fine manual search I discussed in Part 2, but automatic and principled.

Each model's hyperparameters are tuned with 5-fold `TimeSeriesSplit` cross-validation — no look-ahead bias. The HP search runs on a 200–300k row subsample (the full dataset is ~4M rows for training, which would take days to grid-search over).

## The Reality Check: OOS R² is Tiny

Here are the out-of-sample results:

| Model | OOS R² | Sign Accuracy |
|-------|--------|---------------|
| Linear Regression | 0.1% | 50.1% |
| RBF Kernel Ridge | 0.3% | 50.9% |
| **Random Forest** | **5.0%** | **51.1%** |
| DNN (L2 + Huber) | 0.4% | 52.4% |
| AdaBoost | — | — |
| MSRR | -145% | 51.1% |
| IPCA | 0.1% | 50.2% |

Let me be honest about these numbers. An OOS R² of 5% means the model explains 5% of the variation in next-month returns. That sounds awful. The sign accuracy is barely above a coin flip.

But here's the thing: **this is completely expected.** In the cross-section of equity returns, even the best models achieve R² in the low single digits. Gu, Kelly, and Xiu (2020) — the seminal ML-for-asset-pricing paper — report OOS R² of about 0.4% monthly for their best neural network. The fact that my Random Forest gets 5% is actually suspiciously high and warrants scrutiny (it may be capturing something nonlinear in the feature interactions, or it may be overfitting to the subsample period).

The MSRR result (R² = -145%) looks terrible, but MSRR doesn't optimize for prediction accuracy — it optimizes for portfolio Sharpe ratio. Its predictions are wildly scaled (mean prediction ~0.13, vs actual returns ~0.01) because it's trying to maximize the *ranking* of returns, not predict their *level*. The portfolio-level evaluation is where it should be judged.

## What Actually Matters: Portfolio Performance

This is where the ML models earn their keep. Even tiny improvements in prediction accuracy can translate to meaningful portfolio improvements when you sort on the predictions:

At the 30/70 cutoff with equal weighting, the DNN achieves a Sharpe of about 2.5 — better than the best single-factor sort in our entire grid search (size at 2.26). And the DNN's advantage grows at tighter cutoffs.

Random Forest similarly produces strong portfolio Sharpe ratios, driven not by getting the level of returns right, but by getting the **ranking** approximately right. In cross-sectional investing, you only need to know that stock A will outperform stock B — you don't need to know by how much.

## The Engineering That Nobody Talks About

Building these models at scale required solving problems that no ML textbook covers:

**Memory**: The full training set is ~4M rows × 64 features. With 7 models in one notebook, peak memory easily exceeds 15 GB. I split the data into parquet files, load only what's needed for each model, and free test data after evaluation. This is boring infrastructure work, but without it, the notebook just crashes.

**HP tuning at scale**: Running Optuna with 50 trials × 5 CV folds × 100 epochs on 4M rows is infeasible. Each trial would take 10+ minutes; 250 trials would take 40+ hours. Solution: subsample 200–300k rows for the HP search, then refit the best configuration on the full dataset.

**Temporal CV**: A subtle but critical detail. Keras's `validation_split=0.2` takes the last 20% of the array, which may mix time periods. For time-series data, I split temporally: first 80% of the training period for fitting, last 20% for early stopping. All HP tuning uses `TimeSeriesSplit`, not random K-fold.

**DNN specifics**: The original architecture used L1 regularization on biases, which is harmful for BatchNormalization layers. It also used `batch_size=2048`, which gives too few gradient steps per epoch on datasets of 200k–4M rows. The final version uses L2 regularization, Huber loss (robust to return outliers), batch_size=4096, and no BatchNorm or Dropout.

## Key Takeaway

1. **OOS R² is tiny but that's normal.** Cross-sectional return prediction is fundamentally hard. Even 0.5% R² can generate profitable portfolios.
2. **Portfolio Sharpe is what matters**, not prediction MSE. ML models earn their keep by improving *rankings*, not *levels*.
3. **Random Forest is the workhorse.** It's the best performer in my experiments — nonlinear, handles mixed features natively, and doesn't need feature scaling.
4. **The engineering is the hard part.** Memory management, temporal CV, subsample HP tuning — this is where most of the actual work goes.
5. **MSRR is an interesting idea but fragile.** Optimizing directly for Sharpe ratio is appealing but produces unstable predictions.

---

*All results use the JKP global factor dataset, US common equities, train ≤2015, test ≥2016. Hyperparameter tuning via Optuna (Akiba et al., KDD 2019). Code on [GitHub](https://github.com/gunsslashroses/quant-trading-experiments).*
