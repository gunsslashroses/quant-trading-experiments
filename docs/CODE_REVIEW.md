# Code Review Notes

Issues, observations, and suggestions found during code review of the original notebooks.

## Issues Found

### 1. Cumulative Return Scaling (Notebook 02)
**Location**: Strategy evaluation loop, cumret_dict line.
```python
cumret_dict[strategy_name] = (1 + rets).cumprod() - 1
```
There is an inline comment: *"confirm this scaling is correct. Adding this manually because spread was calculated as Long - Short, instead of long*0.5 - short*0.5"*. This is correct for a dollar-neutral long/short portfolio where you invest $1 long and $1 short. The spread return is simply Long − Short, and `(1 + spread).cumprod() - 1` gives the cumulative return. However, the plot title says "Cumulative Log Returns" while showing simple cumulative returns — this is a **labeling inconsistency**.

**Recommendation**: Use `np.log(1 + rets).cumsum()` if you want log returns, or fix the chart title.

### 2. DeprecationWarning: `include_groups` in GroupBy.apply
**Location**: Multiple functions using `.groupby(...).apply()`.

Pandas warns that grouping columns will be excluded from the operation in future versions. The refactored library code explicitly passes `include_groups=False` where needed to future-proof against this change.

### 3. Duplicate column 'qmj' in JKP data
**Location**: Data loading.

The raw JKP CSV may contain `qmj` and `qmj.1` (duplicate column names). The code handles this with:
```python
df_jkp = df_jkp.loc[:, ~df_jkp.columns.duplicated(keep="first")]
```
This is a defensive workaround. A better approach would be to investigate why the duplicate exists in the WRDS query and fix the SQL.

### 4. Mixed dtypes warning on column 3 (iid)
**Location**: `pd.read_csv('jkp_data.csv')`.

Column `iid` has mixed types (some numeric, some string like "01"). Fix by specifying `dtype={"iid": str}` in the read_csv call, or by using `low_memory=False`.

### 5. DNN: multiple hyperparameter tuning issues (FIXED)
**Location**: Notebook 04, DNN section.

All issues below were fixed by switching to **Optuna** (Bayesian optimization with TPE sampler) for HP search, plus architectural corrections.

**5a. Look-ahead bias in final training validation split.**
The original code used `validation_split=0.2` in `model.fit()`. Keras implements this as taking the last 20% of the *array* (which, for unsorted panel data, may mix time periods). For time-series panel data, this risks leaking future information into the validation set. **Fixed**: explicit temporal split — first 80% of training data for fitting, last 20% (chronologically) for early stopping.

**5b. Only L1 regularization was tuned.**
Learning rate was left at Adam's default (1e-3). In practice, the L1–LR interaction is strong: high L1 with high LR can zero out weights too aggressively, while low L1 with low LR may underfit. **Fixed**: Optuna jointly searches L1 (`1e-6` to `1e-1`, log), learning rate (`1e-5` to `1e-2`, log), dropout rate (`0.0` to `0.5`), and layer widths.

**5c. Only 10 epochs during CV search.**
With `batch_size=2048` on datasets of ~50k–500k rows, each epoch has very few gradient steps (e.g., 50k/2048 ≈ 24 steps). 10 epochs = ~240 steps total — the model barely starts learning. **Fixed**: 100 epochs per CV fold with early stopping (patience=10).

**5d. batch_size=2048 too large.**
Reduces the number of weight updates per epoch and can hurt generalization. **Fixed**: batch_size=512.

**5e. L1 regularization on biases.**
`bias_regularizer=regularizers.L1(l1=1e-05)` pushes biases toward zero. This is harmful for BatchNormalization layers (which rely on learned bias-like parameters) and for the output layer's intercept. It's also unusual practice. **Fixed**: removed bias regularization, added Dropout instead.

**5f. Only 2 CV folds.**
TimeSeriesSplit(n_splits=2) gives high variance in HP estimates. **Fixed**: 5-fold `TimeSeriesSplit`.

**5g. GPU memory.**
The original code doesn't set GPU memory growth limits. On shared environments this can cause OOM crashes. Consider adding `tf.config.experimental.set_memory_growth(gpu, True)`.

### 6. AdaBoost: hyperparameter tuning issues (FIXED)
**Location**: Notebook 04, AdaBoost section.

All issues below were fixed by switching to **Optuna** for HP search.

**6a. Search space too narrow.**
Only searched `max_depth=[1,2]` and `n_estimators=[10,50,100]`. **Fixed**: Optuna searches `max_depth` in `[1, 6]`, `n_estimators` in `[50, 1000]` (log-scale), `learning_rate` in `[1e-4, 2.0]` (log-scale) — all with proper scale handling.

**6b. Manual fine grid with wrong step types.**
`n_estimators ± 20` and `learning_rate * {0.5, 1.0, 2.0}` is a mismatch. If coarse best was n_estimators=10, the fine grid [-10, 10, 30] is broken. **Fixed**: Optuna handles mixed int/float/log types natively — no manual grid construction needed.

**6c. Fine search froze max_depth.**
The fine grid only searched the coarse-best depth. **Fixed**: Optuna explores all HPs continuously across all trials.

**6d. Only 2 CV folds.**
Same issue as DNN. **Fixed**: 5-fold `TimeSeriesSplit`.

### 7. MSRR loss function: simplified portfolio simulation
**Location**: Notebook 04, MSRR section.

The Sharpe loss function approximates portfolio returns as `predictions * targets`. This is a simplified proxy — in reality, returns should be based on ranked/sorted positions. The approximation assumes that prediction magnitude is proportional to position sizing, which may not hold. The model is still useful for learning directional signals, but the in-sample Sharpe from this loss is not directly comparable to the portfolio Sharpe from sorted-decile evaluation.

### 8. AdaBoost uses `estimator` parameter (sklearn ≥1.2)
**Location**: Notebook 04, AdaBoost section.

The code correctly uses `estimator=` instead of the deprecated `base_estimator=`. This is compatible with scikit-learn ≥1.2.

### 9. IPCA predict with `mean_factor=True`
**Location**: Notebook 04, IPCA section.

Using `mean_factor=True` for OOS prediction means the model assumes the latent factors in the test period equal their historical average. This is a conservative assumption — in practice, you might want to estimate factors using the most recent data point.

## Style & Structure Suggestions (addressed in refactor)

- **Functions separated into modules**: the original notebooks defined all helper functions inline, making them hard to test and reuse.
- **Consistent parameter naming**: standardized `id_col`, `date_col`, `ret_col` across all functions.
- **Type hints added**: all public functions now have full type annotations.
- **Docstrings added**: NumPy-style docstrings for all public functions.
