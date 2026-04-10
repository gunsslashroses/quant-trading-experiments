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

### 5. DNN model: potential GPU memory issues
**Location**: Notebook 04, DNN section.

The code manually calls `keras.backend.clear_session()` and `gc.collect()` between CV folds to prevent GPU OOM. This is good practice, but the notebook doesn't set a GPU memory growth limit, which could still cause issues on shared GPU environments. Consider adding:
```python
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### 6. MSRR loss function: simplified portfolio simulation
**Location**: Notebook 04, MSRR section.

The Sharpe loss function approximates portfolio returns as `predictions * targets`. This is a simplified proxy — in reality, returns should be based on ranked/sorted positions. The approximation assumes that prediction magnitude is proportional to position sizing, which may not hold. The model is still useful for learning directional signals, but the in-sample Sharpe from this loss is not directly comparable to the portfolio Sharpe from sorted-decile evaluation.

### 7. AdaBoost uses `estimator` parameter (sklearn ≥1.2)
**Location**: Notebook 04, AdaBoost section.

The code correctly uses `estimator=` instead of the deprecated `base_estimator=`. This is compatible with scikit-learn ≥1.2.

### 8. IPCA predict with `mean_factor=True`
**Location**: Notebook 04, IPCA section.

Using `mean_factor=True` for OOS prediction means the model assumes the latent factors in the test period equal their historical average. This is a conservative assumption — in practice, you might want to estimate factors using the most recent data point.

## Style & Structure Suggestions (addressed in refactor)

- **Functions separated into modules**: the original notebooks defined all helper functions inline, making them hard to test and reuse.
- **Consistent parameter naming**: standardized `id_col`, `date_col`, `ret_col` across all functions.
- **Type hints added**: all public functions now have full type annotations.
- **Docstrings added**: NumPy-style docstrings for all public functions.
