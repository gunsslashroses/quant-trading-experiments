#!/usr/bin/env python3
"""End-to-end test of DNN and AdaBoost on synthetic data.

Uses a small dataset with strong signal to quickly validate model correctness.
"""
import gc
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


def make_synthetic_data(n_stocks=100, n_months=60, n_features=10, seed=42):
    rng = np.random.default_rng(seed)
    months = pd.date_range("2010-01-01", periods=n_months, freq="MS")
    rows = []
    for m in months:
        for i in range(n_stocks):
            feats = rng.normal(0, 1, n_features)
            true_signal = 0.03 * feats[0] - 0.02 * feats[1] + 0.015 * feats[2]
            ret = true_signal + rng.normal(0, 0.08)
            rows.append({
                "id": i, "month_date": m, "ret_exc_lead1m_w": ret,
                **{f"f{j}": feats[j] for j in range(n_features)},
            })
    df = pd.DataFrame(rows)
    feat_cols = [f"f{j}" for j in range(n_features)]
    X = df[feat_cols].astype(float)
    y = df["ret_exc_lead1m_w"].astype(float)
    meta = df[["id", "month_date", "ret_exc_lead1m_w"]]
    return X, y, meta


def run_adaboost_original(X_train, y_train, X_test, y_test):
    """ORIGINAL notebook code."""
    base = DecisionTreeRegressor(random_state=42)
    ada = AdaBoostRegressor(estimator=base, random_state=42)
    tscv = TimeSeriesSplit(n_splits=2)

    coarse_grid = {
        "estimator__max_depth": [1, 2],
        "n_estimators": [10, 50, 100],
        "learning_rate": [0.001, 0.01, 0.1, 1.0],
    }
    s1 = GridSearchCV(ada, coarse_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1)
    s1.fit(X_train, y_train)
    best = s1.best_params_
    print(f"  Coarse best: {best} (score={s1.best_score_:.6f})")

    fine_grid = {
        "estimator__max_depth": [best["estimator__max_depth"]],
        "n_estimators": [max(1, best["n_estimators"] - 20), best["n_estimators"], best["n_estimators"] + 20],
        "learning_rate": [best["learning_rate"] * 0.5, best["learning_rate"], best["learning_rate"] * 2.0],
    }
    s2 = GridSearchCV(ada, fine_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1)
    s2.fit(X_train, y_train)
    best_f = s2.best_params_
    print(f"  Fine best: {best_f} (score={s2.best_score_:.6f})")

    pred = s2.best_estimator_.predict(X_test)
    return pred, best_f


def run_adaboost_fixed(X_train, y_train, X_test, y_test):
    """FIXED AdaBoost with improved HP tuning."""
    base = DecisionTreeRegressor(random_state=42)
    ada = AdaBoostRegressor(estimator=base, random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)

    coarse_grid = {
        "estimator__max_depth": [1, 2, 3, 4],
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
    }
    s1 = GridSearchCV(ada, coarse_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1)
    s1.fit(X_train, y_train)
    best = s1.best_params_
    print(f"  Coarse best: {best} (score={s1.best_score_:.6f})")

    best_n = best["n_estimators"]
    best_lr = best["learning_rate"]
    best_depth = best["estimator__max_depth"]

    fine_grid = {
        "estimator__max_depth": sorted(set([max(1, best_depth - 1), best_depth, best_depth + 1])),
        "n_estimators": sorted(set([max(10, int(best_n * 0.7)), best_n, int(best_n * 1.3)])),
        "learning_rate": sorted(set([best_lr * 0.5, best_lr * 0.75, best_lr, best_lr * 1.5, best_lr * 2.0])),
    }
    s2 = GridSearchCV(ada, fine_grid, cv=tscv, scoring="neg_mean_squared_error", n_jobs=-1)
    s2.fit(X_train, y_train)
    best_f = s2.best_params_
    print(f"  Fine best: {best_f} (score={s2.best_score_:.6f})")

    pred = s2.best_estimator_.predict(X_test)
    return pred, best_f


def run_dnn_original(X_train_sc, y_train, X_test_sc, y_test):
    """ORIGINAL notebook DNN."""
    import tensorflow as tf
    import tensorflow.keras as keras
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import BatchNormalization, Dense, Input

    def build_nn(l1_reg):
        keras.backend.clear_session()
        m = keras.models.Sequential([
            Input(shape=(X_train_sc.shape[1],)),
            BatchNormalization(),
            Dense(32, activation="relu", kernel_regularizer=regularizers.L1(l1=l1_reg),
                  bias_regularizer=regularizers.L1(l1=1e-05)),
            BatchNormalization(),
            Dense(16, activation="relu", kernel_regularizer=regularizers.L1(l1=l1_reg),
                  bias_regularizer=regularizers.L1(l1=1e-05)),
            BatchNormalization(),
            Dense(1, activation="linear"),
        ])
        m.compile(optimizer="adam", loss="mse")
        return m

    tscv = TimeSeriesSplit(n_splits=2)
    best_l1, best_loss = 1e-4, float("inf")
    for l1 in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        losses = []
        for tr_i, val_i in tscv.split(X_train_sc):
            with tf.device("/CPU:0"):
                model = build_nn(l1)
                cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
                h = model.fit(X_train_sc[tr_i], y_train.iloc[tr_i].values.astype(np.float32),
                              epochs=10, batch_size=2048,
                              validation_data=(X_train_sc[val_i], y_train.iloc[val_i].values.astype(np.float32)),
                              callbacks=[cb], verbose=0)
                losses.append(min(h.history["val_loss"]))
            keras.backend.clear_session(); del model; gc.collect()
        avg = np.mean(losses)
        print(f"  L1={l1:.5f} -> {avg:.6f}")
        if avg < best_loss: best_loss, best_l1 = avg, l1

    # Fine
    for l1 in [best_l1 * 0.5, best_l1, best_l1 * 2.0]:
        losses = []
        for tr_i, val_i in tscv.split(X_train_sc):
            with tf.device("/CPU:0"):
                model = build_nn(l1)
                cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)
                h = model.fit(X_train_sc[tr_i], y_train.iloc[tr_i].values.astype(np.float32),
                              epochs=10, batch_size=2048,
                              validation_data=(X_train_sc[val_i], y_train.iloc[val_i].values.astype(np.float32)),
                              callbacks=[cb], verbose=0)
                losses.append(min(h.history["val_loss"]))
            keras.backend.clear_session(); del model; gc.collect()
        avg = np.mean(losses)
        if avg < best_loss: best_loss, best_l1 = avg, l1

    print(f"  Best L1: {best_l1}")

    # Final — PROBLEM: validation_split=0.2 is random, not temporal
    with tf.device("/CPU:0"):
        model = build_nn(best_l1)
        cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        model.fit(X_train_sc, y_train.values.astype(np.float32),
                  epochs=50, batch_size=1024, validation_split=0.2, callbacks=[cb], verbose=0)
        pred = model.predict(X_test_sc, verbose=0).flatten()
    keras.backend.clear_session(); gc.collect()
    return pred, best_l1


def run_dnn_fixed(X_train_sc, y_train, X_test_sc, y_test):
    """FIXED DNN with improved HP tuning."""
    import tensorflow as tf
    import tensorflow.keras as keras
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input

    def build_nn(l1_reg, lr=1e-3):
        keras.backend.clear_session()
        m = keras.models.Sequential([
            Input(shape=(X_train_sc.shape[1],)),
            BatchNormalization(),
            Dense(64, activation="relu", kernel_regularizer=regularizers.L1(l1=l1_reg)),
            Dropout(0.1),
            BatchNormalization(),
            Dense(32, activation="relu", kernel_regularizer=regularizers.L1(l1=l1_reg)),
            Dropout(0.1),
            BatchNormalization(),
            Dense(1, activation="linear"),
        ])
        m.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")
        return m

    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = [{"l1": l1, "lr": lr}
                  for l1 in [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
                  for lr in [1e-4, 5e-4, 1e-3]]
    best_params, best_loss = None, float("inf")

    for p in param_grid:
        losses = []
        for tr_i, val_i in tscv.split(X_train_sc):
            with tf.device("/CPU:0"):
                model = build_nn(p["l1"], p["lr"])
                cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
                h = model.fit(X_train_sc[tr_i], y_train.iloc[tr_i].values.astype(np.float32),
                              epochs=30, batch_size=512,
                              validation_data=(X_train_sc[val_i], y_train.iloc[val_i].values.astype(np.float32)),
                              callbacks=[cb], verbose=0)
                losses.append(min(h.history["val_loss"]))
            keras.backend.clear_session(); del model; gc.collect()
        avg = np.mean(losses)
        if avg < best_loss: best_loss, best_params = avg, p

    print(f"  Best: L1={best_params['l1']}, LR={best_params['lr']} (loss={best_loss:.6f})")

    # Temporal validation split for final training
    n_train = int(len(X_train_sc) * 0.8)
    with tf.device("/CPU:0"):
        model = build_nn(best_params["l1"], best_params["lr"])
        cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        model.fit(X_train_sc[:n_train], y_train.iloc[:n_train].values.astype(np.float32),
                  epochs=100, batch_size=512,
                  validation_data=(X_train_sc[n_train:], y_train.iloc[n_train:].values.astype(np.float32)),
                  callbacks=[cb], verbose=0)
        pred = model.predict(X_test_sc, verbose=0).flatten()
    keras.backend.clear_session(); gc.collect()
    return pred, best_params


def main():
    print("Generating synthetic data (100 stocks x 60 months)...")
    X, y, meta = make_synthetic_data(n_stocks=100, n_months=60, n_features=10)

    split = meta["month_date"].unique()[48]  # 80/20 split
    train_mask = meta["month_date"] < split
    test_mask = meta["month_date"] >= split

    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
    X_test_sc = scaler.transform(X_test).astype(np.float32)

    # Baseline
    lr_pred = LinearRegression().fit(X_train, y_train).predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)
    lr_sa = np.mean(np.sign(y_test) == np.sign(lr_pred))
    print(f"\nLinear Regression: R²={lr_r2:.4f}, SignAcc={lr_sa:.2%}")

    results = {"LR": {"r2": lr_r2, "sign_acc": lr_sa, "mse": mean_squared_error(y_test, lr_pred)}}

    print("\n--- AdaBoost ORIGINAL ---")
    t0 = time.time()
    pred, params = run_adaboost_original(X_train, y_train, X_test, y_test)
    r2, sa = r2_score(y_test, pred), np.mean(np.sign(y_test) == np.sign(pred))
    print(f"  Result: R²={r2:.4f}, SignAcc={sa:.2%} ({time.time()-t0:.1f}s)")
    results["AdaBoost_orig"] = {"r2": r2, "sign_acc": sa, "mse": mean_squared_error(y_test, pred), "params": params}

    print("\n--- AdaBoost FIXED ---")
    t0 = time.time()
    pred, params = run_adaboost_fixed(X_train, y_train, X_test, y_test)
    r2, sa = r2_score(y_test, pred), np.mean(np.sign(y_test) == np.sign(pred))
    print(f"  Result: R²={r2:.4f}, SignAcc={sa:.2%} ({time.time()-t0:.1f}s)")
    results["AdaBoost_fixed"] = {"r2": r2, "sign_acc": sa, "mse": mean_squared_error(y_test, pred), "params": params}

    print("\n--- DNN ORIGINAL ---")
    t0 = time.time()
    pred, l1 = run_dnn_original(X_train_sc, y_train, X_test_sc, y_test)
    r2, sa = r2_score(y_test, pred), np.mean(np.sign(y_test) == np.sign(pred))
    print(f"  Result: R²={r2:.4f}, SignAcc={sa:.2%} ({time.time()-t0:.1f}s)")
    results["DNN_orig"] = {"r2": r2, "sign_acc": sa, "mse": mean_squared_error(y_test, pred)}

    print("\n--- DNN FIXED ---")
    t0 = time.time()
    pred, params = run_dnn_fixed(X_train_sc, y_train, X_test_sc, y_test)
    r2, sa = r2_score(y_test, pred), np.mean(np.sign(y_test) == np.sign(pred))
    print(f"  Result: R²={r2:.4f}, SignAcc={sa:.2%} ({time.time()-t0:.1f}s)")
    results["DNN_fixed"] = {"r2": r2, "sign_acc": sa, "mse": mean_squared_error(y_test, pred)}

    print("\n" + "=" * 55)
    print(f"{'Model':<20} {'R²':>8} {'MSE':>10} {'Sign%':>8}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<20} {r['r2']:>8.4f} {r['mse']:>10.6f} {r['sign_acc']:>7.2%}")


if __name__ == "__main__":
    main()
