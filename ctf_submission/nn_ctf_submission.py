"""CTF submission-ready neural network model.

This script is designed to be submitted to https://jkpfactors.com/ctf/submit.
It exposes a `main(chars, features, daily_ret)` function with the required
signature and returns weights with columns: id, eom, w.

You can also run it locally (see __main__) to generate the CSV upload file.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input, LeakyReLU


# =============================================================================
# Editable configuration
# =============================================================================

WeightMethod = Literal["equal", "char_rank_weighted"]
TrainWindowMode = Literal["rolling", "expanding"]


@dataclass
class SubmissionConfig:
    # Universe
    use_us_only: bool = True
    require_common: bool = False  # set True only if `common` exists in chars

    # Feature selection
    feature_source: Literal["ctf_features", "manual"] = "ctf_features"
    manual_feature_list: tuple[str, ...] = (
        "beta_60m",
        "market_equity",
        "be_me",
        "op_at",
        "at_gr1",
        "qmj",
        "ret_1_0",
        "ret_6_1",
        "oaccruals_at",
        "dbnetis_at",
        "debt_at",
        "niq_at_chg1",
        "ni_be",
        "rvol_252d",
    )

    # Optional date guards (None = no extra restriction)
    min_train_eom: str | None = None  # e.g. "1970-01-31"
    max_train_eom: str | None = None  # e.g. "2015-12-31"
    min_test_eom: str | None = None   # e.g. "2016-01-31"
    max_test_eom: str | None = None   # e.g. "2024-12-31"

    # Training window logic
    train_window_mode: TrainWindowMode = "rolling"
    rolling_train_years: int = 10
    min_train_months_required: int = 36

    # Target preprocessing
    target_col: str = "ret_exc_lead1m"
    winsorize_target: bool = True
    winsor_lower_q: float = 0.01
    winsor_upper_q: float = 0.99

    # Portfolio construction (long-short percentiles)
    lower_pct: float = 0.30  # 30/70 default
    upper_pct: float = 0.70
    weight_method: WeightMethod = "equal"
    max_weight_per_leg: float | None = None  # e.g. 0.2 for 20% cap

    # NN architecture / training
    random_seed: int = 42
    layer1_units: int = 128
    layer2_units: int = 64
    activation: Literal["relu", "tanh", "leaky_relu"] = "leaky_relu"
    leaky_relu_negative_slope: float = 0.01
    l2_reg: float = 1e-6
    learning_rate: float = 1e-3
    loss: Literal["mse", "huber"] = "huber"
    batch_size: int = 4096
    epochs: int = 30
    early_stopping_patience: int = 5


CONFIG = SubmissionConfig()


# =============================================================================
# Helpers
# =============================================================================


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def _resolve_eom_column(df: pd.DataFrame) -> str:
    if "eom" in df.columns:
        return "eom"
    if "eom_ret" in df.columns:
        return "eom_ret"
    raise ValueError("chars must contain either 'eom' or 'eom_ret'")


def _resolve_target_column(chars: pd.DataFrame, cfg: SubmissionConfig) -> str:
    if cfg.target_col in chars.columns:
        return cfg.target_col
    if cfg.target_col == "ret_exc_lead1m" and "ret_exc_lead1m" in chars.columns:
        return "ret_exc_lead1m"
    raise ValueError(f"Target column '{cfg.target_col}' not found in chars")


def _get_feature_list(chars: pd.DataFrame, features: pd.DataFrame, cfg: SubmissionConfig) -> list[str]:
    if cfg.feature_source == "ctf_features":
        if "features" not in features.columns:
            raise ValueError("features DataFrame must contain a 'features' column")
        feat = [f for f in features["features"].astype(str).tolist() if f in chars.columns]
    else:
        feat = [f for f in cfg.manual_feature_list if f in chars.columns]

    if not feat:
        raise ValueError("No usable features found after applying feature selection")
    return sorted(dict.fromkeys(feat))


def _winsorize_monthly(series: pd.Series, groups: pd.Series, lo: float, hi: float) -> pd.Series:
    def _clip(x: pd.Series) -> pd.Series:
        q_lo, q_hi = x.quantile([lo, hi])
        return x.clip(q_lo, q_hi)

    return series.groupby(groups).transform(_clip)


def _apply_optional_date_filter(df: pd.DataFrame, eom_col: str, min_eom: str | None, max_eom: str | None) -> pd.DataFrame:
    out = df
    if min_eom is not None:
        out = out[out[eom_col] >= pd.Timestamp(min_eom)]
    if max_eom is not None:
        out = out[out[eom_col] <= pd.Timestamp(max_eom)]
    return out


def _prepare_chars(chars: pd.DataFrame, cfg: SubmissionConfig) -> tuple[pd.DataFrame, str, str]:
    d = chars.copy()
    eom_col = _resolve_eom_column(d)
    d[eom_col] = pd.to_datetime(d[eom_col])
    if "ctff_test" not in d.columns:
        raise ValueError("chars must contain ctff_test")
    if "id" not in d.columns:
        raise ValueError("chars must contain id")

    # Universe filters
    if cfg.use_us_only:
        if "excntry" not in d.columns:
            raise ValueError("use_us_only=True requires 'excntry' column in chars")
        d = d[d["excntry"] == "USA"].copy()

    if cfg.require_common and "common" in d.columns:
        d = d[d["common"] == 1].copy()

    target = _resolve_target_column(d, cfg)
    if cfg.winsorize_target:
        d[target] = _winsorize_monthly(
            d[target].astype(float),
            d[eom_col],
            cfg.winsor_lower_q,
            cfg.winsor_upper_q,
        )

    return d, eom_col, target


def _normalize_leg_weights(raw_w: pd.Series, side: pd.Series, cap: float | None) -> pd.Series:
    out = pd.Series(0.0, index=raw_w.index)

    long_mask = side == 1
    short_mask = side == -1

    for mask in (long_mask, short_mask):
        if not mask.any():
            continue
        w = raw_w.loc[mask].astype(float).clip(lower=0)
        if (w <= 0).all():
            w = pd.Series(1.0, index=w.index)
        w = w / w.sum()
        if cap is not None:
            w = np.minimum(w, cap)
            s = w.sum()
            if s > 0:
                w = w / s
            else:
                w = pd.Series(1.0 / len(w), index=w.index)
        out.loc[mask] = w

    return out


def _weights_from_predictions(
    frame: pd.DataFrame,
    pred_col: str,
    lower_pct: float,
    upper_pct: float,
    method: WeightMethod,
    cap: float | None,
) -> pd.DataFrame:
    f = frame.copy()
    f["position"] = 0

    lo = f[pred_col].quantile(lower_pct)
    hi = f[pred_col].quantile(upper_pct)

    f.loc[f[pred_col] <= lo, "position"] = -1
    f.loc[f[pred_col] >= hi, "position"] = 1

    active = f[f["position"] != 0].copy()
    if active.empty:
        return pd.DataFrame(columns=["id", "eom", "w"])

    if method == "equal":
        active["raw_w"] = 1.0
    elif method == "char_rank_weighted":
        # More extreme predicted values receive larger absolute allocation.
        ranks = active[pred_col].rank(method="average", pct=True)
        active["raw_w"] = np.where(active["position"] == 1, ranks, 1.0 - ranks)
        active["raw_w"] = active["raw_w"].clip(lower=1e-8)
    else:
        raise ValueError(f"Unsupported weight method: {method}")

    active["norm_w"] = _normalize_leg_weights(active["raw_w"], active["position"], cap)
    active["w"] = np.where(active["position"] == 1, active["norm_w"], -active["norm_w"])

    return active[["id", "eom", "w"]].copy()


def _build_model(n_features: int, cfg: SubmissionConfig) -> keras.Model:
    keras.backend.clear_session()

    loss_obj = tf.keras.losses.Huber() if cfg.loss == "huber" else "mse"

    if cfg.activation == "leaky_relu":
        model = keras.models.Sequential(
            [
                Input(shape=(n_features,)),
                Dense(cfg.layer1_units, kernel_regularizer=regularizers.L2(cfg.l2_reg)),
                LeakyReLU(negative_slope=cfg.leaky_relu_negative_slope),
                Dense(cfg.layer2_units, kernel_regularizer=regularizers.L2(cfg.l2_reg)),
                LeakyReLU(negative_slope=cfg.leaky_relu_negative_slope),
                Dense(1, activation="linear"),
            ]
        )
    else:
        model = keras.models.Sequential(
            [
                Input(shape=(n_features,)),
                Dense(
                    cfg.layer1_units,
                    activation=cfg.activation,
                    kernel_regularizer=regularizers.L2(cfg.l2_reg),
                ),
                Dense(
                    cfg.layer2_units,
                    activation=cfg.activation,
                    kernel_regularizer=regularizers.L2(cfg.l2_reg),
                ),
                Dense(1, activation="linear"),
            ]
        )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss=loss_obj,
    )
    return model


# =============================================================================
# Required entry point for CTF
# =============================================================================


def main(chars: pd.DataFrame, features: pd.DataFrame, daily_ret: pd.DataFrame) -> pd.DataFrame:
    """CTF-required entry point.

    Args:
        chars: Stock characteristics DataFrame.
        features: Feature-name DataFrame (expects a `features` column).
        daily_ret: Historical daily returns DataFrame (unused by this model).

    Returns:
        DataFrame with columns: id, eom, w
    """
    del daily_ret  # Unused in this architecture

    cfg = CONFIG
    _set_seeds(cfg.random_seed)

    d, eom_col, target_col = _prepare_chars(chars, cfg)
    feature_list = _get_feature_list(d, features, cfg)

    # Keep needed columns and drop rows without target/date/id.
    keep_cols = ["id", eom_col, "ctff_test", target_col] + feature_list
    d = d[[c for c in keep_cols if c in d.columns]].copy()
    d = d.dropna(subset=["id", eom_col, "ctff_test", target_col])

    # Fill feature NA with month-wise medians to avoid dropping too much data.
    d[feature_list] = d.groupby(eom_col)[feature_list].transform(lambda x: x.fillna(x.median()))
    d[feature_list] = d[feature_list].fillna(0.0)

    # Standardize naming for downstream logic.
    d = d.rename(columns={eom_col: "eom", target_col: "target"})
    d["eom"] = pd.to_datetime(d["eom"])

    # Optional extra train/test date constraints.
    train_df = d[d["ctff_test"] == 0].copy()
    test_df = d[d["ctff_test"] == 1].copy()

    train_df = _apply_optional_date_filter(train_df, "eom", cfg.min_train_eom, cfg.max_train_eom)
    test_df = _apply_optional_date_filter(test_df, "eom", cfg.min_test_eom, cfg.max_test_eom)

    if test_df.empty:
        raise ValueError("No test rows after filtering; cannot produce submission weights")

    test_months = sorted(test_df["eom"].drop_duplicates().tolist())

    output_parts: list[pd.DataFrame] = []

    for test_month in test_months:
        if cfg.train_window_mode == "rolling":
            start_month = test_month - pd.DateOffset(years=cfg.rolling_train_years)
            this_train = train_df[(train_df["eom"] < test_month) & (train_df["eom"] >= start_month)].copy()
        else:
            this_train = train_df[train_df["eom"] < test_month].copy()

        this_test = test_df[test_df["eom"] == test_month].copy()

        train_month_count = this_train["eom"].nunique()
        if train_month_count < cfg.min_train_months_required or this_test.empty:
            # If insufficient history, emit zero weights for this month.
            z = this_test[["id", "eom"]].copy()
            z["w"] = 0.0
            output_parts.append(z)
            continue

        X_train = this_train[feature_list].to_numpy(dtype=np.float32)
        y_train = this_train["target"].to_numpy(dtype=np.float32)

        X_test = this_test[feature_list].to_numpy(dtype=np.float32)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)

        model = _build_model(n_features=X_train_scaled.shape[1], cfg=cfg)

        # Temporal split inside each train window (last 20% as validation)
        split_idx = max(1, int(len(X_train_scaled) * 0.8))
        if split_idx >= len(X_train_scaled):
            split_idx = max(1, len(X_train_scaled) - 1)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.early_stopping_patience,
                restore_best_weights=True,
            )
        ]

        model.fit(
            X_train_scaled[:split_idx],
            y_train[:split_idx],
            validation_data=(X_train_scaled[split_idx:], y_train[split_idx:]),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            verbose=0,
            callbacks=callbacks,
        )

        preds = model.predict(X_test_scaled, verbose=0).reshape(-1)
        pred_frame = this_test[["id", "eom"]].copy()
        pred_frame["pred"] = preds

        w = _weights_from_predictions(
            pred_frame,
            pred_col="pred",
            lower_pct=cfg.lower_pct,
            upper_pct=cfg.upper_pct,
            method=cfg.weight_method,
            cap=cfg.max_weight_per_leg,
        )

        # If no active names from thresholding, emit zeros for the month.
        if w.empty:
            z = this_test[["id", "eom"]].copy()
            z["w"] = 0.0
            output_parts.append(z)
        else:
            output_parts.append(w)

    out = pd.concat(output_parts, ignore_index=True)
    out = out.dropna(subset=["id", "eom", "w"]).copy()

    # Enforce required output schema and types
    out["id"] = pd.to_numeric(out["id"], errors="coerce")
    out = out.dropna(subset=["id"])
    out["id"] = out["id"].astype(np.int64)
    out["eom"] = pd.to_datetime(out["eom"]).dt.normalize()
    out["w"] = pd.to_numeric(out["w"], errors="coerce")
    out = out.dropna(subset=["w"])

    out = out[["id", "eom", "w"]].sort_values(["eom", "id"]).reset_index(drop=True)

    if out.empty:
        raise ValueError("Submission output is empty")

    return out


# =============================================================================
# Local runner (not used by CTF evaluation)
# =============================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CTF NN submission script locally")
    parser.add_argument("--chars", required=True, help="Path to ctff_chars.parquet")
    parser.add_argument("--features", required=True, help="Path to ctff_features.parquet")
    parser.add_argument("--daily-ret", required=True, help="Path to ctff_daily_ret.parquet")
    parser.add_argument("--output-csv", required=True, help="Where to write weights CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    chars_df = pd.read_parquet(args.chars)
    features_df = pd.read_parquet(args.features)
    daily_ret_df = pd.read_parquet(args.daily_ret)

    weights = main(chars=chars_df, features=features_df, daily_ret=daily_ret_df)
    weights.to_csv(args.output_csv, index=False)

    print("Wrote weights CSV:", args.output_csv)
    print("Rows:", len(weights))
    print("Months:", weights["eom"].nunique())
    print(weights.head())
