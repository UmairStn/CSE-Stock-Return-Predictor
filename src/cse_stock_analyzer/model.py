"""Training, persistence, and inference for the cross-sectional return regressor."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import load_history
from .features import FEATURE_COLUMNS, SLICE_BARS, prepare_features

MAX_ABS_DAILY_RETURN = 0.40
TRAIN_SPLIT = 0.80
# Kept here only for clarity inside `train`; reuse via xgb.XGBRegressor(**MODEL_PARAMS).
MODEL_PARAMS = dict(
    n_estimators=150, max_depth=5, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    eval_metric="mae", n_jobs=-1,
)


def _target_poisoned(current_return: pd.Series, horizon: int) -> pd.Series:
    """True for rows whose target window (t+1..t+h) contains a split-sized jump.

    For horizon 1 this is the original neighbour rule (a jump at row i poisons
    the label at row i-1). For longer horizons the window widens: any daily
    return above the threshold inside the window makes the cumulative label
    untrustworthy.
    """
    bad = current_return.abs() > MAX_ABS_DAILY_RETURN
    if horizon <= 1:
        return bad.shift(-1, fill_value=False)
    window = bad.iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1]
    return window.shift(-1).fillna(False).astype(bool)


def prepare_training_frame(
    frame: pd.DataFrame, train_end: str | None = None, horizon: int = 1
) -> pd.DataFrame:
    """Create features and cumulative-horizon targets with contaminated jumps removed."""
    if horizon < 1:
        raise ValueError("horizon must be a positive number of trading sessions")
    data = frame.copy()
    if train_end:
        if "Date" not in data:
            raise ValueError("Training data must include a Date column")
        data = data[pd.to_datetime(data["Date"]) <= pd.Timestamp(train_end)].copy()
    data = prepare_features(data)

    # Cumulative close-to-close return over the next `horizon` trading rows.
    close = pd.to_numeric(data["Close"], errors="coerce")
    data["Target_Return"] = close.shift(-horizon) / close - 1

    # Split-sized jumps anywhere in the row's 5-bar slice (current bar plus
    # three previous intraslice moves) can enter any slice-scoped indicator,
    # so the whole slice is poisoned for the row. This supersedes the old
    # "current + 2 lags" rule, which only covered three of the five bars.
    bad_jump = data["Current_Return"].abs() > MAX_ABS_DAILY_RETURN
    contaminated = bad_jump.rolling(SLICE_BARS - 1, min_periods=1).max().fillna(0).astype(bool)
    contaminated |= _target_poisoned(data["Current_Return"], horizon)
    data = data.loc[~contaminated]
    return data.dropna(subset=FEATURE_COLUMNS + ["Target_Return"])


def _heldout_mae(
    columns: list[str], training: pd.DataFrame, testing: pd.DataFrame
) -> float:
    model = xgb.XGBRegressor(**MODEL_PARAMS)
    model.fit(training[columns], training["Target_Return"])
    predictions = model.predict(testing[columns])
    return float(mean_absolute_error(testing["Target_Return"], predictions))


def _ablation_report(
    training: pd.DataFrame,
    testing: pd.DataFrame,
    horizon: int,
    full_mae: float,
    report_dir: Path,
) -> Path:
    """Refit once per dropped feature to quantify each feature's marginal value.

    Trees tolerate redundant features at roughly zero cost, so features are
    kept regardless of the outcome (the 11-name drop-in artifact contract);
    this report is documentation of which features carry the weight, produced
    from the actual training data instead of judgement.
    """
    report: dict[str, Any] = {
        "horizon_days": horizon,
        "full_model_heldout_mae": full_mae,
        "features": {},
    }
    for name in FEATURE_COLUMNS:
        reduced = [column for column in FEATURE_COLUMNS if column != name]
        mae = _heldout_mae(reduced, training, testing)
        report["features"][name] = {
            "heldout_mae": mae,
            "mae_delta_vs_full": mae - full_mae,
        }
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"feature_ablation_h{horizon}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def train(
    csv_paths: list[Path],
    model_path: Path,
    metadata_path: Path,
    train_end: str | None = None,
    horizon: int = 1,
    ablation: bool = True,
    ablation_dir: Path | None = None,
) -> dict[str, Any]:
    """Train one horizon-specific model using per-security chronological splits."""
    if horizon < 1:
        raise ValueError("horizon must be a positive number of trading sessions")
    train_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []
    symbols: list[str] = []

    for path in sorted(csv_paths):
        frame = prepare_training_frame(load_history(path), train_end, horizon)
        if len(frame) < 50:
            continue
        split = int(len(frame) * TRAIN_SPLIT)
        # Embargo: the last `horizon` training rows have targets whose windows
        # reach into the test period, so no test-period close is ever seen in
        # training. Without this, overlapping windows leak across the split.
        train_frames.append(frame.iloc[: max(0, split - horizon)])
        test_frames.append(frame.iloc[split:])
        symbols.append(path.stem)

    if not train_frames:
        raise ValueError("No CSV contains enough usable observations for training")

    training = pd.concat(train_frames, ignore_index=True)
    testing = pd.concat(test_frames, ignore_index=True)
    model = xgb.XGBRegressor(**MODEL_PARAMS)
    model.fit(training[FEATURE_COLUMNS], training["Target_Return"])
    predictions = model.predict(testing[FEATURE_COLUMNS])
    actual = testing["Target_Return"]
    metrics = {
        "mae": float(mean_absolute_error(actual, predictions)),
        "mse": float(mean_squared_error(actual, predictions)),
        "r2": float(r2_score(actual, predictions)),
        "directional_accuracy": float(((predictions >= 0) == (actual.to_numpy() >= 0)).mean()),
        "zero_return_baseline_mae": float(actual.abs().mean()),
    }
    ablation_path = None
    if ablation:
        ablation_path = _ablation_report(
            training, testing, horizon, metrics["mae"],
            ablation_dir if ablation_dir is not None
            else model_path.parent.parent / "reports" / "generated",
        )
    metadata: dict[str, Any] = {
        "model_type": "XGBoost cumulative return regressor",
        "horizon_days": horizon,
        "target_definition": (
            f"Close(t + {horizon}) / Close(t) - 1 over the next {horizon} "
            "trading session(s)"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_end": train_end,
        "features": FEATURE_COLUMNS,
        "training_rows": len(training),
        "test_rows": len(testing),
        "assets": symbols,
        "metrics": metrics,
        "hyperparameters": model.get_params(),
        # Serving tripwire: every live feature value is checked against the
        # p1/p99 band it was trained on; out-of-band values warn, never vanish.
        "min_bars_required": SLICE_BARS,
        "feature_engineering": (
            "trailing 5-bar slice via build_feature_row (same code as serving)"
        ),
        "feature_ranges": {
            name: {
                "p1": float(column.quantile(0.01)),
                "p50": float(column.quantile(0.50)),
                "p99": float(column.quantile(0.99)),
            }
            for name, column in training[FEATURE_COLUMNS].items()
        },
        "feature_importances": {
            name: float(weight)
            for name, weight in zip(
                FEATURE_COLUMNS, model.feature_importances_, strict=False
            )
        },
    }
    if ablation_path is not None:
        metadata["feature_ablation_report"] = str(ablation_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(model_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return metadata


def load_model(model_path: Path) -> xgb.XGBRegressor:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run the train command first.")
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    saved_features = model.get_booster().feature_names
    if saved_features and saved_features != FEATURE_COLUMNS:
        raise ValueError("Model feature schema does not match the application feature schema")
    return model


def predict_latest(
    model: xgb.XGBRegressor, frame: pd.DataFrame, symbol: str, horizon: int = 1
) -> dict[str, Any]:
    """Predict the cumulative return over the next `horizon` trading sessions."""
    featured = prepare_features(frame).dropna(subset=FEATURE_COLUMNS)
    if featured.empty:
        raise ValueError(f"Not enough usable history to calculate features for {symbol}")
    latest = featured.iloc[-1]
    predicted_return = float(model.predict(featured.iloc[[-1]][FEATURE_COLUMNS])[0])
    date = latest.get("Date", featured.index[-1])
    return {
        "symbol": symbol,
        "observation_date": pd.Timestamp(date).date().isoformat(),
        "horizon_days": horizon,
        "predicted_return": predicted_return,
        "predicted_percent": predicted_return * 100,
        "direction": "UP" if predicted_return >= 0 else "DOWN",
    }
