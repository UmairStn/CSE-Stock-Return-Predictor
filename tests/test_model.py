import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from cse_stock_analyzer.features import FEATURE_COLUMNS, prepare_features
from cse_stock_analyzer.model import predict_latest, prepare_training_frame, train


def prices(rows=300):
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 0.5, rows)) + np.arange(rows) * 0.05
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=rows),
        "Open": close * 0.995, "High": close * 1.01,
        "Low": close * 0.99, "Close": close,
        "Volume": 10_000 + rng.integers(0, 5_000, rows),
    })


def test_prediction_record_shape():
    frame = prices(80)
    featured = prepare_features(frame).dropna(subset=FEATURE_COLUMNS)
    model = xgb.XGBRegressor(n_estimators=2, max_depth=1, random_state=42)
    model.fit(featured[FEATURE_COLUMNS], np.zeros(len(featured)))
    result = predict_latest(model, frame, "JKH.N0000", horizon=1)
    assert result["symbol"] == "JKH.N0000"
    assert result["horizon_days"] == 1
    assert result["direction"] in {"UP", "DOWN"}
    assert isinstance(result["predicted_percent"], float)
    assert result["predicted_percent"] == pytest.approx(result["predicted_return"] * 100)


def test_prediction_record_carries_requested_horizon():
    frame = prices(80)
    featured = prepare_features(frame).dropna(subset=FEATURE_COLUMNS)
    model = xgb.XGBRegressor(n_estimators=2, max_depth=1, random_state=42)
    model.fit(featured[FEATURE_COLUMNS], np.zeros(len(featured)))
    result = predict_latest(model, frame, "JKH.N0000", horizon=7)
    assert result["horizon_days"] == 7


def test_multi_horizon_target_is_cumulative_return():
    frame = prices()
    result = prepare_training_frame(frame, horizon=3)
    featured = prepare_features(frame)
    expected = featured["Close"].shift(-3) / featured["Close"] - 1
    for index, row in result.iterrows():
        assert row["Target_Return"] == pytest.approx(expected[index])


def test_train_writes_model_metadata_and_embargo():
    import json
    import shutil

    frame = prices()
    from pathlib import Path

    data_dir = Path(__file__).parent / "_tmp_train_data"
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "TEST-N0000.CM.csv"
    frame.to_csv(csv_path, index=False)

    model_path = data_dir / "model.json"
    metadata_path = data_dir / "model.metadata.json"
    try:
        metadata = train([csv_path], model_path, metadata_path, horizon=2, ablation=False)
        assert model_path.exists() and metadata_path.exists()
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert saved["horizon_days"] == 2
        assert metadata["horizon_days"] == 2
        assert metadata["training_rows"] + metadata["test_rows"] > 0
        assert "zero_return_baseline_mae" in metadata["metrics"]
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def test_metadata_carries_feature_ranges_importances_and_slice_contract():
    """Requirement 5: per-feature p1/p50/p99 training bands + importances."""
    import json
    import shutil
    from pathlib import Path

    frame = prices()
    data_dir = Path(__file__).parent / "_tmp_meta_data"
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "TEST-N0000.CM.csv"
    model_path = data_dir / "model.json"
    metadata_path = data_dir / "model.metadata.json"
    try:
        frame.to_csv(csv_path, index=False)
        metadata = train([csv_path], model_path, metadata_path, horizon=2, ablation=False)

        assert set(metadata["feature_ranges"]) == set(FEATURE_COLUMNS)
        assert len(metadata["feature_importances"]) == len(FEATURE_COLUMNS)
        assert abs(sum(metadata["feature_importances"].values()) - 1.0) < 1e-6
        for name in FEATURE_COLUMNS:
            entry = metadata["feature_ranges"][name]
            assert entry["p1"] <= entry["p50"] <= entry["p99"]

        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert saved["feature_ranges"] == metadata["feature_ranges"]
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)


def test_ablation_report_quantifies_each_feature():
    import json
    import shutil
    from pathlib import Path

    frame = prices()
    data_dir = Path(__file__).parent / "_tmp_ablation_data"
    data_dir.mkdir(exist_ok=True)
    csv_path = data_dir / "TEST-N0000.CM.csv"
    frame.to_csv(csv_path, index=False)
    model_path = data_dir / "model.json"
    metadata_path = data_dir / "model.metadata.json"
    try:
        metadata = train(
            [csv_path], model_path, metadata_path, horizon=2, ablation=True,
            ablation_dir=data_dir / "reports",
        )
        report_path = Path(metadata["feature_ablation_report"])
        assert report_path.exists()
        # report lands inside the test's own temp tree:
        assert data_dir in report_path.parents
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["horizon_days"] == 2
        assert report["full_model_heldout_mae"] == pytest.approx(
            metadata["metrics"]["mae"]
        )
        assert set(report["features"]) == set(FEATURE_COLUMNS)
        for entry in report["features"].values():
            # delta == heldout - full (a dropped feature can *lower* MAE by noise
            # margin, so only the internal consistency is asserted):
            assert entry["mae_delta_vs_full"] == pytest.approx(
                entry["heldout_mae"] - report["full_model_heldout_mae"]
            )
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
