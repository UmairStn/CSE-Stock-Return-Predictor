import datetime as dt

import numpy as np
import pandas as pd
import pytest

from cse_stock_analyzer.features import (
    FEATURE_COLUMNS,
    SLICE_BARS,
    build_feature_row,
    cse_payload_to_ohlcv,
    featurize_history,
    prepare_ohlcv,
)
from cse_stock_analyzer.model import prepare_training_frame

# The exact record observed from the live CSE API (2026-10-23).
OBSERVED_RECORD = {
    "h": 18.1, "l": 17.6, "o": None, "s": 108923744269, "q": 6308277,
    "p": 17.7, "c": None, "pc": None, "t": 1792348200000, "n": None, "id": 297,
}


def prices(rows=120):
    close = 100 + np.sin(np.arange(rows) / 3) + np.arange(rows) * 0.1
    frame = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=rows),
        "Open": close * 0.995,  # real Open: must be discarded by prepare_ohlcv
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": np.arange(rows) * 100 + 1_000,
    })
    return frame


def cse_record(day: dt.date, close: float, quantity: float = 1_000_000.0) -> dict:
    """A live-payload-shaped record for a given Colombo trade date."""
    stamp = dt.datetime(day.year, day.month, day.day, 17, 0, tzinfo=_COLOMBO_TZ())
    return {
        "h": close * 1.005, "l": close * 0.995, "o": None,
        "s": 1234567, "q": quantity, "p": close, "c": None, "pc": None,
        "t": int(stamp.timestamp() * 1000), "n": None, "id": 1,
    }


def _COLOMBO_TZ() -> dt.tzinfo:
    from cse_stock_analyzer.features import _COLOMBO_TZ as tz

    return tz


def test_slice_parity_any_row_depends_only_on_its_trailing_5_bars():
    """THE core guarantee: features for row t are identical whether computed on
    the full history or on exactly its trailing 5-bar slice."""
    frame = prices(240)
    full = featurize_history(frame).dropna(subset=FEATURE_COLUMNS)
    positions = list(range(full.index.min() + 20, full.index.max(), 7))[:20]
    for position in positions:
        window = frame.iloc[position - SLICE_BARS + 1 : position + 1]
        solo = featurize_history(window).dropna(subset=FEATURE_COLUMNS)
        pd.testing.assert_frame_equal(
            solo[FEATURE_COLUMNS].reset_index(drop=True),
            full.loc[[position], FEATURE_COLUMNS].reset_index(drop=True),
        )


def test_open_is_synthesized_from_previous_close_and_real_open_discarded():
    frame = prices()
    cleaned = prepare_ohlcv(frame)
    close = frame["Close"]
    assert cleaned["Open"].iloc[0] is pd.NA or np.isnan(cleaned["Open"].iloc[0])
    np.testing.assert_allclose(cleaned["Open"].iloc[1:], close.iloc[:-1])


def test_validate_accepts_frames_without_open_column():
    frame = prices().drop(columns=["Open"])
    cleaned = prepare_ohlcv(frame)
    np.testing.assert_allclose(cleaned["Open"].iloc[1:], cleaned["Close"].iloc[:-1])


def test_five_bars_are_sufficient_and_four_are_not():
    frame = prices(5)
    result = featurize_history(frame)
    assert len(result) == 5
    assert np.isfinite(result.iloc[-1][FEATURE_COLUMNS].astype(float)).all()
    assert result.iloc[:-1][FEATURE_COLUMNS].isna().all().all()

    slice_frame = frame.drop(columns=["Open"])
    with pytest.raises(ValueError, match="exactly 5 rows"):
        build_feature_row(slice_frame.iloc[-4:])

    four_rows = featurize_history(prices(4))
    assert four_rows[FEATURE_COLUMNS].isna().all().all()


def test_open_pct_is_the_negative_current_return_to_first_order():
    """Open(t) = Close(t-1), so Open_Pct = Close(t-1)/Close(t) - 1 while
    Current_Return = Close(t)/Close(t-1) - 1. In real arithmetic they are
    negatives up to O(r^2); the test pins that exact relationship."""
    result = featurize_history(prices(60)).dropna(subset=["Open_Pct", "Current_Return"])
    residual = (result["Open_Pct"] + result["Current_Return"]).abs()
    bound = 2.0 * result["Current_Return"].abs() ** 2 + 1e-12
    assert (residual < bound).all()


def test_rsi_bounds_and_pure_states():
    up = pd.DataFrame({
        "High": [10, 11, 12, 13, 14], "Low": [10, 11, 12, 13, 14],
        "Close": [10.0, 11.0, 12.0, 13.0, 14.0], "Volume": [1, 1, 1, 1, 1],
    })
    assert build_feature_row(up)["RSI_3"] == 100.0

    flat = up.copy()
    flat["Close"] = 20.0
    flat["High"] = 20.0
    flat["Low"] = 20.0
    assert build_feature_row(flat)["RSI_3"] == 50.0

    # 0-100 on real history: the /100 scale bug in the serving copy would break this.
    result = featurize_history(prices(120)).dropna(subset=["RSI_3"])
    assert result["RSI_3"].between(0, 100).all()

    # Hand-check the Cutler formula: closes 10,11,12,10.5,11.5 -> deltas 1,1,-1.5,1
    # gains mean 2/3, losses mean 0.5 -> RSI = 100 - 100/(1 + 4/3).
    mixed_close = [10.0, 11.0, 12.0, 10.5, 11.5]
    mixed = pd.DataFrame({
        "High": [value + 0.5 for value in mixed_close],
        "Low": [value - 0.5 for value in mixed_close],
        "Close": mixed_close, "Volume": [1, 1, 1, 1, 1],
    })
    expected = 100.0 - 100.0 / (1.0 + (2.0 / 3.0) / 0.5)
    row = build_feature_row(mixed)
    assert row["RSI_3"] == pytest.approx(expected)
    assert np.isfinite(row["High_Pct"]) and np.isfinite(row["Low_Pct"])


def test_rel_volume_uses_three_bar_mean_including_current_bar():
    frame = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=5),
        "High": [1.0] * 5, "Low": [1.0] * 5, "Close": [1.0] * 5,
        "Volume": [1_000, 2_000, 3_000, 4_000, 5_000],
    })
    row = featurize_history(frame).iloc[-1]
    assert row["Rel_Volume"] == pytest.approx(5_000 / 4_000)  # (3+4+5)/3
    assert row["MACD_Pct"] == 0.0  # flat closes -> ema3 == ema5
    assert row["MACD_Hist_Pct"] == 0.0


def test_training_frame_uses_slice_features_and_drops_slice_jumps():
    """A >40% jump anywhere in a row's 5-bar slice poisons that feature row.

    Jump bar J: the 45% close enters the hourly features of rows J..J+3
    (their trailing slices all contain close(J)) -- those are feature-poisoned.
    Row J-1's *target* window contains the jump, so it is poisoned separately
    by `_target_poisoned`.
    """
    frame = prices(120).copy()
    jump = 30
    frame.loc[jump, "Close"] *= 1.45  # >40% daily jump
    frame.loc[jump, "High"] *= 1.45

    result = prepare_training_frame(frame, horizon=1)
    labels = set(result.index)
    assert {jump, jump + 1, jump + 2, jump + 3}.isdisjoint(labels)  # feature side
    assert jump - 1 not in labels                                   # target side
    assert jump - 4 in labels                                       # last clean row


def test_zero_volume_rows_are_removed_by_shared_ingestion():
    frame = prices()
    frame.loc[10, "Volume"] = 0
    assert len(featurize_history(frame)) == len(frame) - 1


def test_cse_payload_mapping():
    """The observed record maps exactly: p->Close, h/l->High/Low, q->Volume,
    t->Date; o/c/pc/n/id/s are never read; Open is the previous close."""
    # OBSERVED_RECORD's Colombo date is 2026-10-19 (t = 1792348200000).
    today = dt.date(2026, 10, 20)
    payload = [OBSERVED_RECORD]  # newest record, feed order
    payload += [
        cse_record(today - dt.timedelta(days=offset), 17.0 + offset * 0.05)
        for offset in range(8, 1, -1)  # 8 older completed sessions: Oct 12..18
    ]

    now = dt.datetime(2026, 10, 20, 10, 0, tzinfo=_COLOMBO_TZ())
    frame = cse_payload_to_ohlcv(payload, now=now)

    assert list(frame["Close"]).count(17.7) == 1  # observed record mapped in
    observed = frame.iloc[-1]  # newest completed session (today's bar not present)
    np.testing.assert_allclose(
        [observed["Close"], observed["High"], observed["Low"]], [17.7, 18.1, 17.6]
    )
    np.testing.assert_allclose(frame["Open"].iloc[1:], frame["Close"].iloc[:-1])
    # decoy fields never read; q -> Volume:
    assert frame["Volume"].iloc[-1] == 6308277
    # fully ready for the model within the same module:
    features = featurize_history(frame).dropna(subset=FEATURE_COLUMNS)
    assert len(features) == len(frame) - 4


def test_cse_payload_rejects_impossible_bars_never_clamps():
    bad = dict(OBSERVED_RECORD)
    bad["p"] = 30.0  # outside [l, h] -> impossible bar
    today = dt.date(2026, 10, 20)
    payload = [bad] + [cse_record(today - dt.timedelta(days=offset), 17.0 + offset * 0.1)
                       for offset in range(8, 0, -1)]
    now = dt.datetime(2026, 10, 21, 0, 0, tzinfo=_COLOMBO_TZ())
    frame = cse_payload_to_ohlcv(payload, now=now)
    # the impossible bar (p=30 vs h=18.1) is rejected, never clamped:
    assert 30.0 not in set(frame["Close"])
    assert (frame["Close"] <= frame["High"]).all()
    assert (frame["Low"] <= frame["Close"]).all()


def test_cse_payload_refuses_when_fewer_than_five_completed_bars():
    today = dt.date(2026, 10, 20)
    payload = [cse_record(today - dt.timedelta(days=offset), 17.0)
               for offset in range(4, 0, -1)]
    now = dt.datetime(2026, 10, 21, 0, 0, tzinfo=_COLOMBO_TZ())
    with pytest.raises(ValueError, match="completed bar"):
        cse_payload_to_ohlcv(payload, now=now)


def test_open_session_is_kept_when_drop_is_disabled():
    """Shadow-testing/parity mode: today's bar can be kept for offline checks."""
    today = dt.date(2026, 10, 20)
    payload = [cse_record(today - dt.timedelta(days=offset), 17.0)
               for offset in range(7, 0, -1)]
    now = dt.datetime(2026, 10, 21, 0, 0, tzinfo=_COLOMBO_TZ())
    frame = cse_payload_to_ohlcv(payload, now=now, drop_open_session=False)
    assert len(frame) == 7


def test_twentythree_bar_payload_matches_the_5_bar_slice_model():
    """Predicting with ~23 live bars must be bit-identical to using only the
    trailing 5 -- the extra headroom cannot silently change the features."""
    start = dt.date(2026, 9, 25)
    records = [cse_record(start + dt.timedelta(days=offset), 15.0 + offset * 0.05)
               for offset in range(23)]
    payload = list(reversed(records))
    now = dt.datetime(2026, 10, 21, 0, 0, tzinfo=_COLOMBO_TZ())
    frame = cse_payload_to_ohlcv(payload, now=now, drop_open_session=False)

    features_from_23 = featurize_history(frame).dropna(subset=FEATURE_COLUMNS)
    window = featurize_history(frame.iloc[-5:]).dropna(subset=FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(
        window[FEATURE_COLUMNS].reset_index(drop=True),
        features_from_23.tail(1)[FEATURE_COLUMNS].reset_index(drop=True),
    )


def test_ema_matches_pandas_reference():
    rng = np.random.default_rng(11)
    for length in (5, 6, 40):
        values = rng.normal(size=length).cumsum() + 100.0
        from cse_stock_analyzer.features import _ema

        for span in (3, 5):
            reference = pd.Series(values).ewm(span=span, adjust=False).mean().to_numpy()
            np.testing.assert_allclose(_ema(values, span), reference, atol=1e-12)
