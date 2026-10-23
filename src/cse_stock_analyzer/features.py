"""Shared feature engineering for model training and inference.

TRAIN = SERVE CONTRACT
----------------------
Every feature value for row ``t`` is derived from exactly its trailing
``SLICE_BARS`` completed bars (rows ``t-4 .. t``) through
:func:`build_feature_row`. The serving backend (TradeVision) runs this same
module byte-for-byte against the live CSE payload, so a row can never contain
information the live feed cannot reproduce:

* the CSE daily record has no open price (``o`` is always ``null``), so
  ``Open`` is *defined* as the previous close on both sides
  (:func:`prepare_ohlcv`);
* every recursive indicator (EMA/MACD) is seeded inside the 5-bar slice and
  never runs across the full history, so values are identical no matter how
  much history the caller happens to hold.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "Open_Pct", "High_Pct", "Low_Pct", "Rel_Volume", "RSI_3",
    "Return_Lag1", "Return_Lag2", "SMA_3_Pct", "EMA_3_Pct",
    "MACD_Pct", "MACD_Hist_Pct",
]
REQUIRED_OHLCV = ["High", "Low", "Close", "Volume"]
SLICE_BARS = 5

# Sri Lanka has used a fixed UTC+05:30 offset since 2006 (no DST), so a fixed
# offset keeps the trade-date conversion dependency-free (no tz database).
_COLOMBO_TZ = _dt.timezone(_dt.timedelta(hours=5, minutes=30), name="Asia/Colombo")


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Check that an OHLCV frame carries every column the featurizer reads.

    ``Open`` is deliberately *not* required: the CSE feed has no open price,
    and :func:`prepare_ohlcv` synthesizes it from the previous close.
    """
    missing = [column for column in REQUIRED_OHLCV if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Price data is empty")


def drop_non_trading_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove session rows that saw no trading.

    This removes Yahoo's forward-filled ghost rows (Volume=0) and, on the
    serving side, any CSE bars reported with no traded quantity. Both sides
    filter *before* slicing so a 5-bar slice always corresponds to the same
    five trading sessions on either side.
    """
    validate_ohlcv(df)
    volume = pd.to_numeric(df["Volume"], errors="coerce")
    return df.loc[volume > 0].copy()


def prepare_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Shared ingestion contract: validate -> drop non-trading -> synthesize Open.

    The real ``Open`` column (Yahoo's, or a live feed's) is always overwritten
    with ``Open(t) = Close(t-1)``, because the CSE payload reports ``o: null``:
    on both training and serving data the "open" is by definition the previous
    close. The first row of a history has no previous close (``NaN``); its
    features are dropped later because a full slice is unavailable.
    """
    clean = drop_non_trading_rows(df)
    close = pd.to_numeric(clean["Close"], errors="coerce").astype("float64")
    clean = clean.assign(Open=close.shift(1))
    return clean


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    """Recursive ``adjust=False`` EMA over a small array.

    Matches ``pd.Series(values).ewm(span=span, adjust=False).mean()`` exactly,
    but runs a 5-element loop instead of building pandas objects per row, so
    featurizing ~60k training rows stays fast.
    """
    alpha = 2.0 / (span + 1.0)
    out = np.empty(values.shape[0], dtype="float64")
    out[0] = values[0]
    for i in range(1, values.shape[0]):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def build_feature_row(slice_df: pd.DataFrame) -> dict[str, float]:
    """Compute the 11 model features for the LAST bar of a trailing slice.

    ``slice_df`` must contain exactly :data:`SLICE_BARS` completed OHLCV rows,
    oldest first. This is the *only* place feature math exists: training
    (:func:`featurize_history`) and the live backend call the same function on
    the same shape of data, so there is no second implementation to drift.

    Slice rows are s0..s4 (oldest -> newest); all features describe s4.
    """
    validate_ohlcv(slice_df)
    if len(slice_df) != SLICE_BARS:
        raise ValueError(
            f"build_feature_row needs exactly {SLICE_BARS} rows, got {len(slice_df)}"
        )
    close = pd.to_numeric(slice_df["Close"], errors="coerce").to_numpy(dtype="float64")
    high = pd.to_numeric(slice_df["High"], errors="coerce").to_numpy(dtype="float64")
    low = pd.to_numeric(slice_df["Low"], errors="coerce").to_numpy(dtype="float64")
    volume = pd.to_numeric(slice_df["Volume"], errors="coerce").to_numpy(dtype="float64")

    features: dict[str, float] = {}
    latest = close[-1]
    if not np.isfinite(latest) or latest <= 0:
        return {name: np.nan for name in FEATURE_COLUMNS}

    # 1. Open_Pct -- Open(s4) is the previous close (o is null in the CSE feed).
    features["Open_Pct"] = close[-2] / latest - 1
    # 2-3. Intraday extreme position.
    features["High_Pct"] = high[-1] / latest - 1
    features["Low_Pct"] = low[-1] / latest - 1
    # 4. Today's traded quantity vs the 3-bar mean *including* today.
    #    A within-slice ratio is invariant to volume units, so a difference
    #    between Yahoo's and CSE's quantity conventions cannot skew it.
    volume_mean = volume[-3:].mean()
    if np.isfinite(volume_mean) and volume_mean > 0:
        features["Rel_Volume"] = float(volume[-1]) / volume_mean
    else:
        features["Rel_Volume"] = np.nan
    # 5. Cutler-style RSI over daily changes s2-s1 .. s4-s3 (equal-weight mean,
    #    0-100 scale; never rescaled anywhere downstream).
    deltas = np.diff(close)[-3:]
    avg_gain = float(np.clip(deltas, 0.0, None).mean())
    avg_loss = float(-np.clip(deltas, None, 0.0).mean())
    if not np.isfinite(avg_gain) or not np.isfinite(avg_loss):
        features["RSI_3"] = np.nan
    elif avg_gain == 0 and avg_loss == 0:
        features["RSI_3"] = 50.0
    elif avg_loss == 0:
        features["RSI_3"] = 100.0
    else:
        features["RSI_3"] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    # 6-8. Raw momentum and short-trend position.
    features["Return_Lag1"] = close[-2] / close[-3] - 1
    features["Return_Lag2"] = close[-3] / close[-4] - 1
    features["SMA_3_Pct"] = close[-3:].mean() / latest - 1
    # 9-11. Slice-scoped short MACD (EMA3 - EMA5, signal EMA3), seeded at s0:
    # the only definition that can be bit-identical on any history length.
    ema3 = _ema(close, 3)
    ema5 = _ema(close, 5)
    macd = ema3 - ema5
    signal = _ema(macd, 3)
    features["EMA_3_Pct"] = ema3[-1] / latest - 1
    features["MACD_Pct"] = macd[-1] / latest
    features["MACD_Hist_Pct"] = (macd[-1] - signal[-1]) / latest
    return features


def featurize_history(df: pd.DataFrame) -> pd.DataFrame:
    """Featurize every completed bar of a daily OHLCV history.

    Runs :func:`build_feature_row` on each bar's trailing 5-bar slice. The
    first ``SLICE_BARS - 1`` rows have no full slice yet and carry ``NaN``
    features (clearing them is the caller's guard, via ``dropna``).
    """
    validate_ohlcv(df)
    clean = prepare_ohlcv(df).reset_index(drop=True)
    result = clean.copy()
    new_columns = [
        (name, np.full(len(clean), np.nan, dtype="float64"))
        for name in FEATURE_COLUMNS
    ]
    for end in range(SLICE_BARS - 1, len(clean)):
        row = build_feature_row(clean.iloc[end - (SLICE_BARS - 1) : end + 1])
        for name, column in new_columns:
            column[end] = row[name]
    for name, column in new_columns:
        result[name] = column
    close = pd.to_numeric(result["Close"], errors="coerce").astype("float64")
    result["Current_Return"] = close.pct_change(fill_method=None)
    return result


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Alias of :func:`featurize_history` kept for the public analyst surface."""
    return featurize_history(df)


def _trade_date(raw_timestamp: float) -> _dt.date:
    """Convert a CSE epoch timestamp (ms, or s tolerated) to its Colombo date."""
    seconds = raw_timestamp if abs(raw_timestamp) < 1e12 else raw_timestamp / 1000.0
    return _dt.datetime.fromtimestamp(seconds, tz=_COLOMBO_TZ).date()


def cse_payload_to_ohlcv(
    payload: Sequence[Mapping],
    *,
    now: _dt.datetime | None = None,
    drop_open_session: bool = True,
) -> pd.DataFrame:
    """Map raw CSE API price-history records to the canonical OHLCV frame.

    Observed record shape::

        {"h": 18.1, "l": 17.6, "o": null, "s": 108923744269,
         "q": 6308277, "p": 17.7, "c": null, "pc": null,
         "t": 1792348200000, "n": null, "id": 297}

    Mapping and rules (mirrors the training-time hygiene exactly):

    * ``p`` -> Close, ``h`` -> High, ``l`` -> Low, ``q`` -> Volume,
      ``t`` -> trade date (Asia/Colombo, fixed UTC+05:30);
    * ``o``, ``c``, ``pc``, ``n``, ``id``, ``s`` are *never read* -- Open is
      synthesized from the previous close by :func:`prepare_ohlcv`;
    * a bar with ``not (l <= p <= h)`` or missing/invalid price fields is
      rejected (never clamped); malformed records are skipped;
    * duplicate trade dates are collapsed to the latest record;
    * today's still-open session is dropped by default (the model only ever
      predicts *from* a completed close -- override ``drop_open_session`` for
      replay/shadow testing);
    * raises ``ValueError`` when fewer than :data:`SLICE_BARS` completed bars
      survive: the predictor refuses instead of guessing.
    """
    if not payload:
        raise ValueError("CSE payload is empty")
    now = now or _dt.datetime.now(tz=_COLOMBO_TZ)
    today = now.astimezone(_COLOMBO_TZ).date()

    parsed: list[tuple[float, _dt.date, dict[str, float]]] = []
    for record in payload:
        try:
            close = float(record["p"])
            high = float(record["h"])
            low = float(record["l"])
            trade_date = _trade_date(float(record["t"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not all(np.isfinite(value) for value in (high, low, close)):
            continue
        if not low <= close <= high:
            continue  # impossible bar: reject, never clamp
        quantity: float
        try:
            quantity = float(record["q"]) if record.get("q") is not None else np.nan
        except (TypeError, ValueError):
            quantity = np.nan
        parsed.append((float(record["t"]), trade_date, {
            "Date": pd.Timestamp(trade_date), "High": high, "Low": low,
            "Close": close, "Volume": quantity,
        }))
    if not parsed:
        raise ValueError("CSE payload contained no price bars")

    parsed.sort(key=lambda entry: entry[0])
    frame = pd.DataFrame(
        [entry[2] for entry in parsed],
        columns=["Date", "High", "Low", "Close", "Volume"],
    )
    frame = frame.drop_duplicates(subset="Date", keep="last")
    if drop_open_session and frame["Date"].iloc[-1].date() == today:
        frame = frame.iloc[:-1]
    if len(frame) < SLICE_BARS:
        raise ValueError(
            f"CSE payload has {len(frame)} completed bar(s); "
            f"at least {SLICE_BARS} are required for prediction"
        )
    return prepare_ohlcv(frame).reset_index(drop=True)
