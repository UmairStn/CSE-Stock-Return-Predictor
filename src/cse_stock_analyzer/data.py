"""Price data acquisition and CSV loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from .features import REQUIRED_OHLCV, validate_ohlcv
from .universe import to_yahoo_symbol

MIN_USABLE_ROWS = 200  # TRAINING-time guard only; serving requires SLICE_BARS (features.py)


def fetch_history(symbol: str, period: str = "5y") -> pd.DataFrame:
    yahoo_symbol = to_yahoo_symbol(symbol)
    frame = yf.download(
        yahoo_symbol, period=period, interval="1d", auto_adjust=False,
        progress=False, threads=False,
    )
    if frame is None or frame.empty:
        raise ValueError(f"No Yahoo Finance data available for {yahoo_symbol}")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    validate_ohlcv(frame)
    frame = frame[REQUIRED_OHLCV].dropna(subset=["Close"]).sort_index()
    if len(frame) < MIN_USABLE_ROWS:
        raise ValueError(f"Only {len(frame)} rows available; at least {MIN_USABLE_ROWS} required")
    frame.index.name = "Date"
    return frame


def save_history(frame: pd.DataFrame, symbol: str, data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{to_yahoo_symbol(symbol)}.csv"
    frame.to_csv(path, index_label="Date")
    return path


def load_history(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["Date"])
    validate_ohlcv(frame)
    return frame.sort_values("Date").reset_index(drop=True)
