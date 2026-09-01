"""Data loading utilities for fetching daily price series from Yahoo Finance.

Snapshot behaviour
------------------
download_prices(ticker) reads from the locked snapshot named by
config.LOCKED_SNAPSHOT_DATE in config.DATA_DIR, and raises if that file is
absent; there is no fallback to any other snapshot or to a live fetch, so every
run is tied to the locked data state.  Pass refresh=True to re-fetch from
yfinance and write a new dated snapshot file
(data/raw_snapshot_YYYY-MM-DD.parquet); the new file becomes the default read
only when config.LOCKED_SNAPSHOT_DATE is updated to name it.

The snapshot is a single parquet with one column per ticker (SPY, VIX, SKEW)
and a union DatetimeIndex.  Dates where a ticker has no data appear as NaN in
that column; read-back drops those NaN rows so the returned Series is identical
to a live fetch.  ^SKEW data gaps are therefore preserved, not filled.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from . import config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ticker_key(ticker: str) -> str:
    """Map a ticker symbol to its parquet column name (strip leading ^)."""
    return ticker.lstrip("^")


def _snapshot_path(date_str: str) -> Path:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return config.DATA_DIR / f"{config.SNAPSHOT_PREFIX}{date_str}.parquet"


def _fetch_live(ticker: str, start: str | None) -> pd.DataFrame:
    """Download daily adjusted closes from yfinance and return a clean DataFrame."""
    if start is None:
        start = config.DATA_START_DATE

    raw = yf.download(ticker, start=start, progress=False, auto_adjust=False)

    if raw.empty:
        raise ValueError(
            f"yfinance returned an empty frame for ticker {ticker!r} from {start}"
        )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    if "Close" not in raw.columns:
        raise ValueError(
            f"'Close' column missing in yfinance result for ticker {ticker!r}"
        )

    df = raw[["Close"]].rename(columns={"Close": "close"}).copy()
    df.index = pd.DatetimeIndex(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.index.name = "date"

    if df["close"].isna().any():
        raise ValueError(f"NaN values present in close column for ticker {ticker!r}")

    return df


def _read_ticker_from_snapshot(snap: Path, key: str) -> pd.DataFrame:
    """Read one ticker column from a snapshot parquet, returning a clean DataFrame."""
    snap_df = pd.read_parquet(snap)
    close = snap_df[key].dropna()
    close.index = pd.DatetimeIndex(close.index).normalize().astype("datetime64[s]")
    close.index.name = "date"
    close.name = "close"
    return pd.DataFrame({"close": close})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_prices(
    ticker: str,
    start: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Download daily adjusted closing prices for a single ticker.

    Default (refresh=False): read from the snapshot named
    config.LOCKED_SNAPSHOT_DATE in config.DATA_DIR. Raises FileNotFoundError
    if that file is absent. There is no fallback to a live fetch, so results
    are always tied to the locked data state.

    refresh=True: always fetch live from yfinance, add/update this ticker's
    column in data/raw_snapshot_<today>.parquet (creating it if absent), and
    return the freshly fetched data.

    The returned DataFrame has a tz-naive DatetimeIndex (name "date",
    dtype datetime64[s]) and a single "close" column.  Raises ValueError when
    the live response is empty or contains NaN close values.
    """
    key = _ticker_key(ticker)

    if refresh:
        df = _fetch_live(ticker, start)
        today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
        snap = _snapshot_path(today)
        new_col = df["close"].rename(key).to_frame()
        new_col.index.name = "date"
        if snap.exists():
            existing = pd.read_parquet(snap)
            # Use concat to union the indices; assignment would silently drop
            # rows in new_col that are not already in existing.index.
            existing = pd.concat(
                [existing.drop(columns=[key], errors="ignore"), new_col],
                axis=1,
            )
        else:
            existing = new_col
        existing.index.name = "date"
        existing.to_parquet(snap)
        return df

    # Default: read from the locked snapshot (config.LOCKED_SNAPSHOT_DATE).
    snap = _snapshot_path(config.LOCKED_SNAPSHOT_DATE)
    if not snap.exists():
        raise FileNotFoundError(
            f"Locked snapshot {snap} not found. "
            f"Set REFRESH_SNAPSHOT = True in the notebook snapshot cell to fetch "
            f"from yfinance and write it, then update config.LOCKED_SNAPSHOT_DATE."
        )
    snap_df = pd.read_parquet(snap)
    if key not in snap_df.columns:
        raise KeyError(
            f"Ticker key {key!r} not in locked snapshot {snap.name}. "
            f"Available: {list(snap_df.columns)}"
        )
    result = _read_ticker_from_snapshot(snap, key)
    if start is not None:
        result = result[result.index >= pd.Timestamp(start)]
    return result
