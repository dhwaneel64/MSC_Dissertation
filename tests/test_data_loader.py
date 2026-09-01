"""Sanity and leakage-style checks for the data loader.

These tests read the full locked snapshot pipeline. They carry
@pytest.mark.network so they can be deselected with `pytest -m "not network"`,
and they skip gracefully if a data read fails so offline runs do not error out.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.data_loader import download_prices


@pytest.fixture(scope="module")
def spy_prices() -> pd.DataFrame:
    try:
        return download_prices(config.TICKER_SPY)
    except Exception as exc:
        pytest.skip(f"yfinance unavailable, skipping network tests: {exc}")


@pytest.mark.network
def test_index_is_monotonic_increasing(spy_prices: pd.DataFrame) -> None:
    assert spy_prices.index.is_monotonic_increasing


@pytest.mark.network
def test_data_is_fresh(spy_prices: pd.DataFrame) -> None:
    """The loaded data is exactly the locked snapshot's, not a fresher pull.

    Replaces the pre-lock freshness assertion (within 7 days of today), which
    could only fail under the locked-snapshot policy and would have passed only
    if the lock were violated. The invariant now asserted is the lock itself:
    the loaded series ends on the last SPY date inside the file named by
    config.LOCKED_SNAPSHOT_DATE.
    """
    snap_path = config.DATA_DIR / (
        f"{config.SNAPSHOT_PREFIX}{config.LOCKED_SNAPSHOT_DATE}.parquet"
    )
    snap_spy = pd.read_parquet(snap_path)["SPY"].dropna()
    assert spy_prices.index.max() == snap_spy.index.max(), (
        f"loaded SPY ends {spy_prices.index.max().date()}, locked snapshot "
        f"{config.LOCKED_SNAPSHOT_DATE} ends {snap_spy.index.max().date()}"
    )


@pytest.mark.network
def test_close_has_no_nans(spy_prices: pd.DataFrame) -> None:
    assert spy_prices["close"].isna().sum() == 0


@pytest.mark.network
def test_start_date_is_respected() -> None:
    cutoff = "2020-01-01"
    try:
        df = download_prices(config.TICKER_SPY, start=cutoff)
    except Exception as exc:
        pytest.skip(f"yfinance unavailable, skipping network test: {exc}")
    assert (df.index >= pd.Timestamp(cutoff)).all(), (
        f"download_prices returned rows before {cutoff}: {df.index.min()}"
    )


@pytest.mark.network
def test_skew_data_available_from_1993() -> None:
    """CBOE SKEW from yfinance must start at or before 1993-12-31.

    A failure (not a skip) here means the full sample cannot be used with CBOE SKEW.
    Do not truncate or backfill; investigate the data source.
    """
    try:
        skew = download_prices(config.TICKER_SKEW)
    except Exception as exc:
        pytest.skip(f"yfinance unavailable for TICKER_SKEW: {exc}")
    assert "close" in skew.columns
    assert not skew["close"].isna().any()
    assert skew.index.min() <= pd.Timestamp("1993-12-31"), (
        f"SKEW data from yfinance starts {skew.index.min().date()}, "
        "expected at or before 1993-12-31"
    )


# ── Snapshot tests (no network required) ─────────────────────────────────────

def _write_fake_snapshot(tmp_path: object, date_str: str, columns: dict) -> object:
    """Write a synthetic snapshot parquet with the given column Series."""
    import pandas as pd
    from src import config as _cfg
    snap_df = pd.DataFrame(columns)
    snap_df.index.name = "date"
    snap_path = tmp_path / f"{_cfg.SNAPSHOT_PREFIX}{date_str}.parquet"
    snap_df.to_parquet(snap_path)
    return snap_path


def test_snapshot_read_back_by_default(tmp_path, monkeypatch) -> None:
    """Without refresh, download_prices reads from the locked snapshot."""
    from src import config as _cfg
    monkeypatch.setattr(_cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(_cfg, "LOCKED_SNAPSHOT_DATE", "2020-03-01")

    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-02", periods=50)).astype("datetime64[s]")
    fake_close = pd.Series(range(1, 51), index=idx, name="SPY", dtype=float)
    _write_fake_snapshot(tmp_path, "2020-03-01", {"SPY": fake_close})

    result = download_prices("SPY")

    assert len(result) == 50
    assert list(result.columns) == ["close"]
    assert (result["close"].values == fake_close.values).all()


def test_snapshot_index_dtype_is_datetime64s(tmp_path, monkeypatch) -> None:
    """Index dtype is datetime64[s] after round-trip through snapshot parquet."""
    from src import config as _cfg
    monkeypatch.setattr(_cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(_cfg, "LOCKED_SNAPSHOT_DATE", "2020-02-14")

    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-02", periods=30)).astype("datetime64[s]")
    _write_fake_snapshot(tmp_path, "2020-02-14", {"SPY": pd.Series(range(1, 31), index=idx, dtype=float)})

    result = download_prices("SPY")
    assert str(result.index.dtype) == "datetime64[s]", (
        f"Expected datetime64[s], got {result.index.dtype}"
    )


def test_snapshot_round_trip_exact(tmp_path, monkeypatch) -> None:
    """Values written to snapshot and read back are byte-identical."""
    from src import config as _cfg
    monkeypatch.setattr(_cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(_cfg, "LOCKED_SNAPSHOT_DATE", "2020-03-15")

    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-02", periods=50)).astype("datetime64[s]")
    original = pd.Series(
        [i * 1.0001 for i in range(50)], index=idx, name="VIX", dtype=float
    )
    _write_fake_snapshot(tmp_path, "2020-03-15", {"VIX": original})

    result = download_prices("^VIX")

    assert len(result) == 50
    for i, (got, exp) in enumerate(zip(result["close"].tolist(), original.tolist())):
        assert got == exp, f"Value mismatch at row {i}: got {got}, expected {exp}"


def test_snapshot_skew_gap_dates_absent_on_readback(tmp_path, monkeypatch) -> None:
    """Dates absent from SKEW daily (the gap dates) are not filled on read-back.

    The combined snapshot parquet has NaN in the SKEW column for dates where SPY/VIX
    have data but SKEW does not.  Read-back drops those NaN rows, returning the same
    sparse SKEW series that produced the 4-month cboe_skew gaps in the monthly frame.
    """
    from src import config as _cfg
    monkeypatch.setattr(_cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(_cfg, "LOCKED_SNAPSHOT_DATE", "2020-01-20")

    # SPY has 10 dates; SKEW has only 5 of them (every other day = simulated gaps).
    spy_idx = pd.DatetimeIndex(pd.bdate_range("2020-01-02", periods=10)).astype("datetime64[s]")
    skew_idx = spy_idx[::2]  # 5 dates

    snap_df = pd.DataFrame(index=spy_idx)
    snap_df.index.name = "date"
    snap_df["SPY"] = pd.Series(range(1, 11), index=spy_idx, dtype=float)
    snap_df["SKEW"] = pd.Series(range(1, 6), index=skew_idx, dtype=float)
    snap_path = tmp_path / f"{_cfg.SNAPSHOT_PREFIX}2020-01-20.parquet"
    snap_df.to_parquet(snap_path)

    result = download_prices("^SKEW")

    assert len(result) == 5, f"Expected 5 rows (gap dates absent), got {len(result)}"
    assert result["close"].isna().sum() == 0, "Gap rows must be absent, not NaN-filled"
    pd.testing.assert_index_equal(result.index, skew_idx, check_names=False)


def test_snapshot_reads_locked_date_not_newest(tmp_path, monkeypatch) -> None:
    """download_prices reads from LOCKED_SNAPSHOT_DATE specifically, not the newest file."""
    from src import config as _cfg
    monkeypatch.setattr(_cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(_cfg, "LOCKED_SNAPSHOT_DATE", "2020-01-01")

    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-02", periods=5)).astype("datetime64[s]")
    # Locked snapshot: SPY close = 100
    _write_fake_snapshot(tmp_path, "2020-01-01", {"SPY": pd.Series([100.0] * 5, index=idx)})
    # Newer snapshot present but NOT the locked date: SPY close = 200
    _write_fake_snapshot(tmp_path, "2020-06-01", {"SPY": pd.Series([200.0] * 5, index=idx)})

    result = download_prices("SPY")
    assert result["close"].iloc[0] == 100.0, "Should read from LOCKED_SNAPSHOT_DATE, not the newer file"


def test_snapshot_raises_when_locked_snapshot_absent(tmp_path, monkeypatch) -> None:
    """With no locked snapshot on disk, download_prices raises FileNotFoundError."""
    from src import config as _cfg
    monkeypatch.setattr(_cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(_cfg, "LOCKED_SNAPSHOT_DATE", "2020-01-01")

    # tmp_path is empty — the locked snapshot does not exist.
    with pytest.raises(FileNotFoundError, match="Locked snapshot"):
        download_prices("SPY")


def test_skew_monthly_alignment_equals_first_trading_day() -> None:
    """cboe_skew at month t equals skew_daily close at the first trading day of t.

    Uses synthetic daily data with distinct values per day so any misalignment
    (month-end, forward date, or inter-month average) produces a detectable error.
    """
    from src.vrp import resample_to_month_start

    idx = pd.bdate_range("2020-01-02", periods=130)
    skew_daily = pd.Series(
        range(1, len(idx) + 1), index=idx, dtype=float, name="close"
    )
    skew_monthly = resample_to_month_start(skew_daily)

    for t in skew_monthly.index[1:4]:
        assert skew_monthly.at[t] == skew_daily.at[t], (
            f"cboe_skew at {t.date()} is {skew_monthly.at[t]}, "
            f"but skew_daily at {t.date()} is {skew_daily.at[t]}"
        )


def test_skew_monthly_leakage_input_truncation() -> None:
    """Strong leakage test: truncating daily SKEW input at t must not change cboe_skew at t.

    Cuts the underlying Series at t (not just the dates list) and verifies the
    resampled+reindexed value at t is byte-identical. This rules out resample_to_month_start
    or reindex pulling a value from a future date within the same month or a later month.
    """
    from src.vrp import resample_to_month_start

    idx = pd.bdate_range("2020-01-02", periods=200)
    skew_daily = pd.Series(
        range(1, len(idx) + 1), index=idx, dtype=float, name="close"
    )
    skew_monthly_full = resample_to_month_start(skew_daily)
    t = skew_monthly_full.index[3]

    cboe_full = skew_monthly_full.reindex([t]).at[t]

    skew_monthly_trunc = resample_to_month_start(skew_daily.loc[:t])
    assert t in skew_monthly_trunc.index, "t absent after truncation at t"
    cboe_trunc = skew_monthly_trunc.reindex([t]).at[t]

    assert cboe_full == cboe_trunc, (
        f"Leakage detected: full={cboe_full}, truncated={cboe_trunc}"
    )
    assert cboe_full == skew_daily.at[t], (
        f"cboe_skew at {t.date()} is {cboe_full} but raw daily close is {skew_daily.at[t]}"
    )
