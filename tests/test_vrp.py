import pandas as pd
import pytest
from src.vrp import resample_to_month_start


def _make_daily(start: str, end: str, value: float, name: str) -> pd.Series:
    idx = pd.bdate_range(start, end)
    return pd.Series(value, index=idx, name=name, dtype=float)


# ── resample_to_month_start ──────────────────────────────────────────────────

def test_resample_three_months_returns_three_rows():
    daily = _make_daily("2020-01-02", "2020-03-31", 10.0, "x")
    result = resample_to_month_start(daily)
    assert len(result) == 3
    assert result.index[0] == pd.Timestamp("2020-01-02")
    assert result.index[1] == pd.Timestamp("2020-02-03")
    assert result.index[2] == pd.Timestamp("2020-03-02")


def test_resample_index_is_actual_trading_dates():
    daily = _make_daily("2020-01-02", "2020-06-30", 1.0, "x")
    result = resample_to_month_start(daily)
    assert result.index.isin(daily.index).all()


def test_resample_preserves_name():
    daily = _make_daily("2020-01-02", "2020-02-28", 5.0, "myseries")
    result = resample_to_month_start(daily)
    assert result.name == "myseries"


def test_resample_accepts_dataframe():
    idx = pd.bdate_range("2020-01-02", "2020-02-28")
    df = pd.DataFrame({"x": 7.0}, index=idx)
    result = resample_to_month_start(df)
    assert isinstance(result, pd.Series)
    assert len(result) == 2



# ── build_vrp_series ─────────────────────────────────────────────────────────

import numpy as np

from src import config
from src.vrp import build_vrp_series


def _make_synthetic_vrp_inputs(n_returns: int = 1000, seed: int = 0):
    """Synthetic daily returns and monthly VIX (constant at 20.0) for build_vrp_series tests."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n_returns)
    returns = pd.Series(rng.normal(0, 0.01, n_returns), index=idx, name="log_return")
    vix_monthly = resample_to_month_start(
        pd.Series(20.0, index=idx, name="close")
    )
    return returns, vix_monthly


def test_build_vrp_series_output_is_series():
    returns, vix_monthly = _make_synthetic_vrp_inputs()
    result = build_vrp_series(vix_monthly, returns, vix_monthly.index[15:20])
    assert isinstance(result, pd.Series)


def test_build_vrp_series_name():
    returns, vix_monthly = _make_synthetic_vrp_inputs()
    result = build_vrp_series(vix_monthly, returns, vix_monthly.index[15:20])
    assert result.name == "vrp"


def test_build_vrp_series_no_nan():
    returns, vix_monthly = _make_synthetic_vrp_inputs()
    result = build_vrp_series(vix_monthly, returns, vix_monthly.index[15:20])
    assert result.isna().sum() == 0


def test_build_vrp_series_vix_variance_conversion():
    """VIX=20 vol-points -> variance = 20^2 / VIX_VARIANCE_SCALE = 0.04 decimal annualised."""
    assert 20.0 ** 2 / config.VIX_VARIANCE_SCALE == pytest.approx(0.04)


def test_build_vrp_series_leakage():
    """Value at t1 must be identical whether t1 is the only date or one of many."""
    returns, vix_monthly = _make_synthetic_vrp_inputs(seed=7)
    t1 = vix_monthly.index[15]
    t2 = vix_monthly.index[30]

    single = build_vrp_series(vix_monthly, returns, [t1])
    multi = build_vrp_series(vix_monthly, returns, [t1, t2])

    assert t1 in single.index
    assert t1 in multi.index
    assert single.at[t1] == multi.at[t1]


def test_build_vrp_series_input_truncation():
    """Strong leakage test: truncating ALL input series at t1 must not change VRP(t1).

    The weak test (test_build_vrp_series_leakage) varies only forecast_dates
    while passing the full returns and vix_monthly in both calls.  This test cuts
    the underlying data: the truncated call receives returns.loc[:t1] and
    vix_monthly.loc[:t1], containing no data after t1.  Any post-t1 peek inside
    the recursive HAR-RV fit would cause the two values to differ.
    """
    returns, vix_monthly = _make_synthetic_vrp_inputs(seed=7)
    t1 = vix_monthly.index[15]

    full = build_vrp_series(vix_monthly, returns, [t1])
    truncated = build_vrp_series(
        vix_monthly.loc[:t1],
        returns.loc[:t1],
        [t1],
    )

    assert t1 in full.index, "t1 missing from full result"
    assert t1 in truncated.index, "t1 missing from truncated result"
    assert full.at[t1] == truncated.at[t1], (
        f"Leakage detected: full={full.at[t1]:.10f}, "
        f"truncated={truncated.at[t1]:.10f}"
    )


def test_build_vrp_series_raises_on_missing_vix_dates():
    """Raises if vix_monthly lacks entries for the HAR-valid forecast dates."""
    returns, vix_monthly = _make_synthetic_vrp_inputs()
    vix_short = vix_monthly.iloc[:5]  # only first 5 months
    with pytest.raises(ValueError):
        build_vrp_series(vix_short, returns, vix_monthly.index[15:30])


# ── build_vrp_series (network) ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_vrp():
    """Canonical VRP series from real SPY and VIX data."""
    try:
        from src.data_loader import download_prices
        from src.returns import compute_log_returns
        spy = download_prices(config.TICKER_SPY)
        vix_raw = download_prices(config.TICKER_VIX)
        spy_rets = compute_log_returns(spy)
        vix_m = resample_to_month_start(vix_raw["close"])
        return build_vrp_series(vix_m, spy_rets, vix_m.index)
    except Exception as exc:
        pytest.skip(f"yfinance unavailable: {exc}")


@pytest.mark.network
def test_build_vrp_series_mean_positive(real_vrp):
    assert real_vrp.mean() > 0, (
        f"mean VRP {real_vrp.mean():.6f} is not positive; check units"
    )


@pytest.mark.network
def test_build_vrp_series_fraction_positive_above_70pct(real_vrp):
    frac = (real_vrp > 0).mean()
    assert frac > 0.7, (
        f"fraction positive {frac:.3f} is below 0.7; check units"
    )
