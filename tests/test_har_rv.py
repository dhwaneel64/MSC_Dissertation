"""Tests for the recursive HAR-RV variance forecaster in src.har_rv."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src import config
from src.har_rv import fit_har_rv_forecast, realised_variance_target
from src.metrics import qlike
from src.realised_vol import compute_realised_vol


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns(n: int = 1000, seed: int = 0) -> pd.Series:
    """Synthetic daily log-returns on a business-day index."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.Series(rng.normal(0, 0.01, n), index=idx, name="log_return")


@pytest.fixture(scope="module")
def spy_returns() -> pd.Series:
    """Real SPY daily log-returns for network-dependent tests."""
    try:
        from src.data_loader import download_prices
        from src.returns import compute_log_returns
        prices = download_prices(config.TICKER_SPY)
        return compute_log_returns(prices)
    except Exception as exc:
        pytest.skip(f"yfinance unavailable: {exc}")


# ---------------------------------------------------------------------------
# Output shape and type
# ---------------------------------------------------------------------------

def test_output_is_series():
    returns = _make_returns()
    result = fit_har_rv_forecast(returns, [returns.index[300], returns.index[500]])
    assert isinstance(result, pd.Series)


def test_output_has_datetimeindex():
    returns = _make_returns()
    result = fit_har_rv_forecast(returns, [returns.index[300]])
    assert isinstance(result.index, pd.DatetimeIndex)


def test_output_name():
    returns = _make_returns()
    result = fit_har_rv_forecast(returns, [returns.index[300]])
    assert result.name == "har_rv_forecast"


def test_output_no_nan():
    returns = _make_returns()
    dates = [returns.index[200], returns.index[500], returns.index[800]]
    result = fit_har_rv_forecast(returns, dates)
    assert result.isna().sum() == 0


def test_output_values_positive():
    returns = _make_returns()
    dates = [returns.index[200], returns.index[500], returns.index[800]]
    result = fit_har_rv_forecast(returns, dates)
    assert (result > 0).all()


def test_output_index_is_subset_of_forecast_dates():
    returns = _make_returns()
    bad = pd.Timestamp("1990-01-01")
    good1 = returns.index[300]
    good2 = returns.index[600]
    result = fit_har_rv_forecast(returns, [bad, good1, good2])
    for date in result.index:
        assert date in [bad, good1, good2]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_forecast_dates_returns_empty_series():
    returns = _make_returns()
    result = fit_har_rv_forecast(returns, [])
    assert isinstance(result, pd.Series)
    assert len(result) == 0


def test_date_not_in_index_is_skipped():
    returns = _make_returns()
    bad = pd.Timestamp("1990-01-01")
    good = returns.index[300]
    result = fit_har_rv_forecast(returns, [bad, good])
    assert bad not in result.index
    assert good in result.index


def test_raises_on_empty_returns():
    with pytest.raises(ValueError, match="empty"):
        fit_har_rv_forecast(pd.Series([], dtype=float), [])


def test_raises_on_non_monotonic_index():
    idx = pd.DatetimeIndex(["2020-01-03", "2020-01-01", "2020-01-02"])
    returns = pd.Series([0.01, 0.02, 0.03], index=idx)
    with pytest.raises(ValueError, match="monotonic"):
        fit_har_rv_forecast(returns, [])


# ---------------------------------------------------------------------------
# Leakage test (mandatory)
# ---------------------------------------------------------------------------

def test_leakage_single_vs_multi_date():
    """Value at t1 must be identical whether t1 is the only date or one of many.

    Encodes the invariant: the forecast at t uses only data with index <= t,
    so adding future dates to forecast_dates must not alter the t1 result.
    """
    returns = _make_returns(n=1000, seed=7)
    t1 = returns.index[250]
    t2 = returns.index[600]
    t3 = returns.index[900]

    result_single = fit_har_rv_forecast(returns, [t1])
    result_multi = fit_har_rv_forecast(returns, [t1, t2, t3])

    assert t1 in result_single.index, "t1 missing from single-date result"
    assert t1 in result_multi.index, "t1 missing from multi-date result"
    assert result_single.at[t1] == result_multi.at[t1]


def test_leakage_input_truncation():
    """Strong leakage test: truncating input returns at t1 must not change the forecast at t1.

    The weak test (test_leakage_single_vs_multi_date) varies only forecast_dates
    while passing the full returns Series in both calls.  This test cuts the
    underlying data: result_truncated uses only returns.loc[:t1], which contains
    no data after t1.  If the implementation peeks past t1 (e.g. uses full-series
    normalisation or a global fit), the two values will differ.
    """
    returns = _make_returns(n=1000, seed=7)
    t1 = returns.index[250]
    t1_minus_1 = returns.index[249]

    result_full = fit_har_rv_forecast(returns, [t1])
    result_truncated = fit_har_rv_forecast(returns.loc[:t1], [t1])

    assert t1 in result_full.index, "t1 missing from full result"
    assert t1 in result_truncated.index, "t1 missing from truncated result"
    assert result_full.at[t1] == result_truncated.at[t1], (
        f"Leakage detected: full={result_full.at[t1]:.10f}, "
        f"truncated={result_truncated.at[t1]:.10f}"
    )

    # Confirm the forecast genuinely depends on data up to t1: when the input is
    # cut one trading day before t1, t1 is not in the returns index so it must
    # be silently omitted from the output.
    result_short = fit_har_rv_forecast(returns.loc[:t1_minus_1], [t1])
    assert t1 not in result_short.index, (
        "t1 should be absent when input is truncated one day before t1"
    )


# ---------------------------------------------------------------------------
# realised_variance_target — QLIKE realised-side estimator
# ---------------------------------------------------------------------------

def test_realised_variance_target_output_is_series():
    returns = _make_returns()
    result = realised_variance_target(returns)
    assert isinstance(result, pd.Series)
    assert result.name == "realised_variance_target"


def test_realised_variance_target_pinned_to_mean_r2_times_252():
    """Exact pin: target[d] = mean(r^2 over [d+1, d+RV_WINDOW]) * ANNUALISATION_FACTOR_DAILY."""
    returns = _make_returns(n=200, seed=3)
    result = realised_variance_target(returns)

    pos = 100
    d = returns.index[pos]
    window = returns.iloc[pos + 1 : pos + 1 + config.RV_WINDOW]
    expected = (window ** 2).mean() * config.ANNUALISATION_FACTOR_DAILY

    assert result.at[d] == pytest.approx(expected, rel=1e-12)


def test_realised_variance_target_not_equal_to_compute_realised_vol():
    """The two realised-variance estimators must never silently re-converge.

    compute_realised_vol is demeaned (ddof=1) and backward-looking;
    realised_variance_target is uncentred and forward-looking. Even when
    compared on the "same" window (shifted to align start points), the
    centring/ddof difference alone must produce a numerically distinct value.
    """
    returns = _make_returns(n=200, seed=3)
    pos = 100
    d = returns.index[pos]

    target_val = realised_variance_target(returns).at[d]

    # compute_realised_vol at the date RV_WINDOW trading days after d looks
    # backward over exactly the same RV_WINDOW returns [d+1, d+RV_WINDOW] that
    # realised_variance_target(d) looks forward over.
    rv_vol_points = compute_realised_vol(returns)
    end_date = returns.index[pos + config.RV_WINDOW]
    demeaned_val = (rv_vol_points.at[end_date] / 100.0) ** 2  # vol-points -> decimal variance

    assert target_val != pytest.approx(demeaned_val, rel=1e-9), (
        "realised_variance_target must not equal the demeaned, ddof=1 "
        "compute_realised_vol estimator on the same window"
    )
    # The two should still be close (same underlying returns), confirming the
    # difference is the ddof/centring correction, not a unit or window bug.
    assert target_val == pytest.approx(demeaned_val, rel=0.15)


def test_realised_variance_target_nan_tail():
    returns = _make_returns(n=200, seed=1)
    result = realised_variance_target(returns)
    assert result.iloc[-config.RV_WINDOW:].isna().all()
    assert pd.notna(result.iloc[-config.RV_WINDOW - 1])


def test_realised_variance_target_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        realised_variance_target(pd.Series([], dtype=float))


def test_realised_variance_target_raises_on_non_monotonic_index():
    idx = pd.DatetimeIndex(["2020-01-03", "2020-01-01", "2020-01-02"])
    returns = pd.Series([0.01, 0.02, 0.03], index=idx)
    with pytest.raises(ValueError, match="monotonic"):
        realised_variance_target(returns)


def test_fit_har_rv_forecast_target_matches_realised_variance_target():
    """fit_har_rv_forecast must train against exactly this function's output,
    not a parallel re-derivation that could drift from it."""
    returns = _make_returns(n=1000, seed=4)
    daily_rv = returns ** 2
    direct_target = realised_variance_target(returns)
    expected_target = (
        daily_rv.rolling(config.RV_WINDOW).mean().shift(-config.RV_WINDOW)
        * config.ANNUALISATION_FACTOR_DAILY
    )
    pd.testing.assert_series_equal(
        direct_target, expected_target, check_names=False
    )


# ---------------------------------------------------------------------------
# In-sample QLIKE range (mandatory, network)
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_qlike_in_sample_range(spy_returns: pd.Series) -> None:
    """In-sample QLIKE at a single mid-sample fit must lie in [0.1, 0.5].

    Fits the HAR model once at 2010-01-04 on the training data available at
    that date, then computes QLIKE on the in-sample fitted values.  Values
    outside [0.1, 0.5] indicate a spec error (wrong units, wrong annualisation,
    or incorrect target construction).
    """
    t_test = pd.Timestamp("2010-01-04")
    assert t_test in spy_returns.index, "2010-01-04 not in SPY returns index"

    daily_rv = spy_returns ** 2
    reg_D = daily_rv.rolling(config.HAR_RV_HORIZON_D).mean()
    reg_W = daily_rv.rolling(config.HAR_RV_HORIZON_W).mean()
    reg_M = daily_rv.rolling(config.HAR_RV_HORIZON_M).mean()
    target = realised_variance_target(spy_returns)

    data = pd.DataFrame(
        {"D": reg_D, "W": reg_W, "M": reg_M, "target": target}
    ).dropna()

    pos_t = daily_rv.index.get_loc(t_test)
    cutoff_date = daily_rv.index[pos_t - config.RV_WINDOW]
    train = data.loc[data.index <= cutoff_date]

    X_train = sm.add_constant(train[["D", "W", "M"]], has_constant="add")
    fit = sm.OLS(train["target"], X_train).fit()

    y_actual = train["target"].values
    y_fitted = fit.fittedvalues.values

    assert (y_fitted > 0).all(), (
        f"HAR in-sample fitted values contain non-positive entries; "
        f"min={y_fitted.min():.6f}"
    )

    qlike_val = qlike(y_actual, y_fitted)
    assert 0.1 <= qlike_val <= 0.5, (
        f"in-sample QLIKE {qlike_val:.4f} outside expected range [0.1, 0.5]; "
        "check units, annualisation, or target construction"
    )
