import math

import numpy as np
import pandas as pd
import pytest
import scipy.stats as st
import statsmodels.api as sm

from src import config
from src.har_rv import realised_variance_target
from src.mincer_zarnowitz import (
    MZResult,
    build_mz_frame,
    mincer_zarnowitz_regression,
    mz_results_table,
    newey_west_lag,
    run_mincer_zarnowitz,
)
from src.regimes import label_regimes
from src.vrp import resample_to_month_start


def _daily_returns(start: str = "2000-01-03", periods: int = 1400) -> pd.Series:
    """Synthetic daily log-returns on business days with a monotonic index."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range(start, periods=periods)
    return pd.Series(rng.normal(0.0, 0.01, periods), index=idx, name="ret")


def _monthly_vix_from(daily: pd.Series, values=None) -> pd.Series:
    """Monthly first-trading-day VIX series aligned to the daily index months."""
    month_starts = resample_to_month_start(daily).index
    rng = np.random.default_rng(3)
    if values is None:
        vix = rng.uniform(10.0, 40.0, len(month_starts))
    else:
        vix = np.asarray(values, dtype=float)
    return pd.Series(vix[: len(month_starts)], index=month_starts, name="vix")


# ---------------------------------------------------------------------------
# newey_west_lag
# ---------------------------------------------------------------------------

def test_newey_west_lag_at_n100():
    """n=100 -> floor(4 * 1 ** (2/9)) = 4, the locked Newey-West (1994) rule."""
    assert newey_west_lag(100) == 4
    assert newey_west_lag(100) == math.floor(
        config.NW_HAC_MULTIPLIER * (100 / 100) ** config.NW_HAC_EXPONENT
    )


def test_newey_west_lag_shrinks_with_n():
    """A smaller per-regime sample gets a weakly smaller HAC lag."""
    assert newey_west_lag(40) <= newey_west_lag(250)


# ---------------------------------------------------------------------------
# mincer_zarnowitz_regression
# ---------------------------------------------------------------------------

def test_recovers_known_coefficients():
    """OLS recovers the data-generating alpha and beta on a synthetic case."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0.01, 0.20, 300)
    y = 0.30 + 1.80 * x + rng.normal(0.0, 0.005, 300)

    res = mincer_zarnowitz_regression(y, x, sample="full")

    assert isinstance(res, MZResult)
    assert res.n == 300
    assert res.sample == "full"
    assert res.hac_lag == newey_west_lag(300)
    assert res.alpha == pytest.approx(0.30, abs=0.02)
    assert res.beta == pytest.approx(1.80, abs=0.02)


def test_wald_statistic_matches_statsmodels():
    """The manual joint Wald matches statsmodels wald_test on the same HAC fit."""
    rng = np.random.default_rng(1)
    x = rng.uniform(0.01, 0.20, 200)
    y = 0.05 + 1.30 * x + rng.normal(0.0, 0.01, 200)

    res = mincer_zarnowitz_regression(y, x, sample="full")

    X = sm.add_constant(x)
    fit = sm.OLS(y, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lag(200)}
    )
    R = np.array([[1.0, 0.0], [0.0, 1.0]])
    q = np.array([0.0, 1.0])
    w = fit.wald_test((R, q), use_f=False, scalar=True)

    assert res.wald_stat == pytest.approx(float(w.statistic), rel=1e-9)
    assert res.wald_p == pytest.approx(float(w.pvalue), rel=1e-6)
    # Individual beta=1 t-test consistent with the HAC standard error.
    assert res.t_beta1 == pytest.approx((fit.params[1] - 1.0) / fit.bse[1], rel=1e-9)


def test_wald_rejects_when_beta_not_one():
    """A clear beta != 1 (and alpha != 0) data set rejects rationality."""
    rng = np.random.default_rng(2)
    x = rng.uniform(0.01, 0.20, 300)
    y = 0.50 + 2.00 * x + rng.normal(0.0, 0.005, 300)

    res = mincer_zarnowitz_regression(y, x, sample="full")
    assert res.reject_rationality is True
    assert res.wald_p < config.COMPARISON_ALPHA


def test_wald_does_not_reject_when_exactly_rational():
    """Residuals orthogonal to [1, x] force alpha=0, beta=1 exactly: Wald = 0.

    Projecting an arbitrary error onto the orthogonal complement of the design
    makes the OLS fit exactly alpha=0, beta=1 while leaving nonzero residuals
    (so the HAC covariance is well defined). The joint restriction then holds
    exactly, the Wald statistic is ~0, and rationality is not rejected.
    """
    rng = np.random.default_rng(4)
    n = 200
    x = rng.uniform(0.01, 0.20, n)
    X = np.column_stack([np.ones(n), x])
    e = rng.normal(0.0, 0.02, n)
    # Orthogonal projection: e_perp has zero correlation with const and x.
    e_perp = e - X @ np.linalg.solve(X.T @ X, X.T @ e)
    y = 0.0 + 1.0 * x + e_perp

    res = mincer_zarnowitz_regression(y, x, sample="full")

    assert res.alpha == pytest.approx(0.0, abs=1e-10)
    assert res.beta == pytest.approx(1.0, abs=1e-10)
    assert res.wald_stat == pytest.approx(0.0, abs=1e-6)
    assert res.reject_rationality is False


def test_regression_raises_on_nan():
    x = np.array([0.04, 0.05, 0.06, np.nan])
    y = np.array([0.03, 0.05, 0.04, 0.05])
    with pytest.raises(ValueError, match="NaN"):
        mincer_zarnowitz_regression(y, x)


def test_regression_raises_on_too_few_obs():
    with pytest.raises(ValueError, match="at least 3"):
        mincer_zarnowitz_regression([0.04, 0.05], [0.03, 0.06])


# ---------------------------------------------------------------------------
# build_mz_frame: realised side and alignment
# ---------------------------------------------------------------------------

def test_frame_forecast_is_vix_variance():
    """The forecast column is VIX(t)**2 / VIX_VARIANCE_SCALE."""
    daily = _daily_returns()
    vix_monthly = _monthly_vix_from(daily)
    frame = build_mz_frame(vix_monthly, daily)

    expected = (vix_monthly ** 2 / config.VIX_VARIANCE_SCALE).reindex(frame.index)
    np.testing.assert_allclose(frame["vix_variance"].to_numpy(), expected.to_numpy())


def test_realised_side_is_forward_target_not_backward_rv():
    """Realised(t) is the forward target evaluated at t, never compute_realised_vol.

    Confirms the temporal alignment. realised_variance_target at month t already
    covers the RV_WINDOW trading days after t, which is the window VIX(t) prices,
    so the row at t carries the estimator at t and no further shift is applied.
    The locked methodology calls that window RV(t+1) because it names the month
    following t as t+1; the code indexes the same window at t. Applying .shift(-1)
    on top of the estimator's own forward window would reach one month too far,
    which is the defect this test now pins.
    """
    daily = _daily_returns()
    vix_monthly = _monthly_vix_from(daily)
    frame = build_mz_frame(vix_monthly, daily)

    realised_monthly = resample_to_month_start(realised_variance_target(daily))

    for t in frame.index[:: max(1, len(frame) // 10)]:
        pos = realised_monthly.index.get_loc(t)
        t_next = realised_monthly.index[pos + 1]
        assert frame.loc[t, "realised_variance_next"] == pytest.approx(
            realised_monthly.loc[t]
        )
        assert frame.loc[t, "realised_variance_next"] != pytest.approx(
            realised_monthly.loc[t_next]
        )


def test_frame_drops_forward_nan_tail():
    """The last month (no observable t+1 forward target) is dropped."""
    daily = _daily_returns()
    vix_monthly = _monthly_vix_from(daily)
    frame = build_mz_frame(vix_monthly, daily)

    assert frame["realised_variance_next"].notna().all()
    # The final available month start has no fully observed forward t+1 window.
    last_month = resample_to_month_start(daily).index.max()
    assert last_month not in frame.index


def test_frame_regime_matches_label_regimes_at_t():
    """Regime is classified by VIX at month t (the forecast date)."""
    daily = _daily_returns()
    vix_monthly = _monthly_vix_from(daily)
    frame = build_mz_frame(vix_monthly, daily)

    expected = label_regimes(vix_monthly).reindex(frame.index).astype(str)
    assert (frame["regime"].astype(str) == expected).all()


# ---------------------------------------------------------------------------
# run_mincer_zarnowitz: per-regime subsetting
# ---------------------------------------------------------------------------

def test_run_returns_four_samples_in_order():
    daily = _daily_returns()
    vix_monthly = _monthly_vix_from(daily)
    results, _ = run_mincer_zarnowitz(vix_monthly, daily)

    assert [r.sample for r in results] == ["full", "calm", "normal", "stressed"]


def _vix_spanning_regimes(daily: pd.Series, seed: int) -> pd.Series:
    """Monthly VIX that varies continuously inside each of the three regimes."""
    month_starts = resample_to_month_start(daily).index
    rng = np.random.default_rng(seed)
    bands = [(8.0, 14.5), (16.0, 24.0), (26.0, 45.0)]  # calm / normal / stressed
    pick = rng.integers(0, 3, size=len(month_starts))
    vix_vals = np.array(
        [rng.uniform(*bands[b]) for b in pick], dtype=float
    )
    return pd.Series(vix_vals, index=month_starts, name="vix")


def test_regime_subsets_partition_full_sample():
    """The three regime n's sum to the full-sample n (a clean partition)."""
    daily = _daily_returns()
    vix_monthly = _vix_spanning_regimes(daily, seed=11)

    results, frame = run_mincer_zarnowitz(vix_monthly, daily)
    by_sample = {r.sample: r.n for r in results}

    assert by_sample["calm"] + by_sample["normal"] + by_sample["stressed"] == by_sample["full"]
    assert by_sample["full"] == len(frame)
    # Each regime n equals the count of that regime in the frame.
    counts = frame["regime"].astype(str).value_counts()
    for label in config.REGIME_LABELS:
        assert by_sample[label] == int(counts.get(label, 0))


def test_per_regime_hac_lag_uses_own_n():
    """Each regression's HAC lag is computed from its own sample size."""
    daily = _daily_returns()
    vix_monthly = _vix_spanning_regimes(daily, seed=12)

    results, _ = run_mincer_zarnowitz(vix_monthly, daily)
    for r in results:
        assert r.hac_lag == newey_west_lag(r.n)


def test_results_table_columns_and_rows():
    daily = _daily_returns()
    vix_monthly = _monthly_vix_from(daily)
    results, _ = run_mincer_zarnowitz(vix_monthly, daily)
    table = mz_results_table(results)

    assert list(table["sample"]) == ["full", "calm", "normal", "stressed"]
    assert set(table.columns) == {
        "sample", "alpha", "se_alpha", "beta", "se_beta",
        "wald_stat", "wald_p", "n", "reject_rationality",
    }
    assert len(table) == 4
