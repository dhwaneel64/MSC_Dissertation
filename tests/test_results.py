import math

import numpy as np
import pandas as pd
import pytest

from src import config
from src.evaluation import rv_to_variance, vrp_forecast_to_variance
from src.har_rv import realised_variance_target
from src.metrics import qlike
from src.realised_vol import compute_realised_vol
from src.results import score_walk_forward
from src.vrp import resample_to_month_start


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _constant_returns(r: float = 0.01, n_days: int = 700, start: str = "2000-01-03") -> pd.Series:
    """Daily log-returns constant at r, so realised_variance_target is the
    same closed-form value (r**2 * ANNUALISATION_FACTOR_DAILY) everywhere a
    full forward window is available."""
    idx = pd.bdate_range(start, periods=n_days)
    return pd.Series(r, index=idx, name="log_return", dtype=float)


# ---------------------------------------------------------------------------
# Forecast-side conversion (variance-space, not old vix-minus-vrp convention)
# and basic correctness end to end.
# ---------------------------------------------------------------------------

def test_score_walk_forward_basic_correctness():
    r = 0.01
    returns = _constant_returns(r)
    monthly_idx = resample_to_month_start(returns).index
    t0, t1 = monthly_idx[11], monthly_idx[16]

    wf_result = pd.DataFrame(
        {"y_true": [0.012, 0.018], "y_pred": [0.01, 0.02]}, index=[t0, t1]
    )
    vix_next = pd.Series({t0: 20.0, t1: 25.0})

    result = score_walk_forward(wf_result, vix_next, returns)

    realised = r ** 2 * config.ANNUALISATION_FACTOR_DAILY  # 0.0252, both months valid
    implied0 = 20.0 ** 2 / config.VIX_VARIANCE_SCALE - 0.01  # 0.03
    implied1 = 25.0 ** 2 / config.VIX_VARIANCE_SCALE - 0.02  # 0.0425
    ratio0, ratio1 = realised / implied0, realised / implied1
    expected_qlike = np.mean([
        ratio0 - math.log(ratio0) - 1,
        ratio1 - math.log(ratio1) - 1,
    ])
    expected_mse = np.mean([(0.012 - 0.01) ** 2, (0.018 - 0.02) ** 2])

    assert result["qlike"] == pytest.approx(expected_qlike, rel=1e-9)
    assert result["mse"] == pytest.approx(expected_mse, rel=1e-9)
    assert result["directional_accuracy"] == 1.0
    assert result["qlike_n"] == 2
    assert result["mse_n"] == 2
    assert result["da_n"] == 2
    assert result["n_guard_excluded"] == 0
    assert result["n_nan_tail_excluded"] == 0
    assert result["n_obs"] == 2
    assert result["first_oos_date"] == t0
    assert result["last_oos_date"] == t1


def test_score_walk_forward_forecast_side_matches_vrp_forecast_to_variance():
    """implied_variance is vix_next**2/VIX_VARIANCE_SCALE - y_pred, not vix_next - y_pred."""
    returns = _constant_returns()
    monthly_idx = resample_to_month_start(returns).index
    t0 = monthly_idx[11]

    wf_result = pd.DataFrame({"y_true": [0.01], "y_pred": [0.015]}, index=[t0])
    vix_next = pd.Series({t0: 30.0})

    result = score_walk_forward(wf_result, vix_next, returns)

    expected_forecast_var = vrp_forecast_to_variance([0.015], [30.0])[0]
    realised = 0.01 ** 2 * config.ANNUALISATION_FACTOR_DAILY
    ratio = realised / expected_forecast_var
    expected_qlike = ratio - math.log(ratio) - 1

    assert result["qlike"] == pytest.approx(expected_qlike, rel=1e-9)


# ---------------------------------------------------------------------------
# Realised side: must be realised_variance_target, never compute_realised_vol.
# ---------------------------------------------------------------------------

def test_score_walk_forward_realised_side_is_realised_variance_target():
    rng = np.random.default_rng(7)
    returns = pd.Series(
        rng.normal(0, 0.01, 700), index=pd.bdate_range("2000-01-03", periods=700), name="log_return"
    )
    monthly_idx = resample_to_month_start(returns).index
    t_oos, t_next = monthly_idx[15], monthly_idx[16]

    wf_result = pd.DataFrame({"y_true": [0.01], "y_pred": [0.001]}, index=[t_oos])
    vix_next = pd.Series({t_oos: 100.0})  # large VIX: guard never trips, isolates realised side

    result = score_walk_forward(wf_result, vix_next, returns)

    correct_target_monthly = resample_to_month_start(realised_variance_target(returns))
    correct_realised = correct_target_monthly.at[t_next]
    forecast_var = vrp_forecast_to_variance([0.001], [100.0])[0]
    expected_qlike = qlike([correct_realised], [forecast_var])

    assert result["qlike"] == pytest.approx(expected_qlike, rel=1e-9)


def test_score_walk_forward_realised_side_is_not_compute_realised_vol():
    rng = np.random.default_rng(7)
    returns = pd.Series(
        rng.normal(0, 0.01, 700), index=pd.bdate_range("2000-01-03", periods=700), name="log_return"
    )
    monthly_idx = resample_to_month_start(returns).index
    t_oos, t_next = monthly_idx[15], monthly_idx[16]

    wf_result = pd.DataFrame({"y_true": [0.01], "y_pred": [0.001]}, index=[t_oos])
    vix_next = pd.Series({t_oos: 100.0})

    result = score_walk_forward(wf_result, vix_next, returns)

    wrong_rv_monthly = resample_to_month_start(compute_realised_vol(returns))
    wrong_realised = rv_to_variance([wrong_rv_monthly.at[t_next]])[0]
    forecast_var = vrp_forecast_to_variance([0.001], [100.0])[0]
    wrong_qlike = qlike([wrong_realised], [forecast_var])

    assert result["qlike"] != pytest.approx(wrong_qlike, rel=1e-3)


# ---------------------------------------------------------------------------
# Guard exclusions (forecast side): tracked separately from NaN-tail.
# ---------------------------------------------------------------------------

def test_score_walk_forward_guard_exclusion_handled():
    returns = _constant_returns()
    monthly_idx = resample_to_month_start(returns).index
    t0, t1 = monthly_idx[11], monthly_idx[16]  # both have full forward windows

    wf_result = pd.DataFrame(
        {"y_true": [0.01, 0.012], "y_pred": [0.015, 0.01]}, index=[t0, t1]
    )
    # t0: vix=10 -> implied = 100/10000 - 0.015 = -0.005 (guard trip)
    # t1: vix=20 -> implied = 400/10000 - 0.01 = 0.03 (valid)
    vix_next = pd.Series({t0: 10.0, t1: 20.0})

    result = score_walk_forward(wf_result, vix_next, returns)

    assert result["n_guard_excluded"] == 1
    assert result["n_nan_tail_excluded"] == 0
    assert result["qlike_n"] == 1
    assert result["mse_n"] == 2
    assert result["da_n"] == 2


# ---------------------------------------------------------------------------
# NaN-tail exclusion (realised side): forward window not fully in-sample.
# ---------------------------------------------------------------------------

def test_score_walk_forward_nan_tail_exclusion_handled():
    returns = _constant_returns(n_days=46, start="2021-01-01")
    monthly_idx = resample_to_month_start(returns).index
    assert len(monthly_idx) == 3  # 2021-01-01, 2021-02-01, 2021-03-01
    t0, t1 = monthly_idx[0], monthly_idx[1]
    # t0's t+1 (2021-02-01) has a full forward window; t1's t+1 (2021-03-01) does not.

    wf_result = pd.DataFrame(
        {"y_true": [0.01, 0.012], "y_pred": [0.001, 0.001]}, index=[t0, t1]
    )
    vix_next = pd.Series({t0: 20.0, t1: 20.0})

    result = score_walk_forward(wf_result, vix_next, returns)

    assert result["n_guard_excluded"] == 0
    assert result["n_nan_tail_excluded"] == 1
    assert result["qlike_n"] == 1
    assert result["mse_n"] == 2
    assert result["da_n"] == 2


# ---------------------------------------------------------------------------
# Paired-subset exposure: valid_mask / implied_variance / realised_variance_next
# ---------------------------------------------------------------------------

def test_score_walk_forward_exposes_mask_and_variances_for_pairing():
    returns = _constant_returns()
    monthly_idx = resample_to_month_start(returns).index
    t0, t1 = monthly_idx[11], monthly_idx[16]

    wf_result = pd.DataFrame(
        {"y_true": [0.01, 0.012], "y_pred": [0.015, 0.01]}, index=[t0, t1]
    )
    # t0: vix=10 -> implied = 100/10000 - 0.015 = -0.005 (guard trip, invalid)
    # t1: vix=20 -> implied = 400/10000 - 0.01 = 0.03 (valid)
    vix_next = pd.Series({t0: 10.0, t1: 20.0})

    result = score_walk_forward(wf_result, vix_next, returns)

    np.testing.assert_array_equal(result["valid_mask"], np.array([False, True]))
    assert result["valid_mask"].sum() == result["qlike_n"]

    expected_implied = np.array([
        10.0 ** 2 / config.VIX_VARIANCE_SCALE - 0.015,
        20.0 ** 2 / config.VIX_VARIANCE_SCALE - 0.01,
    ])
    np.testing.assert_allclose(result["implied_variance"], expected_implied)

    realised = 0.01 ** 2 * config.ANNUALISATION_FACTOR_DAILY
    np.testing.assert_allclose(result["realised_variance_next"], [realised, realised])

    # QLIKE rebuilt from the exposed arrays must match the scorer's QLIKE,
    # confirming these are the same single variance-space path.
    m = result["valid_mask"]
    rebuilt = qlike(result["realised_variance_next"][m], result["implied_variance"][m])
    assert rebuilt == pytest.approx(result["qlike"], rel=1e-12)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_score_walk_forward_raises_on_missing_vix_next_dates():
    returns = _constant_returns()
    monthly_idx = resample_to_month_start(returns).index
    t0 = monthly_idx[11]
    wf_result = pd.DataFrame({"y_true": [0.01], "y_pred": [0.005]}, index=[t0])
    vix_next = pd.Series({monthly_idx[10]: 20.0})  # wrong date, doesn't cover t0
    with pytest.raises(ValueError, match="not in vix_next.index"):
        score_walk_forward(wf_result, vix_next, returns)


def test_score_walk_forward_raises_on_nan_vix_next():
    returns = _constant_returns()
    monthly_idx = resample_to_month_start(returns).index
    t0 = monthly_idx[11]
    wf_result = pd.DataFrame({"y_true": [0.01], "y_pred": [0.005]}, index=[t0])
    vix_next = pd.Series({t0: float("nan")})
    with pytest.raises(ValueError, match="NaN"):
        score_walk_forward(wf_result, vix_next, returns)


# ---------------------------------------------------------------------------
# Regression check against the validated constant-baseline notebook numbers
# (network: requires the real SPY/VIX/SKEW pipeline).
# ---------------------------------------------------------------------------

@pytest.mark.network
def test_score_walk_forward_reproduces_constant_baseline():
    """score_walk_forward on the real constant-baseline wf_result must reproduce
    QLIKE 1.462384 (n=223), MSE 0.00089641 (n=253), DA 0.6245 (n=253),
    locked to raw_snapshot_2026-06-17.parquet (unadjusted SPY, config.LOCKED_SNAPSHOT_DATE).

    Re-baselined by the lag-to-target alignment correction, which moved the nearest
    VRP predictor from two steps off its target to one and added a row at the front
    of the dataset (387 to 388, first row 1993-11-01 to 1993-10-01). The
    pre-correction floor was QLIKE 1.573981 with MSE 0.00089691 on the same
    snapshot; earlier still, the adjusted-price floor was 1.728298
    (raw_snapshot_2026-06-03) before the auto_adjust=False switch on 2026-06-17.
    The OOS window is unchanged at 253 months, so the guard and NaN-tail exclusion
    counts are unchanged.
    """
    from src.data_loader import download_prices
    from src.returns import compute_log_returns
    from src.vrp import build_vrp_series
    from src.regimes import label_regimes
    from src.dataset import build_model_ready_dataset
    from src.models.baseline import ConstantMeanModel
    from src.walk_forward import walk_forward, make_model_factory_from_class

    spy = download_prices(config.TICKER_SPY)
    vix = download_prices(config.TICKER_VIX)
    skew = download_prices(config.TICKER_SKEW)
    spy_returns = compute_log_returns(spy)

    vix_monthly = resample_to_month_start(vix["close"])
    vrp = build_vrp_series(vix_monthly, spy_returns, vix_monthly.index)
    skew_monthly = resample_to_month_start(skew["close"])
    regime_labels = label_regimes(vix_monthly)
    dataset = build_model_ready_dataset(vrp, vix_monthly, skew_monthly, spy_returns, regime_labels)

    last_train_year = dataset.index[0].year + config.INITIAL_TRAINING_YEARS_FULL - 1
    initial_train_end = dataset.loc[dataset.index.year <= last_train_year].index[-1]
    model_factory = make_model_factory_from_class(ConstantMeanModel)
    wf_constant = walk_forward(
        dataset, feature_cols=[], model_factory=model_factory,
        initial_train_end=initial_train_end, target_col="y",
    )

    vix_next = vix_monthly.shift(-1).reindex(wf_constant.index)
    result = score_walk_forward(wf_constant, vix_next, spy_returns)

    # Locked dataset shape under the corrected alignment. Asserted here because the
    # scored numbers below are only reproducible on this exact sample.
    assert len(dataset) == 388
    assert dataset.index[0] == pd.Timestamp("1993-10-01")

    assert result["qlike"] == pytest.approx(1.462384, abs=1e-5)
    assert result["qlike_n"] == 223
    assert result["mse"] == pytest.approx(0.00089641, abs=1e-7)
    assert result["mse_n"] == 253
    assert result["directional_accuracy"] == pytest.approx(0.6245, abs=1e-4)
    assert result["da_n"] == 253
    assert result["n_guard_excluded"] == 29
    assert result["n_nan_tail_excluded"] == 1
