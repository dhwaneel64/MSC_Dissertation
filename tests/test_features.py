import numpy as np
import pandas as pd
import pytest
from scipy.stats import skew as scipy_skew

from src import config
from src.features import (
    compute_realised_skew_21d,
    build_feature_matrix,
)
from src.regimes import label_regimes
from src.validation import VRP_HORIZON_COLS
from src.vrp import resample_to_month_start


def _make_returns(values, start="2000-01-03"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(values, index=idx, name="log_return", dtype=float)


# ── compute_realised_skew_21d ─────────────────────────────────────────────────
# This is the function called by build_feature_matrix (src/features.py:158).
# Uses scipy.stats.skew(bias=False) — adjusted Fisher-Pearson — on the trailing
# config.RV_WINDOW daily returns ending at and including each monthly date t.
# Dates with fewer than window returns are omitted from the output.


def test_compute_realised_skew_21d_output_is_series():
    returns = _make_returns(np.random.default_rng(0).standard_normal(200))
    result = compute_realised_skew_21d(returns, returns.index[21::21])
    assert isinstance(result, pd.Series)


def test_compute_realised_skew_21d_name():
    returns = _make_returns(np.random.default_rng(0).standard_normal(200))
    result = compute_realised_skew_21d(returns, [returns.index[100]])
    assert result.name == "realised_skew_21d"


def test_compute_realised_skew_21d_uses_scipy_bias_false():
    """Value at t equals scipy.stats.skew(bias=False) on the trailing config.RV_WINDOW returns."""
    rng = np.random.default_rng(42)
    returns = _make_returns(rng.standard_normal(500))
    t = returns.index[100]
    result = compute_realised_skew_21d(returns, [t])
    up_to_t = returns.loc[returns.index <= t]
    expected = float(scipy_skew(up_to_t.iloc[-config.RV_WINDOW:].values, bias=False))
    assert result.at[t] == pytest.approx(expected)


def test_compute_realised_skew_21d_default_window_from_config():
    """Calling without explicit window yields the same value as window=config.RV_WINDOW."""
    rng = np.random.default_rng(42)
    returns = _make_returns(rng.standard_normal(500))
    t = returns.index[200]
    assert (
        compute_realised_skew_21d(returns, [t]).at[t]
        == compute_realised_skew_21d(returns, [t], window=config.RV_WINDOW).at[t]
    )


def test_compute_realised_skew_21d_no_leakage():
    """Value at t is identical whether or not daily returns after t exist in the input."""
    rng = np.random.default_rng(42)
    returns = _make_returns(rng.standard_normal(500))
    t1 = returns.index[100]
    full = compute_realised_skew_21d(returns, [t1])
    truncated = compute_realised_skew_21d(returns.loc[:t1], [t1])
    assert t1 in full.index
    assert t1 in truncated.index
    assert full.at[t1] == truncated.at[t1]


def test_compute_realised_skew_21d_insufficient_history_skipped():
    """Dates with fewer than window available returns are omitted from output."""
    returns = _make_returns(np.random.default_rng(0).standard_normal(200))
    t_short = returns.index[config.RV_WINDOW - 2]  # one fewer return than needed
    t_ok = returns.index[config.RV_WINDOW - 1]     # exactly window returns available
    result = compute_realised_skew_21d(returns, [t_short, t_ok])
    assert t_short not in result.index
    assert t_ok in result.index


def test_compute_realised_skew_21d_index_subset_of_dates():
    """Output index is a subset of the requested dates."""
    returns = _make_returns(np.random.default_rng(0).standard_normal(200))
    dates = returns.index[21::21]
    result = compute_realised_skew_21d(returns, dates)
    assert result.index.isin(dates).all()


def test_compute_realised_skew_21d_empty_dates_returns_empty():
    returns = _make_returns(np.random.default_rng(0).standard_normal(200))
    result = compute_realised_skew_21d(returns, [])
    assert len(result) == 0


def test_compute_realised_skew_21d_raises_on_empty_returns():
    with pytest.raises(ValueError, match="empty"):
        compute_realised_skew_21d(pd.Series([], dtype=float), [])


def test_compute_realised_skew_21d_raises_on_non_monotonic():
    idx = pd.DatetimeIndex(["2020-01-03", "2020-01-01", "2020-01-02"])
    returns = pd.Series([0.01, 0.02, 0.03], index=idx)
    with pytest.raises(ValueError, match="monotonic"):
        compute_realised_skew_21d(returns, [])


# ── build_feature_matrix ─────────────────────────────────────────────────────

def _make_bfm_inputs(n_days: int = 800, seed: int = 0):
    """Synthetic inputs for build_feature_matrix tests."""
    rng = np.random.default_rng(seed)
    day_idx = pd.bdate_range("2000-01-03", periods=n_days)
    returns = pd.Series(rng.normal(0, 0.01, n_days), index=day_idx, name="log_return")
    vix_daily = pd.Series(20.0, index=day_idx, name="close")
    vix_monthly = resample_to_month_start(vix_daily)
    month_idx = vix_monthly.index
    vrp = pd.Series(rng.normal(0.01, 0.005, len(month_idx)), index=month_idx, name="vrp")
    skew_monthly = pd.Series(rng.uniform(110, 140, len(month_idx)), index=month_idx, name="close")
    regime_labels = label_regimes(vix_monthly)
    return vrp, vix_monthly, skew_monthly, returns, regime_labels


def test_build_feature_matrix_has_7_columns():
    vrp, vix_m, skew_m, returns, regimes = _make_bfm_inputs()
    assert len(build_feature_matrix(vrp, vix_m, skew_m, returns, regimes).columns) == 7


def test_build_feature_matrix_column_names_match_locked_set():
    from src.validation import LOCKED_FEATURE_SET
    vrp, vix_m, skew_m, returns, regimes = _make_bfm_inputs()
    result = build_feature_matrix(vrp, vix_m, skew_m, returns, regimes)
    assert set(result.columns) == set(LOCKED_FEATURE_SET)


def test_build_feature_matrix_index_equals_vrp_index():
    vrp, vix_m, skew_m, returns, regimes = _make_bfm_inputs()
    result = build_feature_matrix(vrp, vix_m, skew_m, returns, regimes)
    pd.testing.assert_index_equal(result.index, vrp.index)


def test_build_feature_matrix_vix_level_matches_reindexed_vix():
    vrp, vix_m, skew_m, returns, regimes = _make_bfm_inputs()
    result = build_feature_matrix(vrp, vix_m, skew_m, returns, regimes)
    pd.testing.assert_series_equal(
        result["vix_level"].dropna(),
        vix_m.reindex(result.index).rename("vix_level").dropna(),
    )


def test_build_feature_matrix_cboe_skew_matches_reindexed_skew():
    vrp, vix_m, skew_m, returns, regimes = _make_bfm_inputs()
    result = build_feature_matrix(vrp, vix_m, skew_m, returns, regimes)
    pd.testing.assert_series_equal(
        result["cboe_skew"].dropna(),
        skew_m.reindex(result.index).rename("cboe_skew").dropna(),
    )


def test_build_feature_matrix_vrp_columns_named_by_horizon_from_config():
    """VRP predictor columns are named vrp_h{k}m for each k in config.HAR_LAGS_MONTHS."""
    vrp, vix_m, skew_m, returns, regimes = _make_bfm_inputs()
    result = build_feature_matrix(vrp, vix_m, skew_m, returns, regimes)
    for col in VRP_HORIZON_COLS:
        assert col in result.columns


# ── lag-to-target distance ────────────────────────────────────────────────────
# The invariant is a distance, not a name. A column called vrp_h1m that happens to
# contain a one-month lag says nothing about how far that value sits from the
# target on its own row, which is the quantity that matters and the quantity the
# previous version of this test never measured. Distance is therefore measured, by
# encoding each VRP observation with its own index position and reading the source
# position back off the value.


def _position_encoded_dataset():
    """Model-ready dataset built from a VRP series whose value at position p is p.

    Every VRP-derived cell then carries the position it came from, so the distance
    between any predictor and the target on the same row is read directly rather
    than inferred from the shift expression that produced it.
    """
    from src.dataset import build_model_dataset
    from src.validation import LOCKED_FEATURE_SET

    vrp, vix_m, skew_m, returns, regimes = _make_bfm_inputs()
    coded = pd.Series(np.arange(len(vrp), dtype=float), index=vrp.index, name="vrp")
    features = build_feature_matrix(coded, vix_m, skew_m, returns, regimes).copy()
    features["vrp"] = coded
    dataset = build_model_dataset(
        features, list(LOCKED_FEATURE_SET), target_col="vrp", target_horizon=1
    )
    return coded, dataset, (vix_m, skew_m, returns, regimes)


def _probe_row(dataset):
    """An interior row, away from both the warm-up head and the target-NaN tail."""
    return dataset.index[len(dataset) // 2]


def test_nearest_vrp_predictor_is_one_step_from_target():
    """The nearest VRP predictor on a row sits exactly ONE step from its target.

    This is the whole invariant. It fails on the pre-correction construction, where
    the nearest predictor was VRP(t-1) against a target of VRP(t+1), a distance of
    two, and no naming check could detect it.
    """
    _, dataset, _ = _position_encoded_dataset()
    row = _probe_row(dataset)
    target_pos = int(dataset.at[row, "y"])
    distances = [target_pos - int(dataset.at[row, col]) for col in VRP_HORIZON_COLS]
    assert min(distances) == 1, (
        f"nearest VRP predictor sits {min(distances)} steps from the target at "
        f"{row.date()}; distances by column were "
        f"{dict(zip(VRP_HORIZON_COLS, distances))}"
    )


@pytest.mark.parametrize("k", config.HAR_LAGS_MONTHS)
def test_each_vrp_predictor_sits_at_its_named_horizon_from_target(k):
    """Column vrp_h{k}m sits exactly k steps from the target on the same row."""
    _, dataset, _ = _position_encoded_dataset()
    row = _probe_row(dataset)
    col = f"vrp_h{k}m"
    distance = int(dataset.at[row, "y"]) - int(dataset.at[row, col])
    assert distance == k, (
        f"{col} sits {distance} steps from the target at {row.date()}, expected {k}"
    )


def test_every_non_horizon_feature_sits_one_step_from_target():
    """The four non-VRP features are sourced at t, one step from the target at t+1.

    Guards the other side of the asymmetry that produced the defect: the non-VRP
    columns were already at t while the VRP columns were at t-k, and only the
    latter were ever checked.
    """
    coded, dataset, (vix_m, skew_m, returns, regimes) = _position_encoded_dataset()
    row = _probe_row(dataset)

    row_pos = coded.index.get_loc(row)
    assert int(dataset.at[row, "y"]) - row_pos == 1, "target is not one step after t"

    assert dataset.at[row, "vix_level"] == vix_m.at[row]
    assert dataset.at[row, "cboe_skew"] == skew_m.at[row]
    assert str(dataset.at[row, "regime"]) == str(regimes.at[row])
    expected_skew = compute_realised_skew_21d(returns, [row]).at[row]
    assert dataset.at[row, "realised_skew_21d"] == pytest.approx(expected_skew)


def test_build_feature_matrix_realised_skew_uses_scipy_bias_false():
    """realised_skew_21d at t equals scipy.stats.skew(bias=False) on trailing returns."""
    vrp, vix_m, skew_m, returns, regimes = _make_bfm_inputs()
    full = build_feature_matrix(vrp, vix_m, skew_m, returns, regimes)
    non_nan = full.dropna()
    for t in non_nan.index[:5]:
        up_to_t = returns.loc[returns.index <= t]
        expected = float(scipy_skew(up_to_t.iloc[-config.RV_WINDOW:].values, bias=False))
        assert full.loc[t, "realised_skew_21d"] == pytest.approx(expected, rel=1e-9)


def test_build_feature_matrix_regime_is_categorical_with_locked_labels():
    vrp, vix_m, skew_m, returns, regimes = _make_bfm_inputs()
    result = build_feature_matrix(vrp, vix_m, skew_m, returns, regimes)
    assert isinstance(result["regime"].dtype, pd.CategoricalDtype)
    assert list(result["regime"].cat.categories) == list(config.REGIME_LABELS)
