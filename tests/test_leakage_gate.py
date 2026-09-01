"""Full leakage gate under the corrected horizon alignment.

Every feature, target and model input is held to the strong input-truncation form
and its complement, on the same synthetic chain the pipeline runs on real data:

  strong      hard-cut every input series at t, then assert the value at t is
              bit-for-bit identical to the full-run value. Compared by float.hex()
              so the assertion is on the exact bit pattern, not a tolerance that
              could absorb a small leak.
  complement  cut the inputs before t, then assert the value at t is gone rather
              than filled. Without this a constant or NaN column would satisfy the
              strong form vacuously.

The target y is the one quantity that is not knowable at t by construction: it is
VRP(t+1), a label the walk-forward engine only ever trains on once realised, and
never passes to predict(). Its observability date is t+1, so it is held to the same
two forms anchored one month later: identical under a cut at t+1, absent under a
cut at t.

The correction this gate protects moved VRP(t) into the design matrix at distance 1
from the target. The failure direction is therefore one step further forward, which
would place VRP(t+1), the target itself, into the feature row. That specific failure
is tested directly by position-encoding the VRP series, not inferred from the shift
expression.

Columns are always addressed through LOCKED_FEATURE_SET and VRP_HORIZON_COLS, never
by literal name, so renaming a column cannot silently orphan a test here.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.circle3a.pnl import build_pnl_frame
from src.circle3a.positions import build_position_series
from src.dataset import build_model_dataset, build_model_ready_dataset
from src.features import build_feature_matrix
from src.har_rv import fit_har_rv_forecast, realised_variance_target
from src.mincer_zarnowitz import build_mz_frame
from src.models.extended_ols import ExtendedOLSModel
from src.models.xgboost_vrp import XGB_FEATURE_ORDER, tune_xgboost_hyperparameters
from src.regimes import label_regimes
from src.results import score_walk_forward
from src.validation import LOCKED_FEATURE_SET, VRP_HORIZON_COLS
from src.vrp import build_vrp_series, resample_to_month_start
from src.walk_forward import make_model_factory_from_class, walk_forward

NUMERIC_LOCKED = tuple(c for c in LOCKED_FEATURE_SET if c != "regime")


# ---------------------------------------------------------------------------
# Shared synthetic chain
# ---------------------------------------------------------------------------

def _inputs(n_days: int = 2600, seed: int = 17):
    """Daily inputs long enough for the HAR-RV warm-up, the 6-month horizon and CV."""
    rng = np.random.default_rng(seed)
    day_idx = pd.bdate_range("2000-01-03", periods=n_days)
    returns = pd.Series(rng.normal(0, 0.01, n_days), index=day_idx, name="log_return")
    vix_daily = pd.Series(rng.uniform(12.0, 34.0, n_days), index=day_idx, name="close")
    skew_daily = pd.Series(rng.uniform(110.0, 145.0, n_days), index=day_idx, name="close")
    vix_m = resample_to_month_start(vix_daily)
    skew_m = resample_to_month_start(skew_daily)
    return returns, vix_m, skew_m, label_regimes(vix_m)


def _cut(series_tuple, cut):
    """Hard-cut every input series at `cut`, or return them untouched when cut is None."""
    if cut is None:
        return series_tuple
    return tuple(s.loc[:cut] for s in series_tuple)


def _features_from(returns, vix_m, skew_m, regimes, cut=None):
    """Run the full chain, VRP then feature matrix, on optionally truncated inputs."""
    returns, vix_m, skew_m, regimes = _cut((returns, vix_m, skew_m, regimes), cut)
    vrp = build_vrp_series(vix_m, returns, vix_m.index)
    return vrp, build_feature_matrix(vrp, vix_m, skew_m, returns, regimes)


def _dataset_from(returns, vix_m, skew_m, regimes, cut=None):
    """Model-ready dataset from optionally truncated inputs."""
    returns, vix_m, skew_m, regimes = _cut((returns, vix_m, skew_m, regimes), cut)
    vrp = build_vrp_series(vix_m, returns, vix_m.index)
    return build_model_ready_dataset(vrp, vix_m, skew_m, returns, regimes)


def _probe(vix_m, returns):
    """A late interior month.

    Late enough that a hard cut at t still leaves a usable dataset behind it (the
    builder requires 24 rows), and far enough from the end that the target at t and
    the forward realised window are both available in the full run.
    """
    vrp = build_vrp_series(vix_m, returns, vix_m.index)
    return vrp.index[int(len(vrp) * 0.7)]


def _hex(v):
    """Exact bit pattern of a float, for byte-identity assertions."""
    return float(v).hex()


def _cell(frame, row, col):
    """One cell as a comparable token, handling the categorical regime column."""
    v = frame.at[row, col]
    return str(v) if col == "regime" else _hex(v)


@pytest.fixture(scope="module")
def chain():
    returns, vix_m, skew_m, regimes = _inputs()
    t = _probe(vix_m, returns)
    return returns, vix_m, skew_m, regimes, t


# ---------------------------------------------------------------------------
# HAR-RV forecaster and the VRP series
# ---------------------------------------------------------------------------

def test_har_rv_forecast_strong(chain):
    """The HAR-RV variance forecast at t is unchanged when the inputs stop at t."""
    returns, _, _, _, t = chain
    full = fit_har_rv_forecast(returns, [t])
    cut = fit_har_rv_forecast(returns.loc[:t], [t])
    assert t in full.index and t in cut.index
    assert _hex(full.at[t]) == _hex(cut.at[t])


def test_har_rv_forecast_complement(chain):
    """Cutting the returns before t leaves no HAR-RV forecast at t."""
    returns, vix_m, _, _, t = chain
    t_prev = vix_m.index[vix_m.index.get_loc(t) - 1]
    assert t not in fit_har_rv_forecast(returns.loc[:t_prev], [t]).index


def test_vrp_series_strong(chain):
    """VRP(t) is unchanged when every input stops at t."""
    returns, vix_m, _, _, t = chain
    full = build_vrp_series(vix_m, returns, vix_m.index)
    cut = build_vrp_series(vix_m.loc[:t], returns.loc[:t], vix_m.loc[:t].index)
    assert _hex(full.at[t]) == _hex(cut.at[t])


def test_vrp_series_complement(chain):
    """VRP(t) disappears when the inputs stop before t."""
    returns, vix_m, _, _, t = chain
    t_prev = vix_m.index[vix_m.index.get_loc(t) - 1]
    cut = build_vrp_series(vix_m.loc[:t_prev], returns.loc[:t_prev],
                           vix_m.loc[:t_prev].index)
    assert t not in cut.index


# ---------------------------------------------------------------------------
# The seven locked features
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("col", LOCKED_FEATURE_SET)
def test_locked_feature_strong(chain, col):
    """Each locked feature at t is unchanged when every input is hard-cut at t."""
    returns, vix_m, skew_m, regimes, t = chain
    _, full = _features_from(returns, vix_m, skew_m, regimes)
    _, cut = _features_from(returns, vix_m, skew_m, regimes, cut=t)

    assert t in full.index and t in cut.index
    if col in NUMERIC_LOCKED:
        assert not np.isnan(float(full.at[t, col])), f"{col} is NaN at t, test vacuous"
    assert _cell(full, t, col) == _cell(cut, t, col), (
        f"{col} at {t.date()} changed when inputs were cut at t"
    )


@pytest.mark.parametrize("col", LOCKED_FEATURE_SET)
def test_locked_feature_complement(chain, col):
    """Each locked feature at t is gone when the inputs are cut before t."""
    returns, vix_m, skew_m, regimes, t = chain
    t_prev = vix_m.index[vix_m.index.get_loc(t) - 1]
    _, before = _features_from(returns, vix_m, skew_m, regimes, cut=t_prev)
    assert t not in before.index
    assert before.index.max() <= t_prev


def test_locked_features_are_live(chain):
    """Perturbing the return dated t moves the VRP and skew features at t.

    Without this the truncation tests above could pass on a dead column.
    """
    returns, vix_m, skew_m, regimes, t = chain
    _, base = _features_from(returns, vix_m, skew_m, regimes)
    bumped = returns.copy()
    bumped.at[t] = bumped.at[t] + 0.05
    _, moved = _features_from(bumped, vix_m, skew_m, regimes)
    for col in (VRP_HORIZON_COLS[0], "realised_skew_21d"):
        assert base.at[t, col] != moved.at[t, col], f"{col} did not respond to r(t)"


# ---------------------------------------------------------------------------
# The target
# ---------------------------------------------------------------------------

def test_target_strong_at_its_observability_date(chain):
    """y at row t is unchanged when the inputs are cut at t+1, where it is realised."""
    returns, vix_m, skew_m, regimes, t = chain
    t_next = vix_m.index[vix_m.index.get_loc(t) + 1]
    full = _dataset_from(returns, vix_m, skew_m, regimes)
    cut = _dataset_from(returns, vix_m, skew_m, regimes, cut=t_next)
    assert t in full.index and t in cut.index
    assert _hex(full.at[t, "y"]) == _hex(cut.at[t, "y"])


def test_target_absent_under_cut_at_t(chain):
    """y at row t is absent when the inputs stop at t, because it is a t+1 quantity."""
    returns, vix_m, skew_m, regimes, t = chain
    cut = _dataset_from(returns, vix_m, skew_m, regimes, cut=t)
    assert t not in cut.index


# ---------------------------------------------------------------------------
# The failure direction: VRP(t+1) must not reach the feature row
# ---------------------------------------------------------------------------

def test_no_feature_row_carries_vrp_at_or_after_t_plus_1(chain):
    """No VRP column on any row carries a VRP observation dated later than that row.

    The VRP series is replaced by one whose value at position p is p, so each cell
    reports the position it was sourced from. The correction moved the nearest
    column to VRP(t), one step from the target; a further step forward would put
    VRP(t+1), the target itself, into the row. This asserts that never happens on
    any row, for any horizon column.
    """
    returns, vix_m, skew_m, regimes, _ = chain
    vrp = build_vrp_series(vix_m, returns, vix_m.index)
    coded = pd.Series(np.arange(len(vrp), dtype=float), index=vrp.index, name="vrp")
    features = build_feature_matrix(coded, vix_m, skew_m, returns, regimes).copy()
    features["vrp"] = coded
    dataset = build_model_dataset(features, list(LOCKED_FEATURE_SET),
                                  target_col="vrp", target_horizon=1)

    for row in dataset.index:
        p = coded.index.get_loc(row)
        for col in VRP_HORIZON_COLS:
            src = int(dataset.at[row, col])
            assert src <= p, (
                f"{col} at {row.date()} carries VRP from position {src}, which is "
                f"{src - p} step(s) after the row date"
            )
        assert int(dataset.at[row, "y"]) == p + 1, "target is not VRP(t+1)"


def test_nearest_horizon_column_is_exactly_one_step_from_target(chain):
    """The nearest VRP column sits at distance 1, measured by position."""
    returns, vix_m, skew_m, regimes, _ = chain
    vrp = build_vrp_series(vix_m, returns, vix_m.index)
    coded = pd.Series(np.arange(len(vrp), dtype=float), index=vrp.index, name="vrp")
    features = build_feature_matrix(coded, vix_m, skew_m, returns, regimes).copy()
    features["vrp"] = coded
    dataset = build_model_dataset(features, list(LOCKED_FEATURE_SET),
                                  target_col="vrp", target_horizon=1)
    row = dataset.index[len(dataset) // 2]
    target = int(dataset.at[row, "y"])
    dists = [target - int(dataset.at[row, c]) for c in VRP_HORIZON_COLS]
    assert min(dists) == 1
    assert dists == list(config.HAR_LAGS_MONTHS)


# ---------------------------------------------------------------------------
# Per-step scaler and per-step refit
# ---------------------------------------------------------------------------

def _split(dataset):
    """A training window and the single step predicted immediately after it."""
    pos = len(dataset) // 2
    return dataset.index[pos], dataset.index[pos + 1]


def test_per_step_scaler_ignores_post_boundary_rows(chain):
    """The Extended OLS scaler fitted at a step is unchanged when later rows change.

    mu_ and sigma_ come from X_train only, so corrupting every row after the
    training boundary must leave the stored scaler and the fitted coefficients
    bit-for-bit identical.
    """
    returns, vix_m, skew_m, regimes, _ = chain
    dataset = _dataset_from(returns, vix_m, skew_m, regimes)
    train_end, _ = _split(dataset)
    train = dataset.loc[:train_end]

    corrupted = dataset.copy()
    after = corrupted.index > train_end
    for col in NUMERIC_LOCKED + ("y",):
        corrupted.loc[after, col] = 99.0

    a = ExtendedOLSModel(dataset)
    a.fit(train[list(NUMERIC_LOCKED)], train["y"])
    b = ExtendedOLSModel(corrupted)
    b.fit(corrupted.loc[:train_end, list(NUMERIC_LOCKED)], corrupted.loc[:train_end, "y"])

    for col in NUMERIC_LOCKED:
        assert _hex(a.mu_[col]) == _hex(b.mu_[col])
        assert _hex(a.sigma_[col]) == _hex(b.sigma_[col])
        assert _hex(a.params_[col]) == _hex(b.params_[col])


def test_per_step_refit_prediction_strong(chain):
    """The walk-forward prediction for step t is unchanged when the data stops at t."""
    returns, vix_m, skew_m, regimes, _ = chain
    dataset = _dataset_from(returns, vix_m, skew_m, regimes)
    train_end, step = _split(dataset)
    factory = make_model_factory_from_class(ExtendedOLSModel, monthly_vrp=dataset)

    full = walk_forward(dataset, list(NUMERIC_LOCKED), factory, train_end, "y")
    cut = walk_forward(dataset.loc[:step], list(NUMERIC_LOCKED), factory, train_end, "y")
    assert step in full.index and step in cut.index
    assert _hex(full.at[step, "y_pred"]) == _hex(cut.at[step, "y_pred"])


def test_per_step_refit_prediction_complement(chain):
    """No prediction is produced for step t when the data stops before t."""
    returns, vix_m, skew_m, regimes, _ = chain
    dataset = _dataset_from(returns, vix_m, skew_m, regimes)
    train_end, step = _split(dataset)
    factory = make_model_factory_from_class(ExtendedOLSModel, monthly_vrp=dataset)
    before = walk_forward(dataset.loc[:train_end], list(NUMERIC_LOCKED), factory,
                          dataset.index[dataset.index.get_loc(train_end) - 1], "y")
    assert step not in before.index


# ---------------------------------------------------------------------------
# XGBoost tuning selection
# ---------------------------------------------------------------------------

def test_xgboost_tuning_selection_ignores_post_boundary_data(chain):
    """The tuned hyperparameters are unchanged when post-boundary inputs are corrupted.

    tune_xgboost_hyperparameters truncates the daily returns and the monthly VIX to
    the training boundary internally, so replacing everything after the boundary
    must not move the selection. A small explicit grid is used: the property under
    test is truncation-invariance of the choice, not the locked grid itself.
    """
    returns, vix_m, skew_m, regimes, _ = chain
    dataset = _dataset_from(returns, vix_m, skew_m, regimes)
    # Far enough in that the training window clears XGB_CV_MIN_TRAIN_MONTHS plus the
    # folds, which is what _expanding_cv_folds requires.
    boundary = dataset.index[int(len(dataset) * 0.8)]
    train = dataset.loc[:boundary]
    grid = [
        {"max_depth": 2, "learning_rate": 0.1, "n_estimators": 100, "min_child_weight": 1},
        {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 100, "min_child_weight": 5},
    ]

    base = tune_xgboost_hyperparameters(train, returns, vix_m, monthly_vrp=dataset,
                                        grid=grid)
    bad_returns = returns.copy()
    bad_returns.loc[bad_returns.index > boundary] = 0.25
    bad_vix = vix_m.copy()
    bad_vix.loc[bad_vix.index > boundary] = 90.0
    after = tune_xgboost_hyperparameters(train, bad_returns, bad_vix,
                                         monthly_vrp=dataset, grid=grid)

    assert base.best_params == after.best_params
    assert _hex(base.grid_scores["mean_val_qlike"].iloc[0]) == _hex(
        after.grid_scores["mean_val_qlike"].iloc[0]
    )


# ---------------------------------------------------------------------------
# Family 2 sites: the corrected realised pairing
# ---------------------------------------------------------------------------

def test_mincer_zarnowitz_row_strong_at_its_observability_date(chain):
    """The MZ row at t is unchanged when the inputs stop at its observability date.

    The MZ pairing is a scoring-stage quantity: its realised leg is the forward
    window that opens at t, so the row is observable once that window closes rather
    than at t itself. Cutting there must leave both legs bit-for-bit unchanged.
    """
    returns, vix_m, _, _, t = chain
    obs = vix_m.index[vix_m.index.get_loc(t) + 2]
    full = build_mz_frame(vix_m, returns)
    cut = build_mz_frame(vix_m.loc[:obs], returns.loc[:obs])
    assert t in full.index and t in cut.index
    assert _hex(full.at[t, "vix_variance"]) == _hex(cut.at[t, "vix_variance"])
    assert _hex(full.at[t, "realised_variance_next"]) == _hex(
        cut.at[t, "realised_variance_next"]
    )


def test_mincer_zarnowitz_realised_pairing_is_the_estimator_at_t(chain):
    """The MZ realised leg at t is realised_variance_target at t, not at t+1."""
    returns, vix_m, _, _, t = chain
    frame = build_mz_frame(vix_m, returns)
    realised_m = resample_to_month_start(realised_variance_target(returns))
    t_next = realised_m.index[realised_m.index.get_loc(t) + 1]
    assert _hex(frame.at[t, "realised_variance_next"]) == _hex(realised_m.at[t])
    assert _hex(frame.at[t, "realised_variance_next"]) != _hex(realised_m.at[t_next])


def test_mincer_zarnowitz_complement(chain):
    """The MZ row at t drops when the forward window behind its realised leg is cut."""
    returns, vix_m, _, _, t = chain
    t_prev = vix_m.index[vix_m.index.get_loc(t) - 1]
    before = build_mz_frame(vix_m.loc[:t_prev], returns.loc[:t_prev])
    assert t not in before.index


def test_scorer_realised_leg_is_the_estimator_at_t_plus_1(chain):
    """score_walk_forward pairs its t+1 implied leg with the estimator at t+1.

    results.py is unchanged by the Family 2 correction and must stay that way: it
    scores a leg built from VIX(t+1), so its realised counterpart is one month
    further forward than the mincer_zarnowitz and circle3a legs.
    """
    returns, vix_m, skew_m, regimes, _ = chain
    dataset = _dataset_from(returns, vix_m, skew_m, regimes)
    train_end, _ = _split(dataset)
    factory = make_model_factory_from_class(ExtendedOLSModel, monthly_vrp=dataset)
    wf = walk_forward(dataset, list(NUMERIC_LOCKED), factory, train_end, "y")

    score = score_walk_forward(wf, vix_m.shift(-1).reindex(wf.index), returns)
    realised_m = resample_to_month_start(realised_variance_target(returns))
    expected = realised_m.shift(-1).reindex(wf.index).to_numpy(dtype=float)
    np.testing.assert_array_equal(score["realised_variance_next"], expected)


# ---------------------------------------------------------------------------
# Circle 3A
# ---------------------------------------------------------------------------

def _positions_from(returns, vix_m, skew_m, regimes, train_end, cut=None):
    """Position series built through the full chain on optionally truncated inputs.

    train_end is passed in rather than derived from the truncated dataset, so a cut
    changes only how much data is available and never which step is being scored.
    """
    dataset = _dataset_from(returns, vix_m, skew_m, regimes, cut=cut)
    factory = make_model_factory_from_class(ExtendedOLSModel, monthly_vrp=dataset)
    wf = walk_forward(dataset, list(NUMERIC_LOCKED), factory, train_end, "y")
    return build_position_series(wf["y_pred"], dataset.loc[wf.index, "regime"])


def _position_setup(returns, vix_m, skew_m, regimes):
    """Full-run positions plus the fixed train_end and the probe step inside them."""
    dataset = _dataset_from(returns, vix_m, skew_m, regimes)
    train_end = dataset.index[int(len(dataset) * 0.5)]
    full = _positions_from(returns, vix_m, skew_m, regimes, train_end)
    return full, train_end, full.index[len(full) // 2]


def _corrupt_after(t, returns, vix_m, skew_m):
    """Replace every input observation after t with an extreme value.

    Truncation cannot be used at this site. build_model_ready_dataset drops any row
    whose target is unavailable, so hard-cutting at t removes row t itself and the
    engine has no step to score, which would make the strong form untestable rather
    than passing or failing. Corruption keeps the row and is the harder probe: the
    post-t data is present and wrong, so any dependence on it moves the value.
    """
    r = returns.copy()
    r.loc[r.index > t] = 0.25
    v = vix_m.copy()
    v.loc[v.index > t] = 95.0
    s = skew_m.copy()
    s.loc[s.index > t] = 200.0
    return r, v, s


def test_circle3a_position_strong(chain):
    """The position at t is unchanged when every input after t is corrupted."""
    returns, vix_m, skew_m, regimes, _ = chain
    full, train_end, t = _position_setup(returns, vix_m, skew_m, regimes)
    r, v, s = _corrupt_after(t, returns, vix_m, skew_m)
    cut = _positions_from(r, v, s, label_regimes(v), train_end)
    assert t in cut.index
    for col in ("forecast", "expanding_median"):
        assert _hex(full.at[t, col]) == _hex(cut.at[t, col])
    assert int(full.at[t, "position_naive"]) == int(cut.at[t, "position_naive"])
    assert int(full.at[t, "position_conditioned"]) == int(cut.at[t, "position_conditioned"])


def test_circle3a_position_complement(chain):
    """The position at t is gone when the inputs are cut before t."""
    returns, vix_m, skew_m, regimes, _ = chain
    _, train_end, t = _position_setup(returns, vix_m, skew_m, regimes)
    t_prev = vix_m.index[vix_m.index.get_loc(t) - 1]
    before = _positions_from(returns, vix_m, skew_m, regimes, train_end, cut=t_prev)
    assert t not in before.index


def test_circle3a_pnl_realised_leg_is_the_estimator_at_t(chain):
    """The 3A payoff pairs the entry-month strike with the estimator at t.

    Both legs are dated t, which is what makes the payoff the realised VRP by
    construction. Pairing the strike with the estimator at t+1 is the alignment
    defect this pins.
    """
    returns, vix_m, skew_m, regimes, _ = chain
    positions, _, _ = _position_setup(returns, vix_m, skew_m, regimes)
    frame = build_pnl_frame(positions, vix_m, returns)
    realised_m = resample_to_month_start(realised_variance_target(returns))
    strike = vix_m.reindex(frame.index) ** 2 / config.VIX_VARIANCE_SCALE

    np.testing.assert_array_equal(
        frame["payoff"].to_numpy(),
        (strike - realised_m.reindex(frame.index)).to_numpy(),
    )
    misaligned = strike - realised_m.shift(-1).reindex(frame.index)
    assert not np.allclose(frame["payoff"].to_numpy(), misaligned.to_numpy(),
                           equal_nan=True)


def test_circle3a_pnl_information_sets_are_disjoint(chain):
    """The realised leg reaches past t; the position it multiplies does not.

    Corrupting every return after t moves the payoff at t and leaves the position
    at t untouched, which is the disjointness the spec requires.
    """
    returns, vix_m, skew_m, regimes, _ = chain
    positions, train_end, t = _position_setup(returns, vix_m, skew_m, regimes)

    base = build_pnl_frame(positions, vix_m, returns)
    corrupted = returns.copy()
    corrupted.loc[corrupted.index > t] = 0.2
    moved = build_pnl_frame(positions, vix_m, corrupted)
    assert base.at[t, "payoff"] != moved.at[t, "payoff"], "realised leg is not live"

    r, v, s = _corrupt_after(t, returns, vix_m, skew_m)
    cut_positions = _positions_from(r, v, s, label_regimes(v), train_end)
    assert int(cut_positions.at[t, "position_naive"]) == int(
        positions.at[t, "position_naive"]
    )
    assert int(cut_positions.at[t, "position_conditioned"]) == int(
        positions.at[t, "position_conditioned"]
    )


# ---------------------------------------------------------------------------
# The rename cannot orphan a test
# ---------------------------------------------------------------------------

def test_no_test_references_a_horizon_column_by_literal_name():
    """No test file hardcodes a VRP horizon column name.

    Every reference goes through LOCKED_FEATURE_SET or VRP_HORIZON_COLS, so
    renaming the columns cannot leave a test silently asserting on a column that
    no longer exists.
    """
    pattern = re.compile(r"[\"']vrp_h\d+m[\"']")
    offenders = {}
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        hits = pattern.findall(path.read_text(encoding="utf-8"))
        if hits:
            offenders[path.name] = len(hits)
    assert not offenders, f"literal horizon column names found: {offenders}"
