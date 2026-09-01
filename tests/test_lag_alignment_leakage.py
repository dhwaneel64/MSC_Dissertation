"""Targeted leakage gate for the corrected VRP lag alignment.

The lag columns were changed from vrp.shift(k) to vrp.shift(k-1), so vrp_h1m now
carries VRP(t) rather than VRP(t-1). That column sits exactly on the decision-date
boundary: it is the newest VRP value the row is allowed to see, and a one-month error
in the other direction would be look-ahead rather than staleness. These tests hold
that boundary by truncation, not by reading the shift arithmetic.

Two properties are checked, each demonstrated rather than argued:
  1. vrp_h1m at t is unchanged when every input series is hard-cut at t, and the
     row disappears entirely when the inputs are cut before t.
  2. The VRP(t) value flowing into vrp_h1m is itself a function of data dated <= t
     only, which is what the Bekaert-Hoerova construction requires: VIX squared at t
     minus a HAR-RV forecast fitted recursively on data with index <= t.

Each truncation test is paired with a live-wire check that perturbs in-sample data and
asserts the value does move, so a passing truncation test cannot be passing vacuously.
"""
import numpy as np
import pandas as pd
from src.validation import VRP_HORIZON_COLS
import pytest

from src import config
from src.features import build_feature_matrix
from src.regimes import label_regimes
from src.vrp import build_vrp_series, resample_to_month_start


def _make_inputs(n_days: int = 1200, seed: int = 11):
    """Synthetic daily inputs long enough for the HAR-RV warm-up plus a 6-month lag."""
    rng = np.random.default_rng(seed)
    day_idx = pd.bdate_range("2000-01-03", periods=n_days)
    returns = pd.Series(rng.normal(0, 0.01, n_days), index=day_idx, name="log_return")
    vix_daily = pd.Series(
        rng.uniform(12.0, 30.0, n_days), index=day_idx, name="close"
    )
    skew_daily = pd.Series(
        rng.uniform(110.0, 145.0, n_days), index=day_idx, name="close"
    )
    vix_monthly = resample_to_month_start(vix_daily)
    skew_monthly = resample_to_month_start(skew_daily)
    regimes = label_regimes(vix_monthly)
    return returns, vix_monthly, skew_monthly, regimes


def _features_from(returns, vix_monthly, skew_monthly, regimes, cut=None):
    """Run the full chain (VRP then feature matrix) on inputs optionally hard-cut at `cut`."""
    if cut is not None:
        returns = returns.loc[:cut]
        vix_monthly = vix_monthly.loc[:cut]
        skew_monthly = skew_monthly.loc[:cut]
        regimes = regimes.loc[:cut]
    vrp = build_vrp_series(vix_monthly, returns, vix_monthly.index)
    features = build_feature_matrix(vrp, vix_monthly, skew_monthly, returns, regimes)
    return vrp, features


def _probe_date(vix_monthly, returns):
    """An interior month-start date with enough history for every lag column."""
    vrp = build_vrp_series(vix_monthly, returns, vix_monthly.index)
    # 6 months past the first HAR-valid month, and not the last month.
    return vrp.index[max(config.HAR_LAGS_MONTHS) + 4]


# ---------------------------------------------------------------------------
# 1. vrp_h1m input truncation
# ---------------------------------------------------------------------------

def test_vrp_h1m_input_truncation_at_t_is_byte_identical():
    """Hard-cutting every input at t leaves vrp_h1m at t bit-for-bit unchanged.

    vrp_h1m is VRP(t) under the corrected convention, so t is the last date it is
    allowed to read. The truncated call receives no data whatsoever after t: the daily
    returns, the monthly VIX, the monthly SKEW and the regime labels are all cut, and
    the VRP series is rebuilt from the cut inputs so the recursive HAR-RV fit inside it
    is recomputed too. Any peek past t anywhere in that chain moves the value.

    Compared by float.hex() so the assertion is on the exact bit pattern, not on a
    tolerance that could absorb a small leak.
    """
    returns, vix_m, skew_m, regimes = _make_inputs()
    t = _probe_date(vix_m, returns)

    _, features_full = _features_from(returns, vix_m, skew_m, regimes)
    _, features_cut = _features_from(returns, vix_m, skew_m, regimes, cut=t)

    assert t in features_full.index, "probe date missing from the full feature matrix"
    assert t in features_cut.index, "probe date missing from the truncated feature matrix"

    v_full = float(features_full.at[t, VRP_HORIZON_COLS[0]])
    v_cut = float(features_cut.at[t, VRP_HORIZON_COLS[0]])

    assert not np.isnan(v_full), "vrp_h1m is NaN at the probe date; test is vacuous"
    assert v_full.hex() == v_cut.hex(), (
        f"vrp_h1m at {t.date()} changed when inputs were cut at t: "
        f"full={v_full!r}, truncated={v_cut!r}"
    )


def test_vrp_h1m_row_drops_when_inputs_cut_before_t():
    """Cutting the inputs one month before t removes the row rather than filling it.

    The complement of the truncation test. If the row at t survived a cut that removes
    the data it is built from, the column would be reading something other than VRP(t).
    """
    returns, vix_m, skew_m, regimes = _make_inputs()
    t = _probe_date(vix_m, returns)
    t_prev = vix_m.index[vix_m.index.get_loc(t) - 1]

    _, features_before = _features_from(returns, vix_m, skew_m, regimes, cut=t_prev)

    assert t not in features_before.index, (
        f"row {t.date()} still present after inputs were cut at {t_prev.date()}"
    )
    assert features_before.index.max() <= t_prev, (
        "feature matrix extends past the truncation point"
    )


def test_vrp_h1m_moves_when_in_sample_data_changes():
    """Live-wire check: perturbing the return at t does move vrp_h1m at t.

    Without this, the two truncation tests above could both pass on a column that was
    constant or NaN. The perturbation is applied to the single daily return dated t,
    which feeds the HAR-RV daily regressor and therefore VRP(t).
    """
    returns, vix_m, skew_m, regimes = _make_inputs()
    t = _probe_date(vix_m, returns)

    _, features_base = _features_from(returns, vix_m, skew_m, regimes)

    bumped = returns.copy()
    bumped.at[t] = bumped.at[t] + 0.05
    _, features_bumped = _features_from(bumped, vix_m, skew_m, regimes)

    v_base = float(features_base.at[t, VRP_HORIZON_COLS[0]])
    v_bumped = float(features_bumped.at[t, VRP_HORIZON_COLS[0]])

    assert v_base != v_bumped, (
        "vrp_h1m did not respond to a change in the return dated t; "
        "the truncation tests would pass vacuously"
    )


def test_vrp_h1m_equals_vrp_at_t_not_t_minus_1():
    """The corrected alignment itself: vrp_h1m at t is VRP(t), and is not VRP(t-1)."""
    returns, vix_m, skew_m, regimes = _make_inputs()
    t = _probe_date(vix_m, returns)
    vrp, features = _features_from(returns, vix_m, skew_m, regimes)
    t_prev = vrp.index[vrp.index.get_loc(t) - 1]

    assert float(features.at[t, VRP_HORIZON_COLS[0]]).hex() == float(vrp.at[t]).hex()
    assert float(features.at[t, VRP_HORIZON_COLS[0]]) != float(vrp.at[t_prev])


# ---------------------------------------------------------------------------
# 2. VRP(t) itself carries only time-t information
# ---------------------------------------------------------------------------

def test_vrp_at_t_unchanged_when_all_inputs_truncated_at_t():
    """VRP(t) is bit-for-bit unchanged when the inputs carry nothing after t.

    Demonstrated by truncation rather than argued from the construction. The truncated
    call sees no returns and no VIX after t, so the recursive HAR-RV fit inside
    build_vrp_series has strictly less data available; if it were reaching forward, the
    two values would differ.
    """
    returns, vix_m, _, _ = _make_inputs()
    t = _probe_date(vix_m, returns)

    full = build_vrp_series(vix_m, returns, [t])
    truncated = build_vrp_series(vix_m.loc[:t], returns.loc[:t], [t])

    assert t in full.index and t in truncated.index, "VRP undefined at the probe date"
    assert float(full.at[t]).hex() == float(truncated.at[t]).hex(), (
        f"VRP({t.date()}) changed under truncation at t: "
        f"full={float(full.at[t])!r}, truncated={float(truncated.at[t])!r}"
    )


def test_vrp_at_t_ignores_data_after_t():
    """Overwriting every return after t leaves VRP(t) unchanged.

    A different probe from truncation: the post-t data is present but replaced with
    values large enough that any dependence on it would be obvious. Covers the case
    where a forward window is read but happens to be absent under truncation.
    """
    returns, vix_m, _, _ = _make_inputs()
    t = _probe_date(vix_m, returns)

    corrupted = returns.copy()
    corrupted.loc[corrupted.index > t] = 0.25

    base = build_vrp_series(vix_m, returns, [t])
    after = build_vrp_series(vix_m, corrupted, [t])

    assert float(base.at[t]).hex() == float(after.at[t]).hex(), (
        f"VRP({t.date()}) responded to returns dated after t"
    )


def test_vrp_at_t_responds_to_data_at_or_before_t():
    """Live-wire check for the two tests above: VRP(t) does move on in-sample changes.

    The VIX leg is changed at t and the return leg is changed at t separately, so both
    halves of VRP(t) = VIX^2(t)/scale - E_t[RV^2(t+1)] are shown to be live.
    """
    returns, vix_m, _, _ = _make_inputs()
    t = _probe_date(vix_m, returns)
    base = float(build_vrp_series(vix_m, returns, [t]).at[t])

    vix_bumped = vix_m.copy()
    vix_bumped.at[t] = vix_bumped.at[t] + 5.0
    moved_vix = float(build_vrp_series(vix_bumped, returns, [t]).at[t])
    assert moved_vix != base, "VRP(t) did not respond to VIX(t)"

    ret_bumped = returns.copy()
    ret_bumped.at[t] = ret_bumped.at[t] + 0.05
    moved_ret = float(build_vrp_series(vix_m, ret_bumped, [t]).at[t])
    assert moved_ret != base, "VRP(t) did not respond to the return dated t"


def test_vrp_h1m_carries_the_truncation_invariant_vrp_value():
    """Ties the two halves together.

    The value the feature column carries at t is the same value build_vrp_series
    produces at t when it is given nothing after t. This is the property the
    correction has to preserve: moving the lag forward by one month put VRP(t) into
    the row, and VRP(t) is admissible at t.
    """
    returns, vix_m, skew_m, regimes = _make_inputs()
    t = _probe_date(vix_m, returns)

    _, features = _features_from(returns, vix_m, skew_m, regimes)
    truncated_vrp = build_vrp_series(vix_m.loc[:t], returns.loc[:t], [t])

    assert float(features.at[t, VRP_HORIZON_COLS[0]]).hex() == float(truncated_vrp.at[t]).hex()
