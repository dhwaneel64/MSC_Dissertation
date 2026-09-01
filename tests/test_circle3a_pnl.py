"""Guards for the Circle 3A synthetic variance-swap P&L.

The mandatory preconditions before the benchmark/metrics tasks: the position
columns are consumed unchanged (P&L does not mutate them), the realised leg is the
one pipeline forward variance target that score_walk_forward uses (not
compute_realised_vol), costs are charged on every transition including gate-exit
closes, and NaN-realised months are dropped and counted.
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.circle3a.positions import build_position_series
from src.circle3a.pnl import build_pnl_frame
from src.har_rv import realised_variance_target
from src.realised_vol import compute_realised_vol
from src.results import score_walk_forward
from src.vrp import resample_to_month_start

PNL_COLUMNS = [
    "payoff",
    "position_naive",
    "position_conditioned",
    "gross_naive",
    "gross_conditioned",
    "net_naive_base",
    "net_conditioned_base",
    "net_naive_stress",
    "net_conditioned_stress",
]


def _make_pipeline(seed: int = 0, n_days: int = 900):
    """Synthetic daily returns, month-start VIX, and a position frame over the
    full monthly index (the last months carry a NaN realised leg by construction)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2004-01-01", periods=n_days)
    returns = pd.Series(rng.normal(0.0, 0.011, n_days), index=idx, name="spy")

    month_idx = resample_to_month_start(returns).index
    vix_monthly = pd.Series(
        rng.uniform(12.0, 38.0, len(month_idx)), index=month_idx, name="vix"
    )
    forecast = pd.Series(
        rng.normal(0.01, 0.02, len(month_idx)), index=month_idx, name="y_pred"
    )
    labels = np.array(config.REGIME_LABELS)
    regime = pd.Series(
        labels[rng.integers(0, len(labels), len(month_idx))],
        index=month_idx,
        name="regime",
    )
    positions = build_position_series(forecast, regime)
    return returns, vix_monthly, positions


def test_columns_and_index_subset():
    returns, vix_monthly, positions = _make_pipeline()
    frame = build_pnl_frame(positions, vix_monthly, returns)
    assert list(frame.columns) == PNL_COLUMNS
    assert frame.index.isin(positions.index).all()
    assert "n_nan_dropped" in frame.attrs


def test_positions_columns_unchanged():
    """P&L must not mutate the position columns it consumes."""
    returns, vix_monthly, positions = _make_pipeline(seed=1)
    frame = build_pnl_frame(positions, vix_monthly, returns)
    for col in ("position_naive", "position_conditioned"):
        pd.testing.assert_series_equal(
            frame[col], positions.loc[frame.index, col], check_names=True
        )


def test_realised_leg_matches_score_walk_forward():
    """Byte-identical realised leg to score_walk_forward on shared dates (one truth),
    and the payoff equals strike minus that same realised leg exactly."""
    returns, vix_monthly, positions = _make_pipeline(seed=2)
    frame = build_pnl_frame(positions, vix_monthly, returns)

    realised_monthly = resample_to_month_start(realised_variance_target(returns))

    # vix_next must be non-NaN over the scored index; restrict to dates where the
    # next month exists in vix_monthly.
    vix_next = vix_monthly.shift(-1).reindex(positions.index)
    scored_idx = positions.index[vix_next.notna().to_numpy()]
    wf_result = pd.DataFrame({"y_true": 0.0, "y_pred": 0.0}, index=scored_idx)
    sw = score_walk_forward(wf_result, vix_monthly.shift(-1), returns)
    sw_realised = sw["realised_variance_next"]

    # One estimator, two dates. score_walk_forward scores a leg dated t+1 (its
    # implied side is built from VIX(t+1)), so its realised counterpart is the
    # estimator at t+1. This module scores a leg dated t (the entry-month strike),
    # so its realised counterpart is the estimator at t. Both come from
    # realised_variance_target; neither introduces a second estimator.
    a = realised_monthly.shift(-1).reindex(scored_idx).to_numpy()
    valid = ~np.isnan(sw_realised)
    assert valid.sum() > 0
    np.testing.assert_array_equal(a[valid], sw_realised[valid])

    # payoff = strike(t) - realised(t), byte-identical to the same operation, and
    # this pairing is what makes the payoff the realised VRP by construction.
    strike = vix_monthly.reindex(frame.index) ** 2 / config.VIX_VARIANCE_SCALE
    expected_payoff = strike - realised_monthly.reindex(frame.index)
    np.testing.assert_array_equal(frame["payoff"].to_numpy(), expected_payoff.to_numpy())

    # The scorer's leg is one month further forward than this module's, so pairing
    # the entry-month strike with it would be the alignment defect.
    misaligned = strike - realised_monthly.shift(-1).reindex(frame.index)
    assert not np.allclose(
        frame["payoff"].to_numpy(), misaligned.to_numpy(), equal_nan=True
    )


def test_realised_leg_is_not_compute_realised_vol():
    """Unit-consistency: the realised leg is the pipeline variance target, not the
    demeaned vol-points compute_realised_vol series."""
    returns, vix_monthly, positions = _make_pipeline(seed=3)
    frame = build_pnl_frame(positions, vix_monthly, returns)

    pipeline = (
        resample_to_month_start(realised_variance_target(returns))
        .reindex(frame.index)
    )
    strike = vix_monthly.reindex(frame.index) ** 2 / config.VIX_VARIANCE_SCALE
    np.testing.assert_array_equal(
        frame["payoff"].to_numpy(), (strike - pipeline).to_numpy()
    )

    # A compute_realised_vol-based variance series differs from the pipeline target.
    wrong = resample_to_month_start(compute_realised_vol(returns)) ** 2 / config.VIX_VARIANCE_SCALE
    wrong = wrong.reindex(frame.index)
    assert not np.allclose(pipeline.to_numpy(), wrong.to_numpy(), equal_nan=True)


def _known_positions(vix_monthly):
    """Deterministic position frame with an explicit open/close pattern:
    flat, open(0->1 hold), close(1->0), open again."""
    idx = vix_monthly.index[:6]
    naive = pd.Series([0, 1, 1, 0, 1, 1], index=idx, name="position_naive")
    conditioned = pd.Series([0, 1, 1, 0, 0, 0], index=idx, name="position_conditioned")
    return pd.DataFrame(
        {
            "forecast": 0.0,
            "expanding_median": 0.0,
            "regime": "calm",
            "position_naive": naive,
            "position_conditioned": conditioned,
        }
    )


def test_cost_charged_on_every_transition_including_gate_exit():
    returns, vix_monthly, _ = _make_pipeline(seed=4)
    positions = _known_positions(vix_monthly)
    frame = build_pnl_frame(positions, vix_monthly, returns)

    strike = vix_monthly.reindex(frame.index) ** 2 / config.VIX_VARIANCE_SCALE
    # naive transitions at rows 1 (open), 3 (close), 4 (open): 3 charges.
    naive_cost = frame["gross_naive"] - frame["net_naive_base"]
    charged = naive_cost != 0.0
    expected_charge_dates = positions.index[[1, 3, 4]]
    assert set(frame.index[charged.to_numpy()]) == set(expected_charge_dates)
    # each charge equals base haircut * that month's variance strike.
    for d in expected_charge_dates:
        if d in frame.index:
            assert naive_cost.loc[d] == pytest.approx(
                config.COST_HAIRCUT_BASE * strike.loc[d]
            )

    # conditioned: gate exit at row 3 (1 -> 0) is a close and must be charged.
    cond_cost = frame["gross_conditioned"] - frame["net_conditioned_base"]
    close_date = positions.index[3]
    if close_date in frame.index:
        assert cond_cost.loc[close_date] == pytest.approx(
            config.COST_HAIRCUT_BASE * strike.loc[close_date]
        )


def test_stress_net_is_base_minus_extra_haircut_on_traded_months():
    returns, vix_monthly, positions = _make_pipeline(seed=5)
    frame = build_pnl_frame(positions, vix_monthly, returns)
    for strat in ("naive", "conditioned"):
        base_cost = frame[f"gross_{strat}"] - frame[f"net_{strat}_base"]
        extra = frame[f"net_{strat}_base"] - frame[f"net_{strat}_stress"]
        # stress net = base net - (stress_multiplier - 1) * base cost.
        expected_extra = (config.COST_STRESS_MULTIPLIER - 1) * base_cost
        pd.testing.assert_series_equal(
            extra, expected_extra, check_names=False
        )


def test_nan_tail_dropped_and_counted():
    returns, vix_monthly, positions = _make_pipeline(seed=6)
    frame = build_pnl_frame(positions, vix_monthly, returns)

    realised = (
        resample_to_month_start(realised_variance_target(returns))
        .reindex(positions.index)
    )
    expected_dropped = int(realised.isna().sum())
    assert expected_dropped > 0
    assert frame.attrs["n_nan_dropped"] == expected_dropped
    assert len(frame) == len(positions) - expected_dropped
    assert not frame.isna().any().any()


def test_raises_on_vix_nan_and_missing_column():
    returns, vix_monthly, positions = _make_pipeline(seed=7)

    bad_vix = vix_monthly.copy()
    bad_vix.loc[positions.index[3]] = np.nan
    with pytest.raises(ValueError):
        build_pnl_frame(positions, bad_vix, returns)

    with pytest.raises(ValueError):
        build_pnl_frame(positions.drop(columns=["position_naive"]), vix_monthly, returns)
