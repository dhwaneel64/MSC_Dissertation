"""Leakage and correctness tests for the Circle 3A position series.

The position at t must be a strict function of information with index <= t: the
forecast made at t and the regime label at t. The strong-form input-truncation
test (truncate all inputs at t, position at t byte-identical to the full run) and
its complement (truncate before t, position at t drops) are the mandatory
precondition before any P&L is built on top of these positions.
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.circle3a.positions import build_position_series


def _make_inputs(n: int = 40, seed: int = 0):
    """Synthetic forecast and regime series on a monthly first-of-month index.

    The forecast is a noisy series so the expanding median moves over time; the
    regime cycles through the three labels so both gate branches are exercised.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-01", periods=n, freq="MS")
    forecast = pd.Series(rng.normal(0.01, 0.02, n), index=idx, name="y_pred")
    labels = np.array(config.REGIME_LABELS)
    regime = pd.Series(labels[rng.integers(0, len(labels), n)], index=idx, name="regime")
    return forecast, regime


def test_returns_expected_columns_and_index():
    forecast, regime = _make_inputs()
    result = build_position_series(forecast, regime)
    assert list(result.columns) == [
        "forecast",
        "expanding_median",
        "regime",
        "position_naive",
        "position_conditioned",
    ]
    assert result.index.equals(forecast.index)


def test_expanding_median_includes_t_first_row_flat():
    """The expanding median at t includes t, so the first row can never be short
    (forecast(0) > median(forecast(0)) is False)."""
    forecast, regime = _make_inputs()
    result = build_position_series(forecast, regime)
    assert result["expanding_median"].iloc[0] == forecast.iloc[0]
    assert result["position_naive"].iloc[0] == 0
    # expanding median equals the causal running median at every point.
    expected = forecast.expanding().median()
    pd.testing.assert_series_equal(
        result["expanding_median"], expected, check_names=False
    )


def test_strong_form_input_truncation():
    """Truncate ALL inputs at t: both position series at t are byte-identical to
    the full-run values. Any peek at forecasts after t would move the expanding
    median at t and break this."""
    forecast, regime = _make_inputs(seed=3)
    full = build_position_series(forecast, regime)

    for t in forecast.index[[1, 7, 20, len(forecast) - 1]]:
        truncated = build_position_series(forecast.loc[:t], regime.loc[:t])
        assert t in truncated.index
        for col in ("position_naive", "position_conditioned", "expanding_median"):
            assert truncated.at[t, col] == full.at[t, col], (
                f"Leakage at {t.date()} in {col}: "
                f"full={full.at[t, col]}, truncated={truncated.at[t, col]}"
            )


def test_complement_truncation_before_t_drops_position():
    """Truncating inputs strictly before t drops the position at t: t is not in
    the result built from data up to t-1."""
    forecast, regime = _make_inputs(seed=4)
    t = forecast.index[15]
    t_prev = forecast.index[14]
    truncated = build_position_series(forecast.loc[:t_prev], regime.loc[:t_prev])
    assert t not in truncated.index
    assert t_prev in truncated.index


def test_gate_reads_regime_at_t_no_shift():
    """Flipping the regime at t changes position_conditioned only at t, never at
    t-1 or t+1. This pins the gate to the contemporaneous regime (no shift, no
    t+1 read)."""
    forecast, regime = _make_inputs(seed=5)
    # Force a short signal everywhere so the gate is the only driver of the
    # conditioned column: a strictly increasing forecast is above its expanding
    # median at every row after the first.
    forecast = pd.Series(
        np.arange(1, len(forecast) + 1, dtype=float),
        index=forecast.index,
        name="y_pred",
    )
    regime = pd.Series(config.REGIME_LABELS[0], index=forecast.index, name="regime")

    base = build_position_series(forecast, regime)
    t = forecast.index[20]
    flipped = regime.copy()
    flipped.loc[t] = config.REGIME_STRESSED_LABEL
    after = build_position_series(forecast, flipped)

    changed = after["position_conditioned"] != base["position_conditioned"]
    assert changed.loc[t]
    assert changed.sum() == 1, "regime flip at t altered a position at another date"


def test_conditioned_short_count_le_naive():
    forecast, regime = _make_inputs(seed=6)
    result = build_position_series(forecast, regime)
    assert result["position_conditioned"].sum() <= result["position_naive"].sum()


def test_stressed_months_are_flat_in_conditioned():
    forecast, regime = _make_inputs(seed=7)
    result = build_position_series(forecast, regime)
    stressed = result["regime"].astype(str) == config.REGIME_STRESSED_LABEL
    assert (result.loc[stressed, "position_conditioned"] == 0).all()


def test_positions_are_binary():
    forecast, regime = _make_inputs(seed=8)
    result = build_position_series(forecast, regime)
    for col in ("position_naive", "position_conditioned"):
        assert set(result[col].unique()).issubset({0, 1})


def test_disjoint_information_future_forecast_change_does_not_move_earlier_positions():
    """A change to the forecast at a later date leaves every earlier position
    untouched: positions depend only on data with index <= t."""
    forecast, regime = _make_inputs(seed=9)
    base = build_position_series(forecast, regime)
    t_future = forecast.index[30]
    perturbed = forecast.copy()
    perturbed.loc[t_future] = perturbed.loc[t_future] + 10.0
    after = build_position_series(perturbed, regime)

    earlier = forecast.index[forecast.index < t_future]
    pd.testing.assert_frame_equal(base.loc[earlier], after.loc[earlier])


def test_raises_on_nan_and_misaligned_inputs():
    forecast, regime = _make_inputs(seed=10)

    bad_forecast = forecast.copy()
    bad_forecast.iloc[3] = np.nan
    with pytest.raises(ValueError):
        build_position_series(bad_forecast, regime)

    bad_regime = regime.copy()
    bad_regime.iloc[3] = np.nan
    with pytest.raises(ValueError):
        build_position_series(forecast, bad_regime)

    with pytest.raises(ValueError):
        build_position_series(forecast, regime.iloc[:-1])
