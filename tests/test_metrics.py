import math

import numpy as np
import pytest

from src.metrics import (
    directional_accuracy,
    mse,
    mse_per_obs,
    qlike,
    qlike_per_obs,
)


# ── qlike ────────────────────────────────────────────────────────────────────

def test_qlike_perfect_forecast():
    assert qlike([0.04, 0.09], [0.04, 0.09]) < 1e-12


def test_qlike_known_closed_form():
    # y_true=[1.0, 4.0], y_pred=[1.0, 1.0]
    # obs 0: 1/1 - log(1) - 1 = 0
    # obs 1: 4/1 - log(4) - 1 = 3 - log(4) ≈ 1.6137056
    # mean = 0.8068528
    result = qlike([1.0, 4.0], [1.0, 1.0])
    assert math.isclose(result, 0.8068528, abs_tol=1e-4)


def test_qlike_asymmetry_underprediction_worse():
    # y_true=2.0; underprediction by 2x vs overprediction by 2x
    under = qlike([2.0], [1.0])   # forecast too low
    over = qlike([2.0], [4.0])    # forecast too high
    assert under > over


def test_qlike_raises_on_zero_y_pred():
    with pytest.raises(ValueError):
        qlike([1.0], [0.0])


def test_qlike_raises_on_negative_y_true():
    with pytest.raises(ValueError):
        qlike([-0.5], [1.0])


def test_qlike_raises_on_nan_y_true():
    with pytest.raises(ValueError):
        qlike([float("nan"), 1.0], [1.0, 1.0])


def test_qlike_raises_on_nan_y_pred():
    with pytest.raises(ValueError):
        qlike([1.0, 1.0], [1.0, float("nan")])


def test_qlike_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        qlike([1.0, 2.0], [1.0])


# ── qlike_per_obs ─────────────────────────────────────────────────────────────

def test_qlike_per_obs_length():
    out = qlike_per_obs([1.0, 4.0, 2.0], [1.0, 2.0, 1.0])
    assert len(out) == 3


def test_qlike_per_obs_mean_equals_scalar():
    y_true = [1.0, 4.0, 2.0, 0.5]
    y_pred = [1.0, 2.0, 1.0, 0.25]
    np.testing.assert_allclose(
        qlike_per_obs(y_true, y_pred).mean(), qlike(y_true, y_pred), atol=1e-12
    )


def test_qlike_per_obs_raises_on_non_positive():
    with pytest.raises(ValueError):
        qlike_per_obs([1.0], [0.0])


def test_qlike_per_obs_raises_on_nan():
    with pytest.raises(ValueError):
        qlike_per_obs([float("nan")], [1.0])


def test_qlike_per_obs_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        qlike_per_obs([1.0, 2.0], [1.0])


# ── mse ─────────────────────────────────────────────────────────────────────

def test_mse_perfect_forecast():
    y = [1.0, 2.0, 3.0]
    assert mse(y, y) == 0.0


def test_mse_known_value():
    # y_true=[1,2,3], y_pred=[1,2,4]: errors=[0,0,1], mse=1/3
    result = mse([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert math.isclose(result, 1 / 3, abs_tol=1e-10)


def test_mse_raises_on_nan_y_true():
    with pytest.raises(ValueError):
        mse([float("nan"), 1.0], [1.0, 1.0])


def test_mse_raises_on_nan_y_pred():
    with pytest.raises(ValueError):
        mse([1.0, 1.0], [float("nan"), 1.0])


def test_mse_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        mse([1.0, 2.0], [1.0])


# ── mse_per_obs ───────────────────────────────────────────────────────────────

def test_mse_per_obs_length():
    out = mse_per_obs([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert len(out) == 3


def test_mse_per_obs_mean_equals_scalar():
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.5, 2.5, 2.5, 3.5]
    np.testing.assert_allclose(
        mse_per_obs(y_true, y_pred).mean(), mse(y_true, y_pred), atol=1e-12
    )


def test_mse_per_obs_known_values():
    # errors = [0, 0, 1], squared = [0, 0, 1]
    out = mse_per_obs([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    np.testing.assert_array_equal(out, [0.0, 0.0, 1.0])


def test_mse_per_obs_raises_on_nan():
    with pytest.raises(ValueError):
        mse_per_obs([float("nan"), 1.0], [1.0, 1.0])


def test_mse_per_obs_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        mse_per_obs([1.0, 2.0], [1.0])


# ── directional_accuracy ─────────────────────────────────────────────────────

def test_directional_accuracy_all_correct():
    assert directional_accuracy([1.0, -1.0, 2.0], [0.5, -3.0, 1.0]) == 1.0


def test_directional_accuracy_all_wrong():
    assert directional_accuracy([1.0, -1.0], [-1.0, 1.0]) == 0.0


def test_directional_accuracy_mixed():
    # [1,1] ✓  [-1,1] ✗  [1,1] ✓  [-1,-1] ✓  → 3/4 = 0.75
    result = directional_accuracy([1.0, -1.0, 1.0, -1.0], [1.0, 1.0, 1.0, -1.0])
    assert math.isclose(result, 0.75, abs_tol=1e-10)


def test_directional_accuracy_zero_in_y_true_matches_anything():
    assert directional_accuracy([0.0], [5.0]) == 1.0


def test_directional_accuracy_raises_on_nan():
    with pytest.raises(ValueError):
        directional_accuracy([float("nan")], [1.0])


def test_directional_accuracy_raises_on_length_mismatch():
    with pytest.raises(ValueError):
        directional_accuracy([1.0, 2.0], [1.0])
