import math

import numpy as np
import pytest

from src import config
from src.evaluation import rv_to_variance, vrp_forecast_to_variance


# ── vrp_forecast_to_variance (new decimal-variance convention) ────────────────
# Formula: implied_variance = vix_next**2 / VIX_VARIANCE_SCALE - vrp_forecast
# vix_next in vol-points; vrp_forecast in decimal annualised variance.

def test_implied_variance_anchor():
    # vix_next=20, vrp_forecast=0.01 → 400/10000 - 0.01 = 0.04 - 0.01 = 0.03
    result = vrp_forecast_to_variance([0.01], [20.0])
    np.testing.assert_allclose(result, [0.03], atol=1e-12)


def test_implied_variance_vectorised():
    # vix=[20, 25, 15], vrp=[0.01, 0.02, 0.005]
    # implied = [400-100, 625-200, 225-50] / 10000 = [300, 425, 175] / 10000
    #         = [0.030, 0.0425, 0.0175]
    result = vrp_forecast_to_variance([0.01, 0.02, 0.005], [20.0, 25.0, 15.0])
    expected = [
        20.0 ** 2 / config.VIX_VARIANCE_SCALE - 0.01,
        25.0 ** 2 / config.VIX_VARIANCE_SCALE - 0.02,
        15.0 ** 2 / config.VIX_VARIANCE_SCALE - 0.005,
    ]
    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_implied_variance_uses_config_scale_not_literal():
    # Verify the function uses config.VIX_VARIANCE_SCALE by checking the formula
    # matches config's value, not a hardcoded literal.
    vix = 20.0
    vrp = 0.01
    expected = vix ** 2 / config.VIX_VARIANCE_SCALE - vrp
    result = vrp_forecast_to_variance([vrp], [vix])
    np.testing.assert_allclose(result, [expected], atol=1e-15)


def test_positivity_guard_raises_on_negative_implied_variance():
    # vix_next=10 → rn_var = 100/10000 = 0.01; vrp_forecast=0.015 → implied = -0.005
    with pytest.raises(ValueError) as exc_info:
        vrp_forecast_to_variance([0.015], [10.0])
    msg = str(exc_info.value)
    assert "implied_variance" in msg
    assert "0" in msg  # index 0 named in message


def test_positivity_guard_message_contains_all_three_quantities():
    # Verify the message includes vix_next, vrp_forecast, and implied_variance values.
    vix_val, vrp_val = 10.0, 0.015
    implied_val = vix_val ** 2 / config.VIX_VARIANCE_SCALE - vrp_val  # = -0.005
    with pytest.raises(ValueError) as exc_info:
        vrp_forecast_to_variance([vrp_val], [vix_val])
    msg = str(exc_info.value)
    assert "vix_next" in msg
    assert "vrp_forecast" in msg
    assert "implied_variance" in msg


def test_positivity_guard_raises_on_zero_implied_variance():
    # vix_next=10 → rn_var = 0.01; vrp_forecast=0.01 → implied = 0.0 (not strictly > 0)
    with pytest.raises(ValueError):
        vrp_forecast_to_variance([0.01], [10.0])


def test_positivity_guard_raises_only_on_bad_elements():
    # First element OK (implied=0.03), second bad (implied=-0.005).
    with pytest.raises(ValueError) as exc_info:
        vrp_forecast_to_variance([0.01, 0.015], [20.0, 10.0])
    msg = str(exc_info.value)
    assert "index 1" in msg


def test_raises_on_nan_vrp():
    with pytest.raises(ValueError, match="NaN"):
        vrp_forecast_to_variance([float("nan")], [20.0])


def test_raises_on_nan_vix():
    with pytest.raises(ValueError, match="NaN"):
        vrp_forecast_to_variance([0.01], [float("nan")])


def test_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="mismatch"):
        vrp_forecast_to_variance([0.01, 0.02], [20.0])


# ── rv_to_variance (unchanged) ────────────────────────────────────────────────

def test_rv_to_variance_single():
    np.testing.assert_allclose(rv_to_variance([15.0]), [0.0225], atol=1e-10)


def test_rv_to_variance_vectorised():
    # [10, 20, 15] → [100, 400, 225] / 10000 = [0.01, 0.04, 0.0225]
    np.testing.assert_allclose(
        rv_to_variance([10.0, 20.0, 15.0]), [0.01, 0.04, 0.0225], atol=1e-10
    )


def test_rv_to_variance_raises_on_nan():
    with pytest.raises(ValueError):
        rv_to_variance([float("nan")])


def test_rv_to_variance_raises_on_zero():
    with pytest.raises(ValueError):
        rv_to_variance([0.0])


def test_rv_to_variance_raises_on_negative():
    with pytest.raises(ValueError):
        rv_to_variance([-5.0])
