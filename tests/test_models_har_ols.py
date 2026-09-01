import math

import numpy as np
import pandas as pd
from src.validation import VRP_HORIZON_COLS
import pytest
import statsmodels.api as sm

from src import config
from src.models.har_ols import HAROLSModel


def _make_monthly_vrp(n: int = 60) -> pd.DataFrame:
    """Minimal monthly_vrp with all locked features present."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    return pd.DataFrame(
        {
            "vix_level": rng.uniform(10, 30, n),
            "cboe_skew": rng.uniform(100, 150, n),
            **{_c: rng.normal(3, 2, n) for _c in VRP_HORIZON_COLS},
            "realised_skew_21d": rng.normal(0, 0.5, n),
            "regime": rng.choice(["calm", "normal", "stressed"], n),
        },
        index=dates,
    )


def _make_har_arrays(n: int = 60):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    X = pd.DataFrame(
        {
            **{_c: rng.normal(3, 2, n) for _c in VRP_HORIZON_COLS},
        },
        index=dates,
    )
    y = pd.Series(rng.normal(3, 2, n), index=dates, name="y")
    return X, y


def test_fit_stores_params_and_bse():
    model = HAROLSModel(_make_monthly_vrp())
    X, y = _make_har_arrays()
    model.fit(X, y)

    assert model.params_ is not None
    assert model.bse_ is not None
    # const + 3 lag coefficients
    assert len(model.params_) == 4
    assert len(model.bse_) == 4
    assert "const" in model.params_.index


def test_predict_before_fit_raises():
    model = HAROLSModel(_make_monthly_vrp())
    X_test = pd.DataFrame({_c: [1.0] for _c in VRP_HORIZON_COLS})
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        model.predict(X_test)


def test_predict_shape():
    model = HAROLSModel(_make_monthly_vrp())
    X, y = _make_har_arrays()
    model.fit(X, y)
    preds = model.predict(X.iloc[:5])
    assert preds.shape == (5,)


def test_predict_matches_manual_dot_product():
    model = HAROLSModel(_make_monthly_vrp())
    X, y = _make_har_arrays()
    model.fit(X, y)

    X_test = X.iloc[:4]
    preds = model.predict(X_test)

    X_aug = sm.add_constant(X_test, has_constant="add")
    expected = X_aug[model.params_.index].values @ model.params_.values
    np.testing.assert_allclose(preds, expected, rtol=1e-10)


def test_hac_lag_formula_at_t100():
    """T=100 -> floor(4 * 1^(2/9)) = 4."""
    result = math.floor(config.NW_HAC_MULTIPLIER * (100 / 100) ** config.NW_HAC_EXPONENT)
    assert result == 4


def test_hac_lag_increases_with_t():
    """Lag should be non-decreasing as T grows."""
    lags = [
        math.floor(config.NW_HAC_MULTIPLIER * (T / 100) ** config.NW_HAC_EXPONENT)
        for T in [100, 200, 500, 1000]
    ]
    assert lags == sorted(lags)


def test_assert_feature_set_called_on_monthly_vrp():
    """fit() raises if monthly_vrp is missing a locked feature."""
    bad_vrp = pd.DataFrame({"vrp_lag_1": [1.0]})  # missing most locked features
    model = HAROLSModel(bad_vrp)
    X, y = _make_har_arrays()
    with pytest.raises(ValueError, match="Missing locked features"):
        model.fit(X, y)


def test_assert_feature_set_not_checked_on_x_train():
    """fit() should not raise even though X_train lacks vix_level/cboe_skew/regime."""
    monthly_vrp = _make_monthly_vrp()  # has all locked features
    X, y = _make_har_arrays()         # X has only the three HAR lags
    model = HAROLSModel(monthly_vrp)
    model.fit(X, y)  # should not raise


def test_fresh_instance_per_fit():
    """Two separate fit calls on different data produce independent params."""
    monthly_vrp = _make_monthly_vrp()
    X, y = _make_har_arrays()

    model_a = HAROLSModel(monthly_vrp)
    model_b = HAROLSModel(monthly_vrp)
    model_a.fit(X.iloc[:30], y.iloc[:30])
    model_b.fit(X.iloc[:60], y.iloc[:60])

    # Params should differ because training windows differ
    assert not np.allclose(model_a.params_.values, model_b.params_.values)
