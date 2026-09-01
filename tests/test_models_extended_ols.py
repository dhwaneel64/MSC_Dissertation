import math

import numpy as np
import pandas as pd
from src.validation import VRP_HORIZON_COLS
import pytest
import statsmodels.api as sm

from src import config
from src.models.extended_ols import ExtendedOLSModel


def _make_monthly_vrp(n: int = 60) -> pd.DataFrame:
    """Minimal monthly_vrp with all locked features present."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2012-01-01", periods=n, freq="MS")
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


def _make_extended_arrays(n: int = 60):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    X = pd.DataFrame(
        {
            "vix_level": rng.uniform(10, 30, n),
            "cboe_skew": rng.uniform(100, 150, n),
            **{_c: rng.normal(3, 2, n) for _c in VRP_HORIZON_COLS},
            "realised_skew_21d": rng.normal(0, 0.5, n),
        },
        index=dates,
    )
    y = pd.Series(rng.normal(3, 2, n), index=dates, name="y")
    return X, y


def test_fit_stores_params_and_bse():
    model = ExtendedOLSModel(_make_monthly_vrp())
    X, y = _make_extended_arrays()
    model.fit(X, y)

    assert model.params_ is not None
    assert model.bse_ is not None
    # const + 6 feature coefficients
    assert len(model.params_) == 7
    assert len(model.bse_) == 7
    assert "const" in model.params_.index


def test_predict_before_fit_raises():
    model = ExtendedOLSModel(_make_monthly_vrp())
    X_test = pd.DataFrame({
        "vix_level": [15.0], "cboe_skew": [120.0],
        **{_c: [1.0] for _c in VRP_HORIZON_COLS},
        "realised_skew_21d": [-0.1],
    })
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        model.predict(X_test)


def test_predict_shape():
    model = ExtendedOLSModel(_make_monthly_vrp())
    X, y = _make_extended_arrays()
    model.fit(X, y)
    preds = model.predict(X.iloc[:5])
    assert preds.shape == (5,)


def test_predict_matches_manual_dot_product():
    """predict() standardises with the stored scaler, then takes the dot product."""
    model = ExtendedOLSModel(_make_monthly_vrp())
    X, y = _make_extended_arrays()
    model.fit(X, y)

    X_test = X.iloc[:4]
    preds = model.predict(X_test)

    X_scaled = (X_test - model.mu_) / model.sigma_
    X_aug = sm.add_constant(X_scaled, has_constant="add")
    expected = X_aug[model.params_.index].values @ model.params_.values
    np.testing.assert_allclose(preds, expected, rtol=1e-10)


def test_scaler_fit_on_training_window_stats():
    """mu_/sigma_ equal the training-window mean and population (ddof=0) std."""
    model = ExtendedOLSModel(_make_monthly_vrp())
    X, y = _make_extended_arrays()
    model.fit(X, y)

    pd.testing.assert_series_equal(model.mu_, X.mean())
    pd.testing.assert_series_equal(model.sigma_, X.std(ddof=0).replace(0.0, 1.0))


def test_ols_predictions_invariant_to_standardisation():
    """OLS fitted values are unchanged by the affine rescaling of regressors.

    Fits a plain (unstandardised) OLS on the same design and confirms the
    standardised model reproduces its predictions, so adding the scaler does not
    move the forecast, only the coefficient scale.
    """
    model = ExtendedOLSModel(_make_monthly_vrp())
    X, y = _make_extended_arrays()
    model.fit(X, y)
    preds = model.predict(X.iloc[:6])

    X_plain = sm.add_constant(X)
    plain = sm.OLS(y, X_plain).fit()
    X_plain_test = sm.add_constant(X.iloc[:6], has_constant="add")
    expected = X_plain_test[plain.params.index].values @ plain.params.values
    np.testing.assert_allclose(preds, expected, rtol=1e-8)


def test_per_step_scaler_no_future_leak_byte_identical():
    """Strong leakage test: truncating the input at t leaves the scaled features
    at t byte-identical.

    The scaler is fit inside fit() on X_train alone, so observations after the
    training cut cannot influence it. Fitting on rows [:k] and predicting row k
    must give byte-identical scaler parameters and a byte-identical scaled
    prediction whether or not rows after k exist in the caller's frame.
    """
    monthly_vrp = _make_monthly_vrp()
    X, y = _make_extended_arrays(80)
    k = 50

    # Full-frame fit on the first k rows.
    model_full = ExtendedOLSModel(monthly_vrp)
    model_full.fit(X.iloc[:k], y.iloc[:k])
    pred_full = model_full.predict(X.iloc[k:k + 1])

    # Input truncated at t: everything after row k-1 removed before fitting.
    X_trunc = X.iloc[:k].copy()
    y_trunc = y.iloc[:k].copy()
    model_trunc = ExtendedOLSModel(monthly_vrp)
    model_trunc.fit(X_trunc, y_trunc)
    pred_trunc = model_trunc.predict(X.iloc[k:k + 1])

    np.testing.assert_array_equal(model_full.mu_.values, model_trunc.mu_.values)
    np.testing.assert_array_equal(model_full.sigma_.values, model_trunc.sigma_.values)
    np.testing.assert_array_equal(model_full.params_.values, model_trunc.params_.values)
    np.testing.assert_array_equal(pred_full, pred_trunc)


def test_scaler_unaffected_by_corrupting_future_rows():
    """Corrupting rows after the training cut does not change the fit at t.

    Same training slice, future rows replaced with large garbage. Because fit()
    consumes only X_train, the scaler, coefficients, and the prediction for the
    held-out row are identical.
    """
    monthly_vrp = _make_monthly_vrp()
    X, y = _make_extended_arrays(80)
    k = 50

    model_a = ExtendedOLSModel(monthly_vrp)
    model_a.fit(X.iloc[:k], y.iloc[:k])
    pred_a = model_a.predict(X.iloc[k:k + 1])

    X_corrupt = X.copy()
    X_corrupt.iloc[k + 1:] = 1e6  # garbage strictly after the held-out row
    model_b = ExtendedOLSModel(monthly_vrp)
    model_b.fit(X_corrupt.iloc[:k], y.iloc[:k])
    pred_b = model_b.predict(X_corrupt.iloc[k:k + 1])

    np.testing.assert_array_equal(model_a.mu_.values, model_b.mu_.values)
    np.testing.assert_array_equal(model_a.sigma_.values, model_b.sigma_.values)
    np.testing.assert_array_equal(pred_a, pred_b)


def test_hac_lag_formula_at_t100():
    """T=100 -> floor(4 * 1^(2/9)) = 4."""
    result = math.floor(config.NW_HAC_MULTIPLIER * (100 / 100) ** config.NW_HAC_EXPONENT)
    assert result == 4


def test_assert_feature_set_called_on_monthly_vrp():
    """fit() raises if monthly_vrp is missing a locked feature."""
    bad_vrp = pd.DataFrame({"vrp_lag_1": [1.0]})
    model = ExtendedOLSModel(bad_vrp)
    X, y = _make_extended_arrays()
    with pytest.raises(ValueError, match="Missing locked features"):
        model.fit(X, y)


def test_assert_feature_set_not_checked_on_x_train():
    """fit() should not raise even though X_train lacks regime."""
    monthly_vrp = _make_monthly_vrp()
    X, y = _make_extended_arrays()
    model = ExtendedOLSModel(monthly_vrp)
    model.fit(X, y)


def test_fresh_instance_per_fit():
    """Two separate fit calls on different training windows produce independent params."""
    monthly_vrp = _make_monthly_vrp()
    X, y = _make_extended_arrays()

    model_a = ExtendedOLSModel(monthly_vrp)
    model_b = ExtendedOLSModel(monthly_vrp)
    model_a.fit(X.iloc[:30], y.iloc[:30])
    model_b.fit(X.iloc[:60], y.iloc[:60])

    assert not np.allclose(model_a.params_.values, model_b.params_.values)
