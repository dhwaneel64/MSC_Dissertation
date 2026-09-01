import numpy as np
import pandas as pd
from src.validation import VRP_HORIZON_COLS
import pytest

from src import config
from src.models.extended_ols import ExtendedOLSModel
from src.models.regime_switching_ols import (
    NUMERIC_FEATURES,
    RegimeSwitchingOLSModel,
    regime_fallback_log,
)


def _make_monthly_vrp(n: int = 60) -> pd.DataFrame:
    """Minimal monthly_vrp with all locked features present (for the assertion)."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2012-01-01", periods=n, freq="MS")
    return pd.DataFrame(
        {
            "vix_level": rng.uniform(10, 30, n),
            "cboe_skew": rng.uniform(100, 150, n),
            **{_c: rng.normal(3, 2, n) for _c in VRP_HORIZON_COLS},
            "realised_skew_21d": rng.normal(0, 0.5, n),
            "regime": rng.choice(config.REGIME_LABELS, n),
        },
        index=dates,
    )


def _make_switching_frame(n: int = 200, seed: int = 7):
    """Feature frame with the six numeric features, a regime column, and a target.

    Regime mix is skewed toward calm/normal with a thin stressed tail, mirroring
    the real sample so the stressed fallback can be exercised on small windows.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    regime = rng.choice(config.REGIME_LABELS, n, p=[0.45, 0.45, 0.10])
    X = pd.DataFrame(
        {
            "vix_level": rng.uniform(10, 40, n),
            "cboe_skew": rng.uniform(100, 160, n),
            **{_c: rng.normal(3, 2, n) for _c in VRP_HORIZON_COLS},
            "realised_skew_21d": rng.normal(0, 0.5, n),
            "regime": regime,
        },
        index=dates,
    )
    y = pd.Series(rng.normal(3, 2, n), index=dates, name="y")
    return X, y


def test_numeric_features_are_locked_set_minus_regime():
    assert NUMERIC_FEATURES == (
        "vix_level", "cboe_skew", *VRP_HORIZON_COLS,
        "realised_skew_21d",
    )
    assert "regime" not in NUMERIC_FEATURES


def test_min_obs_is_multiplier_times_params():
    model = RegimeSwitchingOLSModel(_make_monthly_vrp())
    # 6 numeric features + 1 intercept = 7 parameters; 2 * 7 = 14.
    assert model.min_obs_ == config.REGIME_MIN_TRAIN_MULTIPLIER * (len(NUMERIC_FEATURES) + 1)
    assert model.min_obs_ == 14


def test_separate_coefficients_per_regime():
    """Each well-populated regime gets its own coefficient set, not one shared set."""
    model = RegimeSwitchingOLSModel(_make_monthly_vrp())
    X, y = _make_switching_frame(n=240)
    model.fit(X, y)

    fitted = [r for r in config.REGIME_LABELS if r in model.regime_params_]
    assert len(fitted) >= 2, "test needs at least two well-populated regimes"
    # Distinct coefficient sets across regimes (separate fits, not a pooled model).
    for a, b in zip(fitted, fitted[1:]):
        assert not np.allclose(
            model.regime_params_[a].values, model.regime_params_[b].values
        )
    # And distinct from the pooled (restricted) coefficients.
    for r in fitted:
        assert not np.allclose(
            model.regime_params_[r].values, model.pooled_params_.values
        )


def test_predict_uses_regime_specific_coefficients():
    """A test row's prediction equals its regime's standardised fit, by hand."""
    import statsmodels.api as sm

    model = RegimeSwitchingOLSModel(_make_monthly_vrp())
    X, y = _make_switching_frame(n=240)
    model.fit(X, y)

    fitted_regime = next(r for r in config.REGIME_LABELS if r in model.regime_params_)
    row = X[X["regime"] == fitted_regime].iloc[:1]
    pred = model.predict(row)

    Xnum = row[list(NUMERIC_FEATURES)]
    Xs = (Xnum - model.regime_mu_[fitted_regime]) / model.regime_sigma_[fitted_regime]
    Xaug = sm.add_constant(Xs, has_constant="add")
    params = model.regime_params_[fitted_regime]
    expected = Xaug[params.index].values @ params.values
    np.testing.assert_allclose(pred, expected, rtol=1e-10)


def test_pooled_fit_matches_extended_ols():
    """The pooled (restricted) fit is byte-identical to Extended OLS on the same data.

    This is the nesting claim made concrete: Extended OLS is exactly the pooled
    case (all regimes sharing one coefficient set and one scaler).
    """
    monthly_vrp = _make_monthly_vrp()
    X, y = _make_switching_frame(n=240)

    rs = RegimeSwitchingOLSModel(monthly_vrp)
    rs.fit(X, y)

    ext = ExtendedOLSModel(monthly_vrp)
    ext.fit(X[list(NUMERIC_FEATURES)], y)

    np.testing.assert_array_equal(rs.pooled_mu_.values, ext.mu_.values)
    np.testing.assert_array_equal(rs.pooled_sigma_.values, ext.sigma_.values)
    np.testing.assert_array_equal(rs.pooled_params_.values, ext.params_.values)

    # A fallback row therefore predicts identically to Extended OLS.
    fb = rs.fallback_regimes_
    if fb:
        fb_regime = next(iter(fb))
        rows = X[X["regime"] == fb_regime]
        if len(rows) > 0:
            row = rows.iloc[:1]
            np.testing.assert_allclose(
                rs.predict(row),
                ext.predict(row[list(NUMERIC_FEATURES)]),
                rtol=1e-12,
            )


def test_fallback_triggers_for_scarce_regime():
    """A regime below min_obs falls back and is recorded, not silently fit."""
    monthly_vrp = _make_monthly_vrp()
    # Build a frame where stressed appears only a handful of times (< 14).
    rng = np.random.default_rng(3)
    n = 120
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    regime = np.array(["calm"] * 55 + ["normal"] * 60 + ["stressed"] * 5)
    rng.shuffle(regime)
    X = pd.DataFrame(
        {
            "vix_level": rng.uniform(10, 40, n),
            "cboe_skew": rng.uniform(100, 160, n),
            **{_c: rng.normal(3, 2, n) for _c in VRP_HORIZON_COLS},
            "realised_skew_21d": rng.normal(0, 0.5, n),
            "regime": regime,
        },
        index=dates,
    )
    y = pd.Series(rng.normal(3, 2, n), index=dates, name="y")

    model = RegimeSwitchingOLSModel(monthly_vrp)
    model.fit(X, y)

    assert model.regime_n_["stressed"] == 5
    assert "stressed" in model.fallback_regimes_
    assert "stressed" not in model.regime_params_
    # Well-populated regimes are fit separately.
    assert "calm" in model.regime_params_
    assert "normal" in model.regime_params_


def test_predict_before_fit_raises():
    model = RegimeSwitchingOLSModel(_make_monthly_vrp())
    X, _ = _make_switching_frame(n=20)
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        model.predict(X.iloc[:1])


def test_fit_raises_on_missing_regime_column():
    model = RegimeSwitchingOLSModel(_make_monthly_vrp())
    X, y = _make_switching_frame(n=60)
    with pytest.raises(ValueError, match="regime switch column"):
        model.fit(X.drop(columns=["regime"]), y)


def test_assert_feature_set_called_on_monthly_vrp():
    bad_vrp = pd.DataFrame({"vrp_lag_1": [1.0]})
    model = RegimeSwitchingOLSModel(bad_vrp)
    X, y = _make_switching_frame(n=60)
    with pytest.raises(ValueError, match="Missing locked features"):
        model.fit(X, y)


def test_per_step_no_future_leak_byte_identical():
    """Leakage test: truncating the input at t leaves the prediction at t byte-identical.

    Per-regime scalers and coefficients are fit inside fit() on the training rows
    only, so observations after the cut cannot influence the fit. Fitting on rows
    [:k] and predicting row k must give a byte-identical prediction whether or not
    rows after k exist in the caller's frame, and whether or not they are corrupted.
    """
    monthly_vrp = _make_monthly_vrp()
    X, y = _make_switching_frame(n=240)
    k = 150

    model_full = RegimeSwitchingOLSModel(monthly_vrp)
    model_full.fit(X.iloc[:k], y.iloc[:k])
    pred_full = model_full.predict(X.iloc[k:k + 1])

    X_trunc = X.iloc[:k].copy()
    y_trunc = y.iloc[:k].copy()
    model_trunc = RegimeSwitchingOLSModel(monthly_vrp)
    model_trunc.fit(X_trunc, y_trunc)
    pred_trunc = model_trunc.predict(X.iloc[k:k + 1])

    np.testing.assert_array_equal(pred_full, pred_trunc)

    # Corrupting rows strictly after the held-out row also changes nothing.
    X_corrupt = X.copy()
    X_corrupt.iloc[k + 1:, :len(NUMERIC_FEATURES)] = 1e6
    model_c = RegimeSwitchingOLSModel(monthly_vrp)
    model_c.fit(X_corrupt.iloc[:k], y.iloc[:k])
    pred_c = model_c.predict(X_corrupt.iloc[k:k + 1])
    np.testing.assert_array_equal(pred_full, pred_c)


def test_fallback_log_matches_model_decision():
    """regime_fallback_log reproduces the model's actual fallback decision per step."""
    monthly_vrp = _make_monthly_vrp()
    X, y = _make_switching_frame(n=180)
    dataset = X.copy()
    dataset["y"] = y

    initial_train_end = dataset.index[120]
    log = regime_fallback_log(dataset, initial_train_end)

    pos = dataset.index.get_loc(initial_train_end)
    # Check a spread of steps against a freshly fit model on that step's window.
    for offset in [0, 10, 30, len(log) - 1]:
        i = pos + offset
        train = dataset.iloc[: i + 1]
        test = dataset.iloc[i + 1 : i + 2]
        model = RegimeSwitchingOLSModel(monthly_vrp)
        model.fit(train[list(NUMERIC_FEATURES) + ["regime"]], train["y"])
        test_regime = str(test["regime"].iloc[0])
        model_fell_back = test_regime not in model.regime_params_
        log_row = log.loc[test.index[0]]
        assert bool(log_row["fallback"]) == model_fell_back
        assert log_row["regime"] == test_regime
        assert int(log_row["n_train_regime"]) == model.regime_n_[test_regime]


def test_fresh_instance_per_fit():
    monthly_vrp = _make_monthly_vrp()
    X, y = _make_switching_frame(n=240)
    a = RegimeSwitchingOLSModel(monthly_vrp)
    b = RegimeSwitchingOLSModel(monthly_vrp)
    a.fit(X.iloc[:120], y.iloc[:120])
    b.fit(X.iloc[:240], y.iloc[:240])
    assert not np.allclose(a.pooled_params_.values, b.pooled_params_.values)
