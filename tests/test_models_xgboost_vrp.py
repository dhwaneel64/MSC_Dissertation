import numpy as np
import pandas as pd
from src.validation import VRP_HORIZON_COLS
import pytest

from src import config
from src.walk_forward import walk_forward, make_model_factory_from_class
from src.models.xgboost_vrp import (
    NUMERIC_FEATURES,
    XGB_FEATURE_ORDER,
    REGIME_TO_ORDINAL,
    XGBoostVRPModel,
    encode_features,
    default_grid,
    tune_xgboost_hyperparameters,
    walk_forward_with_shap,
    mean_abs_shap_by_regime,
    shap_rank_changes,
)


def _make_monthly_vrp(n: int = 60) -> pd.DataFrame:
    """Minimal monthly_vrp with all locked features present (for the assertion)."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2012-01-01", periods=n, freq="MS")
    return pd.DataFrame(
        {
            "vix_level": rng.uniform(10, 30, n),
            "cboe_skew": rng.uniform(100, 150, n),
            **{_c: rng.normal(0.01, 0.005, n) for _c in VRP_HORIZON_COLS},
            "realised_skew_21d": rng.normal(0, 0.5, n),
            "regime": rng.choice(config.REGIME_LABELS, n),
        },
        index=dates,
    )


def _make_feature_frame(n: int = 200, seed: int = 7):
    """Feature frame with the seven locked features and a small-VRP target."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    regime = rng.choice(config.REGIME_LABELS, n, p=[0.45, 0.45, 0.10])
    X = pd.DataFrame(
        {
            "vix_level": rng.uniform(12, 35, n),
            "cboe_skew": rng.uniform(100, 160, n),
            **{_c: rng.normal(0.01, 0.004, n) for _c in VRP_HORIZON_COLS},
            "realised_skew_21d": rng.normal(0, 0.5, n),
            "regime": regime,
        },
        index=dates,
    )
    y = pd.Series(rng.normal(0.01, 0.004, n), index=dates, name="y")
    return X, y


# ── Feature order / encoding ──────────────────────────────────────────────────

def test_feature_order_is_locked_set_with_regime_last():
    assert NUMERIC_FEATURES == (
        "vix_level", "cboe_skew", *VRP_HORIZON_COLS,
        "realised_skew_21d",
    )
    assert XGB_FEATURE_ORDER == NUMERIC_FEATURES + ("regime",)
    # All seven locked features are consumed (regime included), unlike the OLS models.
    from src.validation import LOCKED_FEATURE_SET
    assert set(XGB_FEATURE_ORDER) == set(LOCKED_FEATURE_SET)


def test_encode_regime_ordinal_and_order():
    assert REGIME_TO_ORDINAL == {"calm": 0, "normal": 1, "stressed": 2}
    X, _ = _make_feature_frame(n=10)
    enc = encode_features(X)
    assert list(enc.columns) == list(XGB_FEATURE_ORDER)
    # regime column is now integer codes matching the canonical order.
    expected = X["regime"].astype(str).map(REGIME_TO_ORDINAL).astype(int)
    np.testing.assert_array_equal(enc["regime"].to_numpy(), expected.to_numpy())
    # numeric columns are untouched.
    np.testing.assert_array_equal(enc["vix_level"].to_numpy(), X["vix_level"].to_numpy())


def test_encode_raises_on_unknown_regime():
    X, _ = _make_feature_frame(n=10)
    X = X.copy()
    X.iloc[0, X.columns.get_loc("regime")] = "panic"
    with pytest.raises(ValueError, match="Unknown regime label"):
        encode_features(X)


def test_default_grid_is_full_cartesian_product():
    grid = default_grid()
    assert len(grid) == (
        len(config.XGB_MAX_DEPTH_GRID)
        * len(config.XGB_LEARNING_RATE_GRID)
        * len(config.XGB_N_ESTIMATORS_GRID)
        * len(config.XGB_MIN_CHILD_WEIGHT_GRID)
    )
    keys = {"max_depth", "learning_rate", "n_estimators", "min_child_weight"}
    assert all(set(c) == keys for c in grid)


# ── Model fit / predict ───────────────────────────────────────────────────────

def test_predict_before_fit_raises():
    model = XGBoostVRPModel(_make_monthly_vrp(), default_grid()[0])
    X, _ = _make_feature_frame(n=10)
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        model.predict(X.iloc[:1])


def test_shap_before_fit_raises():
    model = XGBoostVRPModel(_make_monthly_vrp(), default_grid()[0])
    X, _ = _make_feature_frame(n=10)
    with pytest.raises(RuntimeError, match="fit\\(\\)"):
        model.shap_values(X.iloc[:1])


def test_assert_feature_set_called_on_monthly_vrp():
    bad_vrp = pd.DataFrame({"vrp_lag_1": [1.0]})
    model = XGBoostVRPModel(bad_vrp, default_grid()[0])
    X, y = _make_feature_frame(n=40)
    with pytest.raises(ValueError, match="Missing locked features"):
        model.fit(X[list(XGB_FEATURE_ORDER)], y)


def test_fit_predict_shapes():
    params = {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 50, "min_child_weight": 1}
    model = XGBoostVRPModel(_make_monthly_vrp(), params)
    X, y = _make_feature_frame(n=120)
    model.fit(X[list(XGB_FEATURE_ORDER)], y)
    pred = model.predict(X.iloc[100:105][list(XGB_FEATURE_ORDER)])
    assert pred.shape == (5,)
    assert np.isfinite(pred).all()


def test_shap_additivity_reproduces_prediction():
    """expected_value + row-sum of SHAP must reproduce predict() (exact tree explainer)."""
    params = {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 80, "min_child_weight": 1}
    model = XGBoostVRPModel(_make_monthly_vrp(), params)
    X, y = _make_feature_frame(n=140)
    model.fit(X[list(XGB_FEATURE_ORDER)], y)
    rows = X.iloc[120:130][list(XGB_FEATURE_ORDER)]
    sv = model.shap_values(rows)
    assert sv.shape == (10, len(XGB_FEATURE_ORDER))
    recon = model.expected_value_ + sv.sum(axis=1)
    # XGBoost stores base_score/leaf values in float32, so the exact tree-path
    # explainer reconstruction carries a ~1e-5 float32 offset, not a logic gap.
    np.testing.assert_allclose(recon, model.predict(rows), rtol=1e-2, atol=2e-5)


# ── Leakage: per-step refit, byte-identical prediction ────────────────────────

def test_per_step_no_future_leak_byte_identical():
    """Truncating or corrupting inputs after t leaves the prediction at t identical."""
    params = {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 100, "min_child_weight": 1}
    monthly_vrp = _make_monthly_vrp()
    X, y = _make_feature_frame(n=200)
    k = 150
    cols = list(XGB_FEATURE_ORDER)

    m_full = XGBoostVRPModel(monthly_vrp, params)
    m_full.fit(X.iloc[:k][cols], y.iloc[:k])
    pred_full = m_full.predict(X.iloc[k:k + 1][cols])

    # Truncated frame: identical fit, identical prediction.
    m_trunc = XGBoostVRPModel(monthly_vrp, params)
    m_trunc.fit(X.iloc[:k][cols].copy(), y.iloc[:k].copy())
    pred_trunc = m_trunc.predict(X.iloc[k:k + 1][cols])
    np.testing.assert_array_equal(pred_full, pred_trunc)

    # Corrupting rows strictly after the held-out row also changes nothing.
    X_corrupt = X.copy()
    X_corrupt.iloc[k + 1:, : len(NUMERIC_FEATURES)] = 1e6
    m_corrupt = XGBoostVRPModel(monthly_vrp, params)
    m_corrupt.fit(X_corrupt.iloc[:k][cols], y.iloc[:k])
    pred_corrupt = m_corrupt.predict(X_corrupt.iloc[k:k + 1][cols])
    np.testing.assert_array_equal(pred_full, pred_corrupt)


# ── Walk-forward with SHAP matches the shared engine ──────────────────────────

def test_walk_forward_with_shap_matches_engine():
    params = {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 60, "min_child_weight": 1}
    monthly_vrp = _make_monthly_vrp()
    X, y = _make_feature_frame(n=120)
    dataset = X.copy()
    dataset["y"] = y
    cols = list(XGB_FEATURE_ORDER)
    initial_train_end = dataset.index[90]

    wf_shap, shap_df = walk_forward_with_shap(
        dataset, cols, params, initial_train_end, monthly_vrp=monthly_vrp,
    )

    factory = make_model_factory_from_class(
        XGBoostVRPModel, monthly_vrp=monthly_vrp, hyperparams=params, feature_cols=cols,
    )
    wf_engine = walk_forward(dataset, cols, factory, initial_train_end, target_col="y")

    # Predictions byte-identical to the shared engine.
    assert wf_shap.index.equals(wf_engine.index)
    np.testing.assert_array_equal(wf_shap["y_pred"].to_numpy(), wf_engine["y_pred"].to_numpy())
    np.testing.assert_array_equal(wf_shap["y_true"].to_numpy(), wf_engine["y_true"].to_numpy())

    # SHAP frame aligns to OOS dates and feature columns.
    assert list(shap_df.columns) == cols
    assert shap_df.index.equals(wf_shap.index)
    assert shap_df.shape == (len(wf_shap), len(cols))
    assert np.isfinite(shap_df.to_numpy()).all()


# ── SHAP per-regime aggregation ───────────────────────────────────────────────

def test_mean_abs_shap_by_regime_matches_manual():
    dates = pd.date_range("2005-01-01", periods=6, freq="MS")
    shap_df = pd.DataFrame(
        {
            "vix_level": [1.0, -2.0, 3.0, -4.0, 0.5, -0.5],
            "regime": [0.1, -0.1, 0.2, -0.2, 0.3, -0.3],
        },
        index=dates,
    )
    regimes = pd.Series(
        ["calm", "calm", "stressed", "stressed", "normal", "normal"], index=dates
    )
    table = mean_abs_shap_by_regime(shap_df, regimes)
    # calm vix_level mean abs = mean(|1|, |2|) = 1.5
    assert table.loc["vix_level", "calm"] == pytest.approx(1.5)
    assert table.loc["vix_level", "stressed"] == pytest.approx(3.5)
    assert table.loc["vix_level", "normal"] == pytest.approx(0.5)
    assert table.loc["vix_level", "all"] == pytest.approx(np.mean([1, 2, 3, 4, 0.5, 0.5]))
    # Columns in canonical order, "all" last.
    assert list(table.columns) == ["calm", "normal", "stressed", "all"]


def test_shap_rank_changes_flags_reordering():
    table = pd.DataFrame(
        {
            "calm": [3.0, 1.0, 2.0],
            "normal": [3.0, 1.0, 2.0],
            "stressed": [1.0, 3.0, 2.0],  # feature A and B swap rank in stressed
        },
        index=["A", "B", "C"],
    )
    table.index.name = "feature"
    ranks = shap_rank_changes(table)
    assert bool(ranks.loc["A", "rank_changes"]) is True
    assert bool(ranks.loc["B", "rank_changes"]) is True
    assert bool(ranks.loc["C", "rank_changes"]) is False  # rank 2 in every regime
    assert ranks.loc["A", "rank_calm"] == 1
    assert ranks.loc["A", "rank_stressed"] == 3


# ── Tuning ────────────────────────────────────────────────────────────────────

def _make_tuning_inputs(seed: int = 11):
    """Synthetic daily returns, a monthly dataset, and vix_monthly for tuning tests."""
    from src.vrp import resample_to_month_start

    rng = np.random.default_rng(seed)
    daily_dates = pd.bdate_range("1995-01-02", periods=3000)
    daily = pd.Series(rng.normal(0, 0.01, len(daily_dates)), index=daily_dates, name="r")

    monthly_dates = resample_to_month_start(daily).index
    n = len(monthly_dates)
    vix = pd.Series(rng.uniform(15, 25, n), index=monthly_dates, name="vix")
    dataset = pd.DataFrame(
        {
            "vix_level": vix.to_numpy(),
            "cboe_skew": rng.uniform(110, 150, n),
            **{_c: rng.normal(0.008, 0.003, n) for _c in VRP_HORIZON_COLS},
            "realised_skew_21d": rng.normal(0, 0.4, n),
            "regime": rng.choice(config.REGIME_LABELS, n, p=[0.45, 0.45, 0.10]),
            "y": rng.normal(0.008, 0.003, n),
        },
        index=monthly_dates,
    )
    return dataset, daily, vix


_SMALL_GRID = [
    {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 50, "min_child_weight": 1},
    {"max_depth": 3, "learning_rate": 0.1, "n_estimators": 80, "min_child_weight": 5},
]


def test_tuning_selects_min_mean_qlike():
    dataset, daily, vix = _make_tuning_inputs()
    boundary = dataset.index[80]
    dtr = dataset.loc[:boundary]
    result = tune_xgboost_hyperparameters(
        dtr, daily, vix, grid=_SMALL_GRID, n_folds=2, min_train=40,
    )
    assert result.best_params in _SMALL_GRID
    # grid_scores sorted ascending; the selected best matches the first row.
    assert result.grid_scores["mean_val_qlike"].is_monotonic_increasing
    assert result.best_qlike == pytest.approx(result.grid_scores["mean_val_qlike"].iloc[0])


def test_tuning_ignores_out_of_sample_data():
    """Leakage: corrupting or truncating data after the boundary must not change selection."""
    dataset, daily, vix = _make_tuning_inputs()
    boundary = dataset.index[80]
    dtr = dataset.loc[:boundary]

    base = tune_xgboost_hyperparameters(
        dtr, daily, vix, grid=_SMALL_GRID, n_folds=2, min_train=40,
    )

    # Corrupt everything strictly after the boundary: monthly rows, daily returns, VIX.
    dataset_c = dataset.copy()
    after = dataset_c.index > boundary
    dataset_c.loc[after, [*VRP_HORIZON_COLS, "vix_level", "cboe_skew",
                          "realised_skew_21d", "y"]] = 1e3
    daily_c = daily.copy()
    daily_c.loc[daily_c.index > boundary] = 5.0
    vix_c = vix.copy()
    vix_c.loc[vix_c.index > boundary] = 999.0
    dtr_c = dataset_c.loc[:boundary]

    corrupt = tune_xgboost_hyperparameters(
        dtr_c, daily_c, vix_c, grid=_SMALL_GRID, n_folds=2, min_train=40,
    )

    assert corrupt.best_params == base.best_params
    pd.testing.assert_frame_equal(corrupt.grid_scores, base.grid_scores)

    # Truncating the OOS rows entirely also leaves it identical.
    trunc = tune_xgboost_hyperparameters(
        dataset.loc[:boundary], daily.loc[:boundary], vix.loc[:boundary],
        grid=_SMALL_GRID, n_folds=2, min_train=40,
    )
    assert trunc.best_params == base.best_params
    pd.testing.assert_frame_equal(trunc.grid_scores, base.grid_scores)
