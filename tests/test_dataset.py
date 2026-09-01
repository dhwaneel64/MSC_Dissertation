import numpy as np
import pandas as pd
import pytest

from src.dataset import build_model_dataset


def _make_monthly(n: int, start: str = "2000-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="MS")
    return pd.DataFrame(
        {
            "vrp": np.arange(float(n)),
            "feat_a": np.arange(float(n)) * 2,
            "feat_b": np.arange(float(n)) * 3,
            "feat_c": np.arange(float(n)) * 4,
        },
        index=idx,
    )


# ── Basic shape and column order ──────────────────────────────────────────────

def test_basic_shape_and_column_order():
    df = _make_monthly(50)
    feat = ["feat_a", "feat_b", "feat_c"]
    out = build_model_dataset(df, feat, target_col="vrp", target_horizon=1)
    # shift(-1) drops the last row → 49 rows
    assert out.shape[0] == 49
    assert list(out.columns) == feat + ["y"]


# ── Validation: missing columns ───────────────────────────────────────────────

def test_missing_feature_column_raises():
    df = _make_monthly(50)
    with pytest.raises(ValueError, match="missing_col"):
        build_model_dataset(df, ["feat_a", "missing_col"])


def test_missing_target_column_raises():
    df = _make_monthly(50)
    with pytest.raises(ValueError, match="no_such_target"):
        build_model_dataset(df, ["feat_a"], target_col="no_such_target")


# ── Validation: target_horizon ───────────────────────────────────────────────

def test_target_horizon_zero_raises():
    df = _make_monthly(50)
    with pytest.raises(ValueError):
        build_model_dataset(df, ["feat_a"], target_horizon=0)


def test_target_horizon_negative_raises():
    df = _make_monthly(50)
    with pytest.raises(ValueError):
        build_model_dataset(df, ["feat_a"], target_horizon=-1)


# ── Validation: minimum sample size ──────────────────────────────────────────

def test_fewer_than_24_rows_raises():
    # 24 rows: shift(-1) leaves 23, which is < 24 → raises
    df = _make_monthly(24)
    with pytest.raises(ValueError):
        build_model_dataset(df, ["feat_a"])


def test_exactly_24_rows_does_not_raise():
    # 25 rows: shift(-1) leaves 24, which is not < 24 → OK
    df = _make_monthly(25)
    out = build_model_dataset(df, ["feat_a"])
    assert len(out) == 24


# ── Feature-list-aware NaN drop ───────────────────────────────────────────────

def test_nan_drop_respects_feature_list():
    """
    col_early has NaN in rows 0-5; col_late has NaN in the last 2 rows.
    Calling without col_early retains the early rows.
    Calling with col_early drops them.
    This is the key slope-availability behaviour: including a column with
    early NaNs (like slope before 2012) shrinks the dataset start date.
    """
    n = 60
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    vrp = np.arange(float(n))
    col_early = vrp.copy()
    col_early[:6] = np.nan   # NaN in rows 0-5
    col_late = vrp.copy()
    col_late[-2:] = np.nan   # NaN in rows 58-59

    df = pd.DataFrame(
        {"vrp": vrp, "col_early": col_early, "col_late": col_late},
        index=idx,
    )

    # Without col_early: rows 58 and 59 dropped (NaN col_late or NaN y)
    without_early = build_model_dataset(df, ["col_late"])
    assert len(without_early) == n - 2

    # With col_early: rows 0-5 additionally dropped (6 early + 2 late)
    with_early = build_model_dataset(df, ["col_early", "col_late"])
    assert len(with_early) == n - 8

    # Excluding col_early keeps the early rows that including it would drop
    assert without_early.index[0] < with_early.index[0]


# ── Leakage test ──────────────────────────────────────────────────────────────

def test_no_feature_leakage_and_y_alignment():
    """
    For each row t in the output:
      (a) every feature value equals the feature value at input row t (no shifting).
      (b) y equals target_col at input row t+1.
    Target is set to its own row index so shifts are visually obvious.
    """
    n = 50
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    # vrp = 0, 1, ..., 49 so shift(-1) gives y = 1, 2, ..., 49 at rows 0-48
    target = np.arange(float(n))
    feat_vals = np.arange(float(n)) * 10  # feat_x = 0, 10, 20, ..., 490
    df = pd.DataFrame({"vrp": target, "feat_x": feat_vals}, index=idx)

    out = build_model_dataset(df, ["feat_x"], target_col="vrp", target_horizon=1)

    print("\n--- head(3) ---")
    print(out.head(3))
    print("--- tail(3) ---")
    print(out.tail(3))

    for row_idx, row in out.iterrows():
        pos = df.index.get_loc(row_idx)
        assert row["feat_x"] == df.loc[row_idx, "feat_x"], (
            f"Feature shifted at {row_idx}: got {row['feat_x']}, "
            f"expected {df.loc[row_idx, 'feat_x']}"
        )
        assert row["y"] == df["vrp"].iloc[pos + 1], (
            f"y misaligned at {row_idx}: got {row['y']}, "
            f"expected {df['vrp'].iloc[pos + 1]}"
        )


# ── Categorical dtype passthrough ─────────────────────────────────────────────

def test_categorical_passthrough():
    """Categorical columns are accepted, not coerced, and preserved through dropna."""
    n = 50
    idx = pd.date_range("2000-01-01", periods=n, freq="MS")
    cat_dtype = pd.CategoricalDtype(
        categories=["calm", "normal", "stressed"], ordered=True
    )
    regime_vals = pd.Categorical(
        ["calm"] * 25 + ["normal"] * 25, dtype=cat_dtype
    )
    df = pd.DataFrame(
        {"vrp": np.arange(float(n)), "regime": regime_vals},
        index=idx,
    )
    out = build_model_dataset(df, ["regime"], target_col="vrp", target_horizon=1)
    assert out["regime"].dtype == cat_dtype
    assert list(out["regime"].cat.categories) == ["calm", "normal", "stressed"]
    assert out["regime"].cat.ordered


# ── build_model_ready_dataset ─────────────────────────────────────────────────

from src import config as _config
from src.dataset import build_model_ready_dataset
from src.validation import LOCKED_FEATURE_SET, assert_feature_set_complete
from src.vrp import resample_to_month_start
from src.regimes import label_regimes


def _make_full_inputs(n_days: int = 800, seed: int = 0):
    """Synthetic inputs for build_model_ready_dataset tests."""
    rng = np.random.default_rng(seed)
    day_idx = pd.bdate_range("2000-01-03", periods=n_days)
    returns = pd.Series(rng.normal(0, 0.01, n_days), index=day_idx, name="log_return")
    vix_daily = pd.Series(20.0, index=day_idx, name="close")
    vix_monthly = resample_to_month_start(vix_daily)
    month_idx = vix_monthly.index
    vrp = pd.Series(rng.normal(0.01, 0.005, len(month_idx)), index=month_idx, name="vrp")
    skew_monthly = pd.Series(rng.uniform(110, 140, len(month_idx)), index=month_idx, name="close")
    regime_labels = label_regimes(vix_monthly)
    return vrp, vix_monthly, skew_monthly, returns, regime_labels


def test_build_model_ready_dataset_no_nan():
    vrp, vix_m, skew_m, returns, regimes = _make_full_inputs()
    assert build_model_ready_dataset(vrp, vix_m, skew_m, returns, regimes).isna().sum().sum() == 0


def test_build_model_ready_dataset_has_all_locked_features():
    vrp, vix_m, skew_m, returns, regimes = _make_full_inputs()
    dataset = build_model_ready_dataset(vrp, vix_m, skew_m, returns, regimes)
    for col in LOCKED_FEATURE_SET:
        assert col in dataset.columns, f"missing column: {col}"


def test_build_model_ready_dataset_has_y_column():
    vrp, vix_m, skew_m, returns, regimes = _make_full_inputs()
    assert "y" in build_model_ready_dataset(vrp, vix_m, skew_m, returns, regimes).columns


def test_build_model_ready_dataset_assert_feature_set_passes():
    vrp, vix_m, skew_m, returns, regimes = _make_full_inputs()
    assert_feature_set_complete(build_model_ready_dataset(vrp, vix_m, skew_m, returns, regimes))


def test_build_model_ready_dataset_regime_is_categorical():
    vrp, vix_m, skew_m, returns, regimes = _make_full_inputs()
    dataset = build_model_ready_dataset(vrp, vix_m, skew_m, returns, regimes)
    assert isinstance(dataset["regime"].dtype, pd.CategoricalDtype)
    assert list(dataset["regime"].cat.categories) == list(_config.REGIME_LABELS)


def test_build_model_ready_dataset_lag_warmup_rows_dropped():
    """First dataset row has no NaN lag columns — all warmup rows removed by dropna."""
    vrp, vix_m, skew_m, returns, regimes = _make_full_inputs()
    dataset = build_model_ready_dataset(vrp, vix_m, skew_m, returns, regimes)
    for k in _config.HAR_LAGS_MONTHS:
        assert pd.notna(dataset[f"vrp_h{k}m"].iloc[0]), (
            f"vrp_h{k}m NaN in first row — lag warmup not fully dropped"
        )


def test_build_model_ready_dataset_trailing_row_dropped():
    """Last vrp date must be absent because it has no t+1 target."""
    vrp, vix_m, skew_m, returns, regimes = _make_full_inputs()
    dataset = build_model_ready_dataset(vrp, vix_m, skew_m, returns, regimes)
    assert vrp.index[-1] not in dataset.index, "last vrp date should be absent (no forward target)"


def test_build_model_ready_dataset_y_is_forward_vrp():
    """y at row t equals vrp at the next vrp index date after t."""
    vrp, vix_m, skew_m, returns, regimes = _make_full_inputs()
    dataset = build_model_ready_dataset(vrp, vix_m, skew_m, returns, regimes)
    for i in range(min(5, len(dataset))):
        t = dataset.index[i]
        t_next = vrp.index[vrp.index.get_loc(t) + 1]
        assert dataset.loc[t, "y"] == pytest.approx(vrp.loc[t_next])


def test_build_model_ready_dataset_leakage_input_truncation():
    """STANDING LEAKAGE GATE: all feature values at t are byte-identical when inputs truncated at t_next.

    Truncates every input Series at t_next (the next vrp date after t) so that no data
    beyond t_next is visible to the function. If any feature at t secretly uses post-t data,
    the truncated and full values will differ and the equality assertion will fail.

    Uses n_days=1200 to ensure the truncated dataset contains >= 24 rows after dropna,
    satisfying the minimum-sample-size guard in build_model_dataset.
    """
    vrp, vix_m, skew_m, returns, regimes = _make_full_inputs(n_days=1200, seed=42)
    dataset_full = build_model_ready_dataset(vrp, vix_m, skew_m, returns, regimes)

    # Pick an interior date with enough history that truncation leaves >= 24 rows.
    t = dataset_full.index[30]
    t_next = vrp.index[vrp.index.get_loc(t) + 1]

    dataset_trunc = build_model_ready_dataset(
        vrp.loc[:t_next],
        vix_m.loc[:t_next],
        skew_m.loc[:t_next],
        returns.loc[:t_next],
        regimes.loc[:t_next],
    )

    assert t in dataset_full.index, "t not in full dataset"
    assert t in dataset_trunc.index, "t not in truncated dataset"

    for col in LOCKED_FEATURE_SET:
        v_full = dataset_full.loc[t, col]
        v_trunc = dataset_trunc.loc[t, col]
        assert v_full == v_trunc, (
            f"Leakage in '{col}' at {t.date()}: full={v_full!r}, truncated={v_trunc!r}"
        )

    assert dataset_full.loc[t, "y"] == dataset_trunc.loc[t, "y"], (
        f"y mismatch at {t.date()}: full={dataset_full.loc[t, 'y']}, "
        f"trunc={dataset_trunc.loc[t, 'y']}"
    )
