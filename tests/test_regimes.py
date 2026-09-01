import pandas as pd
import pytest

from src.regimes import classify_regime, label_regimes
from src.config import REGIME_LABELS, VIX_CALM_UPPER, VIX_STRESSED_LOWER


def _vix(values, start="2000-01-03"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.Series(list(values), index=idx, name="close", dtype=float)


# ── classify_regime ──────────────────────────────────────────────────────────
# Boundary convention: calm = VIX < VIX_CALM_UPPER (strict open),
# normal = [VIX_CALM_UPPER, VIX_STRESSED_LOWER] (closed),
# stressed = VIX > VIX_STRESSED_LOWER (strict open).

def test_just_below_calm_upper_is_calm():
    assert classify_regime(VIX_CALM_UPPER - 0.01) == "calm"


def test_at_calm_upper_is_normal():
    assert classify_regime(float(VIX_CALM_UPPER)) == "normal"


def test_at_stressed_lower_is_normal():
    assert classify_regime(float(VIX_STRESSED_LOWER)) == "normal"


def test_just_above_stressed_lower_is_stressed():
    assert classify_regime(VIX_STRESSED_LOWER + 0.01) == "stressed"


def test_classify_regime_nan_raises():
    with pytest.raises(ValueError):
        classify_regime(float("nan"))


# ── label_regimes ────────────────────────────────────────────────────────────

def test_all_three_labels_producible():
    s = _vix([VIX_CALM_UPPER - 1, VIX_CALM_UPPER, VIX_STRESSED_LOWER + 1])
    assert set(label_regimes(s)) == set(REGIME_LABELS)


def test_categories_match_config():
    result = label_regimes(_vix([10.0, 20.0, 30.0]))
    assert list(result.cat.categories) == list(REGIME_LABELS)


def test_output_is_categorical():
    result = label_regimes(_vix([10.0, 20.0, 30.0]))
    assert isinstance(result.dtype, pd.CategoricalDtype)


def test_name_is_regime():
    assert label_regimes(_vix([20.0])).name == "regime"


def test_index_preserved():
    s = _vix([10.0, 20.0, 30.0])
    pd.testing.assert_index_equal(label_regimes(s).index, s.index)


def test_no_nan_introduced():
    s = _vix([VIX_CALM_UPPER - 0.01, VIX_CALM_UPPER, VIX_STRESSED_LOWER, VIX_STRESSED_LOWER + 0.01])
    assert label_regimes(s).isna().sum() == 0


def test_boundary_values_map_correctly():
    s = _vix([VIX_CALM_UPPER - 0.01, VIX_CALM_UPPER, VIX_STRESSED_LOWER, VIX_STRESSED_LOWER + 0.01])
    assert list(label_regimes(s)) == ["calm", "normal", "normal", "stressed"]


def test_empty_raises():
    with pytest.raises(ValueError):
        label_regimes(pd.Series([], dtype=float))


def test_nan_input_raises():
    with pytest.raises(ValueError):
        label_regimes(_vix([10.0, float("nan"), 20.0]))


# Leakage note: classify_regime is a pointwise threshold on the contemporaneous VIX
# level only — no rolling window, no lookahead, no leakage possible.
