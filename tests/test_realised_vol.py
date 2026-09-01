import numpy as np
import pandas as pd
import pytest

from src.realised_vol import compute_realised_vol


def _make_returns(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx, name="log_return", dtype=float)


# ---------------------------------------------------------------------------
# Shape and index properties
# ---------------------------------------------------------------------------

def test_output_length(window=21):
    n = 50
    s = _make_returns(np.random.default_rng(0).normal(0, 0.01, n))
    result = compute_realised_vol(s, window=window)
    assert len(result) == n - window + 1


def test_index_monotonic_increasing():
    s = _make_returns(np.random.default_rng(1).normal(0, 0.01, 30))
    result = compute_realised_vol(s, window=21)
    assert result.index.is_monotonic_increasing


def test_no_nans_in_output():
    s = _make_returns(np.random.default_rng(2).normal(0, 0.01, 30))
    result = compute_realised_vol(s, window=21)
    assert result.isna().sum() == 0


# ---------------------------------------------------------------------------
# Numerical checks
# ---------------------------------------------------------------------------

def test_constant_returns_give_zero_rv():
    s = _make_returns([0.01] * 30)
    result = compute_realised_vol(s, window=21)
    assert result.values == pytest.approx(0.0, abs=1e-10)


def test_rv_mean_in_plausible_range():
    rng = np.random.default_rng(42)
    s = _make_returns(rng.normal(0, 0.01, 1000))
    result = compute_realised_vol(s, window=21)
    # 0.01 * sqrt(252) * 100 ≈ 15.87; allow ±2 for sampling noise
    assert 14.0 < result.mean() < 18.0


# ---------------------------------------------------------------------------
# Window-boundary checks
# ---------------------------------------------------------------------------

def test_one_short_of_window_gives_empty():
    window = 21
    s = _make_returns([0.01] * (window - 1))
    result = compute_realised_vol(s, window=window)
    assert len(result) == 0


def test_exactly_window_gives_one_row():
    window = 21
    s = _make_returns([0.01] * window)
    result = compute_realised_vol(s, window=window)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Leakage check
# ---------------------------------------------------------------------------

def test_leakage():
    """RV at index t must use only returns in [t - window + 1, t].

    For each cutoff position k, the RV computed on returns[:k] must equal the
    RV at position k-1 in the full-series result. Any forward-looking
    contamination would break this equality.
    """
    window = 21
    rng = np.random.default_rng(7)
    returns = _make_returns(rng.normal(0, 0.01, 60))
    full_rv = compute_realised_vol(returns, window=window)

    for k in [window, window + 5, window + 10]:
        rv_at_k = compute_realised_vol(returns.iloc[:k], window=window)
        # Last value of rv_at_k must equal value at index k-1 in full_rv
        pd.testing.assert_series_equal(
            rv_at_k.iloc[[-1]],
            full_rv.loc[rv_at_k.index[-1:]],
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        compute_realised_vol(pd.Series([], dtype=float))


def test_raises_on_nan_in_input():
    s = _make_returns([0.01, np.nan, 0.02] + [0.01] * 20)
    with pytest.raises(ValueError, match="NaN"):
        compute_realised_vol(s)
