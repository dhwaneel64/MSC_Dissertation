import numpy as np
import pandas as pd
import pytest

from src.returns import compute_log_returns


def _make_series(values, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx, name="close", dtype=float)


def test_output_length():
    s = _make_series([100, 110, 99, 105, 102])
    assert len(compute_log_returns(s)) == len(s) - 1


def test_index_monotonic_increasing():
    s = _make_series([100, 110, 99, 105])
    assert compute_log_returns(s).index.is_monotonic_increasing


def test_no_nans_in_output():
    s = _make_series([100, 110, 99, 105])
    assert compute_log_returns(s).isna().sum() == 0


def test_numerical_values():
    s = _make_series([100, 110, 99])
    result = compute_log_returns(s)
    expected = np.array([np.log(110 / 100), np.log(99 / 110)])
    np.testing.assert_allclose(result.values, expected, rtol=1e-10)


def test_series_name():
    s = _make_series([100, 110, 99])
    assert compute_log_returns(s).name == "log_return"


def test_accepts_dataframe():
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    df = pd.DataFrame({"close": [100.0, 110.0, 99.0]}, index=idx)
    result = compute_log_returns(df)
    assert len(result) == 2


def test_raises_on_empty():
    with pytest.raises(ValueError, match="empty"):
        compute_log_returns(pd.Series([], dtype=float))


def test_raises_on_nan_in_input():
    s = _make_series([100, np.nan, 99])
    with pytest.raises(ValueError, match="NaN"):
        compute_log_returns(s)


def test_leakage():
    """Output at index t uses only prices at indices <= t.

    For each cutoff k, the returns computed on prices[:k] must equal the first
    k-1 returns computed on the full series.  Any look-ahead would cause a
    mismatch because the full-series computation has access to future prices.
    """
    prices = _make_series([100, 105, 98, 110, 103, 97, 108])
    full_result = compute_log_returns(prices)
    for k in [3, 4, 5]:
        truncated = compute_log_returns(prices.iloc[:k])
        pd.testing.assert_series_equal(
            truncated,
            full_result.iloc[: k - 1],
            check_names=False,
        )
