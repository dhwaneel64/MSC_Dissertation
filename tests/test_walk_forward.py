import numpy as np
import pandas as pd
import pytest

from src.walk_forward import walk_forward, make_model_factory_from_class


# ---------------------------------------------------------------------------
# Dummy models used across tests
# ---------------------------------------------------------------------------

class DummyMeanModel:
    """Predicts the historical mean of y_train. Stateless, no randomness."""
    def __init__(self):
        self._mean = None

    def fit(self, X_train, y_train):
        self._mean = float(y_train.mean())

    def predict(self, X_test):
        return np.full(len(X_test), self._mean)


class LeakyModel:
    """Deliberately tries to access t+1 data via a saved reference. Used to test
    that the engine's API prevents leakage by structure, not by trust."""
    def __init__(self):
        self.full_dataset_ref = None

    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test):
        # Even if a malicious model wanted to peek, the engine only hands it
        # X_test built from dataset rows. The test below confirms the X_test
        # passed in here contains only one row, the OOS row, with feature
        # values from time t (not t+1).
        return np.zeros(len(X_test))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_synthetic_dataset(n_rows: int = 50) -> pd.DataFrame:
    """50 monthly rows; y = row index (0, 1, ..., n-1); three arbitrary feature cols."""
    dates = pd.date_range("2000-01-01", periods=n_rows, freq="MS")
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "f1": rng.standard_normal(n_rows),
            "f2": rng.standard_normal(n_rows),
            "f3": rng.standard_normal(n_rows),
            "y": np.arange(n_rows, dtype=float),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


FEATURE_COLS = ["f1", "f2", "f3"]


# ---------------------------------------------------------------------------
# Test 1: Basic correctness with DummyMeanModel
# ---------------------------------------------------------------------------

def test_basic_correctness():
    ds = make_synthetic_dataset(50)
    # First 10 rows train; OOS starts at row 10
    initial_train_end = ds.index[9]

    result = walk_forward(
        ds, FEATURE_COLS, lambda: DummyMeanModel(), initial_train_end
    )

    # 40 OOS rows (rows 10 through 49)
    assert len(result) == 40, f"Expected 40 OOS rows, got {len(result)}"

    # y_true must equal original y for rows 10..49
    expected_y_true = ds["y"].iloc[10:].values
    np.testing.assert_array_equal(result["y_true"].values, expected_y_true)

    # y_pred at step k (0-indexed OOS) should equal mean of y[0..k+9] (rows 0..k+9)
    # First prediction (OOS row 10) = mean(y[0:10]) = 4.5
    assert abs(result["y_pred"].iloc[0] - 4.5) < 1e-10, (
        f"First OOS prediction should be 4.5, got {result['y_pred'].iloc[0]}"
    )

    for oos_idx in range(40):
        train_size = 10 + oos_idx  # rows 0..train_size-1
        expected_pred = float(np.arange(train_size).mean())
        actual_pred = float(result["y_pred"].iloc[oos_idx])
        assert abs(actual_pred - expected_pred) < 1e-10, (
            f"OOS step {oos_idx}: expected {expected_pred}, got {actual_pred}"
        )


# ---------------------------------------------------------------------------
# Test 2: Index integrity
# ---------------------------------------------------------------------------

def test_index_integrity():
    ds = make_synthetic_dataset(50)
    initial_train_end = ds.index[9]
    result = walk_forward(ds, FEATURE_COLS, lambda: DummyMeanModel(), initial_train_end)

    assert result.index.is_monotonic_increasing, "Result index must be monotonic increasing"

    assert set(result.index).issubset(set(ds.index)), (
        "Result index must be a strict subset of dataset.index"
    )

    assert (result.index > initial_train_end).all(), (
        "Result index must not contain any date <= initial_train_end"
    )


# ---------------------------------------------------------------------------
# Test 3: Fresh model per step
# ---------------------------------------------------------------------------

class CountingModel:
    _instance_count = 0

    def __init__(self):
        CountingModel._instance_count += 1
        self._mean = None

    def fit(self, X_train, y_train):
        self._mean = float(y_train.mean())

    def predict(self, X_test):
        return np.full(len(X_test), self._mean)


def test_fresh_model_per_step():
    ds = make_synthetic_dataset(50)
    initial_train_end = ds.index[9]
    CountingModel._instance_count = 0

    walk_forward(ds, FEATURE_COLS, lambda: CountingModel(), initial_train_end)

    expected_steps = 40  # rows 10..49
    assert CountingModel._instance_count == expected_steps, (
        f"Expected {expected_steps} model instantiations, got {CountingModel._instance_count}"
    )


# ---------------------------------------------------------------------------
# Test 4: Leakage-by-structure test
# ---------------------------------------------------------------------------

class InspectorModel:
    """Raises AssertionError if the engine ever hands it future data."""
    def __init__(self):
        self._train_max_date = None

    def fit(self, X_train, y_train):
        self._train_max_date = X_train.index.max()

    def predict(self, X_test):
        assert len(X_test) == 1, (
            f"predict() received {len(X_test)} rows; engine must pass exactly 1"
        )
        oos_date = X_test.index[0]
        assert oos_date > self._train_max_date, (
            f"OOS date {oos_date} is not strictly after training max {self._train_max_date}"
        )
        assert oos_date != self._train_max_date, (
            "OOS date equals the training max date; OOS row must be strictly later"
        )
        return np.zeros(len(X_test))


def test_no_leakage_by_structure():
    ds = make_synthetic_dataset(50)
    initial_train_end = ds.index[9]
    # If the engine passes any future or duplicate data, InspectorModel will raise
    walk_forward(ds, FEATURE_COLS, lambda: InspectorModel(), initial_train_end)


# ---------------------------------------------------------------------------
# Test 5: Validation errors
# ---------------------------------------------------------------------------

def test_raises_if_initial_train_end_not_in_index():
    ds = make_synthetic_dataset(50)
    bad_date = pd.Timestamp("1999-01-01")
    with pytest.raises(ValueError, match="not in dataset.index"):
        walk_forward(ds, FEATURE_COLS, lambda: DummyMeanModel(), bad_date)


def test_raises_if_index_not_monotonic():
    ds = make_synthetic_dataset(50)
    # Swap two rows to break monotonicity
    shuffled = ds.copy()
    idx = list(shuffled.index)
    idx[1], idx[2] = idx[2], idx[1]
    shuffled.index = pd.DatetimeIndex(idx)
    with pytest.raises(ValueError, match="monotonic increasing"):
        walk_forward(shuffled, FEATURE_COLS, lambda: DummyMeanModel(), shuffled.index[0])


def test_raises_if_initial_train_end_is_last_row():
    ds = make_synthetic_dataset(50)
    last_date = ds.index[-1]
    with pytest.raises(ValueError, match="no OOS rows available"):
        walk_forward(ds, FEATURE_COLS, lambda: DummyMeanModel(), last_date)
