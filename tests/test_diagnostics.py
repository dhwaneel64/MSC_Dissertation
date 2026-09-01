import numpy as np
import pandas as pd
import pytest

from src.diagnostics import compute_prediction_diagnostics, largest_absolute_errors


def _known_case():
    """A small hand-checkable case.

    y_true = [1, 2, 3, 4], y_pred = [2, 2, 2, 2].
    error  = y_pred - y_true = [1, 0, -1, -2].
      mean_signed_error = -0.5
      abs_error = [1, 0, 1, 2] -> mae = 1.0
      rmse = sqrt((1 + 0 + 1 + 4) / 4) = sqrt(1.5)
      ss_res = 6, ss_tot = 5 (y_true mean 2.5) -> r2 = 1 - 6/5 = -0.2
      n_under_predicted = 2 (pred < true at rows 2, 3)
      n_over_predicted  = 1 (pred > true at row 0)
    regimes = [calm, calm, normal, stressed]:
      regime MAE -> calm 0.5, normal 1.0, stressed 2.0 -> worst stressed, 2.0
    """
    dates = pd.date_range("2005-01-01", periods=4, freq="MS")
    wf = pd.DataFrame({"y_true": [1.0, 2.0, 3.0, 4.0], "y_pred": [2.0, 2.0, 2.0, 2.0]}, index=dates)
    dataset = pd.DataFrame({"regime": ["calm", "calm", "normal", "stressed"]}, index=dates)
    return wf, dataset


def test_compute_prediction_diagnostics_known_values():
    wf, dataset = _known_case()
    d = compute_prediction_diagnostics(wf, dataset, "toy")

    assert d["model"] == "toy"
    assert d["mean_signed_error"] == pytest.approx(-0.5)
    assert d["mae"] == pytest.approx(1.0)
    assert d["rmse"] == pytest.approx(np.sqrt(1.5))
    assert d["r2"] == pytest.approx(-0.2)
    assert d["n_under_predicted"] == 2
    assert d["n_over_predicted"] == 1
    assert d["worst_regime"] == "stressed"
    assert d["worst_regime_mae"] == pytest.approx(2.0)


def test_under_over_counts_exclude_exact_ties():
    """Equality counts toward neither under nor over."""
    dates = pd.date_range("2005-01-01", periods=3, freq="MS")
    wf = pd.DataFrame({"y_true": [1.0, 1.0, 1.0], "y_pred": [1.0, 0.5, 1.5]}, index=dates)
    dataset = pd.DataFrame({"regime": ["calm", "calm", "calm"]}, index=dates)
    d = compute_prediction_diagnostics(wf, dataset, "toy")
    assert d["n_under_predicted"] == 1
    assert d["n_over_predicted"] == 1


def test_perfect_predictions_r2_one_zero_errors():
    dates = pd.date_range("2005-01-01", periods=4, freq="MS")
    wf = pd.DataFrame({"y_true": [1.0, 2.0, 3.0, 4.0], "y_pred": [1.0, 2.0, 3.0, 4.0]}, index=dates)
    dataset = pd.DataFrame({"regime": ["calm", "normal", "normal", "stressed"]}, index=dates)
    d = compute_prediction_diagnostics(wf, dataset, "perfect")
    assert d["mae"] == pytest.approx(0.0)
    assert d["rmse"] == pytest.approx(0.0)
    assert d["r2"] == pytest.approx(1.0)
    assert d["n_under_predicted"] == 0
    assert d["n_over_predicted"] == 0


def test_raises_on_missing_regime_date():
    wf, dataset = _known_case()
    dataset_short = dataset.iloc[:3]  # drop the last date
    with pytest.raises(ValueError, match="no row in dataset"):
        compute_prediction_diagnostics(wf, dataset_short, "toy")


def test_raises_on_nan():
    wf, dataset = _known_case()
    wf = wf.copy()
    wf.iloc[0, wf.columns.get_loc("y_pred")] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        compute_prediction_diagnostics(wf, dataset, "toy")


def test_largest_absolute_errors_sorted_with_regime():
    # Tie-free abs errors so the ordering is unambiguous: [0.5, 1.5, 3.0, 0.2].
    dates = pd.date_range("2005-01-01", periods=4, freq="MS")
    wf = pd.DataFrame(
        {"y_true": [1.0, 2.0, 3.0, 4.0], "y_pred": [1.5, 0.5, 6.0, 3.8]}, index=dates
    )
    dataset = pd.DataFrame(
        {"regime": ["calm", "normal", "stressed", "calm"]}, index=dates
    )
    top2 = largest_absolute_errors(wf, dataset, 2)
    assert list(top2.index) == [dates[2], dates[1]]
    assert top2["abs_error"].tolist() == pytest.approx([3.0, 1.5])
    assert top2["regime"].tolist() == ["stressed", "normal"]
    assert top2.loc[dates[2], "y_true"] == 3.0 and top2.loc[dates[2], "y_pred"] == 6.0
