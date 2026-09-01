"""Descriptive prediction diagnostics, applied to every model in the sequence.

These quantities are interpretability only: tracking, bias direction, error size,
and where the largest misses cluster by regime. They do not feed model comparison
and do not change any locked metric. QLIKE remains the sole adjudicating loss for
model selection (the locked methodology, Loss functions). Nothing here is used to rank models.
"""
import numpy as np
import pandas as pd


def compute_prediction_diagnostics(
    wf_result: pd.DataFrame,
    dataset: pd.DataFrame,
    model_name: str,
) -> dict:
    """Descriptive diagnostics for one model's walk-forward predictions.

    Read-only on wf_result and dataset. Pure computation, no plotting.

    Args:
        wf_result: walk-forward output with columns "y_true" and "y_pred" (VRP in
            decimal annualised variance), indexed by OOS dates.
        dataset: model-ready DataFrame carrying the "regime" label, indexed so that
            every wf_result date is present. Used only to attribute the largest
            misses to a regime (where misses cluster); it does not enter any of the
            scalar error statistics.
        model_name: label recorded on the returned row.

    Returns:
        dict (one diagnostics row):
          - model: model_name.
          - mean_signed_error: mean(y_pred - y_true). Positive means the model
            over-predicts VRP on average, negative means it under-predicts.
          - mae: mean absolute error.
          - rmse: root mean squared error.
          - r2: 1 - SS_res / SS_tot, the fraction of VRP variance explained
            (SS_tot uses the OOS mean of y_true). May be low or negative for
            monthly VRP; reported as-is.
          - n_under_predicted: count of months with y_pred < y_true.
          - n_over_predicted: count of months with y_pred > y_true.
            (Months with y_pred == y_true fall in neither count.)
          - worst_regime: regime label with the highest mean absolute error.
          - worst_regime_mae: that regime's mean absolute error.

    Raises:
        ValueError if wf_result lacks the required columns, contains NaN, or has a
        date with no regime label in dataset.
    """
    for col in ("y_true", "y_pred"):
        if col not in wf_result.columns:
            raise ValueError(f"wf_result is missing required column {col!r}")

    missing_dates = wf_result.index.difference(dataset.index)
    if len(missing_dates) > 0:
        raise ValueError(
            f"wf_result has dates with no row in dataset: {list(missing_dates)}"
        )
    if "regime" not in dataset.columns:
        raise ValueError("dataset is missing the 'regime' column for diagnostics")

    y_true = wf_result["y_true"].to_numpy(dtype=float)
    y_pred = wf_result["y_pred"].to_numpy(dtype=float)
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("wf_result contains NaN in y_true or y_pred")

    error = y_pred - y_true  # signed: positive => over-prediction
    abs_error = np.abs(error)

    mean_signed_error = float(error.mean())
    mae = float(abs_error.mean())
    rmse = float(np.sqrt((error ** 2).mean()))

    ss_res = float((error ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    n_under_predicted = int((y_pred < y_true).sum())
    n_over_predicted = int((y_pred > y_true).sum())

    regimes = dataset.loc[wf_result.index, "regime"].astype(str).to_numpy()
    regime_mae = (
        pd.Series(abs_error, index=regimes).groupby(level=0).mean()
    )
    worst_regime = str(regime_mae.idxmax())
    worst_regime_mae = float(regime_mae.max())

    return {
        "model": model_name,
        "mean_signed_error": mean_signed_error,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "n_under_predicted": n_under_predicted,
        "n_over_predicted": n_over_predicted,
        "worst_regime": worst_regime,
        "worst_regime_mae": worst_regime_mae,
    }


def largest_absolute_errors(
    wf_result: pd.DataFrame,
    dataset: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """The k OOS months with the largest absolute prediction error, with regime.

    Read-only. Returns a DataFrame indexed by date with columns y_true, y_pred,
    abs_error, and regime, sorted by abs_error descending. Used to show whether a
    model's biggest misses cluster by market regime.
    """
    y_true = wf_result["y_true"].to_numpy(dtype=float)
    y_pred = wf_result["y_pred"].to_numpy(dtype=float)
    abs_error = np.abs(y_pred - y_true)

    table = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "abs_error": abs_error,
            "regime": dataset.loc[wf_result.index, "regime"].astype(str).to_numpy(),
        },
        index=wf_result.index,
    )
    return table.sort_values("abs_error", ascending=False).head(k)
