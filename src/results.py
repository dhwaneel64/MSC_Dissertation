import numpy as np
import pandas as pd

from src import config
from src.evaluation import vrp_forecast_to_variance
from src.har_rv import realised_variance_target
from src.metrics import directional_accuracy, mse, qlike
from src.vrp import resample_to_month_start


def score_walk_forward(
    wf_result: pd.DataFrame,
    vix_next: pd.Series,
    daily_log_returns: pd.Series,
) -> dict:
    
    missing = wf_result.index.difference(vix_next.index)
    if len(missing) > 0:
        raise ValueError(
            f"wf_result.index has dates not in vix_next.index: {list(missing)}"
        )

    vix_next = vix_next.reindex(wf_result.index)
    if vix_next.isna().any():
        raise ValueError("vix_next contains NaN at one or more wf_result OOS dates")

    realised_var_target_daily = realised_variance_target(daily_log_returns)
    realised_var_target_monthly = resample_to_month_start(realised_var_target_daily)
    # The .shift(-1) is correct HERE and only here, and must not be removed to match
    # mincer_zarnowitz or circle3a/pnl. Those two score a leg dated t (VIX(t)), so
    # their realised counterpart is the estimator at t. This function scores a leg
    # dated t+1: y_pred is a forecast of VRP(t+1) and vix_next is VIX(t+1), so the
    # realised counterpart is the estimator at t+1, which is one shift forward.
    realised_var_next = realised_var_target_monthly.shift(-1).reindex(wf_result.index)

    y_true = wf_result["y_true"].to_numpy(dtype=float)
    y_pred = wf_result["y_pred"].to_numpy(dtype=float)
    vix_arr = vix_next.to_numpy(dtype=float)
    realised_arr = realised_var_next.to_numpy(dtype=float)

    implied_all = vix_arr ** 2 / config.VIX_VARIANCE_SCALE - y_pred
    guard_mask = implied_all > 0
    nan_tail_mask = ~np.isnan(realised_arr)
    valid_mask = guard_mask & nan_tail_mask

    n_guard_excluded = int((~guard_mask).sum())
    n_nan_tail_excluded = int((guard_mask & ~nan_tail_mask).sum())
    n_valid = int(valid_mask.sum())

    forecast_var_valid = vrp_forecast_to_variance(y_pred[valid_mask], vix_arr[valid_mask])
    realised_var_valid = realised_arr[valid_mask]
    score_qlike = qlike(realised_var_valid, forecast_var_valid)

    score_mse = mse(y_true, y_pred)
    score_da = directional_accuracy(y_true, y_pred)

    return {
        "qlike": score_qlike,
        "qlike_n": n_valid,
        "mse": score_mse,
        "mse_n": len(wf_result),
        "directional_accuracy": score_da,
        "da_n": len(wf_result),
        "n_guard_excluded": n_guard_excluded,
        "n_nan_tail_excluded": n_nan_tail_excluded,
        "n_obs": len(wf_result),
        "first_oos_date": wf_result.index.min(),
        "last_oos_date": wf_result.index.max(),
        "valid_mask": valid_mask,
        "implied_variance": implied_all,
        "realised_variance_next": realised_arr,
    }
