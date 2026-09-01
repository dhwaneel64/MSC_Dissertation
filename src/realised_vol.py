import numpy as np
import pandas as pd

from src import config


def compute_realised_vol(
    log_returns: pd.Series,
    window: int = config.RV_WINDOW,
    annualisation: int = config.ANNUALISATION_FACTOR_DAILY,
    vol_points_scale: int = config.VOL_POINTS_SCALE,
) -> pd.Series:
    """Rolling annualised realised volatility, expressed in volatility points.

    Computes the rolling standard deviation of log_returns over `window`
    trading days (ddof=1), annualises by multiplying by sqrt(annualisation),
    and scales by vol_points_scale so the units match VIX (e.g. 16.99 = 16.99%
    annualised vol). The window is strictly backward-looking: each value at
    index t uses only returns in [t - window + 1, t]. Initial NaN rows are
    dropped.
    """
    if log_returns.empty:
        raise ValueError("log_returns is empty")

    if log_returns.isna().any():
        raise ValueError("log_returns contains NaN values")

    rv = log_returns.rolling(window).std() * np.sqrt(annualisation) * vol_points_scale
    rv = rv.dropna()
    rv.name = "rv"
    return rv
