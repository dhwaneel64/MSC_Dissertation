import numpy as np
import pandas as pd


def compute_log_returns(prices: "pd.Series | pd.DataFrame", price_col: str = "close") -> pd.Series:
    """Return ln(P_t / P_{t-1}) with the first (NaN) row dropped.

    Accepts a Series or a single-column DataFrame; if a DataFrame is supplied
    the column named price_col is extracted before computing returns.
    """
    if isinstance(prices, pd.DataFrame):
        prices = prices[price_col]

    if prices.empty:
        raise ValueError("prices is empty")

    if prices.isna().any():
        raise ValueError("prices contains NaN values")

    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns.name = "log_return"
    return log_returns
