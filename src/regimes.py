import math

import pandas as pd

from src import config


def classify_regime(
    vix_value: float,
    calm_upper: float = config.VIX_CALM_UPPER,
    stressed_lower: float = config.VIX_STRESSED_LOWER,
) -> str:
    """Return "calm", "normal", or "stressed" based on vix_value.

    Boundaries: calm is strictly below calm_upper; stressed is strictly above
    stressed_lower; normal is the closed interval [calm_upper, stressed_lower].
    Raises ValueError on NaN input.
    """
    if math.isnan(vix_value):
        raise ValueError(f"vix_value is NaN")
    if vix_value < calm_upper:
        return "calm"
    if vix_value > stressed_lower:
        return "stressed"
    return "normal"


def label_regimes(
    vix_series: pd.Series,
    calm_upper: float = config.VIX_CALM_UPPER,
    stressed_lower: float = config.VIX_STRESSED_LOWER,
) -> pd.Series:
    """Apply classify_regime element-wise to vix_series.

    Returns a categorical Series with categories ordered as config.REGIME_LABELS,
    index preserved, and name set to "regime". Raises ValueError on empty input
    or any NaN values.
    """
    if vix_series.empty:
        raise ValueError("vix_series is empty")
    if vix_series.isna().any():
        raise ValueError("vix_series contains NaN values")
    labels = vix_series.map(lambda v: classify_regime(v, calm_upper, stressed_lower))
    cat_dtype = pd.CategoricalDtype(categories=list(config.REGIME_LABELS), ordered=True)
    result = labels.astype(cat_dtype)
    result.name = "regime"
    return result
