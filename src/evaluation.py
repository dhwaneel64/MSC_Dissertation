from __future__ import annotations

import numpy as np

from . import config


def vrp_forecast_to_variance(vrp_forecast, vix_next) -> np.ndarray:
    
    vrp_forecast = np.asarray(vrp_forecast, dtype=float)
    vix_next = np.asarray(vix_next, dtype=float)

    if vrp_forecast.shape != vix_next.shape:
        raise ValueError(
            f"Length mismatch: vrp_forecast has {vrp_forecast.shape}, "
            f"vix_next has {vix_next.shape}"
        )
    if np.any(np.isnan(vrp_forecast)):
        raise ValueError("vrp_forecast contains NaN")
    if np.any(np.isnan(vix_next)):
        raise ValueError("vix_next contains NaN")

    implied_variance = vix_next ** 2 / config.VIX_VARIANCE_SCALE - vrp_forecast

    bad = np.where(implied_variance <= 0)[0]
    if len(bad) > 0:
        details = "; ".join(
            f"index {i}: vix_next={vix_next[i]:.4f}, "
            f"vrp_forecast={vrp_forecast[i]:.6f}, "
            f"implied_variance={implied_variance[i]:.6f}"
            for i in bad
        )
        raise ValueError(
            f"implied_variance must be strictly > 0; invalid at: {details}"
        )

    return implied_variance


def rv_to_variance(rv_realised) -> np.ndarray:
    """Convert realised vol (vol points) to variance (decimal).
    variance = rv ** 2 / VIX_VARIANCE_SCALE. Raises if any rv <= 0 or NaN.
    """
    rv = np.asarray(rv_realised, dtype=float)

    if np.any(np.isnan(rv)):
        raise ValueError("rv_realised contains NaN")
    if np.any(rv <= 0):
        raise ValueError("rv_realised must be strictly > 0; got values <= 0")

    return rv ** 2 / config.VIX_VARIANCE_SCALE
