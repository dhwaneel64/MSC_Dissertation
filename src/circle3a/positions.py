"""Circle 3A position series for the naive and regime-conditioned strategies.

Positions only. No P&L, no costs, and no realised t+1 quantity enter this module
(Circle 3A spec, Leakage section). The position at month t is built entirely from
information knowable at entry: the Extended OLS one-month-ahead VRP forecast made
at t, and the VIX-threshold regime label at t. The forward realised outcome is a
scoring-stage input and is deliberately absent here, so the position series can be
verified as a strict function of data with index <= t.

VRP is the volatility risk premium, the gap between implied and subsequent
realised variance. A short-vol position collects that premium when it is
positive. The signal is one-sided: the strategy is either short (1) or flat (0),
with no long-vol leg.
"""
import pandas as pd

from src import config


def build_position_series(
    forecast: pd.Series,
    regime: pd.Series,
    stressed_label: str = config.REGIME_STRESSED_LABEL,
) -> pd.DataFrame:
    """Monthly short-vol positions for the naive and conditioned strategies.

    No model is fit and no forecast is recomputed here: the driver is the
    Extended OLS walk-forward forecast passed in as ``forecast`` (wf_extended
    ["y_pred"]). Both strategies share this one forecast series, one threshold,
    and one sample.

    Threshold: the expanding-window median of the forecast up to and including
    t, recomputed at every t via ``forecast.expanding().median()``. This uses
    only forecasts with index <= t, never a fixed value and never a full-sample
    quantile.

    Base signal (naive strategy): position_naive(t) = 1 (short vol) if
    forecast(t) is strictly greater than the expanding median at t, else 0. At
    the first observation the expanding median equals the single forecast, so
    the strict inequality is False and the position is flat.

    Gate (conditioned strategy): position_conditioned(t) = position_naive(t) AND
    regime(t) != stressed_label. The gate reads the regime at t, which is built
    from the VIX level at the first trading day of t and is therefore knowable at
    entry. The gate
    variable is the observable VIX-threshold regime, never any Mincer-Zarnowitz
    output, so the conditioning is non-circular.

    Parameters
    ----------
    forecast : pd.Series
        Extended OLS one-month-ahead VRP forecast, indexed by OOS month t.
        Monotonic increasing index, no NaN.
    regime : pd.Series
        VIX-threshold regime label at t, aligned to ``forecast`` (same index).
        No NaN.
    stressed_label : str
        Regime label that closes the gate. Defaults to
        config.REGIME_STRESSED_LABEL.

    Returns
    -------
    pd.DataFrame
        Indexed by OOS month t, columns in order:
          - forecast: the driver forecast at t.
          - expanding_median: expanding median of the forecast up to and
            including t.
          - regime: the regime label at t (as passed in).
          - position_naive: 1 if short, 0 if flat (signal only).
          - position_conditioned: position_naive gated to non-stressed months.

    Raises
    ------
    ValueError
        If forecast is empty, forecast or regime contains NaN, the two indexes
        differ, or forecast.index is not monotonic increasing.
    """
    if forecast.empty:
        raise ValueError("forecast is empty")
    if forecast.isna().any():
        raise ValueError("forecast contains NaN values")
    if regime.isna().any():
        raise ValueError("regime contains NaN values")
    if not forecast.index.equals(regime.index):
        raise ValueError("forecast and regime must share the same index")
    if not forecast.index.is_monotonic_increasing:
        raise ValueError("forecast.index must be monotonic increasing")

    expanding_median = forecast.expanding().median()

    naive_arr = forecast.to_numpy() > expanding_median.to_numpy()
    gate_open = regime.astype(str).to_numpy() != str(stressed_label)
    conditioned_arr = naive_arr & gate_open

    position_naive = pd.Series(naive_arr.astype(int), index=forecast.index)
    position_conditioned = pd.Series(conditioned_arr.astype(int), index=forecast.index)

    result = pd.DataFrame(
        {
            "forecast": forecast,
            "expanding_median": expanding_median,
            "regime": regime,
            "position_naive": position_naive,
            "position_conditioned": position_conditioned,
        }
    )
    result.index.name = forecast.index.name
    return result
