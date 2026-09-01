import pandas as pd

from src import config
from src.har_rv import fit_har_rv_forecast


def resample_to_month_start(daily: pd.Series) -> pd.Series:
    """Return values from `daily` at the first available trading day of each calendar month."""
    if isinstance(daily, pd.DataFrame):
        daily = daily.iloc[:, 0]
    mask = daily.groupby([daily.index.year, daily.index.month]).cumcount() == 0
    result = daily[mask]
    result.name = daily.name
    return result



def build_vrp_series(
    vix_monthly: pd.Series,
    daily_log_returns: pd.Series,
    forecast_dates,
) -> pd.Series:
    """Construct VRP(t) = VIX^2(t)/VIX_VARIANCE_SCALE - E_t[RV^2(t+1)].

    Both terms are in annualised decimal variance units.  VIX is quoted in
    vol-points (e.g., VIX=20 means 20% annualised vol), so squaring and
    dividing by VIX_VARIANCE_SCALE (10000) converts to decimal annualised
    variance, matching the units produced by fit_har_rv_forecast.

    Parameters
    ----------
    vix_monthly : pd.Series
        VIX levels at monthly first-trading-day dates, in vol-points.
    daily_log_returns : pd.Series
        Daily SPY log-returns with a monotonic DatetimeIndex.
    forecast_dates : sequence of date-like
        Monthly first-trading-day dates at which to compute VRP.

    Returns
    -------
    pd.Series
        VRP in annualised decimal variance units, indexed by the HAR-valid
        subset of forecast_dates.  Name: "vrp".

    Raises
    ------
    ValueError
        If vix_monthly lacks entries for any HAR-valid forecast date.
    """
    # Physical variance forecast: E_t[RV^2(t+1)] in annualised decimal variance.
    physical_var = fit_har_rv_forecast(daily_log_returns, forecast_dates)

    # Risk-neutral variance: VIX^2 / VIX_VARIANCE_SCALE.
    # VIX in vol-points (VIX=20 = 20% annualised vol) -> decimal variance = (20/100)^2 = 0.04.
    # Equivalently: 20^2 / 10000 = 0.04.
    vix_at_dates = vix_monthly.reindex(physical_var.index)
    if vix_at_dates.isna().any():
        missing = vix_at_dates.index[vix_at_dates.isna()].tolist()
        raise ValueError(
            f"vix_monthly is missing values for {len(missing)} forecast date(s); "
            f"first missing: {missing[0]}"
        )
    # vix_at_dates: vol-points -> vix_var: decimal annualised variance
    vix_var = vix_at_dates ** 2 / config.VIX_VARIANCE_SCALE

    vrp = vix_var - physical_var
    vrp.name = "vrp"
    return vrp
