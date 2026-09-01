import pandas as pd
from scipy.stats import skew as scipy_skew

from src import config
from src.validation import VRP_HORIZON_COLS


def compute_realised_skew_21d(
    daily_log_returns: pd.Series,
    monthly_dates,
    window: int = config.RV_WINDOW,
) -> pd.Series:
    
    if daily_log_returns.empty:
        raise ValueError("daily_log_returns is empty")
    if not daily_log_returns.index.is_monotonic_increasing:
        raise ValueError("daily_log_returns index must be monotonic increasing")

    results: dict = {}
    for t in pd.DatetimeIndex(monthly_dates):
        up_to_t = daily_log_returns.loc[daily_log_returns.index <= t]
        if len(up_to_t) < window:
            continue
        results[t] = float(scipy_skew(up_to_t.iloc[-window:].values, bias=False))

    return pd.Series(results, name="realised_skew_21d")


def build_feature_matrix(
    vrp: pd.Series,
    vix_monthly: pd.Series,
    skew_monthly: pd.Series,
    daily_log_returns: pd.Series,
    regime_labels: pd.Series,
) -> pd.DataFrame:
    """Build the 7-column locked feature matrix indexed by vrp.index.

    The VRP columns are named by horizon to the target, not by lag from t. The
    target on row t is VRP(t+1) (see build_model_dataset), so a column at horizon k
    is vrp.shift(k - 1) and sits k steps from that target. The nearest column is
    therefore VRP(t), which is knowable at t: it is VIX^2(t) / VIX_VARIANCE_SCALE
    minus a HAR-RV forecast fitted only on data indexed <= t.

    Columns in LOCKED_FEATURE_SET order:
      vix_level          VIX level in vol-points at the first trading day of month t.
      cboe_skew          CBOE SKEW index value at the first trading day of month t.
      vrp_h1m            VRP at horizon 1 from the target: VRP(t), vrp.shift(0).
      vrp_h3m            VRP at horizon 3 from the target: VRP(t-2), vrp.shift(2).
      vrp_h6m            VRP at horizon 6 from the target: VRP(t-5), vrp.shift(5).
      realised_skew_21d  scipy.stats.skew on the trailing config.RV_WINDOW daily returns at t.
      regime             Categorical regime label at t.

    NaN rows introduced by lags are NOT dropped here; they are removed in
    build_model_ready_dataset.

    Parameters
    ----------
    vrp : pd.Series
        Canonical VRP series from build_vrp_series, monthly indexed.
    vix_monthly : pd.Series
        VIX levels in vol-points, indexed by monthly first-trading-day dates.
    skew_monthly : pd.Series
        CBOE SKEW index values, indexed by monthly first-trading-day dates.
    daily_log_returns : pd.Series
        Daily SPY log-returns with a monotonic DatetimeIndex.
    regime_labels : pd.Series
        Categorical regime labels, indexed by monthly first-trading-day dates.

    Returns
    -------
    pd.DataFrame
        7-column feature DataFrame indexed by vrp.index.
    """
    idx = vrp.index

    features = pd.DataFrame(index=idx)
    features["vix_level"] = vix_monthly.reindex(idx)
    features["cboe_skew"] = skew_monthly.reindex(idx)
    for k, col in zip(config.HAR_LAGS_MONTHS, VRP_HORIZON_COLS):
        features[col] = vrp.shift(k - 1)
    features["realised_skew_21d"] = compute_realised_skew_21d(daily_log_returns, idx)
    features["regime"] = regime_labels.reindex(idx)

    return features
