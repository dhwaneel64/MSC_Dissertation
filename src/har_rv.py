"""Recursive HAR-RV variance forecaster for the Bekaert-Hoerova VRP construction."""
import pandas as pd
import statsmodels.api as sm

from src import config


def realised_variance_target(daily_log_returns: pd.Series) -> pd.Series:
    
    if daily_log_returns.empty:
        raise ValueError("daily_log_returns is empty")
    if not daily_log_returns.index.is_monotonic_increasing:
        raise ValueError("daily_log_returns index must be monotonic increasing")

    daily_rv = daily_log_returns ** 2
    target = (
        daily_rv.rolling(config.RV_WINDOW).mean().shift(-config.RV_WINDOW)
        * config.ANNUALISATION_FACTOR_DAILY
    )
    target.name = "realised_variance_target"
    return target


def fit_har_rv_forecast(
    daily_log_returns: pd.Series,
    forecast_dates,
) -> pd.Series:
   
    if daily_log_returns.empty:
        raise ValueError("daily_log_returns is empty")
    if not daily_log_returns.index.is_monotonic_increasing:
        raise ValueError("daily_log_returns index must be monotonic increasing")

    # Build all series once from the full return history.
    daily_rv = daily_log_returns ** 2
    reg_D = daily_rv.rolling(config.HAR_RV_HORIZON_D).mean()
    reg_W = daily_rv.rolling(config.HAR_RV_HORIZON_W).mean()
    reg_M = daily_rv.rolling(config.HAR_RV_HORIZON_M).mean()
    # target[d] covers positions [d+1, d+RV_WINDOW]; see realised_variance_target.
    target = realised_variance_target(daily_log_returns)

    # Drop rows where any regressor or target is NaN (first ~M-1 rows lack reg_M;
    # last RV_WINDOW rows lack target after the shift).
    data = pd.DataFrame(
        {"D": reg_D, "W": reg_W, "M": reg_M, "target": target}
    ).dropna()

    results: dict = {}
    for t in pd.DatetimeIndex(forecast_dates):
        if t not in daily_rv.index:
            continue

        pos_t = daily_rv.index.get_loc(t)

        # Training rows: position i where i + RV_WINDOW <= pos_t, meaning the
        # forward window [i+1, i+RV_WINDOW] is fully observed by t.
        if pos_t < config.RV_WINDOW:
            continue
        cutoff_date = daily_rv.index[pos_t - config.RV_WINDOW]

        train = data.loc[data.index <= cutoff_date]
        if len(train) < config.HAR_RV_MIN_TRAIN_ROWS:
            continue

        d_t = reg_D.at[t]
        w_t = reg_W.at[t]
        m_t = reg_M.at[t]
        if pd.isna(d_t) or pd.isna(w_t) or pd.isna(m_t):
            continue

        X_train = sm.add_constant(train[["D", "W", "M"]], has_constant="add")
        fit = sm.OLS(train["target"], X_train).fit()

        x_t = sm.add_constant(
            pd.DataFrame({"D": [d_t], "W": [w_t], "M": [m_t]}),
            has_constant="add",
        )
        results[t] = float(fit.predict(x_t).iloc[0])

    return pd.Series(results, name="har_rv_forecast")
