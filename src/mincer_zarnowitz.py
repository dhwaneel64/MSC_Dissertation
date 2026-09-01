
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from src import config
from src.har_rv import realised_variance_target
from src.regimes import label_regimes
from src.vrp import resample_to_month_start


@dataclass
class MZResult:
    """One Mincer-Zarnowitz regression: realised(t+1) = alpha + beta * VIX_var(t)."""
    sample: str          # "full", "calm", "normal", or "stressed"
    n: int               # observations in this regression
    alpha: float         # intercept
    se_alpha: float      # Newey-West HAC standard error of alpha
    beta: float          # slope on VIX_variance
    se_beta: float       # Newey-West HAC standard error of beta
    t_alpha: float       # t-stat for H0: alpha = 0
    p_alpha: float       # two-sided p-value for H0: alpha = 0
    t_beta1: float       # t-stat for H0: beta = 1
    p_beta1: float       # two-sided p-value for H0: beta = 1
    wald_stat: float     # joint Wald stat for H0: alpha = 0 AND beta = 1
    wald_p: float        # p-value for the joint Wald test (chi-square, df=2)
    hac_lag: int         # HAC truncation lag used (Newey-West 1994 rule on this n)
    reject_rationality: bool  # True if wald_p < alpha_level


def newey_west_lag(n: int) -> int:
    """Newey-West (1994) automatic HAC lag: floor(mult * (n / 100) ** exp).

    Uses config.NW_HAC_MULTIPLIER and config.NW_HAC_EXPONENT, computed from the
    regression's own sample size n, so a smaller per-regime sample gets a
    smaller lag.
    """
    return math.floor(config.NW_HAC_MULTIPLIER * (n / 100) ** config.NW_HAC_EXPONENT)


def mincer_zarnowitz_regression(
    realised,
    forecast,
    sample: str = "full",
    alpha_level: float = config.COMPARISON_ALPHA,
) -> MZResult:
    """Estimate realised = alpha + beta * forecast with NW HAC and a joint Wald test.

    OLS gives the point estimates; the covariance is Newey-West HAC with lag set
    by newey_west_lag on this regression's sample size. The joint rationality
    restriction alpha = 0 AND beta = 1 is tested by a Wald statistic built from
    the HAC covariance:

        W = (theta - q)' [R V R']^{-1} (theta - q),  theta = (alpha, beta),
        R = I_2, q = (0, 1), V = HAC covariance, W ~ chi-square(2) under H0.

    Individual t-tests (alpha = 0; beta = 1) use the same HAC standard errors
    against the standard normal, matching statsmodels' use_t=False default for a
    HAC-covariance fit.

    Args:
        realised: realised_variance(t+1) outcomes (forward target), 1-D.
        forecast: VIX_variance(t) regressor, 1-D, same length as realised.
        sample: label recorded on the result ("full"/"calm"/"normal"/"stressed").
        alpha_level: significance threshold for reject_rationality.

    Returns:
        MZResult.

    Raises:
        ValueError on shape mismatch, non-1-D input, NaN, or fewer than 3 obs.
    """
    realised = np.asarray(realised, dtype=float)
    forecast = np.asarray(forecast, dtype=float)

    if realised.ndim != 1 or forecast.ndim != 1:
        raise ValueError("realised and forecast must both be 1-D")
    if realised.shape != forecast.shape:
        raise ValueError(
            f"length mismatch: realised {realised.shape}, forecast {forecast.shape}"
        )
    if np.isnan(realised).any() or np.isnan(forecast).any():
        raise ValueError("realised/forecast contain NaN")

    n = len(realised)
    if n < 3:
        raise ValueError(
            f"sample {sample!r} has n={n}; need at least 3 observations for MZ"
        )

    hac_lag = newey_west_lag(n)
    X = sm.add_constant(forecast)  # columns: const, x1 (slope on forecast)
    fit = sm.OLS(realised, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lag})

    alpha_hat, beta_hat = float(fit.params[0]), float(fit.params[1])
    se_alpha, se_beta = float(fit.bse[0]), float(fit.bse[1])

    # Individual t-tests with the HAC standard errors (standard normal reference).
    t_alpha = alpha_hat / se_alpha
    p_alpha = float(2.0 * stats.norm.sf(abs(t_alpha)))
    t_beta1 = (beta_hat - 1.0) / se_beta
    p_beta1 = float(2.0 * stats.norm.sf(abs(t_beta1)))

    # Joint Wald test of alpha = 0 AND beta = 1 from the HAC covariance.
    theta = fit.params
    q = np.array([0.0, 1.0])
    V = np.asarray(fit.cov_params())  # R = I_2, so R V R' = V
    diff = theta - q
    wald_stat = float(diff @ np.linalg.inv(V) @ diff)
    wald_p = float(stats.chi2.sf(wald_stat, df=2))

    return MZResult(
        sample=sample,
        n=n,
        alpha=alpha_hat,
        se_alpha=se_alpha,
        beta=beta_hat,
        se_beta=se_beta,
        t_alpha=float(t_alpha),
        p_alpha=p_alpha,
        t_beta1=float(t_beta1),
        p_beta1=p_beta1,
        wald_stat=wald_stat,
        wald_p=wald_p,
        hac_lag=hac_lag,
        reject_rationality=wald_p < alpha_level,
    )


def build_mz_frame(
    vix_monthly: pd.Series,
    daily_log_returns: pd.Series,
) -> pd.DataFrame:
    """Pair VIX_variance(t) with the variance realised after t, and regime(t).

    The forecast is VIX_variance(t) = VIX(t)**2 / VIX_VARIANCE_SCALE at month t.
    The realised outcome is realised_variance_target evaluated at month t, which
    already covers the RV_WINDOW trading days AFTER t. That forward window is the
    period VIX(t) prices, so no further shift is applied. In the notation of the
    locked methodology (RV(t+1) = alpha + beta * VIX(t)) this value IS RV(t+1);
    the methodology writes the month following t as t+1 while the code indexes it
    at t, and applying .shift(-1) on top of the estimator's own forward window
    would reach one month too far. It is the same estimator the QLIKE realised
    side uses, not the backward compute_realised_vol. Regime is classified by
    VIX(t) at the forecast date, consistent with the rest of the pipeline.

    Rows where either side is NaN are dropped: the initial rows are kept (both
    sides are available early), and the forward-window NaN tail (the most recent
    month, whose realised target is not yet observable) is removed.

    Args:
        vix_monthly: VIX levels in vol-points at monthly first-trading-day dates.
        daily_log_returns: daily SPY log-returns with a monotonic DatetimeIndex.

    Returns:
        DataFrame indexed by month t with columns:
          - vix_variance: VIX_variance(t), decimal annualised variance.
          - realised_variance_next: realised_variance(t+1), forward target.
          - regime: regime label at month t.
    """
    forecast = vix_monthly ** 2 / config.VIX_VARIANCE_SCALE

    realised_daily = realised_variance_target(daily_log_returns)
    realised_monthly = resample_to_month_start(realised_daily)
    # No shift: realised_monthly at t already covers the RV_WINDOW days after t,
    # which is the window VIX(t) prices.
    realised_next = realised_monthly.reindex(forecast.index)

    regime = label_regimes(vix_monthly).reindex(forecast.index)

    frame = pd.DataFrame(
        {
            "vix_variance": forecast,
            "realised_variance_next": realised_next,
            "regime": regime,
        }
    )
    return frame.dropna(subset=["vix_variance", "realised_variance_next"])


def run_mincer_zarnowitz(
    vix_monthly: pd.Series,
    daily_log_returns: pd.Series,
    alpha_level: float = config.COMPARISON_ALPHA,
) -> tuple[list[MZResult], pd.DataFrame]:
    """Run the MZ rationality test full-sample and separately within each regime.

    Builds the paired (VIX_variance(t), realised_variance(t+1)) frame, runs the
    regression on the full sample and on each regime subset (regime defined by
    VIX at month t), and returns the four results in canonical order plus the
    frame used.

    Args:
        vix_monthly: VIX levels in vol-points at monthly first-trading-day dates.
        daily_log_returns: daily SPY log-returns with a monotonic DatetimeIndex.
        alpha_level: significance threshold for reject_rationality.

    Returns:
        (results, frame). results is [full, calm, normal, stressed].
    """
    frame = build_mz_frame(vix_monthly, daily_log_returns)

    results = [
        mincer_zarnowitz_regression(
            frame["realised_variance_next"],
            frame["vix_variance"],
            sample="full",
            alpha_level=alpha_level,
        )
    ]
    regime_str = frame["regime"].astype(str)
    for label in config.REGIME_LABELS:
        sub = frame[regime_str == label]
        results.append(
            mincer_zarnowitz_regression(
                sub["realised_variance_next"],
                sub["vix_variance"],
                sample=label,
                alpha_level=alpha_level,
            )
        )
    return results, frame


def mz_results_table(results: list[MZResult]) -> pd.DataFrame:
    """Assemble MZResult rows into the reporting DataFrame (no hardcoded values).

    One row per sample with alpha, SE(alpha), beta, SE(beta), the joint Wald
    statistic and p-value, n, and the reject/not-reject decision.
    """
    rows = [
        {
            "sample": r.sample,
            "alpha": r.alpha,
            "se_alpha": r.se_alpha,
            "beta": r.beta,
            "se_beta": r.se_beta,
            "wald_stat": r.wald_stat,
            "wald_p": r.wald_p,
            "n": r.n,
            "reject_rationality": r.reject_rationality,
        }
        for r in results
    ]
    return pd.DataFrame(rows)
