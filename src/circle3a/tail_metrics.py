"""Circle 3A tail-risk episode table and full-sample tail metrics.

Descriptive read-only layer over the synthetic variance-swap P&L frame produced by
pnl.build_pnl_frame. No model is refit, no position is rebuilt, and no payoff is
recomputed: every number here is an aggregation of the existing net_*_base and
net_*_stress columns. The point of Circle 3A task 3 is to characterise how the
regime gate reshapes the loss tail relative to the naive short-vol rule, so the two
outputs focus on the crash episodes and on the shape of the monthly P&L
distribution rather than on any headline average.

Two strategies are reported throughout, naive (signal only, position_naive) and
conditioned (signal gated to non-stressed months, position_conditioned), plus a
flat zero-position benchmark whose P&L is identically zero.
"""
import numpy as np
import pandas as pd

from src import config
from src.forecast_comparison import block_bootstrap_pvalue

# Strategy key -> (position column, net-base column, net-stress column) in the pnl
# frame. Row order in every output follows this mapping.
_STRATEGIES = {
    "naive": ("position_naive", "net_naive_base", "net_naive_stress"),
    "conditioned": ("position_conditioned", "net_conditioned_base", "net_conditioned_stress"),
}


def episode_pnl_table(
    pnl: pd.DataFrame,
    vix_at_t: pd.Series,
    crash_windows=config.CIRCLE3A_CRASH_WINDOWS,
    vix_threshold: float = config.CIRCLE3A_TAIL_VIX_THRESHOLD,
) -> pd.DataFrame:
    """Summed net-base P&L per strategy over each crash episode.

    Episodes are the named crash windows in ``crash_windows`` plus one row for each
    OOS month in ``pnl`` whose entry-month VIX exceeds ``vix_threshold`` and does not
    already fall inside a named window (so rows stay disjoint). For each episode the
    naive and conditioned columns sum net_naive_base / net_conditioned_base over the
    ``pnl`` rows inside the window; the flat column is identically zero (a
    zero-position benchmark bears no P&L). Costs are included via the net-base
    columns; payoff is never recomputed here.

    Parameters
    ----------
    pnl : pd.DataFrame
        Output of build_pnl_frame, indexed by OOS month t. Must carry
        net_naive_base and net_conditioned_base.
    vix_at_t : pd.Series
        Entry-month VIX at t (month-start, vol points), covering pnl.index.
    crash_windows : sequence of (label, start, end)
        Named windows, bounds inclusive and calendar-date parseable.
    vix_threshold : float
        Single-month episode trigger on the entry-month VIX.

    Returns
    -------
    pd.DataFrame
        Indexed by episode label, columns naive_net, conditioned_net, flat_net.

    Raises
    ------
    ValueError
        If pnl lacks a required net-base column or vix_at_t has NaN on pnl.index.
    """
    for col in ("net_naive_base", "net_conditioned_base"):
        if col not in pnl.columns:
            raise ValueError(f"pnl is missing required column {col!r}")

    vix = vix_at_t.reindex(pnl.index)
    if vix.isna().any():
        raise ValueError("vix_at_t has NaN at one or more pnl.index dates")

    windows = [
        (label, pd.Timestamp(start), pd.Timestamp(end)) for label, start, end in crash_windows
    ]

    # Single-month VIX-spike episodes, excluding any month already inside a named
    # window so the episode rows do not double count.
    in_named = pd.Series(False, index=pnl.index)
    for _, start, end in windows:
        in_named |= (pnl.index >= start) & (pnl.index <= end)
    spike_months = pnl.index[(vix.to_numpy() > vix_threshold) & (~in_named.to_numpy())]
    for t in spike_months:
        label = f"vix>{vix_threshold}:{t.date()}"
        windows.append((label, t, t))

    rows = []
    labels = []
    for label, start, end in windows:
        mask = (pnl.index >= start) & (pnl.index <= end)
        rows.append(
            {
                "naive_net": float(pnl.loc[mask, "net_naive_base"].sum()),
                "conditioned_net": float(pnl.loc[mask, "net_conditioned_base"].sum()),
                "flat_net": 0.0,
            }
        )
        labels.append(label)

    table = pd.DataFrame(rows, index=pd.Index(labels, name="episode"))
    return table


def _max_drawdown(net: pd.Series) -> float:
    """Max peak-to-trough drop of the cumulative-sum equity curve, as a positive
    magnitude (0.0 if the curve never falls below a running peak)."""
    cum = net.cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    return float(-drawdown.min())


def tail_metrics_table(
    pnl: pd.DataFrame,
    months_per_year: int = config.MONTHS_PER_YEAR,
    worst_roll_months: int = config.CIRCLE3A_WORST_ROLL_MONTHS,
) -> pd.DataFrame:
    """Full-sample tail metrics per strategy on the monthly net-P&L series.

    Annualisation is from monthly frequency: annualised mean = monthly mean *
    months_per_year, annualised vol = monthly std (ddof=1) * sqrt(months_per_year),
    Sharpe = annualised mean / annualised vol. total_net, max_drawdown, worst_month,
    worst_3m, skew and hit_rate are computed on the net-base series; total_net and
    Sharpe are additionally reported on the net-stress series. Max drawdown is a
    positive magnitude on the cumulative-sum curve. Hit rate is the fraction of
    traded months (position == 1) with strictly positive net base.

    Parameters
    ----------
    pnl : pd.DataFrame
        Output of build_pnl_frame, indexed by OOS month t.
    months_per_year : int
        Annualisation factor for the monthly series.
    worst_roll_months : int
        Rolling window length for the worst-rolling-sum metric.

    Returns
    -------
    pd.DataFrame
        Indexed by strategy (naive, conditioned), columns total_net_base,
        annualised_mean_base, annualised_vol_base, sharpe_base, max_drawdown,
        worst_month, worst_3m, skew, hit_rate, total_net_stress,
        annualised_mean_stress, annualised_vol_stress, sharpe_stress.

    Raises
    ------
    ValueError
        If pnl lacks a required strategy column.
    """
    required = [c for cols in _STRATEGIES.values() for c in cols]
    for col in required:
        if col not in pnl.columns:
            raise ValueError(f"pnl is missing required column {col!r}")

    root = np.sqrt(months_per_year)
    rows = []
    for key, (pos_col, base_col, stress_col) in _STRATEGIES.items():
        base = pnl[base_col]
        stress = pnl[stress_col]
        traded = pnl[pos_col] == 1

        ann_mean_base = float(base.mean()) * months_per_year
        ann_vol_base = float(base.std(ddof=1)) * root
        ann_mean_stress = float(stress.mean()) * months_per_year
        ann_vol_stress = float(stress.std(ddof=1)) * root

        n_traded = int(traded.sum())
        hit_rate = float((base[traded] > 0).mean()) if n_traded else float("nan")

        rows.append(
            {
                "total_net_base": float(base.sum()),
                "annualised_mean_base": ann_mean_base,
                "annualised_vol_base": ann_vol_base,
                "sharpe_base": ann_mean_base / ann_vol_base if ann_vol_base else float("nan"),
                "max_drawdown": _max_drawdown(base),
                "worst_month": float(base.min()),
                "worst_3m": float(base.rolling(worst_roll_months).sum().min()),
                "skew": float(base.skew()),
                "hit_rate": hit_rate,
                "total_net_stress": float(stress.sum()),
                "annualised_mean_stress": ann_mean_stress,
                "annualised_vol_stress": ann_vol_stress,
                "sharpe_stress": ann_mean_stress / ann_vol_stress if ann_vol_stress else float("nan"),
            }
        )

    table = pd.DataFrame(rows, index=pd.Index(list(_STRATEGIES), name="strategy"))
    return table


def _annualised_sharpe(net: np.ndarray, months_per_year: int) -> float:
    """Annualised Sharpe of a monthly net series: sqrt(m/y) * mean / std(ddof=1)."""
    mean = net.mean()
    std = net.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(np.sqrt(months_per_year) * mean / std)


def _loo_sharpe_diff(cond: np.ndarray, naive: np.ndarray, months_per_year: int) -> np.ndarray:
    """Leave-one-out annualised Sharpe difference (conditioned minus naive).

    Vectorised: for each month i the Sharpe of each strategy is recomputed on the
    n-1 remaining months (ddof=1), and the difference is returned. Used to form the
    jackknife pseudo-values fed to the shared block bootstrap.
    """
    root = np.sqrt(months_per_year)
    n = len(cond)

    def _loo(x):
        total = x.sum()
        total_sq = (x ** 2).sum()
        loo_mean = (total - x) / (n - 1)
        # Sample variance (ddof=1) of the n-1 remaining points.
        loo_ss = total_sq - x ** 2
        loo_var = (loo_ss - (n - 1) * loo_mean ** 2) / (n - 2)
        loo_std = np.sqrt(loo_var)
        return root * loo_mean / loo_std

    return _loo(cond) - _loo(naive)


def sharpe_diff_bootstrap(
    pnl: pd.DataFrame,
    months_per_year: int = config.MONTHS_PER_YEAR,
) -> pd.DataFrame:
    """Block-bootstrap p-value for the conditioned-minus-naive Sharpe difference.

    The point statistic is the annualised Sharpe difference (conditioned minus
    naive) on the monthly net series, using the same annualisation as
    tail_metrics_table (x months_per_year on the mean, x sqrt(months_per_year) on
    the vol). The p-value reuses the repo block-bootstrap util
    (forecast_comparison.block_bootstrap_pvalue): the nonlinear Sharpe difference is
    linearised into per-month jackknife pseudo-values whose sample mean equals the
    jackknife Sharpe difference, and those pseudo-values (month order preserved, so
    the circular block structure still applies) are passed to the util. The test is
    one-sided under H0 that the difference is <= 0 (conditioned does not beat naive)
    against H1 that it is > 0; a negative difference therefore yields a high p-value,
    which is reported as-is. Block length, replications, and seed come from config
    via the util. Reported for both base and stress costs.

    Parameters
    ----------
    pnl : pd.DataFrame
        Output of build_pnl_frame, indexed by OOS month t.
    months_per_year : int
        Annualisation factor for the monthly series.

    Returns
    -------
    pd.DataFrame
        Indexed by cost basis (base, stress), columns sharpe_naive,
        sharpe_conditioned, sharpe_diff, p_value, block_length, n_months.

    Raises
    ------
    ValueError
        If pnl lacks a required net column.
    """
    bases = {
        "base": ("net_naive_base", "net_conditioned_base"),
        "stress": ("net_naive_stress", "net_conditioned_stress"),
    }
    for naive_col, cond_col in bases.values():
        for col in (naive_col, cond_col):
            if col not in pnl.columns:
                raise ValueError(f"pnl is missing required column {col!r}")

    n = len(pnl)
    rows = []
    for label, (naive_col, cond_col) in bases.items():
        naive = pnl[naive_col].to_numpy(dtype=float)
        cond = pnl[cond_col].to_numpy(dtype=float)

        sharpe_naive = _annualised_sharpe(naive, months_per_year)
        sharpe_cond = _annualised_sharpe(cond, months_per_year)
        sharpe_diff = sharpe_cond - sharpe_naive

        # Jackknife pseudo-values: pseudo_i = n*S - (n-1)*S_{-i}, mean(pseudo) = S_jack.
        loo_diff = _loo_sharpe_diff(cond, naive, months_per_year)
        pseudo = n * sharpe_diff - (n - 1) * loo_diff

        p_value, _observed_mean, block_length = block_bootstrap_pvalue(pseudo)

        rows.append(
            {
                "sharpe_naive": sharpe_naive,
                "sharpe_conditioned": sharpe_cond,
                "sharpe_diff": sharpe_diff,
                "p_value": p_value,
                "block_length": block_length,
                "n_months": n,
            }
        )

    table = pd.DataFrame(rows, index=pd.Index(list(bases), name="cost_basis"))
    return table
