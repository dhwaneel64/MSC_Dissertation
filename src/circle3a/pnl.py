"""Circle 3A synthetic variance-swap P&L for the naive and conditioned strategies.

Consumes the position frame from positions.build_position_series. No positions are
created here and no forecast is refit. The monthly payoff is a synthetic
short-variance-swap proxy: enter at month t, receive the variance strike implied by
the entry-month VIX and pay the variance realised over the window that strike
prices. Both legs are in decimal annualised variance. By construction the payoff
equals the realised VRP (implied variance minus realised variance); this identity is
intended and is what fixes the alignment of the two legs.

Both legs are dated t. The strike is VIX(t)**2 / VIX_VARIANCE_SCALE and the realised
leg is realised_variance_target evaluated at t, which already covers the RV_WINDOW
trading days after t. The locked spec writes that window as realised_variance(t+1)
because it names the month following t as t+1, while the code indexes it at t;
applying .shift(-1) on top of the estimator's own forward window would pay the
variance of the month after the one the strike prices and would break the realised
VRP identity above.

Two information sets are kept disjoint. The position at t uses only data with index
<= t (see positions.py). The realised leg is a forward realisation used only in the
payoff (the scoring stage), never fed back into the position rule. It is the same
forward, uncentred estimator score_walk_forward uses; it is deliberately not
compute_realised_vol.
"""
import pandas as pd

from src import config
from src.har_rv import realised_variance_target
from src.vrp import resample_to_month_start


def build_pnl_frame(
    positions: pd.DataFrame,
    vix_monthly: pd.Series,
    daily_log_returns: pd.Series,
    haircut_base: float = config.COST_HAIRCUT_BASE,
    stress_multiplier: float = config.COST_STRESS_MULTIPLIER,
    vix_variance_scale: int = config.VIX_VARIANCE_SCALE,
) -> pd.DataFrame:
    """Synthetic variance-swap P&L for both strategies, base and stress costs.

    Payoff per unit short-variance position entered at t:
        payoff(t) = vix_monthly(t)**2 / vix_variance_scale - realised_variance_next(t)
    where vix_monthly(t) is the entry-month VIX (month-start, vol points, not
    vix_next) and realised_variance_next(t) is
        resample_to_month_start(realised_variance_target(daily_log_returns))
            .reindex(positions.index),
    with no further shift: the estimator at t already covers the RV_WINDOW trading
    days after t, the window the entry-month strike prices.

    Gross return(t) = position(t) * payoff(t), for position_naive and
    position_conditioned.

    Cost is proportional to the entry-month variance strike:
        cost per round trip = haircut_base * vix_monthly(t)**2 / vix_variance_scale,
    charged on every position transition (open or close, including a gate exit
    1 -> 0 and the first entry from a flat start). The stress cost multiplies the
    base by stress_multiplier. Net = gross - cost, per strategy, for base and
    stress.

    Months whose realised leg is NaN (the final OOS months whose forward window is
    not fully in sample) are dropped from the frame; the drop count is available as
    the frame attribute ``n_nan_dropped``.

    Parameters
    ----------
    positions : pd.DataFrame
        Output of build_position_series; must carry position_naive and
        position_conditioned, indexed by OOS month t.
    vix_monthly : pd.Series
        Entry-month VIX at t (month-start, vol points). Must cover positions.index
        with no NaN there.
    daily_log_returns : pd.Series
        Daily SPY log-returns with a monotonic DatetimeIndex, spanning the OOS span
        plus the forward RV_WINDOW days.
    haircut_base, stress_multiplier, vix_variance_scale
        Cost and unit-conversion constants, defaulting to config.

    Returns
    -------
    pd.DataFrame
        Indexed by the retained (non-NaN-realised) subset of positions.index,
        columns in order: payoff, position_naive, position_conditioned,
        gross_naive, gross_conditioned, net_naive_base, net_conditioned_base,
        net_naive_stress, net_conditioned_stress. Carries an ``n_nan_dropped``
        attribute (int) in ``DataFrame.attrs``.

    Raises
    ------
    ValueError
        If positions lacks a required position column, or vix_monthly has NaN at
        any positions.index date.
    """
    for col in ("position_naive", "position_conditioned"):
        if col not in positions.columns:
            raise ValueError(f"positions is missing required column {col!r}")

    vix_t = vix_monthly.reindex(positions.index)
    if vix_t.isna().any():
        raise ValueError("vix_monthly has NaN at one or more positions.index dates")

    realised_daily = realised_variance_target(daily_log_returns)
    realised_monthly = resample_to_month_start(realised_daily)
    realised_variance_next = realised_monthly.reindex(positions.index)

    variance_strike = vix_t ** 2 / vix_variance_scale
    payoff = variance_strike - realised_variance_next

    pos_naive = positions["position_naive"]
    pos_conditioned = positions["position_conditioned"]

    # Transition detection treats the pre-sample state as flat (0), so the first
    # entry from a flat start is charged and a gate exit 1 -> 0 is charged.
    trans_naive = pos_naive != pos_naive.shift(1, fill_value=0)
    trans_conditioned = pos_conditioned != pos_conditioned.shift(1, fill_value=0)

    gross_naive = pos_naive * payoff
    gross_conditioned = pos_conditioned * payoff

    cost_base_naive = trans_naive * haircut_base * variance_strike
    cost_base_conditioned = trans_conditioned * haircut_base * variance_strike

    frame = pd.DataFrame(
        {
            "payoff": payoff,
            "position_naive": pos_naive,
            "position_conditioned": pos_conditioned,
            "gross_naive": gross_naive,
            "gross_conditioned": gross_conditioned,
            "net_naive_base": gross_naive - cost_base_naive,
            "net_conditioned_base": gross_conditioned - cost_base_conditioned,
            "net_naive_stress": gross_naive - cost_base_naive * stress_multiplier,
            "net_conditioned_stress": gross_conditioned - cost_base_conditioned * stress_multiplier,
        }
    )
    frame.index.name = positions.index.name

    nan_mask = realised_variance_next.isna().to_numpy()
    n_nan_dropped = int(nan_mask.sum())
    frame = frame.loc[~nan_mask]
    frame.attrs["n_nan_dropped"] = n_nan_dropped
    return frame
