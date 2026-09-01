"""Data-generating process for the CW/QLIKE bootstrap simulation study.

Simulates a latent monthly log-variance AR(1) with fat-tailed shocks, a
persistent exogenous predictor, and a realised-variance measurement built
from the latent variance times positive mean-1 noise. Under the alternative
world the predictor enters the variance equation with a one-period lag.
"""

import numpy as np
from scipy import stats

import config


def generate_history(world, beta, t_total, rng):
    """Simulate one history of the DGP.

    world: "null" or "alt". Under "null" the predictor never enters the
        variance equation and beta is ignored. Under "alt" the term
        beta * x_{t-1} is added to log h_t.
    beta: loading on the lagged predictor under the alternative world.
    t_total: number of usable observations to return after burn-in. This must
        equal T_TRAIN + T_OOS for the caller.
    rng: a numpy Generator supplying all randomness for this path.

    Returns a dict with arrays h (latent variance), rv (realised variance),
    and x (predictor), each of length t_total.

    The predictor is simulated independently of the log-variance shocks, so it
    is exogenous. The predictor enters with a one-period lag (x_{t-1} drives
    h_t) so a model forecasting t+1 from the level of x_t is correctly
    specified. A burn-in from config is discarded from the front so the initial
    condition is irrelevant.
    """
    if world not in ("null", "alt"):
        raise ValueError("world must be 'null' or 'alt'")

    total = config.BURN_IN + t_total

    # Predictor x: AR(1) with persistence psi and Gaussian innovations scaled
    # by the config shock scale. Drawn first and independently of the
    # log-variance shocks so the predictor is exogenous.
    x_shocks = config.PREDICTOR_SHOCK_SCALE * rng.standard_normal(total)
    x = np.empty(total)
    x[0] = x_shocks[0]
    for t in range(1, total):
        x[t] = config.PREDICTOR_PSI * x[t - 1] + x_shocks[t]

    # Log-variance shocks: Student-t with config df, scaled. Fat tails so
    # crash-like months occur and QLIKE differentials come out right-skewed.
    eta = config.LOGVAR_SHOCK_SCALE * rng.standard_t(config.SHOCK_T_DF, size=total)

    # Log-variance AR(1). Under the alternative, beta * x_{t-1} enters the
    # equation for log h_t (the one-period lag).
    use_predictor = world == "alt"
    log_h = np.empty(total)
    log_h[0] = config.LOGVAR_INTERCEPT + eta[0]
    for t in range(1, total):
        mean = config.LOGVAR_INTERCEPT + config.LOGVAR_PHI * log_h[t - 1] + eta[t]
        if use_predictor:
            mean = mean + beta * x[t - 1]
        log_h[t] = mean

    h = np.exp(log_h)

    # Realised variance: RV_t = h_t * u_t, with u_t a chi-square scaled to mean
    # 1 so the measurement noise is positive and unbiased for h_t.
    u = rng.chisquare(config.RV_NOISE_CHI2_DF, size=total) / config.RV_NOISE_CHI2_DF
    rv = h * u

    # Discard burn-in so the returned arrays begin in the stationary regime.
    sl = slice(config.BURN_IN, total)
    return {"h": h[sl], "rv": rv[sl], "x": x[sl]}


def _sanity_check():
    """Generate one null history with a fixed seed and print diagnostics."""
    t_total = config.T_TRAIN + config.T_OOS_GRID[0]
    rng = np.random.default_rng(config.MASTER_SEED)
    hist = generate_history("null", beta=0.0, t_total=t_total, rng=rng)

    rv = hist["rv"]
    log_rv = np.log(rv)
    lag1_acf = np.corrcoef(log_rv[:-1], log_rv[1:])[0, 1]

    print("DGP sanity check, null world, fixed seed")
    print("-" * 55)
    print(f"length of returned arrays={rv.size} (expected {t_total})")
    print(f"log RV mean={log_rv.mean():.4f}, std={log_rv.std():.4f}")
    print(f"log RV lag-1 autocorrelation={lag1_acf:.4f} (phi={config.LOGVAR_PHI})")
    print(f"RV skewness={stats.skew(rv):.4f}")

    assert np.all(rv > 0), "found non-positive RV"
    assert not np.any(np.isnan(rv)), "found NaN in RV"
    assert rv.size == t_total, "returned length does not equal t_total"
    print("assertions passed: all RV > 0, no NaNs, length equals t_total")


if __name__ == "__main__":
    _sanity_check()
