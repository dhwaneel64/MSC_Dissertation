"""Forecast evaluation, Clark-West statistic and inference for the study.

Runs the expanding walk-forward over the nested model pair, builds the
CW-adjusted QLIKE differential, forms the CW statistic, and returns two
p-values: the normal approximation and a circular block bootstrap.
"""

import warnings

import numpy as np
from scipy import stats

import config


def _ols_coef(y, X):
    """Ordinary least squares coefficients via least squares solve."""
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def walk_forward(rv, x, t_train):
    """Expanding walk-forward evaluation of the nested model pair.

    rv: realised variance series of length T_TRAIN + T_OOS.
    x: predictor series, same length.
    t_train: length of the initial training window.

    Both models regress log rv_{t+1} on predictors known at t and are fit in
    log space. The small model uses [const, log rv_t]; the big model adds x_t.
    At each step the window expands by one, both models refit, forecast the
    next month, and forecasts are exponentiated back to variance space before
    scoring with QLIKE per obs: rv/f - log(rv/f) - 1.

    Returns a dict with arrays qlike_small, qlike_big, f_small, f_big, each of
    length T_OOS = len(rv) - t_train.
    """
    n = rv.size
    log_rv = np.log(rv)

    n_oos = n - t_train
    qlike_small = np.empty(n_oos)
    qlike_big = np.empty(n_oos)
    f_small = np.empty(n_oos)
    f_big = np.empty(n_oos)

    # Forecast targets run from index t_train to n-1. For target tau the
    # predictors are the quantities dated tau-1, and the models are fit on all
    # pairs (log rv_s, predictors_{s-1}) for s from 1 to tau-1.
    for i, tau in enumerate(range(t_train, n)):
        y = log_rv[1:tau]                       # responses log rv_s, s = 1..tau-1
        const = np.ones(tau - 1)
        lag_lrv = log_rv[0:tau - 1]             # log rv_{s-1}
        lag_x = x[0:tau - 1]                    # x_{s-1}

        X_small = np.column_stack([const, lag_lrv])
        X_big = np.column_stack([const, lag_lrv, lag_x])

        coef_small = _ols_coef(y, X_small)
        coef_big = _ols_coef(y, X_big)

        # Predictors dated tau-1 used to forecast the target at tau.
        row_small = np.array([1.0, log_rv[tau - 1]])
        row_big = np.array([1.0, log_rv[tau - 1], x[tau - 1]])

        fs = np.exp(row_small @ coef_small)
        fb = np.exp(row_big @ coef_big)

        f_small[i] = fs
        f_big[i] = fb

        actual = rv[tau]
        qlike_small[i] = actual / fs - np.log(actual / fs) - 1.0
        qlike_big[i] = actual / fb - np.log(actual / fb) - 1.0

    # Forecasts come from exponentiation so they must be strictly positive.
    assert np.all(f_small > 0), "non-positive small-model forecast"
    assert np.all(f_big > 0), "non-positive big-model forecast"

    return {
        "qlike_small": qlike_small,
        "qlike_big": qlike_big,
        "f_small": f_small,
        "f_big": f_big,
    }


def cw_adjusted_diff(qlike_small, qlike_big, f_small, f_big, rv):
    """CW-adjusted QLIKE differential series, weighted form.

    Raw differential is qlike_small - qlike_big (positive when the big model
    wins). The Clark-West adjustment adds the estimation-noise penalty. Because
    both models are fit in log space, the forecast gap is taken on the
    log-forecast scale, g = log f_small - log f_big. Under QLIKE the penalty is
    weighted by the local QLIKE curvature rv/f_small rather than being a flat
    squared gap, so the adjustment is CW_ADJ_SCALE * (rv/f_small) * g^2. This is
    the settled form from diagnostic 3.

    rv is the full realised-variance series of length T_TRAIN + T_OOS; the OOS
    tail aligned with the forecasts is taken here.
    """
    raw = qlike_small - qlike_big
    g = np.log(f_small) - np.log(f_big)
    rv_scored = rv[rv.size - f_small.size:]      # OOS tail aligned with the forecasts
    # Weighted by the local QLIKE curvature rv/f_small; CW_ADJ_SCALE halves the
    # penalty because QLIKE curvature is half that of squared error.
    adjustment = config.CW_ADJ_SCALE * (rv_scored / f_small) * g ** 2
    return raw + adjustment


def _nw_lag(n):
    """Newey-West automatic lag: floor(coef * (n/base)^exp), the locked rule."""
    return int(np.floor(config.NW_LAG_COEF * (n / config.NW_LAG_BASE) ** config.NW_LAG_EXP))


def _nw_long_run_var(d, lag):
    """Newey-West long-run variance of d with Bartlett weights.

    LRV = gamma_0 + 2 * sum_{k=1}^{lag} (1 - k/(lag+1)) * gamma_k, where the
    autocovariances gamma_k use the 1/n normalisation. Bartlett weights make the
    estimate non-negative. lag = 0 returns the plain variance gamma_0.
    """
    n = d.size
    x = d - d.mean()
    lrv = np.dot(x, x) / n
    for k in range(1, lag + 1):
        wk = 1.0 - k / (lag + 1.0)
        gk = np.dot(x[k:], x[:-k]) / n
        lrv += 2.0 * wk * gk
    return lrv


def cw_stat(d_adj):
    """Clark-West statistic, HAC-studentised.

    Numerator is the mean of d_adj. The denominator is the standard error of the
    mean built from the Newey-West long-run variance, sqrt(LRV / n), with the lag
    set by the automatic rule in config. This absorbs the serial dependence of
    the differential into the denominator rather than leaving it to the block
    bootstrap alone.
    """
    n = d_adj.size
    lag = _nw_lag(n)
    lrv = _nw_long_run_var(d_adj, lag)
    return d_adj.mean() / np.sqrt(lrv / n)


def _fixed_block_indices(n, block_len, rng):
    """Indices for one fixed-length circular block bootstrap resample of length n.

    ceil(n/block_len) blocks are drawn at random start points, each wrapping
    modulo n, then concatenated and truncated to n.
    """
    n_blocks = int(np.ceil(n / block_len))
    offsets = np.arange(block_len)
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + offsets[None, :]).ravel() % n
    return idx[:n]


def _stationary_indices(n, expected_len, rng):
    """Indices for one stationary (Politis-Romano) bootstrap resample of length n.

    Block lengths are geometric with mean expected_len: at each step, with
    probability p = 1/expected_len a new block begins at a fresh random start,
    otherwise the index advances by one with circular wrap. Vectorised: coin
    flips mark block starts, a running maximum recovers each block's start
    position, and the offset within a block is added to that block's random base.
    """
    p = 1.0 / expected_len
    coin = rng.random(n) < p
    coin[0] = True                          # the first position always starts a block
    base_starts = rng.integers(0, n, size=n)

    pos = np.arange(n)
    start_pos = np.maximum.accumulate(np.where(coin, pos, 0))  # most recent block start at each t
    offset = pos - start_pos
    base = base_starts[start_pos]
    return (base + offset) % n


def _resample_indices(n, block_len, rng):
    """Resample indices under the scheme selected by config.BOOTSTRAP_SCHEME."""
    if config.BOOTSTRAP_SCHEME == "stationary":
        return _stationary_indices(n, block_len, rng)
    if config.BOOTSTRAP_SCHEME == "fixed":
        return _fixed_block_indices(n, block_len, rng)
    raise ValueError(f"unknown BOOTSTRAP_SCHEME {config.BOOTSTRAP_SCHEME!r}")


def bootstrap_pvalue(d_adj, s_obs, block_len, b_boot, rng):
    """Bootstrap p-value for the HAC-studentised CW statistic.

    The pool is centred at zero (subtract its mean) so the resamples represent
    the null. Each resample of length n is drawn under the scheme in config
    (stationary or fixed block, block_len being the expected/fixed length) and
    scored with the same HAC-studentised cw_stat, so the automatic NW lag is
    recomputed per resample from its length. The p-value is the fraction of
    bootstrap statistics at least as large as s_obs; s_obs is computed on the
    uncentred series and is never centred.

    Returns (p_value, boot_stat_std) where boot_stat_std is the standard
    deviation of the b_boot bootstrap statistics, a diagnostic of the resampled
    null's dispersion.
    """
    n = d_adj.size
    pool = d_adj - d_adj.mean()             # centre the pool, never s_obs

    boot_stats = np.empty(b_boot)
    for b in range(b_boot):
        idx = _resample_indices(n, block_len, rng)
        boot_stats[b] = cw_stat(pool[idx])

    return np.mean(boot_stats >= s_obs), boot_stats.std()


def data_bootstrap_pvalue(rv, x, t_train, block_len, b_boot, rng):
    """Data-level bootstrap p-value for the CW statistic, Calhoun (2015).

    Unlike bootstrap_pvalue, which resamples the loss differential series, this
    resamples the raw data and re-runs the whole forecasting pipeline on each
    fake sample, so the between-history variance that the loss-series bootstrap
    misses is reproduced. Steps per draw:

      1. Draw one set of stationary-bootstrap indices of length t_total (the full
         series length) with geometric mean block length block_len and circular
         wrap. The same index vector is applied to rv and x so the (rv_t, x_t)
         pairs stay together and their cross-dependence survives.
      2. Run the identical pipeline on the resampled series: walk_forward, the
         weighted CW adjustment, then the HAC-studentised cw_stat. Store the
         draw's statistic.

    Centring follows Calhoun: the reference distribution is the set of draw
    statistics recentred at their own mean, so the p-value is the fraction of
    draws with (stat_draw - mean(draws)) >= s_obs. The centre is the bootstrap
    mean, not zero and not the original-series mean.

    s_obs is recomputed here from the original rv, x through the same pipeline so
    it matches the draws exactly.

    Returns (p_value, draw_std) where draw_std is the standard deviation of the
    b_boot draw statistics.
    """
    n = rv.size

    wf0 = walk_forward(rv, x, t_train)
    d0 = cw_adjusted_diff(wf0["qlike_small"], wf0["qlike_big"],
                          wf0["f_small"], wf0["f_big"], rv)
    s_obs = cw_stat(d0)

    draw_stats = np.empty(b_boot)
    for b in range(b_boot):
        idx = _stationary_indices(n, block_len, rng)
        rv_b = rv[idx]
        x_b = x[idx]
        wf = walk_forward(rv_b, x_b, t_train)
        d_b = cw_adjusted_diff(wf["qlike_small"], wf["qlike_big"],
                               wf["f_small"], wf["f_big"], rv_b)
        draw_stats[b] = cw_stat(d_b)

    centred = draw_stats - draw_stats.mean()
    return np.mean(centred >= s_obs), draw_stats.std()


def _fit_ar1(series):
    """Fit an AR(1) with constant by OLS: series_{t+1} on [const, series_t].

    Returns a dict with the recursion intercept c, slope phi, and the demeaned
    residuals (the one-step innovations). Demeaning is enforced so resampled
    innovations carry no residual mean into the simulated recursion.
    """
    n = series.size
    y = series[1:]
    X = np.column_stack([np.ones(n - 1), series[:-1]])
    coef = _ols_coef(y, X)
    resid = y - X @ coef
    return {"c": coef[0], "phi": coef[1], "resid": resid - resid.mean()}


def _fit_arma11_logrv(log_rv):
    """Fit ARMA(1,1) with constant to log rv via statsmodels ARIMA(1,0,1).

    Returns a dict with the explicit-recursion intercept c, AR coefficient phi,
    MA coefficient theta, demeaned residuals, and a converged flag. statsmodels
    parameterises the constant as the process mean m (y_t = m + zero-mean ARMA),
    so the intercept of the AR-form recursion
    log rv_t = c + phi*log rv_{t-1} + eps_t + theta*eps_{t-1}
    that reproduces the same mean is c = m * (1 - phi). Any optimiser warnings are
    captured rather than printed; the converged flag reports non-convergence.
    """
    from statsmodels.tsa.arima.model import ARIMA

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res = ARIMA(log_rv, order=(1, 0, 1), trend="c").fit()

    mean = float(res.params[0])
    phi = float(res.arparams[0])
    theta = float(res.maparams[0])
    resid = np.asarray(res.resid, dtype=float)
    converged = bool(res.mle_retvals.get("converged", True)) \
        if isinstance(res.mle_retvals, dict) else True
    return {"c": mean * (1.0 - phi), "phi": phi, "theta": theta,
            "resid": resid - resid.mean(), "converged": converged}


def _resample_residuals(resid, n_draw, block_len, rng):
    """Draw n_draw innovations from the residual pool under the given scheme.

    block_len 1 is iid resampling with replacement, one residual at a time, which
    is the scheme the validated runs use. block_len above 1 draws
    ceil(n_draw / block_len) blocks of consecutive residuals at random start
    points, each wrapping circularly around the pool, concatenated and truncated
    to n_draw. Blocks carry whatever serial dependence the fitted model left in
    its residuals into the simulated path; iid resampling discards it.
    """
    m = resid.size
    if block_len == 1:
        return resid[rng.integers(0, m, size=n_draw)]
    n_blocks = int(np.ceil(n_draw / block_len))
    starts = rng.integers(0, m, size=n_blocks)
    idx = (starts[:, None] + np.arange(block_len)[None, :]).ravel() % m
    return resid[idx[:n_draw]]


def _simulate_ar1_path(gen, y0, n, rng, resid_block_len=1):
    """Simulate one AR(1) path of length n from a _fit_ar1 generator.

    Iterates y_t = c + phi*y_{t-1} + e_t from the observed initial value y0 with
    innovations resampled from the fitted residuals by _resample_residuals, drawn
    from the per-draw rng stream. resid_block_len 1 is the iid default.
    """
    e = _resample_residuals(gen["resid"], n - 1, resid_block_len, rng)
    out = np.empty(n)
    out[0] = y0
    c, phi = gen["c"], gen["phi"]
    for t in range(1, n):
        out[t] = c + phi * out[t - 1] + e[t - 1]
    return out


def _simulate_arma11_path(gen, y0, n, rng, resid_block_len=1):
    """Simulate one ARMA(1,1) path of length n from a _fit_arma11_logrv generator.

    Iterates the ARMA recursion explicitly,
    y_t = c + phi*y_{t-1} + eps_t + theta*eps_{t-1}, from the observed initial
    value y0 with innovations resampled by _resample_residuals. resid_block_len 1
    is the iid default. The pre-sample lagged innovation is set to zero. Not
    statsmodels' simulate(): the recursion is iterated here so the caller supplies
    the resampled shock sequence directly.
    """
    e = _resample_residuals(gen["resid"], n - 1, resid_block_len, rng)
    out = np.empty(n)
    out[0] = y0
    c, phi, theta = gen["c"], gen["phi"], gen["theta"]
    eps_prev = 0.0
    for t in range(1, n):
        out[t] = c + phi * out[t - 1] + e[t - 1] + theta * eps_prev
        eps_prev = e[t - 1]
    return out


def model_bootstrap_pvalue(rv, x, t_train, b_boot, rng,
                           generator_spec="arma", resid_block_len=1):
    """Model-based null-imposed bootstrap p-value for the CW statistic.

    This is the Kilian / Clark-McCracken class of bootstrap. Rather than
    resampling the observed data (data_bootstrap_pvalue) or the loss differential
    (bootstrap_pvalue), it fits a parametric null recursion to the observed
    history and simulates fresh fake histories from that recursion. The null is
    imposed by construction: the fake rv is generated from a model in which x
    plays no part, and fake x is generated independently, so under these fake
    worlds the big model can only add estimation noise. The draw statistics are
    therefore already null-world statistics and need no centring.

    Setup (once per call):
      - Null generator for rv: fit ARMA(1,1) with constant to log rv of the
        observed history (statsmodels ARIMA(1,0,1), trend="c"). Store the fitted
        c, phi, theta and the demeaned residuals. The ARMA(1,1) form is the A2
        amendment: the earlier plain AR(1) generator under-represented the
        short-run dynamics of log rv, so its fake histories had too little serial
        dependence. If the fit does not converge, or if the fitted |phi| >= 1
        (non-stationary, so the recursion would not have a stable mean), the
        generator falls back to a plain AR(1) fit on log rv, a warning is logged,
        and the reason is returned so the caller can count occurrences.
      - Fit x's own AR(1) on the observed x, unchanged from the A scheme.

    Per draw:
      - Simulate a fake log rv path of the full observed length by iterating the
        fitted ARMA recursion (or the AR(1) fallback) forward from the observed
        initial log rv value with iid-resampled residuals, then exponentiating to
        rv. Simulate a fake x path the same way from its fitted AR and
        iid-resampled residuals. rv and x are simulated independently because
        under the null x does not enter rv, so independence is the null.
      - Run the identical pipeline on the fake data: walk_forward, the weighted
        CW adjustment, then the HAC-studentised cw_stat. Store the draw statistic.

    All fitted parameters come from this observed history, never from the config
    DGP values, so the simulation follows the recursion's own estimated dynamics.
    s_obs is recomputed here from the original rv, x through the same pipeline so
    it matches the draws exactly. The p-value is the fraction of draw statistics
    at least as large as s_obs, with no centring.

    generator_spec selects the null generator for log rv: "arma" is the validated
    ARMA(1,1) described above, "ar1" forces the plain AR(1) with no ARMA fit
    attempted. The forced AR(1) is the mis-specified case the sensitivity runs
    report; it is not the same event as the ARMA fallback, which only fires when
    an attempted ARMA fit fails, so the two are distinguished in diag.

    resid_block_len selects how residuals are resampled inside both the rv and the
    x recursions: 1 is iid, the scheme the validated runs use, and any larger
    value draws circular blocks of that many consecutive residuals.

    Returns (p_value, draw_std, diag) where draw_std is the standard deviation of
    the b_boot draw statistics and diag is a dict {"generator", "reason"}:
    generator is "arma" or "ar1"; reason is None when the ARMA generator was used,
    "specified" when AR(1) was requested outright, and "nonconvergence" or
    "nonstationary" when an attempted ARMA fit fell back to AR(1).
    """
    n = rv.size
    log_rv = np.log(rv)

    # AR(1) generator on log rv, always fit so a fallback is ready and so the
    # forced-AR(1) specification has its generator without a second code path.
    ar_gen = _fit_ar1(log_rv)

    if generator_spec == "ar1":
        rv_gen = ar_gen
        generator = "ar1"
        reason = "specified"
    elif generator_spec != "arma":
        raise ValueError(f"unknown generator_spec {generator_spec!r}")
    else:
        generator = "arma"
        reason = None
        try:
            rv_gen = _fit_arma11_logrv(log_rv)
        except Exception as exc:
            rv_gen = ar_gen
            generator = "ar1"
            reason = "nonconvergence"
            warnings.warn(f"ARMA(1,1) fit raised {type(exc).__name__}, AR(1) fallback")
        else:
            if not rv_gen["converged"]:
                rv_gen = ar_gen
                generator = "ar1"
                reason = "nonconvergence"
                warnings.warn("ARMA(1,1) fit did not converge, AR(1) fallback")
            elif abs(rv_gen["phi"]) >= 1.0:
                rv_gen = ar_gen
                generator = "ar1"
                reason = "nonstationary"
                warnings.warn(f"ARMA(1,1) phi={rv_gen['phi']:.3f} not stationary, AR(1) fallback")

    # Predictor's own AR(1) generator, unchanged.
    x_gen = _fit_ar1(x)

    # Observed statistic through the same pipeline the draws use.
    wf0 = walk_forward(rv, x, t_train)
    d0 = cw_adjusted_diff(wf0["qlike_small"], wf0["qlike_big"],
                          wf0["f_small"], wf0["f_big"], rv)
    s_obs = cw_stat(d0)

    log_rv0 = log_rv[0]                      # observed initial value for the recursion
    x0 = x[0]

    draw_stats = np.empty(b_boot)
    for b in range(b_boot):
        if generator == "arma":
            log_rv_fake = _simulate_arma11_path(rv_gen, log_rv0, n, rng, resid_block_len)
        else:
            log_rv_fake = _simulate_ar1_path(rv_gen, log_rv0, n, rng, resid_block_len)
        rv_fake = np.exp(log_rv_fake)

        # The pipeline takes logs of rv, so the simulated rv must be strictly
        # positive and finite; exponentiation of an overflowing log path is the
        # only way this can fail.
        assert np.all(np.isfinite(rv_fake)) and np.all(rv_fake > 0), \
            "simulated fake rv not strictly positive and finite"

        x_fake = _simulate_ar1_path(x_gen, x0, n, rng, resid_block_len)

        wf = walk_forward(rv_fake, x_fake, t_train)
        d_b = cw_adjusted_diff(wf["qlike_small"], wf["qlike_big"],
                               wf["f_small"], wf["f_big"], rv_fake)
        draw_stats[b] = cw_stat(d_b)

    # No centring: the null is imposed by construction, so the draws are already
    # null-world statistics.
    diag = {"generator": generator, "reason": reason}
    return np.mean(draw_stats >= s_obs), draw_stats.std(), diag


def normal_pvalue(s_obs):
    """One-sided normal-approximation p-value for the CW statistic."""
    return stats.norm.sf(s_obs)


def _sanity_check():
    """One null history through the full pipeline with diagnostics."""
    import dgp

    t_oos = config.T_OOS_GRID[0]
    t_total = config.T_TRAIN + t_oos

    seed_seq = np.random.SeedSequence(config.MASTER_SEED)
    hist_seq, boot_seq = seed_seq.spawn(2)
    rng_hist = np.random.default_rng(hist_seq)
    rng_boot = np.random.default_rng(boot_seq)

    hist = dgp.generate_history("null", beta=0.0, t_total=t_total, rng=rng_hist)
    wf = walk_forward(hist["rv"], hist["x"], config.T_TRAIN)

    d_adj = cw_adjusted_diff(wf["qlike_small"], wf["qlike_big"],
                             wf["f_small"], wf["f_big"], hist["rv"])
    s_obs = cw_stat(d_adj)
    p_norm = normal_pvalue(s_obs)
    p_boot, boot_std = bootstrap_pvalue(d_adj, s_obs, config.BLOCK_BASELINE, config.B_BOOT, rng_boot)

    n_oos = d_adj.size
    print("forecast_eval.py sanity check, null world, fixed seed")
    print("-" * 55)
    print(f"n_oos={n_oos} (expected {t_oos})")
    print(f"d_adj mean={d_adj.mean():.6f}, skewness={stats.skew(d_adj):.4f}")
    print(f"s_obs={s_obs:.4f}")
    print(f"normal p-value={p_norm:.4f}")
    print(f"bootstrap p-value (block={config.BLOCK_BASELINE}, scheme={config.BOOTSTRAP_SCHEME})={p_boot:.4f}")
    print(f"bootstrap-statistic std={boot_std:.4f}")

    assert not np.any(np.isnan(d_adj)), "found NaN in d_adj"
    assert n_oos == t_oos, "n_oos does not equal T_OOS"
    print("assertions passed: no NaNs in d_adj, n_oos equals T_OOS")


if __name__ == "__main__":
    _sanity_check()
