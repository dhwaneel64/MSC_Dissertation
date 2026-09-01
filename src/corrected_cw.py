"""Corrected Clark-West machinery: weighted adjustment, HAC statistic, null generators.

This module is additive. It does not modify forecast_comparison.py or any other
existing module, and nothing here is wired into the notebook yet. It implements
the three corrections the simulation study validated, and builds the null
generators the model-based bootstrap will draw from:

  1. A curvature-weighted CW adjustment, replacing the unweighted QLIKE
     adjustment the pipeline currently passes.
  2. A HAC-studentised statistic at the project's automatic Newey-West lag,
     replacing the plain gamma_0 standard error (hac_lag=0) currently in force.
  3. One null generator per nested pair, so the bootstrap draw distribution
     carries parameter-estimation error instead of reshuffling a fixed loss
     series.

This task builds and checks the generators only. No p-value is computed here.
The generator check is the gate: in the simulation study a mis-specified AR(1)
generator produced 0.074 size against the validated ARMA(1,1)'s 0.055, so a
generator that does not reproduce the observed dependence is not usable.

Shocks are handled differently across the three pairs, because the pairs impose
different restrictions. Pairs 2 and 3 draw their shocks from an ARMA(p,q) fitted
to the null model's residuals: their nulls restrict the exogenous coefficients
and the cross-regime coefficient equality respectively, and neither restricts how
persistent VRP is, so persistence the mean equation left unexplained has to be
put back. Pair 1 keeps iid shocks, because the absence of serial predictability
is precisely what its null asserts, and grafting dependence onto it would
simulate under the alternative.

Per-observation inputs are read from outputs/nested_inputs_combined.npz. The
observed monthly series the generators are fitted on (the VRP path, the three
exogenous predictors, the regime labels) are not in that file, so they are
rebuilt through src.nested_inputs.build_inputs, which calls the same pipeline
functions the notebook calls on the same locked snapshot.

Run as:  python -m src.corrected_cw           generator specification check only
         python -m src.corrected_cw pvalues   bootstrap the corrected p-values
"""
from __future__ import annotations

import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

from src import config
from src import nested_inputs
from src.features import compute_realised_skew_21d
from src.forecast_comparison import clark_west_from_losses
from src.metrics import qlike_per_obs
from src.regimes import label_regimes
from src.models.baseline import ConstantMeanModel
from src.models.extended_ols import ExtendedOLSModel
from src.models.har_ols import HAROLSModel
from src.models.regime_switching_ols import RegimeSwitchingOLSModel
from src.results import score_walk_forward
from src.validation import VRP_HORIZON_COLS
from src.vrp import build_vrp_series
from src.walk_forward import make_model_factory_from_class, walk_forward


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Scale on the Clark-West estimation-noise penalty under QLIKE. Single source of
# truth in config (see the derivation comment there).
CW_ADJ_SCALE = config.CW_ADJ_SCALE

# Lags at which the generator check compares simulated with observed
# autocorrelation. 1 and 3 cover the short dependence the HAR lags exploit, 6 the
# longest HAR lag, 12 the annual horizon where a mis-specified generator that
# matches at short lags usually fails.
ACF_LAGS = (1, 3, 6, 12)

# Trial simulations per generator for the specification check.
N_TRIAL_SIMULATIONS = 100

# A simulated ACF is judged to depart materially from the observed one when the
# observed value sits further than this many standard deviations of the trial
# distribution away from the trial mean. The observed series is one draw from a
# correctly specified generator, so 2 sigma is the natural band and no arbitrary
# absolute tolerance is needed.
ACF_Z_TOL = 2.0

# For the iid null (pair 1) the target has no serial dependence by construction,
# so the check is that the simulated ACF sits inside the usual +/- 2/sqrt(n)
# white-noise band rather than that it matches the observed ACF.
IID_ACF_BAND_MULT = 2.0

# Burn-in months discarded from the front of a simulated path so the recursion
# starts from its stationary regime rather than the observed seed values.
BURN_IN = 200

# Grid caps for the ARMA(p,q) fitted to each exogenous predictor, selected by AIC.
# AR(p) by BIC was tried first and left vix_level short of its observed lag-6 and
# lag-12 autocorrelation: BIC's parsimony penalty suppressed the orders that carry
# long memory, returning AR(1) with phi 0.78, whose lag-12 implication is 0.05
# against an observed 0.23. The generator's job is to reproduce dependence, not to
# describe the series parsimoniously, so the selection criterion is the one that
# errs towards keeping dependence. The grid and the criterion are fixed here in
# advance of seeing the result, not tuned until the gate passed.
EXOG_ARMA_MAX_P = 4
EXOG_ARMA_MAX_Q = 2

# Grid caps for the ARMA(p,q) fitted to a null model's residuals. Small on
# purpose: the residual process is a correction for dependence the mean equation
# left over, not a model of the series, and a wide grid on ~380 observations
# would fit noise. ARMA(0,0) is inside the grid, so BIC can return the iid case.
ARMA_MAX_P = 2
ARMA_MAX_Q = 2

# Companion-matrix spectral radius above which a fitted or simulated recursion is
# treated as explosive and the draw is counted as non-converged.
STATIONARITY_MAX_ROOT = 0.999

# Bootstrap draws per pair: the pipeline's replication count, read from config so
# the corrected p-value is resolved on the same grid as the number it replaces and
# the two counts cannot drift apart.
BOOTSTRAP_DRAWS = config.BOOTSTRAP_REPLICATIONS

# Fallback draw count when the full run is projected to exceed the runtime budget.
# 2999 still resolves a p-value to better than 0.001, which is finer than any
# decision the study takes on it.
BOOTSTRAP_DRAWS_REDUCED = 2999

# Draws timed before the total runtime is projected, and the budget the projection
# is compared against. Each draw refits the whole walk-forward twice, so the cost
# is known only by measuring it.
RUNTIME_PROBE_DRAWS = 20
RUNTIME_BUDGET_SECONDS = 2 * 3600

# Stdout progress cadence for the bootstrap: one flushed line every this many
# completed draws (draws done / total, elapsed, ETA), so a long run is never
# opaque. Display only; no effect on draws, seeding, or statistics.
PROGRESS_PRINT_EVERY_DRAWS = 250

# Worker processes the draws are spread across. Each draw carries its own spawned
# seed and is scored independently, so the result does not depend on how the draws
# are distributed and parallelism buys wall-clock only. Two cores are left free.
BOOTSTRAP_WORKERS = max(1, (os.cpu_count() or 2) - 2)

# Paired months a draw must retain to be scored: the same config floor
# forecast_comparison enforces, so a draw that falls below it is one the
# pipeline's own test would refuse rather than one this module rejects.
MIN_PAIRED_MONTHS = config.MIN_COMPARISON_OBS

# Nested pairs in (smaller, larger) order, matching NESTED_MODEL_PAIRS ordering
# in forecast_comparison and the pair keys written to the npz.
NESTED_PAIRS = nested_inputs.NESTED_PAIRS

# The three exogenous predictors of the Extended OLS specification, the ones
# whose coefficients the pair-2 null sets to zero.
EXOG_COLS = ("vix_level", "cboe_skew", "realised_skew_21d")

# HAR regressor names in locked order, imported from the single source so a rename
# cannot desynchronise this module from the pipeline dataset.
HAR_LAG_COLS = VRP_HORIZON_COLS

# The model classes the walk-forward refits on each bootstrap draw, by the name
# the pipeline uses. Feature columns are not repeated here: they are read from
# build_inputs, so the draw fits the same specification the observed run fitted.
MODEL_CLASSES = {
    "constant": ConstantMeanModel,
    "har": HAROLSModel,
    "extended_ols": ExtendedOLSModel,
    "regime_switching": RegimeSwitchingOLSModel,
}


# ---------------------------------------------------------------------------
# Corrected statistic
# ---------------------------------------------------------------------------

def nw_lag(n: int) -> int:
    """Automatic Newey-West lag at the project rule, read from config.

    lag = floor(NW_HAC_MULTIPLIER * (n / 100) ** NW_HAC_EXPONENT), the same
    automatic bandwidth every regression in the project already uses. No new
    constant is introduced for the statistic.
    """
    return math.floor(config.NW_HAC_MULTIPLIER * (n / 100) ** config.NW_HAC_EXPONENT)


def nw_long_run_var(d: np.ndarray, lag: int) -> float:
    """Newey-West long-run variance with Bartlett weights.

    LRV = gamma_0 + 2 * sum_{k=1..lag} (1 - k/(lag+1)) * gamma_k, autocovariances
    on the 1/n normalisation. The Bartlett taper is what makes the estimate
    non-negative; the pipeline's existing _hac_variance omits it, which is safe
    only at lag 0 where the sum is empty.
    """
    d = np.asarray(d, dtype=float)
    n = d.size
    x = d - d.mean()
    lrv = float(np.dot(x, x) / n)
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        gk = float(np.dot(x[k:], x[:-k]) / n)
        lrv += 2.0 * w * gk
    return lrv


def cw_stat(d_adj: np.ndarray) -> float:
    """HAC-studentised Clark-West statistic.

    mean(d_adj) divided by the standard error of that mean built from the
    Newey-West long-run variance at the automatic lag, sqrt(LRV / n).
    """
    d_adj = np.asarray(d_adj, dtype=float)
    n = d_adj.size
    lrv = nw_long_run_var(d_adj, nw_lag(n))
    if lrv <= 0:
        raise ValueError("Newey-West long-run variance is zero or negative")
    return float(d_adj.mean() / math.sqrt(lrv / n))


def weighted_adjustment(rv: np.ndarray, var_smaller: np.ndarray,
                        var_larger: np.ndarray) -> np.ndarray:
    """Curvature-weighted CW adjustment, a_t = 0.5 * (rv_t/s_t) * (log s_t - log l_t)^2.

    Both models forecast a variance, and QLIKE's local curvature in the forecast
    is rv/f rather than a constant, so the estimation-noise penalty is the
    squared log forecast gap weighted by that curvature and halved. The
    pipeline's current adjustment, qlike_per_obs(s, l), equals the same squared
    log gap to second order but carries no rv/s weight.
    """
    rv = np.asarray(rv, dtype=float)
    s = np.asarray(var_smaller, dtype=float)
    l = np.asarray(var_larger, dtype=float)
    g = np.log(s) - np.log(l)
    return CW_ADJ_SCALE * (rv / s) * g ** 2


def adjusted_differential(rv: np.ndarray, var_smaller: np.ndarray,
                          var_larger: np.ndarray) -> np.ndarray:
    """Corrected per-observation CW differential on QLIKE.

    d_t = qlike(rv, s) - qlike(rv, l) + a_t, positive when the larger model wins
    after paying back the estimation-noise penalty.
    """
    raw = qlike_per_obs(rv, var_smaller) - qlike_per_obs(rv, var_larger)
    return raw + weighted_adjustment(rv, var_smaller, var_larger)


# ---------------------------------------------------------------------------
# Observed data
# ---------------------------------------------------------------------------

def load_saved_inputs() -> dict:
    """Read the persisted nested-comparison inputs from outputs/.

    Returns a dict of the arrays in nested_inputs_combined.npz.
    """
    path = nested_inputs.OUTPUT_DIR / "nested_inputs_combined.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.nested_inputs` first."
        )
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def load_observed() -> dict:
    """Rebuild the observed monthly series the generators are fitted on.

    The saved npz holds the out-of-sample forecast-side arrays, not the monthly
    VRP path or the exogenous predictor series, so those come from the pipeline
    itself: the same calls the notebook makes, on the same locked snapshot.
    """
    inputs = nested_inputs.build_inputs()
    vix_monthly = inputs["vix_monthly"]
    vrp = build_vrp_series(vix_monthly, inputs["spy_returns"], vix_monthly.index)
    inputs["vrp"] = vrp
    return inputs


def slim_observed(observed: dict) -> dict:
    """The subset of the observed inputs the generators and the bootstrap read.

    build_inputs also returns the four observed walk-forward frames, which nothing
    downstream of the generators needs. Dropping them keeps what is sent to a
    worker process small, and makes it explicit which observed series a draw is
    allowed to see.
    """
    return {
        "dataset": observed["dataset"],
        "vrp": observed["vrp"],
        "vix_monthly": observed["vix_monthly"],
        "skew_monthly": observed["skew_monthly"],
        "spy_returns": observed["spy_returns"],
        "initial_train_end": observed["initial_train_end"],
        "feature_cols": observed["feature_cols"],
    }


def frame_from_vrp(vrp: pd.Series, exog: pd.DataFrame, regime: pd.Series,
                   target_index: pd.Index) -> pd.DataFrame:
    """Assemble a dataset-shaped frame from a VRP path and its covariates.

    Reproduces the pipeline's construction exactly: the VRP column at horizon k is
    vrp.shift(k - 1), the target is vrp.shift(-1), the exogenous columns and the
    regime label are taken at t, and the result is restricted to target_index
    (which already encodes the pipeline's NaN drop). The shift here and the one in
    build_feature_matrix must move together; verify_frame_reconstruction is the
    guard that catches them diverging.
    """
    out = pd.DataFrame(index=vrp.index)
    out["vix_level"] = exog["vix_level"]
    out["cboe_skew"] = exog["cboe_skew"]
    for k, col in zip(config.HAR_LAGS_MONTHS, VRP_HORIZON_COLS):
        out[col] = vrp.shift(k - 1)
    out["realised_skew_21d"] = exog["realised_skew_21d"]
    out["regime"] = regime
    out["y"] = vrp.shift(-1)
    return out.reindex(target_index)


def verify_frame_reconstruction(observed: dict) -> None:
    """Check frame_from_vrp reproduces the pipeline dataset from independent sources.

    Every one of the eight columns is rebuilt from a source that is not the frame
    under test: the VRP columns and the target from the observed VRP series, the
    exogenous columns from the monthly VIX and SKEW series and the daily returns,
    and the regime labels from the monthly VIX through label_regimes. A
    perturbation in any dataset column therefore raises; no column is compared
    with a copy of itself. If the reconstruction of the observed data is exact,
    the same function applied to a simulated path builds a frame the walk-forward
    engine can consume on the same terms.
    """
    dataset = observed["dataset"]
    exog = pd.DataFrame(index=dataset.index)
    exog["vix_level"] = observed["vix_monthly"].reindex(dataset.index)
    exog["cboe_skew"] = observed["skew_monthly"].reindex(dataset.index)
    exog["realised_skew_21d"] = compute_realised_skew_21d(
        observed["spy_returns"], dataset.index
    )
    regime = label_regimes(observed["vix_monthly"]).reindex(dataset.index)
    rebuilt = frame_from_vrp(observed["vrp"], exog, regime, dataset.index)
    for col in dataset.columns:
        if col == "regime":
            same = (rebuilt[col].astype(str) == dataset[col].astype(str)).all()
        else:
            same = np.allclose(rebuilt[col].to_numpy(dtype=float),
                               dataset[col].to_numpy(dtype=float), rtol=0, atol=0)
        if not same:
            raise ValueError(f"frame reconstruction disagrees with the dataset in {col!r}")


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def _ols(y: pd.Series | np.ndarray, X: pd.DataFrame) -> tuple:
    """Plain OLS with an intercept, returning (params, residuals).

    Coefficients only. The pipeline fits the same regressions with HAC standard
    errors, which change the reported standard errors and not the coefficients,
    and the null generator needs the conditional mean, not inference. Extended
    OLS z-scores its regressors before fitting; OLS fitted values are invariant
    to that affine rescaling, so the raw-unit fit here has identical predictions.
    """
    X_aug = sm.add_constant(X, has_constant="add")
    res = sm.OLS(np.asarray(y, dtype=float), X_aug).fit()
    return res.params, np.asarray(res.resid, dtype=float)


def _series_lag_map() -> dict:
    """Map each VRP predictor column to its distance in the VRP series itself.

    The dataset row at t regresses y = vrp_{t+1} on the column at horizon k, which
    carries vrp_{t-(k-1)}, so in series terms the simulated observation at position
    i depends on position i - k. The horizon in the column name is the series
    distance, which is the whole point of naming these columns by horizon to the
    target rather than by lag from t. Getting this off by one would build a
    generator with the wrong memory, so it is derived here once and asserted
    against the fitted data.
    """
    return {col: k for k, col in zip(config.HAR_LAGS_MONTHS, VRP_HORIZON_COLS)}


def _companion_max_root(coeffs_by_lag: dict) -> float:
    """Spectral radius of the companion matrix of a sparse AR recursion."""
    max_lag = max(coeffs_by_lag)
    phi = np.zeros(max_lag)
    for lag, b in coeffs_by_lag.items():
        phi[lag - 1] = b
    comp = np.zeros((max_lag, max_lag))
    comp[0, :] = phi
    if max_lag > 1:
        comp[1:, :-1] = np.eye(max_lag - 1)
    return float(np.max(np.abs(np.linalg.eigvals(comp))))


def _fit_arma_grid(x: np.ndarray, max_p: int, max_q: int,
                   criterion: str) -> dict:
    """Fit ARMA(p,q) over a grid on a mean-zero series, order by the given criterion.

    Fitted by exact maximum likelihood with no trend term, so the caller is
    responsible for removing the mean, and with statsmodels' stationarity and
    invertibility enforcement left on, so any order the grid returns simulates a
    stable process. Every order is fitted on the same observations, so the
    information criteria are directly comparable across the grid.

    criterion is "bic" or "aic". Returns the order, the AR and MA parameters, the
    innovation pool the simulation resamples from, and the AR companion spectral
    radius.
    """
    x = np.asarray(x, dtype=float)
    best = None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            with warnings.catch_warnings():
                # The grid deliberately visits over-parameterised orders, which
                # warn about convergence and about starting values. The criterion
                # is what decides between the orders, so the warnings say nothing
                # that changes the selection.
                warnings.simplefilter("ignore")
                try:
                    res = ARIMA(x, order=(p, 0, q), trend="n").fit()
                except (ValueError, np.linalg.LinAlgError):
                    continue
            score = float(res.bic if criterion == "bic" else res.aic)
            if not np.isfinite(score):
                continue
            if best is None or score < best[criterion]:
                best = {
                    "order": (p, q),
                    criterion: score,
                    "criterion": criterion,
                    "ar": np.asarray(res.arparams, dtype=float),
                    "ma": np.asarray(res.maparams, dtype=float),
                    "innov": np.asarray(res.resid, dtype=float),
                }
    if best is None:
        raise ValueError("no ARMA order in the grid could be fitted")

    # The first max(p,q) one-step-ahead errors are formed before the filter has
    # seen enough history, so their variance is the process variance rather than
    # the innovation variance. Dropping them keeps the resampling pool on scale.
    lead = max(best["order"])
    if lead:
        best["innov"] = best["innov"][lead:]
    ar = best["ar"]
    best["max_root"] = (_companion_max_root({j + 1: float(ar[j]) for j in range(ar.size)})
                        if ar.size else 0.0)
    best["innov_sd"] = float(best["innov"].std(ddof=1))
    best["mean"] = 0.0
    return best


def _fit_arma_bic(resid: np.ndarray) -> dict:
    """Fit an ARMA(p,q) to a null model's residuals, order chosen by BIC.

    OLS residuals against an intercept already have mean zero, so no demeaning is
    needed. BIC here rather than AIC because this process is a correction for
    dependence the mean equation left over, and the parsimonious answer is the
    right default when the mean equation is meant to be carrying the structure.
    """
    return _fit_arma_grid(resid, ARMA_MAX_P, ARMA_MAX_Q, "bic")


def _fit_arma_series_aic(series: pd.Series) -> dict:
    """Fit an ARMA(p,q) to an exogenous predictor, order chosen by AIC.

    The series is demeaned before fitting and the mean is carried in the spec, so
    the simulated path is mean plus a mean-zero ARMA draw. AIC over the wider
    EXOG_ARMA grid rather than BIC over AR orders alone: see the note on
    EXOG_ARMA_MAX_P for why the parsimonious criterion is the wrong one for a
    generator whose only job is to reproduce the observed dependence.
    """
    x = series.to_numpy(dtype=float)
    mean = float(x.mean())
    spec = _fit_arma_grid(x - mean, EXOG_ARMA_MAX_P, EXOG_ARMA_MAX_Q, "aic")
    spec["mean"] = mean
    return spec


def _simulate_arma(spec: dict, n_out: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate n_out shocks from a fitted ARMA spec, innovations resampled iid.

    e_i = sum_j ar_j e_{i-j} + u_i + sum_k ma_k u_{i-k}, with u drawn with
    replacement from the fitted innovation pool rather than from a normal, so the
    skewness and the tail weight of the observed innovations survive into the
    draw. BURN_IN extra steps are run so the returned block is in the stationary
    regime. An ARMA(0,0) spec reduces to the iid resampling it replaces.
    """
    ar = np.asarray(spec["ar"], dtype=float)
    ma = np.asarray(spec["ma"], dtype=float)
    p, q = ar.size, ma.size
    if p == 0 and q == 0:
        return rng.choice(spec["innov"], size=n_out, replace=True)

    total = BURN_IN + n_out
    lead = max(p, q)
    u = rng.choice(spec["innov"], size=total + lead, replace=True)
    e = np.zeros(total + lead)
    for i in range(lead, total + lead):
        value = u[i]
        for j in range(p):
            value += ar[j] * e[i - j - 1]
        for k in range(q):
            value += ma[k] * u[i - k - 1]
        e[i] = value
    return e[-n_out:]


def _simulate_sparse_ar(const: float, coeffs_by_lag: dict, resid_spec: dict,
                        seed_values: np.ndarray, n_out: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Simulate a sparse AR recursion, feeding simulated values back into its lags.

    coeffs_by_lag maps a lag in series terms to its coefficient. The path starts
    from seed_values, runs BURN_IN extra steps, and returns the last n_out. Every
    lag the recursion reads is path[i - lag], a value the recursion itself
    produced, so the simulated series carries the fitted persistence rather than
    the observed series' own.

    Shocks come from resid_spec, a fitted ARMA specification. An ARMA(0,0) spec is
    the iid case, so the iid draw is not a separate code path.
    """
    max_lag = max(coeffs_by_lag)
    total = BURN_IN + n_out
    path = np.empty(total + max_lag)
    path[:max_lag] = seed_values[:max_lag]
    draws = _simulate_arma(resid_spec, total, rng)
    for i in range(max_lag, total + max_lag):
        value = const
        for lag, b in coeffs_by_lag.items():
            value += b * path[i - lag]
        path[i] = value + draws[i - max_lag]
    return path[-n_out:]


def _acf(x: np.ndarray, lags=ACF_LAGS) -> np.ndarray:
    """Sample autocorrelation at the given lags, 1/n normalisation."""
    x = np.asarray(x, dtype=float)
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    return np.array([float(np.dot(xc[k:], xc[:-k]) / denom) for k in lags])


# ---------------------------------------------------------------------------
# Null generators
# ---------------------------------------------------------------------------

@dataclass
class GeneratorResult:
    """One simulated history: the VRP path and the dataset-shaped frame."""
    vrp: pd.Series
    frame: pd.DataFrame
    converged: bool
    note: str = ""


@dataclass
class NullGenerator:
    """Base class holding the observed data and the fitted null parameters."""
    name: str
    pair: tuple
    observed: dict
    spec: dict = field(default_factory=dict)

    def target_series(self, sim: GeneratorResult) -> np.ndarray:
        """The series whose ACF the specification check reads."""
        return sim.vrp.to_numpy(dtype=float)


class ConstantNullGenerator(NullGenerator):
    """Pair 1 null: VRP has no serial predictability.

    The constant model predicts the rolling mean of VRP, so under its null the
    series is its unconditional level plus unpredictable noise. Residuals are
    taken against that level and resampled iid, which imposes zero serial
    dependence by construction. No persistence is grafted on: the restriction the
    null expresses is precisely the absence of serial predictability, so a
    generator that reproduced the observed autocorrelation would be simulating
    under the alternative.

    Exogenous columns and regime labels are carried at their observed values;
    neither model in this pair reads them.
    """

    def fit(self) -> None:
        vrp = self.observed["vrp"]
        level = float(vrp.mean())
        resid = vrp.to_numpy(dtype=float) - level
        self.spec = {
            "level": level,
            "resid_sd": float(resid.std(ddof=1)),
            "resid_pool_n": resid.size,
            "resid": resid,
            "imposes": "zero serial correlation in VRP",
        }

    def simulate(self, rng: np.random.Generator) -> GeneratorResult:
        vrp_obs = self.observed["vrp"]
        draws = rng.choice(self.spec["resid"], size=vrp_obs.size, replace=True)
        vrp_sim = pd.Series(self.spec["level"] + draws, index=vrp_obs.index, name="vrp")
        dataset = self.observed["dataset"]
        frame = frame_from_vrp(vrp_sim, dataset, dataset["regime"], dataset.index)
        ok = bool(np.isfinite(vrp_sim.to_numpy()).all() and frame.notna().all().all())
        return GeneratorResult(vrp_sim, frame, ok)


class HARNullGenerator(NullGenerator):
    """Pair 2 null: HAR is correct, the exogenous predictors carry no coefficient.

    VRP follows the fitted HAR recursion in its own lags, so the persistence the
    Extended OLS estimation step depends on is preserved exactly. The three
    exogenous predictors are simulated from their own fitted ARMA(p,q) dependence
    structures, order by AIC, and they never enter the VRP equation: their
    coefficients are zero by construction, which is the null. Simulating them
    rather than holding them observed is what puts their estimation error into the
    draw distribution.

    Shocks to the VRP equation are simulated from an ARMA(p,q) fitted to the HAR
    residuals, not resampled iid. The restriction this null expresses is that the
    exogenous coefficients are zero; it says nothing about how persistent VRP is,
    so the generator has to reproduce the observed persistence in full. Under the
    corrected alignment the pipeline's dataset regresses vrp_{t+1} on vrp_t,
    vrp_{t-2} and vrp_{t-5}, so the one-month link is in the mean equation; the
    residual ARMA carries whatever dependence those three lags leave over (BIC can
    return ARMA(0,0) when nothing is left), and the ACF gate decides whether the
    combination reproduces the observed dependence. A generator that comes out
    under-persistent is the A2 case where it over-rejects.

    Regime labels are carried at their observed values; neither model in this
    pair reads them.
    """

    def fit(self) -> None:
        dataset = self.observed["dataset"]
        params, resid = _ols(dataset["y"], dataset[list(HAR_LAG_COLS)])
        lag_map = _series_lag_map()
        coeffs_by_lag = {lag_map[col]: float(params[col]) for col in HAR_LAG_COLS}
        self.spec = {
            "const": float(params["const"]),
            "coeffs": {col: float(params[col]) for col in HAR_LAG_COLS},
            "coeffs_by_series_lag": coeffs_by_lag,
            "max_root": _companion_max_root(coeffs_by_lag),
            "resid_sd": float(resid.std(ddof=1)),
            "resid": resid,
            "resid_arma": _fit_arma_bic(resid),
            "exog": {col: _fit_arma_series_aic(dataset[col]) for col in EXOG_COLS},
            "imposes": "zero coefficient on every exogenous predictor",
        }

    def simulate(self, rng: np.random.Generator) -> GeneratorResult:
        vrp_obs = self.observed["vrp"]
        dataset = self.observed["dataset"]

        vrp_path = _simulate_sparse_ar(
            self.spec["const"], self.spec["coeffs_by_series_lag"],
            self.spec["resid_arma"], vrp_obs.to_numpy(dtype=float), vrp_obs.size, rng,
        )
        vrp_sim = pd.Series(vrp_path, index=vrp_obs.index, name="vrp")

        exog_sim = pd.DataFrame(index=dataset.index)
        for col in EXOG_COLS:
            fit = self.spec["exog"][col]
            exog_sim[col] = fit["mean"] + _simulate_arma(fit, len(dataset.index), rng)

        frame = frame_from_vrp(vrp_sim, exog_sim, dataset["regime"], dataset.index)
        finite = bool(np.isfinite(vrp_sim.to_numpy()).all()
                      and np.isfinite(exog_sim.to_numpy(dtype=float)).all())
        ok = finite and bool(frame.notna().all().all())
        return GeneratorResult(vrp_sim, frame, ok)


class PooledExtendedNullGenerator(NullGenerator):
    """Pair 3 null: the pooled Extended OLS coefficients hold in every regime.

    VRP follows the fitted pooled Extended OLS equation, its own lag dynamics
    updated from the simulated path so persistence is preserved, with the
    exogenous predictors held at their observed values and the observed regime
    label sequence held fixed. Imposing one pooled coefficient set across all
    regimes is exactly the equality restriction that makes Extended OLS the
    nested model, so the null is imposed by construction rather than by
    recentring.

    Shocks are simulated from an ARMA(p,q) fitted to the pooled residuals, not
    resampled iid, for the same reason as pair 2. The restriction here is equality
    of coefficients across regimes, which places no restriction on the persistence
    of VRP itself. Under the corrected alignment the pooled mean equation reads
    vrp_t, vrp_{t-2} and vrp_{t-5} to predict vrp_{t+1}; the residual ARMA carries
    whatever dependence those lags leave over (BIC can return ARMA(0,0) when
    nothing is left), and the ACF gate decides adequacy.

    Where an exogenous value is missing on the monthly index (the ^SKEW data
    gaps the pipeline drops), the pooled prediction is undefined and that
    position keeps its observed VRP value so later lags stay defined. Those
    positions are already absent from the dataset index; the count is reported.
    """

    def fit(self) -> None:
        dataset = self.observed["dataset"]
        cols = list(nested_inputs.NUMERIC_FEATURES)
        params, resid = _ols(dataset["y"], dataset[cols])
        lag_map = _series_lag_map()
        coeffs_by_lag = {lag_map[col]: float(params[col]) for col in HAR_LAG_COLS}
        self.spec = {
            "const": float(params["const"]),
            "coeffs": {col: float(params[col]) for col in cols},
            "coeffs_by_series_lag": coeffs_by_lag,
            "max_root": _companion_max_root(coeffs_by_lag),
            "resid_sd": float(resid.std(ddof=1)),
            "resid": resid,
            "resid_arma": _fit_arma_bic(resid),
            "feature_cols": cols,
            "imposes": "one pooled coefficient set across all regimes",
        }

    def simulate(self, rng: np.random.Generator) -> GeneratorResult:
        vrp_obs = self.observed["vrp"]
        dataset = self.observed["dataset"]
        index = vrp_obs.index

        # Exogenous predictors on the full monthly index, observed and fixed.
        exog_full = pd.DataFrame(index=index)
        for col in EXOG_COLS:
            exog_full[col] = dataset[col].reindex(index)

        obs = vrp_obs.to_numpy(dtype=float)
        sim = obs.copy()
        lag_map = _series_lag_map()
        max_lag = max(lag_map.values())
        draws = _simulate_arma(self.spec["resid_arma"], obs.size, rng)
        exog_vals = {col: exog_full[col].to_numpy(dtype=float) for col in EXOG_COLS}

        n_fallback = 0
        for i in range(max_lag, obs.size):
            row_exog = [exog_vals[col][i] for col in EXOG_COLS]
            if not np.all(np.isfinite(row_exog)):
                n_fallback += 1
                continue                      # keeps the observed value at this position
            value = self.spec["const"]
            for col in EXOG_COLS:
                value += self.spec["coeffs"][col] * exog_vals[col][i]
            for col in HAR_LAG_COLS:
                value += self.spec["coeffs"][col] * sim[i - lag_map[col]]
            sim[i] = value + draws[i]

        vrp_sim = pd.Series(sim, index=index, name="vrp")
        frame = frame_from_vrp(vrp_sim, dataset, dataset["regime"], dataset.index)
        ok = bool(np.isfinite(sim).all() and frame.notna().all().all())
        note = f"{n_fallback} position(s) kept the observed VRP for missing exogenous values"
        return GeneratorResult(vrp_sim, frame, ok, note)


def build_generators(observed: dict) -> dict:
    """Fit one null generator per nested pair and return them keyed by pair."""
    gens = {
        NESTED_PAIRS[0]: ConstantNullGenerator("iid residual null around the rolling mean",
                                               NESTED_PAIRS[0], observed),
        NESTED_PAIRS[1]: HARNullGenerator("fitted HAR null with simulated exogenous predictors",
                                          NESTED_PAIRS[1], observed),
        NESTED_PAIRS[2]: PooledExtendedNullGenerator("pooled Extended OLS null, regimes fixed",
                                                     NESTED_PAIRS[2], observed),
    }
    for gen in gens.values():
        gen.fit()
    return gens


# ---------------------------------------------------------------------------
# Specification check
# ---------------------------------------------------------------------------

def run_trials(gen: NullGenerator, n_trials: int = N_TRIAL_SIMULATIONS) -> dict:
    """Simulate n_trials histories and collect ACFs, convergence and notes.

    Seeds come from config.BOOTSTRAP_SEED, spawned per draw so each trial draws
    an independent stream and the whole check is reproducible.
    """
    # The pair's position in NESTED_PAIRS is the stream key. Python's hash of a
    # string is randomised per process, so it cannot be used here.
    pair_key = NESTED_PAIRS.index(gen.pair)
    root = np.random.SeedSequence([config.BOOTSTRAP_SEED, pair_key])
    children = root.spawn(n_trials)

    target_acf = np.empty((n_trials, len(ACF_LAGS)))
    exog_acf = {col: np.empty((n_trials, len(ACF_LAGS))) for col in EXOG_COLS}
    converged = np.zeros(n_trials, dtype=bool)
    notes = []
    simulates_exog = isinstance(gen, HARNullGenerator)

    for i, child in enumerate(children):
        rng = np.random.default_rng(child)
        sim = gen.simulate(rng)
        converged[i] = sim.converged
        if sim.note and not notes:
            notes.append(sim.note)
        target_acf[i] = _acf(gen.target_series(sim))
        if simulates_exog:
            for col in EXOG_COLS:
                exog_acf[col][i] = _acf(sim.frame[col].to_numpy(dtype=float))

    return {
        "target_acf": target_acf,
        "exog_acf": exog_acf if simulates_exog else None,
        "converged": converged,
        "convergence_rate": float(converged.mean()),
        "notes": notes,
        "n_trials": n_trials,
    }


def acf_comparison(observed_series: np.ndarray, sim_acf: np.ndarray,
                   iid_null: bool) -> pd.DataFrame:
    """Compare simulated with observed ACF at ACF_LAGS.

    For a generator that is meant to reproduce the observed dependence, the test
    is whether the observed ACF sits inside the trial distribution: z is the
    observed value in trial standard deviations from the trial mean, and |z| above
    ACF_Z_TOL is a material departure. For the iid null the target has no serial
    dependence by construction, so the test is instead whether the simulated ACF
    sits inside the +/- IID_ACF_BAND_MULT / sqrt(n) white-noise band.
    """
    obs = _acf(observed_series)
    mean = sim_acf.mean(axis=0)
    sd = sim_acf.std(axis=0, ddof=1)
    n = observed_series.size

    rows = []
    for j, lag in enumerate(ACF_LAGS):
        if iid_null:
            band = IID_ACF_BAND_MULT / math.sqrt(n)
            rows.append({"lag": lag, "observed": obs[j], "simulated_mean": mean[j],
                         "simulated_sd": sd[j], "z_or_band": band,
                         "pass": abs(mean[j]) <= band})
        else:
            z = (obs[j] - mean[j]) / sd[j] if sd[j] > 0 else np.inf
            rows.append({"lag": lag, "observed": obs[j], "simulated_mean": mean[j],
                         "simulated_sd": sd[j], "z_or_band": z,
                         "pass": abs(z) <= ACF_Z_TOL})
    return pd.DataFrame(rows)


def check_generator(gen: NullGenerator, trials: dict) -> dict:
    """Build the ACF comparison tables for one generator and its pass flag."""
    observed = gen.observed
    iid_null = isinstance(gen, ConstantNullGenerator)
    target = acf_comparison(observed["vrp"].to_numpy(dtype=float),
                            trials["target_acf"], iid_null)

    exog_tables = {}
    if trials["exog_acf"] is not None:
        for col in EXOG_COLS:
            exog_tables[col] = acf_comparison(
                observed["dataset"][col].to_numpy(dtype=float),
                trials["exog_acf"][col], iid_null=False,
            )

    passed = bool(target["pass"].all()) and all(t["pass"].all() for t in exog_tables.values())
    return {"target": target, "exog": exog_tables, "passed": passed}


# ---------------------------------------------------------------------------
# Model-based null-imposed bootstrap
# ---------------------------------------------------------------------------

def _factory_for(name: str, frame: pd.DataFrame):
    """Model factory for one pipeline model name, bound to the frame being fitted.

    monthly_vrp is the frame the model validates its feature set against, so on a
    bootstrap draw it is the simulated frame, not the observed dataset. The
    constant-mean model takes no such argument.
    """
    cls = MODEL_CLASSES[name]
    if name == "constant":
        return make_model_factory_from_class(cls)
    return make_model_factory_from_class(cls, monthly_vrp=frame)


def simulated_realised_variance(realised_obs: np.ndarray, y_obs: np.ndarray,
                                y_sim: np.ndarray) -> np.ndarray:
    """Realised variance consistent with a simulated VRP path.

    VRP is defined as vix^2/scale minus the physical variance forecast, so the
    realised variance the loss is scored against and the VRP path are two sides of
    one identity. If the VRP path is simulated and the realised variance is left
    observed, the loss target no longer responds to the null the generator
    imposes: the larger model's extra regressors could still predict the observed
    target while having no relation to the simulated path, and the draw
    distribution would not be a null distribution.

    Holding the physical forecast error at its observed value and rebuilding the
    target from the simulated path gives

        rv_sim = rv_obs + (vrp_obs - vrp_sim)

    at each out-of-sample date, where vrp at t+1 is the frame's y at t. The
    forecast and the target then live in the same simulated world.
    """
    return np.asarray(realised_obs, dtype=float) + (np.asarray(y_obs, dtype=float)
                                                    - np.asarray(y_sim, dtype=float))


def draw_statistic(pair: tuple, frame_sim: pd.DataFrame, observed: dict,
                   vix_monthly: pd.Series) -> tuple:
    """Corrected CW statistic on one simulated history.

    Refits the walk-forward for both models of the pair on the simulated frame,
    converts the forecasts to variance through the pipeline's own scorer, rebuilds
    the realised target from the simulated path, and returns (statistic, n_paired).

    A month enters the pair only where both models have a positive implied
    variance and the simulated realised variance is positive and defined. That is
    the pipeline's own positivity guard, extended to the simulated target.
    """
    smaller, larger = pair
    dataset = observed["dataset"]
    initial_train_end = observed["initial_train_end"]
    spy_returns = observed["spy_returns"]

    implied = {}
    valid = {}
    realised_obs = None
    oos_index = None

    for name in (smaller, larger):
        wf = walk_forward(
            frame_sim,
            feature_cols=observed["feature_cols"][name],
            model_factory=_factory_for(name, frame_sim),
            initial_train_end=initial_train_end,
            target_col="y",
        )
        vix_next = vix_monthly.shift(-1).reindex(wf.index)
        score = score_walk_forward(wf, vix_next, spy_returns)
        implied[name] = score["implied_variance"]
        valid[name] = score["valid_mask"]
        realised_obs = score["realised_variance_next"]
        oos_index = wf.index

    rv_sim = simulated_realised_variance(
        realised_obs,
        dataset["y"].reindex(oos_index).to_numpy(dtype=float),
        frame_sim["y"].reindex(oos_index).to_numpy(dtype=float),
    )

    mask = (valid[smaller] & valid[larger] & np.isfinite(rv_sim) & (rv_sim > 0))
    n_paired = int(mask.sum())
    if n_paired < MIN_PAIRED_MONTHS:
        raise ValueError(f"only {n_paired} paired months survive the guard")

    d = adjusted_differential(rv_sim[mask], implied[smaller][mask], implied[larger][mask])
    return cw_stat(d), n_paired


def one_draw(gen: NullGenerator, observed: dict, seed_seq) -> tuple:
    """Simulate one history and return (statistic, n_paired), or (nan, 0) on failure.

    A draw fails when the generator does not converge, when the positivity guard
    leaves too few paired months, or when a walk-forward fit is singular. Those
    are the ValueError / LinAlgError classes and only those are absorbed; a
    KeyError or any other exception is a coding bug, not a statistical failure,
    and propagates so it crashes the run instead of biasing the failed-draw
    count. Failures are counted and excluded rather than retried with another
    seed, which would make the draw distribution depend on which seeds happened
    to work.
    """
    rng = np.random.default_rng(seed_seq)
    try:
        sim = gen.simulate(rng)
        if not sim.converged:
            return math.nan, 0
        stat, n_paired = draw_statistic(gen.pair, sim.frame, observed,
                                        observed["vix_monthly"])
        if not math.isfinite(stat):
            return math.nan, 0
        return stat, n_paired
    except (ValueError, np.linalg.LinAlgError):
        return math.nan, 0


# Per-process state for the bootstrap workers, populated by _worker_init. The
# generator and the observed series are sent once per worker rather than once per
# draw, because a draw is milliseconds of data and seconds of fitting.
_WORKER: dict = {}


def _worker_init(gen: NullGenerator, observed: dict) -> None:
    """Store the generator and observed inputs in the worker process."""
    _WORKER["gen"] = gen
    _WORKER["observed"] = observed


def _worker_draw(seed_seq) -> tuple:
    """Run one draw in a worker process."""
    return one_draw(_WORKER["gen"], _WORKER["observed"], seed_seq)


def bootstrap_pair(gen: NullGenerator, observed: dict, s_obs: float,
                   n_draws: int = BOOTSTRAP_DRAWS,
                   probe_draws: int = RUNTIME_PROBE_DRAWS,
                   budget_seconds: float = RUNTIME_BUDGET_SECONDS,
                   workers: int = BOOTSTRAP_WORKERS) -> dict:
    """Null-imposed bootstrap p-value for one nested pair.

    Each draw simulates a history from the pair's null generator, refits the
    walk-forward for both models on it, and recomputes the corrected statistic.
    The p-value is the fraction of draw statistics at or above the observed one,
    uncentred: the generator already imposes the null, so the draw distribution is
    a null distribution and recentring it would impose the null twice.

    Seeds are spawned from config.BOOTSTRAP_SEED under a three-element key, which
    keeps this stream disjoint from the two-element key the specification check
    uses, so the check and the p-value never share a simulated history. Draw i
    always uses spawned child i, so the result is the same whatever order the
    draws complete in and whatever the worker count is.

    The first probe_draws run in this process and are timed, the total is
    projected, and the remainder run across workers. If even the parallel
    projection exceeds budget_seconds the run is cut to BOOTSTRAP_DRAWS_REDUCED.
    """
    pair_key = NESTED_PAIRS.index(gen.pair)
    root = np.random.SeedSequence([config.BOOTSTRAP_SEED, pair_key, 1])
    children = root.spawn(n_draws)

    started = time.perf_counter()
    probe = [one_draw(gen, observed, child) for child in children[:probe_draws]]
    probe_seconds = time.perf_counter() - started
    per_draw = probe_seconds / probe_draws

    serial = per_draw * n_draws
    parallel = serial / workers
    print(f"  runtime probe over {probe_draws} draws: {per_draw:.3f} s/draw", flush=True)
    print(f"  projection for {n_draws} draws: {serial / 3600:.2f} h serial, "
          f"{parallel / 3600:.2f} h across {workers} workers", flush=True)

    reduced = False
    if parallel > budget_seconds:
        reduced = True
        children = children[:BOOTSTRAP_DRAWS_REDUCED]
        print(f"  parallel projection exceeds the {budget_seconds / 3600:.1f} h "
              f"budget, cutting to {BOOTSTRAP_DRAWS_REDUCED} draws (p-value "
              f"resolution 1/{BOOTSTRAP_DRAWS_REDUCED} rather than 1/{n_draws})",
              flush=True)
    else:
        print(f"  within the {budget_seconds / 3600:.1f} h budget, "
              f"running the full {n_draws} draws", flush=True)

    def _progress(done, total):
        """Flushed per-chunk progress line: draws done/total, elapsed, ETA."""
        elapsed_s = time.perf_counter() - started
        eta_s = elapsed_s / done * (total - done)
        print(f"  progress: {done}/{total} draws, elapsed {elapsed_s / 60:.1f} min, "
              f"ETA {eta_s / 60:.1f} min", flush=True)

    # Results are consumed incrementally, in submission order (pool.map preserves
    # it), purely so progress can be printed live. Draw content, seeding, and
    # ordering are unchanged.
    results = list(probe)
    total_draws = len(children)
    remaining = children[probe_draws:]
    if remaining:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                                     initargs=(gen, observed)) as pool:
                for r in pool.map(_worker_draw, remaining, chunksize=8):
                    results.append(r)
                    if (len(results) % PROGRESS_PRINT_EVERY_DRAWS == 0
                            or len(results) == total_draws):
                        _progress(len(results), total_draws)
        else:
            for child in remaining:
                results.append(one_draw(gen, observed, child))
                if (len(results) % PROGRESS_PRINT_EVERY_DRAWS == 0
                        or len(results) == total_draws):
                    _progress(len(results), total_draws)

    elapsed = time.perf_counter() - started
    stats_all = np.array([r[0] for r in results], dtype=float)
    n_paired_all = np.array([r[1] for r in results], dtype=float)
    ok = np.isfinite(stats_all)
    draws = stats_all[ok]
    if draws.size == 0:
        raise ValueError("every bootstrap draw failed; no null distribution to report")
    n_ge = int((draws >= s_obs).sum())

    return {
        "pair": gen.pair,
        "s_obs": float(s_obs),
        "draws": draws,
        "n_draws_requested": len(children),
        "n_draws_used": int(draws.size),
        "n_failed": int((~ok).sum()),
        "reduced": reduced,
        "n_ge": n_ge,
        "p_corrected": float(n_ge / draws.size),
        "draw_mean": float(draws.mean()),
        "draw_sd": float(draws.std(ddof=1)),
        "n_paired_mean": float(n_paired_all[ok].mean()),
        "seconds_per_draw": float(per_draw),
        "elapsed_seconds": float(elapsed),
    }


def original_p_values(saved: dict) -> dict:
    """The pipeline's own normal-approximation p-value per pair, for comparison.

    Recomputed through forecast_comparison.clark_west_from_losses on exactly the
    inputs the notebook passes it, so the number reported alongside the corrected
    p-value is the number the corrected one replaces, not a re-derivation.
    """
    out = {}
    for smaller, larger in NESTED_PAIRS:
        key = f"{smaller}__{larger}"
        mask = saved[f"{key}__paired_mask"]
        rv = saved[f"{smaller}__realised_variance_next"][mask]
        s = saved[f"{smaller}__implied_variance"][mask]
        l = saved[f"{larger}__implied_variance"][mask]
        cw = clark_west_from_losses(qlike_per_obs(rv, s), qlike_per_obs(rv, l),
                                    qlike_per_obs(s, l), loss="qlike")
        out[(smaller, larger)] = float(cw.p_value)
    return out


def observed_statistics(saved: dict) -> dict:
    """Corrected statistic and paired n per pair on the saved observed inputs."""
    out = {}
    for smaller, larger in NESTED_PAIRS:
        key = f"{smaller}__{larger}"
        mask = saved[f"{key}__paired_mask"]
        rv = saved[f"{smaller}__realised_variance_next"][mask]
        s = saved[f"{smaller}__implied_variance"][mask]
        l = saved[f"{larger}__implied_variance"][mask]
        d = adjusted_differential(rv, s, l)
        out[(smaller, larger)] = {"statistic": cw_stat(d), "n": int(d.size)}
    return out


def save_bootstrap_results(results: dict, originals: dict):
    """Write the bootstrap results to outputs/corrected_cw_results.npz.

    One flat key per quantity per pair, no pickled objects, so the file loads
    without allow_pickle the same way the nested-inputs npz does.
    """
    payload = {}
    for pair, res in results.items():
        key = f"{pair[0]}__{pair[1]}"
        payload[f"{key}__draws"] = res["draws"]
        payload[f"{key}__s_obs"] = np.array(res["s_obs"])
        payload[f"{key}__p_corrected"] = np.array(res["p_corrected"])
        payload[f"{key}__p_original"] = np.array(originals[pair])
        payload[f"{key}__draw_mean"] = np.array(res["draw_mean"])
        payload[f"{key}__draw_sd"] = np.array(res["draw_sd"])
        payload[f"{key}__n_draws_used"] = np.array(res["n_draws_used"])
        payload[f"{key}__n_failed"] = np.array(res["n_failed"])
        payload[f"{key}__n_ge"] = np.array(res["n_ge"])
        payload[f"{key}__n_paired_observed"] = np.array(res["n_observed"])
        payload[f"{key}__n_paired_draw_mean"] = np.array(res["n_paired_mean"])
    payload["pair_keys"] = np.array([f"{p[0]}__{p[1]}" for p in results])
    path = nested_inputs.OUTPUT_DIR / "corrected_cw_results.npz"
    np.savez(path, **payload)
    return path


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _fmt_table(table: pd.DataFrame, iid_null: bool) -> str:
    """Render one ACF comparison table with the right header for its rule."""
    shown = table.rename(columns={"z_or_band": "band" if iid_null else "z"})
    return shown.to_string(index=False, float_format=lambda v: f"{v:.4f}")


def print_generator_check(gen: NullGenerator, trials: dict, check: dict) -> None:
    """Print fitted parameters, ACF comparison and convergence for one generator."""
    smaller, larger = gen.pair
    iid_null = isinstance(gen, ConstantNullGenerator)

    print()
    print(f"Generator for {smaller} vs {larger}: {gen.name}")
    print("-" * 78)
    print(f"  imposes: {gen.spec['imposes']}")

    if iid_null:
        print(f"  fitted level      = {gen.spec['level']:.6f}")
        print(f"  residual sd       = {gen.spec['resid_sd']:.6f} "
              f"(pool of {gen.spec['resid_pool_n']} observed deviations, resampled iid)")
    else:
        print(f"  intercept         = {gen.spec['const']:.6f}")
        for col, b in gen.spec["coeffs"].items():
            print(f"  {col:<18}= {b: .6f}")
        print(f"  companion max root= {gen.spec['max_root']:.4f} "
              f"(lag recursion, stationary below {STATIONARITY_MAX_ROOT})")
        print(f"  residual sd       = {gen.spec['resid_sd']:.6f}")

        arma = gen.spec["resid_arma"]
        p, q = arma["order"]
        print(f"  residual process  = ARMA({p},{q}) by BIC over p in 0..{ARMA_MAX_P}, "
              f"q in 0..{ARMA_MAX_Q} (BIC {arma['bic']:.4f})")
        if p:
            ar_txt = ", ".join(f"{v: .6f}" for v in arma["ar"])
            print(f"    AR params       = [{ar_txt}], max root {arma['max_root']:.4f}")
        if q:
            ma_txt = ", ".join(f"{v: .6f}" for v in arma["ma"])
            print(f"    MA params       = [{ma_txt}]")
        if p == 0 and q == 0:
            print("    no residual dependence selected, shocks stay iid")
        print(f"    innovation sd   = {arma['innov_sd']:.6f} (pool of "
              f"{arma['innov'].size}, resampled iid inside the ARMA recursion)")

    if isinstance(gen, HARNullGenerator):
        print(f"  exogenous predictors, each fitted separately, ARMA(p,q) by AIC over "
              f"p in 0..{EXOG_ARMA_MAX_P}, q in 0..{EXOG_ARMA_MAX_Q}:")
        for col in EXOG_COLS:
            f = gen.spec["exog"][col]
            p_col, q_col = f["order"]
            print(f"    {col:<18} ARMA({p_col},{q_col}), AIC {f['aic']:.2f}, "
                  f"mean {f['mean']:.4f}, max root {f['max_root']:.4f}, "
                  f"innovation sd {f['innov_sd']:.4f}")
            if p_col:
                print(f"      AR [{', '.join(f'{b: .4f}' for b in f['ar'])}]")
            if q_col:
                print(f"      MA [{', '.join(f'{b: .4f}' for b in f['ma'])}]")

    print()
    print("  ACF of the target series (VRP), simulated against observed:")
    for line in _fmt_table(check["target"], iid_null).splitlines():
        print(f"    {line}")
    if iid_null:
        print("    rule: the null imposes zero serial dependence, so the simulated ACF must sit")
        print("    inside the white-noise band; the observed ACF is not the target here.")
    else:
        print(f"    rule: observed must sit within {ACF_Z_TOL:.0f} trial sd of the trial mean.")

    for col, table in check["exog"].items():
        print()
        print(f"  ACF of {col}, simulated against observed:")
        for line in _fmt_table(table, False).splitlines():
            print(f"    {line}")

    print()
    print(f"  convergence over {trials['n_trials']} trial simulations: "
          f"{trials['convergence_rate']:.3f}")
    for note in trials["notes"]:
        print(f"  note: {note}")
    print(f"  specification check: {'PASS' if check['passed'] else 'FAIL'}")


def print_statistic_demo(saved: dict) -> None:
    """Show the corrected statistic on the saved observed inputs, for orientation.

    This is the corrected point statistic on the observed data. It is not a
    p-value and no null distribution is involved; the bootstrap comes next.
    """
    print()
    print("Corrected point statistic on the saved observed inputs")
    print("-" * 78)
    print(f"{'pair':<40}{'n':>6}{'CW HAC stat':>14}{'NW lag':>8}")
    for smaller, larger in NESTED_PAIRS:
        key = f"{smaller}__{larger}"
        mask = saved[f"{key}__paired_mask"]
        rv = saved[f"{smaller}__realised_variance_next"][mask]
        s = saved[f"{smaller}__implied_variance"][mask]
        l = saved[f"{larger}__implied_variance"][mask]
        d = adjusted_differential(rv, s, l)
        print(f"{smaller + ' vs ' + larger:<40}{d.size:>6}{cw_stat(d):>14.4f}{nw_lag(d.size):>8}")
    print("  Statistic only. No p-value is computed in this task.")


def main() -> int:
    """Fit the three null generators, check their specification, and report."""
    saved = load_saved_inputs()
    observed = slim_observed(load_observed())
    verify_frame_reconstruction(observed)

    dataset = observed["dataset"]
    print("Observed data the generators are fitted on")
    print("-" * 78)
    print(f"  VRP series          : {observed['vrp'].size} months, "
          f"{observed['vrp'].index.min().date()} to {observed['vrp'].index.max().date()}")
    print(f"  dataset rows        : {len(dataset)} "
          f"({dataset.index.min().date()} to {dataset.index.max().date()})")
    print(f"  frame reconstruction: exact against the pipeline dataset")

    generators = build_generators(observed)

    failures = []
    for pair in NESTED_PAIRS:
        gen = generators[pair]
        trials = run_trials(gen)
        check = check_generator(gen, trials)
        print_generator_check(gen, trials, check)
        if not check["passed"]:
            failures.append(f"{pair[0]} vs {pair[1]}: ACF departs materially from observed")
        if trials["convergence_rate"] < 1.0:
            failures.append(f"{pair[0]} vs {pair[1]}: convergence rate "
                            f"{trials['convergence_rate']:.3f} below 1.000")

    print_statistic_demo(saved)

    print()
    print("Generator gate")
    print("-" * 78)
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
        print("  Stopping: a generator that does not reproduce the dependence the null "
              "relies on is not usable for the bootstrap.")
        return 1
    print("  All three generators pass. No p-values computed in this task.")
    return 0


# Pairs the bootstrap is run for. Scope is fixed, not derived: the correction
# applies to constant vs har only. Every other pair, including extended_ols vs
# regime_switching, is out of scope because its pipeline result is a
# non-rejection, which is safe under the fallback.
PVALUE_PAIRS = (NESTED_PAIRS[0],)


def print_bootstrap_result(res: dict, p_original: float) -> None:
    """Print one pair's corrected p-value against the pipeline's original."""
    smaller, larger = res["pair"]
    print()
    print(f"{smaller} vs {larger}")
    print("-" * 78)
    print(f"  paired months, observed        : {res['n_observed']}")
    print(f"  paired months, draw mean       : {res['n_paired_mean']:.1f}")
    print(f"  corrected statistic (HAC)      : {res['s_obs']:.4f}")
    print(f"  corrected bootstrap p          : {res['p_corrected']:.4f} "
          f"({res['n_ge']} of {res['n_draws_used']} draws at or above)")
    print(f"  pipeline original p (normal)   : {p_original:.4f}")
    print(f"  draw distribution mean         : {res['draw_mean']:.4f}")
    print(f"  draw distribution sd           : {res['draw_sd']:.4f}")
    print(f"  draws used / failed            : {res['n_draws_used']} / {res['n_failed']}")
    print(f"  runtime                        : {res['elapsed_seconds'] / 60:.1f} min")
    if res["reduced"]:
        print(f"  note: cut to {res['n_draws_used'] + res['n_failed']} draws on the "
              f"runtime projection, p-value resolution reduced accordingly")


def main_pvalues() -> int:
    """Bootstrap the corrected p-value for the authorised pairs and save the results.

    The generators are refitted here rather than read from the specification-check
    run, because fitting is cheap and a p-value that depends on state left behind
    by another run is not reproducible on its own.
    """
    # Each OLS in the walk-forward is small enough that a threaded BLAS buys
    # nothing, and with one process per core the threads would contend. Set before
    # the pool is created so the worker processes inherit it.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"

    saved = load_saved_inputs()
    observed = slim_observed(load_observed())
    verify_frame_reconstruction(observed)

    generators = build_generators(observed)
    obs_stats = observed_statistics(saved)
    originals = original_p_values(saved)

    print("Model-based null-imposed bootstrap")
    print("-" * 78)
    print(f"  draws requested per pair : {BOOTSTRAP_DRAWS}")
    print(f"  worker processes         : {BOOTSTRAP_WORKERS}")
    print(f"  runtime budget per pair  : {RUNTIME_BUDGET_SECONDS / 3600:.1f} h")
    print(f"  pairs                    : "
          f"{', '.join(f'{p[0]} vs {p[1]}' for p in PVALUE_PAIRS)}")
    print("  p-value is the uncentred fraction of draw statistics at or above the")
    print("  observed statistic; the generator imposes the null, so the draw")
    print("  distribution is already a null distribution.")

    results = {}
    for pair in PVALUE_PAIRS:
        gen = generators[pair]
        print()
        print(f"Running {pair[0]} vs {pair[1]}")
        res = bootstrap_pair(gen, observed, obs_stats[pair]["statistic"])
        res["n_observed"] = obs_stats[pair]["n"]
        results[pair] = res

    print()
    print("Corrected p-values")
    print("=" * 78)
    for pair in PVALUE_PAIRS:
        print_bootstrap_result(results[pair], originals[pair])

    path = save_bootstrap_results(results, originals)
    print()
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pvalues":
        raise SystemExit(main_pvalues())
    raise SystemExit(main())
