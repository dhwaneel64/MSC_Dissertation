"""Central configuration for the CW/QLIKE bootstrap simulation study.

Every parameter used anywhere in the study lives here. No magic numbers
elsewhere. Each entry carries a one-line comment stating its role and,
where the value is a modelling choice, why it was set that way.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Data generating process (DGP)
# ---------------------------------------------------------------------------

# Latent monthly log-variance follows AR(1): log h_t = c + phi * log h_{t-1} + shock.
LOGVAR_INTERCEPT = -0.5      # c, the AR(1) intercept, sets the unconditional level of log-variance
LOGVAR_PHI = 0.95           # phi, persistence of monthly log-variance, high to mimic volatility clustering
LOGVAR_SHOCK_SCALE = 0.30   # scale multiplying the Student-t shock in the log-variance equation

# Shock/innovation distribution. Student-t with low df gives fat tails so
# crash-like months occur and QLIKE differentials come out right-skewed.
SHOCK_T_DF = 5              # degrees of freedom for the Student-t innovations, df=5 for fat tails

# Realised variance is the latent variance times positive multiplicative noise
# with mean 1: RV_t = h_t * u_t. We use a chi-square scaled to mean 1 so the
# measurement error is positive and unbiased for h_t.
RV_NOISE_CHI2_DF = 8       # df of the chi-square used for RV multiplicative noise, then scaled to mean 1

# One persistent AR(1) predictor x_t: x_t = psi * x_{t-1} + shock.
PREDICTOR_PSI = 0.9         # psi, persistence of the predictor, high so x_t is a slow-moving signal
PREDICTOR_SHOCK_SCALE = 1.0 # scale of the predictor's Gaussian shock

# Burn-in length discarded from the front of every simulated path so the AR(1)
# processes start from their stationary regime rather than the initial value.
BURN_IN = 200              # number of initial observations discarded to remove transient dynamics

# ---------------------------------------------------------------------------
# Alternative-world strength grid
# ---------------------------------------------------------------------------

# Coefficient on x_t in the variance equation under the alternative world.
# Two strengths, weak and moderate. Set from the calibration run: weak 0.03
# gives 1.66% mean QLIKE reduction and a 66% big-model win rate, with the true
# effect about 2.5x the null-world estimation-noise penalty; moderate 0.05 gives
# 4.08% and 86%. Betas above 0.10 saturate (0.20 gives 23.6% and a 99.7% win
# rate), so power there would be trivially 1. Calibrated at t_oos=220, R=300.
BETA_GRID = [0.03, 0.05]   # [weak, moderate] loading of x_t under the alternative

# Candidate betas scanned by the calibration run, from which the two BETA_GRID
# values are chosen. Concentrated on 0.02 to 0.10 because a pilot scan showed
# the whole target range (a few percent QLIKE gain) sits inside it: by beta=0.20
# the gain is above 20% and the big model wins in every replication, so anything
# larger makes power trivially 1. The 0.20 point is kept as an upper anchor
# recording where the effect saturates.
CAL_BETA_CANDIDATES = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.20]  # betas scanned for the population QLIKE gain
R_CAL = 300                # replications per calibration beta, enough to pin the mean QLIKE gain to well under a percentage point without bootstrap cost
CAL_T_OOS = 220            # out-of-sample length used for calibration, the shorter of the two horizons so the gain is not overstated

# ---------------------------------------------------------------------------
# Sample sizes
# ---------------------------------------------------------------------------

T_TRAIN = 140              # length of the initial training window before walk-forward evaluation begins
T_OOS_GRID = [220, 250]    # out-of-sample lengths evaluated, the two horizons the study reports

# ---------------------------------------------------------------------------
# Monte Carlo settings
# ---------------------------------------------------------------------------

R_SIZE = 2000              # replications for size (null-world) runs, large for tight rejection-rate estimates
R_POWER = 1000             # replications per alternative cell for power runs
B_BOOT = 999               # bootstrap draws per replication, odd so the 95th percentile is well defined
B_BOOT_SIZE = 299          # Bootstrap draws for size runs. Davidson-MacKinnon rule: alpha*(B+1) integer at both alpha levels (0.05*300=15, 0.10*300=30), so the bootstrap p-value has no boundary granularity bias at the tested levels.
B_BOOT_POWER = 299         # Bootstrap draws for power runs, same value and same Davidson-MacKinnon reason as B_BOOT_SIZE, kept as its own name so the two experiments can be varied independently.

# ---------------------------------------------------------------------------
# Block length for the circular block bootstrap
# ---------------------------------------------------------------------------

# Baseline follows the n^(1/3) rate rule (Hall, Horowitz and Jing 1995).
# n = 220 gives 220^(1/3) which is about 6, so 5 sits at the rule. The
# sensitivity grid confirms the result is stable to the block choice.
BLOCK_BASELINE = 5         # baseline block length, at the n^(1/3) rate rule for n=220
BLOCK_SENSITIVITY = [3, 5, 10]  # block lengths for the robustness check around the baseline

# Resampling scheme for the bootstrap p-value. "stationary" is the Politis-Romano
# stationary bootstrap (geometric block lengths, mean BLOCK_BASELINE, circular
# wrap); "fixed" is the fixed-length circular block bootstrap. Default stationary:
# geometric lengths avoid the artefacts of a single fixed block while keeping the
# same expected block length.
BOOTSTRAP_SCHEME = "stationary"  # "stationary" or "fixed", selects the resampling scheme

# CW penalty scale under QLIKE. QLIKE per obs in log-error u is exp(-u)+u-1,
# approx u^2/2 for small u, half the curvature of squared error u^2, so the
# estimation-noise penalty is half the log-MSPE penalty (f_s-f_b)^2 in logs.
# Measured on the null, 2026-07-07: the ratio d_raw/adjustment is -0.40, and the
# residual mean under the half adjustment is +0.0005.
CW_ADJ_SCALE = 0.5

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

ALPHA_LEVELS = [0.05, 0.10]  # nominal significance levels at which rejection rates are reported

# ---------------------------------------------------------------------------
# Sensitivity runs
# ---------------------------------------------------------------------------

# Null-world sensitivity runs at a single horizon, one row per setting on three
# axes: the generator specification, the number of bootstrap draws, and the
# residual resampling scheme inside the generator.
R_SENS = 1000              # replications per sensitivity cell, matched to R_POWER so each row has power-run Monte Carlo precision
SENS_T_OOS = 220           # out-of-sample length for the sensitivity runs, the shorter horizon where the size distortion is largest
SENS_GENERATOR_GRID = ["arma", "ar1"]   # generator specifications compared: the validated ARMA(1,1) and the mis-specified AR(1)
SENS_B_GRID = [299, 999]   # bootstrap-draw settings compared, the value the size and power runs use and the larger B_BOOT value
SENS_RESID_IID = 1         # residual block length that denotes iid resampling, one residual drawn at a time, the current scheme
SENS_RESID_BLOCK = 3       # residual block length on the resampling axis, short because the ARMA residuals should already be near-white so only low-order dependence is at stake
R_SENS_REDUCED = 500       # fallback replication count for the B=999 cell if the projected total runtime exceeds the budget, half precision on that row only
SENS_RUNTIME_BUDGET_H = 8.0  # wall-clock budget in hours for the whole sensitivity run, above which the B=999 cell drops to R_SENS_REDUCED
SENS_PROBE_REPS = 3        # replications timed per cell before the run starts, enough to project the total without materially adding to it

# Newey-West automatic lag for the HAC long-run variance in the CW statistic.
# Lag = floor(NW_LAG_COEF * (n / NW_LAG_BASE) ** NW_LAG_EXP), the locked project
# rule (the same automatic bandwidth the dissertation pipeline uses). At n=220
# this gives lag 4.
NW_LAG_COEF = 4            # multiplier in the Newey-West automatic lag rule
NW_LAG_BASE = 100         # sample-size scaling base in the lag rule
NW_LAG_EXP = 2.0 / 9.0    # exponent in the lag rule

# ---------------------------------------------------------------------------
# Reporting layer: the four bootstrap constructions compared
# ---------------------------------------------------------------------------

# The four constructions tried during the study, in the order they were built.
# The reporting table has one row per key, in this order.
CONSTRUCTION_KEYS = ["a", "b", "c", "d"]   # roster of constructions, ordered as they were built

# Short label naming each construction in the comparison table.
CONSTRUCTION_LABEL = {
    "a": "Loss-series block bootstrap, plain SE, full-scale adjustment",
    "b": "Loss-series stationary bootstrap, HAC statistic",
    "c": "Calhoun data-level block bootstrap, HAC statistic",
    "d": "Model-based null-imposed ARMA bootstrap, HAC statistic",
}

# One-line mechanism statement per construction, why that construction behaves
# the way the size column shows. Supplied text, used verbatim in the table.
CONSTRUCTION_MECHANISM = {
    "a": ("Resamples the loss series only; parameter-estimation error is a per-history "
          "common shock invisible to any reshuffling of the differentials."),
    "b": ("HAC absorbs serial dependence within a history but not the between-history "
          "component arising from parameter estimation, which no resampling of the loss "
          "series can reproduce."),
    "c": ("Resampling raw data with per-draw refitting recreates estimation error in "
          "principle, but chunk-gluing destroys the persistence the estimation step "
          "depends on; no block length balances realism against variety."),
    "d": ("Simulating seamless histories from a fitted null generator preserves "
          "persistence exactly and imposes the null by construction, so estimation "
          "error enters the draw distribution."),
}

# Constructions whose per-replication output is no longer on disk. Their row is
# filled from these recorded values instead of from an npz. None marks a
# quantity that was not recorded, so the gap shows in the table rather than
# being silently filled.
#
# Construction (a) is the first size run in the programme: fixed circular block
# length 5, plain standard error, full-scale adjustment. Its per-replication
# arrays are not available, so the row is filled from the values recorded from
# that run's output at the time. Two cells of that row were not recorded and are
# None below: draw_std, the mean draw-statistic std, and s_obs_std, the target
# s_obs std across replications.
CONSTRUCTION_RECORDED = {
    "a": {
        "r_reps": 2000,          # replications in the original-spec size run
        "b_boot": 999,           # bootstrap draws per replication in that run
        "rej": [0.189, 0.247],   # rejection rate at each ALPHA_LEVELS entry, recorded from the run output
        "se": [0.009, 0.010],    # Monte Carlo standard error at each ALPHA_LEVELS entry, as recorded
        "draw_std": None,        # mean draw-statistic std, not recorded by that run
        "s_obs_std": None,       # target s_obs std across replications, not recorded by that run
        "t_oos": 220,            # horizon of that run, matching CONSTRUCTION_T_OOS so the row is comparable
    },
}

# Which of the two defects each construction still carries. The adjustment
# defect is the full-scale CW penalty, later halved to match QLIKE curvature;
# the resampling defect is resampling the loss series rather than the data or a
# fitted null generator, which leaves estimation error out of the draws.
CONSTRUCTION_DEFECTS = {
    "a": "adjustment scale + resampling level",
    "b": "resampling level",
    "c": "resampling level",
    "d": "none",
}

# Trailing note printed under the construction table, stating what the (a) row
# does and does not measure.
CONSTRUCTION_TABLE_NOTE = (
    "Construction (a) carries both the over-scaled adjustment and the loss-series "
    "resampling, so its size is not a clean measurement of the resampling defect alone. "
    "With the adjustment corrected but resampling unchanged, size at t_oos=220 was 0.131 "
    "(plain SE, fixed block) and 0.128 (HAC, stationary), isolating the resampling "
    "contribution."
)

# Horizon at which the four constructions are compared. All four were run at the
# shorter horizon, so this is the only horizon on which the comparison is
# like-for-like.
CONSTRUCTION_T_OOS = 220   # out-of-sample length for the construction comparison table

# ---------------------------------------------------------------------------
# Reporting layer: diagnostic figure
# ---------------------------------------------------------------------------

QQ_T_OOS = 220             # horizon whose null s_obs values the QQ-plot uses, the shorter horizon where the distortion is largest

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

MASTER_SEED = 20260707     # single master seed for the whole study, fixed for reproducibility

# Root SeedSequence. Per-replication RNGs are spawned from disjoint children so
# size and power runs never share a stream. Downstream code spawns from these.
_ROOT_SEED_SEQUENCE = np.random.SeedSequence(MASTER_SEED)

# Four disjoint child sequences, one for size runs, one for power runs, one for
# the beta calibration and one for the sensitivity runs, so the four experiments
# draw independent randomness. Children are indexed by position, so the first
# three sequences are byte-identical to what spawn(3) produced before the
# sensitivity child was added, and the size, power and calibration runs already
# on disk remain reproducible.
(SIZE_SEED_SEQUENCE, POWER_SEED_SEQUENCE, CAL_SEED_SEQUENCE,
 SENS_SEED_SEQUENCE) = _ROOT_SEED_SEQUENCE.spawn(4)
