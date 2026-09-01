"""Single source of truth for all methodology parameters and paths.

Constants are added incrementally as tasks require them. Each constant has a
short comment stating what it represents and why the value was chosen.
"""
from pathlib import Path

# Project root is the parent of /src/, derived from this file's location so imports and filesystem access work regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folder that holds raw and processed data artefacts.
DATA_DIR = PROJECT_ROOT / "data"

# Earliest date requested from yfinance. SPY began trading on 1993-01-29, so the
# usable sample for the dissertation cannot start before this, even though VIX
# history extends to 1990.
DATA_START_DATE = "1993-01-29"

# Yahoo Finance ticker for the SPY ETF, used as the S&P 500 proxy because it is directly tradable.
TICKER_SPY = "SPY"

# Yahoo Finance ticker for the VIX spot index.
TICKER_VIX = "^VIX"

# Yahoo Finance ticker for the CBOE SKEW index (option-implied tail asymmetry).
# Available from 1993, covering the full dissertation sample.
TICKER_SKEW = "^SKEW"

# Conversion factor from VIX vol-points to decimal annualised variance.
# VIX is quoted in percent (e.g., VIX=20 means 20% annualised vol), so
# decimal annualised variance = (VIX / 100)^2 = VIX^2 / VIX_VARIANCE_SCALE.
VIX_VARIANCE_SCALE = 10_000

# Scale from decimal annualised volatility to vol-points (VIX quote units).
# Derived as the square root of VIX_VARIANCE_SCALE so the vol-point and the
# variance conversions cannot drift apart: 0.20 decimal vol x 100 = 20 vol-points,
# and 20^2 / 10000 = 0.04 decimal variance.
VOL_POINTS_SCALE = int(VIX_VARIANCE_SCALE ** 0.5)

# Rolling window length for realised volatility, in trading days. 21 trading
# days approximates one calendar month, the standard window in the vol
# forecasting literature (Andersen, Bollerslev et al.).
RV_WINDOW = 21

# Trading days per year, used to annualise daily-frequency variance estimates
# by multiplying the rolling standard deviation by sqrt(ANNUALISATION_FACTOR_DAILY).
ANNUALISATION_FACTOR_DAILY = 252

# VIX thresholds for regime classification (locked methodology, Regime classification section).
# calm: VIX < VIX_CALM_UPPER; normal: VIX_CALM_UPPER <= VIX <= VIX_STRESSED_LOWER; stressed: VIX > VIX_STRESSED_LOWER.
VIX_CALM_UPPER = 15      # VIX strictly below this is "calm"
VIX_STRESSED_LOWER = 25  # VIX strictly above this is "stressed"; "normal" is the closed interval [VIX_CALM_UPPER, VIX_STRESSED_LOWER]
REGIME_LABELS = ("calm", "normal", "stressed")  # canonical ordering used everywhere

# Circle 3A gate label. The conditioned strategy suppresses the short-vol entry
# whenever the VIX-threshold regime at t is stressed (VIX > VIX_STRESSED_LOWER).
# Derived from REGIME_LABELS so the label string has a single source of truth.
REGIME_STRESSED_LABEL = REGIME_LABELS[-1]  # "stressed"

# Lag horizons (in months) for the HAR-style regression on monthly VRP.
# The 1-, 3-, and 6-month structure follows Corsi (2009) adapted to monthly frequency.
HAR_LAGS_MONTHS = (1, 3, 6)

# Forecast horizon h used in the HLN small-sample correction for Diebold-Mariano,
# and as the default HAC lag truncation (h - 1). Set to 1 because all models
# produce one-month-ahead VRP forecasts.
DM_DEFAULT_HORIZON = 1

# Significance threshold for the "favoured" label in DMResult and CWResult.
# 0.05 is the standard threshold in the forecasting literature.
COMPARISON_ALPHA = 0.05

# Block-bootstrap inference for the nested Clark-West test on QLIKE differentials.
# The N(0,1) reference distribution is squared-error-specific and is not valid for
# the QLIKE-weighted statistic (only the bias-correction centering carries over),
# so the nested-pair p-value is obtained by a moving-block bootstrap under H0.
# BOOTSTRAP_REPLICATIONS: number of resamples. 9999 gives p-value granularity of
#   1 / (9999 + 1) = 1e-4 under the (1 + count) / (B + 1) convention.
BOOTSTRAP_REPLICATIONS = 9999
# BOOTSTRAP_SEED: fixed seed so the bootstrap p-value is reproducible across runs.
BOOTSTRAP_SEED = 20260617
# BOOTSTRAP_BLOCK_LENGTH_EXPONENT: moving-block length rule, block = floor(n ** e).
# e = 1/3 is the standard growth rate for block-bootstrap consistency
# (Hall, Horowitz and Jing 1995); for the paired subset n = 222 this gives a
# block length of 6 months, which guards any residual serial dependence in the
# one-step-ahead f_t series while leaving enough distinct blocks to resample.
BOOTSTRAP_BLOCK_LENGTH_EXPONENT = 1 / 3

# Chosen to preserve at least 100 OOS observations for Clark-West reliability
# given the 1993-present sample length.
INITIAL_TRAINING_YEARS_FULL = 12

# Minimum rows a model-ready dataset must retain after the NaN drop. Below this a
# monthly regression has too few observations to be meaningful; the builder raises
# rather than returning a degenerate frame.
MIN_MODEL_DATASET_ROWS = 24

# Minimum training rows for one HAR-RV recursive fit. Below this the 4-parameter
# OLS (const + D + W + M) is near-saturated, so the forecast date is skipped.
HAR_RV_MIN_TRAIN_ROWS = 5

# Minimum paired observations for any forecast-comparison statistic (DM with the
# HLN correction, Clark-West, the block bootstrap, and a corrected-CW bootstrap
# draw). 8 is the smallest n at which the HLN small-sample correction factor is
# well defined and the block bootstrap has more than one block; a single floor is
# used everywhere so the tests cannot disagree about admissible sample sizes.
MIN_COMPARISON_OBS = 8

# Scale on the Clark-West estimation-noise penalty under QLIKE. QLIKE per
# observation in the log forecast error u = log(rv) - log(f) is exp(u) - u - 1,
# whose second-order term is u^2/2: half the curvature of squared error u^2. The
# canonical CW penalty is the full squared forecast gap because it is derived
# under squared-error loss, so under QLIKE the same derivation returns half of it.
CW_ADJ_SCALE = 0.5

# Newey-West (1994) automatic HAC lag selection: lag = floor(NW_HAC_MULTIPLIER * (T / 100) ** NW_HAC_EXPONENT)
# where T is the regression sample size. Values 4 and 2/9 are the standard Newey-West (1994)
# data-driven bandwidth parameters, locked by the methodology for all regressions in this project.
NW_HAC_MULTIPLIER = 4.0
NW_HAC_EXPONENT = 2 / 9

# Regime-switching OLS (Model 4) per-regime minimum-training-observations rule.
# A separate OLS per regime needs at least (number of numeric features + intercept)
# = 7 parameters; fitting that many on a barely-larger subset gives an unstable,
# near-saturated fit. REGIME_MIN_TRAIN_MULTIPLIER sets the minimum subset size as
# multiplier * (n_features + 1) = 2 * 7 = 14 observations. Below that the step
# falls back to the pooled (Extended OLS) coefficients for that prediction rather
# than fitting an unstable per-regime OLS. 2x is the conventional rule-of-thumb
# floor on observations-per-parameter; the stressed regime (~71 total months) is
# the expected fallback trigger in the early expanding-window years.
REGIME_MIN_TRAIN_MULTIPLIER = 2

# HAR-RV variance forecaster: Corsi (2009) cascade horizons in trading days.
# D=1 (daily), W=5 (weekly), M=22 (monthly) regressors. Forward target uses RV_WINDOW=21.
HAR_RV_HORIZON_D = 1
HAR_RV_HORIZON_W = 5
HAR_RV_HORIZON_M = 22

# Prefix for the dated raw-data snapshot parquets written to DATA_DIR.
# A snapshot is named DATA_DIR / f"{SNAPSHOT_PREFIX}{YYYY-MM-DD}.parquet" and
# stores daily close columns for SPY, VIX, and SKEW under sanitised column names.
SNAPSHOT_PREFIX = "raw_snapshot_"

# Date of the locked raw-data snapshot used for all dissertation results.
# download_prices reads from this specific file by default (refresh=False),
# making every run reproducible regardless of execution date. To update the
# sample, set REFRESH_SNAPSHOT = True in the notebook snapshot cell, then
# change this date and delete the old snapshot files.
LOCKED_SNAPSHOT_DATE = "2026-06-17"

# Minimum paired out-of-sample observations for a regime-conditional test to be
# treated as decisive. Below this the Diebold-Mariano / Clark-West p-value is still
# computed and reported but flagged low_power and read descriptively, because the
# tests have little power on a short subset. 50 is a conventional small-sample floor
# for forecast-comparison tests; the stressed regime (~41 OOS months) sits just
# below it and so is expected to trip the flag.
LOW_POWER_MIN_N = 50

# XGBoost (Model 5, final)
# Reproducibility seed for XGBoost tree construction. Fixed so every walk-forward
# refit and every SHAP attribution is byte-reproducible across runs.
XGBOOST_SEED = 20260623

# Single-threaded tree construction. XGBoost's multithreaded histogram builder can
# reorder floating-point reductions and break byte-for-byte reproducibility; the
# per-step-refit leakage test requires identical fits, so n_jobs is pinned to 1.
# The monthly sample (~350 rows) is small enough that single-threaded cost is trivial.
XGBOOST_N_JOBS = 1

# Hyperparameter search grid for the tune-once step. The selected configuration is
# locked after selection on the 1993-2004 training window and reused at every
# walk-forward step (Bergmeir and Benitez 2012: tune once, refit recursively). The
# grid is deliberately small and conservative for a ~130-row training window:
# shallow trees, low learning rates, modest tree counts, and a light
# min_child_weight regulariser. min_child_weight is the chosen regulariser (over
# subsample) because it keeps each fit deterministic given the seed, which the
# leakage test depends on.
XGB_MAX_DEPTH_GRID = (2, 3, 4)
XGB_LEARNING_RATE_GRID = (0.01, 0.05, 0.1)
XGB_N_ESTIMATORS_GRID = (100, 300, 500)
XGB_MIN_CHILD_WEIGHT_GRID = (1, 5)

# Expanding-window time-series cross-validation INSIDE the initial training window,
# used to score each grid candidate on QLIKE. XGB_CV_FOLDS contiguous validation
# blocks; the first fold trains on the first XGB_CV_MIN_TRAIN_MONTHS rows and each
# later fold expands the training set. 5 years is the minimum initial training
# block; 4 folds leaves a usable validation block per fold given the ~134-month
# training window. The out-of-sample period (2005+) never enters this CV.
XGB_CV_FOLDS = 4
XGB_CV_MIN_TRAIN_MONTHS = 60

# Circle 3A transaction-cost model (variance units, spec amendment).
# The cost is proportional to the variance strike, not a flat variance amount. A
# realistic haircut is an approximately fixed round-trip spread in vol-points at the
# VIX-futures level; the vol-to-variance map d(sigma^2) = 2 * sigma * d(sigma) makes
# the equivalent variance-space cost level-dependent, so stressed (high-VIX) months
# cost more in absolute variance terms. That property strengthens the honesty of the
# test rather than weakening it, and both strategies bear identical costs, so the
# naive-vs-conditioned headline difference is robust to the choice.
# Derivation chain (reproducible from this comment alone):
#   round-trip futures spread s ~ 0.1 vol-points
#   variance-space cost per round trip = 2 * sigma * s / VIX_VARIANCE_SCALE
#   as a fraction of the variance strike (sigma^2 / VIX_VARIANCE_SCALE) this is
#     2 * s / sigma = 2 * 0.1 / 20 ~ 0.01 at sigma ~ 20 (a representative VIX level),
#   i.e. ~1% of the variance strike per round trip; the 2x stress case is ~2%.
# The applied per-round-trip cost at t is therefore COST_HAIRCUT_BASE * VIX(t)^2 / VIX_VARIANCE_SCALE.
# Primary citation (peer-reviewed, goes in the reference list): Simon and Campasano
# (2014), "The VIX Futures Basis: Evidence and Trading Strategies", Journal of
# Derivatives 21(3), for the round-trip VIX-futures trading cost.
# Secondary corroboration (working paper, named in a footnote): arXiv working paper
# on VIX-futures round-trip costs, exact handle to be confirmed by the user.
COST_HAIRCUT_BASE = 0.01     # base round-trip cost as a fraction of the variance strike (~1%, from a 0.1 vol-point futures spread via 2 * sigma * s / VIX_VARIANCE_SCALE at sigma ~ 20)
COST_STRESS_MULTIPLIER = 2   # stress case: 2x base, ~2% of the variance strike per round trip, reported alongside base everywhere

# Circle 3A tail-risk episode and metrics parameters (task 3, episode table plus
# full-sample tail metrics on the monthly net-base P&L series). Descriptive only;
# no model is refit and no payoff is recomputed.
# Named crash windows for the episode table, each (label, start, end) inclusive.
# Chosen as the canonical equity-vol stress events inside the 2005+ OOS span:
# the 2008 global financial crisis, the February 2018 volmageddon, and the
# March 2020 COVID crash. Bounds are calendar dates; monthly first-trading-day
# observations falling within [start, end] are summed for that episode.
CIRCLE3A_CRASH_WINDOWS = (
    ("2008-09_2009-03", "2008-09-01", "2009-03-31"),
    ("2018-02",         "2018-02-01", "2018-02-28"),
    ("2020-02_2020-04", "2020-02-01", "2020-04-30"),
)
# Any single OOS month whose entry-month VIX exceeds this level is added as its own
# episode row (if not already inside a named window). 40 is a conventional extreme
# panic level (VIX > 40 is roughly the 99th percentile of the daily VIX history).
CIRCLE3A_TAIL_VIX_THRESHOLD = 40
# Rolling window length, in months, for the worst-rolling-sum tail metric.
CIRCLE3A_WORST_ROLL_MONTHS = 3
# Months per year, used to annualise the monthly net-P&L series: annualised mean =
# monthly mean * MONTHS_PER_YEAR, annualised vol = monthly vol * sqrt(MONTHS_PER_YEAR).
MONTHS_PER_YEAR = 12
