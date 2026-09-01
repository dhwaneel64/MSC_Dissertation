import math

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import config
from src.validation import LOCKED_FEATURE_SET, assert_feature_set_complete

# The six numeric regressors, derived from the locked feature set minus the
# categorical regime label. Same six features as Extended OLS, in locked order.
NUMERIC_FEATURES = tuple(f for f in LOCKED_FEATURE_SET if f != "regime")


def _fit_standardised_ols(X: pd.DataFrame, y: pd.Series):
    """Fit one standardised OLS and return (mu, sigma, params, bse).

    Identical construction to ExtendedOLSModel: z-score the regressors on the
    supplied window (ddof=0, zero-variance columns mapped to sigma 1.0), add a
    constant, and fit with Newey-West HAC standard errors at the locked lag rule
    computed from this window's size T. The scaler is fit on exactly the rows
    passed in, so when X is a regime subset the standardisation uses only that
    regime's training observations.
    """
    T = len(y)
    hac_lag = math.floor(
        config.NW_HAC_MULTIPLIER * (T / 100) ** config.NW_HAC_EXPONENT
    )
    mu = X.mean()
    sigma = X.std(ddof=0).replace(0.0, 1.0)
    X_scaled = (X - mu) / sigma
    X_aug = sm.add_constant(X_scaled)
    res = sm.OLS(y, X_aug).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lag})
    return mu, sigma, res.params, res.bse


def _predict_standardised(
    X: pd.DataFrame, mu: pd.Series, sigma: pd.Series, params: pd.Series
) -> np.ndarray:
    """Apply a stored scaler and coefficient set to X, aligning by column label."""
    X_scaled = (X - mu) / sigma
    X_aug = sm.add_constant(X_scaled, has_constant="add")
    return X_aug[params.index].values @ params.values


class RegimeSwitchingOLSModel:
    """Model 4: regime-switching OLS with separate coefficients per regime.

    This is the locked regime-switching definition: a separate standardised OLS
    is fit on each regime's training observations, with its own per-regime scaler
    (mean/std) and coefficient set. It is NOT a regime dummy added to one pooled
    regression. At predict time the test month's regime label selects which
    fitted scaler and coefficient set to apply.

    The same six numeric features as Extended OLS are used (NUMERIC_FEATURES). The
    regime label is consumed as the switch key, not as a numeric regressor. The
    walk-forward engine therefore passes feature_cols = NUMERIC_FEATURES + the
    regime column; fit() splits on the regime column and fits the numeric features
    within each regime.

    Stressed-regime scarcity fallback (methodology-relevant, documented):
    the stressed regime has roughly 71 months in the whole sample and far fewer in
    any early expanding-window training set, so at some walk-forward steps a regime
    subset is too small to fit six features plus an intercept (seven parameters)
    stably. If a regime's training subset has fewer than min_obs_ =
    REGIME_MIN_TRAIN_MULTIPLIER * (n_numeric_features + 1) observations, that
    regime falls back to the pooled (Extended OLS) coefficients for its predictions
    rather than producing an unstable near-saturated per-regime fit. The pooled fit
    is the equality-restricted special case (all regimes sharing one coefficient
    set), which is exactly Extended OLS; this is why Extended OLS nests this model.
    Fallback is recorded on fallback_regimes_ and is never silent.

    Attributes after fit:
        pooled_params_, pooled_bse_, pooled_mu_, pooled_sigma_: the pooled
            (restricted) Extended-OLS-equivalent fit on all training rows; the
            fallback coefficients.
        regime_params_, regime_bse_, regime_mu_, regime_sigma_: dicts keyed by
            regime label, present only for regimes with enough observations.
        regime_n_: dict regime -> training-observation count in this window.
        fallback_regimes_: set of regime labels that fell back to pooled at this
            fit (subset too small or absent in the training window).
        min_obs_: the minimum per-regime observation count for a separate fit.
    """

    def __init__(
        self,
        monthly_vrp: pd.DataFrame,
        numeric_features: tuple[str, ...] = NUMERIC_FEATURES,
        regime_col: str = "regime",
        min_train_multiplier: int = config.REGIME_MIN_TRAIN_MULTIPLIER,
    ) -> None:
        self._monthly_vrp = monthly_vrp
        self._numeric_features = list(numeric_features)
        self._regime_col = regime_col
        self._min_train_multiplier = min_train_multiplier
        self.min_obs_ = min_train_multiplier * (len(self._numeric_features) + 1)

        self.pooled_mu_: pd.Series | None = None
        self.pooled_sigma_: pd.Series | None = None
        self.pooled_params_: pd.Series | None = None
        self.pooled_bse_: pd.Series | None = None

        self.regime_mu_: dict[str, pd.Series] = {}
        self.regime_sigma_: dict[str, pd.Series] = {}
        self.regime_params_: dict[str, pd.Series] = {}
        self.regime_bse_: dict[str, pd.Series] = {}
        self.regime_n_: dict[str, int] = {}
        self.fallback_regimes_: set[str] = set()
        self._fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        assert_feature_set_complete(self._monthly_vrp)
        if self._regime_col not in X_train.columns:
            raise ValueError(
                f"X_train is missing the regime switch column {self._regime_col!r}"
            )

        X_num = X_train[self._numeric_features]
        regimes = X_train[self._regime_col].astype(str)

        # Pooled (restricted) fit on all training rows. Identical construction to
        # ExtendedOLSModel, so this is the equality-restricted special case used as
        # the documented fallback.
        (
            self.pooled_mu_,
            self.pooled_sigma_,
            self.pooled_params_,
            self.pooled_bse_,
        ) = _fit_standardised_ols(X_num, y_train)

        # Separate fit per regime, with the documented small-subset fallback.
        for regime in config.REGIME_LABELS:
            mask = (regimes == regime).to_numpy()
            n_regime = int(mask.sum())
            self.regime_n_[regime] = n_regime
            if n_regime >= self.min_obs_:
                mu, sigma, params, bse = _fit_standardised_ols(
                    X_num.loc[mask], y_train.loc[mask]
                )
                self.regime_mu_[regime] = mu
                self.regime_sigma_[regime] = sigma
                self.regime_params_[regime] = params
                self.regime_bse_[regime] = bse
            else:
                self.fallback_regimes_.add(regime)

        self._fitted = True

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("fit() must be called before predict()")

        X_num = X_test[self._numeric_features]
        regimes = X_test[self._regime_col].astype(str)

        preds = np.empty(len(X_test), dtype=float)
        for i in range(len(X_test)):
            regime = regimes.iloc[i]
            row = X_num.iloc[i : i + 1]
            if regime in self.regime_params_:
                preds[i] = _predict_standardised(
                    row,
                    self.regime_mu_[regime],
                    self.regime_sigma_[regime],
                    self.regime_params_[regime],
                )[0]
            else:
                # Fallback: regime too small in this training window (or absent),
                # so use the pooled Extended-OLS-equivalent coefficients.
                preds[i] = _predict_standardised(
                    row, self.pooled_mu_, self.pooled_sigma_, self.pooled_params_
                )[0]
        return preds


def regime_fallback_log(
    dataset: pd.DataFrame,
    initial_train_end: pd.Timestamp,
    numeric_features: tuple[str, ...] = NUMERIC_FEATURES,
    regime_col: str = "regime",
    min_train_multiplier: int = config.REGIME_MIN_TRAIN_MULTIPLIER,
    target_col: str = "y",
) -> pd.DataFrame:
    """Replay the expanding-window walk-forward and record the fallback decision.

    Deterministic replay of the engine's stepping that records, for each
    out-of-sample month, the test month's regime, how many observations of that
    regime are in the training window up to t, the min_obs threshold, and whether
    the step falls back to the pooled coefficients. The decision depends only on
    the test month's regime label and that regime's count in the training window,
    so this reproduces RegimeSwitchingOLSModel's fallback behaviour at every step
    without refitting. It is used to report how many steps hit the fallback and in
    which regimes; it never feeds any metric.

    Returns a DataFrame indexed by the OOS dates with columns:
        regime, n_train_regime, min_obs, fallback (bool).

    Raises the same index errors as walk_forward (non-monotonic index, missing or
    last-row initial_train_end).
    """
    if not dataset.index.is_monotonic_increasing:
        raise ValueError("dataset.index must be monotonic increasing")
    if initial_train_end not in dataset.index:
        raise ValueError(
            f"initial_train_end {initial_train_end} is not in dataset.index"
        )
    pos = dataset.index.get_loc(initial_train_end)
    if pos >= len(dataset) - 1:
        raise ValueError(
            f"initial_train_end {initial_train_end} is the last row; no OOS rows available"
        )

    min_obs = min_train_multiplier * (len(numeric_features) + 1)
    regimes = dataset[regime_col].astype(str)

    records = []
    for i in range(pos, len(dataset) - 1):
        train_regimes = regimes.iloc[: i + 1]
        test_date = dataset.index[i + 1]
        test_regime = regimes.iloc[i + 1]
        n_train_regime = int((train_regimes == test_regime).sum())
        records.append(
            {
                "date": test_date,
                "regime": test_regime,
                "n_train_regime": n_train_regime,
                "min_obs": min_obs,
                "fallback": n_train_regime < min_obs,
            }
        )

    log = pd.DataFrame(records).set_index("date")
    log.index.name = dataset.index.name
    return log
