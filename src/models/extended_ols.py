import math

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import config
from src.validation import assert_feature_set_complete


class ExtendedOLSModel:
    """Model 3: extended HAR OLS on lagged VRP, VIX level, CBOE skew, and realised skewness.

    Numeric features: vix_level, cboe_skew, vrp_h1m, vrp_h3m, vrp_h6m,
    realised_skew_21d (six features; cboe_skew available from 1993, no 2012+ restriction).
    Refit at every walk-forward step. Standard errors use Newey-West HAC with
    lag = floor(NW_HAC_MULTIPLIER * (T / 100) ** NW_HAC_EXPONENT).

    The six features span roughly three orders of magnitude (vix_level ~10-80,
    cboe_skew ~100-170, vrp lags ~0.001-0.05, realised_skew_21d ~-3 to +2).
    Features are z-scored before fitting so the design matrix is
    well conditioned. The scaler (per-feature mean and standard deviation) is fit
    on the training window only, inside fit(), and stored on the instance; predict()
    applies the stored training-window scaler to the test point. Because fit()
    sees only X_train, the scaler cannot absorb information from the test point or
    any later observation, so standardisation introduces no look-ahead. OLS fitted
    values are invariant to this affine rescaling of the regressors; the rescaling
    changes the coefficient scale (params_/bse_ are in standardised-feature units),
    not the predictions.

    Attributes after fit:
        params_: fitted coefficients (const + six features, standardised-feature units).
        bse_: HAC standard errors for each coefficient.
        mu_: per-feature training-window means used by the scaler.
        sigma_: per-feature training-window standard deviations used by the scaler.
    """

    def __init__(self, monthly_vrp: pd.DataFrame) -> None:
        self._monthly_vrp = monthly_vrp
        self.params_: pd.Series | None = None
        self.bse_: pd.Series | None = None
        self.mu_: pd.Series | None = None
        self.sigma_: pd.Series | None = None
        self._fit_result = None

    def _standardise(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the stored training-window scaler to X, aligning by column label."""
        return (X - self.mu_) / self.sigma_

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        assert_feature_set_complete(self._monthly_vrp)

        T = len(y_train)
        hac_lag = math.floor(
            config.NW_HAC_MULTIPLIER * (T / 100) ** config.NW_HAC_EXPONENT
        )

        # Scaler fit on the training window only (leakage-critical). ddof=0 is the
        # standard z-score convention; a zero-variance training column would divide
        # by zero, so map sigma == 0 to 1.0 (the centred column is then all zeros).
        self.mu_ = X_train.mean()
        self.sigma_ = X_train.std(ddof=0).replace(0.0, 1.0)
        X_scaled = self._standardise(X_train)

        X = sm.add_constant(X_scaled)
        ols = sm.OLS(y_train, X)
        self._fit_result = ols.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lag})
        self.params_ = self._fit_result.params
        self.bse_ = self._fit_result.bse

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.params_ is None:
            raise RuntimeError("fit() must be called before predict()")
        X_scaled = self._standardise(X_test)
        X = sm.add_constant(X_scaled, has_constant="add")
        return X[self.params_.index].values @ self.params_.values
