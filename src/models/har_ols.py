import math

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src import config
from src.validation import assert_feature_set_complete


class HAROLSModel:
    """Model 2: HAR-style OLS on lagged VRP at 1, 3, and 6 months.

    Refit at every walk-forward step. Standard errors use Newey-West HAC with
    lag = floor(NW_HAC_MULTIPLIER * (T / 100) ** NW_HAC_EXPONENT) computed
    from the current training-set size T. Intercept added by statsmodels.

    Attributes after fit:
        params_: fitted coefficients (const + three lags).
        bse_: HAC standard errors for each coefficient.
    """

    def __init__(self, monthly_vrp: pd.DataFrame) -> None:
        self._monthly_vrp = monthly_vrp
        self.params_: pd.Series | None = None
        self.bse_: pd.Series | None = None
        self._fit_result = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        assert_feature_set_complete(self._monthly_vrp)

        T = len(y_train)
        hac_lag = math.floor(
            config.NW_HAC_MULTIPLIER * (T / 100) ** config.NW_HAC_EXPONENT
        )

        X = sm.add_constant(X_train)
        ols = sm.OLS(y_train, X)
        self._fit_result = ols.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lag})
        self.params_ = self._fit_result.params
        self.bse_ = self._fit_result.bse

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.params_ is None:
            raise RuntimeError("fit() must be called before predict()")
        X = sm.add_constant(X_test, has_constant="add")
        return X[self.params_.index].values @ self.params_.values
