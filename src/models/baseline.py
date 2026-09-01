import numpy as np
import pandas as pd


class ConstantMeanModel:
    """Model 1 in the locked sequence: rolling historical mean of VRP up to time T.

    At each walk-forward step the model is fit on VRP observations 1..T and
    predicts the same constant, mean(VRP[1..T]), for T+1.
    """

    def __init__(self) -> None:
        self._mean: float | None = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        self._mean = float(y_train.mean())

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("fit() must be called before predict()")
        return np.full(len(X_test), self._mean)
