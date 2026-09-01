import numpy as np


def _validate(y_true: np.ndarray, y_pred: np.ndarray, require_positive: bool = False) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Length mismatch: y_true has {y_true.shape}, y_pred has {y_pred.shape}"
        )
    if np.any(np.isnan(y_true)):
        raise ValueError("y_true contains NaN")
    if np.any(np.isnan(y_pred)):
        raise ValueError("y_pred contains NaN")
    if require_positive:
        if np.any(y_true <= 0):
            raise ValueError("y_true must be strictly positive; got values <= 0")
        if np.any(y_pred <= 0):
            raise ValueError("y_pred must be strictly positive; got values <= 0")


def qlike_per_obs(y_true, y_pred) -> np.ndarray:
    """Per-observation QLIKE losses. Same input rules as qlike (positive variance only)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _validate(y_true, y_pred, require_positive=True)
    ratio = y_true / y_pred
    return ratio - np.log(ratio) - 1


def qlike(y_true, y_pred) -> float:

    return float(np.mean(qlike_per_obs(y_true, y_pred)))


def mse_per_obs(y_true, y_pred) -> np.ndarray:
    """Per-observation squared errors. Same input rules as mse."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _validate(y_true, y_pred)
    return (y_true - y_pred) ** 2


def mse(y_true, y_pred) -> float:
    """Mean squared error. Raises on NaN or length mismatch."""
    return float(np.mean(mse_per_obs(y_true, y_pred)))


def directional_accuracy(y_true, y_pred) -> float:
    """Fraction of observations where sign(y_true) == sign(y_pred).
    Convention: sign(0) treated as matching either sign (returns True).
    Returns a value in [0, 1]. Raises on NaN or length mismatch.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    _validate(y_true, y_pred)
    match = (np.sign(y_true) == np.sign(y_pred)) | (y_true == 0) | (y_pred == 0)
    return float(np.mean(match))
