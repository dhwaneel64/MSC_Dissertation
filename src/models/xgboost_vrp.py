"""Model 5 (final): XGBoost VRP forecaster with tune-once hyperparameters and SHAP.

This is the only nonlinear model in the sequence. It consumes all seven locked
features, including regime, which is encoded as a single ordinal column so the
tree learner can split on the market state and learn regime-dependent structure
through interactions on the full pooled sample. There is no manual per-regime
split and no per-regime tuning: regime is a feature, not a partition key.

Hyperparameters are tuned once on the initial training window (1993-2004) by
expanding-window time-series cross-validation scored on QLIKE, then locked. At
every walk-forward step the trees are refit on the expanding data with those
locked hyperparameters (Bergmeir and Benitez 2012). SHAP values are computed at
each step for regime-conditional feature importance (Objective 4).

Determinism: fits are single-threaded with a fixed seed (config.XGBOOST_SEED,
config.XGBOOST_N_JOBS) so the per-step refit is byte-reproducible, which the
leakage test relies on.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src import config
from src.results import score_walk_forward
from src.validation import LOCKED_FEATURE_SET, assert_feature_set_complete

# Six numeric features (locked set minus regime) followed by the regime switch
# column. XGBoost consumes all seven; the order is fixed so the design matrix,
# feature_importances_, and SHAP columns line up across every walk-forward step.
NUMERIC_FEATURES = tuple(f for f in LOCKED_FEATURE_SET if f != "regime")
XGB_FEATURE_ORDER = NUMERIC_FEATURES + ("regime",)

# Ordinal encoding of the regime label, in the canonical calm < normal < stressed
# order from config. A single ordinal column (not one-hot) keeps regime as one
# feature with one SHAP value, which is what the Objective-4 per-regime importance
# breakdown reports against. The ordering is monotone in market stress, so a tree
# split on the ordinal is interpretable as a split on the regime boundary.
REGIME_TO_ORDINAL = {label: i for i, label in enumerate(config.REGIME_LABELS)}


def encode_features(
    X: pd.DataFrame, feature_cols=XGB_FEATURE_ORDER, regime_col: str = "regime"
) -> pd.DataFrame:
    """Return X reduced to feature_cols with the regime column ordinal-encoded.

    Numeric features are passed through unchanged; the regime label column is
    mapped to its integer code via REGIME_TO_ORDINAL. Column order follows
    feature_cols exactly. Raises if the regime column holds an unknown label, so
    a silent miscode cannot occur.
    """
    feature_cols = list(feature_cols)
    out = X.reindex(columns=feature_cols).copy()
    if regime_col in feature_cols:
        labels = out[regime_col].astype(str)
        unknown = sorted(set(labels) - set(REGIME_TO_ORDINAL))
        if unknown:
            raise ValueError(f"Unknown regime label(s) for ordinal encoding: {unknown}")
        out[regime_col] = labels.map(REGIME_TO_ORDINAL).astype(int)
    return out


def default_grid() -> list[dict]:
    """Cartesian product of the config hyperparameter grids, as a list of dicts.

    Each dict has keys max_depth, learning_rate, n_estimators, min_child_weight.
    The order is deterministic (itertools.product over the config tuples), so the
    tuning selection is reproducible and the grid carries no notebook literals.
    """
    grid = []
    for max_depth, lr, n_est, mcw in itertools.product(
        config.XGB_MAX_DEPTH_GRID,
        config.XGB_LEARNING_RATE_GRID,
        config.XGB_N_ESTIMATORS_GRID,
        config.XGB_MIN_CHILD_WEIGHT_GRID,
    ):
        grid.append(
            {
                "max_depth": int(max_depth),
                "learning_rate": float(lr),
                "n_estimators": int(n_est),
                "min_child_weight": int(mcw),
            }
        )
    return grid


class XGBoostVRPModel:
    """Gradient-boosted-tree VRP forecaster on the seven locked features.

    fit() trains on the supplied training window with the locked hyperparameters;
    predict() scores the test rows; shap_values() returns per-feature SHAP
    contributions for the supplied rows under the fitted trees. The regime label
    is ordinal-encoded inside the model (encode_features); callers pass the raw
    seven-column feature frame.

    Attributes after fit:
        model_: the fitted xgboost.XGBRegressor.
        expected_value_: SHAP base value (set lazily on the first shap_values call).
    """

    def __init__(
        self,
        monthly_vrp: pd.DataFrame,
        hyperparams: dict,
        feature_cols=XGB_FEATURE_ORDER,
        regime_col: str = "regime",
        seed: int = config.XGBOOST_SEED,
    ) -> None:
        self._monthly_vrp = monthly_vrp
        self._hyperparams = dict(hyperparams)
        self._feature_cols = list(feature_cols)
        self._regime_col = regime_col
        self._seed = seed
        self.model_: xgb.XGBRegressor | None = None
        self.expected_value_: float | None = None
        self._explainer = None

    def _encode(self, X: pd.DataFrame) -> pd.DataFrame:
        return encode_features(X, self._feature_cols, self._regime_col)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        assert_feature_set_complete(self._monthly_vrp)
        X_enc = self._encode(X_train)
        self.model_ = xgb.XGBRegressor(
            random_state=self._seed,
            n_jobs=config.XGBOOST_N_JOBS,
            tree_method="hist",
            verbosity=0,
            **self._hyperparams,
        )
        self.model_.fit(X_enc, np.asarray(y_train, dtype=float))
        self._explainer = None
        self.expected_value_ = None

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("fit() must be called before predict()")
        return np.asarray(self.model_.predict(self._encode(X_test)), dtype=float)

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Per-feature SHAP contributions for X under the fitted trees.

        Returns an array of shape (len(X), len(feature_cols)) whose columns align
        to self._feature_cols. The tree-path-dependent explainer is exact for the
        fitted booster, so expected_value_ + row-sum of the SHAP values reproduces
        predict() to numerical precision.
        """
        if self.model_ is None:
            raise RuntimeError("fit() must be called before shap_values()")
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model_)
            self.expected_value_ = float(np.asarray(self._explainer.expected_value).ravel()[0])
        values = self._explainer.shap_values(self._encode(X))
        return np.asarray(values, dtype=float)


@dataclass
class XGBTuningResult:
    """Outcome of the tune-once hyperparameter search."""
    best_params: dict
    best_qlike: float
    grid_scores: pd.DataFrame  # one row per candidate, sorted by mean_val_qlike ascending
    n_folds: int
    fold_val_spans: list[tuple]  # (first_val_date, last_val_date) per fold
    seed: int


def _expanding_cv_folds(n_rows: int, min_train: int, n_folds: int) -> list[tuple[int, int, int]]:
    """Expanding-window CV fold boundaries as (train_end_pos, val_start, val_end).

    The final dataset row is never used as a validation point: its target month
    t+1 falls outside the training window, so it has no in-window VIX to score
    against. Validation positions therefore span [min_train, n_rows - 1) split
    into n_folds contiguous blocks; fold k trains on rows [0, val_start) and
    validates on [val_start, val_end).
    """
    last_val = n_rows - 1  # exclusive upper bound: drop the final (unscoreable) row
    if last_val - min_train < n_folds:
        raise ValueError(
            f"Not enough rows for CV: need > {min_train + n_folds}, have {n_rows}"
        )
    edges = np.linspace(min_train, last_val, n_folds + 1).astype(int)
    folds = []
    for k in range(n_folds):
        val_start, val_end = int(edges[k]), int(edges[k + 1])
        if val_end <= val_start:
            continue
        folds.append((val_start, val_start, val_end))
    return folds


def tune_xgboost_hyperparameters(
    dataset_train: pd.DataFrame,
    daily_log_returns: pd.Series,
    vix_monthly: pd.Series,
    feature_cols=XGB_FEATURE_ORDER,
    monthly_vrp: pd.DataFrame | None = None,
    grid: list[dict] | None = None,
    n_folds: int = config.XGB_CV_FOLDS,
    min_train: int = config.XGB_CV_MIN_TRAIN_MONTHS,
    seed: int = config.XGBOOST_SEED,
    target_col: str = "y",
) -> XGBTuningResult:
    """Tune-once XGBoost hyperparameters by expanding-window CV scored on QLIKE.

    dataset_train must already be restricted to the initial training window
    (index <= initial_train_end). To guarantee the selection is a pure function
    of in-window data, daily_log_returns and vix_monthly are truncated internally
    to the training boundary, so no out-of-sample value (returns, VIX, or target)
    can influence the choice. Validation months near the boundary whose forward
    realised-variance window is not fully in-window are nan-tail-excluded from
    QLIKE by the shared scorer, exactly as in the live walk-forward.

    Each grid candidate is fit on each fold's expanding training block, predicted
    on that fold's validation block, and scored on QLIKE through
    src.results.score_walk_forward (the single shared scorer, no parallel QLIKE
    path). The candidate's score is the mean QLIKE across folds that produced at
    least one valid (guard-passing, observable-target) validation month. The
    candidate with the lowest mean validation QLIKE is selected and returned.

    Returns an XGBTuningResult; grid_scores is sorted ascending by mean_val_qlike.
    """
    if not dataset_train.index.is_monotonic_increasing:
        raise ValueError("dataset_train.index must be monotonic increasing")
    feature_cols = list(feature_cols)
    if monthly_vrp is None:
        monthly_vrp = dataset_train
    if grid is None:
        grid = default_grid()

    boundary = dataset_train.index[-1]
    # Truncate to the training boundary: tuning must not see any OOS data.
    dlr = daily_log_returns.loc[:boundary]
    vix_next_full = vix_monthly.loc[:boundary].shift(-1)

    n_rows = len(dataset_train)
    folds = _expanding_cv_folds(n_rows, min_train, n_folds)
    fold_val_spans = [
        (dataset_train.index[vs], dataset_train.index[ve - 1]) for (_, vs, ve) in folds
    ]

    records = []
    for params in grid:
        fold_qlikes = []
        for (val_start, _, val_end) in folds:
            train = dataset_train.iloc[:val_start]
            val = dataset_train.iloc[val_start:val_end]

            model = XGBoostVRPModel(monthly_vrp, params, feature_cols, seed=seed)
            model.fit(train[feature_cols], train[target_col])
            y_pred = model.predict(val[feature_cols])

            wf_cv = pd.DataFrame(
                {"y_true": val[target_col].to_numpy(dtype=float), "y_pred": y_pred},
                index=val.index,
            )
            vix_next_cv = vix_next_full.reindex(val.index)
            fold_score = score_walk_forward(wf_cv, vix_next_cv, dlr)
            if fold_score["qlike_n"] > 0:
                fold_qlikes.append(fold_score["qlike"])

        mean_qlike = float(np.mean(fold_qlikes)) if fold_qlikes else float("inf")
        records.append({**params, "mean_val_qlike": mean_qlike, "n_folds_scored": len(fold_qlikes)})

    grid_scores = (
        pd.DataFrame(records)
        .sort_values("mean_val_qlike", kind="stable")
        .reset_index(drop=True)
    )
    best = grid_scores.iloc[0]
    best_params = {
        "max_depth": int(best["max_depth"]),
        "learning_rate": float(best["learning_rate"]),
        "n_estimators": int(best["n_estimators"]),
        "min_child_weight": int(best["min_child_weight"]),
    }
    return XGBTuningResult(
        best_params=best_params,
        best_qlike=float(best["mean_val_qlike"]),
        grid_scores=grid_scores,
        n_folds=n_folds,
        fold_val_spans=fold_val_spans,
        seed=seed,
    )


def walk_forward_with_shap(
    dataset: pd.DataFrame,
    feature_cols,
    hyperparams: dict,
    initial_train_end: pd.Timestamp,
    monthly_vrp: pd.DataFrame | None = None,
    seed: int = config.XGBOOST_SEED,
    target_col: str = "y",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expanding-window walk-forward for XGBoost, returning predictions and SHAP.

    Steps identically to src.walk_forward.walk_forward (same iloc slicing, same
    one-step-ahead test row), so the prediction frame is byte-identical to running
    the shared engine with an XGBoostVRPModel factory. At each step the trees are
    refit on the expanding window with the locked hyperparameters (no re-tuning),
    and the SHAP values for that step's single test row are recorded from the
    same fitted model.

    Returns:
        (wf_result, shap_df).
        wf_result: columns y_true, y_pred, indexed by the OOS dates.
        shap_df: per-step SHAP contributions, columns = feature_cols, indexed by
            the OOS dates. expected_value + row-sum reproduces y_pred.
    """
    if not dataset.index.is_monotonic_increasing:
        raise ValueError("dataset.index must be monotonic increasing")
    if initial_train_end not in dataset.index:
        raise ValueError(f"initial_train_end {initial_train_end} is not in dataset.index")
    feature_cols = list(feature_cols)
    if monthly_vrp is None:
        monthly_vrp = dataset

    pos = dataset.index.get_loc(initial_train_end)
    if pos >= len(dataset) - 1:
        raise ValueError(
            f"initial_train_end {initial_train_end} is the last row; no OOS rows available"
        )

    pred_records = []
    shap_rows = []
    shap_index = []
    for i in range(pos, len(dataset) - 1):
        train = dataset.iloc[: i + 1]
        test = dataset.iloc[i + 1 : i + 2]

        model = XGBoostVRPModel(monthly_vrp, hyperparams, feature_cols, seed=seed)
        model.fit(train[feature_cols], train[target_col])
        y_pred = model.predict(test[feature_cols])
        shap_row = model.shap_values(test[feature_cols])[0]

        pred_records.append(
            {
                "date": test.index[0],
                "y_true": float(test[target_col].iloc[0]),
                "y_pred": float(y_pred[0]),
            }
        )
        shap_rows.append(shap_row)
        shap_index.append(test.index[0])

    wf_result = pd.DataFrame(pred_records).set_index("date")
    wf_result.index.name = dataset.index.name
    shap_df = pd.DataFrame(shap_rows, index=pd.DatetimeIndex(shap_index, name=dataset.index.name), columns=feature_cols)
    return wf_result, shap_df


def mean_abs_shap_by_regime(
    shap_df: pd.DataFrame,
    regime_labels: pd.Series,
) -> pd.DataFrame:
    """Mean absolute SHAP per feature, broken down by regime (Objective 4).

    Args:
        shap_df: per-step SHAP contributions, columns = features, indexed by OOS dates.
        regime_labels: regime label per OOS date, aligned to shap_df.index.

    Returns:
        DataFrame indexed by feature name, with one column per regime in the
        canonical config.REGIME_LABELS order (regimes absent from the OOS sample
        are omitted) plus an "all" column over every OOS month. Values are the
        mean of |SHAP| within each group, the standard global feature-importance
        summary applied per market state.
    """
    regimes = regime_labels.reindex(shap_df.index).astype(str)
    abs_shap = shap_df.abs()

    cols = {}
    for regime in config.REGIME_LABELS:
        mask = (regimes == regime).to_numpy()
        if mask.any():
            cols[regime] = abs_shap.loc[mask].mean()
    cols["all"] = abs_shap.mean()
    table = pd.DataFrame(cols)
    table.index.name = "feature"
    return table


def shap_rank_changes(importance_table: pd.DataFrame, regimes=None) -> pd.DataFrame:
    """Feature importance ranks per regime and whether each feature's rank shifts.

    Args:
        importance_table: output of mean_abs_shap_by_regime (features x regimes).
        regimes: subset of columns to rank over. Defaults to the regime columns
            present in importance_table (config.REGIME_LABELS order), excluding "all".

    Returns:
        DataFrame indexed by feature, columns = the per-regime ranks (1 = most
        important in that regime) plus "rank_changes" (True where the feature does
        not hold the same rank in every ranked regime). Rank instability across
        regimes is the Objective-4 signal that different features drive predictions
        in different market states.
    """
    if regimes is None:
        regimes = [r for r in config.REGIME_LABELS if r in importance_table.columns]
    ranks = importance_table[list(regimes)].rank(ascending=False, method="min").astype(int)
    ranks.columns = [f"rank_{r}" for r in regimes]
    ranks["rank_changes"] = ranks.nunique(axis=1) > 1
    return ranks
