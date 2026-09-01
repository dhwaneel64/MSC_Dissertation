import pandas as pd

from src import config
from src.features import build_feature_matrix
from src.validation import LOCKED_FEATURE_SET, assert_feature_set_complete


def build_model_dataset(
    monthly_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "vrp",
    target_horizon: int = 1,
) -> pd.DataFrame:
    """Build a model-ready DataFrame for a given feature subset.

    Constructs prediction label y = target_col.shift(-target_horizon), selects
    only [*feature_cols, "y"], drops NaN rows, and validates minimum sample size.
    The NaN drop considers only the selected columns, so including or excluding a
    column with early or late NaNs changes the retained row range.
    """
    missing = [c for c in [*feature_cols, target_col] if c not in monthly_df.columns]
    if missing:
        raise ValueError(f"Columns not found in monthly_df: {missing}")

    if target_horizon < 1:
        raise ValueError(f"target_horizon must be >= 1, got {target_horizon}")

    working = monthly_df[feature_cols].copy()
    working["y"] = monthly_df[target_col].shift(-target_horizon)
    result = working.dropna()

    if len(result) < config.MIN_MODEL_DATASET_ROWS:
        raise ValueError(
            f"Fewer than {config.MIN_MODEL_DATASET_ROWS} rows remain after "
            f"NaN drop ({len(result)} rows)"
        )

    return result


def build_model_ready_dataset(
    vrp: pd.Series,
    vix_monthly: pd.Series,
    skew_monthly: pd.Series,
    daily_log_returns: pd.Series,
    regime_labels: pd.Series,
) -> pd.DataFrame:
    """Build the model-ready DataFrame for the full 7-column locked feature set.

    Assembles the locked feature matrix via build_feature_matrix, adds the
    one-month-ahead VRP as column "y", drops rows with any NaN (which removes
    the initial rows missing lagged VRP values), and verifies the result passes
    assert_feature_set_complete.

    Parameters
    ----------
    vrp : pd.Series
        Canonical VRP series from build_vrp_series, monthly indexed.
    vix_monthly : pd.Series
        VIX levels in vol-points at monthly dates.
    skew_monthly : pd.Series
        CBOE SKEW index values at monthly dates.
    daily_log_returns : pd.Series
        Daily SPY log-returns with a monotonic DatetimeIndex.
    regime_labels : pd.Series
        Categorical regime labels at monthly dates.

    Returns
    -------
    pd.DataFrame
        Columns: vix_level, cboe_skew, vrp_h1m, vrp_h3m, vrp_h6m,
                 realised_skew_21d, regime, y.  No NaN rows.
    """
    features = build_feature_matrix(
        vrp, vix_monthly, skew_monthly, daily_log_returns, regime_labels
    )
    features = features.copy()
    features["vrp"] = vrp.reindex(features.index)

    dataset = build_model_dataset(
        features,
        list(LOCKED_FEATURE_SET),
        target_col="vrp",
        target_horizon=1,
    )
    assert_feature_set_complete(dataset)
    return dataset
