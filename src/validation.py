import pandas as pd

from src import config

# Horizon-to-target names for the VRP predictor columns. The number in each name is
# the distance in months from that column to the target on the same row, not a lag
# measured from t. On a row whose target is VRP(t+1), vrp_h1m carries VRP(t),
# vrp_h3m carries VRP(t-2) and vrp_h6m carries VRP(t-5). The columns are built from
# config.HAR_LAGS_MONTHS so the names and the horizons cannot drift apart.
VRP_HORIZON_COLS = tuple(f"vrp_h{k}m" for k in config.HAR_LAGS_MONTHS)

LOCKED_FEATURE_SET = (
    "vix_level", "cboe_skew", *VRP_HORIZON_COLS, "realised_skew_21d", "regime",
)
# Single source of truth for the locked feature set per the locked methodology. Any drift from
# this is a methodology error, not a stylistic choice.


def assert_feature_set_complete(
    df: pd.DataFrame,
    expected_features: tuple[str, ...] = LOCKED_FEATURE_SET,
) -> None:
    """Raise ValueError naming any feature in expected_features missing from df.columns."""
    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing locked features in df: {missing}")


def print_verification_block(
    df: pd.DataFrame,
    feature_cols: list[str],
    config_values: dict | None = None,
) -> None:
    """Print the fixed-format verification block for the current model-ready dataset."""
    print("--- VERIFICATION BLOCK ---")
    print(f"Locked feature set (from context.md): {list(LOCKED_FEATURE_SET)}")
    print(f"Features used in this task: {feature_cols}")
    print("Locked features present in df:")
    for f in LOCKED_FEATURE_SET:
        status = "OK" if f in df.columns else "MISSING"
        print(f"  {f}: {status}")
    excluded = [f for f in LOCKED_FEATURE_SET if f not in feature_cols]
    print(
        f"Locked features absent from feature list "
        f"(intentional exclusions, e.g. HAR): {excluded}"
    )
    if config_values is not None:
        print("Config values referenced:")
        for k, v in config_values.items():
            print(f"  {k}: {v}")
    print("--- END VERIFICATION BLOCK ---")
