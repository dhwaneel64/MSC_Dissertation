import pandas as pd
import pytest

from src import config
from src.validation import (
    LOCKED_FEATURE_SET,
    VRP_HORIZON_COLS,
    assert_feature_set_complete,
    print_verification_block,
)


def _full_df() -> pd.DataFrame:
    return pd.DataFrame({f: [1.0] * 5 for f in LOCKED_FEATURE_SET})


# ── LOCKED_FEATURE_SET contents ───────────────────────────────────────────────

def test_locked_feature_set_is_exact_7_column_tuple():
    """Pin the 7-column structure so any drift is caught immediately.

    The VRP columns are pinned through VRP_HORIZON_COLS rather than by literal name,
    so a rename moves both together, but their count, position and encoded horizons
    are still asserted: one column per entry in config.HAR_LAGS_MONTHS, each naming
    its own horizon to the target.
    """
    assert LOCKED_FEATURE_SET == (
        "vix_level", "cboe_skew", *VRP_HORIZON_COLS,
        "realised_skew_21d", "regime",
    )
    assert len(VRP_HORIZON_COLS) == len(config.HAR_LAGS_MONTHS)
    for k, col in zip(config.HAR_LAGS_MONTHS, VRP_HORIZON_COLS):
        assert str(k) in col, f"{col} does not name its horizon {k}"


def test_locked_feature_set_has_7_elements():
    assert len(LOCKED_FEATURE_SET) == 7


# ── assert_feature_set_complete ───────────────────────────────────────────────

def test_passes_silently_when_all_features_present():
    assert_feature_set_complete(_full_df())


def test_passes_when_df_has_extra_columns():
    """Extra columns such as 'y' do not cause a failure."""
    df = _full_df()
    df["y"] = 1.0
    assert_feature_set_complete(df)


def test_raises_naming_missing_feature():
    df = _full_df().drop(columns=["regime"])
    with pytest.raises(ValueError, match="regime"):
        assert_feature_set_complete(df)


def test_raises_naming_multiple_missing_features():
    df = _full_df().drop(columns=["regime", "cboe_skew"])
    with pytest.raises(ValueError) as exc_info:
        assert_feature_set_complete(df)
    msg = str(exc_info.value)
    assert "regime" in msg
    assert "cboe_skew" in msg


# ── print_verification_block ──────────────────────────────────────────────────

def test_output_contains_verification_block_marker(capsys):
    print_verification_block(_full_df(), list(LOCKED_FEATURE_SET))
    out = capsys.readouterr().out
    assert "VERIFICATION BLOCK" in out


def test_output_lists_each_locked_feature(capsys):
    print_verification_block(_full_df(), list(LOCKED_FEATURE_SET))
    out = capsys.readouterr().out
    for f in LOCKED_FEATURE_SET:
        assert f in out
