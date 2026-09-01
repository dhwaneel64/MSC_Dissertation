"""Guards for the corrected Clark-West machinery.

Two properties are pinned, both demonstrated rather than argued:
  1. verify_frame_reconstruction is falsifiable in every one of the eight
     dataset columns. The pre-fix version rebuilt the exogenous columns and the
     regime label from the frame under test, so a perturbation in four of the
     eight columns could never fire the guard.
  2. one_draw absorbs only the legitimate statistical failure classes
     (ValueError, LinAlgError). A KeyError raised inside a draw is a coding bug
     and must crash the run, not become a silently counted failed draw.
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src import corrected_cw as cc
from src.dataset import build_model_ready_dataset
from src.regimes import label_regimes
from src.validation import LOCKED_FEATURE_SET
from src.vrp import build_vrp_series, resample_to_month_start

# Every dataset column, addressed through the locked constant so a rename cannot
# orphan this guard (and the literal-name tripwire in test_leakage_gate holds).
ALL_DATASET_COLUMNS = tuple(LOCKED_FEATURE_SET) + ("y",)


def _observed_chain(n_days: int = 1400, seed: int = 23) -> dict:
    """Synthetic observed dict shaped like slim_observed's output, built through
    the same pipeline functions the real inputs come through."""
    rng = np.random.default_rng(seed)
    day_idx = pd.bdate_range("2000-01-03", periods=n_days)
    returns = pd.Series(rng.normal(0, 0.01, n_days), index=day_idx, name="log_return")
    vix_daily = pd.Series(rng.uniform(12.0, 34.0, n_days), index=day_idx, name="close")
    skew_daily = pd.Series(rng.uniform(110.0, 145.0, n_days), index=day_idx, name="close")
    vix_m = resample_to_month_start(vix_daily)
    skew_m = resample_to_month_start(skew_daily)
    regimes = label_regimes(vix_m)
    vrp = build_vrp_series(vix_m, returns, vix_m.index)
    dataset = build_model_ready_dataset(vrp, vix_m, skew_m, returns, regimes)
    return {
        "dataset": dataset,
        "vrp": vrp,
        "vix_monthly": vix_m,
        "skew_monthly": skew_m,
        "spy_returns": returns,
    }


@pytest.fixture(scope="module")
def observed():
    return _observed_chain()


def test_verify_frame_reconstruction_passes_on_clean_data(observed):
    cc.verify_frame_reconstruction(observed)


@pytest.mark.parametrize("col", ALL_DATASET_COLUMNS)
def test_verify_frame_reconstruction_fires_on_every_column(observed, col):
    """Perturbing any single dataset column raises: the guard is 8/8 falsifiable."""
    bad = dict(observed)
    frame = observed["dataset"].copy()
    row = frame.index[len(frame) // 2]
    if col == "regime":
        current = str(frame.at[row, "regime"])
        replacement = "stressed" if current != "stressed" else "calm"
        frame.loc[row, "regime"] = replacement
    else:
        frame.loc[row, col] = float(frame.at[row, col]) + 1.0
    bad["dataset"] = frame
    with pytest.raises(ValueError, match=col):
        cc.verify_frame_reconstruction(bad)


class _KeyErrorGenerator:
    """Simulate() raising KeyError: the shape of a coding bug, not a failed draw."""
    pair = cc.NESTED_PAIRS[0]

    def simulate(self, rng):
        raise KeyError("renamed_spec_key")


class _ValueErrorGenerator:
    """Simulate() raising ValueError: the legitimate statistical failure class."""
    pair = cc.NESTED_PAIRS[0]

    def simulate(self, rng):
        raise ValueError("guard-type failure")


def test_one_draw_propagates_keyerror():
    with pytest.raises(KeyError, match="renamed_spec_key"):
        cc.one_draw(_KeyErrorGenerator(), {}, np.random.SeedSequence(0))


def test_one_draw_still_absorbs_valueerror():
    stat, n_paired = cc.one_draw(_ValueErrorGenerator(), {}, np.random.SeedSequence(0))
    assert np.isnan(stat)
    assert n_paired == 0


def test_module_constants_are_config_bound():
    """The de-duplicated constants cannot drift from their config source."""
    assert cc.BOOTSTRAP_DRAWS == config.BOOTSTRAP_REPLICATIONS
    assert cc.MIN_PAIRED_MONTHS == config.MIN_COMPARISON_OBS
    assert cc.CW_ADJ_SCALE == config.CW_ADJ_SCALE
