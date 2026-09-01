"""Tests for regime-conditional model comparison (Objective 3).

Two required properties:
  1. Per pair, the three regime n_paired sum to the full paired n (the regimes
     partition the common-valid subset of the OOS index).
  2. The regime mask is dataset["regime"] read at the decision date t on wf.index.
"""
import numpy as np
import pandas as pd

from src import config
from src.regime_comparison import (
    _regime_labels,
    compare_models_by_regime,
    per_regime_paired_qlike,
    regime_qlike_decomposition,
)
from src.results import score_walk_forward
from src.vrp import resample_to_month_start


# ---------------------------------------------------------------------------
# Fixtures: synthetic but valid inputs for the shared scorer.
# ---------------------------------------------------------------------------

def _make_inputs(seed: int = 11):
    """Random daily log-returns, a monthly VIX series, an interior OOS window,
    a dataset whose regime labels are assigned independently of VIX (so the test
    can verify the function reads the label column, not a recomputed VIX rule),
    and five walk-forward frames sharing one OOS index.
    """
    rng = np.random.default_rng(seed)
    returns = pd.Series(
        rng.normal(0.0, 0.01, 1500),
        index=pd.bdate_range("2000-01-03", periods=1500),
        name="log_return",
    )
    monthly_idx = resample_to_month_start(returns).index

    # VIX flat at 20 everywhere: guard never trips (implied = 0.04 - y_pred > 0)
    # and shift(-1) is defined on every interior OOS date.
    vix_monthly = pd.Series(20.0, index=monthly_idx, name="vix")

    # Interior OOS window: full forward realised windows, t+1 always defined.
    oos_index = monthly_idx[12:60]
    n = len(oos_index)

    # Regime labels cycle over the canonical set, deliberately unrelated to VIX,
    # giving each regime >= 8 months on the OOS window.
    regime_col = pd.Series(
        [config.REGIME_LABELS[i % len(config.REGIME_LABELS)] for i in range(len(monthly_idx))],
        index=monthly_idx,
        name="regime",
    )
    dataset = pd.DataFrame({"regime": regime_col})

    def _frame(offset):
        # Small positive y_pred with variation so per-obs QLIKE (and thus the CW
        # f_t / DM d_t series) is non-degenerate; implied variance stays positive.
        y_pred = 0.003 + offset + rng.uniform(0.0, 0.004, n)
        y_true = rng.uniform(0.0, 0.02, n)
        return pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=oos_index)

    wf_frames = {
        "constant": _frame(0.0000),
        "har": _frame(0.0005),
        "extended_ols": _frame(0.0010),
        "regime_switching": _frame(0.0015),
        "xgboost": _frame(0.0020),
    }
    return wf_frames, vix_monthly, returns, dataset, oos_index


_PAIRS = [
    ("constant", "har"),
    ("har", "extended_ols"),
    ("extended_ols", "regime_switching"),
    ("xgboost", "extended_ols"),
]


# ---------------------------------------------------------------------------
# Property 1: regime n_paired partition the full paired subset.
# ---------------------------------------------------------------------------

def test_regime_n_paired_sum_to_full_paired_n_per_pair():
    wf_frames, vix_monthly, returns, dataset, _ = _make_inputs()
    tests = compare_models_by_regime(wf_frames, vix_monthly, returns, dataset, _PAIRS)

    for m_a, m_b in _PAIRS:
        score_a = score_walk_forward(
            wf_frames[m_a], vix_monthly.shift(-1).reindex(wf_frames[m_a].index), returns
        )
        score_b = score_walk_forward(
            wf_frames[m_b], vix_monthly.shift(-1).reindex(wf_frames[m_b].index), returns
        )
        full_paired = int((score_a["valid_mask"] & score_b["valid_mask"]).sum())

        pair_rows = tests[tests["pair"] == f"{m_a} vs {m_b}"]
        assert len(pair_rows) == len(config.REGIME_LABELS)
        assert int(pair_rows["n_paired"].sum()) == full_paired


def test_each_pair_routes_via_is_nested():
    wf_frames, vix_monthly, returns, dataset, _ = _make_inputs()
    tests = compare_models_by_regime(wf_frames, vix_monthly, returns, dataset, _PAIRS)

    nested = tests[tests["pair"] != "xgboost vs extended_ols"]
    non_nested = tests[tests["pair"] == "xgboost vs extended_ols"]
    assert (nested["test_type"] == "clark_west_bootstrap").all()
    assert (non_nested["test_type"] == "diebold_mariano").all()
    # Block length is recomputed per regime for the nested (bootstrap) rows only.
    assert nested["block_length"].notna().all()
    assert non_nested["block_length"].isna().all()


# ---------------------------------------------------------------------------
# Property 2: regime mask equals dataset["regime"] at the decision date t.
# ---------------------------------------------------------------------------

def test_regime_mask_equals_dataset_labels_on_wf_index():
    wf_frames, _, _, dataset, oos_index = _make_inputs()
    expected = dataset.loc[oos_index, "regime"].astype(str).to_numpy()
    np.testing.assert_array_equal(_regime_labels(dataset, oos_index), expected)


def test_per_regime_n_paired_matches_label_and_valid_intersection():
    wf_frames, vix_monthly, returns, dataset, oos_index = _make_inputs()
    tests = compare_models_by_regime(wf_frames, vix_monthly, returns, dataset, _PAIRS)
    labels = dataset.loc[oos_index, "regime"].astype(str).to_numpy()

    m_a, m_b = ("constant", "har")
    score_a = score_walk_forward(
        wf_frames[m_a], vix_monthly.shift(-1).reindex(wf_frames[m_a].index), returns
    )
    score_b = score_walk_forward(
        wf_frames[m_b], vix_monthly.shift(-1).reindex(wf_frames[m_b].index), returns
    )
    both_valid = score_a["valid_mask"] & score_b["valid_mask"]

    pair_rows = tests[tests["pair"] == f"{m_a} vs {m_b}"].set_index("regime")
    for regime in config.REGIME_LABELS:
        expected_n = int(((labels == regime) & both_valid).sum())
        assert int(pair_rows.loc[regime, "n_paired"]) == expected_n


# ---------------------------------------------------------------------------
# Low-power flag and the auxiliary tables.
# ---------------------------------------------------------------------------

def test_low_power_flag_tracks_threshold():
    wf_frames, vix_monthly, returns, dataset, _ = _make_inputs()
    tests = compare_models_by_regime(wf_frames, vix_monthly, returns, dataset, _PAIRS)
    expected = tests["n_paired"] < config.LOW_POWER_MIN_N
    pd.testing.assert_series_equal(tests["low_power"], expected, check_names=False)


def test_per_regime_paired_qlike_partitions_all_regimes():
    wf_frames, vix_monthly, returns, dataset, _ = _make_inputs()
    table = per_regime_paired_qlike(wf_frames, vix_monthly, returns, dataset)
    regime_total = int(table.loc[list(config.REGIME_LABELS), "n_paired"].sum())
    assert regime_total == int(table.loc["all_regimes", "n_paired"])


def test_winner_is_raw_qlike_argmin_for_every_row():
    wf_frames, vix_monthly, returns, dataset, _ = _make_inputs()
    tests = compare_models_by_regime(wf_frames, vix_monthly, returns, dataset, _PAIRS)
    for _, row in tests.iterrows():
        m_a, m_b = row["pair"].split(" vs ")
        if row["qlike_smaller"] < row["qlike_larger"]:
            expected = m_a
        elif row["qlike_larger"] < row["qlike_smaller"]:
            expected = m_b
        else:
            expected = "none"
        assert row["winner"] == expected


def _make_conflict_inputs():
    """A near-null nested case where the bootstrap Clark-West direction (larger
    model wins) contradicts the raw-QLIKE winner (smaller model has lower loss).
    Constant returns give a constant realised target; both models' implied variance
    sit close to it with small independent noise, so QLIKE differences are tiny and
    the CW adjustment term drives the centering artifact. Seed 1 lands the conflict
    in the calm block with a bootstrap p-value far below alpha.
    """
    r = 0.01
    returns = pd.Series(
        r, index=pd.bdate_range("2000-01-03", periods=1500), name="log_return"
    )
    monthly_idx = resample_to_month_start(returns).index
    vix_monthly = pd.Series(20.0, index=monthly_idx, name="vix")
    oos_index = monthly_idx[12:60]
    n = len(oos_index)

    c = r ** 2 * config.ANNUALISATION_FACTOR_DAILY
    vix_var = 20.0 ** 2 / config.VIX_VARIANCE_SCALE

    n3 = n // 3
    labels = np.array(["calm"] * n3 + ["normal"] * n3 + ["stressed"] * (n - 2 * n3))
    dataset = pd.DataFrame({"regime": labels}, index=oos_index)

    rng = np.random.default_rng(1)
    implied_a = c + rng.normal(0, 0.0015, n)
    implied_b = c + rng.normal(0, 0.0015, n)
    wf_a = pd.DataFrame(
        {"y_true": np.full(n, 0.01), "y_pred": vix_var - implied_a}, index=oos_index
    )
    wf_b = pd.DataFrame(
        {"y_true": np.full(n, 0.01), "y_pred": vix_var - implied_b}, index=oos_index
    )
    # ("constant", "har") is nested, so the row routes to bootstrap Clark-West.
    return {"constant": wf_a, "har": wf_b}, vix_monthly, returns, dataset


def test_winner_follows_raw_qlike_when_cw_direction_disagrees():
    wf_frames, vix_monthly, returns, dataset = _make_conflict_inputs()
    tests = compare_models_by_regime(
        wf_frames, vix_monthly, returns, dataset, [("constant", "har")]
    )
    calm = tests[tests["regime"] == "calm"].iloc[0]

    # CW direction: the larger model (har) significantly wins on the adjusted loss.
    assert calm["test_type"] == "clark_west_bootstrap"
    assert calm["p_value"] < config.COMPARISON_ALPHA
    # Raw QLIKE: the smaller model (constant) has the lower loss.
    assert calm["qlike_smaller"] < calm["qlike_larger"]
    # Winner is the raw-QLIKE argmin, not the CW-favoured larger model.
    assert calm["winner"] == "constant"
    # The disagreement is flagged, not suppressed.
    assert bool(calm["cw_qlike_direction_conflict"]) is True


def test_no_conflict_flag_when_cw_agrees_with_raw_qlike():
    # Random benign inputs: most rows have the test either insignificant or pointing
    # the same way as raw QLIKE. The flag must never fire against the raw winner.
    wf_frames, vix_monthly, returns, dataset, _ = _make_inputs()
    tests = compare_models_by_regime(wf_frames, vix_monthly, returns, dataset, _PAIRS)
    for _, row in tests.iterrows():
        if row["cw_qlike_direction_conflict"]:
            # A flagged row must have the larger model winning the nested test while
            # raw QLIKE prefers the smaller one.
            m_a, m_b = row["pair"].split(" vs ")
            assert row["winner"] == m_a
            assert row["test_type"] == "clark_west_bootstrap"


def test_decomposition_returns_single_row_with_share_in_unit_interval():
    wf_frames, vix_monthly, returns, dataset, _ = _make_inputs()
    decomp = regime_qlike_decomposition(
        wf_frames["xgboost"], vix_monthly, returns, dataset,
        regime_label="stressed", model_name="xgboost",
    )
    assert len(decomp) == 1
    share = float(decomp["largest_month_qlike_share"].iloc[0])
    assert 0.0 < share <= 1.0
    assert decomp["largest_month_regime"].iloc[0] == "stressed"
