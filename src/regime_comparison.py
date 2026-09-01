"""Regime-conditional model comparison (Objective 3, final empirical task).

The full-sample model-comparison cells establish each pair's verdict over the whole
out-of-sample window. This module repeats those same paired comparisons within each
market regime classified at the decision date t (calm, normal, stressed), to test
whether a model's QLIKE advantage is concentrated in one market state rather than
spread evenly.

No rescoring happens here. score_walk_forward owns the only variance-space QLIKE
conversion and forward realised target; this module calls it once per model and
slices the exposed implied_variance / realised_variance_next / valid_mask by regime.
The regime label is taken at the decision date t via dataset.loc[wf.index, "regime"]
(the same idiom as the SHAP-by-regime cell), never the t+1 label. Each pair routes
through the locked dispatcher: is_nested -> bootstrap Clark-West on QLIKE for nested
pairs, Diebold-Mariano (HLN) on QLIKE for non-nested pairs. The test choice is never
hardcoded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.forecast_comparison import clark_west_bootstrap, diebold_mariano, is_nested
from src.metrics import qlike, qlike_per_obs
from src.results import score_walk_forward


def _score_all(wf_frames: dict, vix_monthly: pd.Series, spy_returns: pd.Series):
    """Score every model once through the shared scorer.

    Asserts all wf frames share one OOS index, because the per-regime subset masks
    are positionally aligned across models and that alignment is only valid when the
    indices are identical.

    Returns (scores, ref_index) where scores maps model name -> score_walk_forward
    dict and ref_index is the common OOS index.
    """
    items = list(wf_frames.items())
    if not items:
        raise ValueError("wf_frames is empty")

    ref_name, ref_frame = items[0]
    ref_index = ref_frame.index
    for name, wf in items[1:]:
        if not wf.index.equals(ref_index):
            raise ValueError(
                f"all wf frames must share one OOS index; {name!r} differs from {ref_name!r}"
            )

    scores = {}
    for name, wf in items:
        vix_next = vix_monthly.shift(-1).reindex(wf.index)
        scores[name] = score_walk_forward(wf, vix_next, spy_returns)
    return scores, ref_index


def _regime_labels(dataset: pd.DataFrame, index: pd.Index) -> np.ndarray:
    """Regime label at the decision date t for each OOS month, as a str array.

    Uses dataset.loc[index, "regime"], positionally aligned to index. This is the
    decision-date label (time t), never the realised regime at t+1.
    """
    return dataset.loc[index, "regime"].astype(str).to_numpy()


def per_regime_paired_qlike(
    wf_frames: dict,
    vix_monthly: pd.Series,
    spy_returns: pd.Series,
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Each model's QLIKE within each regime, on the subset of months valid for ALL
    models (the common-valid paired subset), classified at the decision date t.

    Holding the subset to the months every model can score keeps each model's
    per-regime QLIKE on identical dates, so the numbers are directly comparable
    within a regime.

    Returns a DataFrame: rows = config.REGIME_LABELS plus "all_regimes", columns =
    one per model in wf_frames (paired-subset QLIKE) followed by "n_paired".
    """
    scores, ref_index = _score_all(wf_frames, vix_monthly, spy_returns)
    regimes = _regime_labels(dataset, ref_index)

    names = list(wf_frames.keys())
    common_valid = scores[names[0]]["valid_mask"].copy()
    for name in names[1:]:
        common_valid = common_valid & scores[name]["valid_mask"]

    rows = {}
    for regime in config.REGIME_LABELS + ("all_regimes",):
        subset = common_valid if regime == "all_regimes" else (regimes == regime) & common_valid
        n_paired = int(subset.sum())
        realised = scores[names[0]]["realised_variance_next"][subset]
        row = {}
        for name in names:
            implied = scores[name]["implied_variance"][subset]
            row[name] = qlike(realised, implied) if n_paired > 0 else float("nan")
        row["n_paired"] = n_paired
        rows[regime] = row

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "regime"
    return table[names + ["n_paired"]]


def compare_models_by_regime(
    wf_frames: dict,
    vix_monthly: pd.Series,
    spy_returns: pd.Series,
    dataset: pd.DataFrame,
    pairs: list,
) -> pd.DataFrame:
    """Regime-conditional paired forecast comparison, one tidy row per regime x pair.

    For each regime in config.REGIME_LABELS and each (model_a, model_b) pair, the
    comparison runs on the subset where both models give a strictly positive implied
    variance (both valid_masks) AND the decision-date regime label matches. The pair
    routes through is_nested: nested -> clark_west_bootstrap on QLIKE (block length
    recomputed from the regime n), non-nested -> diebold_mariano (HLN) on QLIKE. The
    realised target is the single shared forward estimator; it is asserted identical
    across the two models on the subset.

    Pairs are supplied in (smaller, larger) nesting order for nested pairs. For the
    non-nested pair the two entries are simply model_a and model_b in the order
    given, mapped to the qlike_smaller / qlike_larger columns respectively, with the
    Diebold-Mariano differential d_t = loss_a - loss_b (a negative statistic favours
    model_a).

    The winner is decided strictly on raw paired QLIKE: the model with the lower of
    qlike_smaller / qlike_larger wins (lower QLIKE is better), regardless of the CW
    statistic sign or the DM favoured field; exact ties give "none". This is the
    QLIKE-primary adjudication rule. cw_qlike_direction_conflict flags rows where the
    test points to one model while raw QLIKE points to the other. For a nested pair
    the test direction is the larger model when larger_model_wins is True (and no
    direction otherwise); for a non-nested pair it is the DM favoured field. The
    conflict is the near-null Clark-West centering artifact (the adjustment term can
    push f_t positive even when the larger model's raw loss is higher) and is a
    reportable finding, not suppressed.

    Columns: regime, pair, test_type, n_paired, qlike_smaller, qlike_larger,
    statistic, p_value, block_length (NA for Diebold-Mariano), winner, low_power,
    cw_qlike_direction_conflict. low_power is True where n_paired <
    config.LOW_POWER_MIN_N; the p-value is kept but should not be treated as decisive
    there.
    """
    scores, ref_index = _score_all(wf_frames, vix_monthly, spy_returns)
    regimes = _regime_labels(dataset, ref_index)

    rows = []
    for m_a, m_b in pairs:
        score_a = scores[m_a]
        score_b = scores[m_b]
        both_valid = score_a["valid_mask"] & score_b["valid_mask"]
        nested = is_nested(m_a, m_b)
        pair_label = f"{m_a} vs {m_b}"

        for regime in config.REGIME_LABELS:
            subset = (regimes == regime) & both_valid
            n_paired = int(subset.sum())

            realised = score_a["realised_variance_next"][subset]
            # Realised target is the same forward estimator for both models.
            np.testing.assert_allclose(realised, score_b["realised_variance_next"][subset])
            implied_a = score_a["implied_variance"][subset]
            implied_b = score_b["implied_variance"][subset]

            qlike_a = qlike(realised, implied_a)
            qlike_b = qlike(realised, implied_b)

            loss_a = qlike_per_obs(realised, implied_a)
            loss_b = qlike_per_obs(realised, implied_b)

            if nested:
                # smaller = m_a, larger = m_b. The adjustment scores the larger
                # model's forecast against the smaller (nested, correct-under-H0)
                # forecast as proxy-truth. Block length is recomputed from this
                # regime's n inside clark_west_bootstrap (block_length=None).
                adjustment = qlike_per_obs(implied_a, implied_b)
                res = clark_west_bootstrap(loss_a, loss_b, adjustment, loss="qlike")
                test_type = res.test_type
                statistic = res.statistic
                p_value = res.p_value
                block_length = res.block_length
                # Test points at the larger model only when it significantly wins.
                test_direction = m_b if res.larger_model_wins else "none"
            else:
                res = diebold_mariano(loss_a, loss_b)
                test_type = res.test_type
                statistic = res.statistic
                p_value = res.p_value
                block_length = pd.NA
                test_direction = {"model_a": m_a, "model_b": m_b, "tie": "none"}[res.favoured]

            # Winner is the lower raw paired QLIKE (QLIKE-primary), never read off
            # the test direction. Exact ties give "none".
            if qlike_a < qlike_b:
                winner = m_a
            elif qlike_b < qlike_a:
                winner = m_b
            else:
                winner = "none"

            # Conflict: the test claims a direction that contradicts the raw-loss
            # winner. The reportable near-null Clark-West centering artifact.
            conflict = test_direction != "none" and test_direction != winner

            rows.append(
                {
                    "regime": regime,
                    "pair": pair_label,
                    "test_type": test_type,
                    "n_paired": n_paired,
                    "qlike_smaller": qlike_a,
                    "qlike_larger": qlike_b,
                    "statistic": statistic,
                    "p_value": p_value,
                    "block_length": block_length,
                    "winner": winner,
                    "low_power": n_paired < config.LOW_POWER_MIN_N,
                    "cw_qlike_direction_conflict": conflict,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "regime",
            "pair",
            "test_type",
            "n_paired",
            "qlike_smaller",
            "qlike_larger",
            "statistic",
            "p_value",
            "block_length",
            "winner",
            "low_power",
            "cw_qlike_direction_conflict",
        ],
    )


def regime_qlike_decomposition(
    wf: pd.DataFrame,
    vix_monthly: pd.Series,
    spy_returns: pd.Series,
    dataset: pd.DataFrame,
    regime_label: str,
    model_name: str,
) -> pd.DataFrame:
    """Per-observation QLIKE decomposition for one model within one regime.

    Shows how concentrated the regime's mean QLIKE is in a single month: the full
    (untrimmed) mean, the median per-observation loss, the share of summed QLIKE
    contributed by the single largest month, and that month's date and regime. The
    XGBoost QLIKE mean is dominated by one month, and within
    the stressed regime it dominates harder, so the decomposition is reported and the
    series is never trimmed.

    Read-only and descriptive: it does not adjudicate any model comparison.

    Returns a one-row DataFrame.
    """
    vix_next = vix_monthly.shift(-1).reindex(wf.index)
    score = score_walk_forward(wf, vix_next, spy_returns)
    regimes = _regime_labels(dataset, wf.index)

    subset = (regimes == regime_label) & score["valid_mask"]
    n = int(subset.sum())

    realised = score["realised_variance_next"][subset]
    implied = score["implied_variance"][subset]
    per_obs = qlike_per_obs(realised, implied)

    argmax = int(np.argmax(per_obs))
    largest_date = wf.index[subset][argmax]
    largest_regime = regimes[subset][argmax]
    total = float(per_obs.sum())

    return pd.DataFrame(
        [
            {
                "model": model_name,
                "regime": regime_label,
                "n": n,
                "full_mean_qlike": float(per_obs.mean()),
                "median_per_obs_qlike": float(np.median(per_obs)),
                "largest_month_qlike_share": float(per_obs[argmax] / total)
                if total > 0
                else float("nan"),
                "largest_month_date": largest_date,
                "largest_month_regime": largest_regime,
            }
        ]
    )
