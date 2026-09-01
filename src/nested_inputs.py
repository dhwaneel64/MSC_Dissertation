"""Persistence layer for the nested-comparison inputs.

The nested Clark-West comparisons (Objective 2 full-sample, Objective 3
regime-conditional) consume per-observation series that until now existed only
inside a live kernel: nothing was written to disk, so every downstream use meant
rerunning the walk-forward. This module rebuilds the four walk-forwards those
comparisons need (constant, HAR, extended OLS, regime-switching OLS; XGBoost is
not nested with anything and is not needed), writes their per-observation arrays
to outputs/, and records the pairing each nested comparison used so the pairing
is reproduced rather than recomputed.

Nothing about the methodology changes here. The dataset, the models, the
walk-forward engine, the scorer and the loss functions are the same objects the
notebook calls, invoked in the same order with the same arguments, on the same
locked snapshot. The module adds persistence and prints diagnostics; it does not
re-derive any quantity by a second path.

Before writing anything, the module recomputes the three Objective 2 Clark-West
statistics and paired QLIKE means and checks them against the values reported by
the executed notebook. A mismatch stops the run before any file is
written, because a saved artefact that disagrees with the reported results is
worse than no artefact.

Run as:  python -m src.nested_inputs
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data_loader import download_prices
from src.dataset import build_model_ready_dataset
from src.forecast_comparison import clark_west_from_losses
from src.metrics import qlike, qlike_per_obs
from src.models.baseline import ConstantMeanModel
from src.models.extended_ols import ExtendedOLSModel
from src.models.har_ols import HAROLSModel
from src.models.regime_switching_ols import NUMERIC_FEATURES, RegimeSwitchingOLSModel
from src.regimes import label_regimes
from src.results import score_walk_forward
from src.returns import compute_log_returns
from src.validation import VRP_HORIZON_COLS
from src.vrp import build_vrp_series, resample_to_month_start
from src.walk_forward import make_model_factory_from_class, walk_forward


OUTPUT_DIR = config.PROJECT_ROOT / "outputs"

# Model order is the nesting sequence: each model nests the one before it.
MODEL_ORDER = ("constant", "har", "extended_ols", "regime_switching")

# Nested pairs in (smaller, larger) order, the same three the notebook runs and
# the same three regime_comparison routes through Clark-West.
NESTED_PAIRS = (
    ("constant", "har"),
    ("har", "extended_ols"),
    ("extended_ols", "regime_switching"),
)

# Clark-West statistics used as the gate on this rerun, to 4 decimals.
#
# Re-baselined by the lag-to-target alignment correction. The pre-correction values
# (3.0334, 3.0536, 1.1027) were produced by a dataset whose nearest VRP predictor
# sat two steps from its target, so they are not comparable with these and are not
# a fallback. The dataset gained one row at the front (387 to 388, first row
# 1993-11-01 to 1993-10-01) because the maximum shift dropped from 6 to 5; the OOS
# index is unchanged at 253 months.
AUDIT_CW_STATISTIC = {
    ("constant", "har"): 3.465788,
    ("har", "extended_ols"): -0.982124,
    ("extended_ols", "regime_switching"): 1.287212,
}

# Paired-subset QLIKE means from the same corrected run, keyed by pair then model.
AUDIT_PAIRED_QLIKE = {
    ("constant", "har"): {"constant": 1.466938, "har": 0.443401},
    ("har", "extended_ols"): {"har": 0.419371, "extended_ols": 2.375025},
    ("extended_ols", "regime_switching"): {"extended_ols": 0.408569,
                                           "regime_switching": 0.817034},
}

# Paired n from the same corrected run.
AUDIT_N_PAIRED = {
    ("constant", "har"): 222,
    ("har", "extended_ols"): 249,
    ("extended_ols", "regime_switching"): 246,
}

# Match tolerances. The reported values are printed to 4 decimals (statistic) and 6
# decimals (QLIKE), so half a unit in the last printed place is the tightest a
# comparison against them can be.
CW_STAT_TOL = 5e-5
QLIKE_TOL = 5e-7


# ---------------------------------------------------------------------------
# Rebuild the notebook's walk-forwards
# ---------------------------------------------------------------------------

def build_inputs() -> dict:
    """Rebuild dataset, monthly series and the four walk-forward frames.

    Mirrors notebook cells 2 to 58 for the models the nested comparisons use,
    in the same order with the same arguments. Returns a dict holding the
    dataset, the monthly VIX series, the daily SPY log returns, the derived
    initial_train_end, and the walk-forward frame per model.
    """
    spy = download_prices(config.TICKER_SPY)
    vix = download_prices(config.TICKER_VIX)
    skew = download_prices(config.TICKER_SKEW)

    spy_returns = compute_log_returns(spy)
    vix_monthly = resample_to_month_start(vix["close"])
    skew_monthly = resample_to_month_start(skew["close"])

    vrp = build_vrp_series(vix_monthly, spy_returns, vix_monthly.index)
    regime_labels = label_regimes(vix_monthly)
    dataset = build_model_ready_dataset(vrp, vix_monthly, skew_monthly,
                                        spy_returns, regime_labels)

    # Same derivation as the notebook: the last dataset row inside the first
    # INITIAL_TRAINING_YEARS_FULL calendar years of the sample.
    last_train_year = dataset.index[0].year + config.INITIAL_TRAINING_YEARS_FULL - 1
    initial_train_end = dataset.loc[dataset.index.year <= last_train_year].index[-1]

    har_feature_cols = list(VRP_HORIZON_COLS)
    extended_feature_cols = ["vix_level", "cboe_skew", *VRP_HORIZON_COLS,
                             "realised_skew_21d"]
    regime_feature_cols = list(NUMERIC_FEATURES) + ["regime"]

    specs = {
        "constant": ([], make_model_factory_from_class(ConstantMeanModel)),
        "har": (har_feature_cols,
                make_model_factory_from_class(HAROLSModel, monthly_vrp=dataset)),
        "extended_ols": (extended_feature_cols,
                         make_model_factory_from_class(ExtendedOLSModel, monthly_vrp=dataset)),
        "regime_switching": (regime_feature_cols,
                             make_model_factory_from_class(RegimeSwitchingOLSModel,
                                                           monthly_vrp=dataset)),
    }

    wf_frames = {}
    for name in MODEL_ORDER:
        feature_cols, factory = specs[name]
        wf_frames[name] = walk_forward(
            dataset,
            feature_cols=feature_cols,
            model_factory=factory,
            initial_train_end=initial_train_end,
            target_col="y",
        )

    ref_index = wf_frames[MODEL_ORDER[0]].index
    for name in MODEL_ORDER[1:]:
        if not wf_frames[name].index.equals(ref_index):
            raise ValueError(f"{name} OOS index differs from {MODEL_ORDER[0]}")

    return {
        "dataset": dataset,
        "vix_monthly": vix_monthly,
        "skew_monthly": skew_monthly,
        "spy_returns": spy_returns,
        "initial_train_end": initial_train_end,
        "wf_frames": wf_frames,
        "feature_cols": {name: specs[name][0] for name in MODEL_ORDER},
        "oos_index": ref_index,
    }


def build_model_arrays(inputs: dict) -> dict:
    """Per-observation arrays per model, from the shared scorer.

    score_walk_forward owns the only variance-space conversion and the only
    forward realised target, so every series here comes from it. QLIKE per
    observation is defined only where the model is valid (positive implied
    variance and a non-missing realised target); invalid months carry NaN so the
    array stays aligned with the full OOS index.
    """
    vix_monthly = inputs["vix_monthly"]
    spy_returns = inputs["spy_returns"]
    dataset = inputs["dataset"]
    index = inputs["oos_index"]

    # Fixed-width unicode, not object dtype, so the npz loads without allow_pickle.
    regime_at_t = np.asarray(dataset.loc[index, "regime"].astype(str).to_list())
    dates = np.array([d.strftime("%Y-%m-%d") for d in index])

    arrays = {}
    for name, wf in inputs["wf_frames"].items():
        vix_next = vix_monthly.shift(-1).reindex(wf.index)
        score = score_walk_forward(wf, vix_next, spy_returns)

        valid = score["valid_mask"]
        implied = score["implied_variance"]
        realised = score["realised_variance_next"]

        qlike_obs = np.full(len(index), np.nan)
        qlike_obs[valid] = qlike_per_obs(realised[valid], implied[valid])

        arrays[name] = {
            "date": dates,
            "vrp_forecast": wf["y_pred"].to_numpy(dtype=float),
            "vrp_realised": wf["y_true"].to_numpy(dtype=float),
            "implied_variance": implied,
            "realised_variance_next": realised,
            "qlike_per_obs": qlike_obs,
            "regime_at_t": regime_at_t,
            "valid_mask": valid,
            "score": score,
        }
    return arrays


def build_pair_alignment(arrays: dict) -> dict:
    """Paired mask and paired dates per nested pair.

    The pairing rule is the one the comparison cells use: the months valid for
    both models. Saving the mask means a later correction pairs the series by
    reading it, not by recomputing the rule.
    """
    pairs = {}
    for smaller, larger in NESTED_PAIRS:
        mask = arrays[smaller]["valid_mask"] & arrays[larger]["valid_mask"]
        pairs[(smaller, larger)] = {
            "paired_mask": mask,
            "paired_dates": arrays[smaller]["date"][mask],
            "n_paired": int(mask.sum()),
        }
    return pairs


# ---------------------------------------------------------------------------
# Verification against the reported values
# ---------------------------------------------------------------------------

def recompute_pair_scalars(arrays: dict, pairs: dict) -> dict:
    """Paired QLIKE means and the Clark-West statistic per nested pair.

    The statistic is produced by clark_west_from_losses on exactly the inputs
    the notebook passes it: per-observation QLIKE for each model and the
    adjustment qlike_per_obs(var_smaller, var_larger). No change of method, this
    is the existing statistic recomputed so it can be compared with the recorded
    value.
    """
    out = {}
    for smaller, larger in NESTED_PAIRS:
        mask = pairs[(smaller, larger)]["paired_mask"]
        realised = arrays[smaller]["realised_variance_next"][mask]
        implied_s = arrays[smaller]["implied_variance"][mask]
        implied_l = arrays[larger]["implied_variance"][mask]

        loss_s = qlike_per_obs(realised, implied_s)
        loss_l = qlike_per_obs(realised, implied_l)
        adjustment = qlike_per_obs(implied_s, implied_l)
        cw = clark_west_from_losses(loss_s, loss_l, adjustment, loss="qlike")

        out[(smaller, larger)] = {
            "n_paired": int(mask.sum()),
            "qlike": {smaller: qlike(realised, implied_s),
                      larger: qlike(realised, implied_l)},
            "statistic": cw.statistic,
            "p_value_normal": cw.p_value,
        }
    return out


def check_against_audit(scalars: dict) -> list:
    """Compare recomputed scalars with the recorded notebook values.

    Returns a list of mismatch descriptions, empty when everything agrees within
    the printed precision of the recorded values.
    """
    mismatches = []
    for pair in NESTED_PAIRS:
        got = scalars[pair]
        label = f"{pair[0]} vs {pair[1]}"

        if got["n_paired"] != AUDIT_N_PAIRED[pair]:
            mismatches.append(
                f"{label}: n_paired recomputed {got['n_paired']}, "
                f"recorded {AUDIT_N_PAIRED[pair]}"
            )

        diff = abs(got["statistic"] - AUDIT_CW_STATISTIC[pair])
        if diff > CW_STAT_TOL:
            mismatches.append(
                f"{label}: CW statistic recomputed {got['statistic']:.6f}, "
                f"recorded {AUDIT_CW_STATISTIC[pair]:.4f}, difference {diff:.2e} "
                f"(tolerance {CW_STAT_TOL:.0e})"
            )

        for model, recorded in AUDIT_PAIRED_QLIKE[pair].items():
            diff = abs(got["qlike"][model] - recorded)
            if diff > QLIKE_TOL:
                mismatches.append(
                    f"{label}: QLIKE {model} recomputed {got['qlike'][model]:.8f}, "
                    f"recorded {recorded:.6f}, difference {diff:.2e} "
                    f"(tolerance {QLIKE_TOL:.0e})"
                )
    return mismatches


# ---------------------------------------------------------------------------
# Regime-conditional QLIKE decomposition (read-only diagnostic)
# ---------------------------------------------------------------------------

def regime_qlike_by_model(arrays: dict, pair: tuple) -> pd.DataFrame:
    """Raw QLIKE mean per model per regime on one pair's paired subset.

    No test, no adjustment, no bootstrap: the paired subset is split by the
    decision-date regime label and each model's mean QLIKE is reported with the
    n in that regime. The all_regimes row is the whole paired subset.
    """
    smaller, larger = pair
    mask = arrays[smaller]["valid_mask"] & arrays[larger]["valid_mask"]
    regimes = arrays[smaller]["regime_at_t"]
    realised = arrays[smaller]["realised_variance_next"]
    implied_s = arrays[smaller]["implied_variance"]
    implied_l = arrays[larger]["implied_variance"]

    rows = {}
    for regime in list(config.REGIME_LABELS) + ["all_regimes"]:
        subset = mask if regime == "all_regimes" else mask & (regimes == regime)
        n = int(subset.sum())
        rows[regime] = {
            smaller: qlike(realised[subset], implied_s[subset]) if n else float("nan"),
            larger: qlike(realised[subset], implied_l[subset]) if n else float("nan"),
            "n": n,
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "regime"
    return table[[smaller, larger, "n"]]


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

# Per-observation fields written to each model's npz, in this order.
MODEL_FIELDS = ("date", "vrp_forecast", "vrp_realised", "implied_variance",
                "realised_variance_next", "qlike_per_obs", "regime_at_t", "valid_mask")


def _pair_key(pair: tuple) -> str:
    """Filename-safe key for a nested pair."""
    return f"{pair[0]}__{pair[1]}"


def _sha256(path: Path) -> str:
    """SHA-256 of a file, used as the content hash in the metadata."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_safe(value):
    """Convert a config value to something json can write, or None if it cannot."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return None


def config_snapshot() -> dict:
    """Every public config constant that json can represent, as it is now.

    Taken by enumeration rather than by naming a subset, so a later config change
    cannot silently fall outside the record.
    """
    out = {}
    for name in dir(config):
        if not name.isupper():
            continue
        safe = _json_safe(getattr(config, name))
        if safe is not None:
            out[name] = safe
    return out


def save_arrays(arrays: dict, pairs: dict, inputs: dict) -> dict:
    """Write one npz per model, one combined npz, and return the paths written."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = {}

    for name in MODEL_ORDER:
        payload = {f: arrays[name][f] for f in MODEL_FIELDS}
        path = OUTPUT_DIR / f"nested_inputs_{name}.npz"
        np.savez(path, **payload)
        written[name] = path

    combined = {"models": np.array(MODEL_ORDER),
                "date": arrays[MODEL_ORDER[0]]["date"],
                "regime_at_t": arrays[MODEL_ORDER[0]]["regime_at_t"],
                "pair_keys": np.array([_pair_key(p) for p in NESTED_PAIRS])}
    for name in MODEL_ORDER:
        for field in MODEL_FIELDS:
            if field in ("date", "regime_at_t"):
                continue          # shared across models, stored once above
            combined[f"{name}__{field}"] = arrays[name][field]
    for pair in NESTED_PAIRS:
        key = _pair_key(pair)
        combined[f"{key}__paired_mask"] = pairs[pair]["paired_mask"]
        combined[f"{key}__paired_dates"] = pairs[pair]["paired_dates"]
        combined[f"{key}__n_paired"] = pairs[pair]["n_paired"]
        combined[f"{key}__smaller"] = pair[0]
        combined[f"{key}__larger"] = pair[1]

    combined_path = OUTPUT_DIR / "nested_inputs_combined.npz"
    np.savez(combined_path, **combined)
    written["combined"] = combined_path
    return written


def save_metadata(written: dict, arrays: dict, pairs: dict, scalars: dict,
                  inputs: dict) -> Path:
    """Write the metadata json describing what was saved and under what settings."""
    snapshot_path = config.DATA_DIR / f"{config.SNAPSHOT_PREFIX}{config.LOCKED_SNAPSHOT_DATE}.parquet"
    index = inputs["oos_index"]

    meta = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot": {
            "locked_snapshot_date": config.LOCKED_SNAPSHOT_DATE,
            "file": snapshot_path.name,
            "sha256": _sha256(snapshot_path),
        },
        "oos": {
            "n_obs": len(index),
            "first_date": index.min().strftime("%Y-%m-%d"),
            "last_date": index.max().strftime("%Y-%m-%d"),
            "initial_train_end": inputs["initial_train_end"].strftime("%Y-%m-%d"),
        },
        "models": {
            name: {
                "feature_cols": list(inputs["feature_cols"][name]),
                "n_obs": int(arrays[name]["score"]["n_obs"]),
                "n_valid": int(arrays[name]["score"]["qlike_n"]),
                "n_guard_excluded": int(arrays[name]["score"]["n_guard_excluded"]),
                "n_nan_tail_excluded": int(arrays[name]["score"]["n_nan_tail_excluded"]),
                "qlike_full_valid": float(arrays[name]["score"]["qlike"]),
                "file": written[name].name,
                "sha256": _sha256(written[name]),
            }
            for name in MODEL_ORDER
        },
        "pairs": {
            _pair_key(pair): {
                "smaller": pair[0],
                "larger": pair[1],
                "n_paired": pairs[pair]["n_paired"],
                "qlike_paired": {m: float(v) for m, v in scalars[pair]["qlike"].items()},
                "cw_statistic": float(scalars[pair]["statistic"]),
                "cw_p_value_normal_reference": float(scalars[pair]["p_value_normal"]),
                "audit_cw_statistic": AUDIT_CW_STATISTIC[pair],
            }
            for pair in NESTED_PAIRS
        },
        "combined_file": {
            "name": written["combined"].name,
            "sha256": _sha256(written["combined"]),
        },
        "config": config_snapshot(),
    }

    path = OUTPUT_DIR / "nested_inputs_metadata.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return path


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_verification(arrays: dict, pairs: dict, scalars: dict, mismatches: list) -> None:
    """Print n per model, n paired per pair, and the recomputed-versus-recorded check."""
    print("Verification: n per model (full OOS window)")
    print("-" * 70)
    print(f"{'model':<20}{'n_obs':>8}{'n_valid':>9}{'guard':>8}{'nan_tail':>10}{'QLIKE':>12}")
    for name in MODEL_ORDER:
        s = arrays[name]["score"]
        print(f"{name:<20}{s['n_obs']:>8}{s['qlike_n']:>9}{s['n_guard_excluded']:>8}"
              f"{s['n_nan_tail_excluded']:>10}{s['qlike']:>12.6f}")

    print()
    print("Verification: n paired per nested pair")
    print("-" * 70)
    print(f"{'pair':<40}{'n_paired':>10}{'recorded':>10}")
    for pair in NESTED_PAIRS:
        label = f"{pair[0]} vs {pair[1]}"
        print(f"{label:<40}{pairs[pair]['n_paired']:>10}{AUDIT_N_PAIRED[pair]:>10}")

    print()
    print("Verification: recomputed against recorded (notebook cells 48, 56, 64)")
    print("-" * 70)
    print(f"{'pair':<34}{'quantity':<20}{'recomputed':>14}{'recorded':>12}")
    for pair in NESTED_PAIRS:
        label = f"{pair[0]} vs {pair[1]}"
        got = scalars[pair]
        print(f"{label:<34}{'CW statistic':<20}{got['statistic']:>14.6f}"
              f"{AUDIT_CW_STATISTIC[pair]:>12.4f}")
        for model, recorded in AUDIT_PAIRED_QLIKE[pair].items():
            print(f"{'':<34}{'QLIKE ' + model:<20}{got['qlike'][model]:>14.6f}{recorded:>12.6f}")

    print()
    if mismatches:
        print(f"MISMATCH: {len(mismatches)} recomputed value(s) disagree with the record.")
        for m in mismatches:
            print(f"  {m}")
    else:
        print("All recomputed statistics agree with the recorded values within tolerance "
              f"(CW {CW_STAT_TOL:.0e}, QLIKE {QLIKE_TOL:.0e}).")


def print_regime_decomposition(arrays: dict) -> None:
    """Print the raw QLIKE decomposition for the extended OLS / regime-switching pair."""
    pair = ("extended_ols", "regime_switching")
    table = regime_qlike_by_model(arrays, pair)
    print()
    print(f"Raw QLIKE mean per model per regime, {pair[0]} vs {pair[1]} paired subset")
    print("-" * 70)
    print(table.to_string(float_format=lambda v: f"{v:.6f}"))
    print("Regime is the decision-date label at t. No test, no adjustment, no bootstrap.")


def main() -> int:
    """Rebuild, verify, save, and print. Returns 0 on success, 1 on a mismatch."""
    inputs = build_inputs()
    arrays = build_model_arrays(inputs)
    pairs = build_pair_alignment(arrays)
    scalars = recompute_pair_scalars(arrays, pairs)
    mismatches = check_against_audit(scalars)

    print_verification(arrays, pairs, scalars, mismatches)

    if mismatches:
        print()
        print("Nothing was written. Resolve the mismatch before saving.")
        return 1

    written = save_arrays(arrays, pairs, inputs)
    meta_path = save_metadata(written, arrays, pairs, scalars, inputs)

    print()
    print("Saved files")
    print("-" * 70)
    for name in MODEL_ORDER:
        print(f"  {written[name]}")
    print(f"  {written['combined']}")
    print(f"  {meta_path}")

    print_regime_decomposition(arrays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
