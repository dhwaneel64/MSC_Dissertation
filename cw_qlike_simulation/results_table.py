"""Reporting layer for the CW/QLIKE bootstrap simulation study.

Assembles the three final tables and the diagnostic figure from output already
on disk. Nothing here simulates: every number is read from an npz written by a
completed run, or from the recorded values in config for the one construction
whose per-replication output no longer exists. If a cell is missing the row is
still built, with np.nan in place of the numbers, and the gap is printed.

pandas is used here and only here. The simulation modules stay on numpy, scipy
and statsmodels; this file is the reporting layer, where a DataFrame is the
natural carrier for a table that is written to CSV and printed.

Tables produced:
  1. Construction comparison, null world at CONSTRUCTION_T_OOS, one row per
     bootstrap construction tried.
  2. Size and power for the validated construction, null and both alternative
     strengths at both horizons.
  3. Sensitivity rows for the validated construction, with the difference
     standard error against the reference row.

Figure produced:
  QQ-plot of the null-world s_obs values at QQ_T_OOS against N(0,1).
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")            # file-only backend, nothing here opens a window
import matplotlib.pyplot as plt  # noqa: E402, import after the backend is fixed

import config
import montecarlo as mc


# ---------------------------------------------------------------------------
# Output file names, all under outputs/
# ---------------------------------------------------------------------------

CONSTRUCTION_CSV = "table1_constructions.csv"   # construction comparison table
SIZE_POWER_CSV = "table2_size_power.csv"        # size and power for the validated construction
SENSITIVITY_CSV = "table3_sensitivity.csv"      # sensitivity rows with difference standard errors
QQ_FIGURE_NAME = "fig_qq_null_s_obs_toos{t_oos}.png"   # QQ-plot file name, horizon in the name

# Input file the construction (b) row is read from. montecarlo writes this name
# as a literal in its size-run entry point rather than as a named constant, so
# the reporting layer repeats it here.
SIZE_RESULTS_NAME = "size_results.npz"   # size run for the loss-series stationary bootstrap

# ---------------------------------------------------------------------------
# Figure presentation constants, cosmetics only
# ---------------------------------------------------------------------------

FIG_SIZE = (6.5, 6.0)      # figure size in inches, square-ish because a QQ-plot is read against a 45-degree line
FIG_DPI = 200              # raster resolution, high enough for print in the methods chapter
POINT_COLOR = "#2a78d6"    # colour of the quantile points
CUTOFF_COLOR = "#eb6834"   # colour of the empirical percentile line, distinct from the points under colour-vision deficiency
REF_COLOR = "#52514e"      # colour of the 45-degree line and the normal-cutoff line
TEXT_COLOR = "#0b0b0b"     # primary text colour
SURFACE_COLOR = "#fcfcfb"  # figure and axes background
POINT_SIZE = 9             # marker area in points squared, small because 2000 points are plotted
POINT_ALPHA = 0.45         # marker transparency, so the dense centre of the plot stays readable
GRID_ALPHA = 0.25          # grid line transparency, recessive against the data


def _out_dir():
    """Absolute path of the outputs directory next to this file."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def _load(name):
    """Load an npz from outputs/ into a dict of arrays, or None if absent.

    Returns a plain dict so the caller does not hold the file handle open.
    """
    path = os.path.join(_out_dir(), name)
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _rej_and_se(p_arr, alphas):
    """Rejection rates and Monte Carlo standard errors from a p-value array.

    A test rejects when its p-value is at or below the nominal level, so the
    rate at each level is the fraction of replications satisfying that, and the
    standard error is the binomial sqrt(p (1 - p) / R).
    """
    r = p_arr.size
    rej = np.array([np.mean(p_arr <= a) for a in alphas])
    se = np.sqrt(rej * (1.0 - rej) / r)
    return rej, se


def _or_nan(value):
    """Return the value as a float, or np.nan when it was not recorded."""
    return np.nan if value is None else float(value)


def _fmt_count(value):
    """Format a replication or draw count, or say so when it was not recorded.

    Kept as text so a row with a missing count does not turn the whole column
    into floats and print 2000.000 next to it.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "not recorded"
    return str(int(value))


def _fmt_pair(values, digits=3):
    """Format a two-element sequence as "first/second", or "-" if both are nan.

    Used for the columns that carry one number per alpha level in a single cell.
    """
    if np.all(np.isnan(values)):
        return "-"
    return "/".join(f"{v:.{digits}f}" for v in values)


# ---------------------------------------------------------------------------
# Table 1: the four constructions
# ---------------------------------------------------------------------------

def _construction_rows():
    """One record per construction, with the source of each row's numbers.

    Constructions (b), (c) and (d) are read from their saved per-replication
    arrays. Construction (a) has no saved output, so it is filled from
    config.CONSTRUCTION_RECORDED, which currently records nothing. Missing files
    produce a row of np.nan and an entry in the returned gap list.
    """
    alphas = config.ALPHA_LEVELS
    t_oos = config.CONSTRUCTION_T_OOS
    rows = {}
    gaps = []

    # (b) Loss-series stationary bootstrap on the HAC-studentised statistic.
    # run_cell passes config.B_BOOT, so that is the draw count behind this file.
    size_results = _load(SIZE_RESULTS_NAME)
    if size_results is None:
        gaps.append(f"(b) outputs/{SIZE_RESULTS_NAME} not found, loss-series stationary bootstrap row is empty")
    else:
        tag = mc.cell_tag("null", 0.0, t_oos, config.BLOCK_BASELINE)
        s_obs = size_results[f"{tag}__s_obs"]
        rej, se = _rej_and_se(size_results[f"{tag}__p_boot"], alphas)
        rows["b"] = {
            "r_reps": int(size_results[f"{tag}__r_reps"]),
            "b_boot": config.B_BOOT,
            "rej": rej,
            "se": se,
            "draw_std": float(size_results[f"{tag}__boot_std"].mean()),
            "s_obs_std": float(s_obs.std()),
        }

    # (c) Calhoun data-level block bootstrap, from the reduced-scale validation.
    valid = _load(mc.VALID_PARTIAL_NAME)
    if valid is None or int(valid["n_done"]) < mc.VALID_R:
        gaps.append(f"(c) data-level bootstrap validation incomplete or missing, expected {mc.VALID_R} replications")
    else:
        rej, se = _rej_and_se(valid["p_data"], alphas)
        rows["c"] = {
            "r_reps": int(valid["n_done"]),
            "b_boot": mc.VALID_B,
            "rej": rej,
            "se": se,
            "draw_std": float(valid["draw_std"].mean()),
            "s_obs_std": float(valid["s_obs"].std()),
        }

    # (d) Model-based null-imposed ARMA bootstrap, from the full size run.
    size_full = _load(mc.SIZE_FULL_RESULTS_NAME)
    if size_full is None:
        gaps.append(f"(d) outputs/{mc.SIZE_FULL_RESULTS_NAME} not found, validated construction row is empty")
    else:
        s_obs = size_full[f"toos{t_oos}__s_obs"]
        rej, se = _rej_and_se(size_full[f"toos{t_oos}__p_model"], alphas)
        rows["d"] = {
            "r_reps": int(size_full["r_size"]),
            "b_boot": int(size_full["b_boot_size"]),
            "rej": rej,
            "se": se,
            "draw_std": float(size_full[f"toos{t_oos}__draw_std"].mean()),
            "s_obs_std": float(s_obs.std()),
        }

    # Constructions with no saved output fall back to the recorded values. A
    # recorded standard error is used as recorded; if none was kept, it is
    # rebuilt from the rate and R.
    for key, rec in config.CONSTRUCTION_RECORDED.items():
        if key in rows:
            continue
        rej = np.array(rec["rej"], dtype=float)
        r_reps = rec["r_reps"]
        if rec.get("se") is not None:
            se = np.array(rec["se"], dtype=float)
        elif r_reps is not None:
            se = np.sqrt(rej * (1.0 - rej) / r_reps)
        else:
            se = np.full(rej.shape, np.nan)
        rows[key] = {
            "r_reps": r_reps,
            "b_boot": rec["b_boot"],
            "rej": rej,
            "se": se,
            "draw_std": _or_nan(rec["draw_std"]),
            "s_obs_std": _or_nan(rec["s_obs_std"]),
        }
        if rec.get("t_oos") is not None and rec["t_oos"] != t_oos:
            gaps.append(f"({key}) recorded at t_oos={rec['t_oos']}, table horizon is {t_oos}, "
                        "the row is not comparable with the rest")
        missing = [name for name in ["draw_std", "s_obs_std"] if rec[name] is None]
        if missing:
            gaps.append(f"({key}) {' and '.join(missing)} not recorded by that run, cells carry nan")
        if np.all(np.isnan(rej)):
            gaps.append(f"({key}) no saved output and no recorded rejection rates, row carries nan")

    return rows, gaps


def build_construction_table():
    """Comparison table of the four bootstrap constructions, null world.

    One row per construction at config.CONSTRUCTION_T_OOS. R and B are carried
    in their own columns so the precision differences between rows are visible.
    Returns (DataFrame, list of gap messages).
    """
    alphas = config.ALPHA_LEVELS
    i5 = alphas.index(0.05)
    i10 = alphas.index(0.10)
    rows, gaps = _construction_rows()

    records = []
    for key in config.CONSTRUCTION_KEYS:
        if key not in rows:
            continue
        r = rows[key]
        records.append({
            "construction": f"({key}) {config.CONSTRUCTION_LABEL[key]}",
            "R": _fmt_count(r["r_reps"]),
            "B": _fmt_count(r["b_boot"]),
            "size@5%": r["rej"][i5],
            "size@10%": r["rej"][i10],
            "MC se 5/10%": _fmt_pair([r["se"][i5], r["se"][i10]]),
            "mean draw std": r["draw_std"],
            "target s_obs std": r["s_obs_std"],
            "defects present": config.CONSTRUCTION_DEFECTS[key],
            "mechanism": config.CONSTRUCTION_MECHANISM[key],
        })
    return pd.DataFrame.from_records(records), gaps


# ---------------------------------------------------------------------------
# Table 2: size and power for the validated construction
# ---------------------------------------------------------------------------

# Column order for the size/power table, raw tests first then the size-adjusted
# versions, each reported at both alpha levels.
SIZE_POWER_COLUMNS = ["CW normal raw", "CW normal size-adj",
                      "CW model-boot raw", "CW model-boot size-adj"]


def _cell_rates(arrays, cutoffs, alphas):
    """Rejection rates for one cell, one entry per column of the size/power table.

    Raw columns use the nominal levels. The size-adjusted columns use the
    empirical null cutoffs in cutoffs: the s_obs cutoff calibrates the statistic
    both tests are read from, and the p-value cutoff calibrates the bootstrap
    test on its own null distribution. This is the same definition the power run
    used, taken from montecarlo so the exhibit and the run logs agree.
    """
    out = {
        "CW normal raw": np.array([np.mean(arrays["p_norm"] <= a) for a in alphas]),
        "CW model-boot raw": np.array([np.mean(arrays["p_model"] <= a) for a in alphas]),
        "CW normal size-adj": np.array([np.mean(arrays["s_obs"] >= c) for c in cutoffs["s_obs_cut"]]),
        "CW model-boot size-adj": np.array([np.mean(arrays["p_model"] <= c) for c in cutoffs["p_model_cut"]]),
    }
    return out


def build_size_power_table():
    """Size and power table for the validated model-based construction.

    Rows are the null world at each horizon and each alternative strength at
    each horizon; columns are the four tests at each alpha level. Null-world
    size-adjusted entries sit at the nominal level by construction, because the
    cutoffs are quantiles of the same null sample.
    Returns (DataFrame, list of gap messages).
    """
    alphas = config.ALPHA_LEVELS
    gaps = []
    records = []

    size_full = _load(mc.SIZE_FULL_RESULTS_NAME)
    power_full = _load(mc.POWER_RESULTS_NAME)
    if size_full is None:
        gaps.append(f"outputs/{mc.SIZE_FULL_RESULTS_NAME} not found, no null rows and no size-adjusted cutoffs")
        return pd.DataFrame(), gaps

    for t_oos in config.T_OOS_GRID:
        cutoffs = mc._null_cutoffs(t_oos)
        arrays = {f: size_full[f"toos{t_oos}__{f}"] for f in ["s_obs", "p_norm", "p_model"]}
        rates = _cell_rates(arrays, cutoffs, alphas)
        rec = {"world": "null", "beta": 0.0, "t_oos": t_oos,
               "R": int(size_full["r_size"]), "B": int(size_full["b_boot_size"])}
        for col in SIZE_POWER_COLUMNS:
            for a, v in zip(alphas, rates[col]):
                rec[f"{col} @{a:.0%}"] = v
        records.append(rec)

    if power_full is None:
        gaps.append(f"outputs/{mc.POWER_RESULTS_NAME} not found, no alternative rows")
    else:
        for beta in config.BETA_GRID:
            for t_oos in config.T_OOS_GRID:
                tag = mc.power_cell_tag(beta, t_oos)
                if f"{tag}__s_obs" not in power_full:
                    gaps.append(f"power cell {tag} not present in {mc.POWER_RESULTS_NAME}")
                    continue
                cutoffs = mc._null_cutoffs(t_oos)
                arrays = {f: power_full[f"{tag}__{f}"] for f in ["s_obs", "p_norm", "p_model"]}
                rates = _cell_rates(arrays, cutoffs, alphas)
                rec = {"world": "alt", "beta": beta, "t_oos": t_oos,
                       "R": int(power_full["r_power"]), "B": int(power_full["b_boot_power"])}
                for col in SIZE_POWER_COLUMNS:
                    for a, v in zip(alphas, rates[col]):
                        rec[f"{col} @{a:.0%}"] = v
                records.append(rec)

    return pd.DataFrame.from_records(records), gaps


# ---------------------------------------------------------------------------
# Table 3: sensitivity
# ---------------------------------------------------------------------------

def build_sensitivity_table():
    """Sensitivity rows for the validated construction, with difference standard errors.

    Rows are as the sensitivity run reported them: three axes, each with the
    validated cell as its reference row. The cells were run on separate seed
    streams, so no two rows share histories and every comparison against the
    reference is unpaired. The difference standard error is therefore
    sqrt(se_row^2 + se_reference^2), reported on the non-reference rows only.
    Returns (DataFrame, list of gap messages).
    """
    alphas = config.ALPHA_LEVELS
    i5 = alphas.index(0.05)
    i10 = alphas.index(0.10)
    gaps = []

    sens = _load(mc.SENS_RESULTS_NAME)
    if sens is None:
        gaps.append(f"outputs/{mc.SENS_RESULTS_NAME} not found, sensitivity table is empty")
        return pd.DataFrame(), gaps

    # Per-cell rejection rates and standard errors, keyed by cell.
    cell = {}
    for key in sens["cell_keys"]:
        rej, se = _rej_and_se(sens[f"{key}__p_model"], alphas)
        cell[str(key)] = {
            "rej": rej,
            "se": se,
            "r_reps": int(sens[f"{key}__r_reps"]),
            "draw_std": float(sens[f"{key}__draw_std"].mean()),
        }

    # The validated cell is the reference row on every axis: its generator,
    # bootstrap-draw count and residual scheme are the settled settings.
    ref_key = mc._sens_cells_and_rows()[0][0]["key"]
    if ref_key not in cell:
        gaps.append(f"reference sensitivity cell {ref_key} missing, difference standard errors not computed")

    records = []
    for axis, setting, key in zip(sens["row_axis"], sens["row_setting"], sens["row_key"]):
        key = str(key)
        c = cell[key]
        if key == ref_key:
            diff_se = "reference"
        elif ref_key in cell:
            ref = cell[ref_key]
            d = np.sqrt(c["se"] ** 2 + ref["se"] ** 2)
            diff_se = _fmt_pair([d[i5], d[i10]])
        else:
            diff_se = "-"
        records.append({
            "axis": str(axis),
            "setting": str(setting),
            "R": c["r_reps"],
            "rej@5%": c["rej"][i5],
            "rej@10%": c["rej"][i10],
            "MC se 5/10%": _fmt_pair([c["se"][i5], c["se"][i10]]),
            "mean draw std": c["draw_std"],
            "difference se vs reference 5/10%": diff_se,
        })
    return pd.DataFrame.from_records(records), gaps


# ---------------------------------------------------------------------------
# Figure: QQ-plot of the null s_obs against N(0,1)
# ---------------------------------------------------------------------------

def make_qq_figure():
    """QQ-plot of the null-world s_obs values at config.QQ_T_OOS against N(0,1).

    Sample quantiles are the sorted s_obs; theoretical quantiles use the
    (i - 0.5)/n plotting positions. Two horizontal references are marked: the
    normal cutoff at the smaller alpha level, the value the standard test uses,
    and the empirical quantile of s_obs at the same level, the cutoff the null
    distribution actually implies. The vertical gap between them is the size
    distortion the study reports.

    Returns (path to the saved PNG, normal cutoff, empirical cutoff), or
    (None, nan, nan) if the size results are missing.
    """
    size_full = _load(mc.SIZE_FULL_RESULTS_NAME)
    if size_full is None:
        return None, np.nan, np.nan

    t_oos = config.QQ_T_OOS
    s_obs = np.sort(size_full[f"toos{t_oos}__s_obs"])
    n = s_obs.size
    alpha = min(config.ALPHA_LEVELS)
    theo = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    normal_cut = stats.norm.ppf(1.0 - alpha)
    emp_cut = np.percentile(s_obs, 100.0 * (1.0 - alpha))

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI)
    fig.patch.set_facecolor(SURFACE_COLOR)
    ax.set_facecolor(SURFACE_COLOR)

    lo = min(theo[0], s_obs[0])
    hi = max(theo[-1], s_obs[-1])
    ax.plot([lo, hi], [lo, hi], color=REF_COLOR, linewidth=1.2, linestyle="--", zorder=1)
    ax.scatter(theo, s_obs, s=POINT_SIZE, color=POINT_COLOR, alpha=POINT_ALPHA,
               linewidths=0, zorder=3)
    ax.axhline(normal_cut, color=REF_COLOR, linewidth=1.6, zorder=2)
    ax.axhline(emp_cut, color=CUTOFF_COLOR, linewidth=1.6, zorder=2)

    ax.annotate(f"N(0,1) cutoff at {alpha:.0%}: {normal_cut:.3f}",
                xy=(lo, normal_cut), xytext=(4, 4), textcoords="offset points",
                color=TEXT_COLOR, fontsize=9, ha="left", va="bottom")
    ax.annotate(f"empirical {1.0 - alpha:.0%} quantile of s_obs: {emp_cut:.3f}",
                xy=(lo, emp_cut), xytext=(4, 4), textcoords="offset points",
                color=CUTOFF_COLOR, fontsize=9, ha="left", va="bottom")
    # The 45-degree line is labelled in the empty lower-right corner rather than
    # on the line itself, where the points sit on top of it.
    ax.text(0.98, 0.03, "dashed: 45-degree line", transform=ax.transAxes,
            color=REF_COLOR, fontsize=9, ha="right", va="bottom")

    ax.set_xlabel("N(0,1) quantile", color=TEXT_COLOR)
    ax.set_ylabel("CW statistic s_obs, null world", color=TEXT_COLOR)
    ax.set_title(f"Null distribution of the CW statistic against N(0,1)\n"
                 f"t_oos={t_oos}, {n} replications", color=TEXT_COLOR, fontsize=11)
    ax.grid(True, alpha=GRID_ALPHA, linewidth=0.6)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(REF_COLOR)
        spine.set_linewidth(0.8)

    path = os.path.join(_out_dir(), QQ_FIGURE_NAME.format(t_oos=t_oos))
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, facecolor=SURFACE_COLOR)
    plt.close(fig)
    return path, normal_cut, emp_cut


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _save_csv(df, name):
    """Write a table to outputs/ as CSV and return the path."""
    path = os.path.join(_out_dir(), name)
    df.to_csv(path, index=False)
    return path


def _print_table(title, df, note=None):
    """Print a table with a title line, and an optional one-line note under it."""
    print()
    print(title)
    print("-" * len(title))
    if df.empty:
        print("  (no rows, see gaps below)")
    else:
        print(df.to_string(index=False))
    if note is not None:
        print(note)


def main():
    """Build the three tables and the figure, print them, and save to outputs/."""
    # Print every table in full: no column dropped, no cell truncated, no
    # wrapping onto a second block of columns. The mechanism strings make the
    # construction table wide, which is the intended form for the report.
    pd.set_option("display.width", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    gaps = []

    t1, g1 = build_construction_table()
    gaps += g1
    _print_table(
        f"Table 1. Bootstrap constructions compared, null world, t_oos={config.CONSTRUCTION_T_OOS}",
        t1,
        "Mean draw std is the average across replications of the spread of that replication's "
        "bootstrap statistics;\ntarget s_obs std is the spread of the statistic across "
        "replications, the value a correct draw distribution has to match.\n"
        f"Note: {config.CONSTRUCTION_TABLE_NOTE}")

    t2, g2 = build_size_power_table()
    gaps += g2
    max_se = {int(r): 0.5 / np.sqrt(int(r)) for r in t2["R"].unique()} if not t2.empty else {}
    se_note = ", ".join(f"R={r}: {v:.3f}" for r, v in sorted(max_se.items()))
    _print_table(
        "Table 2. Size and power, validated model-based ARMA bootstrap",
        t2,
        f"Largest possible Monte Carlo se per row, 0.5/sqrt(R): {se_note}.\n"
        "Null-world size-adjusted entries equal the nominal level by construction: "
        "the cutoffs are quantiles of that same null sample.")

    t3, g3 = build_sensitivity_table()
    gaps += g3
    _print_table(
        f"Table 3. Sensitivity, validated construction, null world, t_oos={config.SENS_T_OOS}",
        t3,
        "Cells were run on separate seed streams, so every comparison against the reference row "
        "is unpaired\nand the difference se is sqrt(se_row^2 + se_reference^2).")

    fig_path, normal_cut, emp_cut = make_qq_figure()

    print()
    print("Saved files")
    print("-" * 11)
    for df, name in [(t1, CONSTRUCTION_CSV), (t2, SIZE_POWER_CSV), (t3, SENSITIVITY_CSV)]:
        print(f"  {_save_csv(df, name)}")
    if fig_path is None:
        gaps.append(f"outputs/{mc.SIZE_FULL_RESULTS_NAME} not found, QQ-plot not produced")
    else:
        print(f"  figure: {fig_path}")
        print(f"  QQ-plot cutoffs: normal={normal_cut:.4f}, empirical={emp_cut:.4f}")

    print()
    print("Gaps")
    print("-" * 4)
    if not gaps:
        print("  none, every reported cell was read from a saved run")
    for g in gaps:
        print(f"  {g}")

    return t1, t2, t3


if __name__ == "__main__":
    main()
