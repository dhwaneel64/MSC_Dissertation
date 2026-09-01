"""Renders persisted artifacts only.

Every series plotted here is read directly from an artifact under outputs/.
No statistic, aggregation, filter or threshold is computed in this script. The exclusion
marking uses the persisted valid_mask boolean, which already encodes both
components recorded at write time: the positivity-guard exclusions, where the
strike VIX squared over VIX_VARIANCE_SCALE minus the model's VRP forecast is at
or below zero, and the unscorable trailing month, whose forward
realised-variance target falls outside the sample. Three of the five
commissioned figures are dropped for want of persisted inputs and are not
reconstructed here.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures"
ART = ROOT / "outputs"

MODELS = ["constant", "har", "extended_ols", "regime_switching", "xgboost"]
REGIME_COLOURS = {"calm": "#d9ead3", "normal": "#e8e8e8", "stressed": "#f4cccc"}


def load(model):
    return np.load(ART / f"nested_inputs_{model}.npz", allow_pickle=True)


def figure_qlike_per_month():
    fig, ax = plt.subplots(figsize=(13, 6))
    marked = None
    for model in MODELS:
        d = load(model)
        dates = pd.to_datetime(d["date"])
        ax.plot(dates, d["qlike_per_obs"], linewidth=1.0, label=model)
        invalid = ~d["valid_mask"]
        marked = invalid if marked is None else (marked | invalid)

    dates = pd.to_datetime(load("har")["date"])
    for x in dates[marked]:
        ax.axvline(x, color="#b00020", linewidth=0.7, alpha=0.45, zorder=0)

    ax.set_yscale("log")
    ax.set_xlabel("Month (walk-forward out-of-sample, first trading day)")
    ax.set_ylabel("QLIKE per observation (dimensionless, log scale)")
    ax.set_title(
        "Per-month QLIKE by model, with months excluded from scoring for any "
        "model marked"
    )
    handles, labels = ax.get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="#b00020", linewidth=0.7, alpha=0.45))
    labels.append("excluded (implied variance minus forecast at or below zero, or forward realised target absent)")
    ax.legend(handles, labels, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = OUT / "fig_qlike_per_month_by_model.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_vrp_with_regimes():
    d = load("har")
    dates = pd.to_datetime(d["date"])
    vrp = d["vrp_realised"]
    regimes = d["regime_at_t"]

    fig, ax = plt.subplots(figsize=(13, 6))
    edges = list(dates) + [dates[-1] + (dates[-1] - dates[-2])]
    for i, label in enumerate(regimes):
        ax.axvspan(edges[i], edges[i + 1], color=REGIME_COLOURS[label], linewidth=0, zorder=0)
    ax.plot(dates, vrp, color="#1a1a1a", linewidth=1.1, zorder=2)
    ax.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--", zorder=1)

    ax.set_xlabel("Month (walk-forward out-of-sample, first trading day)")
    ax.set_ylabel("Constructed VRP (decimal annualised variance)")
    ax.legend(
        handles=[mpatches.Patch(color=c, label=k) for k, c in REGIME_COLOURS.items()],
        fontsize=9,
        title="Regime at t",
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = OUT / "fig_vrp_series_regime_shaded.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for p in (figure_qlike_per_month(), figure_vrp_with_regimes()):
        print(p.relative_to(ROOT))
