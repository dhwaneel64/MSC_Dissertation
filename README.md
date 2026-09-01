# Volatility risk premium pipeline and simulation study

Code, data and result artifacts for an MSc dissertation. This file covers
installation and running only.

    https://github.com/dhwaneel64/MSC_Dissertation.git

## Contents

    README.md                     this file
    requirements.txt              pipeline dependencies, minimum versions
    pytest.ini                    pytest configuration, registers the network marker
    .gitignore

    data/
      raw_snapshot_2026-06-17.parquet   daily closes for SPY, VIX and CBOE SKEW
                                        on a union date index

    src/                          pipeline modules, imported by the notebook
      config.py                   all parameters, paths, thresholds and seeds
      data_loader.py              snapshot read
      returns.py, realised_vol.py, har_rv.py, vrp.py, regimes.py, features.py,
      dataset.py                  series construction and the monthly frame
      walk_forward.py             expanding-window fit and predict loop
      models/                     the five forecasting models
      metrics.py, evaluation.py, results.py
                                  loss computation and walk-forward scoring
      forecast_comparison.py, corrected_cw.py, nested_inputs.py,
      mincer_zarnowitz.py, regime_comparison.py, diagnostics.py, validation.py
                                  statistical tests, persistence and checks
      circle3a/                   trading rule: position series, profit and loss,
                                  tail and episode summaries

    tests/                        393 tests

    notebooks/
      dissertation.ipynb          95 cells, 48 code, 46 carrying stored outputs

    outputs/                      per-observation arrays per model, the bootstrap
                                  result from corrected_cw.py, paired losses for
                                  all ten model pairs, and two figures
      figures/                    two PNGs

    scripts/
      make_figures.py             renders the two figures from the arrays in
                                  outputs/. Computes no statistic

    cw_qlike_simulation/          separate Monte Carlo study with its own README,
                                  its own pinned requirements and its own outputs

The simulation does not import from the pipeline and the pipeline does not
import from the simulation.

## Environment

Both parts run on Python 3.14.0. Earlier 3.x versions were not tested.

They need two virtual environments, because the simulation pins exact versions
that conflict with what the pipeline's minimums resolve to.

Pipeline, from the repository root:

    python -m venv .venv
    .venv\Scripts\activate            # Windows; source .venv/bin/activate on Unix
    python -m pip install -r requirements.txt

`requirements.txt` includes pyarrow. It is required, not optional: the data file
is parquet and pyarrow is the engine pandas uses to read it. Without pyarrow the
loader raises on the first data call.

Simulation, in its own environment:

    python -m venv .venv-sim
    .venv-sim\Scripts\activate
    python -m pip install -r cw_qlike_simulation/requirements.txt

That file pins numpy 2.3.5, scipy 1.17.0, statsmodels 0.14.6, pandas 2.3.3 and
matplotlib 3.10.7 exactly.

## Version sensitivity

The committed notebook outputs were produced under xgboost 3.2.0 and the
committed figures under matplotlib 3.10.7. `requirements.txt` sets minimums with
no upper bound, so a fresh install resolves to later versions.

Under a later xgboost, cells 68, 70, 72, 74, 76 and the XGBoost rows of cell 80
produce different numbers from the committed ones. The remaining cells do not.
Cell 66 prints a wall-clock timing line that differs on any machine.

Under a later matplotlib, `scripts/make_figures.py` draws the same data on the
same axes but the rasterisation differs, so the PNG bytes do not match.

To match the committed outputs, install those two versions after the
requirements install:

    python -m pip install -r requirements.txt
    python -m pip install xgboost==3.2.0 matplotlib==3.10.7

The simulation's own condition is the statsmodels pin above. Its README explains
what depends on it.

## Data

    data/raw_snapshot_2026-06-17.parquet
    sha256  7323f72f8dc3cb0e3d912c3e6bc111e81904499dc9d60cb627e0bfbae0c215d6
    size    190,301 bytes

No network access is required to run anything in this repository.
`src/data_loader.download_prices` reads only the snapshot named by
`config.LOCKED_SNAPSHOT_DATE` and raises if it is absent. There is no fallback
to a live fetch and no fallback to another snapshot. Passing `refresh=True`
fetches new data and writes a new dated file, which becomes the default only if
`config.LOCKED_SNAPSHOT_DATE` is edited by hand. No code path edits it.

The simulation reads no market data. It generates its own series.

## Running the pipeline

    python -m pytest
    jupyter lab notebooks/dissertation.ipynb      # then run all cells
    python scripts/make_figures.py

Measured on the reference machine, single run, fresh environment:

| Step | Wall clock |
|---|---|
| Full test suite, 393 tests | 48 s |
| The single test below, alone | 5 s |
| Notebook, all 48 code cells, top to bottom | 68 s |
| `python scripts/make_figures.py` | under 10 s |

The notebook does not repeat the two expensive computations. The bootstrap in
`src/corrected_cw.py` (9999 draws, tens of minutes) was run once and its result
read from `outputs/corrected_cw_results.npz`. The arrays in
`outputs/nested_inputs_*.npz` are read, not recomputed.

## The single test to run

    python -m pytest tests/test_results.py::test_score_walk_forward_reproduces_constant_baseline -v

It rebuilds the pipeline from `data/raw_snapshot_2026-06-17.parquet` end to end,
through returns, variance forecasts, the premium series, the regime labels, the
monthly dataset, the walk-forward loop and the scorer, and asserts that the
result matches the values recorded in the test. Those values live in the test
file. It passes in this tree.

## Running the simulation

Re-running is expensive: `cw_qlike_simulation/README.md` measures 1.35 hours for
the cheapest single run, 9.34 hours for the size run plus the power run through
its third cell, and about 19 hours for every table and figure. The outputs are
committed for that reason. Read the per-cell cost table there before starting
any of the commands below.

    cd cw_qlike_simulation
    python montecarlo.py size_full     # -> outputs/size_full.npz
    python montecarlo.py power         # -> outputs/power_full.npz
    python results_table.py            # writes the three tables and the figure

`cw_qlike_simulation/README.md` is the reference for this part.

Runs resume from `*_partial.npz` checkpoints and skip completed replications, so
a from-scratch re-run needs those cleared first. Keep `validation_partial.npz`,
which is a reporting input rather than a checkpoint.

The runtimes quoted there were measured on the machine that produced the
outputs. They are reported rather than reproducible in this tree.

## Determinism

Every stochastic component runs from a fixed seed, set in `src/config.py` and
`cw_qlike_simulation/config.py`. All runs are single-threaded. Nothing reads the
network. Repeating a run on the same data produces identical output, subject to
the two version conditions above: xgboost 3.2.0 for the XGBoost cells, and
matplotlib 3.10.7 for the figure bytes.
