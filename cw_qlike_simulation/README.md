# CW/QLIKE Bootstrap Simulation Study

A Monte Carlo study of Clark-West tests on QLIKE loss differentials, with
p-values taken from a bootstrap rather than the standard N(0,1) approximation.
This file covers installation and running only.

This directory is self-contained. It does not import from, read from, or write
to the pipeline at the repository root.

## Citation address

Cite this study at:

    https://github.com/dhwaneel64/MSC_Dissertation.git
    path: cw_qlike_simulation/

The result artifacts in `outputs/` are committed rather than gitignored.

## Runtime before you start

Every figure below is wall clock measured on the machine that produced the
outputs. The run logs those measurements were taken from are not included here.
All runs are single-threaded.

| What you want | Cost |
|---|---|
| Cheapest single run: the first power cell, beta 0.03, t_oos 220 | **1.35 h** |
| The size run plus the power run through its third cell | **9.34 h** |
| Full replication, every table and figure | **about 19 h** |

The 19 hour figure covers the sensitivity run and the two validation runs, which
produce Table 1 and Table 3. It is not the cost of the size and power runs,
which is 9.34 hours.

The outputs are committed for that reason. To check one cell, read the per-cell
cost table below and run only what that cell needs, not the whole study.

## Environment

    python -m pip install -r requirements.txt

Python 3.14.0, numpy 2.3.5, scipy 1.17.0, statsmodels 0.14.6, pandas 2.3.3,
matplotlib 3.10.7.

### The statsmodels version is a reproduction condition

Reproduction is exact under a fixed seed **and** statsmodels 0.14.6, not under a
fixed seed alone.

The model-based bootstrap fits an ARMA(1,1) to log realised variance on every
replication. Whether the AR(1) fallback branch fires is decided by the
optimiser, so a different statsmodels version can converge where 0.14.6 did not,
or fail where it succeeded, and the reported rates move. Pin the version before
treating a mismatch as a reproduction failure.

## Layout

    config.py          every parameter in the study, one comment per entry
    dgp.py             data generating process, null and alternative worlds
    forecast_eval.py   walk-forward evaluation, CW statistic, the p-value functions
    montecarlo.py      all run entry points and their checkpointing
    results_table.py   reporting layer: reads npz, writes the tables and the figure
    outputs/           committed results

`results_table.py` computes nothing. Every number it reports is read from an npz
written by a completed run, or from the recorded values in `config.py` for the
one construction whose per-replication output no longer exists.

### Which output file feeds which exhibit

| Exhibit | Built from |
|---|---|
| `table1_constructions.csv` | `size_results.npz`, `validation_partial.npz`, `size_full.npz`, plus `config.CONSTRUCTION_RECORDED` for row (a) |
| `table2_size_power.csv` | `size_full.npz`, `power_full.npz` |
| `table3_sensitivity.csv` | `sensitivity.npz` |
| `fig_qq_null_s_obs_toos220.png` | `size_full.npz` |

Note that `validation_partial.npz` is a required input despite its name. The
other files ending `_partial.npz` are resume checkpoints and are not read by the
reporting layer.

## Entry points

    python montecarlo.py size_full        null world, both horizons  -> size_full.npz
    python montecarlo.py power            alt world, both betas      -> power_full.npz
    python montecarlo.py sensitivity      three sensitivity axes     -> sensitivity.npz
    python montecarlo.py validation       data-level bootstrap       -> validation_partial.npz
    python montecarlo.py validation_model model bootstrap, A2 scheme -> validation_model2_partial.npz
    python montecarlo.py beta_calibration alternative-strength scan  (no committed artifact)
    python montecarlo.py                  loss-series bootstrap      -> size_results.npz
    python results_table.py               all three tables and the figure

## Reproducing the cells of Table 2

### Which command writes which cell

| Cell | Command | Output file | Lands in `table2_size_power.csv` |
|---|---|---|---|
| Size, standard test, t_oos 220 | `montecarlo.py size_full` | `size_full.npz` | col `CW normal raw @5%`, row `null,0.0,220` |
| Size, corrected test, t_oos 220 | `montecarlo.py size_full` | `size_full.npz` | col `CW model-boot raw @5%`, row `null,0.0,220` |
| Size, standard test, t_oos 250 | `montecarlo.py size_full` | `size_full.npz` | col `CW normal raw @5%`, row `null,0.0,250` |
| Size, corrected test, t_oos 250 | `montecarlo.py size_full` | `size_full.npz` | col `CW model-boot raw @5%`, row `null,0.0,250` |
| Power, model bootstrap, beta 0.03, t_oos 220 | `montecarlo.py power` | `power_full.npz` | col `CW model-boot raw @5%`, row `alt,0.03,220` |
| Power, model bootstrap, beta 0.05, t_oos 220 | `montecarlo.py power` | `power_full.npz` | col `CW model-boot raw @5%`, row `alt,0.05,220` |

### Cost, and what can be run on its own

| Cell | Command | Reachable on its own? | Cost |
|---|---|---|---|
| Both t_oos 220 size columns | `python montecarlo.py size_full` with `T_OOS_GRID = [220]` | Yes, one-line config edit | **2.29 h** |
| Both t_oos 220 size columns, no config edit | `python montecarlo.py size_full` | Printed in the cell 1 summary before cell 2 starts, but no `size_full.npz` until both cells finish | 2.29 h to the print, 5.19 h to the file |
| Both t_oos 250 size columns | `python montecarlo.py size_full` | No. Cell 2 cannot be reached without cell 1 | **5.19 h** |
| Power, beta 0.03, t_oos 220 | `python montecarlo.py power` | Yes. Cell 1 of 4, and `power_full.npz` is written when it finishes | **1.35 h** |
| Power, beta 0.05, t_oos 220 | `python montecarlo.py power` | Cell 3 of 4, so cells 1 and 2 must run first | **4.15 h** |

The two t_oos 220 size columns come out of the same cell, as do the two at
t_oos 250, so that cell costs 2.29 hours once, not twice.

Size and power together cost 9.34 hours: 5.19 for the size run plus 4.15 for the
power run through cell 3. The fourth power cell is not needed for any of the six
cells above.

### The power run does not need the size run

`python montecarlo.py power` runs standalone. With `size_full.npz` absent,
`_null_cutoffs` returns `None`, the run prints "size-adjusted columns skipped",
`_save_power_results` omits the two cutoff keys, and the run completes and
writes a valid `power_full.npz`. The size-adjusted columns are not produced.

The dependency exists in one place only: `results_table.py` needs
`size_full.npz` to emit the size-adjusted columns and the null rows of Table 2.

### The single-horizon size shortcut

Setting `T_OOS_GRID = [220]` in `config.py` runs the t_oos=220 size cell over
the same replications as the two-horizon run and writes a real `size_full.npz`
for that horizon, at 2.29 hours instead of 5.19.

Both runs spawn their per-replication seeds once and slice them positionally,
and `SeedSequence.spawn` is deterministic by position:
`SIZE_SEED_SEQUENCE.spawn(4000)[:2000]` has spawn keys identical to
`SIZE_SEED_SEQUENCE.spawn(2000)`, so cell 1 sees the same 2000 histories either
way.

Two consequences of that same positional slicing:

- **The size run's cell 2 cannot be reached without cell 1.** Cell 2 uses
  `children[2000:4000]`, a slice that only exists when `T_OOS_GRID` holds both
  entries. There is no way to run t_oos=250 alone.
- **Restore `T_OOS_GRID = [220, 250]` before running `power` or
  `results_table.py`.** The power run builds its cell list from `BETA_GRID`
  crossed with `T_OOS_GRID`, so a shortened grid changes every power cell's seed
  slice and the power cells will not reproduce. The shortcut is valid for the
  size run and nothing else.

### Constants that govern these runs

Common to both:

    MASTER_SEED = 20260707
    _ROOT_SEED_SEQUENCE.spawn(4) -> SIZE, POWER, CAL, SENS   (positional, order matters)
    T_TRAIN = 140
    T_OOS_GRID = [220, 250]
    ALPHA_LEVELS = [0.05, 0.10]
    CW_ADJ_SCALE = 0.5
    NW_LAG_COEF = 4, NW_LAG_BASE = 100, NW_LAG_EXP = 2/9      (lag 4 at n=220)

The spawn count of 4 and the positional child order are part of the seed
discipline. Changing either, including adding a fifth child, silently changes
every number in the study.

The size run, `python montecarlo.py size_full`:

    seed stream   SIZE_SEED_SEQUENCE, child 0 of the root spawn
    R_SIZE        2000
    B_BOOT_SIZE   299
    output        outputs/size_full.npz
    keys          toos220__p_norm, toos220__p_model,
                  toos250__p_norm, toos250__p_model
    metadata      r_size = 2000, b_boot_size = 299, t_oos_grid

The power run, `python montecarlo.py power`:

    seed stream    POWER_SEED_SEQUENCE, child 1 of the root spawn
    R_POWER        1000
    B_BOOT_POWER   299
    BETA_GRID      [0.03, 0.05]
    output         outputs/power_full.npz
    keys           beta0p03_toos220__p_model, beta0p05_toos220__p_model
    metadata       r_power = 1000, b_boot_power = 299

`B_BOOT = 999` is a different constant and governs neither run. It applies to
the loss-series bootstrap behind Table 1 row (b).

### The npz stores vectors, not rates

Each `__p_norm` and `__p_model` key holds one p-value per replication: 2000
floats for a size cell, 1000 for a power cell. The reported rate is not stored.
It is computed on read as the fraction at or below the level:

    rate = np.mean(p_arr <= 0.05)

in `results_table.py`. Opening `size_full.npz` returns those per-replication
vectors, not a scalar.

## Reproduction mechanics

**Working directory.** Every entry point resolves `outputs/` relative to its own
file, so commands run from anywhere, but always write into
`cw_qlike_simulation/outputs/`.

**Clear the checkpoints first.** Every run resumes from `*_partial.npz` if
present and skips completed replications. A from-scratch reproduction needs
`outputs/` to contain no partial files, or the run will finish instantly having
recomputed nothing. Keep `validation_partial.npz` if you want Table 1 row (c) to
survive, since that one is a reporting input rather than a checkpoint.

**Order.** `size_full` before `power` if you want the size-adjusted columns,
though not for the raw power columns. Both before `results_table.py`.

**Interruption.** The power run writes `power_full.npz` after each cell, so
stopping it part way leaves a usable file covering the finished cells. The size
run writes `size_full.npz` only after both cells finish, so stopping it after
cell 1 leaves the printed summary and a checkpoint but no results file.

## Reading the outputs

`results_table.py` prints a `Gaps` block after the tables. Partial reproduction
produces `np.nan` rows and an entry in that block **by design, not a crash**: a
missing input file is reported, the row is still built, and the run completes.

The pass condition for a complete reproduction is that the block reads:

    none, every reported cell was read from a saved run

Anything else lists exactly which cell could not be filled and why.
