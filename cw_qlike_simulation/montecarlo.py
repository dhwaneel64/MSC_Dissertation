"""Monte Carlo replication loop for the CW/QLIKE bootstrap simulation study.

One cell is a fixed (world, beta, T_OOS, block length). run_cell repeats the
full pipeline r_reps times with independent per-replication randomness and
returns rejection rates, Monte Carlo standard errors, and diagnostic arrays.
main drives the size runs (null world) at the baseline block length and saves
the output for the table module.
"""

import os
import sys
import time

import numpy as np
from scipy import stats

import config
import dgp
import forecast_eval


def cell_tag(world, beta, t_oos, block_len):
    """Stable string identifier for a cell, used as a key prefix when saving."""
    return f"{world}_beta{beta}_toos{t_oos}_block{block_len}"


def run_cell(world, beta, t_oos, block_len, r_reps, master_children):
    """Run one Monte Carlo cell.

    world, beta, t_oos, block_len: define the cell.
    r_reps: number of replications.
    master_children: sequence of SeedSequence objects, length r_reps, one per
        replication. Each is spawned into two disjoint streams inside, one for
        history generation and one for the bootstrap. Because master_children
        is fixed by the cell identity and not by block_len, the generated
        history for replication i is identical across block-length variants.

    Returns a dict with the per-replication diagnostic arrays (s_obs, skew,
    p_norm, p_boot), the rejection rates at each ALPHA level for both tests,
    and the Monte Carlo standard errors sqrt(p_hat (1 - p_hat) / R).
    """
    t_total = config.T_TRAIN + t_oos
    tag = cell_tag(world, beta, t_oos, block_len)

    s_obs_arr = np.empty(r_reps)
    skew_arr = np.empty(r_reps)
    p_norm_arr = np.empty(r_reps)
    p_boot_arr = np.empty(r_reps)
    boot_std_arr = np.empty(r_reps)     # std of each replication's bootstrap statistics

    start = time.perf_counter()
    for i in range(r_reps):
        # Two disjoint streams per replication: history and bootstrap. The
        # history stream depends only on the replication's child seed, so it
        # is reproducible when block_len changes in later sensitivity runs.
        hist_seq, boot_seq = master_children[i].spawn(2)
        rng_hist = np.random.default_rng(hist_seq)
        rng_boot = np.random.default_rng(boot_seq)

        hist = dgp.generate_history(world, beta, t_total, rng_hist)
        wf = forecast_eval.walk_forward(hist["rv"], hist["x"], config.T_TRAIN)
        d_adj = forecast_eval.cw_adjusted_diff(wf["qlike_small"], wf["qlike_big"],
                                       wf["f_small"], wf["f_big"], hist["rv"])
        s_obs = forecast_eval.cw_stat(d_adj)

        s_obs_arr[i] = s_obs
        skew_arr[i] = stats.skew(d_adj)
        p_norm_arr[i] = forecast_eval.normal_pvalue(s_obs)
        p_boot_arr[i], boot_std_arr[i] = forecast_eval.bootstrap_pvalue(d_adj, s_obs, block_len,
                                                                config.B_BOOT, rng_boot)

        if i + 1 == 10:
            elapsed = time.perf_counter() - start
            est = elapsed / 10 * r_reps
            print(f"  {tag}: {elapsed:.2f}s for 10 reps, estimated {est:.0f}s for {r_reps} reps")
        if (i + 1) % 100 == 0:
            print(f"  {tag}: replication {i + 1}/{r_reps}")

    alpha_levels = np.array(config.ALPHA_LEVELS)
    rej_normal = np.array([np.mean(p_norm_arr <= a) for a in alpha_levels])
    rej_boot = np.array([np.mean(p_boot_arr <= a) for a in alpha_levels])
    se_normal = np.sqrt(rej_normal * (1.0 - rej_normal) / r_reps)
    se_boot = np.sqrt(rej_boot * (1.0 - rej_boot) / r_reps)

    return {
        "tag": tag,
        "world": world,
        "beta": beta,
        "t_oos": t_oos,
        "block_len": block_len,
        "r_reps": r_reps,
        "alpha_levels": alpha_levels,
        "s_obs": s_obs_arr,
        "skew": skew_arr,
        "p_norm": p_norm_arr,
        "p_boot": p_boot_arr,
        "boot_std": boot_std_arr,
        "rej_normal": rej_normal,
        "se_normal": se_normal,
        "rej_boot": rej_boot,
        "se_boot": se_boot,
    }


# Fields written to the npz per cell, prefixed by the cell tag.
_SAVE_FIELDS = ["alpha_levels", "s_obs", "skew", "p_norm", "p_boot", "boot_std",
                "rej_normal", "se_normal", "rej_boot", "se_boot"]


def save_results(results, path):
    """Save a list of cell result dicts to a single npz.

    Array fields are stored under keys "{tag}__{field}". Scalar cell metadata
    (world, beta, t_oos, block_len, r_reps) is stored the same way so the table
    module can reconstruct each cell from the tag list alone.
    """
    payload = {"cell_tags": np.array([r["tag"] for r in results])}
    for r in results:
        tag = r["tag"]
        for field in _SAVE_FIELDS:
            payload[f"{tag}__{field}"] = r[field]
        payload[f"{tag}__world"] = r["world"]
        payload[f"{tag}__beta"] = r["beta"]
        payload[f"{tag}__t_oos"] = r["t_oos"]
        payload[f"{tag}__block_len"] = r["block_len"]
        payload[f"{tag}__r_reps"] = r["r_reps"]
    np.savez(path, **payload)


def print_size_table(results):
    """Print a compact size table, one row per (t_oos, test).

    Columns are the rejection rate at 5% and at 10%, each with its Monte Carlo
    standard error in parentheses.
    """
    alphas = config.ALPHA_LEVELS
    i5 = alphas.index(0.05)
    i10 = alphas.index(0.10)

    print("Size table, null world, baseline block length")
    print("-" * 66)
    print(f"{'t_oos':>6}  {'test':<10}  {'rej@5%':>16}  {'rej@10%':>16}")
    for r in results:
        for name, rej, se in [("CW normal", r["rej_normal"], r["se_normal"]),
                              ("CW boot", r["rej_boot"], r["se_boot"])]:
            c5 = f"{rej[i5]:.3f} ({se[i5]:.3f})"
            c10 = f"{rej[i10]:.3f} ({se[i10]:.3f})"
            print(f"{r['t_oos']:>6}  {name:<10}  {c5:>16}  {c10:>16}")


def print_stat_diagnostics(results):
    """Print per-cell s_obs mean/std and mean bootstrap-statistic std.

    s_obs std near 1.0 confirms the HAC denominator has absorbed the dispersion
    of the CW statistic under the null. The bootstrap-statistic std is the mean
    over replications of each replication's spread of B_BOOT bootstrap
    statistics, at the baseline block length.
    """
    print("Statistic diagnostics, null world, baseline block length")
    print("-" * 66)
    print(f"{'t_oos':>6}  {'s_obs mean':>12}  {'s_obs std':>12}  {'mean boot std':>14}")
    for r in results:
        m = r["s_obs"].mean()
        s = r["s_obs"].std()
        bstd = r["boot_std"].mean()
        print(f"{r['t_oos']:>6}  {m:>12.4f}  {s:>12.4f}  {bstd:>14.4f}")


# ---------------------------------------------------------------------------
# Reduced-scale validation of the data-level (Calhoun) bootstrap
# ---------------------------------------------------------------------------

# Reduced-scale settings for the data-bootstrap size validation. Kept here as
# named constants rather than in config because they define a one-off gating run
# at a deliberately smaller scale, not a study-wide parameter.
VALID_R = 500          # replications for the reduced-scale validation
VALID_B = 299          # data-bootstrap draws per replication at reduced scale
VALID_T_OOS = 220      # out-of-sample length for the validation, the shorter horizon only
VALID_CHECKPOINT_EVERY = 50   # write a checkpoint every this many replications so an interruption can resume
VALID_PARTIAL_NAME = "validation_partial.npz"   # checkpoint file under outputs/, holds the completed prefix
VALID_MODEL_PARTIAL_NAME = "validation_model_partial.npz"   # checkpoint file for the model-bootstrap validation (A scheme, AR(1) generator)
VALID_MODEL2_PARTIAL_NAME = "validation_model2_partial.npz"   # checkpoint file for the A2 amendment (ARMA(1,1) generator), disjoint from A
VALID_ARMA_NONCONV_MAX = 0.01   # stop-and-report threshold: max fraction of replications whose ARMA fit may fail to converge before the run is not gate-valid

# Full-scale size run settings. These orchestrate a study-wide run at the config
# scale (R_SIZE, both T_OOS, B_BOOT_SIZE), so the study parameters come from
# config; only the run-orchestration cadence and file naming live here.
SIZE_FULL_CHECKPOINT_EVERY = 50    # checkpoint cadence for the full size run, one write every this many replications per cell so an interruption loses at most this many replications
SIZE_FULL_PARTIAL_TEMPLATE = "size_full_toos{t_oos}_partial.npz"   # per-cell checkpoint filename under outputs/, one file per T_OOS so a completed cell is never re-run
SIZE_FULL_RESULTS_NAME = "size_full.npz"   # final combined results file under outputs/
# Fields recorded per replication in the full size run. gen_code stores the
# generator actually used (ARMA vs AR(1) fallback) so the non-convergence tally
# and the consistency check against the validation checkpoint reconstruct exactly.
SIZE_FULL_FIELDS = ["s_obs", "p_norm", "p_model", "draw_std", "skew",
                    "phi", "theta", "converged", "gen_code"]


def _validation_paths():
    """Return (outputs dir, checkpoint path) for the validation run, creating the dir."""
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir, os.path.join(out_dir, VALID_PARTIAL_NAME)


def _save_validation_checkpoint(path, n_done, s_obs, p_norm, p_data, draw_std):
    """Write the validation arrays and a done-counter to the checkpoint file.

    n_done is the number of replications completed; only entries [0:n_done] of
    the arrays are meaningful, the rest are NaN placeholders. Written to a
    temporary file then renamed so an interruption mid-write cannot corrupt the
    checkpoint.
    """
    # np.savez appends .npz when the name lacks it, so the temp name ends in
    # .npz already to keep the written path and the rename source in sync.
    tmp = path + ".tmp.npz"
    np.savez(tmp, n_done=n_done, s_obs=s_obs, p_norm=p_norm,
             p_data=p_data, draw_std=draw_std)
    os.replace(tmp, path)


def _load_validation_checkpoint(path):
    """Return (n_done, arrays dict) from an existing checkpoint, or (0, None).

    arrays dict has keys s_obs, p_norm, p_data, draw_std. Absent file means a
    fresh run, signalled by (0, None).
    """
    if not os.path.exists(path):
        return 0, None
    with np.load(path) as data:
        n_done = int(data["n_done"])
        arrays = {k: data[k] for k in ("s_obs", "p_norm", "p_data", "draw_std")}
    return n_done, arrays


def run_validation():
    """Reduced-scale size validation of the data-level (Calhoun) bootstrap.

    Null world, t_oos=220 only, VALID_R replications, VALID_B data-bootstrap
    draws each, baseline block length. Per replication it records the normal
    p-value (unchanged) and the data-bootstrap p-value, plus s_obs and the std
    of that replication's draw statistics. Prints a size table for both tests at
    5% and 10% with Monte Carlo standard errors, the s_obs std across
    replications, the mean draw-statistic std, and the per-replication runtime
    measured after the first ten replications.

    Seeds: children are spawned up front and deterministically from the size
    seed sequence (the same master seed as every other run), each split into a
    disjoint history stream and a disjoint bootstrap stream. Because child i is
    fixed by its position and not by the restart point, replication i always
    draws the same randomness, so a run resumed from a checkpoint reproduces the
    identical results an uninterrupted run would have produced.

    Checkpointing: every VALID_CHECKPOINT_EVERY replications the completed prefix
    of the arrays is written to outputs/validation_partial.npz. On start, if that
    file exists, the finished replications are loaded and the loop continues from
    where it stopped rather than restarting.
    """
    t_oos = VALID_T_OOS
    t_total = config.T_TRAIN + t_oos
    block_len = config.BLOCK_BASELINE

    _, partial_path = _validation_paths()

    # Spawned once for all VALID_R replications; child i is fixed by position so
    # a resumed run reuses the same seeds it would have used uninterrupted.
    children = config.SIZE_SEED_SEQUENCE.spawn(VALID_R)

    # NaN placeholders mark not-yet-computed replications so a partial checkpoint
    # never masquerades as finished data.
    s_obs_arr = np.full(VALID_R, np.nan)
    p_norm_arr = np.full(VALID_R, np.nan)
    p_data_arr = np.full(VALID_R, np.nan)
    draw_std_arr = np.full(VALID_R, np.nan)

    n_done, saved = _load_validation_checkpoint(partial_path)
    if saved is not None:
        s_obs_arr[:n_done] = saved["s_obs"][:n_done]
        p_norm_arr[:n_done] = saved["p_norm"][:n_done]
        p_data_arr[:n_done] = saved["p_data"][:n_done]
        draw_std_arr[:n_done] = saved["draw_std"][:n_done]
        print(f"resuming validation from checkpoint at replication {n_done}/{VALID_R}")

    print(f"running data-bootstrap validation: null world, t_oos={t_oos}, "
          f"block={block_len}, R={VALID_R}, B={VALID_B}")

    # Timing is measured over replications actually computed in this invocation,
    # so the per-replication estimate is meaningful whether the run is fresh or
    # resumed. session_count counts only freshly computed replications.
    session_start = time.perf_counter()
    session_count = 0
    for i in range(n_done, VALID_R):
        hist_seq, boot_seq = children[i].spawn(2)
        rng_hist = np.random.default_rng(hist_seq)
        rng_boot = np.random.default_rng(boot_seq)

        hist = dgp.generate_history("null", 0.0, t_total, rng_hist)

        # s_obs is computed inside data_bootstrap_pvalue on the same code path as
        # the draws, so read it back from a single pipeline call here for the
        # normal p-value rather than recomputing it a second way.
        wf = forecast_eval.walk_forward(hist["rv"], hist["x"], config.T_TRAIN)
        d_adj = forecast_eval.cw_adjusted_diff(wf["qlike_small"], wf["qlike_big"],
                                       wf["f_small"], wf["f_big"], hist["rv"])
        s_obs = forecast_eval.cw_stat(d_adj)

        p_data, draw_std = forecast_eval.data_bootstrap_pvalue(
            hist["rv"], hist["x"], config.T_TRAIN, block_len, VALID_B, rng_boot)

        s_obs_arr[i] = s_obs
        p_norm_arr[i] = forecast_eval.normal_pvalue(s_obs)
        p_data_arr[i] = p_data
        draw_std_arr[i] = draw_std
        session_count += 1

        if session_count == 10:
            elapsed = time.perf_counter() - session_start
            per = elapsed / 10
            print(f"  {per:.3f}s per replication, estimated {per * VALID_R / 60:.1f} min "
                  f"({per * VALID_R / 3600:.2f} h) for {VALID_R} replications")
        if (i + 1) % VALID_CHECKPOINT_EVERY == 0:
            _save_validation_checkpoint(partial_path, i + 1, s_obs_arr,
                                        p_norm_arr, p_data_arr, draw_std_arr)
            print(f"  checkpoint saved at replication {i + 1}/{VALID_R}")

    # Final checkpoint so the full result set is on disk even if VALID_R is not a
    # multiple of the checkpoint interval.
    _save_validation_checkpoint(partial_path, VALID_R, s_obs_arr,
                                p_norm_arr, p_data_arr, draw_std_arr)

    alpha_levels = np.array(config.ALPHA_LEVELS)
    i5 = config.ALPHA_LEVELS.index(0.05)
    i10 = config.ALPHA_LEVELS.index(0.10)
    rej_normal = np.array([np.mean(p_norm_arr <= a) for a in alpha_levels])
    rej_data = np.array([np.mean(p_data_arr <= a) for a in alpha_levels])
    se_normal = np.sqrt(rej_normal * (1.0 - rej_normal) / VALID_R)
    se_data = np.sqrt(rej_data * (1.0 - rej_data) / VALID_R)

    print("Data-bootstrap validation size table, null world, t_oos=220, baseline block")
    print("-" * 70)
    print(f"{'test':<14}  {'rej@5%':>16}  {'rej@10%':>16}")
    for name, rej, se in [("CW normal", rej_normal, se_normal),
                          ("CW data-boot", rej_data, se_data)]:
        c5 = f"{rej[i5]:.3f} ({se[i5]:.3f})"
        c10 = f"{rej[i10]:.3f} ({se[i10]:.3f})"
        print(f"{name:<14}  {c5:>16}  {c10:>16}")
    print(f"s_obs std across reps = {s_obs_arr.std():.4f}")
    print(f"mean draw-statistic std = {draw_std_arr.mean():.4f}")


# Integer codes stored per replication to record which generator produced the
# fake histories, so the fallback tallies reconstruct exactly after a resume.
_GEN_CODE = {"arma": 0, "nonconvergence": 1, "nonstationary": 2, "specified": 3}


def _gen_code(diag):
    """Map a model_bootstrap_pvalue diag dict to its integer generator code.

    Code 0 is the ARMA(1,1) generator, 1 and 2 are the two AR(1) fallbacks after
    an attempted ARMA fit failed, and 3 is a forced AR(1) generator where no ARMA
    fit was attempted, which is what the sensitivity run's mis-specified cell uses.
    """
    if diag["generator"] == "arma":
        return _GEN_CODE["arma"]
    return _GEN_CODE[diag["reason"]]


def _save_validation_model2_checkpoint(path, n_done, s_obs, p_norm, p_model, draw_std, gen_code):
    """Write the A2 validation arrays and a done-counter to the checkpoint file.

    Same atomic temp-then-rename discipline as _save_validation_checkpoint, with
    the extra gen_code array recording the per-replication generator so the
    fallback tallies survive a resume. Only entries [0:n_done] are meaningful.
    """
    tmp = path + ".tmp.npz"
    np.savez(tmp, n_done=n_done, s_obs=s_obs, p_norm=p_norm,
             p_model=p_model, draw_std=draw_std, gen_code=gen_code)
    os.replace(tmp, path)


def _load_validation_model2_checkpoint(path):
    """Return (n_done, arrays dict) from an existing A2 checkpoint, or (0, None).

    arrays dict has keys s_obs, p_norm, p_model, draw_std, gen_code.
    """
    if not os.path.exists(path):
        return 0, None
    with np.load(path) as data:
        n_done = int(data["n_done"])
        arrays = {k: data[k] for k in ("s_obs", "p_norm", "p_model", "draw_std", "gen_code")}
    return n_done, arrays


def _acf(series, lag):
    """Sample autocorrelation of series at the given lag, 1/n normalisation."""
    centred = series - series.mean()
    return float(np.dot(centred[lag:], centred[:-lag]) / np.dot(centred, centred))


def _model2_sanity_print():
    """Print an ARMA(1,1) generator sanity check on one null history.

    Fits the ARMA(1,1) log-rv generator to a single null history and simulates
    one fake path from it, then reports the fitted phi and theta and the lag-1
    and lag-5 autocorrelations of the fake log rv against the observed log rv. If
    the ARMA generator captures the short-run dynamics the fake ACF should track
    the observed ACF closely, unlike the plain AR(1) generator of the A scheme.

    The diagnostic draws from a fresh SeedSequence built on the master seed, which
    is independent of config.SIZE_SEED_SEQUENCE, so it does not disturb the seeds
    the replications draw from.
    """
    t_total = config.T_TRAIN + VALID_T_OOS
    sanity_seq = np.random.SeedSequence(config.MASTER_SEED)
    hist_seq, boot_seq = sanity_seq.spawn(2)

    hist = dgp.generate_history("null", 0.0, t_total, np.random.default_rng(hist_seq))
    log_rv = np.log(hist["rv"])
    arma = forecast_eval._fit_arma11_logrv(log_rv)
    fake = forecast_eval._simulate_arma11_path(arma, log_rv[0], log_rv.size,
                                       np.random.default_rng(boot_seq))

    print("ARMA(1,1) generator sanity, one null history")
    print("-" * 66)
    print(f"fitted phi={arma['phi']:.4f}, theta={arma['theta']:.4f}, converged={arma['converged']}")
    print(f"observed log rv ACF: lag1={_acf(log_rv, 1):.4f}, lag5={_acf(log_rv, 5):.4f}")
    print(f"fake     log rv ACF: lag1={_acf(fake, 1):.4f}, lag5={_acf(fake, 5):.4f}")


def run_validation_model():
    """Reduced-scale size validation of the A2 model bootstrap (ARMA(1,1) generator).

    Identical harness to run_validation and to the A scheme: null world,
    t_oos=220 only, VALID_R replications, VALID_B bootstrap draws each, same
    master seed and spawn discipline, checkpoint every VALID_CHECKPOINT_EVERY
    replications with seed-exact resume. The only change from A is the null
    generator inside forecast_eval.model_bootstrap_pvalue, now ARMA(1,1) on log rv rather
    than AR(1), so results are comparable one-for-one with the A run: each
    replication faces the identical observed history. Records the normal p-value
    and the model-bootstrap p-value, plus the per-replication generator code so
    the ARMA-vs-fallback tallies are known.

    Fresh checkpoint file (VALID_MODEL2_PARTIAL_NAME), disjoint from the A run.
    Prints a sanity check first, then a size table for both tests at 5% and 10%
    with Monte Carlo standard errors, the s_obs std across replications, the mean
    draw-statistic std, the generator-usage tallies, and the runtime.

    If the ARMA fit fails to converge on more than VALID_ARMA_NONCONV_MAX of the
    replications, the run is reported as not gate-valid rather than silently
    resting on AR(1) fallbacks: the non-convergence rate is printed and the size
    table is flagged so the gate is not read off a degraded generator.
    """
    t_oos = VALID_T_OOS
    t_total = config.T_TRAIN + t_oos

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    partial_path = os.path.join(out_dir, VALID_MODEL2_PARTIAL_NAME)

    _model2_sanity_print()

    # Spawned once for all VALID_R replications from the same size seed sequence
    # as the A run; child i is fixed by position so a resumed run reuses the same
    # seeds, and each replication faces the same history the A run gave it.
    children = config.SIZE_SEED_SEQUENCE.spawn(VALID_R)

    s_obs_arr = np.full(VALID_R, np.nan)
    p_norm_arr = np.full(VALID_R, np.nan)
    p_model_arr = np.full(VALID_R, np.nan)
    draw_std_arr = np.full(VALID_R, np.nan)
    gen_code_arr = np.full(VALID_R, -1, dtype=int)     # -1 marks a not-yet-computed replication

    n_done, saved = _load_validation_model2_checkpoint(partial_path)
    if saved is not None:
        s_obs_arr[:n_done] = saved["s_obs"][:n_done]
        p_norm_arr[:n_done] = saved["p_norm"][:n_done]
        p_model_arr[:n_done] = saved["p_model"][:n_done]
        draw_std_arr[:n_done] = saved["draw_std"][:n_done]
        gen_code_arr[:n_done] = saved["gen_code"][:n_done]
        print(f"resuming A2 model-bootstrap validation from checkpoint at replication {n_done}/{VALID_R}")

    print(f"running A2 model-bootstrap validation (ARMA(1,1) generator): null world, "
          f"t_oos={t_oos}, R={VALID_R}, B={VALID_B}")

    session_start = time.perf_counter()
    session_count = 0
    for i in range(n_done, VALID_R):
        hist_seq, boot_seq = children[i].spawn(2)
        rng_hist = np.random.default_rng(hist_seq)
        rng_boot = np.random.default_rng(boot_seq)

        hist = dgp.generate_history("null", 0.0, t_total, rng_hist)

        # s_obs for the normal p-value, on the same code path model_bootstrap_pvalue
        # uses internally for its own s_obs.
        wf = forecast_eval.walk_forward(hist["rv"], hist["x"], config.T_TRAIN)
        d_adj = forecast_eval.cw_adjusted_diff(wf["qlike_small"], wf["qlike_big"],
                                       wf["f_small"], wf["f_big"], hist["rv"])
        s_obs = forecast_eval.cw_stat(d_adj)

        p_model, draw_std, diag = forecast_eval.model_bootstrap_pvalue(
            hist["rv"], hist["x"], config.T_TRAIN, VALID_B, rng_boot)

        s_obs_arr[i] = s_obs
        p_norm_arr[i] = forecast_eval.normal_pvalue(s_obs)
        p_model_arr[i] = p_model
        draw_std_arr[i] = draw_std
        gen_code_arr[i] = _gen_code(diag)
        session_count += 1

        if session_count == 10:
            elapsed = time.perf_counter() - session_start
            per = elapsed / 10
            print(f"  {per:.3f}s per replication, estimated {per * VALID_R / 60:.1f} min "
                  f"({per * VALID_R / 3600:.2f} h) for {VALID_R} replications")
        if (i + 1) % VALID_CHECKPOINT_EVERY == 0:
            _save_validation_model2_checkpoint(partial_path, i + 1, s_obs_arr,
                                               p_norm_arr, p_model_arr, draw_std_arr, gen_code_arr)
            print(f"  checkpoint saved at replication {i + 1}/{VALID_R}")

    _save_validation_model2_checkpoint(partial_path, VALID_R, s_obs_arr,
                                       p_norm_arr, p_model_arr, draw_std_arr, gen_code_arr)

    alpha_levels = np.array(config.ALPHA_LEVELS)
    i5 = config.ALPHA_LEVELS.index(0.05)
    i10 = config.ALPHA_LEVELS.index(0.10)
    rej_normal = np.array([np.mean(p_norm_arr <= a) for a in alpha_levels])
    rej_model = np.array([np.mean(p_model_arr <= a) for a in alpha_levels])
    se_normal = np.sqrt(rej_normal * (1.0 - rej_normal) / VALID_R)
    se_model = np.sqrt(rej_model * (1.0 - rej_model) / VALID_R)

    n_arma = int(np.sum(gen_code_arr == _GEN_CODE["arma"]))
    n_nonconv = int(np.sum(gen_code_arr == _GEN_CODE["nonconvergence"]))
    n_nonstat = int(np.sum(gen_code_arr == _GEN_CODE["nonstationary"]))
    nonconv_rate = n_nonconv / VALID_R

    total_elapsed = time.perf_counter() - session_start
    print("A2 model-bootstrap validation size table, null world, t_oos=220")
    print("-" * 70)
    print(f"{'test':<15}  {'rej@5%':>16}  {'rej@10%':>16}")
    for name, rej, se in [("CW normal", rej_normal, se_normal),
                          ("CW model-boot", rej_model, se_model)]:
        c5 = f"{rej[i5]:.3f} ({se[i5]:.3f})"
        c10 = f"{rej[i10]:.3f} ({se[i10]:.3f})"
        print(f"{name:<15}  {c5:>16}  {c10:>16}")
    print(f"s_obs std across reps = {s_obs_arr.std():.4f}")
    print(f"mean draw-statistic std = {draw_std_arr.mean():.4f}")
    print(f"generator usage: ARMA={n_arma}, AR(1) fallback nonconvergence={n_nonconv}, "
          f"AR(1) fallback nonstationary={n_nonstat}")
    print(f"ARMA non-convergence rate = {nonconv_rate:.4f} "
          f"(threshold {VALID_ARMA_NONCONV_MAX:.4f})")
    if nonconv_rate > VALID_ARMA_NONCONV_MAX:
        print("STOP: ARMA non-convergence rate exceeds the threshold; this run is "
              "NOT gate-valid, the size table rests on degraded AR(1) fallbacks.")
    print(f"runtime this session = {total_elapsed:.1f}s for {session_count} replications")


# ---------------------------------------------------------------------------
# Full-scale size runs (A2 model bootstrap, both T_OOS, R_SIZE replications)
# ---------------------------------------------------------------------------

def _save_size_full_checkpoint(path, n_done, arrays):
    """Write the full-size per-cell arrays and a done-counter atomically.

    arrays is a dict keyed by SIZE_FULL_FIELDS. Only entries [0:n_done] are
    meaningful, the rest hold placeholders. Written to a temp file then renamed so
    an interruption mid-write cannot corrupt the checkpoint.
    """
    tmp = path + ".tmp.npz"
    np.savez(tmp, n_done=n_done, **{k: arrays[k] for k in SIZE_FULL_FIELDS})
    os.replace(tmp, path)


def _load_size_full_checkpoint(path):
    """Return (n_done, arrays dict) from an existing full-size checkpoint, or (0, None).

    arrays dict has the keys in SIZE_FULL_FIELDS. Absent file means a fresh cell,
    signalled by (0, None).
    """
    if not os.path.exists(path):
        return 0, None
    with np.load(path) as data:
        n_done = int(data["n_done"])
        arrays = {k: data[k] for k in SIZE_FULL_FIELDS}
    return n_done, arrays


def _run_size_cell(t_oos, cell_children, partial_path):
    """Run one full-size cell (null world, given t_oos) with checkpoint resume.

    cell_children is the length-R_SIZE slice of spawned SeedSequences for this
    cell; child i is fixed by position, so a resumed run reuses the same seeds and
    replication i is identical whether the run is fresh or resumed. Records the
    per-replication arrays named in SIZE_FULL_FIELDS. Returns (arrays, elapsed)
    where elapsed is the wall-clock time spent computing replications in this
    invocation.
    """
    t_total = config.T_TRAIN + t_oos
    b_boot = config.B_BOOT_SIZE
    r = config.R_SIZE

    s_obs_arr = np.full(r, np.nan)
    p_norm_arr = np.full(r, np.nan)
    p_model_arr = np.full(r, np.nan)
    draw_std_arr = np.full(r, np.nan)
    skew_arr = np.full(r, np.nan)
    phi_arr = np.full(r, np.nan)
    theta_arr = np.full(r, np.nan)
    converged_arr = np.full(r, -1, dtype=int)   # -1 marks a not-yet-computed replication
    gen_code_arr = np.full(r, -1, dtype=int)

    def pack():
        return {"s_obs": s_obs_arr, "p_norm": p_norm_arr, "p_model": p_model_arr,
                "draw_std": draw_std_arr, "skew": skew_arr, "phi": phi_arr,
                "theta": theta_arr, "converged": converged_arr, "gen_code": gen_code_arr}

    n_done, saved = _load_size_full_checkpoint(partial_path)
    if saved is not None:
        for k, dst in pack().items():
            dst[:n_done] = saved[k][:n_done]
        print(f"  resuming t_oos={t_oos} cell from checkpoint at replication {n_done}/{r}")

    # Timing is measured over replications computed in this invocation only, so the
    # per-replication estimate is meaningful whether the cell is fresh or resumed.
    session_start = time.perf_counter()
    session_count = 0
    for i in range(n_done, r):
        hist_seq, boot_seq = cell_children[i].spawn(2)
        rng_hist = np.random.default_rng(hist_seq)
        rng_boot = np.random.default_rng(boot_seq)

        hist = dgp.generate_history("null", 0.0, t_total, rng_hist)

        # s_obs for the normal p-value, on the same pipeline model_bootstrap_pvalue
        # uses internally for its own s_obs.
        wf = forecast_eval.walk_forward(hist["rv"], hist["x"], config.T_TRAIN)
        d_adj = forecast_eval.cw_adjusted_diff(wf["qlike_small"], wf["qlike_big"],
                                       wf["f_small"], wf["f_big"], hist["rv"])
        s_obs = forecast_eval.cw_stat(d_adj)

        p_model, draw_std, diag = forecast_eval.model_bootstrap_pvalue(
            hist["rv"], hist["x"], config.T_TRAIN, b_boot, rng_boot)

        # Re-fit the ARMA(1,1) to record the fitted parameters. The fit uses no
        # randomness, so it reproduces the fit model_bootstrap_pvalue did
        # internally on the same log rv and touches neither replication RNG stream,
        # leaving the seed-exact resume intact. If the fit raises, phi and theta
        # stay NaN and converged is 0, matching the AR(1) nonconvergence fallback
        # the bootstrap took on the same history.
        try:
            fit = forecast_eval._fit_arma11_logrv(np.log(hist["rv"]))
            phi_arr[i] = fit["phi"]
            theta_arr[i] = fit["theta"]
            converged_arr[i] = int(fit["converged"])
        except Exception:
            converged_arr[i] = 0

        s_obs_arr[i] = s_obs
        p_norm_arr[i] = forecast_eval.normal_pvalue(s_obs)
        p_model_arr[i] = p_model
        draw_std_arr[i] = draw_std
        skew_arr[i] = stats.skew(d_adj)
        gen_code_arr[i] = _gen_code(diag)
        session_count += 1

        if session_count == 10:
            elapsed = time.perf_counter() - session_start
            per = elapsed / 10
            print(f"  t_oos={t_oos}: {per:.3f}s per replication, projected "
                  f"{per * r / 60:.1f} min ({per * r / 3600:.2f} h) for {r} replications")
        if (i + 1) % SIZE_FULL_CHECKPOINT_EVERY == 0:
            _save_size_full_checkpoint(partial_path, i + 1, pack())
            print(f"  t_oos={t_oos}: checkpoint saved at replication {i + 1}/{r}")

    # Final checkpoint so the full cell is on disk even if R_SIZE is not a multiple
    # of the checkpoint interval.
    _save_size_full_checkpoint(partial_path, r, pack())
    elapsed = time.perf_counter() - session_start
    return pack(), elapsed


def _size_consistency_check(cell_arrays, t_oos):
    """Compare a full-size cell's first VALID_R replications against the A2 validation checkpoint.

    The t_oos=220 cell reuses the same master seed and spawn discipline as the A2
    validation, so its first VALID_R replications must reproduce the validation
    results exactly. Loads the validation checkpoint and prints the maximum
    absolute differences on s_obs, p_norm, p_model, draw_std, whether gen_code
    matches, and a single exact-match verdict. Runs only for the t_oos the
    validation used; returns immediately otherwise.
    """
    if t_oos != VALID_T_OOS:
        return
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    ref_path = os.path.join(out_dir, VALID_MODEL2_PARTIAL_NAME)
    if not os.path.exists(ref_path):
        print(f"  consistency check skipped: validation checkpoint {VALID_MODEL2_PARTIAL_NAME} not found")
        return
    with np.load(ref_path) as ref:
        n_ref = int(ref["n_done"])
        float_fields = ["s_obs", "p_norm", "p_model", "draw_std"]
        diffs = {f: float(np.max(np.abs(cell_arrays[f][:n_ref] - ref[f][:n_ref])))
                 for f in float_fields}
        gen_match = bool(np.array_equal(cell_arrays["gen_code"][:n_ref], ref["gen_code"][:n_ref]))
    exact = all(d == 0.0 for d in diffs.values()) and gen_match
    print(f"Consistency check: t_oos={t_oos} first {n_ref} reps vs A2 validation checkpoint")
    print("-" * 70)
    for f in ["s_obs", "p_norm", "p_model", "draw_std"]:
        print(f"  {f:<10} max|diff| = {diffs[f]:.3e}")
    print(f"  gen_code identical = {gen_match}")
    print(f"  exact reproduction of validation = {exact}")


def _size_rej_rates(p_arr, r):
    """Rejection rates and Monte Carlo standard errors at the config alpha levels."""
    alpha_levels = np.array(config.ALPHA_LEVELS)
    rej = np.array([np.mean(p_arr <= a) for a in alpha_levels])
    se = np.sqrt(rej * (1.0 - rej) / r)
    return rej, se


def _print_size_full_cell(t_oos, arrays, elapsed):
    """Print the per-cell diagnostics right after a cell finishes."""
    r = config.R_SIZE
    i5 = config.ALPHA_LEVELS.index(0.05)
    i10 = config.ALPHA_LEVELS.index(0.10)
    rej_n, se_n = _size_rej_rates(arrays["p_norm"], r)
    rej_m, se_m = _size_rej_rates(arrays["p_model"], r)
    skew = arrays["skew"]
    p5, p50, p95 = np.percentile(skew, [5, 50, 95])
    n_arma = int(np.sum(arrays["gen_code"] == _GEN_CODE["arma"]))
    n_nonconv = int(np.sum(arrays["gen_code"] == _GEN_CODE["nonconvergence"]))
    n_nonstat = int(np.sum(arrays["gen_code"] == _GEN_CODE["nonstationary"]))
    print(f"cell t_oos={t_oos} summary")
    print("-" * 70)
    print(f"  CW normal     rej@5%={rej_n[i5]:.3f} ({se_n[i5]:.3f})  rej@10%={rej_n[i10]:.3f} ({se_n[i10]:.3f})")
    print(f"  CW model-boot rej@5%={rej_m[i5]:.3f} ({se_m[i5]:.3f})  rej@10%={rej_m[i10]:.3f} ({se_m[i10]:.3f})")
    print(f"  s_obs std = {arrays['s_obs'].std():.4f}, mean draw-statistic std = {arrays['draw_std'].mean():.4f}")
    print(f"  d_adj skew: mean={skew.mean():.4f}, 5th={p5:.4f}, 50th={p50:.4f}, 95th={p95:.4f}")
    print(f"  generator usage: ARMA={n_arma}, nonconvergence fallback={n_nonconv}, "
          f"nonstationary fallback={n_nonstat}")
    print(f"  ARMA non-convergence count = {n_nonconv}")
    print(f"  runtime this cell (this session) = {elapsed:.1f}s")


def _print_size_full_table(cell_results):
    """Print the combined size table, one row per (t_oos, test)."""
    r = config.R_SIZE
    i5 = config.ALPHA_LEVELS.index(0.05)
    i10 = config.ALPHA_LEVELS.index(0.10)
    print("Full size table, null world, ARMA(1,1) model bootstrap")
    print("-" * 74)
    print(f"{'t_oos':>6}  {'test':<14}  {'rej@5%':>16}  {'rej@10%':>16}")
    for t_oos, arrays in cell_results:
        rej_n, se_n = _size_rej_rates(arrays["p_norm"], r)
        rej_m, se_m = _size_rej_rates(arrays["p_model"], r)
        for name, rej, se in [("CW normal", rej_n, se_n), ("CW model-boot", rej_m, se_m)]:
            c5 = f"{rej[i5]:.3f} ({se[i5]:.3f})"
            c10 = f"{rej[i10]:.3f} ({se[i10]:.3f})"
            print(f"{t_oos:>6}  {name:<14}  {c5:>16}  {c10:>16}")


def _save_size_full_results(out_dir, cell_results):
    """Save the combined full-size arrays to outputs/size_full.npz.

    Per-cell arrays are stored under keys "toos{t_oos}__{field}". Run-level
    metadata (the T_OOS grid, R_SIZE, B_BOOT_SIZE) is stored so the table module
    can reconstruct each cell from the file alone.
    """
    path = os.path.join(out_dir, SIZE_FULL_RESULTS_NAME)
    payload = {
        "t_oos_grid": np.array([t for t, _ in cell_results]),
        "r_size": config.R_SIZE,
        "b_boot_size": config.B_BOOT_SIZE,
    }
    for t_oos, arrays in cell_results:
        for f in SIZE_FULL_FIELDS:
            payload[f"toos{t_oos}__{f}"] = arrays[f]
    np.savez(path, **payload)
    print(f"saved full size results to {path}")


def run_size_full():
    """Full-scale size runs with the validated construction (A2 model bootstrap).

    Null world, both T_OOS values, R_SIZE replications each, B_BOOT_SIZE model
    bootstrap draws, ARMA(1,1) generator. Same master seed and spawn discipline as
    every prior run: children are spawned once from the size seed sequence and
    sliced per cell, so the first cell (t_oos=220) reproduces the A2 validation's
    first VALID_R replications exactly, which is checked and reported. Cells run
    sequentially, each finishing and saving its own checkpoint before the next
    starts, so an interruption never loses a completed cell and resumes the
    in-progress cell from its last checkpoint.

    Records per replication: s_obs, normal p-value, model-bootstrap p-value, d_adj
    skewness, draw-statistic std, ARMA fitted phi and theta, and the convergence
    flag. Saves the combined arrays to outputs/size_full.npz and prints the size
    table (both tests, both T_OOS, rejection at 5% and 10% with Monte Carlo
    standard errors), per-cell s_obs std and mean draw-statistic std, the mean and
    5th, 50th, 95th percentiles of d_adj skewness, the ARMA non-convergence count,
    and the runtime. Each cell reports as it finishes rather than waiting for both.
    """
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    num_cells = len(config.T_OOS_GRID)
    # Spawn once from the size seed sequence, then slice per cell exactly as
    # montecarlo.main does. Spawning once (not per cell) is what keeps cell 0's
    # first VALID_R children identical to the validation run's spawn, and gives
    # cell 1 a disjoint seed slice.
    children = config.SIZE_SEED_SEQUENCE.spawn(num_cells * config.R_SIZE)

    print(f"running full size runs: null world, T_OOS={config.T_OOS_GRID}, "
          f"R={config.R_SIZE}, B={config.B_BOOT_SIZE}, ARMA(1,1) model bootstrap")

    cell_results = []
    total_elapsed = 0.0
    for k, t_oos in enumerate(config.T_OOS_GRID):
        cell_children = children[k * config.R_SIZE:(k + 1) * config.R_SIZE]
        partial_path = os.path.join(out_dir, SIZE_FULL_PARTIAL_TEMPLATE.format(t_oos=t_oos))
        print(f"cell {k + 1}/{num_cells}: null world, t_oos={t_oos}")
        arrays, elapsed = _run_size_cell(t_oos, cell_children, partial_path)
        total_elapsed += elapsed
        cell_results.append((t_oos, arrays))
        _size_consistency_check(arrays, t_oos)
        _print_size_full_cell(t_oos, arrays, elapsed)

    _save_size_full_results(out_dir, cell_results)
    _print_size_full_table(cell_results)
    print(f"total runtime this session = {total_elapsed:.1f}s across {num_cells} cells")


# ---------------------------------------------------------------------------
# Full-scale power runs (A2 model bootstrap, both betas, both T_OOS)
# ---------------------------------------------------------------------------

# Run-orchestration settings for the power runs. The study parameters (BETA_GRID,
# T_OOS_GRID, R_POWER, B_BOOT_POWER) come from config; only the cadence and file
# naming live here, matching the full size run.
POWER_CHECKPOINT_EVERY = 50    # checkpoint cadence for the power runs, one write every this many replications per cell
POWER_PARTIAL_TEMPLATE = "power_beta{beta_tag}_toos{t_oos}_partial.npz"   # per-cell checkpoint filename under outputs/, one file per (beta, T_OOS) so a completed cell is never re-run
POWER_RESULTS_NAME = "power_full.npz"   # final combined results file under outputs/
# Fields recorded per replication. pct_gain is the realised percentage QLIKE
# reduction of the big model over the out-of-sample window, on the same
# definition the beta calibration used. gen_code records the generator actually
# used by the bootstrap (ARMA vs AR(1) fallback), as in the size runs.
POWER_FIELDS = ["s_obs", "p_norm", "p_model", "draw_std", "pct_gain", "gen_code"]


def _beta_tag(beta):
    """Filename-safe tag for a beta, e.g. 0.03 becomes 0p03."""
    return f"{beta:g}".replace(".", "p")


def power_cell_tag(beta, t_oos):
    """Stable key prefix for one power cell in the saved npz."""
    return f"beta{_beta_tag(beta)}_toos{t_oos}"


def _save_power_checkpoint(path, n_done, arrays):
    """Write the power per-cell arrays and a done-counter atomically.

    arrays is a dict keyed by POWER_FIELDS. Only entries [0:n_done] are
    meaningful. Written to a temp file then renamed so an interruption mid-write
    cannot corrupt the checkpoint.
    """
    tmp = path + ".tmp.npz"
    np.savez(tmp, n_done=n_done, **{k: arrays[k] for k in POWER_FIELDS})
    os.replace(tmp, path)


def _load_power_checkpoint(path):
    """Return (n_done, arrays dict) from an existing power checkpoint, or (0, None)."""
    if not os.path.exists(path):
        return 0, None
    with np.load(path) as data:
        n_done = int(data["n_done"])
        arrays = {k: data[k] for k in POWER_FIELDS}
    return n_done, arrays


def _run_power_cell(beta, t_oos, cell_children, partial_path):
    """Run one power cell (alt world, given beta and t_oos) with checkpoint resume.

    cell_children is the length-R_POWER slice of spawned SeedSequences for this
    cell; child i is fixed by position, so a resumed run reuses the same seeds and
    replication i is identical whether the run is fresh or resumed. Records the
    per-replication arrays named in POWER_FIELDS. Returns (arrays, elapsed) where
    elapsed is the wall-clock time spent computing replications in this
    invocation.
    """
    t_total = config.T_TRAIN + t_oos
    b_boot = config.B_BOOT_POWER
    r = config.R_POWER

    s_obs_arr = np.full(r, np.nan)
    p_norm_arr = np.full(r, np.nan)
    p_model_arr = np.full(r, np.nan)
    draw_std_arr = np.full(r, np.nan)
    pct_gain_arr = np.full(r, np.nan)
    gen_code_arr = np.full(r, -1, dtype=int)   # -1 marks a not-yet-computed replication

    def pack():
        return {"s_obs": s_obs_arr, "p_norm": p_norm_arr, "p_model": p_model_arr,
                "draw_std": draw_std_arr, "pct_gain": pct_gain_arr,
                "gen_code": gen_code_arr}

    n_done, saved = _load_power_checkpoint(partial_path)
    if saved is not None:
        for k, dst in pack().items():
            dst[:n_done] = saved[k][:n_done]
        print(f"  resuming beta={beta}, t_oos={t_oos} cell from checkpoint at "
              f"replication {n_done}/{r}")

    # Timing is measured over replications computed in this invocation only, so the
    # per-replication estimate is meaningful whether the cell is fresh or resumed.
    session_start = time.perf_counter()
    session_count = 0
    for i in range(n_done, r):
        hist_seq, boot_seq = cell_children[i].spawn(2)
        rng_hist = np.random.default_rng(hist_seq)
        rng_boot = np.random.default_rng(boot_seq)

        hist = dgp.generate_history("alt", beta, t_total, rng_hist)

        # One walk_forward serves both the statistic and the realised QLIKE gain.
        wf = forecast_eval.walk_forward(hist["rv"], hist["x"], config.T_TRAIN)
        d_adj = forecast_eval.cw_adjusted_diff(wf["qlike_small"], wf["qlike_big"],
                                       wf["f_small"], wf["f_big"], hist["rv"])
        s_obs = forecast_eval.cw_stat(d_adj)

        q_small = wf["qlike_small"].mean()
        q_big = wf["qlike_big"].mean()

        p_model, draw_std, diag = forecast_eval.model_bootstrap_pvalue(
            hist["rv"], hist["x"], config.T_TRAIN, b_boot, rng_boot)

        s_obs_arr[i] = s_obs
        p_norm_arr[i] = forecast_eval.normal_pvalue(s_obs)
        p_model_arr[i] = p_model
        draw_std_arr[i] = draw_std
        pct_gain_arr[i] = 100.0 * (q_small - q_big) / q_small
        gen_code_arr[i] = _gen_code(diag)
        session_count += 1

        if session_count == 10:
            elapsed = time.perf_counter() - session_start
            per = elapsed / 10
            print(f"  beta={beta}, t_oos={t_oos}: {per:.3f}s per replication, projected "
                  f"{per * r / 60:.1f} min ({per * r / 3600:.2f} h) for {r} replications")
        if (i + 1) % POWER_CHECKPOINT_EVERY == 0:
            _save_power_checkpoint(partial_path, i + 1, pack())
            print(f"  beta={beta}, t_oos={t_oos}: checkpoint saved at replication {i + 1}/{r}")

    # Final checkpoint so the full cell is on disk even if R_POWER is not a
    # multiple of the checkpoint interval.
    _save_power_checkpoint(partial_path, r, pack())
    elapsed = time.perf_counter() - session_start
    return pack(), elapsed


def _null_cutoffs(t_oos):
    """Empirical null cutoffs for the size-adjusted power columns, matched on t_oos.

    Read from the already-saved outputs/size_full.npz; no null cell is refit or
    rerun. Returns a dict with, per config alpha level, the null s_obs upper
    quantile (reject when s_obs is at or above it) and the null model-bootstrap
    p-value lower quantile (reject when p_model is at or below it), or None if the
    size results file is absent.

    The size adjustment exists because the normal test over-rejects under the
    null, so its raw power is inflated by the same distortion the size table
    reports; only against an empirically correct null cutoff is the comparison
    with the bootstrap test like-for-like.
    """
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    path = os.path.join(out_dir, SIZE_FULL_RESULTS_NAME)
    if not os.path.exists(path):
        return None
    with np.load(path) as data:
        grid = list(data["t_oos_grid"])
        if t_oos not in grid:
            return None
        s_null = data[f"toos{t_oos}__s_obs"]
        p_null = data[f"toos{t_oos}__p_model"]
    alphas = np.array(config.ALPHA_LEVELS)
    # s_obs is a one-sided upper-tail statistic, so its size-adjusted cutoff is the
    # (1 - alpha) null quantile. The bootstrap p-value rejects in its lower tail,
    # so its own cutoff is the alpha null quantile.
    return {
        "s_obs_cut": np.percentile(s_null, 100.0 * (1.0 - alphas)),
        "p_model_cut": np.percentile(p_null, 100.0 * alphas),
        "n_null": s_null.size,
    }


def _power_rej(mask_per_alpha, r):
    """Rejection rates and Monte Carlo standard errors from per-alpha boolean masks."""
    rej = np.array([np.mean(m) for m in mask_per_alpha])
    se = np.sqrt(rej * (1.0 - rej) / r)
    return rej, se


def _power_cell_stats(arrays, cutoffs):
    """Rejection rates for every reported column of one power cell.

    Returns a dict mapping a column name to (rejection rates, MC standard errors)
    over the config alpha levels. Raw columns use the nominal cutoffs; the
    size-adjusted columns use the empirical null cutoffs in cutoffs, which is None
    when the size results are unavailable.
    """
    r = config.R_POWER
    alphas = config.ALPHA_LEVELS
    out = {
        "CW normal": _power_rej([arrays["p_norm"] <= a for a in alphas], r),
        "CW model-boot": _power_rej([arrays["p_model"] <= a for a in alphas], r),
    }
    if cutoffs is not None:
        # Both raw tests are read off the same s_obs, so a cutoff taken from the
        # null s_obs distribution gives one size-adjusted column that applies to
        # both of them. The bootstrap test additionally has its own null p-value
        # distribution, so it also gets a column calibrated on that.
        out["size-adj s_obs"] = _power_rej(
            [arrays["s_obs"] >= c for c in cutoffs["s_obs_cut"]], r)
        out["size-adj boot-p"] = _power_rej(
            [arrays["p_model"] <= c for c in cutoffs["p_model_cut"]], r)
    return out


# Column order for the power tables, raw tests first then the size-adjusted ones.
POWER_COLUMNS = ["CW normal", "CW model-boot", "size-adj s_obs", "size-adj boot-p"]


def _print_power_cell(beta, t_oos, arrays, cutoffs, elapsed):
    """Print the per-cell diagnostics right after a cell finishes."""
    i5 = config.ALPHA_LEVELS.index(0.05)
    i10 = config.ALPHA_LEVELS.index(0.10)
    stats_by_col = _power_cell_stats(arrays, cutoffs)
    n_arma = int(np.sum(arrays["gen_code"] == _GEN_CODE["arma"]))
    n_nonconv = int(np.sum(arrays["gen_code"] == _GEN_CODE["nonconvergence"]))
    n_nonstat = int(np.sum(arrays["gen_code"] == _GEN_CODE["nonstationary"]))

    print(f"cell beta={beta}, t_oos={t_oos} summary")
    print("-" * 74)
    for name in POWER_COLUMNS:
        if name not in stats_by_col:
            continue
        rej, se = stats_by_col[name]
        print(f"  {name:<16} rej@5%={rej[i5]:.3f} ({se[i5]:.3f})  "
              f"rej@10%={rej[i10]:.3f} ({se[i10]:.3f})")
    if cutoffs is None:
        print("  size-adjusted columns skipped: outputs/size_full.npz not found")
    else:
        cuts = cutoffs["s_obs_cut"]
        pcuts = cutoffs["p_model_cut"]
        print(f"  null cutoffs from {cutoffs['n_null']} null reps: "
              f"s_obs 5%={cuts[i5]:.4f}, 10%={cuts[i10]:.4f}; "
              f"boot-p 5%={pcuts[i5]:.4f}, 10%={pcuts[i10]:.4f}")
    print(f"  mean realised QLIKE gain = {arrays['pct_gain'].mean():.4f}%")
    print(f"  s_obs mean = {arrays['s_obs'].mean():.4f}, std = {arrays['s_obs'].std():.4f}")
    print(f"  mean draw-statistic std = {arrays['draw_std'].mean():.4f}")
    print(f"  generator usage: ARMA={n_arma}, nonconvergence fallback={n_nonconv}, "
          f"nonstationary fallback={n_nonstat}")
    print(f"  runtime this cell (this session) = {elapsed:.1f}s")


def _print_power_table(cell_results):
    """Print the combined power table, one row per (beta, t_oos, column)."""
    i5 = config.ALPHA_LEVELS.index(0.05)
    i10 = config.ALPHA_LEVELS.index(0.10)
    print("Full power table, alt world, ARMA(1,1) model bootstrap")
    print("-" * 88)
    print(f"{'beta':>6}  {'t_oos':>6}  {'test':<16}  {'rej@5%':>16}  {'rej@10%':>16}  {'mean gain%':>10}")
    for beta, t_oos, arrays, cutoffs in cell_results:
        stats_by_col = _power_cell_stats(arrays, cutoffs)
        gain = arrays["pct_gain"].mean()
        for name in POWER_COLUMNS:
            if name not in stats_by_col:
                continue
            rej, se = stats_by_col[name]
            c5 = f"{rej[i5]:.3f} ({se[i5]:.3f})"
            c10 = f"{rej[i10]:.3f} ({se[i10]:.3f})"
            print(f"{beta:>6.2f}  {t_oos:>6}  {name:<16}  {c5:>16}  {c10:>16}  {gain:>10.3f}")


def _save_power_results(out_dir, cell_results):
    """Save the combined power arrays to outputs/power_full.npz.

    Per-cell arrays are stored under keys "{cell tag}__{field}", with the cell
    tags and the run-level metadata (beta grid, T_OOS grid, R_POWER,
    B_BOOT_POWER) alongside so the table module can reconstruct each cell from
    the file alone. Called after every cell, so a run interrupted between cells
    still leaves a readable results file for the cells that finished.
    """
    path = os.path.join(out_dir, POWER_RESULTS_NAME)
    payload = {
        "cell_tags": np.array([power_cell_tag(b, t) for b, t, _, _ in cell_results]),
        "cell_beta": np.array([b for b, _, _, _ in cell_results]),
        "cell_t_oos": np.array([t for _, t, _, _ in cell_results]),
        "beta_grid": np.array(config.BETA_GRID),
        "t_oos_grid": np.array(config.T_OOS_GRID),
        "r_power": config.R_POWER,
        "b_boot_power": config.B_BOOT_POWER,
    }
    for beta, t_oos, arrays, cutoffs in cell_results:
        tag = power_cell_tag(beta, t_oos)
        for f in POWER_FIELDS:
            payload[f"{tag}__{f}"] = arrays[f]
        if cutoffs is not None:
            payload[f"{tag}__s_obs_cut"] = cutoffs["s_obs_cut"]
            payload[f"{tag}__p_model_cut"] = cutoffs["p_model_cut"]
    np.savez(path, **payload)
    print(f"saved power results ({len(cell_results)} cells) to {path}")


def run_power():
    """Full-scale power runs with the validated construction (A2 model bootstrap).

    Alt world, both betas in config.BETA_GRID crossed with both T_OOS values,
    R_POWER replications each, B_BOOT_POWER model bootstrap draws. Seeds come
    from config.POWER_SEED_SEQUENCE, a root child disjoint from the size and
    calibration sequences; children are spawned once and sliced per cell, so
    every cell has its own disjoint seed range and child i is fixed by position
    for a seed-exact resume. The four cells run sequentially and each one saves
    its checkpoint, writes the combined results file and prints its summary
    before the next starts, so a run interrupted part way still leaves every
    finished cell on disk and reported.

    Records per replication: s_obs, the normal p-value, the model-bootstrap
    p-value, the draw-statistic std, the realised percentage QLIKE gain, and the
    generator code. Reports raw rejection rates at 5% and 10% for both tests with
    Monte Carlo standard errors, plus size-adjusted rejection rates against
    empirical null cutoffs taken from the saved outputs/size_full.npz matched on
    t_oos. The size adjustment is needed because the normal test over-rejects
    under the null, so its raw power is inflated and only the size-adjusted
    figures compare the two tests like for like.
    """
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Cell order fixes the seed slices, so it must not be reordered once a run has
    # started: beta varies slowest, t_oos fastest.
    cells = [(beta, t_oos) for beta in config.BETA_GRID for t_oos in config.T_OOS_GRID]
    children = config.POWER_SEED_SEQUENCE.spawn(len(cells) * config.R_POWER)

    print(f"running full power runs: alt world, betas={config.BETA_GRID}, "
          f"T_OOS={config.T_OOS_GRID}, R={config.R_POWER}, B={config.B_BOOT_POWER}, "
          f"ARMA(1,1) model bootstrap")

    cell_results = []
    total_elapsed = 0.0
    for k, (beta, t_oos) in enumerate(cells):
        cell_children = children[k * config.R_POWER:(k + 1) * config.R_POWER]
        partial_path = os.path.join(
            out_dir, POWER_PARTIAL_TEMPLATE.format(beta_tag=_beta_tag(beta), t_oos=t_oos))
        print(f"cell {k + 1}/{len(cells)}: alt world, beta={beta}, t_oos={t_oos}")
        arrays, elapsed = _run_power_cell(beta, t_oos, cell_children, partial_path)
        total_elapsed += elapsed
        cutoffs = _null_cutoffs(t_oos)
        cell_results.append((beta, t_oos, arrays, cutoffs))
        _print_power_cell(beta, t_oos, arrays, cutoffs, elapsed)
        _save_power_results(out_dir, cell_results)

    _print_power_table(cell_results)
    print(f"total runtime this session = {total_elapsed:.1f}s across {len(cells)} cells")


# ---------------------------------------------------------------------------
# Beta calibration for the power runs
# ---------------------------------------------------------------------------

def _calibration_cell(beta, t_oos, children):
    """Run the alt-world replications for one candidate beta, no bootstrap.

    beta: loading on the lagged predictor in the alt world.
    t_oos: out-of-sample length.
    children: sequence of SeedSequence objects, one per replication. The same
        sequence is passed for every beta, which gives common random numbers:
        beta does not change how many draws generate_history takes, so
        replication i faces the identical predictor, log-variance shock and
        measurement-noise paths at every beta and only the loading differs. The
        beta-to-beta comparison is then free of sampling noise.

    Per replication this runs walk_forward and QLIKE only, no bootstrap, so a
    replication costs a fraction of a second.

    Returns a dict of per-replication arrays: qlike_small, qlike_big (means over
    the out-of-sample window), d_raw (mean unadjusted differential, equal to
    qlike_small minus qlike_big) and pct_red (percentage QLIKE reduction).
    """
    t_total = config.T_TRAIN + t_oos
    r_reps = len(children)

    qlike_small = np.empty(r_reps)
    qlike_big = np.empty(r_reps)

    for i in range(r_reps):
        rng_hist = np.random.default_rng(children[i])
        hist = dgp.generate_history("alt", beta, t_total, rng_hist)
        wf = forecast_eval.walk_forward(hist["rv"], hist["x"], config.T_TRAIN)
        qlike_small[i] = wf["qlike_small"].mean()
        qlike_big[i] = wf["qlike_big"].mean()

    d_raw = qlike_small - qlike_big
    pct_red = 100.0 * d_raw / qlike_small
    return {"qlike_small": qlike_small, "qlike_big": qlike_big,
            "d_raw": d_raw, "pct_red": pct_red}


def _print_calibration_table(betas, cells, r_reps, t_oos):
    """Print the calibration table, one row per candidate beta.

    Columns are the mean percentage QLIKE reduction, its 10th, 50th and 90th
    percentiles across replications, and the fraction of replications in which
    the big model beats the small one on raw QLIKE.
    """
    print(f"Beta calibration, alt world, t_oos={t_oos}, R={r_reps}, no bootstrap")
    print("-" * 76)
    print(f"{'beta':>6}  {'mean %red':>10}  {'p10 %red':>10}  {'p50 %red':>10}  "
          f"{'p90 %red':>10}  {'big wins':>9}")
    for beta, cell in zip(betas, cells):
        pct = cell["pct_red"]
        p10, p50, p90 = np.percentile(pct, [10, 50, 90])
        win = np.mean(cell["d_raw"] > 0.0)
        print(f"{beta:>6.2f}  {pct.mean():>10.3f}  {p10:>10.3f}  {p50:>10.3f}  "
              f"{p90:>10.3f}  {win:>9.3f}")


def _print_calibration_levels(betas, cells):
    """Print the QLIKE levels and the mean raw differential per candidate beta.

    The raw differential is on the same scale as the null-world mean d_raw, so
    the size of the true effect can be read against the estimation-noise penalty
    the null run measured.
    """
    print("QLIKE levels and raw differential per beta")
    print("-" * 76)
    print(f"{'beta':>6}  {'mean QLIKE small':>17}  {'mean QLIKE big':>15}  "
          f"{'mean d_raw':>12}")
    for beta, cell in zip(betas, cells):
        print(f"{beta:>6.2f}  {cell['qlike_small'].mean():>17.4f}  "
              f"{cell['qlike_big'].mean():>15.4f}  {cell['d_raw'].mean():>12.4f}")


def run_beta_calibration():
    """Scan the candidate betas for the population QLIKE gain in the alt world.

    For each beta in config.CAL_BETA_CANDIDATES, runs config.R_CAL alt-world
    replications at config.CAL_T_OOS and records the mean QLIKE of both models,
    the mean unadjusted differential and the percentage QLIKE reduction. No
    bootstrap is involved anywhere: this fixes the two BETA_GRID strengths for
    the power runs, and only needs the population effect size.

    Seeds come from config.CAL_SEED_SEQUENCE, a root child disjoint from the
    size and power sequences. The same R_CAL children serve every beta, so the
    betas are compared on common random numbers.
    """
    betas = config.CAL_BETA_CANDIDATES
    t_oos = config.CAL_T_OOS
    r_reps = config.R_CAL

    children = config.CAL_SEED_SEQUENCE.spawn(r_reps)

    print(f"running beta calibration: alt world, betas={betas}, "
          f"t_oos={t_oos}, R={r_reps}")

    cells = []
    start = time.perf_counter()
    for beta in betas:
        cell_start = time.perf_counter()
        cells.append(_calibration_cell(beta, t_oos, children))
        elapsed = time.perf_counter() - cell_start
        print(f"  beta={beta}: {elapsed:.1f}s for {r_reps} replications "
              f"({elapsed / r_reps:.3f}s per replication)")

    _print_calibration_table(betas, cells, r_reps, t_oos)
    print()
    _print_calibration_levels(betas, cells)
    print(f"total calibration runtime = {time.perf_counter() - start:.1f}s")


# ---------------------------------------------------------------------------
# Sensitivity runs (null world, one horizon, three axes)
# ---------------------------------------------------------------------------

# Run-orchestration settings for the sensitivity runs. The study parameters
# (R_SENS, SENS_T_OOS, the generator, draw and residual-block grids) come from
# config; only the cadence and file naming live here, matching the size and
# power runs.
SENS_CHECKPOINT_EVERY = 50    # checkpoint cadence, one write every this many replications per cell
SENS_PARTIAL_TEMPLATE = "sensitivity_{key}_partial.npz"   # per-cell checkpoint filename under outputs/, one file per cell so a completed cell is never re-run
SENS_RESULTS_NAME = "sensitivity.npz"   # final combined results file under outputs/
SENS_FIELDS = ["s_obs", "p_norm", "p_model", "draw_std", "gen_code"]   # fields recorded per replication


# Display names for the generator specifications, used in the table's setting
# column. The keys are the generator_spec strings forecast_eval.model_bootstrap_pvalue
# takes.
SENS_GEN_LABEL = {"arma": "ARMA(1,1)", "ar1": "AR(1)"}


def _sens_key(generator, b_boot, resid_block):
    """Stable key for one sensitivity cell, used for its checkpoint and npz prefix."""
    return f"{generator}_b{b_boot}_resid{resid_block}"


def _sens_cells_and_rows():
    """Return (cells, rows) for the sensitivity run.

    cells is the list of distinct computations, in the fixed order that fixes
    their seed slices; it must not be reordered once a run has started. rows is
    the reported table layout, a list of (axis, setting label, cell key).

    Three axes are varied one at a time around the validated construction
    (ARMA(1,1) generator, the baseline number of draws, iid residual resampling).
    That validated setting is one computation and appears as the reference row on
    all three axes, so six reported rows come from four cells rather than six.
    """
    gen_valid, gen_mis = config.SENS_GENERATOR_GRID
    b_base, b_large = config.SENS_B_GRID
    iid, blk = config.SENS_RESID_IID, config.SENS_RESID_BLOCK

    base = {"generator": gen_valid, "b_boot": b_base, "resid_block": iid}
    mis_gen = {"generator": gen_mis, "b_boot": b_base, "resid_block": iid}
    large_b = {"generator": gen_valid, "b_boot": b_large, "resid_block": iid}
    blocked = {"generator": gen_valid, "b_boot": b_base, "resid_block": blk}

    cells = []
    for spec in [base, mis_gen, large_b, blocked]:
        spec = dict(spec)
        spec["key"] = _sens_key(spec["generator"], spec["b_boot"], spec["resid_block"])
        spec["r_reps"] = config.R_SENS
        cells.append(spec)

    k_base, k_mis, k_large, k_blk = [c["key"] for c in cells]
    rows = [
        ("generator", f"{SENS_GEN_LABEL[gen_valid]}, validated", k_base),
        ("generator", f"{SENS_GEN_LABEL[gen_mis]}, mis-specified", k_mis),
        ("bootstrap draws", f"B={b_base}", k_base),
        ("bootstrap draws", f"B={b_large}", k_large),
        ("generator residuals", "iid", k_base),
        ("generator residuals", f"blocks of {blk}", k_blk),
    ]
    return cells, rows


def _save_sens_checkpoint(path, n_done, r_reps, arrays):
    """Write one sensitivity cell's arrays, its done-counter and its R atomically.

    r_reps is stored so a resumed session runs the cell at the same length the
    session that started it chose, even if the runtime budget would now pick a
    different one. Only entries [0:n_done] are meaningful.
    """
    tmp = path + ".tmp.npz"
    np.savez(tmp, n_done=n_done, r_reps=r_reps, **{k: arrays[k] for k in SENS_FIELDS})
    os.replace(tmp, path)


def _load_sens_checkpoint(path):
    """Return (n_done, r_reps, arrays) from an existing checkpoint, or (0, None, None)."""
    if not os.path.exists(path):
        return 0, None, None
    with np.load(path) as data:
        n_done = int(data["n_done"])
        r_reps = int(data["r_reps"])
        arrays = {k: data[k] for k in SENS_FIELDS}
    return n_done, r_reps, arrays


def _sens_replication(cell, child, t_total):
    """Run one sensitivity replication and return its recorded fields.

    Null world throughout: the sensitivity axes are about the bootstrap's own
    construction, so every cell is measuring size. Two disjoint streams per
    replication, history and bootstrap, as in every other run here.
    """
    hist_seq, boot_seq = child.spawn(2)
    rng_hist = np.random.default_rng(hist_seq)
    rng_boot = np.random.default_rng(boot_seq)

    hist = dgp.generate_history("null", 0.0, t_total, rng_hist)

    wf = forecast_eval.walk_forward(hist["rv"], hist["x"], config.T_TRAIN)
    d_adj = forecast_eval.cw_adjusted_diff(wf["qlike_small"], wf["qlike_big"],
                                   wf["f_small"], wf["f_big"], hist["rv"])
    s_obs = forecast_eval.cw_stat(d_adj)

    p_model, draw_std, diag = forecast_eval.model_bootstrap_pvalue(
        hist["rv"], hist["x"], config.T_TRAIN, cell["b_boot"], rng_boot,
        generator_spec=cell["generator"], resid_block_len=cell["resid_block"])

    return {"s_obs": s_obs, "p_norm": forecast_eval.normal_pvalue(s_obs), "p_model": p_model,
            "draw_std": draw_std, "gen_code": _gen_code(diag)}


def _sens_partial_path(out_dir, cell):
    """Checkpoint path for one sensitivity cell."""
    return os.path.join(out_dir, SENS_PARTIAL_TEMPLATE.format(key=cell["key"]))


def _sens_adopt_checkpoint_lengths(cells, out_dir):
    """Set each cell's r_reps from its existing checkpoint, if one is there.

    A cell already part-run keeps the length its first session chose, so the
    runtime budget below cannot change a cell's R underneath a checkpoint.
    """
    for cell in cells:
        _, stored_r, _ = _load_sens_checkpoint(_sens_partial_path(out_dir, cell))
        if stored_r is not None and stored_r != cell["r_reps"]:
            print(f"  cell {cell['key']}: adopting R={stored_r} from its existing checkpoint "
                  f"(config R_SENS={cell['r_reps']})")
            cell["r_reps"] = stored_r


def _sens_probe(cells, probe_children):
    """Time config.SENS_PROBE_REPS replications of each cell, returning seconds per replication.

    probe_children is a per-cell list of SeedSequence objects taken from the tail
    of the sensitivity spawn, disjoint from the children the recorded replications
    use, so timing the probe consumes none of the run's own randomness. The probe
    exists so the projected runtime below is measured on this machine rather than
    assumed.
    """
    t_total = config.T_TRAIN + config.SENS_T_OOS
    per_rep = []
    for cell, children in zip(cells, probe_children):
        start = time.perf_counter()
        for child in children:
            _sens_replication(cell, child, t_total)
        per_rep.append((time.perf_counter() - start) / len(children))
    return per_rep


def _sens_project(cells, per_rep):
    """Projected hours for the whole run and per cell, from the probe's per-replication times."""
    hours = [c["r_reps"] * s / 3600.0 for c, s in zip(cells, per_rep)]
    return sum(hours), hours


def _sens_apply_budget(cells, per_rep, out_dir):
    """Print the projected runtime and cut the largest-B cell's R if it is over budget.

    The rule is fixed in advance: if the projection exceeds
    config.SENS_RUNTIME_BUDGET_H, the cell with the most bootstrap draws (the most
    expensive one, and the one whose row is a resolution check rather than a
    headline size number) drops to config.R_SENS_REDUCED and its row is flagged as
    reduced precision. Cells already holding a checkpoint are left alone so a
    resumed run stays consistent. Returns the set of keys whose R was cut.
    """
    total_h, per_cell_h = _sens_project(cells, per_rep)
    print("Projected runtime from the probe")
    print("-" * 74)
    for cell, s, h in zip(cells, per_rep, per_cell_h):
        print(f"  {cell['key']:<22} R={cell['r_reps']:<5} {s:>6.2f}s per rep  "
              f"{h:>6.2f} h")
    print(f"  projected total = {total_h:.2f} h "
          f"(budget {config.SENS_RUNTIME_BUDGET_H:.1f} h)")

    reduced = set()
    if total_h <= config.SENS_RUNTIME_BUDGET_H:
        return reduced

    target = max(cells, key=lambda c: c["b_boot"])
    has_checkpoint = os.path.exists(_sens_partial_path(out_dir, target))
    if has_checkpoint:
        print(f"  over budget, but cell {target['key']} already has a checkpoint, "
              f"so its R is left at {target['r_reps']}")
        return reduced

    target["r_reps"] = config.R_SENS_REDUCED
    reduced.add(target["key"])
    total_h, _ = _sens_project(cells, per_rep)
    print(f"  over budget, so cell {target['key']} drops to R={config.R_SENS_REDUCED}, "
          f"revised total = {total_h:.2f} h")
    return reduced


def _run_sens_cell(cell, cell_children, partial_path):
    """Run one sensitivity cell with checkpoint resume.

    cell_children is the length-r_reps slice of spawned SeedSequences for this
    cell; child i is fixed by position, so a resumed run reuses the same seeds and
    replication i is identical whether the run is fresh or resumed. Returns
    (arrays, elapsed) with elapsed the time spent computing replications in this
    invocation.
    """
    t_total = config.T_TRAIN + config.SENS_T_OOS
    r = cell["r_reps"]

    arrays = {k: np.full(r, np.nan) for k in SENS_FIELDS if k != "gen_code"}
    arrays["gen_code"] = np.full(r, -1, dtype=int)   # -1 marks a not-yet-computed replication

    n_done, _, saved = _load_sens_checkpoint(partial_path)
    if saved is not None:
        for k, dst in arrays.items():
            dst[:n_done] = saved[k][:n_done]
        print(f"  resuming cell {cell['key']} from checkpoint at replication {n_done}/{r}")

    # Timing covers replications computed in this invocation only, so the estimate
    # is meaningful whether the cell is fresh or resumed.
    session_start = time.perf_counter()
    session_count = 0
    for i in range(n_done, r):
        rec = _sens_replication(cell, cell_children[i], t_total)
        for k, v in rec.items():
            arrays[k][i] = v
        session_count += 1

        if session_count == 10:
            per = (time.perf_counter() - session_start) / 10
            print(f"  {cell['key']}: {per:.3f}s per replication, projected "
                  f"{per * r / 60:.1f} min ({per * r / 3600:.2f} h) for {r} replications")
        if (i + 1) % SENS_CHECKPOINT_EVERY == 0:
            _save_sens_checkpoint(partial_path, i + 1, r, arrays)
            print(f"  {cell['key']}: checkpoint saved at replication {i + 1}/{r}")

    # Final checkpoint so the full cell is on disk even if r is not a multiple of
    # the checkpoint interval.
    _save_sens_checkpoint(partial_path, r, r, arrays)
    return arrays, time.perf_counter() - session_start


def _print_sens_cell(cell, arrays, elapsed):
    """Print one sensitivity cell's summary as soon as it finishes."""
    r = cell["r_reps"]
    i5 = config.ALPHA_LEVELS.index(0.05)
    i10 = config.ALPHA_LEVELS.index(0.10)
    rej_n, se_n = _size_rej_rates(arrays["p_norm"], r)
    rej_m, se_m = _size_rej_rates(arrays["p_model"], r)
    counts = {name: int(np.sum(arrays["gen_code"] == code))
              for name, code in _GEN_CODE.items()}
    print(f"cell {cell['key']} summary (generator={cell['generator']}, "
          f"B={cell['b_boot']}, residual block={cell['resid_block']}, R={r})")
    print("-" * 74)
    print(f"  CW normal     rej@5%={rej_n[i5]:.3f} ({se_n[i5]:.3f})  "
          f"rej@10%={rej_n[i10]:.3f} ({se_n[i10]:.3f})")
    print(f"  CW model-boot rej@5%={rej_m[i5]:.3f} ({se_m[i5]:.3f})  "
          f"rej@10%={rej_m[i10]:.3f} ({se_m[i10]:.3f})")
    print(f"  s_obs std = {arrays['s_obs'].std():.4f}, "
          f"mean draw-statistic std = {arrays['draw_std'].mean():.4f}")
    print(f"  generator usage: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"  runtime this cell (this session) = {elapsed:.1f}s")


def _print_sens_table(cells, rows, cell_arrays, reduced):
    """Print the sensitivity table, one row per axis setting.

    Columns are the axis, the setting, the cell's R, the model-bootstrap rejection
    rate at 5% and at 10%, the Monte Carlo standard errors at those two levels, and
    the mean draw-statistic std. Rows whose cell was cut to a shorter R by the
    runtime budget are marked, since their standard errors are the wider ones.
    """
    by_key = {c["key"]: c for c in cells}
    i5 = config.ALPHA_LEVELS.index(0.05)
    i10 = config.ALPHA_LEVELS.index(0.10)

    print(f"Sensitivity table, null world, t_oos={config.SENS_T_OOS}, "
          f"model-based bootstrap")
    print("-" * 96)
    print(f"{'axis':<20}  {'setting':<24}  {'R':>5}  {'rej@5%':>8}  {'rej@10%':>8}  "
          f"{'MC se 5/10%':>13}  {'mean draw std':>13}")
    for axis, setting, key in rows:
        cell = by_key[key]
        arrays = cell_arrays[key]
        rej, se = _size_rej_rates(arrays["p_model"], cell["r_reps"])
        label = setting + (" (reduced R)" if key in reduced else "")
        print(f"{axis:<20}  {label:<24}  {cell['r_reps']:>5}  {rej[i5]:>8.3f}  "
              f"{rej[i10]:>8.3f}  {se[i5]:>6.3f}/{se[i10]:<6.3f}  "
              f"{arrays['draw_std'].mean():>13.4f}")
    base = by_key[rows[0][2]]
    print(f"The validated setting ({SENS_GEN_LABEL[base['generator']]}, B={base['b_boot']}, "
          f"iid residuals) is one cell and appears as the reference row on all three axes.")


def _save_sens_results(out_dir, cells, rows, cell_arrays):
    """Save the sensitivity arrays, cell settings and row layout to outputs/sensitivity.npz.

    Per-cell arrays are stored under keys "{cell key}__{field}", the cell settings
    under "{cell key}__{setting}", and the reported row layout as three parallel
    string arrays, so the comparison table can be rebuilt from this file alone.
    """
    path = os.path.join(out_dir, SENS_RESULTS_NAME)
    payload = {
        "cell_keys": np.array([c["key"] for c in cells]),
        "row_axis": np.array([r[0] for r in rows]),
        "row_setting": np.array([r[1] for r in rows]),
        "row_key": np.array([r[2] for r in rows]),
        "t_oos": config.SENS_T_OOS,
        "alpha_levels": np.array(config.ALPHA_LEVELS),
    }
    for cell in cells:
        key = cell["key"]
        for f in SENS_FIELDS:
            payload[f"{key}__{f}"] = cell_arrays[key][f]
        for s in ["generator", "b_boot", "resid_block", "r_reps"]:
            payload[f"{key}__{s}"] = cell[s]
    np.savez(path, **payload)
    print(f"saved sensitivity results ({len(cells)} cells) to {path}")


def run_sensitivity():
    """Sensitivity runs for the model-based bootstrap, null world, one horizon.

    Null world at config.SENS_T_OOS only, config.R_SENS replications per cell.
    Three axes are varied one at a time around the validated construction:

      a. Generator specification, the validated ARMA(1,1) against a forced AR(1).
         The AR(1) case is the mis-specified one measured earlier at reduced
         scale; it is rerun here at the full R so its row carries the same Monte
         Carlo precision as the rest of the table.
      b. Bootstrap draws, the two values in config.SENS_B_GRID at the validated
         ARMA specification, which shows whether the size result depends on the
         bootstrap's resolution.
      c. Residual resampling inside the generator, iid against short blocks of
         config.SENS_RESID_BLOCK. The ARMA residuals should already be near-white,
         so this demonstrates that rather than asserting it.

    Seeds come from config.SENS_SEED_SEQUENCE, a root child disjoint from the
    size, power and calibration sequences. Children are spawned once and sliced
    per cell, so every cell has its own disjoint seed range and child i is fixed
    by position for a seed-exact resume. The tail of the spawn serves the timing
    probe, so probing consumes none of the recorded replications' randomness.

    The run first probes config.SENS_PROBE_REPS replications per cell, prints the
    projected total runtime, and if that exceeds config.SENS_RUNTIME_BUDGET_H cuts
    the largest-B cell to config.R_SENS_REDUCED and marks its row as reduced
    precision. Cells run sequentially, each checkpointing, printing its own
    summary and rewriting the results file before the next starts.
    """
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)

    cells, rows = _sens_cells_and_rows()
    print(f"running sensitivity: null world, t_oos={config.SENS_T_OOS}, "
          f"{len(cells)} cells reported as {len(rows)} rows")
    _sens_adopt_checkpoint_lengths(cells, out_dir)

    # One spawn covers every cell at the full R plus the probe tail, so the slices
    # stay fixed even if the budget rule shortens a cell: a shortened cell simply
    # uses fewer of its own children and no other cell's range moves.
    n_run = len(cells) * config.R_SENS
    children = config.SENS_SEED_SEQUENCE.spawn(n_run + len(cells) * config.SENS_PROBE_REPS)
    probe_children = [children[n_run + k * config.SENS_PROBE_REPS:
                               n_run + (k + 1) * config.SENS_PROBE_REPS]
                      for k in range(len(cells))]

    print(f"timing probe: {config.SENS_PROBE_REPS} replications per cell")
    per_rep = _sens_probe(cells, probe_children)
    reduced = _sens_apply_budget(cells, per_rep, out_dir)

    cell_arrays = {}
    total_elapsed = 0.0
    for k, cell in enumerate(cells):
        cell_children = children[k * config.R_SENS:(k + 1) * config.R_SENS]
        print(f"cell {k + 1}/{len(cells)}: {cell['key']}, R={cell['r_reps']}")
        arrays, elapsed = _run_sens_cell(cell, cell_children,
                                         _sens_partial_path(out_dir, cell))
        total_elapsed += elapsed
        cell_arrays[cell["key"]] = arrays
        _print_sens_cell(cell, arrays, elapsed)
        _save_sens_results(out_dir, cells[:k + 1], rows, cell_arrays)

    _print_sens_table(cells, rows, cell_arrays, reduced)
    print(f"total runtime this session = {total_elapsed:.1f}s across {len(cells)} cells")


def main():
    """Drive the size runs: null world, baseline block, both T_OOS, R_SIZE reps."""
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "size_results.npz")

    # Size runs draw from the size seed sequence in config, disjoint from the
    # power runs. Spawn all children upfront and give each cell a disjoint slice.
    num_cells = len(config.T_OOS_GRID)
    children = config.SIZE_SEED_SEQUENCE.spawn(num_cells * config.R_SIZE)

    results = []
    for k, t_oos in enumerate(config.T_OOS_GRID):
        master_children = children[k * config.R_SIZE:(k + 1) * config.R_SIZE]
        print(f"running size cell: null world, t_oos={t_oos}, block={config.BLOCK_BASELINE}")
        res = run_cell("null", 0.0, t_oos, config.BLOCK_BASELINE,
                       config.R_SIZE, master_children)
        results.append(res)

    save_results(results, out_path)
    print(f"saved size results to {out_path}")
    print_size_table(results)
    print()
    print_stat_diagnostics(results)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "validation":
        run_validation()
    elif len(sys.argv) > 1 and sys.argv[1] == "validation_model":
        run_validation_model()
    elif len(sys.argv) > 1 and sys.argv[1] == "size_full":
        run_size_full()
    elif len(sys.argv) > 1 and sys.argv[1] == "beta_calibration":
        run_beta_calibration()
    elif len(sys.argv) > 1 and sys.argv[1] == "power":
        run_power()
    elif len(sys.argv) > 1 and sys.argv[1] == "sensitivity":
        run_sensitivity()
    else:
        main()