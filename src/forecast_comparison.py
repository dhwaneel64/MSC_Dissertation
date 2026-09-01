from dataclasses import dataclass
import numpy as np
from scipy import stats
from src import config


@dataclass
class DMResult:
    test_type: str       # always "diebold_mariano"
    statistic: float     # HLN-corrected by default
    p_value: float       # two-sided
    n: int
    horizon: int
    hac_lag: int
    hln_correction: bool
    favoured: str        # "model_a", "model_b", or "tie"
    alpha: float


@dataclass
class CWResult:
    test_type: str           # always "clark_west"
    statistic: float
    p_value: float           # one-sided: tests whether larger model has lower adjusted loss
    n: int
    hac_lag: int
    larger_model_wins: bool
    alpha: str | float       # significance level used
    loss: str = "mspe"       # loss the CW adjustment operated on: "mspe" or "qlike"


@dataclass
class BootstrapCWResult:
    test_type: str               # always "clark_west_bootstrap"
    loss: str                    # loss the CW adjustment operated on, e.g. "qlike"
    statistic: float             # observed studentised CW statistic (point statistic, unchanged)
    observed_mean: float         # mean of the adjusted differential f_t (quantity tested)
    p_value: float               # block-bootstrap one-sided p-value under H0
    n: int
    block_length: int
    n_boot: int
    seed: int
    larger_model_wins: bool
    alpha: float


def _hac_variance(x: np.ndarray, hac_lag: int) -> float:
    """Biased HAC variance estimate: gamma_0 + 2 * sum_{k=1..hac_lag} gamma_k."""
    N = len(x)
    x_c = x - x.mean()
    gamma = np.empty(hac_lag + 1)
    for k in range(hac_lag + 1):
        gamma[k] = np.dot(x_c[: N - k], x_c[k:]) / N
    return float(gamma[0] + 2.0 * np.sum(gamma[1:]))


def diebold_mariano(
    losses_a,
    losses_b,
    horizon: int = config.DM_DEFAULT_HORIZON,
    hac_lag: int | None = None,
    hln_correction: bool = True,
    alpha: float = config.COMPARISON_ALPHA,
) -> DMResult:
    """Diebold-Mariano test for non-nested model comparison.

    Tests H0: E[loss_a - loss_b] = 0 against two-sided alternative.

    For nested models (one is a strict special case of the other), use
    clark_west() instead. Standard DM has degenerate asymptotic distribution
    under nesting and under-rejects.

    Args:
        losses_a, losses_b: per-observation loss series, equal length, same metric.
        horizon: forecast horizon. Used for HLN correction. Default 1.
        hac_lag: HAC lag truncation. If None, defaults to horizon - 1.
        hln_correction: Harvey-Leybourne-Newbold finite-sample correction. True by default.
        alpha: significance threshold for the "favoured" field.

    Implementation:
        d_t = losses_a[t] - losses_b[t]
        d_bar = mean(d)
        HAC variance: gamma_0 + 2 * sum_{k=1..hac_lag} gamma_k
        Standard DM stat: d_bar / sqrt(HAC_var / N)
        HLN correction factor: sqrt((N + 1 - 2h + h(h-1)/N) / N)
            Multiply standard DM stat by this factor.
        p-value: t(N-1) two-sided under HLN, standard normal otherwise.

    Raises:
        ValueError on length mismatch, NaN, N < 8, or zero-variance differential.
    """
    losses_a = np.asarray(losses_a, dtype=float)
    losses_b = np.asarray(losses_b, dtype=float)

    if losses_a.shape != losses_b.shape:
        raise ValueError(
            f"Length mismatch: losses_a has shape {losses_a.shape}, "
            f"losses_b has shape {losses_b.shape}"
        )
    if np.any(np.isnan(losses_a)):
        raise ValueError("losses_a contains NaN")
    if np.any(np.isnan(losses_b)):
        raise ValueError("losses_b contains NaN")

    N = len(losses_a)
    if N < config.MIN_COMPARISON_OBS:
        raise ValueError(
            f"N={N} is too small; HLN correction requires N >= {config.MIN_COMPARISON_OBS}"
        )

    if hac_lag is None:
        hac_lag = horizon - 1

    d = losses_a - losses_b
    d_bar = float(d.mean())

    hac_var = _hac_variance(d, hac_lag)
    if hac_var <= 0:
        # d is constant. If d_bar == 0 (identical losses), return a tie.
        # If d_bar != 0 (non-zero constant differential), the test stat is +/-inf, undefined.
        if d_bar == 0.0:
            return DMResult(
                test_type="diebold_mariano",
                statistic=0.0,
                p_value=1.0,
                n=N,
                horizon=horizon,
                hac_lag=hac_lag,
                hln_correction=hln_correction,
                favoured="tie",
                alpha=alpha,
            )
        raise ValueError(
            "HAC variance of the loss differential is zero or negative; "
            "DM test is undefined (degenerate loss differential)"
        )

    dm_stat = d_bar / np.sqrt(hac_var / N)

    if hln_correction:
        factor = np.sqrt((N + 1 - 2 * horizon + horizon * (horizon - 1) / N) / N)
        final_stat = float(dm_stat * factor)
        p_value = float(2.0 * stats.t.sf(abs(final_stat), df=N - 1))
    else:
        final_stat = float(dm_stat)
        p_value = float(2.0 * stats.norm.sf(abs(final_stat)))

    if p_value < alpha:
        favoured = "model_a" if final_stat < 0 else "model_b"
    else:
        favoured = "tie"

    return DMResult(
        test_type="diebold_mariano",
        statistic=final_stat,
        p_value=p_value,
        n=N,
        horizon=horizon,
        hac_lag=hac_lag,
        hln_correction=hln_correction,
        favoured=favoured,
        alpha=alpha,
    )


def clark_west(
    y_true,
    y_pred_smaller,
    y_pred_larger,
    hac_lag: int = 0,
    alpha: float = config.COMPARISON_ALPHA,
    loss_type: str = "mspe",
) -> CWResult:
    """Clark-West (2007) MSPE-adjusted test for nested model comparison.

    Tests H0: smaller model has equal or lower MSPE than larger model
    against H1: larger model has lower MSPE. One-sided.

    The N(0,1) reference p-value this function returns is valid for squared-error
    loss only. loss_type is an explicit declaration of the loss the caller wants
    the p-value for: anything other than "mspe" raises, pointing to
    clark_west_bootstrap, whose block-bootstrap p-value is the locked method for
    QLIKE differentials. The function stays importable for MSPE use.

    Use when the smaller model is a strict special case of the larger model
    (e.g. constant baseline subset HAR subset extended OLS). Standard DM is
    invalid in this case because the loss differential has zero variance under
    the null.

    Reference: Clark and West (2007), "Approximately Normal Tests for Equal
    Predictive Accuracy in Nested Models", Journal of Econometrics.

    Implementation:
        e_smaller_t = y_true[t] - y_pred_smaller[t]
        e_larger_t  = y_true[t] - y_pred_larger[t]
        f_t = e_smaller_t^2 - e_larger_t^2 + (y_pred_smaller[t] - y_pred_larger[t])^2
        f_bar = mean(f)
        HAC variance of f.
        CW stat = f_bar / sqrt(HAC_var / N)
        p-value: one-sided standard normal upper tail (P(Z > stat)).

    Args:
        y_true: realised target series.
        y_pred_smaller: predictions from the smaller (nested) model.
        y_pred_larger: predictions from the larger (nesting) model.
        hac_lag: HAC lag truncation. Default 0 for one-step-ahead forecasts.
        alpha: significance threshold for larger_model_wins.

    Returns:
        CWResult. larger_model_wins is True if p_value < alpha.

    Raises:
        ValueError on length mismatch, NaN, N < config.MIN_COMPARISON_OBS,
        zero-variance f series, or loss_type != "mspe".
    """
    if loss_type != "mspe":
        raise ValueError(
            f"clark_west's N(0,1) p-value is squared-error-specific; got "
            f"loss_type={loss_type!r}. Use clark_west_bootstrap for QLIKE "
            f"differentials (block-bootstrap p-value under H0)."
        )

    y_true = np.asarray(y_true, dtype=float)
    y_pred_smaller = np.asarray(y_pred_smaller, dtype=float)
    y_pred_larger = np.asarray(y_pred_larger, dtype=float)

    if not (y_true.shape == y_pred_smaller.shape == y_pred_larger.shape):
        raise ValueError(
            "y_true, y_pred_smaller, and y_pred_larger must all have the same length"
        )
    if np.any(np.isnan(y_true)):
        raise ValueError("y_true contains NaN")
    if np.any(np.isnan(y_pred_smaller)):
        raise ValueError("y_pred_smaller contains NaN")
    if np.any(np.isnan(y_pred_larger)):
        raise ValueError("y_pred_larger contains NaN")

    N = len(y_true)
    if N < config.MIN_COMPARISON_OBS:
        raise ValueError(
            f"N={N} is too small; Clark-West test requires N >= {config.MIN_COMPARISON_OBS}"
        )

    # Standard CW operates on squared-error (MSPE) loss. Build the three
    # per-observation series and hand them to the loss-agnostic CW core. The
    # adjustment (y_pred_smaller - y_pred_larger)**2 is the squared-error loss of
    # the larger model's forecast measured against the smaller (nested) model's
    # forecast as proxy-truth, the canonical Clark-West adjustment.
    e_s = y_true - y_pred_smaller
    e_l = y_true - y_pred_larger
    loss_smaller = e_s ** 2
    loss_larger = e_l ** 2
    adjustment = (y_pred_smaller - y_pred_larger) ** 2

    return clark_west_from_losses(
        loss_smaller,
        loss_larger,
        adjustment,
        hac_lag=hac_lag,
        alpha=alpha,
        loss="mspe",
    )


def clark_west_from_losses(
    loss_smaller,
    loss_larger,
    adjustment,
    hac_lag: int = 0,
    alpha: float = config.COMPARISON_ALPHA,
    loss: str = "mspe",
) -> CWResult:
    
    loss_smaller = np.asarray(loss_smaller, dtype=float)
    loss_larger = np.asarray(loss_larger, dtype=float)
    adjustment = np.asarray(adjustment, dtype=float)

    if not (loss_smaller.shape == loss_larger.shape == adjustment.shape):
        raise ValueError(
            "loss_smaller, loss_larger, and adjustment must all have the same length"
        )
    if np.any(np.isnan(loss_smaller)):
        raise ValueError("loss_smaller contains NaN")
    if np.any(np.isnan(loss_larger)):
        raise ValueError("loss_larger contains NaN")
    if np.any(np.isnan(adjustment)):
        raise ValueError("adjustment contains NaN")

    N = len(loss_smaller)
    if N < config.MIN_COMPARISON_OBS:
        raise ValueError(
            f"N={N} is too small; Clark-West test requires N >= {config.MIN_COMPARISON_OBS}"
        )

    f = loss_smaller - loss_larger + adjustment
    f_bar = float(f.mean())

    hac_var = _hac_variance(f, hac_lag)
    if hac_var <= 0:
        raise ValueError(
            "HAC variance of the f series is zero or negative; "
            "Clark-West test is undefined (degenerate f series)"
        )

    cw_stat = f_bar / np.sqrt(hac_var / N)
    p_value = float(stats.norm.sf(cw_stat))  # one-sided upper tail

    return CWResult(
        test_type="clark_west",
        statistic=float(cw_stat),
        p_value=p_value,
        n=N,
        hac_lag=hac_lag,
        larger_model_wins=p_value < alpha,
        alpha=alpha,
        loss=loss,
    )


def block_bootstrap_pvalue(
    f,
    block_length: int | None = None,
    n_boot: int = config.BOOTSTRAP_REPLICATIONS,
    seed: int = config.BOOTSTRAP_SEED,
) -> tuple[float, float, int]:
    """Moving-block bootstrap one-sided p-value for H0: E[f] <= 0 vs H1: E[f] > 0.

    Used to obtain the nested Clark-West p-value on QLIKE differentials, where the
    N(0,1) reference distribution is not valid (it is squared-error-specific; only
    the bias-correction centering of f carries over to QLIKE). f is the adjusted
    per-observation differential f_t = loss_smaller - loss_larger + adjustment, so
    larger f means the larger (nesting) model improves.

    The null is enforced by RE-CENTERING f to mean zero before resampling: the
    bootstrap distribution is built from g_t = f_t - mean(f_t), so it represents
    H0. The observed test quantity is the raw mean(f_t); the p-value is the share
    of re-centered block-bootstrap means at or beyond it, one-sided upper tail.
    This is a proper null bootstrap, not a resample of the raw statistic.

    Blocks are CIRCULAR (Politis and Romano 1992): block starts are drawn over all
    n positions and indices wrap modulo n. Circular blocks give every observation
    equal selection probability, so the expected bootstrap mean equals mean(g) = 0
    exactly and the null distribution is correctly centered (a non-circular moving
    block bootstrap under-weights the end observations and leaves the null mean
    off-zero, biasing the p-value).

    Block length defaults to floor(n ** BOOTSTRAP_BLOCK_LENGTH_EXPONENT), the
    standard n**(1/3) growth rate (Hall, Horowitz and Jing 1995); the block
    preserves any residual serial dependence in f_t.

    Args:
        f: adjusted per-observation differential series.
        block_length: moving-block length. If None, uses the n**(1/3) rule.
        n_boot: number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        (p_value, observed_mean, block_length). p_value uses the
        (1 + count) / (n_boot + 1) convention, so it is never exactly zero.

    Raises:
        ValueError on NaN, N < 8, or block_length outside [1, N].
    """
    f = np.asarray(f, dtype=float)
    if np.any(np.isnan(f)):
        raise ValueError("f contains NaN")

    n = len(f)
    if n < config.MIN_COMPARISON_OBS:
        raise ValueError(
            f"N={n} is too small; bootstrap requires N >= {config.MIN_COMPARISON_OBS}"
        )

    if block_length is None:
        block_length = max(1, int(n ** config.BOOTSTRAP_BLOCK_LENGTH_EXPONENT))
    if not (1 <= block_length <= n):
        raise ValueError(
            f"block_length={block_length} must be in [1, {n}]"
        )

    observed_mean = float(f.mean())
    g = f - observed_mean  # enforce H0: mean-zero null distribution

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))

    # Circular blocks: starts range over all n positions, indices wrap modulo n.
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block_length)
    idx = ((starts[:, :, None] + offsets[None, None, :]) % n).reshape(
        n_boot, n_blocks * block_length
    )[:, :n]
    boot_means = g[idx].mean(axis=1)

    count = int(np.sum(boot_means >= observed_mean))
    p_value = (1 + count) / (n_boot + 1)
    return p_value, observed_mean, int(block_length)


def clark_west_bootstrap(
    loss_smaller,
    loss_larger,
    adjustment,
    block_length: int | None = None,
    n_boot: int = config.BOOTSTRAP_REPLICATIONS,
    seed: int = config.BOOTSTRAP_SEED,
    alpha: float = config.COMPARISON_ALPHA,
    loss: str = "qlike",
    hac_lag: int = 0,
) -> BootstrapCWResult:
    """Nested Clark-West test with a block-bootstrap p-value under H0.

    The point statistic and its centering are unchanged: f_t and the studentised
    CW statistic are exactly those of clark_west_from_losses (which this function
    calls). Only the p-value source changes, from the N(0,1) table (not valid for
    QLIKE) to block_bootstrap_pvalue, which enforces H0 by re-centering f_t.

    This is the standard nested-significance test for the model sequence. For the
    QLIKE case the caller passes
        loss_smaller = qlike_per_obs(realised, var_smaller),
        loss_larger  = qlike_per_obs(realised, var_larger),
        adjustment   = qlike_per_obs(var_smaller, var_larger).

    Args:
        loss_smaller, loss_larger, adjustment: per-observation series, equal length.
        block_length, n_boot, seed: passed to block_bootstrap_pvalue.
        alpha: significance threshold for larger_model_wins.
        loss: label recorded on the result ("qlike" by default).
        hac_lag: HAC lag for the studentised point statistic. Default 0.

    Returns:
        BootstrapCWResult. larger_model_wins is True if the bootstrap p_value < alpha.
    """
    # Point statistic and centering: identical to clark_west_from_losses.
    cw = clark_west_from_losses(
        loss_smaller, loss_larger, adjustment, hac_lag=hac_lag, alpha=alpha, loss=loss
    )

    # Bootstrap p-value on the same adjusted differential f_t.
    loss_smaller = np.asarray(loss_smaller, dtype=float)
    loss_larger = np.asarray(loss_larger, dtype=float)
    adjustment = np.asarray(adjustment, dtype=float)
    f = loss_smaller - loss_larger + adjustment

    p_value, observed_mean, block_length_used = block_bootstrap_pvalue(
        f, block_length=block_length, n_boot=n_boot, seed=seed
    )

    return BootstrapCWResult(
        test_type="clark_west_bootstrap",
        loss=loss,
        statistic=cw.statistic,
        observed_mean=observed_mean,
        p_value=p_value,
        n=cw.n,
        block_length=block_length_used,
        n_boot=n_boot,
        seed=seed,
        larger_model_wins=p_value < alpha,
        alpha=alpha,
    )


# Each tuple is (smaller_model, larger_model) in canonical nesting order.
NESTED_MODEL_PAIRS: frozenset[tuple[str, str]] = frozenset({
    ("constant", "har"),
    ("constant", "extended_ols"),
    ("har", "extended_ols"),
    # Regime-switching OLS nests Extended OLS: Extended OLS is the equality
    # restriction where all regimes share one coefficient set (and one pooled
    # scaler), so regime-switching is the larger (nesting) model. Clark-West, not
    # Diebold-Mariano, is the valid test for this pair.
    ("extended_ols", "regime_switching"),
})


def is_nested(model_a: str, model_b: str) -> bool:
    """Returns True if (model_a, model_b) or (model_b, model_a) is in NESTED_MODEL_PAIRS."""
    return (model_a, model_b) in NESTED_MODEL_PAIRS or (model_b, model_a) in NESTED_MODEL_PAIRS


def _nested_order(model_a: str, model_b: str) -> tuple[str, str] | None:
    """Return (smaller, larger) if nested, else None."""
    if (model_a, model_b) in NESTED_MODEL_PAIRS:
        return (model_a, model_b)
    if (model_b, model_a) in NESTED_MODEL_PAIRS:
        return (model_b, model_a)
    return None


def compare_models(
    name_a: str,
    name_b: str,
    losses_a=None,
    losses_b=None,
    y_true=None,
    y_pred_a=None,
    y_pred_b=None,
    loss_type: str = "mspe",
    **kwargs,
) -> DMResult | CWResult:
    """Dispatcher: chooses DM (non-nested) or CW (nested) based on model names.

    For non-nested pairs: requires losses_a and losses_b. Calls diebold_mariano.
    For nested pairs: requires y_true, y_pred_a, y_pred_b. Calls clark_west,
        with the smaller-model predictions identified by NESTED_MODEL_PAIRS.

    loss_type declares the loss the p-value is wanted for. The nested route's
    N(0,1) reference is squared-error-specific, so a nested pair with
    loss_type != "mspe" raises and points to clark_west_bootstrap. The
    non-nested Diebold-Mariano route is loss-agnostic and ignores loss_type.

    Raises ValueError if required arguments for the chosen test are missing.

    Canonical names for this dissertation: "constant", "har", "extended_ols",
    "regime_switching", "xgboost".
    """
    order = _nested_order(name_a, name_b)

    if order is not None:
        if loss_type != "mspe":
            raise ValueError(
                f"nested pair ({name_a!r}, {name_b!r}) with loss_type={loss_type!r}: "
                "the dispatcher's Clark-West N(0,1) p-value is squared-error-specific. "
                "Use clark_west_bootstrap for QLIKE differentials."
            )
        if y_true is None or y_pred_a is None or y_pred_b is None:
            raise ValueError(
                f"({name_a!r}, {name_b!r}) is a nested pair; "
                "provide y_true, y_pred_a, and y_pred_b for Clark-West"
            )
        smaller, larger = order
        if smaller == name_a:
            pred_smaller, pred_larger = y_pred_a, y_pred_b
        else:
            pred_smaller, pred_larger = y_pred_b, y_pred_a
        return clark_west(y_true, pred_smaller, pred_larger, loss_type=loss_type, **kwargs)
    else:
        if losses_a is None or losses_b is None:
            raise ValueError(
                f"({name_a!r}, {name_b!r}) is not a nested pair; "
                "provide losses_a and losses_b for Diebold-Mariano"
            )
        return diebold_mariano(losses_a, losses_b, **kwargs)
