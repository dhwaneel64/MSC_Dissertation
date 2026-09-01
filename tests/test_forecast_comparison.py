import numpy as np
import pytest

from src import config
from src.forecast_comparison import (
    BootstrapCWResult,
    CWResult,
    DMResult,
    block_bootstrap_pvalue,
    clark_west,
    clark_west_bootstrap,
    clark_west_from_losses,
    compare_models,
    diebold_mariano,
    is_nested,
)
from src.metrics import qlike_per_obs


# ── helpers ───────────────────────────────────────────────────────────────────

def _losses(seed: int, n: int, scale: float = 1.0) -> np.ndarray:
    return np.abs(np.random.default_rng(seed).standard_normal(n)) * scale + 0.1


# ── DM tests ──────────────────────────────────────────────────────────────────

def test_dm_identical_losses_is_tie():
    losses = _losses(0, 50)
    result = diebold_mariano(losses, losses.copy())
    assert abs(result.statistic) < 1e-10
    assert result.p_value > 0.9
    assert result.favoured == "tie"


def test_dm_sign_correctness():
    # A ~ U(1.5, 2.5), B ~ U(0.5, 1.5): independent, E[A] > E[B], non-constant d
    rng = np.random.default_rng(1)
    N = 50
    losses_b = rng.uniform(0.5, 1.5, N)
    losses_a = rng.uniform(1.5, 2.5, N)
    result = diebold_mariano(losses_a, losses_b)
    assert result.statistic > 0
    assert result.p_value < 0.05
    assert result.favoured == "model_b"


def test_dm_symmetry():
    rng = np.random.default_rng(2)
    N = 50
    losses_b = rng.uniform(0.5, 1.5, N)
    losses_a = rng.uniform(1.5, 2.5, N)
    res_ab = diebold_mariano(losses_a, losses_b)
    res_ba = diebold_mariano(losses_b, losses_a)
    assert abs(res_ab.statistic + res_ba.statistic) < 1e-10
    assert abs(res_ab.p_value - res_ba.p_value) < 1e-12


def test_dm_raises_if_n_too_small():
    with pytest.raises(ValueError, match="too small"):
        diebold_mariano(_losses(0, 7), _losses(1, 7))


def test_dm_raises_on_nan_losses_a():
    losses = _losses(0, 20)
    nan_losses = losses.copy()
    nan_losses[3] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        diebold_mariano(nan_losses, losses)


def test_dm_raises_on_nan_losses_b():
    losses = _losses(0, 20)
    nan_losses = losses.copy()
    nan_losses[0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        diebold_mariano(losses, nan_losses)


def test_dm_raises_on_length_mismatch():
    with pytest.raises(ValueError, match="[Ll]ength"):
        diebold_mariano(_losses(0, 20), _losses(1, 10))


def test_dm_raises_on_zero_variance_differential():
    # Use np.full to guarantee d_t is exactly constant in floating point.
    # losses_b + 1.0 does NOT reliably give d=1.0 exactly due to fp rounding.
    losses_a = np.full(20, 2.0)
    losses_b = np.full(20, 1.0)
    with pytest.raises(ValueError, match="zero"):
        diebold_mariano(losses_a, losses_b)


def test_dm_hln_correction_is_conservative():
    # Independent series, A clearly higher; N=30, h=1 → HLN factor = sqrt(29/30) < 1
    rng = np.random.default_rng(3)
    N = 30
    losses_b = rng.uniform(0.5, 1.5, N)
    losses_a = rng.uniform(1.5, 2.5, N)
    stat_hln = diebold_mariano(losses_a, losses_b, hln_correction=True).statistic
    stat_raw = diebold_mariano(losses_a, losses_b, hln_correction=False).statistic
    assert abs(stat_hln) < abs(stat_raw)


# ── CW tests ──────────────────────────────────────────────────────────────────

def test_cw_identical_predictions_raises():
    # f_t = 2*(pred_l - pred_s)*e_s = 0 when pred_l == pred_s → zero variance
    rng = np.random.default_rng(0)
    y_true = rng.standard_normal(20)
    pred = rng.standard_normal(20)
    with pytest.raises(ValueError, match="zero"):
        clark_west(y_true, pred, pred.copy())


def test_cw_larger_model_better():
    # pred_larger ≈ y_true → f_t ≈ 2*y_true^2 > 0 → CW stat >> 0 → wins
    rng = np.random.default_rng(42)
    N = 50
    y_true = rng.standard_normal(N)
    y_pred_smaller = np.zeros(N)
    y_pred_larger = y_true + rng.standard_normal(N) * 0.01
    result = clark_west(y_true, y_pred_smaller, y_pred_larger)
    assert result.statistic > 0
    assert result.larger_model_wins


def test_cw_larger_model_worse():
    # pred_larger = -0.5 * y_true (wrong direction)
    # f_t = 2*(-0.5*y - 0)*y = -y^2 → f_bar < 0 → stat << 0 → not significant
    rng = np.random.default_rng(42)
    N = 30
    y_true = rng.standard_normal(N)
    y_pred_smaller = np.zeros(N)
    y_pred_larger = -0.5 * y_true
    result = clark_west(y_true, y_pred_smaller, y_pred_larger)
    assert not result.larger_model_wins


def test_cw_validation_errors():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(20)

    with pytest.raises(ValueError, match="NaN"):
        clark_west(np.full(20, np.nan), y, y * 0.9)

    with pytest.raises(ValueError, match="[Ll]ength"):
        clark_west(y, y[:10], y)

    with pytest.raises(ValueError, match="too small"):
        clark_west(y[:7], y[:7] * 0.5, y[:7] * 0.3)


# ── CW-from-losses tests (loss-agnostic core; QLIKE path) ─────────────────────

def test_cw_from_losses_matches_clark_west_for_mspe():
    # clark_west() must be exactly clark_west_from_losses() fed squared-error
    # losses and the squared-prediction-gap adjustment.
    rng = np.random.default_rng(7)
    N = 40
    y_true = rng.standard_normal(N)
    pred_s = rng.standard_normal(N) * 0.5
    pred_l = y_true + rng.standard_normal(N) * 0.3

    direct = clark_west(y_true, pred_s, pred_l)
    via_losses = clark_west_from_losses(
        (y_true - pred_s) ** 2,
        (y_true - pred_l) ** 2,
        (pred_s - pred_l) ** 2,
    )
    assert via_losses.statistic == pytest.approx(direct.statistic)
    assert via_losses.p_value == pytest.approx(direct.p_value)
    assert via_losses.n == direct.n
    assert direct.loss == "mspe"


def test_cw_from_losses_operates_on_qlike_not_squared_error():
    # The core must use the per-obs losses it is GIVEN, not recompute squared
    # errors internally. Feed QLIKE per-obs losses (variance space) and verify
    # the statistic equals a hand-built QLIKE f-series stat and differs from the
    # MSPE stat on the same forecasts.
    rng = np.random.default_rng(11)
    N = 60
    realised = rng.uniform(0.01, 0.06, N)          # realised variance (>0)
    var_smaller = rng.uniform(0.02, 0.05, N)       # constant-like implied variance
    var_larger = var_smaller + rng.normal(0, 0.004, N)  # HAR-like, nested under H0
    var_larger = np.clip(var_larger, 0.005, None)  # keep strictly positive

    loss_s = qlike_per_obs(realised, var_smaller)
    loss_l = qlike_per_obs(realised, var_larger)
    adj = qlike_per_obs(var_smaller, var_larger)    # smaller forecast = proxy-truth

    result = clark_west_from_losses(loss_s, loss_l, adj, loss="qlike")
    assert result.loss == "qlike"

    # Hand-built QLIKE f-series → same statistic (one-sided upper-tail normal).
    f = loss_s - loss_l + adj
    from src.forecast_comparison import _hac_variance
    expected_stat = f.mean() / np.sqrt(_hac_variance(f, 0) / N)
    assert result.statistic == pytest.approx(expected_stat)

    # The MSPE statistic on the same forecasts uses squared errors, not QLIKE,
    # and must be a genuinely different quantity.
    mspe_result = clark_west(realised, var_smaller, var_larger)
    assert abs(result.statistic - mspe_result.statistic) > 1e-6


def test_cw_from_losses_validation_errors():
    good = np.abs(np.random.default_rng(0).standard_normal(20)) + 0.1

    with pytest.raises(ValueError, match="[Ll]ength"):
        clark_west_from_losses(good, good[:10], good)

    nan_arr = good.copy()
    nan_arr[2] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        clark_west_from_losses(nan_arr, good, good)

    with pytest.raises(ValueError, match="too small"):
        clark_west_from_losses(good[:7], good[:7], good[:7])

    # Constant f (zero adjustment, identical losses) → degenerate.
    const = np.full(20, 0.5)
    with pytest.raises(ValueError, match="zero"):
        clark_west_from_losses(const, const.copy(), np.zeros(20))


# ── block-bootstrap p-value tests (nested significance under QLIKE) ───────────

def test_bootstrap_enforces_null_by_recentering():
    # Constant positive f: with re-centering, g_t = 0, every bootstrap mean is 0,
    # and 0 >= observed_mean (0.5) is never true, so p = 1 / (n_boot + 1) (minimum).
    # WITHOUT re-centering, every bootstrap mean would equal 0.5 and p would be 1.0.
    # p at the minimum therefore proves the null is enforced by re-centering.
    n_boot = 999
    f = np.full(40, 0.5)
    p, observed_mean, block_length = block_bootstrap_pvalue(f, n_boot=n_boot)
    assert observed_mean == pytest.approx(0.5)
    assert p == pytest.approx(1 / (n_boot + 1))


def test_bootstrap_centered_series_gives_p_near_half():
    # Symmetric mean-zero f -> observed_mean ~ 0 -> roughly half the re-centered
    # bootstrap means fall at or above it.
    rng = np.random.default_rng(5)
    f = rng.standard_normal(200)
    f = f - f.mean()  # exactly zero mean
    p, observed_mean, _ = block_bootstrap_pvalue(f, n_boot=2000, seed=1)
    assert observed_mean == pytest.approx(0.0, abs=1e-12)
    assert 0.4 < p < 0.6


def test_bootstrap_circular_centers_skewed_null():
    # Skewed but exactly mean-zero series. Circular blocks weight every
    # observation equally, so the null distribution is centered and p ~ 0.5.
    # A non-circular moving block bootstrap would under-weight the end points,
    # shift the null mean off zero, and push p away from 0.5.
    rng = np.random.default_rng(13)
    f = rng.exponential(1.0, 300)  # right-skewed
    f = f - f.mean()              # exactly zero mean
    p, observed_mean, _ = block_bootstrap_pvalue(f, n_boot=4000, seed=2)
    assert observed_mean == pytest.approx(0.0, abs=1e-12)
    assert 0.45 < p < 0.55


def test_bootstrap_strong_positive_mean_rejects():
    rng = np.random.default_rng(6)
    f = rng.standard_normal(150) * 0.2 + 1.0  # mean ~ 1, small noise
    p, _, _ = block_bootstrap_pvalue(f, n_boot=2000)
    assert p < 0.05


def test_bootstrap_negative_mean_does_not_reject():
    rng = np.random.default_rng(6)
    f = rng.standard_normal(150) * 0.2 - 1.0  # mean ~ -1
    p, _, _ = block_bootstrap_pvalue(f, n_boot=2000)
    assert p > 0.95


def test_bootstrap_block_length_default_rule():
    f = np.arange(222, dtype=float)
    _, _, block_length = block_bootstrap_pvalue(f, n_boot=10)
    assert block_length == int(222 ** config.BOOTSTRAP_BLOCK_LENGTH_EXPONENT)
    assert block_length == 6


def test_bootstrap_deterministic_with_seed():
    # Use a (near) zero-mean series so the p-value sits away from the floor and
    # different seeds give genuinely different bootstrap counts.
    rng = np.random.default_rng(9)
    f = rng.standard_normal(120)
    f = f - f.mean()
    p1, _, _ = block_bootstrap_pvalue(f, n_boot=1500, seed=42)
    p2, _, _ = block_bootstrap_pvalue(f, n_boot=1500, seed=42)
    p3, _, _ = block_bootstrap_pvalue(f, n_boot=1500, seed=43)
    assert p1 == p2  # same seed, identical result
    assert p1 != p3  # different seed, different resamples


def test_bootstrap_validation_errors():
    good = np.abs(np.random.default_rng(0).standard_normal(20)) + 0.1
    with pytest.raises(ValueError, match="NaN"):
        bad = good.copy(); bad[1] = float("nan")
        block_bootstrap_pvalue(bad)
    with pytest.raises(ValueError, match="too small"):
        block_bootstrap_pvalue(good[:7])
    with pytest.raises(ValueError, match="block_length"):
        block_bootstrap_pvalue(good, block_length=21)


def test_clark_west_bootstrap_keeps_point_statistic():
    # The studentised point statistic must be byte-for-byte the clark_west_from_losses
    # value; only the p-value source changes.
    rng = np.random.default_rng(11)
    N = 80
    realised = rng.uniform(0.01, 0.06, N)
    var_smaller = rng.uniform(0.02, 0.05, N)
    var_larger = np.clip(var_smaller + rng.normal(0, 0.004, N), 0.005, None)
    loss_s = qlike_per_obs(realised, var_smaller)
    loss_l = qlike_per_obs(realised, var_larger)
    adj = qlike_per_obs(var_smaller, var_larger)

    cw = clark_west_from_losses(loss_s, loss_l, adj, loss="qlike")
    boot = clark_west_bootstrap(loss_s, loss_l, adj, n_boot=999, loss="qlike")

    assert isinstance(boot, BootstrapCWResult)
    assert boot.statistic == cw.statistic  # point statistic unchanged
    assert boot.observed_mean == pytest.approx(float((loss_s - loss_l + adj).mean()))
    assert boot.n == cw.n
    assert boot.seed == config.BOOTSTRAP_SEED
    assert boot.test_type == "clark_west_bootstrap"
    assert 0.0 < boot.p_value <= 1.0
    assert boot.larger_model_wins == (boot.p_value < boot.alpha)


# ── is_nested tests ───────────────────────────────────────────────────────────

def test_is_nested_canonical_order():
    assert is_nested("constant", "har") is True


def test_is_nested_reversed_order():
    assert is_nested("har", "constant") is True


def test_is_nested_extended_ols_regime_switching():
    # Regime-switching OLS nests Extended OLS (the equality-restricted special case).
    assert is_nested("extended_ols", "regime_switching") is True
    assert is_nested("regime_switching", "extended_ols") is True


def test_is_nested_unrelated_pair():
    assert is_nested("har", "regime_switching") is False


def test_is_nested_xgboost_vs_constant():
    assert is_nested("xgboost", "constant") is False


# ── compare_models dispatcher tests ───────────────────────────────────────────

def test_compare_models_nested_returns_cw():
    rng = np.random.default_rng(0)
    N = 30
    y_true = rng.standard_normal(N)
    y_pred_a = np.zeros(N)
    y_pred_b = y_true + rng.standard_normal(N) * 0.1
    result = compare_models("constant", "har", y_true=y_true, y_pred_a=y_pred_a, y_pred_b=y_pred_b)
    assert isinstance(result, CWResult)
    assert result.test_type == "clark_west"


def test_compare_models_non_nested_returns_dm():
    losses_a = _losses(0, 30)
    losses_b = _losses(1, 30)
    result = compare_models("regime_switching", "xgboost", losses_a=losses_a, losses_b=losses_b)
    assert isinstance(result, DMResult)
    assert result.test_type == "diebold_mariano"


def test_compare_models_nested_without_predictions_raises():
    losses = _losses(0, 30)
    with pytest.raises(ValueError, match="nested"):
        compare_models("constant", "har", losses_a=losses, losses_b=losses)


def test_compare_models_non_nested_without_losses_raises():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(30)
    with pytest.raises(ValueError, match="not a nested"):
        compare_models("regime_switching", "xgboost", y_true=y, y_pred_a=y, y_pred_b=y)


# ── loss_type guard: the N(0,1) CW p-value is squared-error-specific ──────────

def test_clark_west_raises_on_qlike_loss_type():
    rng = np.random.default_rng(0)
    y = np.abs(rng.standard_normal(30)) + 0.1
    with pytest.raises(ValueError, match="clark_west_bootstrap"):
        clark_west(y, y * 0.9, y * 1.1, loss_type="qlike")


def test_compare_models_nested_raises_on_qlike_loss_type():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(30)
    with pytest.raises(ValueError, match="clark_west_bootstrap"):
        compare_models("constant", "har", y_true=y, y_pred_a=y, y_pred_b=y,
                       loss_type="qlike")


def test_clark_west_mspe_route_unaffected_by_default_loss_type():
    rng = np.random.default_rng(0)
    y = rng.standard_normal(30)
    pred_b = y + rng.standard_normal(30) * 0.1
    assert clark_west(y, np.zeros(30), pred_b).loss == "mspe"
    assert clark_west(y, np.zeros(30), pred_b, loss_type="mspe").loss == "mspe"


def test_phantom_regime_constant_pair_removed():
    assert is_nested("constant", "regime_constant") is False
    assert is_nested("regime_constant", "constant") is False
