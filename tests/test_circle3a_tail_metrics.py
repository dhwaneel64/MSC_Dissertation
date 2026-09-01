"""Correctness tests for the Circle 3A tail-risk episode table and tail metrics.

These are descriptive aggregations of the existing pnl frame, so the tests pin the
aggregation rules: episodes sum the right net-base column over the right window,
the flat benchmark is zero, VIX-spike months become their own disjoint rows, and
the full-sample metrics (annualisation, drawdown, worst-month, worst-3m, hit rate)
match direct recomputation on the same series.
"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.circle3a.tail_metrics import (
    episode_pnl_table,
    tail_metrics_table,
    sharpe_diff_bootstrap,
)


def _make_pnl(n: int = 60, seed: int = 0):
    """A minimal pnl-shaped frame with the columns the tail layer consumes."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2007-01-01", periods=n, freq="MS")
    pos_naive = rng.integers(0, 2, n)
    # conditioned is naive gated off in some months.
    gate = rng.integers(0, 2, n)
    pos_cond = pos_naive * gate

    payoff = rng.normal(0.002, 0.02, n)
    net_naive_base = pos_naive * payoff - pos_naive * 0.0001
    net_cond_base = pos_cond * payoff - pos_cond * 0.0001
    net_naive_stress = pos_naive * payoff - pos_naive * 0.0002
    net_cond_stress = pos_cond * payoff - pos_cond * 0.0002

    pnl = pd.DataFrame(
        {
            "position_naive": pos_naive,
            "position_conditioned": pos_cond,
            "net_naive_base": net_naive_base,
            "net_conditioned_base": net_cond_base,
            "net_naive_stress": net_naive_stress,
            "net_conditioned_stress": net_cond_stress,
        },
        index=idx,
    )
    vix = pd.Series(rng.uniform(12.0, 30.0, n), index=idx, name="vix")
    return pnl, vix


def test_episode_columns_and_flat_zero():
    pnl, vix = _make_pnl()
    windows = (("w1", "2007-03-01", "2007-06-30"),)
    table = episode_pnl_table(pnl, vix, crash_windows=windows, vix_threshold=100)
    assert list(table.columns) == ["naive_net", "conditioned_net", "flat_net"]
    assert (table["flat_net"] == 0.0).all()
    # naive_net equals the direct window sum.
    mask = (pnl.index >= pd.Timestamp("2007-03-01")) & (pnl.index <= pd.Timestamp("2007-06-30"))
    assert table.loc["w1", "naive_net"] == pytest.approx(pnl.loc[mask, "net_naive_base"].sum())
    assert table.loc["w1", "conditioned_net"] == pytest.approx(pnl.loc[mask, "net_conditioned_base"].sum())


def test_vix_spike_month_added_and_disjoint():
    pnl, vix = _make_pnl()
    spike = pnl.index[10]
    vix.loc[spike] = 45.0
    # A second spike inside a named window must NOT get its own row.
    inside = pnl.index[3]
    vix.loc[inside] = 55.0
    windows = (("named", "2007-03-01", "2007-05-31"),)  # covers index[2..4]
    table = episode_pnl_table(pnl, vix, crash_windows=windows, vix_threshold=40)
    labels = list(table.index)
    assert f"vix>40:{spike.date()}" in labels
    assert f"vix>40:{inside.date()}" not in labels
    # single-month row equals that month's net.
    assert table.loc[f"vix>40:{spike.date()}", "naive_net"] == pytest.approx(
        pnl.loc[spike, "net_naive_base"]
    )


def test_episode_raises_on_missing_col_and_vix_nan():
    pnl, vix = _make_pnl()
    with pytest.raises(ValueError):
        episode_pnl_table(pnl.drop(columns=["net_naive_base"]), vix)
    bad = vix.copy()
    bad.iloc[5] = np.nan
    with pytest.raises(ValueError):
        episode_pnl_table(pnl, bad)


def test_metrics_rows_and_columns():
    pnl, _ = _make_pnl()
    m = tail_metrics_table(pnl)
    assert list(m.index) == ["naive", "conditioned"]
    for col in [
        "total_net_base", "sharpe_base", "total_net_stress", "sharpe_stress",
        "max_drawdown", "worst_month", "worst_3m", "skew", "hit_rate",
    ]:
        assert col in m.columns


def test_metrics_match_direct_recompute():
    pnl, _ = _make_pnl(seed=3)
    m = tail_metrics_table(pnl)
    base = pnl["net_naive_base"]

    assert m.loc["naive", "total_net_base"] == pytest.approx(base.sum())
    ann_mean = base.mean() * config.MONTHS_PER_YEAR
    ann_vol = base.std(ddof=1) * np.sqrt(config.MONTHS_PER_YEAR)
    assert m.loc["naive", "sharpe_base"] == pytest.approx(ann_mean / ann_vol)
    assert m.loc["naive", "worst_month"] == pytest.approx(base.min())
    assert m.loc["naive", "worst_3m"] == pytest.approx(
        base.rolling(config.CIRCLE3A_WORST_ROLL_MONTHS).sum().min()
    )

    # Max drawdown as a positive magnitude on the cumulative sum.
    cum = base.cumsum()
    expected_dd = float(-(cum - cum.cummax()).min())
    assert m.loc["naive", "max_drawdown"] == pytest.approx(expected_dd)
    assert m.loc["naive", "max_drawdown"] >= 0.0

    # Hit rate over traded months only.
    traded = pnl["position_naive"] == 1
    assert m.loc["naive", "hit_rate"] == pytest.approx((base[traded] > 0).mean())


def test_metrics_raises_on_missing_col():
    pnl, _ = _make_pnl()
    with pytest.raises(ValueError):
        tail_metrics_table(pnl.drop(columns=["net_conditioned_stress"]))


def test_sharpe_diff_shape_and_diff_identity():
    pnl, _ = _make_pnl()
    sd = sharpe_diff_bootstrap(pnl)
    assert list(sd.index) == ["base", "stress"]
    assert list(sd.columns) == [
        "sharpe_naive", "sharpe_conditioned", "sharpe_diff",
        "p_value", "block_length", "n_months",
    ]
    # sharpe_diff is exactly conditioned minus naive, and matches the tail-metrics
    # table Sharpes on the base row.
    for row in ("base", "stress"):
        assert sd.loc[row, "sharpe_diff"] == pytest.approx(
            sd.loc[row, "sharpe_conditioned"] - sd.loc[row, "sharpe_naive"]
        )
    m = tail_metrics_table(pnl)
    assert sd.loc["base", "sharpe_naive"] == pytest.approx(m.loc["naive", "sharpe_base"])
    assert sd.loc["base", "sharpe_conditioned"] == pytest.approx(m.loc["conditioned", "sharpe_base"])


def test_sharpe_diff_pvalue_and_block_from_config():
    pnl, _ = _make_pnl(seed=4)
    sd = sharpe_diff_bootstrap(pnl)
    n = len(pnl)
    expected_block = max(1, int(n ** config.BOOTSTRAP_BLOCK_LENGTH_EXPONENT))
    for row in ("base", "stress"):
        p = sd.loc[row, "p_value"]
        assert 0.0 < p <= 1.0
        assert sd.loc[row, "block_length"] == expected_block
        assert sd.loc[row, "n_months"] == n


def test_sharpe_diff_reuses_block_bootstrap_pvalue(monkeypatch):
    """The p-value must come from the shared util, not a second bootstrap."""
    import src.circle3a.tail_metrics as tm

    calls = {"n": 0}
    real = tm.block_bootstrap_pvalue

    def _spy(f, **kwargs):
        calls["n"] += 1
        return real(f, **kwargs)

    monkeypatch.setattr(tm, "block_bootstrap_pvalue", _spy)
    pnl, _ = _make_pnl(seed=5)
    tm.sharpe_diff_bootstrap(pnl)
    assert calls["n"] == 2  # once per cost basis


def test_sharpe_diff_raises_on_missing_col():
    pnl, _ = _make_pnl()
    with pytest.raises(ValueError):
        sharpe_diff_bootstrap(pnl.drop(columns=["net_naive_stress"]))
