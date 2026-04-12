"""Offline unit tests — no network required."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import Config
from optimizer import (
    black_litterman,
    efficient_frontier,
    max_sharpe,
    min_variance,
    risk_parity,
)
from regime import detect_regime, ewma_vol_series, regime_stats
from rebalance import apply_costs, compute_actual_weights, should_rebalance
from backtest import performance_metrics, tracking_metrics


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def simple_cov():
    return np.array([[0.04, 0.01], [0.01, 0.02]])   # 2-asset, annualised


@pytest.fixture
def simple_mu():
    return np.array([0.10, 0.06])


# --------------------------------------------------------------------------- #
# Optimizer — Max Sharpe
# --------------------------------------------------------------------------- #
def test_max_sharpe_weights_sum_to_one(simple_mu, simple_cov):
    w = max_sharpe(simple_mu, simple_cov, rf=0.04)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.all(w >= 0)


def test_max_sharpe_prefers_higher_return_asset(simple_mu, simple_cov):
    w = max_sharpe(simple_mu, simple_cov, rf=0.04, bounds=(0.0, 1.0))
    # Asset 0 has higher expected return and reasonable vol -> should dominate
    assert w[0] > w[1]


def test_max_sharpe_bounds_respected(simple_mu, simple_cov):
    lo, hi = 0.1, 0.7
    w = max_sharpe(simple_mu, simple_cov, rf=0.04, bounds=(lo, hi))
    assert np.all(w >= lo - 1e-6)
    assert np.all(w <= hi + 1e-6)


# --------------------------------------------------------------------------- #
# Optimizer — Min Variance
# --------------------------------------------------------------------------- #
def test_min_variance_lower_vol_than_max_sharpe(simple_mu, simple_cov):
    w_mv = min_variance(simple_mu, simple_cov, bounds=(0.0, 1.0))
    w_ms = max_sharpe(simple_mu, simple_cov, rf=0.04, bounds=(0.0, 1.0))
    vol_mv = float(np.sqrt(w_mv @ simple_cov @ w_mv))
    vol_ms = float(np.sqrt(w_ms @ simple_cov @ w_ms))
    assert vol_mv <= vol_ms + 1e-6


def test_min_variance_weights_sum_to_one(simple_mu, simple_cov):
    w = min_variance(simple_mu, simple_cov)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Optimizer — Risk Parity
# --------------------------------------------------------------------------- #
def test_risk_parity_equal_cov_equal_weights():
    cov = np.eye(3) * 0.02
    w = risk_parity(cov, bounds=(0.0, 1.0))
    assert w == pytest.approx(np.full(3, 1 / 3), abs=0.01)


def test_risk_parity_sums_to_one(simple_cov):
    w = risk_parity(simple_cov)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Optimizer — Efficient Frontier
# --------------------------------------------------------------------------- #
def test_frontier_monotone_vol(simple_mu, simple_cov):
    df = efficient_frontier(simple_mu, simple_cov, rf=0.04,
                            bounds=(0.0, 1.0), n_points=20)
    assert len(df) > 5
    # vol should be (roughly) increasing along the frontier
    vols = df["vol"].to_numpy()
    assert vols[-1] >= vols[0] - 1e-4


# --------------------------------------------------------------------------- #
# Optimizer — Black-Litterman
# --------------------------------------------------------------------------- #
def test_bl_no_views_returns_equilibrium(simple_mu, simple_cov):
    w_mkt = np.array([0.6, 0.4])
    mu_bl, _ = black_litterman(simple_mu, simple_cov, w_mkt, {}, ("A", "B"),
                               tau=0.05, delta=2.5)
    # With no views, posterior = pi (equilibrium)
    pi = 2.5 * simple_cov @ w_mkt
    assert mu_bl == pytest.approx(pi, rel=0.01)


def test_bl_view_pulls_return(simple_mu, simple_cov):
    w_mkt = np.array([0.5, 0.5])
    views = {"A": 0.20}    # bullish view on asset A
    mu_bl, _ = black_litterman(simple_mu, simple_cov, w_mkt, views, ("A", "B"),
                               tau=0.05, delta=2.5)
    pi = 2.5 * simple_cov @ w_mkt
    # BL return for A should be pulled toward 0.20 (view) vs equilibrium
    assert mu_bl[0] > pi[0] - 1e-6   # not below equilibrium when view is bullish


# --------------------------------------------------------------------------- #
# Regime detection
# --------------------------------------------------------------------------- #
def test_ewma_vol_positive_length():
    r = pd.Series(np.random.default_rng(0).normal(0, 0.01, 300))
    v = ewma_vol_series(r, 0.94)
    assert len(v) == 300
    assert np.all(v > 0)


def test_detect_regime_all_calm_when_low_vol():
    idx = pd.date_range("2020-01-01", periods=100)
    vol = pd.Series(np.full(100, 0.05), index=idx)   # well below 20%
    labels = detect_regime(vol, stress_threshold=0.20)
    assert (labels == "CALM").all()


def test_detect_regime_stress_after_streak():
    idx = pd.date_range("2020-01-01", periods=20)
    vals = np.full(20, 0.05)
    vals[10:] = 0.30   # crosses threshold from day 10
    vol = pd.Series(vals, index=idx)
    labels = detect_regime(vol, stress_threshold=0.20, min_stress_days=3)
    assert labels.iloc[12] == "STRESS"   # 3 days after crossing
    assert labels.iloc[0]  == "CALM"


# --------------------------------------------------------------------------- #
# Rebalancing
# --------------------------------------------------------------------------- #
def test_compute_actual_weights_drift():
    prices_prev = np.array([100.0, 100.0])
    prices_today = np.array([110.0, 100.0])   # asset 0 up 10%
    w = np.array([0.5, 0.5])
    w_new = compute_actual_weights(prices_today, prices_prev, w)
    assert w_new[0] > w_new[1]
    assert w_new.sum() == pytest.approx(1.0, abs=1e-8)


def test_compute_actual_weights_sums_to_one():
    rng = np.random.default_rng(0)
    p1 = rng.uniform(50, 150, 5)
    p2 = rng.uniform(50, 150, 5)
    w = np.full(5, 0.2)
    w_new = compute_actual_weights(p2, p1, w)
    assert w_new.sum() == pytest.approx(1.0, abs=1e-8)


def test_apply_costs_proportional():
    cost = apply_costs(np.array([0.5, 0.5]), np.array([0.6, 0.4]),
                       portfolio_value=1_000_000, cost_bps=10.0, min_trade_pct=0.0)
    # 10% turnover one-way, 10bps = 0.001 -> $100
    assert cost == pytest.approx(200.0, rel=0.01)   # both legs trade 10%


def test_no_rebal_in_stress():
    cfg = Config(rebal_use_regime=True, rebal_threshold=0.20)
    w_actual = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    w_target = np.array([0.6, 0.2, 0.1, 0.05, 0.05])
    result = should_rebalance(
        pd.Timestamp("2020-03-20"), w_actual, w_target,
        regime="STRESS", is_calendar_day=True, cfg=cfg,
    )
    assert result is False


def test_rebal_on_calendar_when_calm():
    cfg = Config(rebal_use_regime=True, rebal_threshold=0.20)
    w = np.full(5, 0.2)
    result = should_rebalance(
        pd.Timestamp("2020-06-01"), w, w,
        regime="CALM", is_calendar_day=True, cfg=cfg,
    )
    assert result is True


def test_rebal_on_drift_when_calm():
    cfg = Config(rebal_use_regime=True, rebal_threshold=0.05)
    w_actual = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    w_target = np.array([0.4, 0.3, 0.1, 0.1, 0.1])   # drift > 5%
    result = should_rebalance(
        pd.Timestamp("2020-06-15"), w_actual, w_target,
        regime="CALM", is_calendar_day=False, cfg=cfg,
    )
    assert result is True


# --------------------------------------------------------------------------- #
# Performance metrics
# --------------------------------------------------------------------------- #
def test_performance_metrics_basic():
    idx = pd.date_range("2020-01-01", periods=252)
    nav = pd.Series((1.10 ** (1 / 252)) ** np.arange(252), index=idx)
    m = performance_metrics(nav, rf=0.0)
    assert m["ann_return"] == pytest.approx(0.10, abs=0.01)
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-6)


def test_performance_metrics_drawdown():
    nav = pd.Series([1.0, 1.1, 0.9, 0.95, 1.0])
    m = performance_metrics(nav, rf=0.0)
    assert m["max_drawdown"] < 0


def test_tracking_metrics_identical_series():
    idx = pd.date_range("2020-01-01", periods=252)
    nav = pd.Series(np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, 252)), index=idx)
    t = tracking_metrics(nav, nav)
    assert t["tracking_error"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #
def test_config_bad_bounds_raises():
    with pytest.raises(ValueError, match="min_weight"):
        Config(min_weight=0.8, max_weight=0.2)


def test_config_bad_ewma_raises():
    with pytest.raises(ValueError, match="ewma_lambda"):
        Config(ewma_lambda=1.5)


def test_config_bad_rebal_calendar_raises():
    with pytest.raises(ValueError, match="rebal_calendar"):
        Config(rebal_calendar="annual")


def test_config_single_ticker_raises():
    with pytest.raises(ValueError, match="at least 2"):
        Config(tickers=("SPY",))
