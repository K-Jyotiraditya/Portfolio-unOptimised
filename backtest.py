"""Walk-forward strategy backtest with transaction costs and regime awareness.

Optimization schedule: min-variance weights are re-computed on each calendar
rebalance date. Min-variance is used instead of max-Sharpe because 252-day
sample means have estimation error larger than the signal itself, causing
MVO to make unstable bets. Min-variance relies only on the covariance matrix,
which is estimable with far less data.

Regime suppression: if STRESS, calendar rebalances are deferred. This avoids
forced selling at distressed prices.

Benchmark: buy-and-hold bench_ticker from warmup day 1.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config import Config
from data import sample_stats
from optimizer import (
    black_litterman, ledoit_wolf_cov, min_variance,
    momentum_scores, momentum_tilt, risk_parity,
)
from rebalance import (
    _calendar_dates,
    apply_costs,
    compute_actual_weights,
    should_rebalance,
)
from regime import defensive_weights, detect_regime, portfolio_ewma_vol, regime_stats

LOGGER = logging.getLogger("portfolio.backtest")


def run_backtest(returns: pd.DataFrame, prices: pd.DataFrame,
                 cfg: Config, rf_series: pd.Series = None) -> Dict:
    """Full walk-forward backtest. Returns result dict."""
    tickers = list(cfg.tickers)
    n = len(tickers)
    index = returns.index
    T = len(index)

    warmup = cfg.lookback_days
    if warmup >= T:
        raise ValueError(f"lookback_days ({warmup}) >= history length ({T})")

    # Pre-compute portfolio EWMA vol with equal weights for regime labelling
    w_eq = np.full(n, 1.0 / n)
    port_vol_series = portfolio_ewma_vol(returns, w_eq, cfg.ewma_lambda)
    regime_series = detect_regime(port_vol_series, cfg.stress_vol_threshold)

    cal_dates = set(_calendar_dates(index, cfg.rebal_calendar))
    bnds = (cfg.min_weight, cfg.max_weight)

    # State
    nav = 1.0
    w = w_eq.copy()           # start equal-weight
    w_fixed = w.copy()        # last rebalanced-to weights (drift anchor)
    w_target = w.copy()       # current optimised target (updated on cal dates)

    nav_series   = np.full(T, np.nan)
    w_history    = np.full((T, n), np.nan)
    rebal_dates  = []
    rebal_suppressed = []
    costs_total  = 0.0
    turnover_list = []

    for t in range(T):
        date = index[t]

        # ---- warmup: hold equal-weight, no optimisation ----
        if t < warmup:
            if t > 0:
                port_ret = float(returns.iloc[t].to_numpy() @ w)
                nav *= (1 + port_ret)
            nav_series[t] = nav
            w_history[t] = w
            continue

        # ---- drift actual weights with today's price change ----
        if t > 0:
            p_today = prices.iloc[t].to_numpy()
            p_prev  = prices.iloc[t - 1].to_numpy()
            w = compute_actual_weights(p_today, p_prev, w)

        regime_today = regime_series.iloc[t]
        is_cal = date in cal_dates

        # ---- re-optimise on calendar dates ----
        if is_cal:
            if regime_today == "CALM":
                # Ledoit-Wolf covariance on trailing window
                ret_window = returns.iloc[max(0, t - warmup):t]
                cov_lw = ledoit_wolf_cov(ret_window)

                # Momentum scores from price history
                price_window = prices.iloc[:t]
                mom = momentum_scores(price_window,
                                      lookback=cfg.lookback_days, skip=21)

                try:
                    w_mv = min_variance(ret_window.mean().to_numpy(), cov_lw, bnds)
                    w_target = momentum_tilt(w_mv, mom, alpha=0.30, bounds=bnds)
                except Exception:
                    w_target = risk_parity(cov_lw, bnds)

            else:
                # STRESS regime: shift to defensive allocation immediately
                w_target = defensive_weights(tickers, bounds=bnds)
                rebal_suppressed.append(date)   # log as suppressed (regular rebal)

        # ---- rebalance decision ----
        # In STRESS, force a defensive rebalance on calendar dates
        if regime_today == "STRESS" and is_cal:
            do_rebal = True
        else:
            do_rebal = should_rebalance(date, w, w_target, regime_today, is_cal, cfg)

        if do_rebal:
            cost = apply_costs(w, w_target, nav, cfg.cost_bps, cfg.min_trade_pct)
            nav -= cost
            costs_total += cost
            turnover = float(np.abs(w_target - w).sum()) / 2
            turnover_list.append(turnover)
            rebal_dates.append(date)
            w = w_target.copy()
            w_fixed = w_target.copy()

        # ---- advance NAV ----
        port_ret = float(returns.iloc[t].to_numpy() @ w)
        nav *= (1 + port_ret)
        nav_series[t] = nav
        w_history[t] = w

    nav_s = pd.Series(nav_series, index=index, name="strategy")

    # ---- benchmark: buy-and-hold bench_ticker ----
    bench_col = cfg.bench_ticker if cfg.bench_ticker in tickers else tickers[0]
    bench_idx = tickers.index(bench_col)
    bench_rets = returns.iloc[:, bench_idx]
    bench_nav = (1 + bench_rets).cumprod()
    bench_nav = bench_nav / bench_nav.iloc[warmup] * nav_series[warmup]
    bench_nav = pd.Series(bench_nav.to_numpy(), index=index, name="benchmark")

    # ---- regime stats on post-warmup period ----
    port_daily = nav_s.pct_change().dropna()
    r_stats = regime_stats(regime_series.iloc[warmup:],
                           port_daily.iloc[warmup - 1:])

    return {
        "nav": nav_s,
        "benchmark": bench_nav,
        "weights": pd.DataFrame(w_history, index=index, columns=tickers),
        "regime": regime_series,
        "port_vol": port_vol_series,
        "rebal_dates": rebal_dates,
        "rebal_suppressed": rebal_suppressed,
        "costs_total": costs_total,
        "avg_turnover": float(np.mean(turnover_list)) if turnover_list else 0.0,
        "n_rebal": len(rebal_dates),
        "n_suppressed": len(rebal_suppressed),
        "regime_stats": r_stats,
        "tickers": tickers,
    }


def performance_metrics(nav: pd.Series, rf: float = 0.045,
                        rf_series: pd.Series = None) -> Dict:
    """Annualised return, vol, Sharpe, max drawdown, Calmar.

    If rf_series is provided (daily decimal), uses time-varying risk-free
    rate for the Sharpe calculation — more accurate across rate cycles.
    """
    r = nav.pct_change().dropna()
    ann_ret = float((1 + r).prod() ** (252 / len(r)) - 1)
    ann_vol = float(r.std() * np.sqrt(252))

    if rf_series is not None:
        rf_daily = rf_series.reindex(r.index).ffill().fillna(rf / 252)
        avg_rf_ann = float(rf_daily.mean() * 252)
        excess = r - rf_daily
        sharpe = float(excess.mean() / (excess.std() + 1e-12) * np.sqrt(252))
    else:
        avg_rf_ann = rf
        sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0

    roll_max = nav.cummax()
    dd = (nav - roll_max) / roll_max
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0

    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "avg_rf": avg_rf_ann,
        "max_drawdown": max_dd,
        "calmar": calmar,
    }


def tracking_metrics(strat: pd.Series, bench: pd.Series,
                     rf_series: pd.Series = None, rf: float = 0.045) -> Dict:
    """Alpha (CAPM), beta, tracking error, information ratio."""
    s = strat.pct_change().dropna()
    b = bench.pct_change().dropna()
    common = s.index.intersection(b.index)
    s, b = s[common], b[common]

    if rf_series is not None:
        rf_d = rf_series.reindex(common).ffill().fillna(rf / 252)
    else:
        rf_d = pd.Series(rf / 252, index=common)

    cov_mat = np.cov((s - rf_d).to_numpy(), (b - rf_d).to_numpy())
    beta = cov_mat[0, 1] / (cov_mat[1, 1] + 1e-12)
    alpha = ((s - rf_d).mean() - beta * (b - rf_d).mean()) * 252
    te = float((s - b).std() * np.sqrt(252))
    ir = float((s - b).mean() * 252 / te) if te > 0 else 0.0

    return {"alpha": alpha, "beta": beta, "tracking_error": te, "info_ratio": ir}
