"""Portfolio Optimization + Dynamic Rebalancing Engine.

    python main.py        # downloads + caches prices, runs full pipeline
    python -m pytest -q   # offline unit tests (no network)
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from backtest import performance_metrics, run_backtest, tracking_metrics
from config import Config
from data import compute_returns, load_prices, load_risk_free_rate, sample_stats
from optimizer import (
    black_litterman,
    efficient_frontier,
    max_sharpe,
    min_variance,
    risk_parity,
)
from plotting import plot_efficient_frontier, plot_performance, plot_weights_snapshot
from regime import regime_stats

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
LOGGER = logging.getLogger("portfolio")
HERE = Path(__file__).resolve().parent

W = 64


def _line(char="-"):
    return char * W


def run(cfg: Config) -> dict:
    # ------------------------------------------------------------------ #
    # 1. Data
    # ------------------------------------------------------------------ #
    prices = load_prices(cfg)
    returns = compute_returns(prices)
    tickers = list(cfg.tickers)
    n = len(tickers)

    start_date = returns.index[0].date()
    end_date   = returns.index[-1].date()
    LOGGER.info("Universe: %s  |  history: %d days (%s to %s)",
                tickers, len(returns), start_date, end_date)

    rf_series = load_risk_free_rate(cfg, returns.index)
    avg_rf = float(rf_series.mean() * 252)
    LOGGER.info("Avg risk-free rate over sample: %.2f%%", avg_rf * 100)

    # ------------------------------------------------------------------ #
    # 2. Point-in-time estimates (full history for snapshot)
    # ------------------------------------------------------------------ #
    mu_ann, cov_ann = sample_stats(returns, cfg.lookback_days)
    mu  = mu_ann.to_numpy()
    cov = cov_ann.to_numpy()

    # ------------------------------------------------------------------ #
    # 3. Optimal portfolios
    # ------------------------------------------------------------------ #
    bnds = (cfg.min_weight, cfg.max_weight)
    w_sharpe = max_sharpe(mu, cov, cfg.risk_free_rate, bnds)
    w_mv     = min_variance(mu, cov, bnds)
    w_rp     = risk_parity(cov, bnds)

    # Black-Litterman (if views provided)
    if cfg.bl_views:
        w_mkt = np.full(n, 1.0 / n)
        mu_bl, cov_bl = black_litterman(
            mu, cov, w_mkt, cfg.bl_views, tickers,
            tau=cfg.bl_tau, rf=cfg.risk_free_rate,
        )
        w_bl = max_sharpe(mu_bl, cov_bl, cfg.risk_free_rate, bnds)
    else:
        mu_bl = cov_bl = w_bl = None

    frontier_df = efficient_frontier(mu, cov, cfg.risk_free_rate, bnds, n_points=60)

    # ------------------------------------------------------------------ #
    # 4. Walk-forward backtest
    # ------------------------------------------------------------------ #
    bt = run_backtest(returns, prices, cfg, rf_series=rf_series)

    perf_strat = performance_metrics(bt["nav"].dropna(), cfg.risk_free_rate, rf_series)
    perf_bench = performance_metrics(bt["benchmark"].dropna(), cfg.risk_free_rate, rf_series)
    track = tracking_metrics(bt["nav"].dropna(), bt["benchmark"].dropna(),
                             rf_series, cfg.risk_free_rate)

    # ------------------------------------------------------------------ #
    # 5. Report
    # ------------------------------------------------------------------ #
    print()
    print(_line("="))
    print("  PORTFOLIO OPTIMIZATION + DYNAMIC REBALANCING")
    print("  Universe  : " + " / ".join(tickers))
    print(f"  History   : {start_date} to {end_date}  ({len(returns):,} trading days)")
    print(f"  Risk-free : {avg_rf*100:.2f}% avg (actual T-bill)  |  "
          f"Target vol : {cfg.target_vol*100:.0f}%  |  "
          f"Rebal : {cfg.rebal_calendar}")
    print(_line("="))

    print("\n  POINT-IN-TIME OPTIMAL WEIGHTS  (last %d trading days)" % cfg.lookback_days)
    print(f"  {'Ticker':<8} {'MaxSharpe':>10} {'MinVar':>10} {'RiskParity':>12}")
    print("  " + _line("-")[:44])
    for i, t in enumerate(tickers):
        print(f"  {t:<8} {w_sharpe[i]*100:>9.1f}% {w_mv[i]*100:>9.1f}% {w_rp[i]*100:>11.1f}%")

    print("\n  PORTFOLIO STATISTICS  (annualised, same window)")
    def _pstats(label, w):
        r = float(w @ mu)
        v = float(np.sqrt(w @ cov @ w))
        s = (r - cfg.risk_free_rate) / v if v > 0 else 0.0
        print(f"  {label:<14}  ret {r*100:>6.2f}%  vol {v*100:>6.2f}%  Sharpe {s:>5.2f}")

    _pstats("Max Sharpe", w_sharpe)
    _pstats("Min Variance", w_mv)
    _pstats("Risk Parity", w_rp)
    if w_bl is not None:
        _pstats("BL Max Sharpe", w_bl)

    print("\n  BACKTEST RESULTS  (walk-forward, %d-day warmup)" % cfg.lookback_days)
    print(_line("-"))
    print(f"  {'':24} {'Strategy':>12} {'Benchmark':>12}")
    print(f"  {'Annualised Return':<24} {perf_strat['ann_return']*100:>11.2f}% "
          f"{perf_bench['ann_return']*100:>11.2f}%")
    print(f"  {'Annualised Vol':<24} {perf_strat['ann_vol']*100:>11.2f}% "
          f"{perf_bench['ann_vol']*100:>11.2f}%")
    print(f"  {'Sharpe Ratio':<24} {perf_strat['sharpe']:>12.2f} "
          f"{perf_bench['sharpe']:>12.2f}")
    print(f"  {'Max Drawdown':<24} {perf_strat['max_drawdown']*100:>11.2f}% "
          f"{perf_bench['max_drawdown']*100:>11.2f}%")
    print(f"  {'Calmar Ratio':<24} {perf_strat['calmar']:>12.2f} "
          f"{perf_bench['calmar']:>12.2f}")

    print("\n  RELATIVE PERFORMANCE  (vs benchmark)")
    print(_line("-"))
    print(f"  Alpha              {track['alpha']*100:>7.2f}%  p.a.")
    print(f"  Beta               {track['beta']:>8.3f}")
    print(f"  Tracking Error     {track['tracking_error']*100:>7.2f}%  p.a.")
    print(f"  Information Ratio  {track['info_ratio']:>8.2f}")

    print("\n  REGIME ANALYSIS")
    print(_line("-"))
    rs = bt["regime_stats"]
    for label in ("CALM", "STRESS"):
        s = rs.get(label, {})
        if s:
            print(f"  {label:<8} {s['days']:>5d} days ({s['pct_of_sample']*100:.0f}%)  "
                  f"ret {s['ann_return']*100:>6.2f}%  vol {s['ann_vol']*100:>6.2f}%  "
                  f"Sharpe {s['sharpe']:>5.2f}")

    print("\n  REBALANCING SUMMARY")
    print(_line("-"))
    print(f"  Total rebalances   : {bt['n_rebal']}")
    print(f"  Suppressed (stress): {bt['n_suppressed']}")
    print(f"  Avg turnover/rebal : {bt['avg_turnover']*100:.1f}%  (one-way)")
    print(f"  Total cost drag    : {bt['costs_total']*100:.4f}% of initial NAV")
    print(_line("="))
    print()

    # ------------------------------------------------------------------ #
    # 6. Plots
    # ------------------------------------------------------------------ #
    plot_efficient_frontier(
        frontier_df, w_sharpe, w_mv, w_rp,
        mu, cov, tickers, cfg.risk_free_rate, cfg,
        str(HERE / "efficient_frontier.png")
    )
    plot_performance(bt, cfg, str(HERE / "performance.png"))
    plot_weights_snapshot(w_sharpe, w_mv, w_rp, tickers, cfg,
                          str(HERE / "weights_snapshot.png"))

    return {
        "weights": {"max_sharpe": w_sharpe, "min_var": w_mv, "risk_parity": w_rp},
        "frontier": frontier_df,
        "backtest": bt,
        "perf_strategy": perf_strat,
        "perf_benchmark": perf_bench,
        "tracking": track,
    }


def main() -> int:
    try:
        run(Config(cache_path=str(HERE / "prices.pkl")))
    except RuntimeError as exc:
        LOGGER.error("Aborted: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
