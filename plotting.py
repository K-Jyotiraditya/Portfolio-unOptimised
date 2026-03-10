"""Visualisations: efficient frontier, NAV + benchmark, weights, regime."""
from __future__ import annotations

import logging
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from config import Config

LOGGER = logging.getLogger("portfolio.plot")

_GREY   = "#d3d3d3"
_BLUE   = "#1f77b4"
_ORANGE = "#ff7f0e"
_GREEN  = "#2ca02c"
_RED    = "#d62728"
_PURPLE = "#9467bd"

_TICKER_COLORS = [_BLUE, _ORANGE, _GREEN, _RED, _PURPLE, _GREY]


def plot_efficient_frontier(frontier_df: pd.DataFrame,
                            w_sharpe: np.ndarray, w_mv: np.ndarray,
                            w_rp: np.ndarray,
                            mu: np.ndarray, cov: np.ndarray,
                            tickers: List[str],
                            rf: float, cfg: Config, path: str) -> None:
    """Efficient frontier with key portfolios marked."""
    fig, ax = plt.subplots(figsize=(10, 7), dpi=cfg.dpi)

    # Frontier curve
    ax.plot(frontier_df["vol"] * 100, frontier_df["ret"] * 100,
            color=_BLUE, lw=2.0, label="Efficient frontier", zorder=2)

    def _mark(w, label, marker, color):
        v = float(np.sqrt(w @ cov @ w)) * 100
        r = float(w @ mu) * 100
        ax.scatter(v, r, s=120, marker=marker, color=color, zorder=5)
        ax.annotate(label, (v, r), textcoords="offset points",
                    xytext=(8, 4), fontsize=9)

    _mark(w_sharpe, "Max Sharpe", "D", _ORANGE)
    _mark(w_mv,     "Min Var",    "s", _GREEN)
    _mark(w_rp,     "Risk Parity","^", _PURPLE)

    # Individual assets
    for i, t in enumerate(tickers):
        e_w = np.zeros(len(tickers)); e_w[i] = 1.0
        ax.scatter(np.sqrt(cov[i, i]) * 100, mu[i] * 100,
                   s=60, marker="o", color=_GREY, zorder=4)
        ax.annotate(t, (np.sqrt(cov[i, i]) * 100, mu[i] * 100),
                    textcoords="offset points", xytext=(5, 2), fontsize=8, color="grey")

    # Capital Market Line
    x_rf = 0.0
    x_max = frontier_df["vol"].max() * 100 * 1.1
    sharpe_r = float(w_sharpe @ mu) * 100
    sharpe_v = float(np.sqrt(w_sharpe @ cov @ w_sharpe)) * 100
    if sharpe_v > 0:
        slope = (sharpe_r - rf * 100) / sharpe_v
        cml_x = [x_rf, x_max]
        cml_y = [rf * 100, rf * 100 + slope * x_max]
        ax.plot(cml_x, cml_y, "--", color=_ORANGE, lw=1.0, alpha=0.6, label="CML")

    ax.set_xlabel("Annualised Volatility (%)")
    ax.set_ylabel("Annualised Return (%)")
    ax.set_title("Efficient Frontier", fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved frontier plot -> %s", path)


def plot_performance(bt: Dict, cfg: Config, path: str) -> None:
    """Four-panel performance dashboard."""
    fig, axes = plt.subplots(4, 1, figsize=(15, 16), dpi=cfg.dpi,
                             gridspec_kw={"height_ratios": [3, 2, 2, 1]})

    nav      = bt["nav"].dropna()
    bench    = bt["benchmark"].dropna()
    weights  = bt["weights"].dropna()
    regime   = bt["regime"]
    port_vol = bt["port_vol"]
    tickers  = bt["tickers"]

    common = nav.index.intersection(bench.index)
    nav   = nav[common]
    bench = bench[common]

    # --- Panel 1: NAV vs Benchmark ---
    ax = axes[0]
    ax.plot(nav.index, nav.to_numpy(), color=_BLUE, lw=1.5, label="Strategy")
    ax.plot(bench.index, bench.to_numpy(), color=_GREY, lw=1.2,
            linestyle="--", label=cfg.bench_ticker)

    # Shade stress periods
    stress_mask = (regime[common] == "STRESS").to_numpy()
    _shade_stress(ax, common, stress_mask)

    # Rebalance markers
    for rd in bt["rebal_dates"]:
        if rd in nav.index:
            ax.axvline(rd, color=_ORANGE, alpha=0.15, lw=0.5)

    ax.set_ylabel("NAV (normalised)")
    ax.set_title("Strategy NAV vs Benchmark", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # --- Panel 2: Drawdown ---
    ax = axes[1]
    roll_max = nav.cummax()
    dd = ((nav - roll_max) / roll_max) * 100
    ax.fill_between(dd.index, dd.to_numpy(), 0, color=_RED, alpha=0.4)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Strategy Drawdown", fontsize=11)
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # --- Panel 3: Weight evolution ---
    ax = axes[2]
    w_arr = weights.reindex(common).ffill().to_numpy()
    bottom = np.zeros(len(common))
    for i, t in enumerate(tickers):
        color = _TICKER_COLORS[i % len(_TICKER_COLORS)]
        ax.fill_between(common, bottom, bottom + w_arr[:, i] * 100,
                        color=color, alpha=0.75, label=t)
        bottom += w_arr[:, i] * 100
    ax.set_ylabel("Weight (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Portfolio Weight Evolution", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=len(tickers), loc="upper left")
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # --- Panel 4: Regime indicator ---
    ax = axes[3]
    regime_common = regime.reindex(common)
    stress_indicator = (regime_common == "STRESS").astype(int).to_numpy()
    ax.fill_between(common, stress_indicator, 0, color=_RED, alpha=0.6, label="STRESS")
    ax.set_yticks([])
    ax.set_ylabel("Regime")
    ax.set_title("Stress Regime Indicator", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved performance plot -> %s", path)


def plot_weights_snapshot(w_sharpe: np.ndarray, w_mv: np.ndarray,
                          w_rp: np.ndarray, tickers: List[str],
                          cfg: Config, path: str) -> None:
    """Bar chart comparing weights across 3 strategies."""
    fig, ax = plt.subplots(figsize=(10, 5), dpi=cfg.dpi)
    x = np.arange(len(tickers))
    width = 0.25

    ax.bar(x - width, w_sharpe * 100, width, color=_ORANGE, alpha=0.85, label="Max Sharpe")
    ax.bar(x,         w_mv     * 100, width, color=_GREEN,  alpha=0.85, label="Min Variance")
    ax.bar(x + width, w_rp     * 100, width, color=_PURPLE, alpha=0.85, label="Risk Parity")

    ax.set_xticks(x)
    ax.set_xticklabels(tickers, fontsize=11)
    ax.set_ylabel("Weight (%)")
    ax.set_title("Portfolio Weights: Strategy Comparison", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved weights snapshot -> %s", path)


def _shade_stress(ax, index, stress_mask):
    """Grey shade for stress periods on an existing axis."""
    in_stress = False
    start = None
    for i, (date, is_stress) in enumerate(zip(index, stress_mask)):
        if is_stress and not in_stress:
            start = date
            in_stress = True
        elif not is_stress and in_stress:
            ax.axvspan(start, date, color=_RED, alpha=0.08)
            in_stress = False
    if in_stress and start is not None:
        ax.axvspan(start, index[-1], color=_RED, alpha=0.08)
