"""Volatility regime detection using EWMA realised vol.

Simple but effective: classify each day as CALM or STRESS based on
whether the EWMA annualised vol exceeds a configurable threshold.
A two-state HMM-inspired smoothing (require N consecutive days in
stress before switching) prevents whipsaw rebalancing suppression.
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("portfolio.regime")

RegimeLabel = Literal["CALM", "STRESS"]


def ewma_vol_series(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """Daily EWMA volatility (annualised) for a single return series."""
    r = returns.to_numpy()
    n = len(r)
    v2 = np.empty(n)
    v2[0] = r[0] ** 2
    for t in range(1, n):
        v2[t] = lam * v2[t - 1] + (1 - lam) * r[t] ** 2
    ann_vol = np.sqrt(v2 * 252)
    return pd.Series(ann_vol, index=returns.index)


def portfolio_ewma_vol(returns: pd.DataFrame, weights: np.ndarray,
                       lam: float = 0.94) -> pd.Series:
    """EWMA vol of the weighted portfolio return series."""
    w = np.asarray(weights, float)
    port_ret = returns @ w
    return ewma_vol_series(port_ret, lam)


def detect_regime(port_vol: pd.Series,
                  stress_threshold: float = 0.20,
                  min_stress_days: int = 3) -> pd.Series:
    """Return a Series of 'CALM' / 'STRESS' labels.

    Switches to STRESS only after min_stress_days consecutive days
    above threshold (avoids single-spike false alarms).
    Switches back to CALM immediately when vol drops below threshold.
    """
    above = (port_vol >= stress_threshold).to_numpy()
    labels = np.where(above, "STRESS", "CALM").astype(object)

    # Apply smoothing: only enter STRESS after min_stress_days consecutive
    state = "CALM"
    streak = 0
    result = labels.copy()
    for i, a in enumerate(above):
        if a:
            streak += 1
        else:
            streak = 0
            state = "CALM"
        if streak >= min_stress_days:
            state = "STRESS"
        result[i] = state

    return pd.Series(result, index=port_vol.index, dtype=str)


def defensive_weights(tickers: list,
                      defensive: list = None,
                      risk_off: list = None,
                      bounds: tuple = (0.02, 0.60)) -> np.ndarray:
    """Tactical defensive allocation for STRESS regime.

    Concentrates weight in safe-haven assets (bonds, gold) and floors
    risky assets (equities, REITs) at the minimum bound.

    defensive : assets to overweight (e.g. ["TLT", "GLD", "IEF"])
    risk_off  : assets to underweight to min bound (e.g. ["SPY", "VNQ"])

    Any ticker not in either list gets equal share of the residual.
    """
    if defensive is None:
        defensive = ["TLT", "GLD", "IEF"]
    if risk_off is None:
        risk_off = ["SPY", "VNQ"]

    n = len(tickers)
    lo, hi = bounds
    w = np.full(n, lo)

    # Classify
    def_idx  = [i for i, t in enumerate(tickers) if t in defensive]
    risk_idx = [i for i, t in enumerate(tickers) if t in risk_off]
    neut_idx = [i for i, t in enumerate(tickers)
                if t not in defensive and t not in risk_off]

    # Risk-off assets get floor
    for i in risk_idx:
        w[i] = lo

    # Residual distributed among defensive (equal split)
    residual = 1.0 - w.sum()
    if def_idx:
        per_def = residual / len(def_idx)
        for i in def_idx:
            w[i] = np.clip(per_def, lo, hi)

    # Neutral get whatever is left
    used = w.sum()
    if neut_idx and used < 1.0:
        per_neut = (1.0 - used) / len(neut_idx)
        for i in neut_idx:
            w[i] = np.clip(w[i] + per_neut, lo, hi)

    # Final normalise
    w = np.clip(w, lo, hi)
    return w / w.sum()


def regime_stats(regime: pd.Series, port_returns: pd.Series) -> dict:
    """Summary statistics split by regime label."""
    stats = {}
    for label in ("CALM", "STRESS"):
        mask = regime == label
        r = port_returns[mask]
        if r.empty:
            stats[label] = {}
            continue
        ann = r.mean() * 252
        vol = r.std() * np.sqrt(252)
        stats[label] = {
            "days": int(mask.sum()),
            "pct_of_sample": float(mask.mean()),
            "ann_return": float(ann),
            "ann_vol": float(vol),
            "sharpe": float(ann / vol) if vol > 0 else 0.0,
        }
    return stats
