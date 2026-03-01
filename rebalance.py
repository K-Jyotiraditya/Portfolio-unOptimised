"""Rebalancing logic: calendar, threshold, and regime-suppressed triggers.

A rebalance is triggered on a given day if ANY of these are true:
  1. Calendar: first trading day of the configured period (monthly, etc.)
  2. Threshold: any asset has drifted > rebal_threshold from target

Rebalance is SUPPRESSED if:
  - rebal_use_regime=True and current regime is STRESS

This allows the portfolio to ride through stress without forced selling
at distressed prices — a key differentiator from naive calendar rebalancing.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import Config

LOGGER = logging.getLogger("portfolio.rebal")


def _calendar_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """First trading day of each calendar period."""
    freq_map = {
        "daily":     "D",
        "weekly":    "W-MON",
        "monthly":   "MS",
        "quarterly": "QS",
    }
    period_starts = pd.date_range(
        index[0], index[-1], freq=freq_map[freq]
    )
    # map each period start to the nearest actual trading day >= start
    cal_days = []
    for ps in period_starts:
        future = index[index >= ps]
        if len(future):
            cal_days.append(future[0])
    return pd.DatetimeIndex(sorted(set(cal_days)))


def should_rebalance(date: pd.Timestamp,
                     w_actual: np.ndarray,
                     w_target: np.ndarray,
                     regime: str,
                     is_calendar_day: bool,
                     cfg: Config) -> bool:
    """True if a rebalance should execute on `date`."""
    # Regime suppression
    if cfg.rebal_use_regime and regime == "STRESS":
        return False

    # Calendar trigger
    if is_calendar_day:
        return True

    # Drift trigger
    drift = np.abs(w_actual - w_target)
    if drift.max() > cfg.rebal_threshold:
        return True

    return False


def compute_actual_weights(prices_today: np.ndarray,
                           prices_prev: np.ndarray,
                           w_prev: np.ndarray) -> np.ndarray:
    """Drift weights forward one period given price changes."""
    growth = prices_today / np.maximum(prices_prev, 1e-10)
    w_new = w_prev * growth
    total = w_new.sum()
    if total < 1e-10:
        return w_prev
    return w_new / total


def apply_costs(w_before: np.ndarray, w_after: np.ndarray,
                portfolio_value: float, cost_bps: float,
                min_trade_pct: float) -> float:
    """Return total transaction cost in dollar terms (deducted from NAV)."""
    trade = np.abs(w_after - w_before)
    # Ignore tiny trades
    trade = np.where(trade < min_trade_pct, 0.0, trade)
    total_turnover = trade.sum()   # one-way
    cost = portfolio_value * total_turnover * cost_bps * 1e-4
    return float(cost)
