"""Configuration for the portfolio optimization + dynamic rebalancing engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class Config:
    # Universe
    tickers: Tuple[str, ...] = ("SPY", "TLT", "GLD", "IEF", "VNQ")
    start: str = "2007-01-01"
    end: str = "2024-12-31"

    # Data cache
    cache_path: str = "prices.pkl"
    cache_max_age_days: int = 7

    # Optimization
    risk_free_rate: float = 0.045          # annualised, used in Sharpe + BL
    target_vol: float = 0.10               # 10% annualised target vol (risk-targeting)
    min_weight: float = 0.02              # per-asset floor
    max_weight: float = 0.60              # per-asset cap
    lookback_days: int = 252              # estimation window for cov / returns

    # Black-Litterman
    bl_tau: float = 0.05                  # prior uncertainty scalar
    # Views injected at runtime via run(); leave empty for pure equilibrium
    bl_views: Dict = field(default_factory=dict)   # e.g. {"SPY": 0.08}

    # Rebalancing triggers
    rebal_calendar: str = "monthly"       # "daily" | "weekly" | "monthly" | "quarterly"
    rebal_threshold: float = 0.05         # drift threshold (|w_actual - w_target| > 5%)
    rebal_use_regime: bool = True         # suppress rebal in stress regime

    # Transaction costs
    cost_bps: float = 5.0                 # one-way cost in basis points per trade
    min_trade_pct: float = 0.005          # ignore trades smaller than 0.5% (avoid churn)

    # GARCH regime
    ewma_lambda: float = 0.94
    stress_vol_threshold: float = 0.20    # annualised vol above which = stress regime

    # Reporting
    dpi: int = 150
    bench_ticker: str = "SPY"             # benchmark for alpha / tracking error

    def __post_init__(self) -> None:
        if len(self.tickers) < 2:
            raise ValueError("Need at least 2 tickers")
        if not (0.0 <= self.min_weight < self.max_weight <= 1.0):
            raise ValueError("min_weight / max_weight bounds invalid")
        if not (0.0 < self.ewma_lambda < 1.0):
            raise ValueError("ewma_lambda must be in (0, 1)")
        if not (0 < self.lookback_days):
            raise ValueError("lookback_days must be positive")
        if self.rebal_calendar not in ("daily", "weekly", "monthly", "quarterly"):
            raise ValueError("rebal_calendar must be daily/weekly/monthly/quarterly")
