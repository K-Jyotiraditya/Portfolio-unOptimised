"""Price data: download via yfinance, cache locally, compute log returns."""
from __future__ import annotations

import logging
import os
import pickle
import time

import pandas as pd
import yfinance as yf

from config import Config

LOGGER = logging.getLogger("portfolio.data")


def load_prices(cfg: Config) -> pd.DataFrame:
    """Return adjusted-close prices for all tickers; cache to disk."""
    path = cfg.cache_path
    stale = True
    if os.path.exists(path):
        age_days = (time.time() - os.path.getmtime(path)) / 86400
        if age_days <= cfg.cache_max_age_days:
            stale = False

    if not stale:
        with open(path, "rb") as f:
            prices: pd.DataFrame = pickle.load(f)
        LOGGER.info("Loaded prices from cache (%s)", path)
    else:
        LOGGER.info("Downloading prices %s -> %s for %s",
                    cfg.start, cfg.end, cfg.tickers)
        raw = yf.download(
            list(cfg.tickers), start=cfg.start, end=cfg.end,
            auto_adjust=True, progress=False,
        )
        prices = raw["Close"].dropna(how="all")
        if prices.empty:
            raise RuntimeError("yfinance returned no data — check tickers / dates")
        prices = prices[list(cfg.tickers)].dropna()
        with open(path, "wb") as f:
            pickle.dump(prices, f)
        LOGGER.info("Downloaded and cached %d rows", len(prices))

    missing = [t for t in cfg.tickers if t not in prices.columns]
    if missing:
        raise RuntimeError(f"Tickers not in price data: {missing}")
    return prices[list(cfg.tickers)].dropna()


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns, first row dropped."""
    return prices.apply(lambda s: s.pct_change()).dropna()


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns for compounding math."""
    import numpy as np
    return prices.apply(lambda s: np.log(s / s.shift(1))).dropna()


def load_risk_free_rate(cfg: Config, price_index: pd.DatetimeIndex) -> pd.Series:
    """Daily risk-free rate (decimal) from 3-month T-bill (^IRX).

    ^IRX is quoted as annualised percentage (e.g. 5.25 = 5.25%).
    Converts to daily decimal by dividing by 100*252, then forward-fills
    gaps (weekends / holidays) to align with the equity return dates.
    Falls back to cfg.risk_free_rate if download fails.
    """
    rf_cache = cfg.cache_path.replace("prices.pkl", "rf_rate.pkl")
    stale = True
    if os.path.exists(rf_cache):
        age_days = (time.time() - os.path.getmtime(rf_cache)) / 86400
        if age_days <= cfg.cache_max_age_days:
            stale = False

    if not stale:
        with open(rf_cache, "rb") as f:
            irx = pickle.load(f)
        if isinstance(irx, pd.DataFrame):
            irx = irx.squeeze()
        irx = irx.dropna()
        LOGGER.info("Loaded risk-free rate from cache")
    else:
        try:
            raw = yf.download("^IRX", start=cfg.start, end=cfg.end,
                              auto_adjust=True, progress=False)
            irx = raw["Close"].squeeze().dropna()
            with open(rf_cache, "wb") as f:
                pickle.dump(irx, f)
            LOGGER.info("Downloaded T-bill rate (%d rows)", len(irx))
        except Exception as exc:
            LOGGER.warning("T-bill download failed (%s) — using flat %.1f%%", exc,
                           cfg.risk_free_rate * 100)
            return pd.Series(cfg.risk_free_rate / 252, index=price_index)

    # Convert annualised % to daily decimal, align to equity dates
    daily_rf = (irx / 100 / 252).reindex(price_index).ffill()
    daily_rf = daily_rf.fillna(cfg.risk_free_rate / 252)
    return daily_rf


def rolling_cov(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualised covariance over trailing window (252 scaling)."""
    return returns.iloc[-window:].cov() * 252


def sample_stats(returns: pd.DataFrame, window: int):
    """(mu_ann, cov_ann) over the trailing window."""
    r = returns.iloc[-window:]
    mu = r.mean() * 252
    cov = r.cov() * 252
    return mu, cov
