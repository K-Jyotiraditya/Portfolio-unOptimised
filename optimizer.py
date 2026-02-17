"""Portfolio optimization: MVO, Black-Litterman, momentum tilt.

Covariance estimation:
  - ledoit_wolf_cov: shrinks sample covariance toward scaled identity.
    Reduces estimation error for small T/N ratios (252 obs, 5 assets).

Momentum overlay:
  - momentum_scores: 12-1 month cross-sectional momentum per asset.
  - momentum_tilt: rescales min-variance weights toward momentum winners.
    Blending factor alpha controls tilt strength (0 = pure min-var, 1 = pure momentum).
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

LOGGER = logging.getLogger("portfolio.opt")

_EPS = 1e-10


# --------------------------------------------------------------------------- #
# Core MVO
# --------------------------------------------------------------------------- #

def _portfolio_stats(w: np.ndarray, mu: np.ndarray,
                     cov: np.ndarray, rf: float) -> Tuple[float, float, float]:
    """(expected_return, volatility, sharpe) — all annualised."""
    w = np.asarray(w, float)
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - rf) / (vol + _EPS)
    return ret, vol, sharpe


def max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float,
               bounds: Tuple[float, float] = (0.02, 0.60)) -> np.ndarray:
    """Maximum Sharpe ratio weights."""
    n = len(mu)
    w0 = np.full(n, 1.0 / n)

    def neg_sharpe(w):
        r, v, _ = _portfolio_stats(w, mu, cov, rf)
        return -(r - rf) / (v + _EPS)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [bounds] * n
    res = minimize(neg_sharpe, w0, method="SLSQP",
                   bounds=bnds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter": 1000})
    if not res.success:
        LOGGER.warning("max_sharpe did not converge: %s", res.message)
    w = np.clip(res.x, 0, 1)
    return w / w.sum()


def min_variance(mu: np.ndarray, cov: np.ndarray,
                 bounds: Tuple[float, float] = (0.02, 0.60)) -> np.ndarray:
    """Global minimum variance weights."""
    n = len(mu)
    w0 = np.full(n, 1.0 / n)

    def port_var(w):
        return float(w @ cov @ w)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bnds = [bounds] * n
    res = minimize(port_var, w0, method="SLSQP",
                   bounds=bnds, constraints=constraints,
                   options={"ftol": 1e-9, "maxiter": 1000})
    if not res.success:
        LOGGER.warning("min_variance did not converge: %s", res.message)
    w = np.clip(res.x, 0, 1)
    return w / w.sum()


def risk_target_weights(w_sharpe: np.ndarray, cov: np.ndarray,
                        target_vol: float) -> np.ndarray:
    """Scale max-Sharpe weights so portfolio vol == target_vol.

    If the max-Sharpe portfolio already exceeds target_vol it is de-levered;
    below target_vol it is left at 1x (no leverage assumed).
    """
    port_vol = float(np.sqrt(w_sharpe @ cov @ w_sharpe))
    if port_vol < _EPS:
        return w_sharpe
    scale = min(target_vol / port_vol, 1.0)   # no leverage
    w_scaled = w_sharpe * scale
    cash = 1.0 - w_scaled.sum()
    # cash stays in risk-free; for weight reporting we re-normalise to 1
    if cash > 0:
        return w_sharpe   # already below target — return unscaled
    return w_scaled / w_scaled.sum()


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, rf: float,
                       bounds: Tuple[float, float] = (0.02, 0.60),
                       n_points: int = 60) -> pd.DataFrame:
    """Sweep target returns from min-var to max-mu and solve for min-vol.

    Returns DataFrame with columns: ret, vol, sharpe.
    """
    n = len(mu)
    w_mv = min_variance(mu, cov, bounds)
    ret_min = float(w_mv @ mu)
    ret_max = float(mu.max())
    targets = np.linspace(ret_min, ret_max, n_points)

    rows = []
    for tgt in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
            {"type": "eq", "fun": lambda w, t=tgt: float(w @ mu) - t},
        ]
        res = minimize(lambda w: float(w @ cov @ w),
                       np.full(n, 1 / n), method="SLSQP",
                       bounds=[bounds] * n, constraints=constraints,
                       options={"ftol": 1e-9, "maxiter": 500})
        if res.success:
            w = np.clip(res.x, 0, 1); w /= w.sum()
            v = float(np.sqrt(w @ cov @ w))
            r = float(w @ mu)
            rows.append({"ret": r, "vol": v, "sharpe": (r - rf) / (v + _EPS)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Black-Litterman
# --------------------------------------------------------------------------- #

def black_litterman(mu_hist: np.ndarray, cov: np.ndarray,
                    w_mkt: np.ndarray,
                    views: Dict[str, float],
                    tickers,
                    tau: float = 0.05,
                    delta: float = 2.5,
                    rf: float = 0.045) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mu_BL, cov_BL) after blending market equilibrium with views.

    views: dict mapping ticker -> absolute expected annual return
           e.g. {"SPY": 0.08, "TLT": 0.03}
    """
    n = len(tickers)
    ticker_list = list(tickers)

    # Implied equilibrium returns
    pi = delta * cov @ w_mkt

    if not views:
        return pi, tau * cov

    # Build pick matrix P and view vector q
    k = len(views)
    P = np.zeros((k, n))
    q = np.zeros(k)
    for i, (ticker, view_ret) in enumerate(views.items()):
        if ticker not in ticker_list:
            LOGGER.warning("BL view ticker '%s' not in universe — skipped", ticker)
            continue
        j = ticker_list.index(ticker)
        P[i, j] = 1.0
        q[i] = view_ret

    # Uncertainty on views: proportional to prior uncertainty
    Omega = np.diag(np.diag(P @ (tau * cov) @ P.T))

    # BL posterior
    tau_cov = tau * cov
    M = np.linalg.inv(np.linalg.inv(tau_cov) + P.T @ np.linalg.inv(Omega) @ P)
    mu_bl = M @ (np.linalg.inv(tau_cov) @ pi + P.T @ np.linalg.inv(Omega) @ q)
    cov_bl = cov + M   # posterior uncertainty

    return mu_bl, cov_bl


def equal_weight(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def risk_parity(cov: np.ndarray,
                bounds: Tuple[float, float] = (0.02, 0.60)) -> np.ndarray:
    """Equal Risk Contribution (ERC) weights via convex approximation."""
    n = cov.shape[0]
    w0 = np.full(n, 1.0 / n)

    def rc_objective(w):
        sigma = np.sqrt(w @ cov @ w)
        mrc = cov @ w / (sigma + _EPS)
        rc = w * mrc
        target = sigma / n
        return float(np.sum((rc - target) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    res = minimize(rc_objective, w0, method="SLSQP",
                   bounds=[bounds] * n, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 2000})
    w = np.clip(res.x, 0, 1)
    return w / w.sum()


# --------------------------------------------------------------------------- #
# Covariance shrinkage
# --------------------------------------------------------------------------- #

def ledoit_wolf_cov(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance (annualised).

    Shrinks sample cov toward a scaled identity matrix, dramatically
    reducing estimation error when T (observations) is only moderately
    larger than N (assets). Returns annualised covariance matrix.
    """
    lw = LedoitWolf().fit(returns.to_numpy())
    return lw.covariance_ * 252


# --------------------------------------------------------------------------- #
# Momentum overlay
# --------------------------------------------------------------------------- #

def momentum_scores(prices: pd.DataFrame,
                    lookback: int = 252,
                    skip: int = 21) -> np.ndarray:
    """Cross-sectional momentum: (price[t-skip] / price[t-lookback]) - 1.

    skip=21 excludes the most recent month (short-term reversal noise).
    Returns z-scored momentum ranks across assets.
    """
    if len(prices) < lookback + skip:
        return np.zeros(len(prices.columns))
    p_start = prices.iloc[-(lookback + skip)]
    p_end   = prices.iloc[-skip]
    raw_mom = (p_end / p_start - 1.0).to_numpy()
    std = raw_mom.std()
    if std < _EPS:
        return np.zeros(len(raw_mom))
    return (raw_mom - raw_mom.mean()) / std


def momentum_tilt(w_minvar: np.ndarray, mom: np.ndarray,
                  alpha: float = 0.30,
                  bounds: Tuple[float, float] = (0.02, 0.60)) -> np.ndarray:
    """Blend min-variance weights with momentum-proportional weights.

    alpha=0   -> pure min-variance
    alpha=0.3 -> 30% momentum tilt (default: modest, robust)

    Momentum weights are computed by shifting z-scores to be non-negative
    then normalising to sum to 1.
    """
    mom_pos = np.maximum(mom + np.abs(mom.min()) + _EPS, 0.0)
    mom_w = mom_pos / (mom_pos.sum() + _EPS)

    blended = (1 - alpha) * w_minvar + alpha * mom_w
    blended = np.clip(blended, bounds[0], bounds[1])
    return blended / blended.sum()
