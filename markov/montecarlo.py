"""
montecarlo.py — Monte Carlo simulation with ML-estimated drift
==============================================================

Uses Geometric Brownian Motion (GBM) to simulate N independent price paths:

    S(t+1) = S(t) * exp((μ - 0.5σ²) + σ * Z)    Z ~ N(0,1)

Drift sources
-------------
- **OLS drift** (default): linear regression on log-prices captures the
  recent price trend.  This is the baseline ML estimate.
- **SVM-conditioned drift** (when svm_probs and state_mean_returns are
  supplied): the expected daily return under the SVM's predicted regime
  distribution — Σ P(state=i) * mean_return(state=i).  This makes the
  Monte Carlo regime-aware: if the SVM detects a bearish regime, the drift
  shifts down; if it detects a bullish regime, the drift shifts up.

Volatility (σ) always uses the standard deviation of daily log-returns.
Percentile bands (10/25/50/75/90) are computed across all paths at every
time step and returned for plotting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _estimate_drift_ols(prices: pd.Series) -> float:
    """Estimate daily log-return drift via OLS on log prices (closed-form)."""
    log_prices = np.log(prices.values.astype(float))
    t = np.arange(len(log_prices), dtype=float)
    t_mean = t.mean()
    y_mean = log_prices.mean()
    b = float(np.dot(t - t_mean, log_prices - y_mean) / np.dot(t - t_mean, t - t_mean))
    return b


def svm_expected_drift(
    svm_probs: np.ndarray,
    state_mean_returns: np.ndarray,
) -> float:
    """Compute the SVM-conditioned expected daily drift.

    Parameters
    ----------
    svm_probs:           Probability distribution over states from the SVM
                         (shape: n_states).
    state_mean_returns:  Mean daily return for each state from the Markov
                         model (shape: n_states).

    Returns
    -------
    float — weighted average daily return = Σ P(state=i) * mean_return(i).
    """
    return float(np.dot(svm_probs, state_mean_returns))


def run_monte_carlo(
    prices: pd.Series,
    horizon: int,
    n_simulations: int = 500,
    random_seed: int = 42,
    svm_probs: np.ndarray | None = None,
    state_mean_returns: np.ndarray | None = None,
) -> dict:
    """Simulate *n_simulations* GBM price paths over *horizon* trading days.

    When *svm_probs* and *state_mean_returns* are both provided, the drift
    is replaced with the SVM-conditioned expected return, making the
    simulation sensitive to the current detected market regime.  Otherwise
    the OLS trend estimate is used.

    Parameters
    ----------
    prices:              Historical closing price series.
    horizon:             Number of trading days to simulate.
    n_simulations:       Number of independent paths to generate.
    random_seed:         Seed for reproducibility.
    svm_probs:           (optional) SVM next-state probability array.
    state_mean_returns:  (optional) Mean daily return per state.

    Returns
    -------
    dict with keys:
        bands              — {percentile: np.ndarray of shape (horizon+1,)}.
        drift_daily        — Drift used in the simulation.
        drift_ols          — OLS baseline drift (always computed).
        drift_source       — "SVM-conditioned" or "OLS regression".
        sigma_daily        — Historical daily volatility.
        n_simulations      — Number of paths simulated.
        median_end         — Median end price across all paths.
        mean_end           — Mean end price across all paths.
        p10_end            — 10th-percentile end price.
        p90_end            — 90th-percentile end price.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna().astype(float)
    sigma = float(log_returns.std())
    drift_ols = _estimate_drift_ols(prices)

    if svm_probs is not None and state_mean_returns is not None:
        drift = svm_expected_drift(svm_probs, state_mean_returns)
        drift_source = "SVM-conditioned"
    else:
        drift = drift_ols
        drift_source = "OLS regression"

    rng = np.random.default_rng(random_seed)
    S0 = float(prices.iloc[-1])

    Z = rng.standard_normal((n_simulations, horizon))
    shocks = np.exp((drift - 0.5 * sigma ** 2) + sigma * Z)

    paths = np.empty((n_simulations, horizon + 1))
    paths[:, 0] = S0
    for t in range(horizon):
        paths[:, t + 1] = paths[:, t] * shocks[:, t]

    bands = {p: np.percentile(paths, p, axis=0) for p in (10, 25, 50, 75, 90)}
    end_prices = paths[:, -1]

    return {
        "bands": bands,
        "drift_daily": drift,
        "drift_ols": drift_ols,
        "drift_source": drift_source,
        "sigma_daily": sigma,
        "n_simulations": n_simulations,
        "median_end": float(np.median(end_prices)),
        "mean_end": float(np.mean(end_prices)),
        "p10_end": float(np.percentile(end_prices, 10)),
        "p90_end": float(np.percentile(end_prices, 90)),
    }
