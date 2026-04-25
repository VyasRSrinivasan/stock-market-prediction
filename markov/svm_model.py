"""
svm_model.py — SVM RBF classifier for next-state prediction and price simulation
=================================================================================

Trains a Support Vector Machine with an RBF (Gaussian) kernel to predict the
next trading day's return state from engineered features of recent price history.

Unlike the Markov chain (which uses only the current state) and Monte Carlo
(which samples from a fixed GBM distribution), the SVM captures non-linear
relationships across a multi-day feature window.

Features used to predict state at time t
-----------------------------------------
- lag_1 … lag_5  : log-returns r_{t-1} … r_{t-5}
- roll_mean_5    : mean log-return over the prior 5 days
- roll_mean_10   : mean log-return over the prior 10 days
- roll_std_5     : return volatility over the prior 5 days
- momentum       : lag_1 − roll_mean_5

All features are standardised inside a sklearn Pipeline so the RBF kernel
distances are well-conditioned.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_N_LAGS = 5
_WINDOW = 10  # rolling window size; minimum past returns needed for inference


def _feature_names() -> list[str]:
    return [f"lag_{i}" for i in range(1, _N_LAGS + 1)] + [
        "roll_mean_5", "roll_mean_10", "roll_std_5", "momentum"
    ]


def _build_feature_matrix(log_returns: pd.Series) -> pd.DataFrame:
    """Return a DataFrame of features aligned to the same index as *log_returns*.

    Each row at time t contains only information available *before* t so there
    is no look-ahead bias.
    """
    df = pd.DataFrame({"r": log_returns})
    for lag in range(1, _N_LAGS + 1):
        df[f"lag_{lag}"] = df["r"].shift(lag)
    df["roll_mean_5"] = df["r"].shift(1).rolling(5).mean()
    df["roll_mean_10"] = df["r"].shift(1).rolling(10).mean()
    df["roll_std_5"] = df["r"].shift(1).rolling(5).std().clip(lower=1e-8)
    df["momentum"] = df["lag_1"] - df["roll_mean_5"]
    return df.dropna()


def _row_from_window(window: np.ndarray) -> np.ndarray:
    """Build a single feature row from the last _WINDOW log-returns (oldest first)."""
    lags = window[-1: -_N_LAGS - 1: -1]   # [r_{t-1}, r_{t-2}, ..., r_{t-5}]
    r5 = window[-5:]
    r10 = window[-_WINDOW:]
    roll_mean_5 = float(r5.mean())
    roll_mean_10 = float(r10.mean())
    roll_std_5 = max(float(r5.std()), 1e-8)
    momentum = float(window[-1]) - roll_mean_5
    return np.array([*lags, roll_mean_5, roll_mean_10, roll_std_5, momentum])


def train_svm(
    prices: pd.Series,
    state_bins: np.ndarray,
    n_states: int,
) -> tuple:
    """Train a StandardScaler + SVC(kernel='rbf') pipeline on historical data.

    Parameters
    ----------
    prices:      Historical closing price series.
    state_bins:  Bin edges from the Markov model (shared bucketing scheme).
    n_states:    Number of discrete states.

    Returns
    -------
    (pipeline, n_train_samples)
    """
    try:
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise ImportError(
            "The SVM model requires scikit-learn. "
            "Install it with: pip install scikit-learn"
        ) from exc

    from .states import assign_states

    log_returns = np.log(prices / prices.shift(1)).dropna()
    feat_df = _build_feature_matrix(log_returns)

    all_states = assign_states(log_returns, state_bins)
    state_series = pd.Series(all_states, index=log_returns.index)
    y = state_series.loc[feat_df.index].values
    X = feat_df[_feature_names()].values

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    clf.fit(X, y)
    return clf, len(X)


def predict_next_state_probs(clf, prices: pd.Series) -> np.ndarray:
    """Return probability distribution over next states given the most recent real data.

    Returns
    -------
    np.ndarray of shape (n_states,) — probabilities indexed by state label order
    from clf.classes_.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna().values.astype(float)
    window = log_returns[-_WINDOW:]
    feat = _row_from_window(window).reshape(1, -1)
    return clf.predict_proba(feat)[0]


def simulate_svm_prices(
    clf,
    prices: pd.Series,
    state_mean_returns: np.ndarray,
    horizon: int,
    random_seed: int = 42,
) -> pd.Series:
    """Simulate a price path by rolling the SVM forward step by step.

    At each step, the SVM predicts a probability distribution over states
    from the current feature window (seeded with real history, then extended
    with simulated log-returns).  A state is sampled from that distribution
    and the state's mean return is applied to the price.

    Returns
    -------
    pd.Series of length horizon+1 (day 0 = last known closing price).
    """
    log_returns = np.log(prices / prices.shift(1)).dropna().values.astype(float)
    window = log_returns[-_WINDOW:].copy()

    rng = np.random.default_rng(random_seed)
    path = [float(prices.iloc[-1])]

    for _ in range(horizon):
        feat = _row_from_window(window).reshape(1, -1)
        probs = clf.predict_proba(feat)[0]
        next_state = int(rng.choice(len(probs), p=probs))
        daily_ret = float(state_mean_returns[next_state])
        path.append(path[-1] * (1.0 + daily_ret))
        sim_log_ret = np.log1p(daily_ret) if daily_ret > -1.0 else 0.0
        window = np.append(window[1:], sim_log_ret)

    return pd.Series(path, dtype=float)
