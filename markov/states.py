"""
states.py
---------
Converts a continuous return series into discrete Markov states by dividing
the return distribution into quantile-based bins.

Each state represents a return "regime":
  state 0 → worst returns  (e.g. large losses)
  state N → best returns   (e.g. large gains)
"""

import numpy as np
import pandas as pd


def compute_state_bins(returns: pd.Series, n_states: int) -> np.ndarray:
    """Compute quantile-based bin edges that partition returns into n_states.

    Parameters
    ----------
    returns:  Series of daily percentage returns.
    n_states: Number of discrete states to create.

    Returns
    -------
    A 1-D array of (n_states + 1) bin edges from min to max return.
    """
    quantiles = np.linspace(0, 1, n_states + 1)
    return np.quantile(returns, quantiles)


def assign_states(returns: pd.Series, state_bins: np.ndarray) -> np.ndarray:
    """Map each return value to a discrete state index using bin edges.

    Parameters
    ----------
    returns:    Series of daily percentage returns.
    state_bins: Bin edges produced by compute_state_bins().

    Returns
    -------
    A 1-D integer array of state indices, one per return observation.
    """
    states = np.digitize(returns, bins=state_bins[1:-1], right=False)
    return states.astype(int).ravel()


def return_to_state(ret: float, state_bins: np.ndarray) -> int:
    """Map a single return value to its discrete state index.

    Parameters
    ----------
    ret:        A single daily percentage return (scalar).
    state_bins: Bin edges produced by compute_state_bins().

    Returns
    -------
    Integer state index.
    """
    return int(np.digitize(float(ret), bins=state_bins[1:-1], right=False))


def build_state_series(returns: pd.Series, n_states: int) -> pd.Series:
    """Convenience wrapper — returns a labelled Series of state indices.

    Parameters
    ----------
    returns:  Series of daily percentage returns.
    n_states: Number of discrete states to create.

    Returns
    -------
    A pd.Series of integer state indices sharing the same index as returns.
    """
    state_bins = compute_state_bins(returns, n_states)
    states = assign_states(returns, state_bins)
    return pd.Series(states, index=returns.index)
