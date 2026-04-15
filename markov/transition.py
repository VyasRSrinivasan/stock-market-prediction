"""
transition.py
-------------
Builds the Markov transition matrix and supporting statistics from a
sequence of discrete state assignments.

The transition matrix T[i][j] is the empirical probability of moving
from state i to state j on the next time step.
"""

import numpy as np
import pandas as pd


def compute_transition_matrix(states: np.ndarray, n_states: int) -> np.ndarray:
    """Count state-to-state transitions and normalize each row to probabilities.

    Parameters
    ----------
    states:   1-D integer array of state indices (chronological order).
    n_states: Total number of discrete states.

    Returns
    -------
    A (n_states x n_states) float array where row i sums to 1.0.
    Rows with no observed transitions are left as all-zeros.
    """
    transition_counts = np.zeros((n_states, n_states), dtype=float)
    for src, dst in zip(states[:-1], states[1:]):
        transition_counts[src, dst] += 1.0

    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(
        transition_counts,
        row_sums,
        out=np.zeros_like(transition_counts),
        where=row_sums != 0,
    )
    return transition_matrix


def compute_state_mean_returns(
    returns: pd.Series, states: np.ndarray, n_states: int
) -> np.ndarray:
    """Compute the average observed return for each state.

    Parameters
    ----------
    returns:  Series of daily percentage returns aligned with states.
    states:   1-D integer array of state indices (same length as returns).
    n_states: Total number of discrete states.

    Returns
    -------
    A 1-D float array of length n_states. States with no observations get 0.0.
    """
    state_mean_returns = np.zeros(n_states, dtype=float)
    for s in range(n_states):
        mask = np.where(states == s)[0]
        state_returns = returns.iloc[mask]
        state_mean_returns[s] = float(state_returns.mean()) if len(state_returns) else 0.0
    return state_mean_returns


def compute_initial_state_counts(states: np.ndarray, n_states: int) -> np.ndarray:
    """Count how many times each state appears in the historical sequence.

    Parameters
    ----------
    states:   1-D integer array of state indices.
    n_states: Total number of discrete states.

    Returns
    -------
    A 1-D integer array of length n_states with observation counts.
    """
    return np.bincount(states, minlength=n_states)
