"""
simulation.py
-------------
Simulates future price paths by stepping through Markov state transitions
and applying each state's mean return to compound a starting price.
"""

from typing import Optional

import numpy as np
import pandas as pd


def predict_next_state(
    current_state: int,
    transition_matrix: np.ndarray,
    n_states: int,
) -> int:
    """Sample the next state from the transition probability distribution.

    Parameters
    ----------
    current_state:     The current discrete state index.
    transition_matrix: (n_states x n_states) probability matrix.
    n_states:          Total number of discrete states.

    Returns
    -------
    Integer index of the sampled next state. If the current row has no
    probability mass (all zeros), the current state is returned unchanged.
    """
    probabilities = transition_matrix[current_state]
    if probabilities.sum() <= 0:
        return current_state
    return int(np.random.choice(n_states, p=probabilities))


def most_likely_next_state(
    current_state: int,
    transition_matrix: np.ndarray,
) -> int:
    """Return the state with the highest transition probability from current_state.

    Parameters
    ----------
    current_state:     The current discrete state index.
    transition_matrix: (n_states x n_states) probability matrix.

    Returns
    -------
    Integer index of the most probable next state.
    """
    return int(np.argmax(transition_matrix[current_state]))


def simulate_prices(
    start_price: float,
    start_state: int,
    transition_matrix: np.ndarray,
    state_mean_returns: np.ndarray,
    n_states: int,
    horizon: int = 10,
    random_seed: Optional[int] = None,
) -> pd.Series:
    """Simulate a future price path by chaining Markov state transitions.

    At each step the model:
      1. Samples the next state from the transition distribution.
      2. Compounds the current price by (1 + mean_return[next_state]).

    Parameters
    ----------
    start_price:        Most recent observed closing price.
    start_state:        State index corresponding to the most recent return.
    transition_matrix:  (n_states x n_states) probability matrix.
    state_mean_returns: Mean return for each state (length n_states).
    n_states:           Total number of discrete states.
    horizon:            Number of future days to simulate.
    random_seed:        Optional seed for reproducibility.

    Returns
    -------
    A pd.Series of length (horizon + 1) starting with start_price.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    prices = [float(start_price)]
    state = start_state
    for _ in range(horizon):
        next_state = predict_next_state(state, transition_matrix, n_states)
        drift = state_mean_returns[next_state]
        prices.append(prices[-1] * (1.0 + drift))
        state = next_state

    return pd.Series(prices, dtype=float)
