"""
model.py
--------
Defines MarkovStockModel — a dataclass that holds all fitted parameters and
exposes a high-level API for fitting, state lookup, and simulation.

Internally delegates to the specialised modules:
  preprocessing  →  data loading
  states         →  return-to-state mapping
  transition     →  transition matrix and statistics
  simulation     →  future price path generation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .states import compute_state_bins, assign_states, return_to_state
from .transition import (
    compute_transition_matrix,
    compute_state_mean_returns,
    compute_initial_state_counts,
)
from .simulation import predict_next_state, most_likely_next_state, simulate_prices


@dataclass
class MarkovStockModel:
    """A fitted Markov chain model for stock return state transitions.

    Attributes
    ----------
    n_states:             Number of discrete return states.
    transition_matrix:    (n_states x n_states) row-stochastic probability matrix.
    state_bins:           Quantile-based bin edges used to map returns to states.
    state_mean_returns:   Average historical return observed in each state.
    initial_state_counts: Raw count of observations per state.
    """

    n_states: int
    transition_matrix: np.ndarray
    state_bins: np.ndarray
    state_mean_returns: np.ndarray
    initial_state_counts: np.ndarray

    @classmethod
    def fit(cls, prices: pd.Series, n_states: int = 5) -> MarkovStockModel:
        """Fit a Markov chain model to a price series.

        Parameters
        ----------
        prices:   Chronologically ordered closing price series.
        n_states: Number of discrete return states to use.

        Returns
        -------
        A fitted MarkovStockModel instance.
        """
        returns = prices.pct_change().dropna()
        if returns.empty:
            raise ValueError("Price series must contain at least two values.")

        state_bins = compute_state_bins(returns, n_states)
        states = assign_states(returns, state_bins)
        transition_matrix = compute_transition_matrix(states, n_states)
        state_mean_returns = compute_state_mean_returns(returns, states, n_states)
        initial_state_counts = compute_initial_state_counts(states, n_states)

        return cls(
            n_states=n_states,
            transition_matrix=transition_matrix,
            state_bins=state_bins,
            state_mean_returns=state_mean_returns,
            initial_state_counts=initial_state_counts,
        )

    def state_for_return(self, ret: float) -> int:
        """Map a single return value to its discrete state index."""
        return return_to_state(ret, self.state_bins)

    def predict_next_state(self, current_state: int) -> int:
        """Randomly sample the next state from the transition distribution."""
        return predict_next_state(current_state, self.transition_matrix, self.n_states)

    def most_likely_next_state(self, current_state: int) -> int:
        """Return the state with the highest transition probability."""
        return most_likely_next_state(current_state, self.transition_matrix)

    def simulate_prices(
        self,
        start_price: float,
        start_state: int,
        horizon: int = 10,
        random_seed: Optional[int] = None,
    ) -> pd.Series:
        """Simulate a future price path over the given horizon."""
        return simulate_prices(
            start_price=start_price,
            start_state=start_state,
            transition_matrix=self.transition_matrix,
            state_mean_returns=self.state_mean_returns,
            n_states=self.n_states,
            horizon=horizon,
            random_seed=random_seed,
        )
