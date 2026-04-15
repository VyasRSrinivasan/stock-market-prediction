"""
markov — Markov chain stock prediction package
===============================================

Modules
-------
preprocessing   Load and clean historical price data (CSV or Yahoo Finance).
states          Map continuous returns to discrete Markov states.
transition      Build the transition matrix and per-state statistics.
simulation      Simulate future price paths via state transitions.
model           MarkovStockModel: high-level fit/predict/simulate API.
summary         Print a human-readable model summary.
"""

from .model import MarkovStockModel
from .preprocessing import load_prices_from_csv, download_price_series
from .states import compute_state_bins, assign_states, return_to_state, build_state_series
from .transition import compute_transition_matrix, compute_state_mean_returns, compute_initial_state_counts
from .simulation import predict_next_state, most_likely_next_state, simulate_prices
from .summary import print_model_summary

__all__ = [
    "MarkovStockModel",
    "load_prices_from_csv",
    "download_price_series",
    "compute_state_bins",
    "assign_states",
    "return_to_state",
    "build_state_series",
    "compute_transition_matrix",
    "compute_state_mean_returns",
    "compute_initial_state_counts",
    "predict_next_state",
    "most_likely_next_state",
    "simulate_prices",
    "print_model_summary",
]
