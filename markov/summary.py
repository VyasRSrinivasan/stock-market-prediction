"""
summary.py
----------
Prints a human-readable summary of a fitted MarkovStockModel to stdout.
"""

import pandas as pd

from .model import MarkovStockModel


def print_model_summary(model: MarkovStockModel) -> None:
    """Print the fitted model parameters in a readable format.

    Displays:
      - Number of states
      - Quantile bin edges that define each state
      - Full transition probability matrix
      - Mean return observed in each state

    Parameters
    ----------
    model: A fitted MarkovStockModel instance.
    """
    print("Markov Stock Prediction Model")
    print("============================")
    print(f"Number of states: {model.n_states}")

    print("State bins (return quantiles):")
    for idx, edge in enumerate(model.state_bins):
        print(f"  bin {idx}: {edge:.5f}")

    print("\nTransition matrix:")
    print(pd.DataFrame(model.transition_matrix).round(3))

    print("\nMean return per state:")
    print(pd.Series(model.state_mean_returns).round(5))
