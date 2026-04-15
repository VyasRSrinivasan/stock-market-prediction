import argparse
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class MarkovStockModel:
    n_states: int
    transition_matrix: np.ndarray
    state_bins: np.ndarray
    state_mean_returns: np.ndarray
    initial_state_counts: np.ndarray

    @classmethod
    def fit(cls, prices: pd.Series, n_states: int = 5) -> "MarkovStockModel":
        returns = prices.pct_change().dropna()
        if returns.empty:
            raise ValueError("Price series must contain at least two values.")

        quantiles = np.linspace(0, 1, n_states + 1)
        state_bins = np.quantile(returns, quantiles)

        # Assign each return to a discrete Markov state
        states = np.digitize(returns, bins=state_bins[1:-1], right=False)
        states = states.astype(int).ravel()

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

        state_mean_returns = np.zeros(n_states, dtype=float)
        for s in range(n_states):
            state_returns = returns.iloc[np.where(states == s)[0]]
            state_mean_returns[s] = float(state_returns.mean()) if len(state_returns) else 0.0

        initial_state_counts = np.bincount(states, minlength=n_states)

        return cls(
            n_states=n_states,
            transition_matrix=transition_matrix,
            state_bins=state_bins,
            state_mean_returns=state_mean_returns,
            initial_state_counts=initial_state_counts,
        )

    def state_for_return(self, ret: float) -> int:
        return int(np.digitize(float(ret), bins=self.state_bins[1:-1], right=False))

    def predict_next_state(self, current_state: int) -> int:
        probabilities = self.transition_matrix[current_state]
        if probabilities.sum() <= 0:
            return current_state
        return int(np.random.choice(self.n_states, p=probabilities))

    def simulate_prices(
        self,
        start_price: float,
        start_state: int,
        horizon: int = 10,
        random_seed: Optional[int] = None,
    ) -> pd.Series:
        if random_seed is not None:
            np.random.seed(random_seed)

        prices = [start_price]
        state = start_state
        for _ in range(horizon):
            next_state = self.predict_next_state(state)
            drift = self.state_mean_returns[next_state]
            next_price = prices[-1] * (1.0 + drift)
            prices.append(next_price)
            state = next_state

        return pd.Series(prices)

    def most_likely_next_state(self, current_state: int) -> int:
        probabilities = self.transition_matrix[current_state]
        return int(np.argmax(probabilities))


def load_prices_from_csv(csv_path: str, date_column: str = "Date", price_column: str = "Close") -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=[date_column])
    df = df.sort_values(date_column).dropna(subset=[price_column])
    return pd.Series(df[price_column].values, index=df[date_column])


def download_price_series(ticker: str, period: str = "1y") -> pd.Series:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for ticker downloads. Install it with `pip install yfinance`."
        ) from exc

    df = yf.download(ticker, period=period, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker} and period {period}.")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna().astype(float)


def build_markov_states(returns: pd.Series, n_states: int) -> pd.Series:
    boundaries = np.quantile(returns, np.linspace(0, 1, n_states + 1))
    return pd.Series(np.digitize(returns, bins=boundaries[1:-1], right=False), index=returns.index)


def print_model_summary(model: MarkovStockModel) -> None:
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


def main(args: argparse.Namespace) -> None:
    if args.csv:
        prices = load_prices_from_csv(args.csv, date_column=args.date_column, price_column=args.price_column)
    else:
        prices = download_price_series(args.ticker, args.period)

    model = MarkovStockModel.fit(prices, n_states=args.states)
    print_model_summary(model)

    returns = prices.pct_change().dropna()
    current_return = returns.iloc[-1]
    current_state = model.state_for_return(current_return)

    print(f"\nMost likely next state from current state {current_state}: {model.most_likely_next_state(current_state)}")
    print(f"Current state mean return: {model.state_mean_returns[current_state]:.5f}")

    simulation = model.simulate_prices(
        start_price=prices.iloc[-1], start_state=current_state, horizon=args.horizon, random_seed=args.seed
    )
    print("\nSimulated future price path:")
    print(simulation.round(2).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stochastic Markov chain stock prediction example.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", type=str, help="Ticker symbol for Yahoo Finance download.")
    group.add_argument("--csv", type=str, help="CSV file with historical prices.")
    parser.add_argument("--period", type=str, default="1y", help="Download period for ticker (e.g. 6mo, 1y, 2y).")
    parser.add_argument("--states", type=int, default=5, help="Number of Markov states to use.")
    parser.add_argument("--horizon", type=int, default=10, help="Days to simulate into the future.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for simulation.")
    parser.add_argument("--date-column", type=str, default="Date", help="CSV date column name.")
    parser.add_argument("--price-column", type=str, default="Close", help="CSV price column name.")
    args = parser.parse_args()
    main(args)
