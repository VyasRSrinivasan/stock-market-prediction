import argparse

from markov import (
    MarkovStockModel,
    load_prices_from_csv,
    download_price_series,
    print_model_summary,
)


def main(args: argparse.Namespace) -> None:
    if args.csv:
        prices = load_prices_from_csv(
            args.csv,
            date_column=args.date_column,
            price_column=args.price_column,
        )
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
        start_price=float(prices.iloc[-1]),
        start_state=current_state,
        horizon=args.horizon,
        random_seed=args.seed,
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
