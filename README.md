# Stock Market Predictions

## Problem

Forecasting stock market movements is inherently uncertain. This project uses a simple stochastic Markov chain model on historical daily returns to explore how discrete state transitions can be used for short-term price simulation.

## Objective

- Fit a Markov chain to stock return states.
- Estimate a transition matrix for return state changes.
- Simulate future price paths from the learned chain.
- Provide a lightweight example for educational stock prediction analysis.

## Dataset

- Supports loading historical prices from a local CSV file.
- Supports downloading stock price data from Yahoo Finance using `yfinance`.
- Expects a CSV with `Date` and `Close` columns by default.

## Tech Stack

- Python 3
- NumPy
- pandas
- yfinance

## Project Structure

- `markov_stock_prediction.py` – Main implementation of the Markov chain model, fit and simulation logic, plus command-line interface.
- `requirements.txt` – Python dependency list.
- `README.md` – Project overview and usage instructions.

## Usage

Install requirements:

```bash
pip install -r requirements.txt
```

Run with a ticker symbol:

```bash
python markov_stock_prediction.py --ticker AAPL --period 1y --states 5 --horizon 10
```

Run with a CSV file:

```bash
python markov_stock_prediction.py --csv data/prices.csv --date-column Date --price-column Close --states 5 --horizon 10
```

## Parameters

- `--ticker` – Stock ticker for Yahoo Finance download.
- `--csv` – Path to local CSV file containing historical prices.
- `--period` – Download period for ticker data (default: `1y`).
- `--states` – Number of discrete Markov return states (default: `5`).
- `--horizon` – Number of days to simulate into the future (default: `10`).
- `--seed` – Random seed for reproducible simulation (default: `42`).
- `--date-column` – CSV date column name (default: `Date`).
- `--price-column` – CSV price column name (default: `Close`).

## Notes

- This code is intended for demonstration and research, not financial advice.
- Markov models are a simplified way to analyze market state transitions and do not guarantee predictive accuracy.

## References

- Markov chain models in finance
- `yfinance` for stock price download
- NumPy and pandas for data preparation and numerical modeling

