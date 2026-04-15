"""
preprocessing.py
----------------
Handles loading and cleaning historical price data from either a local CSV
file or a live Yahoo Finance download.
"""

import pandas as pd


def load_prices_from_csv(
    csv_path: str,
    date_column: str = "Date",
    price_column: str = "Close",
) -> pd.Series:
    """Load closing prices from a CSV file.

    The CSV must contain at minimum a date column and a numeric price column.
    Rows with missing prices are dropped, and the result is sorted by date.

    Parameters
    ----------
    csv_path:     Path to the CSV file.
    date_column:  Name of the column containing dates (default: "Date").
    price_column: Name of the column containing prices (default: "Close").

    Returns
    -------
    A float-typed pd.Series indexed by date.
    """
    df = pd.read_csv(csv_path, parse_dates=[date_column])
    df = df.sort_values(date_column).dropna(subset=[price_column])
    return pd.Series(df[price_column].values, index=df[date_column], dtype=float)


def download_price_series(ticker: str, period: str = "1y") -> pd.Series:
    """Download closing prices from Yahoo Finance via yfinance.

    Parameters
    ----------
    ticker: Stock ticker symbol (e.g. "AAPL").
    period: Lookback period string accepted by yfinance (e.g. "6mo", "1y", "2y").

    Returns
    -------
    A float-typed pd.Series of closing prices indexed by date.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for ticker downloads. "
            "Install it with `pip install yfinance`."
        ) from exc

    df = yf.download(ticker, period=period, progress=False)
    if df.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' and period '{period}'."
        )

    close = df["Close"]
    # Newer yfinance versions may return a DataFrame with a multi-level column
    # index even for a single ticker download.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    return close.dropna().astype(float)
