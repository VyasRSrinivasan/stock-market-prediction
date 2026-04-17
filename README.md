# Stock Market Prediction with Markov Chains

## What This Project Does

This project uses a **Markov chain** model to simulate how a stock's price might move in the future, based on its past daily returns. It also includes an optional **AI Analysis** feature that fetches recent news about the ticker and uses Claude (Anthropic's AI) to produce an educational summary alongside the simulation. It is a learning tool — not financial advice.

Here is the basic idea:
1. It downloads historical stock closing prices from Yahoo Finance.
2. It groups each day's return into a "state" (e.g. big drop, small drop, flat, small gain, big gain).
3. It counts how often the market moves from one state to another — this becomes the **transition matrix**.
4. It uses that matrix to randomly simulate probable future price paths.
5. Optionally, it retrieves recent news headlines and passes them to Claude to generate a contextual analysis.

---

## Problem Statement

Predicting stock prices is one of the hardest problems in finance. Markets are noisy, influenced by countless unpredictable factors, and constantly changing. Most simple models fail because they assume prices follow a neat pattern — but in reality, daily returns are highly variable.

This project tackles a narrower question: **can we model the statistical behavior of daily returns using a Markov chain, and use that model to simulate probable short-term price paths?**

A **Markov chain** is a mathematical model that describes a sequence of events where the probability of each event depends only on the state of the previous event — not on anything that came before it. This is known as the "memoryless" property. For example, if the stock market had a bad day today, a Markov chain estimates the probability of tomorrow being good, flat, or bad — purely based on what tends to follow a bad day historically, ignoring everything else. Each possible condition (bad day, good day, etc.) is called a **state**, and the likelihood of moving between states is captured in a **transition matrix**.

Rather than trying to predict exact future prices (which is not reliably possible), the goal is to capture the structure of how return "regimes" tend to transition — for example, whether a bad day is more likely to be followed by a recovery or another bad day.

---

## Objective

- Represent daily stock returns as a sequence of discrete states (e.g. large loss, small loss, flat, small gain, large gain).
- Learn a **transition matrix** from historical data that captures how likely the market is to move from one state to another.
- Use the fitted model to **simulate future price paths** by randomly sampling state transitions.
- Optionally layer in **AI-powered news analysis** using the Claude API and recent headlines.
- Provide a clean, beginner-friendly example of applying Markov chains to financial time series data.

---

## Prerequisites

You need **Python 3.8 or newer**. Check your version:

```bash
python3 --version
```

You also need `pip`, which comes bundled with Python. If you are unsure, run:

```bash
pip --version
```

> **conda users:** The base Anaconda environment ships with Python 3.7, which is too old for some dependencies. Use a conda environment with Python 3.8+ (see Installation below).

---

## Installation

**1. Clone or download this repository**

```bash
git clone https://github.com/your-username/stock-market-prediction.git
cd stock-market-prediction
```

**2. Create an environment with Python 3.8+**

*Option A — virtual environment (recommended for most users):*

```bash
python3 -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

*Option B — conda environment:*

```bash
conda create -n stockenv python=3.10
conda activate stockenv
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## How to Run

There are two ways to use this project: a **web UI** (recommended for beginners) or a **command line**.

---

### Option 1 — Web UI (Streamlit)

The easiest way to use the project. Opens an interactive app in your browser where you can fill in all parameters and see results as charts and tables.

```bash
streamlit run app.py
```

A browser tab will open automatically at `http://localhost:8501`.

**What you can do in the UI:**
- Enter any ticker symbol (e.g. `AAPL`, `MSFT`, `TSLA`)
- Choose a historical period from a dropdown (`6mo`, `1y`, `2y`, `5y`)
- Choose a **state bucketing mode**:
  - **Quantile** — splits states evenly by return percentile; adjust the number of states (3–10) with a slider
  - **Volume (Low / Average / High)** — uses 3 fixed buckets based on the mean return (below mean, near mean, above mean)
- Adjust the simulation horizon with a slider (5–60 trading days)
- Set a random seed for reproducible results
- Click **Run Simulation** to see:
  - A line chart of the simulated price path with high/low reference lines
  - Start price, end price, and simulated % change
  - **Simulated High** (value and % shown in green) and **Simulated Low** (value and % shown in red, with highlighted badges)
  - Current and most likely next market state
  - Transition matrix
  - State definitions table (return range, mean return, observation count)

**Optional — AI Analysis (requires Anthropic API key):**

The AI Analysis feature is entirely opt-in and does not affect the simulation unless explicitly enabled:

1. Paste your [Anthropic API key](https://console.anthropic.com/) into the sidebar under **AI Analysis (optional)**.
2. Tick **Generate AI Analysis**.
3. Click **Run Simulation**.

When enabled, two things happen before results are shown:
- **News sentiment classification** — Claude reads the latest headlines for the ticker and classifies sentiment as bearish (−1), neutral (0), or bullish (+1). A banner explains whether and how the simulation's starting state was adjusted based on this signal.
- **Full AI analysis** — Claude combines the Markov model output with the news to produce a 3–5 paragraph educational summary. A collapsible **Sources** section lists every article used, with clickable links.

If no API key is entered or the checkbox is left unticked, the simulation runs as a pure Markov chain with no external data or Claude calls.

---

### Option 2 — Command Line

**Download live data from Yahoo Finance:**

```bash
python3 markov_stock_prediction.py --ticker AAPL --period 1y --states 5 --horizon 10
```

**Use your own CSV file:**

```bash
python3 markov_stock_prediction.py --csv data/prices.csv
```

The CSV must have at minimum a `Date` column and a `Close` column. Example:

```
Date,Close
2024-01-02,185.20
2024-01-03,184.40
2024-01-04,186.75
...
```

If your columns have different names, use `--date-column` and `--price-column`:

```bash
python3 markov_stock_prediction.py --csv data/prices.csv --date-column date --price-column close_price
```

---

## CLI Parameters

| Flag | Default | What it does |
|---|---|---|
| `--ticker` | *(required if no --csv)* | Ticker symbol to download from Yahoo Finance (e.g. `AAPL`) |
| `--csv` | *(required if no --ticker)* | Path to a local CSV file with price history |
| `--period` | `1y` | How far back to download data. Options: `6mo`, `1y`, `2y`, `5y` |
| `--states` | `5` | Number of return states (more states = finer-grained model) |
| `--horizon` | `10` | How many days into the future to simulate |
| `--seed` | `42` | Random seed — use the same number to get the same simulation output |
| `--date-column` | `Date` | Name of the date column in your CSV |
| `--price-column` | `Close` | Name of the price column in your CSV |

---

## Understanding the Output

**State bins** — The return thresholds that define each state. For example:
```
bin 0: -0.04998   ← very negative returns land in state 0
bin 1: -0.00833
...
bin 5:  0.06315   ← very positive returns land in state 4
```

**Transition matrix** — The probability of moving from one state (row) to another (column) on the next day. Each row sums to 1.0.
```
       0      1      2      3      4
0  0.240  0.140  0.160  0.160  0.300
1  0.220  0.280  0.200  0.140  0.160
...
```
Row 0, column 4 = 0.300 means: after a very bad day, there is a 30% chance the next day is a very good day.

**Mean return per state** — The average daily return observed in each state historically.

**Simulated price path** — A sequence of prices starting from today's closing price, projected forward day by day using random state transitions.

**Simulated high / low** — The maximum and minimum prices reached during the simulated path, shown as dashed green/red reference lines on the chart. The metric cards display the high value and percentage in green and the low value and percentage in red (with a highlighted badge) for quick visual scanning.

**AI Analysis** *(optional, requires API key + checkbox)* — A 3–5 paragraph summary written by Claude that interprets the model output in light of recent news. Includes a sentiment banner explaining any starting-state adjustment and a Sources section with links to every article used. The pure Markov simulation is unaffected when this feature is disabled.

---

## Project Structure

The logic is split into a `markov/` package of focused modules, keeping each file small and easy to understand individually.

```
stock-market-prediction/
├── markov/
│   ├── __init__.py          # Package exports — import anything from here
│   ├── preprocessing.py     # Load prices from CSV or download from Yahoo Finance
│   ├── states.py            # Map continuous returns to discrete Markov states
│   ├── transition.py        # Build the transition matrix and per-state statistics
│   ├── simulation.py        # Simulate future price paths via state transitions
│   ├── model.py             # MarkovStockModel — high-level fit/predict/simulate API
│   ├── summary.py           # Print a human-readable model summary
│   └── rag.py               # AI Analysis — news fetching and Claude API integration
├── markov_stock_prediction.py   # CLI entry point — thin wrapper around the package
├── app.py                       # Streamlit web UI
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

### What each module does

| Module | Responsibility |
|---|---|
| `preprocessing.py` | Loads and cleans historical price data. Handles both local CSV files and live Yahoo Finance downloads. Ensures the output is always a float-typed `pd.Series`. |
| `states.py` | Computes quantile-based bin edges and maps each daily return to an integer state index (0 = worst, N-1 = best). |
| `transition.py` | Counts observed state-to-state transitions, normalizes counts into probabilities (the transition matrix), and computes the mean return and observation count per state. |
| `simulation.py` | Steps forward day by day — randomly sampling the next state from the transition distribution and compounding the price by that state's mean return. |
| `model.py` | `MarkovStockModel` dataclass that ties all modules together. Call `.fit()` to train, then `.simulate_prices()`, `.predict_next_state()`, or `.most_likely_next_state()` to use it. |
| `summary.py` | Prints state bins, the full transition matrix, and per-state mean returns in a readable format. |
| `rag.py` | Fetches recent news via yfinance, builds a prompt combining the news with Markov model output, and calls the Claude API to generate an educational analysis. |

---

## How the Model Works (Beginner-Friendly)

A **Markov chain** is a simple mathematical model where the next state depends only on the current state — not on anything that happened before. Think of it like a weather forecast that only looks at today's weather to guess tomorrow's.

In this project:
- Each "state" represents how well the stock did on a given day (e.g. state 0 = bad day, state 4 = great day).
- The **transition matrix** is learned from historical data and captures patterns like "after a great day, what usually happens next?"
- During simulation, the model randomly picks the next state according to those learned probabilities, then computes the next price from the state's average return.

**State bucketing modes:**
- **Quantile** — divides all historical returns into N equal-sized groups by percentile. More states give a finer picture of the return distribution.
- **Volume (Low / Average / High)** — uses 3 fixed buckets relative to the historical mean return. Returns more than half a standard deviation below the mean are *Low*, within half a standard deviation are *Average*, and above are *High*.

Because the model is stochastic (random), running it twice with different seed values will give different simulated paths. This reflects the genuine uncertainty in future prices.

---

## AI Analysis (Optional)

The **AI Analysis** feature is fully opt-in — the simulation runs as a pure Markov chain until you explicitly enable it. It uses Retrieval-Augmented Generation (RAG) to combine the model's quantitative output with qualitative context from recent news.

**How to enable:** enter your Anthropic API key in the sidebar, tick **Generate AI Analysis**, then click Run Simulation.

**What happens when enabled:**

1. **News sentiment classification** — `rag.py` fetches recent headlines via yfinance and calls `claude-opus-4-6` using tool use to return a structured sentiment score (−1 bearish, 0 neutral, +1 bullish) and a one-sentence explanation. If bearish, the simulation starts in the lowest state; if bullish, in the highest state; if neutral, the return-based state is used unchanged. A colour-coded banner in the UI explains the adjustment.
2. **News fetching** — The same articles retrieved for sentiment are reused (no second network call).
3. **Prompt construction** — All articles are included in a structured prompt alongside the model's simulation results (current price, simulated end price, news-adjusted state labels, etc.).
4. **Claude API call** — The prompt is sent to `claude-opus-4-6` with adaptive thinking enabled. Claude synthesises the quantitative and qualitative signals into a 3–5 paragraph analysis.
5. **Sources** — The articles used are surfaced in a collapsible Sources section in the UI, each with a clickable link to the original article.

To use this feature you need an Anthropic API key from [console.anthropic.com](https://console.anthropic.com/).

---

## Limitations

- This is a **simplified educational model**. Real markets are far more complex.
- The Markov model only uses past return history — it ignores news, earnings, macroeconomic events, and other factors.
- Simulated paths are one possible scenario, not a forecast.
- The AI Analysis reflects recent headlines but cannot access real-time data, predict future events, or account for information not present in the news summaries.
- **Do NOT use this for real investment decisions.**

---

## Disclaimer

This project is intended **for educational and research purposes ONLY**.

- Nothing in this repository constitutes financial, investment, or trading advice.
- Simulated price paths are generated from a simplified statistical model and do not represent predictions of actual future prices.
- AI-generated analysis is produced by a language model and may contain errors or omissions.
- Past market behavior does NOT guarantee future results.
- The authors are NOT responsible for any financial decisions made based on this tool or its output.

**Always consult a qualified financial professional before making any investment decisions.**

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical computations (matrix math, quantiles) |
| `pandas` | Data loading and manipulation |
| `yfinance` | Downloading stock price data and news from Yahoo Finance |
| `streamlit` | Web UI — interactive browser-based interface |
| `anthropic` | Claude API client — used for the optional AI Analysis feature |

---

## Streamlit App

Run the app:

[Stock Market Prediction Simulator](https://vyasrsrinivasan-stock-market-prediction-app-coukef.streamlit.app/)
