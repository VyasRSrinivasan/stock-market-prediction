# Stock Market Prediction with Markov Chains

## What This Project Does

This project uses a **Markov chain** model to simulate how a stock's price might move in the future, based on its past daily returns. On top of that it layers two additional quantitative models — an **SVM (RBF) classifier** and a **Monte Carlo (GBM) simulator** — whose outputs are chained together so each model enriches the next. An optional **AI Analysis** feature fetches recent news and uses Claude to produce an educational summary that covers all three models. It is a learning tool — not financial advice.

Here is the basic idea:
1. It downloads historical stock closing prices from Yahoo Finance.
2. It groups each day's return into a "state" (e.g. big drop, small drop, flat, small gain, big gain).
3. It counts how often the market moves from one state to another — this becomes the **transition matrix**.
4. It uses that matrix to randomly simulate probable future price paths.
5. An **SVM with an RBF kernel** is trained on engineered features (lagged returns, rolling statistics, momentum) to predict the probability distribution over the next state.
6. A **Monte Carlo (GBM) simulation** runs 500 independent price paths. When the SVM is available, its regime probabilities replace the OLS drift estimate, making the simulation regime-aware.
7. Optionally, it retrieves recent news headlines and passes all three models' outputs to Claude to generate a contextual analysis.
8. All results can be **downloaded as a PDF report** with a single click.

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
- Train an **SVM (RBF) classifier** on engineered features to predict the next-state probability distribution and simulate an alternative price path.
- Run a **Monte Carlo (GBM) simulation** whose drift is conditioned on the SVM's regime probabilities when available, otherwise falling back to an OLS trend estimate.
- Optionally layer in **AI-powered news analysis** using the Claude API — Claude is given all three models' outputs and synthesises them with recent headlines.
- Allow users to **download a PDF report** of every result with a single button click.
- Provide a clean, beginner-friendly example of applying Markov chains and ML to financial time series data.

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

## Obtaining an Anthropic API Key

An API key is only needed for the optional **AI Analysis** feature. The Markov chain simulation works without one.

**1. Create a free Anthropic account**

Go to [console.anthropic.com](https://console.anthropic.com/) and sign up with your email address.

**2. Navigate to API Keys**

After logging in, click your account name in the top-right corner and select **API Keys**, or go directly to [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys).

**3. Create a new key**

Click **Create Key**, give it a name (e.g. `stock-predictor`), and click **Create Key** again. Copy the key immediately — it is only shown once.

**4. Add credits (if needed)**

New accounts include a small free credit. For ongoing use, add a payment method under **Billing** at [console.anthropic.com/settings/billing](https://console.anthropic.com/settings/billing). The AI Analysis feature makes two lightweight API calls per simulation run (one for sentiment, one for the full analysis), so costs are minimal.

**5. Paste the key into the app**

In the sidebar under **AI Analysis (optional)**, paste your key into the **Anthropic API Key** field. It is masked as a password and is never stored beyond your browser session.

> **Security note:** Never commit your API key to version control. If you want to set it as an environment variable instead, you can read it in `rag.py` via `os.environ.get("ANTHROPIC_API_KEY")` and skip the sidebar input.

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
  - **SVM (RBF) Prediction** — a separate price path simulated by the SVM model, end-price metrics, and a bar chart of the predicted next-state probability distribution
  - **Monte Carlo Simulation** — a fan chart of 500 GBM paths showing the P10/P25/P50/P75/P90 percentile bands, with median, pessimistic (P10), and optimistic (P90) end prices. The drift is automatically conditioned on SVM regime probabilities when available; an expandable parameters panel shows which drift source was used.
  - **Download Report** — a button at the bottom that exports all results as a PDF

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

**SVM (RBF) Prediction** — An SVM classifier trained on five lagged log-returns, rolling mean (5-day and 10-day), rolling volatility (5-day), and momentum predicts the probability of each state occurring next. The most likely state and its probability are shown above a bar chart of the full distribution. The SVM also simulates its own price path by rolling forward, re-computing features from each newly simulated return.

**Monte Carlo Simulation** — 500 independent Geometric Brownian Motion paths are simulated over the chosen horizon. Percentile fan bands (P10/P25/P50/P75/P90) are shown on the chart. When the SVM has run successfully, the GBM drift is replaced by the SVM-conditioned expected return Σ P(state=i) × mean_return(i) — making the simulation regime-aware. If the SVM is unavailable, drift falls back to an OLS trend estimate on log-prices. An expandable **Model parameters** panel shows the drift source, active drift value, OLS baseline (when SVM is used), daily volatility, and number of paths.

**AI Analysis** *(optional, requires API key + checkbox)* — A 3–5 paragraph summary written by Claude that interprets all three models' output in light of recent news. The prompt includes Markov, Monte Carlo, and SVM outputs so Claude can compare where they agree or diverge. Includes a sentiment banner explaining any starting-state adjustment and a Sources section with links to every article used. The pure Markov simulation is unaffected when this feature is disabled.

**Download Report** — A **Download PDF Report** button appears at the bottom of the results page. Clicking it generates and downloads a PDF containing every section: Markov metrics, transition matrix, state definitions, SVM probabilities, Monte Carlo parameters, AI analysis text, and news sources.

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
│   ├── montecarlo.py        # Monte Carlo (GBM) simulation with OLS / SVM-conditioned drift
│   ├── svm_model.py         # SVM (RBF) next-state classifier and price simulator
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
| `montecarlo.py` | Runs N independent GBM price paths. Drift is either estimated via OLS on log-prices or replaced by the SVM-conditioned expected return. Returns percentile fan bands (P10/25/50/75/90) and summary statistics. |
| `svm_model.py` | Trains a `StandardScaler + SVC(kernel='rbf', probability=True)` pipeline on five lagged log-returns, rolling mean (5-day, 10-day), rolling volatility (5-day), and momentum. Exposes `predict_next_state_probs()` and `simulate_svm_prices()`. |
| `rag.py` | Fetches recent news via yfinance, classifies sentiment using Claude tool use, builds a structured prompt that includes Markov, Monte Carlo, and SVM outputs, and calls the Claude API to generate an educational analysis. |

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

**SVM (RBF) model:**

A Support Vector Machine with an RBF (radial basis function) kernel is trained on each run to predict which state the market is most likely to enter next. The features it uses are:
- Five lagged daily log-returns (yesterday, two days ago, etc.)
- 5-day and 10-day rolling mean log-return
- 5-day rolling standard deviation (short-term volatility)
- Momentum (5-day log-return minus 10-day log-return)

The model outputs a full probability distribution across states, not just a single prediction. It also simulates its own price path by rolling forward one step at a time — at each step it re-computes the feature vector from the most recent (real + simulated) returns, samples a state from the predicted distribution, and compounds the price by that state's historical mean return.

**Monte Carlo (GBM) simulation:**

Geometric Brownian Motion models a price as a random walk where each daily step is:

```
S(t+1) = S(t) × exp((μ − 0.5σ²) + σ × Z),   Z ~ N(0, 1)
```

- **σ (volatility)** is always computed from the standard deviation of historical daily log-returns.
- **μ (drift)** has two modes: by default it is estimated via OLS regression on log-prices (capturing the recent price trend); when the SVM has run, it is replaced by the SVM-conditioned expected return Σ P(state=i) × mean_return(i) — so a bearish SVM prediction pulls the drift down and a bullish prediction pushes it up.

500 independent paths are simulated and their percentile fan bands (P10/P25/P50/P75/P90) are plotted. The expandable **Model parameters** panel shows which drift source was used, the active drift value, the OLS baseline (when SVM-conditioned), daily volatility, and path count.

---

## AI Analysis (Optional)

The **AI Analysis** feature is fully opt-in — the simulation runs as a pure Markov chain until you explicitly enable it. It uses Retrieval-Augmented Generation (RAG) to combine the model's quantitative output with qualitative context from recent news.

**How to enable:** enter your Anthropic API key in the sidebar, tick **Generate AI Analysis**, then click Run Simulation.

**What happens when enabled:**

1. **News sentiment classification** — `rag.py` fetches recent headlines via yfinance and calls `claude-opus-4-6` using tool use to return a structured sentiment score (−1 bearish, 0 neutral, +1 bullish) and a one-sentence explanation. If bearish, the simulation starts in the lowest state; if bullish, in the highest state; if neutral, the return-based state is used unchanged. A colour-coded banner in the UI explains the adjustment.
2. **News fetching** — The same articles retrieved for sentiment are reused (no second network call).
3. **Prompt construction** — All articles are included in a structured prompt alongside the outputs of all three models: the Markov chain simulation, the SVM's next-state probabilities, and the Monte Carlo percentile statistics. The prompt instructs Claude to compare where the models agree or diverge.
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
| `numpy` | Numerical computations (matrix math, quantiles, GBM paths) |
| `pandas` | Data loading and manipulation |
| `yfinance` | Downloading stock price data and news from Yahoo Finance |
| `streamlit` | Web UI — interactive browser-based interface |
| `scikit-learn` | SVM (RBF) classifier and `StandardScaler` pipeline |
| `anthropic` | Claude API client — used for the optional AI Analysis feature |
| `fpdf2` | PDF report generation — no system dependencies required |

---

## Streamlit App

Run the app:

[Stock Market Prediction Simulator](https://vyasrsrinivasan-stock-market-prediction-app-coukef.streamlit.app/)
