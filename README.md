# Stock Market Prediction with Markov Chains

## What This Project Does

This project uses a **Markov chain** model to simulate how a stock's price might move in the future, based on its past daily returns. On top of that it layers two additional quantitative models — an **SVM (RBF) classifier** and a **Monte Carlo (GBM) simulator** — whose outputs are chained together so each model enriches the next. An optional **AI Analysis** feature fetches recent news and uses your choice of AI provider to produce an educational summary that covers all three models. It is a learning tool — not financial advice.

Here is the basic idea:
1. It downloads historical stock closing prices from Yahoo Finance.
2. It groups each day's return into a "state" (e.g. big drop, small drop, flat, small gain, big gain).
3. It counts how often the market moves from one state to another — this becomes the **transition matrix**.
4. It uses that matrix to randomly simulate probable future price paths.
5. An **SVM with an RBF kernel** is trained on engineered features (lagged returns, rolling statistics, momentum) to predict the probability distribution over the next state.
6. A **Monte Carlo (GBM) simulation** runs 500 independent price paths. When the SVM is available, its regime probabilities replace the OLS drift estimate, making the simulation regime-aware.
7. Optionally, it retrieves recent news headlines and passes all three models' outputs to your chosen AI provider (Groq, Anthropic Claude, OpenAI, Google Gemini, or DeepSeek) to generate a contextual analysis, with optional per-article relevance summaries. **Groq is the default provider and is free — no API key required when using the hosted app.**
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
- Optionally layer in **AI-powered news analysis** using a choice of provider (Groq, Anthropic Claude, OpenAI, Google Gemini, or DeepSeek) — the model is given all three quantitative outputs and synthesises them with recent headlines. Optionally generate an AI relevance summary for each source article. **Groq is the default and free — no API key needed on the hosted app.**
- Allow users to **download a PDF report** of every result with a single button click, including charts, AI analysis, provider attribution, and per-article relevance summaries.
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

## Obtaining an API Key

An API key is only needed for the optional **AI Analysis** feature. The Markov chain simulation works without one.

**No API key is required on the hosted app** — shared keys are configured for all five providers. Simply select a provider, tick **Generate AI Analysis**, and run.

If the shared key for a provider runs low on balance, the app will display a warning and prompt you to enter your own key. You can always paste your own key into the sidebar field to use your own quota — it takes priority over the shared key.

The app supports five providers:

| Provider | Free tier | Where to get a key |
|---|---|---|
| **Groq (Free)** | Yes — generous free tier, no credit card | [console.groq.com](https://console.groq.com) |
| **Anthropic (Claude)** | No | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) |
| **OpenAI** | No | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| **Google Gemini** | Yes — limited free tier | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
| **DeepSeek** | No | [platform.deepseek.com/api-keys](https://platform.deepseek.com/api-keys) |

**How to use AI Analysis:**

1. In the sidebar under **AI Analysis (optional)**, select a provider from the **Provider** dropdown. The API key field will show **(optional)** when a shared key is available.
2. Leave the key field blank to use the shared app key, or paste your own key to use your own quota.
3. Tick **Generate AI Analysis** and click **Run Simulation**.

The AI Analysis makes two API calls per run — one for sentiment classification and one for the full analysis — so costs are minimal regardless of provider.

**For app owners — configuring shared keys via Streamlit Secrets:**

Place provider keys in `.streamlit/secrets.toml` (local) or in **App settings → Secrets** on Streamlit Community Cloud. The app uses these as fallbacks when a user leaves the key field blank:

```toml
GROQ_API_KEY      = "gsk_..."    # recommended — free tier
ANTHROPIC_API_KEY = "sk-ant-..."
OPENAI_API_KEY    = "sk-proj-..."
GEMINI_API_KEY    = "AIza..."
DEEPSEEK_API_KEY  = "sk-..."
```

> **Security note:** `secrets.toml` is gitignored by default in this repo. Never commit real keys. If a shared key's balance is exhausted, the app shows a warning and directs the user to enter their own key.

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
  - A line chart of the simulated price path with actual business dates on the x-axis and high/low reference lines
  - Start price, end price, and simulated % change
  - **Simulated High** (value and % shown in green) and **Simulated Low** (value and % shown in red, with highlighted badges)
  - **Current State** and **Most Likely Next State** metric cards, each showing the state label (e.g. State 0), its regime name (e.g. Capitulation), a **Schwab-style letter grade** (F through A), and the corresponding analyst rating meaning (e.g. *Strongly Underperform* → *Strongly Outperform*)
  - Transition matrix
  - **State Definitions table** — return range, mean return, observation count, regime name, letter grade, and rating meaning for every state. An expandable **"What do these states mean?"** legend explains each active state with a colour dot, regime name, grade badge, and plain-English description
  - **SVM (RBF) Prediction** — a separate price path simulated by the SVM model, end-price metrics, a bar chart of the predicted next-state probability distribution, and an inline grade badge on the most likely next state
  - **Monte Carlo Simulation** — a fan chart of 500 GBM paths showing the P10/P25/P50/P75/P90 percentile bands, with median, pessimistic (P10), and optimistic (P90) end prices. The drift is automatically conditioned on SVM regime probabilities when available; a compact **Model parameters** panel shows drift source, active drift, OLS baseline, daily volatility, and path count.
  - **Download Report** — a button at the bottom that exports all results as a PDF including all charts

**UI styling:** the app uses a financial analyst typographic system — Playfair Display (headings), IBM Plex Sans (body), and IBM Plex Mono (metric values and tables) — with a navy and gold color scheme (`#00296b` / `#fdc500`) applied to headings, metric cards, buttons, and accents. The sidebar has a mint green background (`#c2f2e2`) with a gold right border.

**Optional — AI Analysis:**

The AI Analysis feature is entirely opt-in and does not affect the simulation unless explicitly enabled:

1. Select a provider from the **Provider** dropdown (**Groq (Free)** is the default). A colour-coded logo badge for the selected provider is shown above the dropdown.
2. Leave the API key field blank to use the shared app key (the label shows **(optional)** when one is available), or paste your own key to use your own quota.
3. Tick **Generate AI Analysis**.
4. Optionally tick **AI-summarize each source article** to generate a one-sentence relevance note for each news article (uses one additional API call).
5. Click **Run Simulation**.

When enabled, two things happen before results are shown:
- **News sentiment classification** — the selected model reads the latest headlines for the ticker and classifies sentiment as bearish (−1), neutral (0), or bullish (+1). A banner explains whether and how the simulation's starting state was adjusted based on this signal.
- **Full AI analysis** — the model combines the Markov, SVM, and Monte Carlo outputs with recent news to produce a 3–5 paragraph educational summary. A collapsible **Sources** section lists every article used, with clickable links. If source summarization is enabled, each article also shows an AI-generated sentence explaining its relevance to the simulation result.

If the shared key for the selected provider has insufficient balance, a warning banner is shown prompting you to enter your own API key. If the checkbox is left unticked, the simulation runs as a pure Markov chain with no external data or AI calls.

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

**State bins** — The return thresholds that define each regime. For example:
```
bin 0: -0.04998   ← Capitulation / severe drawdown (bottom quintile)
bin 1: -0.00833   ← Mild distribution / corrective pressure
bin 2:  0.00000   ← Consolidation / range-bound session (median)
bin 3:  0.00833   ← Accumulation / moderate upside
bin 5:  0.06315   ← Strong momentum / breakout session (top quintile)
```

**Transition matrix** — The empirical regime-transition probability matrix. Each cell T[i][j] is the historical frequency of moving from regime i (row) to regime j (column) on the next trading day. Each row sums to 1.0.
```
       0      1      2      3      4
0  0.240  0.140  0.160  0.160  0.300
1  0.220  0.280  0.200  0.140  0.160
...
```
Row 0, column 4 = 0.300: following a capitulation session, there is a 30% probability of a momentum/breakout day — consistent with a sharp mean-reversion dynamic (e.g. oversold bounce). The high T[0][0] = 0.240 simultaneously indicates that drawdown sessions also show meaningful persistence (consecutive down days are nearly as likely as an immediate bounce).

**Mean return per state** — The average realised daily log-return within each regime, estimated from the full historical sample. This is the return the simulation compounds when the model occupies that state.

**Simulated price path** — A sequence of prices starting from today's closing price, projected forward day by day using random state transitions.

**Simulated high / low** — The maximum and minimum prices reached during the simulated path, shown as dashed green/red reference lines on the chart. The metric cards display the high value and percentage in green and the low value and percentage in red (with a highlighted badge) for quick visual scanning.

**SVM (RBF) Prediction** — An SVM classifier trained on five lagged log-returns, rolling mean (5-day and 10-day), rolling volatility (5-day), and momentum predicts the probability of each state occurring next. The most likely state is displayed with its regime name, Schwab letter grade, and rating meaning alongside the probability. A bar chart shows the full distribution across all states. The SVM also simulates its own price path by rolling forward, re-computing features from each newly simulated return.

**Monte Carlo Simulation** — 500 independent Geometric Brownian Motion paths are simulated over the chosen horizon. Percentile fan bands (P10/P25/P50/P75/P90) are shown on the chart. When the SVM has run successfully, the GBM drift is replaced by the SVM-conditioned expected return Σ P(state=i) × mean_return(i) — making the simulation regime-aware. If the SVM is unavailable, drift falls back to an OLS trend estimate on log-prices. An expandable **Model parameters** panel shows the drift source, active drift value, OLS baseline (when SVM is used), daily volatility, and number of paths.

**AI Analysis** *(optional, requires API key + checkbox)* — A 3–5 paragraph summary produced by your chosen AI provider that interprets all three models' output in light of recent news. The prompt includes Markov, Monte Carlo, and SVM outputs so the model can compare where they agree or diverge. Includes a sentiment banner explaining any starting-state adjustment and a Sources section with links to every article used. When **AI-summarize each source article** is enabled, each article additionally shows a one-sentence note explaining its relevance to the simulation result. The pure Markov simulation is unaffected when this feature is disabled.

**Download Report** — A **Download PDF Report** button appears at the bottom of the results page. Clicking it generates and downloads a PDF containing every section: Markov metrics, transition matrix, state definitions, SVM probabilities, Monte Carlo parameters, all matplotlib charts (with actual business dates on the x-axis), AI analysis text, provider attribution badge, and news sources with optional per-article relevance summaries (rendered in blue).

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
│   └── rag.py               # AI Analysis — news fetching and multi-provider LLM integration
├── .streamlit/
│   └── secrets.toml         # Provider API keys (gitignored) — used as shared fallback keys
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
| `rag.py` | Fetches recent news via yfinance, classifies sentiment, builds a structured prompt that includes Markov, Monte Carlo, and SVM outputs, and calls the chosen AI provider (Groq, Anthropic, OpenAI, Google Gemini, or DeepSeek) to generate an educational analysis. Optionally generates per-article relevance summaries in a single batched call. |

---

## How the Model Works (Beginner-Friendly)

A **Markov chain** is a simple mathematical model where the next state depends only on the current state — not on anything that happened before. Think of it like a weather forecast that only looks at today's weather to guess tomorrow's.

In this project:
- Each "state" represents a distinct **return regime** observed in the security's daily price history — ranging from severe drawdown at one end to strong momentum at the other.
- The **transition matrix** is estimated from historical data and encodes the empirical regime-transition probabilities: given that the market is in a particular regime today, how likely is each regime tomorrow?
- During simulation, the model samples the next regime according to those learned probabilities, then compounds the price by the mean return historically observed in that regime.

**State bucketing modes:**
- **Quantile** — partitions all historical daily log-returns into N equal-frequency bins. Each bin captures an identically-sized slice of the return distribution, so all states are equally represented in the training data. More states reveal finer granularity in the return distribution at the cost of sparser transition observations per cell.
- **Volume (Low / Average / High)** — assigns each day to one of three regime buckets relative to the security's historical mean daily return (μ) and standard deviation (σ): returns below μ − 0.5σ map to *Low*, returns within ±0.5σ of μ map to *Average*, and returns above μ + 0.5σ map to *High*. This schema is parameter-free and directly interpretable in terms of realized volatility bands.

Because the model is stochastic, running it twice with different seed values produces different simulated paths. This reflects the genuine dispersion of outcomes implied by the historical regime-transition structure.

---

## Market States — Nomenclature & Financial Interpretation

Each state in the Markov chain corresponds to a **daily return regime** calibrated from historical price data. States are labelled with practitioner-standard names drawn from technical analysis and institutional trading terminology. Every state also carries a **Schwab-style letter grade** (F through A) that maps directly to the analyst rating scale used by major brokerages: F = Strongly Underperform, D = Underperform, C = Market Perform, B = Outperform, A = Strongly Outperform. Plus (+) and minus (−) modifiers are used for finer gradations when more than five states are active.

These grades and regime names are shown throughout the app — on the Current State and Most Likely Next State metric cards, in the State Definitions table, in the "What do these states mean?" expander, and on the SVM most likely next state caption — making the model output directly interpretable without reference to raw return percentiles.

### Schwab Letter Grade Scale

| Grade | Analyst Rating | Signal |
|---|---|---|
| **A** | Strongly Outperform | Strong bullish conviction — consider adding to portfolio |
| **A−** | Outperform+ | High bullish conviction |
| **B** | Outperform | Constructive — above-market expected return |
| **B−** | Outperform– | Mild positive conviction |
| **C+** | Market Perform+ | Slightly above neutral |
| **C** | Market Perform | Neutral — hold existing position, no action needed |
| **C−** | Market Perform– | Slightly below neutral |
| **D+** | Underperform+ | Mild negative conviction |
| **D** | Underperform | Weak — consider whether to hold or reduce |
| **D−** | Underperform– | High negative conviction |
| **F** | Strongly Underperform | Strong bearish signal — consider reducing or avoiding |

The grade assigned to each state is determined purely by its **rank position** in the return distribution — State 0 always receives F, the top state always receives A, regardless of the absolute return values. Grades at intermediate positions are interpolated from the table below.

---

### Quantile State Nomenclature

The state names vary with the number of states selected. Each name maps to a specific slice of the historical daily return distribution.

#### 3 States

| # | Grade | Name | Percentile Band | Regime Signal |
|---|---|---|---|---|
| 0 | **F** | **Bear** | Bottom tercile (≤ P33) | Persistent selling pressure, downtrend, or risk-off |
| 1 | **C** | **Neutral** | P33–P67 (median band) | Range-bound, balanced order flow, no directional conviction |
| 2 | **A** | **Bull** | Top tercile (≥ P67) | Positive momentum, accumulation, or trend-continuation |

#### 4 States

| # | Grade | Name | Percentile Band | Regime Signal |
|---|---|---|---|---|
| 0 | **F** | **Drawdown** | ≤ P25 | Significant decline, elevated selling, negative momentum |
| 1 | **D** | **Correction** | P25–P50 | Below-median returns, mild distribution or profit-taking |
| 2 | **B** | **Recovery** | P50–P75 | Above-median returns, constructive tape, early accumulation |
| 3 | **A** | **Breakout** | ≥ P75 | Strong upside, momentum continuation, or positive catalyst |

#### 5 States *(default)*

| # | Grade | Name | Percentile Band | Regime Signal |
|---|---|---|---|---|
| 0 | **F** | **Capitulation** | ≤ P20 | Panic selling, forced liquidation, gap-downs, severe drawdown |
| 1 | **D** | **Distribution** | P20–P40 | Below-median pressure, profit-taking, softening conviction |
| 2 | **C** | **Consolidation** | P40–P60 | Equilibrium, sideways tape, low-conviction range-bound session |
| 3 | **B** | **Accumulation** | P60–P80 | Constructive action, institutional buying, improving breadth |
| 4 | **A** | **Momentum** | ≥ P80 | Strong upside, bullish catalyst, breakout from prior range |

#### 6 States

| # | Grade | Name | Percentile Band | Regime Signal |
|---|---|---|---|---|
| 0 | **F** | **Capitulation** | ≤ P17 | Panic, forced selling, severe drawdown |
| 1 | **D** | **Distribution** | P17–P33 | Moderate decline, distribution phase |
| 2 | **C−** | **Drift Down** | P33–P50 | Mild below-median weakness, soft tape |
| 3 | **C+** | **Drift Up** | P50–P67 | Mild above-median strength, tentative buying |
| 4 | **B** | **Accumulation** | P67–P83 | Constructive, institutional support |
| 5 | **A** | **Momentum** | ≥ P83 | Strong rally, breakout, catalyst-driven upside |

#### 7 States

| # | Grade | Name | Percentile Band | Regime Signal |
|---|---|---|---|---|
| 0 | **F** | **Panic** | ≤ P14 | Extreme selloff, margin calls, fear-driven liquidation |
| 1 | **D−** | **Capitulation** | P14–P29 | Heavy drawdown, persistent selling |
| 2 | **D** | **Distribution** | P29–P43 | Moderate weakness, profit-taking |
| 3 | **C** | **Consolidation** | P43–P57 | Balanced, indecisive, range-bound |
| 4 | **B** | **Accumulation** | P57–P71 | Moderate upside, constructive tape |
| 5 | **A−** | **Rally** | P71–P86 | Strong positive momentum |
| 6 | **A** | **Breakout** | ≥ P86 | Explosive upside, catalyst-driven surge |

#### 8 States

| # | Grade | Name | Percentile Band | Regime Signal |
|---|---|---|---|---|
| 0 | **F** | **Panic** | ≤ P12.5 | Extreme fear, forced liquidation |
| 1 | **D−** | **Capitulation** | P12.5–P25 | Heavy sustained selling |
| 2 | **D** | **Distribution** | P25–P37.5 | Moderate decline, supply overhang |
| 3 | **C−** | **Drift Down** | P37.5–P50 | Mild below-median weakness |
| 4 | **C+** | **Drift Up** | P50–P62.5 | Mild above-median strength |
| 5 | **B** | **Accumulation** | P62.5–P75 | Constructive buying, institutional activity |
| 6 | **A−** | **Rally** | P75–P87.5 | Strong upside, positive momentum |
| 7 | **A** | **Breakout** | ≥ P87.5 | Explosive gain, catalyst-driven surge |

#### 9 States

| # | Grade | Name | Percentile Band | Regime Signal |
|---|---|---|---|---|
| 0 | **F** | **Panic** | ≤ P11 | Extreme selloff |
| 1 | **D−** | **Capitulation** | P11–P22 | Heavy drawdown |
| 2 | **D** | **Distribution** | P22–P33 | Moderate selling |
| 3 | **D+** | **Mild Distribution** | P33–P44 | Soft tape, mild weakness |
| 4 | **C** | **Consolidation** | P44–P56 | Balanced, no direction |
| 5 | **B−** | **Mild Accumulation** | P56–P67 | Tentative buying |
| 6 | **B** | **Accumulation** | P67–P78 | Constructive, institutional support |
| 7 | **A−** | **Rally** | P78–P89 | Strong upside |
| 8 | **A** | **Breakout** | ≥ P89 | Explosive surge |

#### 10 States

| # | Grade | Name | Percentile Band | Regime Signal |
|---|---|---|---|---|
| 0 | **F** | **Panic** | ≤ P10 | Extreme fear, flash crash behavior |
| 1 | **D−** | **Capitulation** | P10–P20 | Forced liquidation, severe drawdown |
| 2 | **D** | **Distribution** | P20–P30 | Heavy selling pressure |
| 3 | **D+** | **Mild Distribution** | P30–P40 | Below-median weakness, profit-taking |
| 4 | **C−** | **Drift Down** | P40–P50 | Soft, slightly below neutral |
| 5 | **C+** | **Drift Up** | P50–P60 | Soft, slightly above neutral |
| 6 | **B−** | **Mild Accumulation** | P60–P70 | Constructive, early buying |
| 7 | **B** | **Accumulation** | P70–P80 | Institutional buying, improving breadth |
| 8 | **A−** | **Rally** | P80–P90 | Strong positive momentum |
| 9 | **A** | **Breakout** | ≥ P90 | Explosive upside, catalyst-driven surge |

---

### Volume State Nomenclature

The three-state volume schema uses fixed labels regardless of the number of states selected:

| Grade | Name | Threshold | Regime Signal |
|---|---|---|---|
| **F** | **Low** | Return < μ − 0.5σ | Bearish — return falls meaningfully below the historical average; risk-off, downtrend, distribution phase |
| **C** | **Average** | μ − 0.5σ ≤ Return ≤ μ + 0.5σ | Neutral — return within half a standard deviation of the mean; balanced order flow, transitional tape |
| **A** | **High** | Return > μ + 0.5σ | Bullish — return materially exceeds the historical average; accumulation, trend continuation, positive catalyst |

---

### Reading the Transition Matrix as a Practitioner

The transition matrix is the core quantitative output of the Markov model. Each cell T[i][j] gives the **empirical probability of transitioning from regime i to regime j on the next trading day**, estimated from the full historical sample.

Key patterns to look for:

- **High diagonal values (T[i][i] > 0.5)** — strong **regime persistence**. A Capitulation state that tends to stay in Capitulation (high T[0][0]) signals downside momentum; a Momentum state that stays in Momentum (high T[N-1][N-1]) signals bullish continuation.
- **High off-diagonal T[0][N-1] or T[N-1][0]** — elevated **mean-reversion probability**. A high probability of jumping directly from Panic to Breakout (or vice versa) is characteristic of high-volatility, whipsaw-prone securities.
- **Flat rows (all values ≈ 1/N)** — **regime-agnostic transitions**, meaning the next day's return is essentially independent of today's. This implies the Markov assumption adds little predictive value for this security over the chosen period.
- **Skewed rows toward higher states** — regardless of the current regime, the security tends to drift upward, indicating a secular bullish bias in the historical sample. The reverse skew signals a bearish secular trend.

### Current State and Most Likely Next State

The **Current State** shown in the UI reflects the regime into which the most recent trading day's return falls, given the model's calibrated bin edges. This is the starting point for all forward simulations.

The **Most Likely Next State** is the column j in the current state's transition row T[current_state][j] with the highest probability. It is the single-step modal forecast under the Markov model — not a certainty, but the historically most probable next regime. When the AI Analysis feature is enabled, the news-sentiment signal may override the starting state: a **bearish** score anchors the simulation in the lowest regime (Panic / Capitulation / Bear / Low), and a **bullish** score anchors it in the highest (Breakout / Momentum / Bull / High), reflecting qualitative macro or company-specific context that pure return history cannot capture.

**SVM (RBF) model:**

A Support Vector Machine with an RBF (Radial Basis Function) kernel is trained on each run to predict which state the market is most likely to enter next. The features it uses are:
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

**Supported providers:**

| Provider | Model used | Free? | Notes |
|---|---|---|---|
| **Groq (Free)** | `llama-3.3-70b-versatile` | Yes | Default — no key needed on hosted app; OpenAI-compatible endpoint |
| **Anthropic (Claude)** | `claude-opus-4-6` | No | Sentiment via tool use; analysis with adaptive thinking |
| **OpenAI** | `gpt-4o` | No | Sentiment and analysis via JSON response format |
| **Google Gemini** | `gemini-2.0-flash` | Limited | OpenAI-compatible endpoint (`generativelanguage.googleapis.com`) |
| **DeepSeek** | `deepseek-chat` | No | OpenAI-compatible endpoint (`api.deepseek.com`) |

**How to enable:** select a provider from the sidebar dropdown (Groq is pre-selected), optionally enter your own API key (the field is marked **optional** when a shared key exists), tick **Generate AI Analysis**, then click Run Simulation. All providers work without a personal key on the hosted app. If a shared key's balance is exhausted, the app surfaces a warning and prompts the user to supply their own key.

**What happens when enabled:**

1. **News sentiment classification** — `rag.py` fetches recent headlines via yfinance and asks the chosen model to return a structured sentiment score (−1 bearish, 0 neutral, +1 bullish) with a one-sentence explanation. Anthropic uses tool use; other providers use JSON response format. If bearish, the simulation starts in the lowest state; if bullish, in the highest; if neutral, the return-based state is used unchanged. A colour-coded banner in the UI explains the adjustment.
2. **News fetching** — The same articles retrieved for sentiment are reused (no second network call to Yahoo Finance).
3. **Prompt construction** — All articles are included in a structured prompt alongside the outputs of all three models: the Markov chain simulation, the SVM's next-state probabilities, and the Monte Carlo percentile statistics. The prompt instructs the model to compare where they agree or diverge.
4. **AI analysis call** — The prompt is sent to the chosen provider. Claude uses adaptive thinking; all others use standard chat completions. The model synthesises the quantitative and qualitative signals into a 3–5 paragraph analysis.
5. **Source summarization** *(optional)* — If **AI-summarize each source article** is ticked, a single additional batched call is made asking for one relevance sentence per article, returned as a JSON array. These summaries are displayed in the Sources expander and included in the PDF in blue (`#0006b1`).
6. **Sources** — The articles used are surfaced in a collapsible Sources section in the UI, each with a clickable link. If source summarization is enabled, each article shows an AI-generated sentence explaining its relevance to the simulation result.

**PDF report** — The AI Analysis section of the PDF includes a colour-coded provider badge (e.g. orange for Anthropic, black for OpenAI, blue for Gemini/DeepSeek) next to the "Generated by" attribution line, followed by the analysis text. Source articles list the title, URL, and — when generated — the relevance summary in blue italic text.

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
| `matplotlib` | Chart rendering for embedded PDF report figures |
| `yfinance` | Downloading stock price data and news from Yahoo Finance |
| `streamlit` | Web UI — interactive browser-based interface |
| `scikit-learn` | SVM (RBF) classifier and `StandardScaler` pipeline |
| `anthropic` | Claude API client — used when Anthropic is selected as the AI provider |
| `openai` | OpenAI-compatible client — used for Groq, OpenAI, Google Gemini, and DeepSeek providers |
| `fpdf2` | PDF report generation — no system dependencies required |

---

## Streamlit App

Run the app:

[Stock Market Prediction Simulator](https://vyasrsrinivasan-stock-market-prediction-app-coukef.streamlit.app/)

## References

https://www.schwab.com/learn/story/buy-hold-sell-what-analyst-stock-ratings-mean

https://www.troweprice.com/personal-investing/resources/insights/how-monte-carlo-analysis-could-improve-your-retirement-plan.html