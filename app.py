import streamlit as st
import pandas as pd
import numpy as np

from markov import (
    MarkovStockModel,
    download_price_series,
    print_model_summary,
)

st.set_page_config(page_title="Markov Stock Predictor", layout="centered")

st.title("Markov Chain Stock Price Simulator")
st.caption("An educational tool for simulating short-term stock price paths using Markov chains.")

# ── Sidebar inputs ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Parameters")

    ticker = st.text_input(
        "Ticker Symbol",
        value="AAPL",
        help="Stock ticker symbol (e.g. AAPL, MSFT, TSLA)",
    ).upper().strip()

    period = st.selectbox(
        "Historical Period",
        options=["6mo", "1y", "2y", "5y"],
        index=1,
        help="How far back to fetch price data from Yahoo Finance.",
    )

    n_states = st.slider(
        "Number of States",
        min_value=3,
        max_value=10,
        value=5,
        help="More states = finer return categories. 5 is a good default.",
    )

    horizon = st.slider(
        "Simulation Horizon (days)",
        min_value=5,
        max_value=60,
        value=10,
        help="How many trading days to simulate into the future.",
    )

    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=99999,
        value=42,
        help="Fix this value to get the same simulation output each run.",
    )

    run = st.button("Run Simulation", type="primary", width="stretch")

# ── Main panel ────────────────────────────────────────────────────────────────
if not run:
    st.info("Set your parameters in the sidebar and click **Run Simulation**.")
    st.stop()

if not ticker:
    st.error("Please enter a ticker symbol.")
    st.stop()

with st.spinner(f"Downloading data for {ticker}..."):
    try:
        prices = download_price_series(ticker, period)
    except Exception as e:
        st.error(f"Could not download data for **{ticker}**: {e}")
        st.stop()

with st.spinner("Fitting Markov model..."):
    try:
        model = MarkovStockModel.fit(prices, n_states=n_states)
    except ValueError as e:
        st.error(str(e))
        st.stop()

returns = prices.pct_change().dropna()
current_return = float(returns.iloc[-1])
current_state = model.state_for_return(current_return)

simulation = model.simulate_prices(
    start_price=float(prices.iloc[-1]),
    start_state=current_state,
    horizon=horizon,
    random_seed=int(seed),
)

# ── Results ───────────────────────────────────────────────────────────────────
st.success(f"Model fitted on **{len(prices)}** trading days of **{ticker}** data.")

# Simulated price chart
st.subheader("Simulated Price Path")
sim_df = pd.DataFrame({
    "Day": range(len(simulation)),
    "Price": simulation.values,
})
sim_df = sim_df.set_index("Day")
st.line_chart(sim_df)

col1, col2, col3 = st.columns(3)
col1.metric("Start Price", f"${float(prices.iloc[-1]):.2f}")
col2.metric("Simulated End Price", f"${simulation.iloc[-1]:.2f}")
delta = simulation.iloc[-1] - float(prices.iloc[-1])
col3.metric("Simulated Change", f"${delta:.2f}", delta=f"{delta / float(prices.iloc[-1]) * 100:.1f}%")

st.divider()

# Current state info
st.subheader("Current Market State")
col_a, col_b = st.columns(2)
col_a.metric("Current State", current_state)
col_b.metric("Most Likely Next State", model.most_likely_next_state(current_state))
st.caption(f"Current state mean return: `{model.state_mean_returns[current_state]:.5f}`")

st.divider()

# Transition matrix
st.subheader("Transition Matrix")
st.caption("Each cell shows the probability of moving from row-state to column-state on the next day.")
tm_df = pd.DataFrame(
    model.transition_matrix,
    index=[f"State {i}" for i in range(n_states)],
    columns=[f"State {i}" for i in range(n_states)],
).round(3)
st.dataframe(tm_df, width="stretch")

st.divider()

# State bins and mean returns
st.subheader("State Definitions")
st.caption("Each state covers a range of daily returns, split by quantile.")
state_rows = []
for i in range(n_states):
    state_rows.append({
        "State": i,
        "Return Range": f"{model.state_bins[i]:.4f}  →  {model.state_bins[i + 1]:.4f}",
        "Mean Return": f"{model.state_mean_returns[i]:.5f}",
        "Observations": int(model.initial_state_counts[i]),
    })
st.dataframe(pd.DataFrame(state_rows).set_index("State"), width="stretch")

st.divider()
st.caption("This tool is for educational purposes only. Not financial advice.")
