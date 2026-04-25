import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys

from markov import (
    MarkovStockModel,
    download_price_series,
    print_model_summary,
    assign_states,
    compute_transition_matrix,
    compute_state_mean_returns,
    compute_initial_state_counts,
    run_monte_carlo,
)

try:
    from markov import train_svm, predict_next_state_probs, simulate_svm_prices
    _svm_importable = True
except ImportError:
    _svm_importable = False

try:
    from markov import get_news_sentiment, run_rag_analysis
    _rag_importable = True
except ImportError:
    _rag_importable = False

st.set_page_config(page_title="Stock Predictor", layout="centered")

try:
    import sklearn  # noqa: F401
    _sklearn_available = True
except ImportError:
    _sklearn_available = False

st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    [data-testid="stMetricDelta"] { font-size: 1.2rem; }
    [data-testid="stMetricLabel"] { font-size: 1rem; }
    </style>
""", unsafe_allow_html=True)

col_l, col_m, col_r = st.columns([1, 2, 1])
with col_m:
    st.image("./images/StockPricePredictionImage.png", width=200)
st.title("Stock Price Prediction Simulator")
st.markdown('<p style="font-size:1.1rem; color:#1f77b4;">An educational tool for simulating short-term stock price paths using Markov chains.</p>', unsafe_allow_html=True)

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

    bucket_mode = st.radio(
        "State Bucketing",
        options=["Quantile", "Volume (Low / Average / High)"],
        help="Quantile splits states evenly by return percentile. Volume uses 3 fixed buckets based on the mean return.",
    )
    if bucket_mode == "Quantile":
        n_states = st.slider(
            "Number of States",
            min_value=3,
            max_value=10,
            value=5,
            help="More states = finer return categories. 5 is a good default.",
        )
    else:
        n_states = 3
        st.caption("3 fixed states: **Low** (below mean), **Average** (near mean), **High** (above mean).")

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

    st.divider()
    st.header("AI Analysis (optional)")
    anthropic_api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        help="Paste your Anthropic API key to enable AI-powered news analysis. Leave blank to skip.",
    )
    run_rag = st.checkbox(
        "Generate AI Analysis",
        value=False,
        disabled=not anthropic_api_key,
        help="Uses Claude + recent news to contextualise the simulation. Requires an API key.",
    )

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

returns = prices.pct_change().dropna()

with st.spinner("Fitting Markov model..."):
    try:
        if bucket_mode == "Volume (Low / Average / High)":
            mean_ret = float(returns.mean())
            std_ret = float(returns.std())
            state_bins = np.array([
                float(returns.min()),
                mean_ret - 0.5 * std_ret,
                mean_ret + 0.5 * std_ret,
                float(returns.max()),
            ])
            states_arr = assign_states(returns, state_bins)
            model = MarkovStockModel(
                n_states=3,
                transition_matrix=compute_transition_matrix(states_arr, 3),
                state_bins=state_bins,
                state_mean_returns=compute_state_mean_returns(returns, states_arr, 3),
                initial_state_counts=compute_initial_state_counts(states_arr, 3),
            )
            state_labels = ["Low", "Average", "High"]
        else:
            model = MarkovStockModel.fit(prices, n_states=n_states)
            state_labels = [f"State {i}" for i in range(n_states)]
    except ValueError as e:
        st.error(str(e))
        st.stop()

current_return = float(returns.iloc[-1])
current_state = model.state_for_return(current_return)

# ── News-conditioned starting state ───────────────────────────────────────────
sentiment_data = None
sim_start_state = current_state

if run_rag and anthropic_api_key:
    with st.spinner("Analyzing news sentiment to condition simulation..."):
        try:
            sentiment_data = get_news_sentiment(ticker, anthropic_api_key)
            s = sentiment_data["sentiment"]
            if s == -1:
                sim_start_state = 0
            elif s == 1:
                sim_start_state = n_states - 1
            # s == 0: keep current_state
        except Exception:
            sentiment_data = None  # fall back silently

simulation = model.simulate_prices(
    start_price=float(prices.iloc[-1]),
    start_state=sim_start_state,
    horizon=horizon,
    random_seed=int(seed),
)

# ── Results ───────────────────────────────────────────────────────────────────
st.success(f"Model fitted on **{len(prices)}** trading days of **{ticker}** data.")

if sentiment_data:
    s = sentiment_data["sentiment"]
    raw_label = state_labels[current_state]
    adj_label = state_labels[sim_start_state]
    if s == -1:
        st.warning(
            f"**Bearish news sentiment detected.** "
            f"Simulation starts in **{adj_label}** (lowest state) "
            f"instead of the return-based state **{raw_label}**. "
            f"_{sentiment_data['reasoning']}_"
        )
    elif s == 1:
        st.success(
            f"**Bullish news sentiment detected.** "
            f"Simulation starts in **{adj_label}** (highest state) "
            f"instead of the return-based state **{raw_label}**. "
            f"_{sentiment_data['reasoning']}_"
        )
    else:
        st.info(
            f"**Neutral news sentiment.** "
            f"Simulation starts in the return-based state **{raw_label}**. "
            f"_{sentiment_data['reasoning']}_"
        )

# Simulated price chart
st.subheader("Simulated Price Path")
sim_high = float(simulation.max())
sim_low = float(simulation.min())
sim_df = pd.DataFrame({
    "Day": range(len(simulation)),
    "Price": simulation.values,
})
y_min = sim_low * 0.99
y_max = sim_high * 1.01

line = (
    alt.Chart(sim_df)
    .mark_line(color="#1f77b4")
    .encode(
        x=alt.X("Day:Q", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("Price:Q", scale=alt.Scale(domain=[y_min, y_max])),
    )
)
high_rule = (
    alt.Chart(pd.DataFrame({"y": [sim_high], "label": ["High"]}))
    .mark_rule(color="green", strokeDash=[4, 4])
    .encode(y="y:Q")
)
low_rule = (
    alt.Chart(pd.DataFrame({"y": [sim_low], "label": ["Low"]}))
    .mark_rule(color="red", strokeDash=[4, 4])
    .encode(y="y:Q")
)
chart = (line + high_rule + low_rule).properties(width="container")
st.altair_chart(chart, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Previous Day Closing Price", f"${float(prices.iloc[-1]):.2f}")
col2.metric("Simulated End Price", f"${simulation.iloc[-1]:.2f}")
delta = simulation.iloc[-1] - float(prices.iloc[-1])
col3.metric("Simulated Change", f"${delta:.2f}", delta=f"{delta / float(prices.iloc[-1]) * 100:.1f}%")

col4, col5 = st.columns(2)
low_pct = (sim_low / float(prices.iloc[-1]) - 1) * 100
high_pct = (sim_high / float(prices.iloc[-1]) - 1) * 100
col4.markdown(f"""
    <div>
        <div style="font-size:1rem; color:inherit">Simulated Low</div>
        <div style="font-size:1.8rem; font-weight:600; color:#6f0000">${sim_low:.2f}</div>
        <div style="display:inline-block; font-size:1.2rem; color:#d62728;
                    background-color:#ffd7d7; border-radius:4px; padding:1px 6px">{low_pct:+.1f}%</div>
    </div>
""", unsafe_allow_html=True)
col5.markdown(f"""
    <div>
        <div style="font-size:1rem; color:inherit">Simulated High</div>
        <div style="font-size:1.8rem; font-weight:600; color:#306844">${sim_high:.2f}</div>
        <div style="display:inline-block; font-size:1.2rem; color:#2ca02c;
                    background-color:#d4edda; border-radius:4px; padding:1px 6px">{high_pct:+.1f}%</div>
    </div>
""", unsafe_allow_html=True)

st.divider()

# Current state info
st.subheader("Current Market State")
col_a, col_b = st.columns(2)
col_a.metric("Current State", state_labels[current_state])
col_b.metric("Most Likely Next State", state_labels[model.most_likely_next_state(current_state)])
st.caption(f"Current state mean return: `{model.state_mean_returns[current_state]:.5f}`")

st.divider()

# Transition matrix
st.subheader("Transition Matrix")
st.caption("Each cell shows the probability of moving from row-state to column-state on the next day.")
tm_df = pd.DataFrame(
    model.transition_matrix,
    index=state_labels,
    columns=state_labels,
).round(3)
st.dataframe(tm_df, width="stretch")

st.divider()

# State bins and mean returns
st.subheader("State Definitions")
st.caption("Each state covers a range of daily returns, split by quantile.")
state_rows = []
for i in range(n_states):
    state_rows.append({
        "State": state_labels[i],
        "Return Range": f"{model.state_bins[i]:.4f}  →  {model.state_bins[i + 1]:.4f}",
        "Mean Return": f"{model.state_mean_returns[i]:.5f}",
        "Observations": int(model.initial_state_counts[i]),
    })
st.dataframe(pd.DataFrame(state_rows).set_index("State"), width="stretch")

st.divider()

# ── Monte Carlo Simulation ────────────────────────────────────────────────────
# ── SVM RBF Prediction ────────────────────────────────────────────────────────
st.subheader("SVM (RBF) Prediction")
st.caption(
    "An SVM with an RBF kernel is trained on engineered features (lagged returns, "
    "rolling mean, volatility, momentum) to predict the next state at each step. "
    "Its regime probability output also conditions the Monte Carlo drift below."
)

svm_clf = None
svm_probs = None
svm_simulation = None

if not _sklearn_available or not _svm_importable:
    py = sys.version.split()[0]
    st.warning(
        f"**scikit-learn is not available in the current Python environment "
        f"(Python {py}).**\n\n"
        "Install it with: `pip install scikit-learn`"
    )

with st.spinner("Training SVM model..."):
    try:
        svm_clf, n_train = train_svm(prices, model.state_bins, n_states)
        svm_probs = predict_next_state_probs(svm_clf, prices)
        svm_simulation = simulate_svm_prices(
            svm_clf, prices, model.state_mean_returns, horizon, random_seed=int(seed)
        )

        # Price path chart
        svm_sim_df = pd.DataFrame({
            "Day": range(len(svm_simulation)),
            "Price": svm_simulation.values,
        })
        svm_y_min = float(svm_simulation.min()) * 0.99
        svm_y_max = float(svm_simulation.max()) * 1.01
        svm_line = (
            alt.Chart(svm_sim_df)
            .mark_line(color="#ff7f0e")
            .encode(
                x=alt.X("Day:Q", axis=alt.Axis(tickMinStep=1)),
                y=alt.Y("Price:Q", scale=alt.Scale(domain=[svm_y_min, svm_y_max])),
            )
            .properties(width="container")
        )
        st.altair_chart(svm_line, use_container_width=True)

        # Summary metrics
        svm_end = float(svm_simulation.iloc[-1])
        svm_delta = svm_end - float(prices.iloc[-1])
        svm_col1, svm_col2, svm_col3 = st.columns(3)
        svm_col1.metric("SVM End Price", f"${svm_end:.2f}")
        svm_col2.metric("SVM Change", f"${svm_delta:.2f}",
                        delta=f"{svm_delta / float(prices.iloc[-1]) * 100:.1f}%")
        svm_col3.metric("Trained on", f"{n_train} samples")

        # Next-state probability bar chart
        st.caption("Predicted probability distribution over next states:")
        prob_df = pd.DataFrame({
            "State": state_labels,
            "Probability": svm_probs,
        })
        top_idx = int(svm_probs.argmax())
        prob_chart = (
            alt.Chart(prob_df)
            .mark_bar()
            .encode(
                x=alt.X("State:N", sort=None),
                y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, 1]),
                        axis=alt.Axis(format="%")),
                color=alt.condition(
                    alt.datum["State"] == state_labels[top_idx],
                    alt.value("#ff7f0e"),
                    alt.value("#ffbb78"),
                ),
                tooltip=["State:N", alt.Tooltip("Probability:Q", format=".1%")],
            )
            .properties(width="container")
        )
        st.altair_chart(prob_chart, use_container_width=True)
        st.caption(
            f"Most likely next state: **{state_labels[top_idx]}** "
            f"({svm_probs[top_idx] * 100:.1f}%)"
        )

    except ImportError:
        pass  # warning already shown above
    except Exception as e:
        st.error(f"SVM model failed: {e}")

st.divider()

# ── Monte Carlo Simulation (SVM-conditioned) ──────────────────────────────────
st.subheader("Monte Carlo Simulation")
if svm_probs is not None:
    st.caption(
        f"500 independent GBM paths over {horizon} trading days. "
        "Drift is conditioned on the SVM's regime probabilities — "
        "the expected return Σ P(state=i) × mean_return(i) replaces the OLS baseline. "
        "Bands show the 10th, 25th, 50th, 75th, and 90th percentiles across all paths."
    )
else:
    st.caption(
        f"500 independent GBM paths over {horizon} trading days. "
        "Drift estimated via OLS regression on log-prices; volatility from historical log-returns. "
        "Bands show the 10th, 25th, 50th, 75th, and 90th percentiles across all paths."
    )

mc = run_monte_carlo(
    prices,
    horizon=horizon,
    n_simulations=500,
    random_seed=int(seed),
    svm_probs=svm_probs,
    state_mean_returns=model.state_mean_returns if svm_probs is not None else None,
)

days = list(range(horizon + 1))
mc_df = pd.DataFrame({
    "Day": days,
    "P10": mc["bands"][10],
    "P25": mc["bands"][25],
    "P50": mc["bands"][50],
    "P75": mc["bands"][75],
    "P90": mc["bands"][90],
})

base = alt.Chart(mc_df)
mc_y_min = float(mc["bands"][10].min()) * 0.99
mc_y_max = float(mc["bands"][90].max()) * 1.01
outer_band = base.mark_area(opacity=0.12, color="#1f77b4").encode(
    x=alt.X("Day:Q", axis=alt.Axis(tickMinStep=1)),
    y=alt.Y("P10:Q", title="Price", scale=alt.Scale(domain=[mc_y_min, mc_y_max])),
    y2=alt.Y2("P90:Q"),
)
inner_band = base.mark_area(opacity=0.25, color="#1f77b4").encode(
    x="Day:Q",
    y="P25:Q",
    y2=alt.Y2("P75:Q"),
)
median_line = base.mark_line(color="#1f77b4", strokeWidth=2).encode(
    x="Day:Q",
    y="P50:Q",
)
mc_chart = (outer_band + inner_band + median_line).properties(width="container")
st.altair_chart(mc_chart, use_container_width=True)

mc_col1, mc_col2, mc_col3 = st.columns(3)
mc_col1.metric("Median End Price", f"${mc['median_end']:.2f}")
mc_col2.metric("Pessimistic (P10)", f"${mc['p10_end']:.2f}")
mc_col3.metric("Optimistic (P90)", f"${mc['p90_end']:.2f}")

with st.expander("Model parameters"):
    drift_label = mc["drift_source"]
    param_text = (
        f"Drift source: `{drift_label}` — `{mc['drift_daily'] * 100:+.4f}%/day`"
    )
    if mc["drift_source"] == "SVM-conditioned":
        param_text += f"  ·  OLS baseline: `{mc['drift_ols'] * 100:+.4f}%/day`"
    param_text += (
        f"  ·  Daily volatility: `{mc['sigma_daily'] * 100:.4f}%`"
        f"  ·  Paths: `{mc['n_simulations']}`"
    )
    st.caption(param_text)

st.divider()

# ── AI Analysis (RAG) ─────────────────────────────────────────────────────────
if run_rag and anthropic_api_key:
    st.subheader("AI Analysis")
    st.caption(
        "Claude reads recent news about this ticker and combines it with the Markov model "
        "output to produce an educational summary. This is NOT financial advice."
    )
    with st.spinner("Fetching news and generating analysis..."):
        try:
            result = run_rag_analysis(
                ticker=ticker,
                api_key=anthropic_api_key,
                current_price=float(prices.iloc[-1]),
                simulated_end_price=float(simulation.iloc[-1]),
                sim_change_pct=float(simulation.iloc[-1] / float(prices.iloc[-1]) - 1) * 100,
                sim_high=sim_high,
                sim_low=sim_low,
                current_state_label=state_labels[sim_start_state],
                next_state_label=state_labels[model.most_likely_next_state(sim_start_state)],
                horizon=horizon,
                articles=sentiment_data["articles"] if sentiment_data else None,
                monte_carlo=mc,
                svm_probs=svm_probs,
                state_labels=state_labels,
            )
            st.markdown(result["analysis"])

            sources = result["sources"]
            if sources:
                with st.expander(f"Sources ({len(sources)} articles used)"):
                    for i, src in enumerate(sources, start=1):
                        title = src["title"] or "Untitled"
                        url = src["url"]
                        summary = src["text"][:280].rstrip() + ("…" if len(src["text"]) > 280 else "")
                        if url:
                            st.markdown(f"**{i}. [{title}]({url})**")
                        else:
                            st.markdown(f"**{i}. {title}**")
                        st.caption(summary)
                        if i < len(sources):
                            st.divider()
        except ImportError as e:
            st.warning(
                f"Missing dependency: {e}. "
                "Install with: `pip install anthropic chromadb sentence-transformers`"
            )
        except Exception as e:
            st.error(f"AI analysis failed: {e}")
    st.divider()
elif run_rag and not anthropic_api_key:
    st.info("Enter your Anthropic API key in the sidebar to enable AI analysis.")
    st.divider()

st.caption("This tool is for educational purposes only. NOT financial advice.")
