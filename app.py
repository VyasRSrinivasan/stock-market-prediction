import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys
import datetime
import os
import io

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
    from markov.rag import InsufficientBalanceError
    _rag_importable = True
except ImportError:
    _rag_importable = False
    InsufficientBalanceError = Exception  # fallback so except clauses still compile

def _pdf_safe(text: str) -> str:
    """Replace Unicode characters outside Latin-1 with ASCII equivalents."""
    return (
        str(text)
        .replace("\u2014", "--")   # em dash
        .replace("\u2013", "-")    # en dash
        .replace("\u2018", "'")    # left single quote
        .replace("\u2019", "'")    # right single quote
        .replace("\u201c", '"')    # left double quote
        .replace("\u201d", '"')    # right double quote
        .replace("\u2026", "...")  # ellipsis
        .replace("\u2022", "-")    # bullet
        .replace("\u00b7", ".")    # middle dot
        .encode("latin-1", errors="replace").decode("latin-1")
    )


def _generate_pdf(
    ticker, period, horizon, seed, n_states,
    current_price, simulation, sim_high, sim_low,
    current_state, state_labels, model,
    svm_probs, svm_simulation,
    mc,
    sentiment_data,
    rag_result,
    ai_provider=None,
    last_price_date=None,
):
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise ImportError("fpdf2 is required for PDF export. Install it with: pip install fpdf2") from exc

    S = _pdf_safe

    # All widths explicit — no w=0 auto-calculations.
    PW = 190   # A4 usable width (210 - 10 left - 10 right)
    LW = 62    # label column for key:value rows
    VW = PW - LW

    # ── Lazy matplotlib import for chart generation ───────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _charts = True
    except ImportError:
        _charts = False

    def _embed_chart(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        pdf.image(buf, x=pdf.l_margin, w=PW)
        pdf.ln(3)

    def _section(title):
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(PW, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)

    def _kv(key, val, multicell=False):
        pdf.cell(LW, 7, S(str(key) + ":"), border=0)
        if multicell:
            pdf.multi_cell(VW, 7, S(str(val)), new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.cell(VW, 7, S(str(val)), new_x="LMARGIN", new_y="NEXT")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Header ────────────────────────────────────────────────────────────────
    _img = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "StockPricePredictionPDFImage.png")
    if os.path.exists(_img):
        _img_w = 200  # mm — compact logo at top
        _img_x = (210 - _img_w) / 2  # centered on A4
        pdf.image(_img, x=_img_x, w=_img_w)
        pdf.ln(4)

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(PW, 10, S(f"Stock Prediction Report: {ticker}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(
        PW, 6,
        S(f"Generated: {datetime.date.today()}  |  Period: {period}  |  Horizon: {horizon} days  |  Seed: {seed}"),
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(PW, 5, "Educational purposes only -- NOT financial advice.", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # ── Markov Chain Simulation ───────────────────────────────────────────────
    _section("Markov Chain Simulation")

    end_price = float(simulation.iloc[-1])
    delta = end_price - current_price
    delta_pct = delta / current_price * 100
    low_pct = (sim_low / current_price - 1) * 100
    high_pct = (sim_high / current_price - 1) * 100

    if _charts:
        import matplotlib.dates as mdates
        if last_price_date is not None:
            _sim_dates = pd.bdate_range(start=last_price_date + pd.Timedelta(days=1), periods=len(simulation))
            x_vals = _sim_dates
        else:
            x_vals = list(range(len(simulation)))
        fig, ax = plt.subplots(figsize=(8, 2.8), facecolor='white')
        ax.plot(x_vals, simulation.values, color='#1f77b4', linewidth=1.8, zorder=3)
        ax.axhline(sim_high, color='#2ca02c', linestyle='--', linewidth=1.2, alpha=0.9,
                   label=f'High  ${sim_high:.2f} ({high_pct:+.1f}%)')
        ax.axhline(sim_low, color='#d62728', linestyle='--', linewidth=1.2, alpha=0.9,
                   label=f'Low  ${sim_low:.2f} ({low_pct:+.1f}%)')
        y_pad = max((sim_high - sim_low) * 0.08, 0.5)
        ax.set_ylim(sim_low - y_pad, sim_high + y_pad)
        if last_price_date is not None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)
            ax.set_xlabel('Date', fontsize=8)
        else:
            ax.set_xlabel('Day', fontsize=8)
        ax.set_ylabel('Price ($)', fontsize=8)
        ax.set_title('Simulated Price Path', fontsize=9, fontweight='bold', pad=4)
        ax.legend(fontsize=7, framealpha=0.6, loc='upper left')
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.18, linewidth=0.5)
        plt.tight_layout(pad=0.5)
        _embed_chart(fig)

    _kv("Previous Close", f"${current_price:.2f}")
    _kv("Simulated End Price", f"${end_price:.2f}  ({delta_pct:+.1f}%)")
    _kv("Simulated High", f"${sim_high:.2f}  ({high_pct:+.1f}%)")
    _kv("Simulated Low", f"${sim_low:.2f}  ({low_pct:+.1f}%)")
    _cur_regime = _QUANTILE_STATE_NAMES.get(n_states, state_labels)[current_state]
    _cur_grade, _, _, _cur_meaning = _schwab_rating(current_state, n_states)
    _kv("Current State", S(f"{state_labels[current_state]}  |  {_cur_regime}  |  {_cur_grade} – {_cur_meaning}"))
    _nxt_i = model.most_likely_next_state(current_state)
    _nxt_regime = _QUANTILE_STATE_NAMES.get(n_states, state_labels)[_nxt_i]
    _nxt_grade, _, _, _nxt_meaning = _schwab_rating(_nxt_i, n_states)
    _kv("Most Likely Next State", S(f"{state_labels[_nxt_i]}  |  {_nxt_regime}  |  {_nxt_grade} – {_nxt_meaning}"))
    if sentiment_data:
        s_val = sentiment_data["sentiment"]
        s_label = {-1: "Bearish", 0: "Neutral", 1: "Bullish"}.get(s_val, "Unknown")
        _kv("News Sentiment", f"{s_label} - {sentiment_data['reasoning']}", multicell=True)

    pdf.ln(3)

    # Transition matrix
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(PW, 7, "Transition Matrix", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    col_w = PW // (n_states + 1)
    pdf.cell(col_w, 6, "", border=1)
    for lbl in state_labels:
        pdf.cell(col_w, 6, S(lbl[:9]), border=1, align="C")
    pdf.ln()
    for i, row_lbl in enumerate(state_labels):
        pdf.cell(col_w, 6, S(row_lbl[:9]), border=1)
        for j in range(n_states):
            pdf.cell(col_w, 6, f"{model.transition_matrix[i, j]:.3f}", border=1, align="C")
        pdf.ln()

    pdf.ln(3)

    # State definitions
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(PW, 7, "State Definitions", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    _sd_names = _QUANTILE_STATE_NAMES.get(n_states, state_labels)
    hdr      = ["State", "Regime",  "Grade", "Return Range",              "Mean Ret", "Obs"]
    col_ws   = [20,       42,        14,      52,                          30,          32]
    for h, w in zip(hdr, col_ws):
        pdf.set_font("Helvetica", "B", 7)
        pdf.cell(w, 6, h, border=1, align="C")
    pdf.ln()
    pdf.set_font("Helvetica", "", 7)
    for i in range(n_states):
        _grade, _, _, _meaning = _schwab_rating(i, n_states)
        pdf.cell(col_ws[0], 6, S(state_labels[i]), border=1)
        pdf.cell(col_ws[1], 6, S(_sd_names[i]), border=1)
        pdf.cell(col_ws[2], 6, _grade, border=1, align="C")
        pdf.cell(col_ws[3], 6, f"{model.state_bins[i]:.4f} -> {model.state_bins[i+1]:.4f}", border=1, align="C")
        pdf.cell(col_ws[4], 6, f"{model.state_mean_returns[i]:.5f}", border=1, align="C")
        pdf.cell(col_ws[5], 6, str(int(model.initial_state_counts[i])), border=1, align="C")
        pdf.ln()
    pdf.set_font("Helvetica", "", 8)

    pdf.ln(4)

    # ── SVM (RBF) Prediction ──────────────────────────────────────────────────
    if svm_probs is not None and svm_simulation is not None:
        _section("SVM (RBF) Prediction")

        svm_end = float(svm_simulation.iloc[-1])
        svm_delta_pct = (svm_end - current_price) / current_price * 100
        top_idx = int(svm_probs.argmax())

        if _charts:
            import matplotlib.dates as mdates
            if last_price_date is not None:
                _svm_dates = pd.bdate_range(start=last_price_date + pd.Timedelta(days=1), periods=len(svm_simulation))
                svm_x = _svm_dates
            else:
                svm_x = list(range(len(svm_simulation)))
            fig, ax = plt.subplots(figsize=(8, 2.8), facecolor='white')
            ax.plot(svm_x, svm_simulation.values, color='#ff7f0e', linewidth=1.8, zorder=3)
            s_lo = float(svm_simulation.min())
            s_hi = float(svm_simulation.max())
            y_pad = max((s_hi - s_lo) * 0.08, 0.5)
            ax.set_ylim(s_lo - y_pad, s_hi + y_pad)
            if last_price_date is not None:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=7)
                ax.set_xlabel('Date', fontsize=8)
            else:
                ax.set_xlabel('Day', fontsize=8)
            ax.set_ylabel('Price ($)', fontsize=8)
            ax.set_title('SVM Simulated Price Path', fontsize=9, fontweight='bold', pad=4)
            ax.tick_params(labelsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.18, linewidth=0.5)
            plt.tight_layout(pad=0.5)
            _embed_chart(fig)

        _kv("SVM End Price", f"${svm_end:.2f}  ({svm_delta_pct:+.1f}%)")
        _svm_names = _QUANTILE_STATE_NAMES.get(n_states, state_labels)
        _svm_regime = _svm_names[top_idx]
        _svm_grade, _, _, _svm_meaning = _schwab_rating(top_idx, n_states)
        _kv("Most Likely Next State", S(
            f"{state_labels[top_idx]}  |  {_svm_regime}  |  {_svm_grade} – {_svm_meaning}"
            f"  ({svm_probs[top_idx]*100:.1f}%)"
        ))
        pdf.ln(2)

        if _charts:
            colors = ['#ff7f0e' if i == top_idx else '#ffbb78' for i in range(len(state_labels))]
            fig, ax = plt.subplots(figsize=(8, 2.6), facecolor='white')
            ax.bar(state_labels, svm_probs * 100, color=colors, zorder=3)
            ax.set_ylabel('Probability (%)', fontsize=8)
            ax.set_xlabel('State', fontsize=8)
            ax.set_title('Predicted Next-State Probabilities', fontsize=9, fontweight='bold', pad=4)
            ax.set_ylim(0, 100)
            ax.tick_params(labelsize=7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.18, linewidth=0.5, axis='y')
            plt.tight_layout(pad=0.5)
            _embed_chart(fig)
        else:
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(PW, 6, "Next-State Probabilities:", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            for lbl, p in zip(state_labels, svm_probs):
                pdf.cell(PW, 5, S(f"    {lbl}: {p*100:.1f}%"), new_x="LMARGIN", new_y="NEXT")

        pdf.ln(4)

    # ── Monte Carlo Simulation ────────────────────────────────────────────────
    _section("Monte Carlo Simulation")

    if _charts:
        mc_days = list(range(horizon + 1))
        fig, ax = plt.subplots(figsize=(8, 3.0), facecolor='white')
        ax.fill_between(mc_days, mc['bands'][10], mc['bands'][90],
                        alpha=0.13, color='#1f77b4', label='P10-P90')
        ax.fill_between(mc_days, mc['bands'][25], mc['bands'][75],
                        alpha=0.28, color='#1f77b4', label='P25-P75')
        ax.plot(mc_days, mc['bands'][50], color='#1f77b4', linewidth=1.8,
                label='Median (P50)', zorder=3)
        y_lo = float(mc['bands'][10].min())
        y_hi = float(mc['bands'][90].max())
        y_pad = max((y_hi - y_lo) * 0.05, 0.5)
        ax.set_ylim(y_lo - y_pad, y_hi + y_pad)
        ax.set_xlabel('Day', fontsize=8)
        ax.set_ylabel('Price ($)', fontsize=8)
        ax.set_title(f'Monte Carlo ({mc["n_simulations"]} paths, GBM)', fontsize=9, fontweight='bold', pad=4)
        ax.legend(fontsize=7, framealpha=0.6, loc='upper left')
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.18, linewidth=0.5)
        plt.tight_layout(pad=0.5)
        _embed_chart(fig)

    mc_rows = [
        ("Median End Price", f"${mc['median_end']:.2f}"),
        ("Pessimistic (P10)", f"${mc['p10_end']:.2f}"),
        ("Optimistic (P90)", f"${mc['p90_end']:.2f}"),
        ("Drift Source", S(mc["drift_source"])),
        ("Active Drift", f"{mc['drift_daily']*100:+.4f}%/day"),
    ]
    if mc["drift_source"] == "SVM-conditioned":
        mc_rows.append(("OLS Baseline Drift", f"{mc['drift_ols']*100:+.4f}%/day"))
    mc_rows += [
        ("Daily Volatility", f"{mc['sigma_daily']*100:.4f}%"),
        ("Simulations", str(mc["n_simulations"])),
    ]
    for key, val in mc_rows:
        _kv(key, val)

    pdf.ln(4)

    # ── AI Analysis ───────────────────────────────────────────────────────────
    if rag_result:
        _section("AI Analysis")
        if ai_provider:
            _badge_colors = {
                "Anthropic (Claude)": (217, 119,  87),  # #D97757 warm orange
                "OpenAI":             ( 16,  16,  16),  # near-black
                "Google Gemini":      ( 66, 133, 244),  # #4285F4 Google blue
                "DeepSeek":           ( 26, 111, 224),  # #1A6FE0 deep blue
                "Groq (Free)":        (107,  33, 168),  # #6B21A8 purple
            }
            _br, _bg, _bb = _badge_colors.get(ai_provider, (80, 80, 80))

            # "Generated by" label
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(120, 120, 120)
            _label = "Generated by "
            pdf.cell(pdf.get_string_width(_label) + 1, 7, _label, border=0)

            # Coloured badge
            pdf.set_font("Helvetica", "B", 8)
            _badge_text = S(f"  {ai_provider}  ")
            _badge_w = pdf.get_string_width(_badge_text) + 2
            pdf.set_fill_color(_br, _bg, _bb)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(_badge_w, 7, _badge_text, fill=True, border=0, new_x="LMARGIN", new_y="NEXT")

            # Reset
            pdf.set_fill_color(255, 255, 255)
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Helvetica", "", 10)
            pdf.ln(2)
        clean = S(rag_result["analysis"].replace("**", "").replace("*", ""))
        pdf.multi_cell(PW, 6, clean, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        sources = rag_result.get("sources", [])
        if sources:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(PW, 7, "Sources", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            for i, src in enumerate(sources, 1):
                title = S(src["title"] or "Untitled")
                pdf.multi_cell(PW, 5, f"{i}. {title}", new_x="LMARGIN", new_y="NEXT")
                if src["url"]:
                    pdf.set_text_color(31, 119, 180)
                    pdf.multi_cell(PW, 5, S(src["url"]), new_x="LMARGIN", new_y="NEXT")
                    pdf.set_text_color(0, 0, 0)
                relevance = src.get("relevance_summary", "")
                if relevance:
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.set_text_color(0, 6, 177)
                    pdf.multi_cell(PW, 5, S(relevance), new_x="LMARGIN", new_y="NEXT")
                    pdf.set_text_color(0, 0, 0)
                    pdf.set_font("Helvetica", "", 9)
                pdf.ln(1)

        pdf.ln(3)

    # ── Footer ────────────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(
        PW, 5,
        "This report is for educational purposes only and does not constitute financial advice. "
        "Past performance is not indicative of future results.",
        new_x="LMARGIN", new_y="NEXT",
    )

    return bytes(pdf.output())


# ── State nomenclature ────────────────────────────────────────────────────────
_QUANTILE_STATE_NAMES = {
    3:  ["Bear", "Neutral", "Bull"],
    4:  ["Drawdown", "Correction", "Recovery", "Breakout"],
    5:  ["Capitulation", "Distribution", "Consolidation", "Accumulation", "Momentum"],
    6:  ["Capitulation", "Distribution", "Drift Down", "Drift Up", "Accumulation", "Momentum"],
    7:  ["Panic", "Capitulation", "Distribution", "Consolidation", "Accumulation", "Rally", "Breakout"],
    8:  ["Panic", "Capitulation", "Distribution", "Drift Down", "Drift Up", "Accumulation", "Rally", "Breakout"],
    9:  ["Panic", "Capitulation", "Distribution", "Mild Distribution", "Consolidation", "Mild Accumulation", "Accumulation", "Rally", "Breakout"],
    10: ["Panic", "Capitulation", "Distribution", "Mild Distribution", "Drift Down", "Drift Up", "Mild Accumulation", "Accumulation", "Rally", "Breakout"],
}

_STATE_DESCRIPTIONS = {
    "Panic":              ("🔴", "Extreme selloff, forced liquidation, flash-crash behavior."),
    "Capitulation":       ("🔴", "Heavy drawdown, persistent selling, fear-driven decline."),
    "Distribution":       ("🟠", "Moderate selling pressure, profit-taking, supply overhang."),
    "Mild Distribution":  ("🟠", "Soft tape, mild below-median weakness."),
    "Drift Down":         ("🟡", "Slightly below-median returns, tentative selling."),
    "Correction":         ("🟠", "Notable decline, below-median returns, softening conviction."),
    "Drawdown":           ("🔴", "Significant decline, elevated selling, negative momentum."),
    "Neutral":            ("⚪", "Balanced order flow, range-bound, no directional conviction."),
    "Consolidation":      ("⚪", "Equilibrium between buyers and sellers, sideways tape."),
    "Drift Up":           ("🟡", "Slightly above-median returns, tentative buying."),
    "Mild Accumulation":  ("🟢", "Constructive early buying, improving sentiment."),
    "Recovery":           ("🟢", "Above-median returns, constructive tape, early accumulation."),
    "Accumulation":       ("🟢", "Institutional buying, improving breadth, positive conviction."),
    "Rally":              ("🟢", "Strong positive momentum, healthy upside continuation."),
    "Bull":               ("🟢", "Positive momentum, accumulation, or trend-continuation."),
    "Momentum":           ("🟢", "Strong upside, bullish catalyst, breakout from prior range."),
    "Breakout":           ("🟢", "Explosive gain, catalyst-driven surge, high-conviction upside."),
    # Volume states
    "Low":                ("🔴", "Return meaningfully below historical average — bearish regime."),
    "Average":            ("⚪", "Return within ½ std-dev of historical mean — neutral, transitional."),
    "High":               ("🟢", "Return materially above historical average — bullish regime."),
}

# Schwab-style A–F ratings by state count (State 0 = F, top state = A)
# Format per entry: (label, badge background, text color, meaning)
_SCHWAB_GRADE_MAP = {
    "A":  ("#14532d", "#ffffff", "Strongly Outperform"),
    "A-": ("#15803d", "#ffffff", "Outperform+"),
    "B":  ("#15803d", "#ffffff", "Outperform"),
    "B-": ("#166534", "#ffffff", "Outperform–"),
    "C+": ("#6b7280", "#ffffff", "Market Perform+"),
    "C":  ("#6b7280", "#ffffff", "Market Perform"),
    "C-": ("#78716c", "#ffffff", "Market Perform–"),
    "D+": ("#b45309", "#ffffff", "Underperform+"),
    "D":  ("#c2410c", "#ffffff", "Underperform"),
    "D-": ("#b91c1c", "#ffffff", "Underperform–"),
    "F":  ("#7f1d1d", "#ffffff", "Strongly Underperform"),
}

_SCHWAB_RATINGS_BY_COUNT = {
    3:  ["F",  "C",  "A"],
    4:  ["F",  "D",  "B",  "A"],
    5:  ["F",  "D",  "C",  "B",  "A"],
    6:  ["F",  "D",  "C-", "C+", "B",  "A"],
    7:  ["F",  "D-", "D",  "C",  "B",  "A-", "A"],
    8:  ["F",  "D-", "D",  "C-", "C+", "B",  "A-", "A"],
    9:  ["F",  "D-", "D",  "D+", "C",  "B-", "B",  "A-", "A"],
    10: ["F",  "D-", "D",  "D+", "C-", "C+", "B-", "B",  "A-", "A"],
}

def _schwab_rating(state_idx: int, total_states: int):
    """Return (grade, bg_color, text_color, meaning) for the given state index."""
    grades = _SCHWAB_RATINGS_BY_COUNT.get(total_states)
    if grades is None:
        # fallback: linearly map to F…A
        grades = list(_SCHWAB_RATINGS_BY_COUNT[5])
    grade = grades[min(state_idx, len(grades) - 1)]
    bg, fg, meaning = _SCHWAB_GRADE_MAP.get(grade, ("#6b7280", "#ffffff", "Market Perform"))
    return grade, bg, fg, meaning


st.set_page_config(page_title="Stock Predictor", layout="centered")

try:
    import sklearn  # noqa: F401
    _sklearn_available = True
except ImportError:
    _sklearn_available = False

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap');

    /* ── Global body & UI text ─────────────────────────────────────────── */
    html, body, p, li, td, th, label, input,
    .stMarkdown, .stAlert, .stCaptionContainer,
    [data-testid="stSidebar"], [data-testid="stExpander"],
    [data-testid="stDataFrame"], button {
        font-family: 'IBM Plex Sans', 'Helvetica Neue', Arial, sans-serif !important;
    }

    /* Exclude Material Icons from font override so expander arrows render correctly */
    .material-icons, .material-icons-outlined, [class*="Icon"], [class*="icon"] {
        font-family: 'Material Icons' !important;
    }

    /* ── Headings — navy serif ──────────────────────────────────────────── */
    h1, h2, h3, h4,
    [data-testid="stHeading"],
    .stTitle, .stHeader, .stSubheader {
        font-family: 'Playfair Display', Georgia, 'Times New Roman', serif !important;
        color: #00296b !important;
        letter-spacing: -0.01em;
    }

    /* ── Metric cards ───────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        border-top: 3px solid #fdc500 !important;
        border-radius: 6px !important;
        padding: 10px 14px !important;
    }
    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
        font-size: 1.8rem;
        font-weight: 500;
        letter-spacing: -0.02em;
        color: #00296b !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.2rem;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #00509d !important;
    }

    /* ── Sidebar ────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #c2f2e2 !important;
        border-right: 3px solid #fdc500 !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        font-family: 'Playfair Display', Georgia, serif !important;
        color: #00296b !important;
    }

    /* ── Primary button ─────────────────────────────────────────────────── */
    [data-testid="stBaseButton-primary"] {
        background: linear-gradient(90deg, #00296b 0%, #00509d 100%) !important;
        color: #ffd500 !important;
        border: none !important;
        font-weight: 700 !important;
        letter-spacing: 0.04em !important;
    }
    [data-testid="stBaseButton-primary"]:hover {
        background: linear-gradient(90deg, #003f88 0%, #00509d 100%) !important;
    }
    [data-testid="stBaseButton-secondary"] {
        color: #00296b !important;
        border-color: #00509d !important;
    }

    /* ── Expander header accent ──────────────────────────────────────────── */
    [data-testid="stExpander"] {
        border-left: 3px solid #fdc500 !important;
        border-radius: 4px !important;
    }

    /* ── Dividers ───────────────────────────────────────────────────────── */
    hr {
        border-color: #fdc50055 !important;
    }

    /* ── Dataframe / table text ─────────────────────────────────────────── */
    .stDataFrame, .stTable, td, th {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.82rem !important;
    }
    th {
        background: #00296b14 !important;
        color: #00296b !important;
    }

    /* ── Links ──────────────────────────────────────────────────────────── */
    a {
        color: #00509d !important;
    }
    a:hover {
        color: #003f88 !important;
    }

    /* ── Code / caption ─────────────────────────────────────────────────── */
    code, pre, .stCode {
        font-family: 'IBM Plex Mono', 'Courier New', monospace !important;
    }
    [data-testid="stCaptionContainer"] p {
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.01em;
        color: #003f88 !important;
    }

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

    _provider_badges = {
        "Anthropic (Claude)": {
            "bg": "#FDF0EC", "border": "#D97757", "text": "#A0522D",
            "icon": (
                '<svg width="18" height="18" viewBox="0 0 24 24" fill="#D97757" xmlns="http://www.w3.org/2000/svg">'
                '<path d="M13.827 3.52h3.603L24 20h-3.603l-6.57-16.48zm-3.654 0H6.57L0 20h3.603l6.57-16.48z"/>'
                '</svg>'
            ),
            "label": "Anthropic",
        },
        "OpenAI": {
            "bg": "#F5F5F5", "border": "#000000", "text": "#000000",
            "icon": (
                '<svg width="18" height="18" viewBox="0 0 24 24" fill="#000000" xmlns="http://www.w3.org/2000/svg">'
                '<path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91 6.046 6.046 0 0 0-6.51-2.9A6.065 6.065 0 0 0 4.981 4.18a5.985 5.985 0 0 0-3.998 2.9 6.046 6.046 0 0 0 .743 7.097 5.98 5.98 0 0 0 .51 4.911 6.051 6.051 0 0 0 6.515 2.9A5.985 5.985 0 0 0 13.26 24a6.056 6.056 0 0 0 5.772-4.206 5.99 5.99 0 0 0 3.997-2.9 6.056 6.056 0 0 0-.747-7.073zM13.26 22.43a4.476 4.476 0 0 1-2.876-1.04l.141-.081 4.779-2.758a.795.795 0 0 0 .392-.681v-6.737l2.02 1.168a.071.071 0 0 1 .038.052v5.583a4.504 4.504 0 0 1-4.494 4.494zM3.6 18.304a4.47 4.47 0 0 1-.535-3.014l.142.085 4.783 2.759a.771.771 0 0 0 .78 0l5.843-3.369v2.332a.08.08 0 0 1-.033.062L9.74 19.95a4.5 4.5 0 0 1-6.14-1.646zM2.34 7.896a4.485 4.485 0 0 1 2.366-1.973V11.6a.766.766 0 0 0 .388.676l5.815 3.355-2.02 1.168a.076.076 0 0 1-.071 0l-4.83-2.786A4.504 4.504 0 0 1 2.34 7.872zm16.597 3.855l-5.843-3.372L15.11 7.21a.076.076 0 0 1 .071 0l4.83 2.791a4.494 4.494 0 0 1-.676 8.105v-5.678a.79.79 0 0 0-.398-.677zm2.01-3.023l-.141-.085-4.774-2.782a.776.776 0 0 0-.785 0L9.409 9.23V6.897a.066.066 0 0 1 .028-.061l4.83-2.787a4.5 4.5 0 0 1 6.68 4.66zm-12.64 4.135l-2.02-1.164a.08.08 0 0 1-.038-.057V6.075a4.5 4.5 0 0 1 7.375-3.453l-.142.08L8.704 5.46a.795.795 0 0 0-.393.681zm1.097-2.365l2.602-1.5 2.607 1.5v2.999l-2.597 1.5-2.607-1.5z"/>'
                '</svg>'
            ),
            "label": "OpenAI",
        },
        "Google Gemini": {
            "bg": "#EEF3FE", "border": "#4285F4", "text": "#1A56C4",
            "icon": (
                '<svg width="18" height="18" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">'
                '<path d="M12 24A14.304 14.304 0 0 0 0 12 14.304 14.304 0 0 0 12 0a14.304 14.304 0 0 0 12 12 14.304 14.304 0 0 0-12 12z" fill="#4285F4"/>'
                '</svg>'
            ),
            "label": "Google Gemini",
        },
        "DeepSeek": {
            "bg": "#EAF2FF", "border": "#1A6FE0", "text": "#0D47A1",
            "icon": (
                '<svg width="18" height="18" viewBox="0 0 24 24" fill="#1A6FE0" xmlns="http://www.w3.org/2000/svg">'
                '<circle cx="12" cy="12" r="10"/>'
                '<path d="M8 12a4 4 0 0 1 8 0" stroke="#fff" stroke-width="2" fill="none" stroke-linecap="round"/>'
                '<circle cx="12" cy="14" r="2" fill="#fff"/>'
                '</svg>'
            ),
            "label": "DeepSeek",
        },
        "Groq (Free)": {
            "bg": "#F5F0FF", "border": "#6B21A8", "text": "#6B21A8",
            "icon": (
                '<svg width="18" height="18" viewBox="0 0 24 24" fill="#6B21A8" xmlns="http://www.w3.org/2000/svg">'
                '<path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/>'
                '</svg>'
            ),
            "label": "Groq (Free)",
        },
    }

    _active = st.session_state.get("_ai_provider_select", "Groq (Free)")
    _badge = _provider_badges.get(_active, _provider_badges["Groq (Free)"])
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;padding:7px 12px;'
        f'background:{_badge["bg"]};border:1.5px solid {_badge["border"]}40;'
        f'border-radius:8px;margin-bottom:6px;width:fit-content;">'
        f'{_badge["icon"]}'
        f'<span style="font-size:0.82rem;font-weight:600;color:{_badge["text"]};'
        f'font-family:\'IBM Plex Sans\',sans-serif;letter-spacing:0.02em;">'
        f'{_badge["label"]}</span></div>',
        unsafe_allow_html=True,
    )

    ai_provider = st.selectbox(
        "Provider",
        options=["Groq (Free)", "Anthropic (Claude)", "OpenAI", "Google Gemini", "DeepSeek"],
        index=0,
        key="_ai_provider_select",
        help="Choose which AI provider powers the news analysis. No API key required — shared keys are provided for all providers.",
    )
    _provider_slug_map = {
        "Anthropic (Claude)": "anthropic",
        "OpenAI": "openai",
        "Google Gemini": "gemini",
        "DeepSeek": "deepseek",
        "Groq (Free)": "groq",
    }
    _secret_key_map = {
        "Anthropic (Claude)": "ANTHROPIC_API_KEY",
        "OpenAI": "OPENAI_API_KEY",
        "Google Gemini": "GEMINI_API_KEY",
        "DeepSeek": "DEEPSEEK_API_KEY",
        "Groq (Free)": "GROQ_API_KEY",
    }
    _api_key_labels = {
        "Anthropic (Claude)": "Anthropic API Key",
        "OpenAI": "OpenAI API Key",
        "Google Gemini": "Google AI API Key",
        "DeepSeek": "DeepSeek API Key",
        "Groq (Free)": "Groq API Key",
    }
    provider_slug = _provider_slug_map[ai_provider]
    _secret_name = _secret_key_map[ai_provider]
    _secret_key = st.secrets.get(_secret_name, "") if hasattr(st, "secrets") else ""
    _has_secret = bool(_secret_key)

    ai_api_key_input = st.text_input(
        _api_key_labels[ai_provider] + (" (optional)" if _has_secret else ""),
        type="password",
        help=(
            "Leave blank to use the shared key provided by this app. Enter your own key to use your own quota."
            if _has_secret else
            "Paste your API key to enable AI-powered news analysis."
        ),
    )
    ai_api_key = ai_api_key_input or _secret_key

    if _has_secret and not ai_api_key_input:
        st.caption("Using shared app key. Enter your own key to use your quota.")

    run_rag = st.checkbox(
        "Generate AI Analysis",
        value=False,
        disabled=not ai_api_key,
        help="Uses your chosen AI provider + recent news to contextualise the simulation.",
    )
    summarize_sources = st.checkbox(
        "AI-summarize each source article",
        value=False,
        disabled=not (ai_api_key and run_rag),
        help="For each news article used, generate a one-sentence summary of its relevance to the simulation results. Uses an extra LLM call.",
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

if run_rag and ai_api_key:
    with st.spinner("Analyzing news sentiment to condition simulation..."):
        try:
            sentiment_data = get_news_sentiment(ticker, ai_api_key, provider=provider_slug)
            s = sentiment_data["sentiment"]
            if s == -1:
                sim_start_state = 0
            elif s == 1:
                sim_start_state = n_states - 1
            # s == 0: keep current_state
        except InsufficientBalanceError:
            st.warning(
                f"The shared **{ai_provider}** key has insufficient balance. "
                "Please enter your own API key in the sidebar to use AI Analysis.",
                icon="💳",
            )
            sentiment_data = None
            run_rag = False
        except Exception:
            sentiment_data = None  # fall back silently

simulation = model.simulate_prices(
    start_price=float(prices.iloc[-1]),
    start_state=sim_start_state,
    horizon=horizon,
    random_seed=int(seed),
)

# ── Results ───────────────────────────────────────────────────────────────────
st.success(f"Model fitted on **{len(prices)}** trading days of **{ticker}** data — as of **{datetime.date.today().strftime('%B %d, %Y')}**.")

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
_sim_dates = pd.bdate_range(start=prices.index[-1] + pd.Timedelta(days=1), periods=len(simulation))
sim_df = pd.DataFrame({
    "Date": _sim_dates.strftime("%b %d"),
    "Price": simulation.values,
})
y_min = sim_low * 0.99
y_max = sim_high * 1.01

line = (
    alt.Chart(sim_df)
    .mark_line(color="#1f77b4")
    .encode(
        x=alt.X("Date:O", sort=None, axis=alt.Axis(labelAngle=-45)),
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
_name_lookup = _QUANTILE_STATE_NAMES.get(n_states, state_labels)

def _state_badge_html(state_idx: int, regime_name: str, total: int) -> str:
    dot, desc = _STATE_DESCRIPTIONS.get(regime_name, ("⚪", ""))
    grade, bg, fg, meaning = _schwab_rating(state_idx, total)
    return (
        f'<div style="margin-top:4px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
        f'<span style="font-size:0.9rem;font-weight:600;color:#00509d;">{dot} {regime_name}</span>'
        f'<span style="font-size:0.82rem;font-weight:800;padding:2px 10px;border-radius:4px;'
        f'background:{bg};color:{fg};letter-spacing:0.04em;">{grade}</span>'
        f'<span style="font-size:0.78rem;color:#555;">{meaning}</span>'
        f'</div>'
        f'<div style="font-size:0.78rem;color:#555;margin-top:2px;">{desc}</div>'
    )

col_a, col_b = st.columns(2)
with col_a:
    _cur_name = _name_lookup[current_state]
    st.metric("Current State", state_labels[current_state])
    st.markdown(_state_badge_html(current_state, _cur_name, n_states), unsafe_allow_html=True)
with col_b:
    _nxt_idx = model.most_likely_next_state(current_state)
    _nxt_name = _name_lookup[_nxt_idx]
    st.metric("Most Likely Next State", state_labels[_nxt_idx])
    st.markdown(_state_badge_html(_nxt_idx, _nxt_name, n_states), unsafe_allow_html=True)
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
    _rname = _name_lookup[i]
    _grade, _, _, _meaning = _schwab_rating(i, n_states)
    state_rows.append({
        "State": state_labels[i],
        "Regime": _rname,
        "Grade": _grade,
        "Rating": _meaning,
        "Return Range": f"{model.state_bins[i]:.4f}  →  {model.state_bins[i + 1]:.4f}",
        "Mean Return": f"{model.state_mean_returns[i]:.5f}",
        "Observations": int(model.initial_state_counts[i]),
    })
st.dataframe(pd.DataFrame(state_rows).set_index("State"), width="stretch")

with st.expander("What do these states mean?"):
    st.caption(
        "Each state represents a **daily return regime** observed in this stock's price history. "
        "States are ordered from most bearish (State 0) to most bullish (State N). "
        "The colour dot gives an at-a-glance directional signal."
    )
    _name_list = _QUANTILE_STATE_NAMES.get(n_states)
    for i, label in enumerate(state_labels):
        regime_name = _name_list[i] if _name_list else label
        dot, desc = _STATE_DESCRIPTIONS.get(regime_name, ("⚪", "Custom state."))
        grade, bg, fg, meaning = _schwab_rating(i, n_states)
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:10px;margin:6px 0;">'
            f'<span style="font-size:1.1rem;line-height:1.5;">{dot}</span>'
            f'<div>'
            f'<span style="font-weight:600;color:#00296b;">{label}</span>'
            f'<span style="color:#00509d;font-size:0.88rem;font-weight:500;"> — {regime_name}</span>'
            f'&nbsp;<span style="font-size:0.78rem;font-weight:800;padding:1px 9px;border-radius:4px;'
            f'background:{bg};color:{fg};letter-spacing:0.04em;">{grade}</span>'
            f'&nbsp;<span style="color:#555;font-size:0.80rem;">{meaning}</span>'
            f'<br><span style="color:#666;font-size:0.82rem;">{desc}</span>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ── Monte Carlo Simulation ────────────────────────────────────────────────────
# ── SVM RBF Prediction ────────────────────────────────────────────────────────
st.subheader("SVM (RBF) Prediction")
st.caption(
    "A Support Vector Machine (SVM) with a Radial Basis Function (RBF) kernel is trained on engineered features (lagged returns, "
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
        _svm_dates = pd.bdate_range(start=prices.index[-1] + pd.Timedelta(days=1), periods=len(svm_simulation))
        svm_sim_df = pd.DataFrame({
            "Date": _svm_dates.strftime("%b %d"),
            "Price": svm_simulation.values,
        })
        svm_y_min = float(svm_simulation.min()) * 0.99
        svm_y_max = float(svm_simulation.max()) * 1.01
        svm_line = (
            alt.Chart(svm_sim_df)
            .mark_line(color="#ff7f0e")
            .encode(
                x=alt.X("Date:O", sort=None, axis=alt.Axis(labelAngle=-45)),
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
        _svm_top_name = _name_lookup[top_idx]
        _svm_dot, _ = _STATE_DESCRIPTIONS.get(_svm_top_name, ("⚪", ""))
        _svm_grade, _svm_bg, _svm_fg, _svm_meaning = _schwab_rating(top_idx, n_states)
        st.markdown(
            f'Most likely next state: **{state_labels[top_idx]}** — '
            f'{_svm_dot} {_svm_top_name} '
            f'<span style="font-size:0.78rem;font-weight:800;padding:1px 9px;border-radius:4px;'
            f'background:{_svm_bg};color:{_svm_fg};letter-spacing:0.04em;">{_svm_grade}</span> '
            f'<span style="font-size:0.80rem;color:#555;">{_svm_meaning}</span> '
            f'({svm_probs[top_idx] * 100:.1f}%)',
            unsafe_allow_html=True,
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
    _params = [
        ("Drift Source", mc["drift_source"]),
        ("Daily Drift", f"{mc['drift_daily'] * 100:+.4f}%"),
    ]
    if mc["drift_source"] == "SVM-conditioned":
        _params.append(("OLS Baseline", f"{mc['drift_ols'] * 100:+.4f}%"))
    _params += [
        ("Daily Volatility", f"{mc['sigma_daily'] * 100:.4f}%"),
        ("Simulated Paths", f"{mc['n_simulations']:,}"),
    ]
    _cells = "".join(
        f'<td style="padding:0 18px 0 0; white-space:nowrap;">'
        f'<span style="display:block;font-size:0.68rem;font-weight:600;letter-spacing:0.07em;'
        f'text-transform:uppercase;color:#888;font-family:\'IBM Plex Sans\',sans-serif;">{k}</span>'
        f'<span style="font-size:0.9rem;font-weight:500;font-family:\'IBM Plex Mono\',monospace;">{v}</span>'
        f'</td>'
        for k, v in _params
    )
    st.markdown(f'<table style="border:none;border-collapse:collapse;margin:4px 0 2px 0;"><tr>{_cells}</tr></table>', unsafe_allow_html=True)

st.divider()

# ── AI Analysis (RAG) ─────────────────────────────────────────────────────────
rag_result = None
if run_rag and ai_api_key:
    st.subheader("AI Analysis")
    st.caption(
        f"{ai_provider} reads recent news about this ticker and combines it with the Markov model "
        "output to produce an educational summary. This is NOT financial advice."
    )
    with st.spinner("Fetching news and generating analysis..."):
        try:
            result = run_rag_analysis(
                ticker=ticker,
                api_key=ai_api_key,
                provider=provider_slug,
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
                summarize_sources=summarize_sources,
            )
            rag_result = result
            st.markdown(result["analysis"])

            sources = result["sources"]
            if sources:
                with st.expander(f"Sources ({len(sources)} articles used)"):
                    for i, src in enumerate(sources, start=1):
                        title = src["title"] or "Untitled"
                        url = src["url"]
                        if url:
                            st.markdown(f"**{i}. [{title}]({url})**")
                        else:
                            st.markdown(f"**{i}. {title}**")
                        ai_summary = src.get("relevance_summary")
                        if ai_summary:
                            st.caption(f"**Relevance:** {ai_summary}")
                        else:
                            fallback = src["text"][:280].rstrip() + ("…" if len(src["text"]) > 280 else "")
                            st.caption(fallback)
                        if i < len(sources):
                            st.divider()
        except ImportError as e:
            st.warning(
                f"Missing dependency: {e}. "
                "Install with: `pip install anthropic chromadb sentence-transformers`"
            )
        except InsufficientBalanceError:
            st.warning(
                f"The shared **{ai_provider}** key has insufficient balance. "
                "Please enter your own API key in the sidebar to use AI Analysis.",
                icon="💳",
            )
        except Exception as e:
            st.error(f"AI analysis failed: {e}")
    st.divider()
elif run_rag and not ai_api_key:
    st.info("Enter your API key in the sidebar to enable AI analysis.")
    st.divider()

# ── PDF Download ──────────────────────────────────────────────────────────────
st.subheader("Download Report")
try:
    pdf_bytes = _generate_pdf(
        ticker=ticker,
        period=period,
        horizon=horizon,
        seed=seed,
        n_states=n_states,
        current_price=float(prices.iloc[-1]),
        simulation=simulation,
        sim_high=sim_high,
        sim_low=sim_low,
        current_state=current_state,
        state_labels=state_labels,
        model=model,
        svm_probs=svm_probs,
        svm_simulation=svm_simulation,
        mc=mc,
        sentiment_data=sentiment_data,
        rag_result=rag_result,
        ai_provider=ai_provider if rag_result else None,
        last_price_date=prices.index[-1],
    )
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name=f"{ticker}_prediction_{datetime.date.today()}.pdf",
        mime="application/pdf",
    )
except ImportError as _pdf_err:
    st.info(f"{_pdf_err}")
except Exception as _pdf_err:
    st.error(f"PDF generation failed: {_pdf_err}")

st.caption("This tool is for educational purposes only. NOT financial advice.")
