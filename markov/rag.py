"""
rag.py — News-augmented analysis using the Claude API
======================================================

Pipeline:
1. fetch_news          — Pull recent news headlines + summaries via yfinance
2. get_news_sentiment  — Ask Claude to classify sentiment (-1 bearish / 0 neutral / 1 bullish)
3. run_rag_analysis    — Pass all articles + model output to Claude, return analysis + sources
"""

from __future__ import annotations

from typing import Any

import yfinance as yf


# ── News fetching ─────────────────────────────────────────────────────────────

def fetch_news(ticker: str, max_articles: int = 20) -> list[dict[str, str]]:
    """Return recent news articles for *ticker*.

    Each dict has keys: ``title``, ``summary``, ``url``, ``text``.
    Articles with no usable text are silently dropped.
    """
    stock = yf.Ticker(ticker)
    raw: list[dict[str, Any]] = stock.news or []

    articles: list[dict[str, str]] = []
    for item in raw[:max_articles]:
        content = item.get("content", {})
        title: str = content.get("title") or item.get("title") or ""
        summary: str = (
            content.get("summary")
            or content.get("description")
            or item.get("summary")
            or ""
        )
        url: str = (
            content.get("canonicalUrl", {}).get("url")
            or content.get("clickThroughUrl", {}).get("url")
            or item.get("link")
            or ""
        )
        text = f"{title} {summary}".strip()
        if text:
            articles.append({"title": title, "summary": summary, "url": url, "text": text})

    return articles


# ── Sentiment classification ──────────────────────────────────────────────────

_SENTIMENT_TOOL = {
    "name": "report_sentiment",
    "description": (
        "Report the overall market sentiment for the stock based on recent news headlines "
        "and summaries. Use -1 for clearly bearish signals (bad earnings, downgrades, macro "
        "headwinds, scandal), 0 for mixed or neutral news, and 1 for clearly bullish signals "
        "(strong earnings, upgrades, major product launches, positive guidance)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "integer",
                "enum": [-1, 0, 1],
                "description": "-1 = bearish, 0 = neutral, 1 = bullish",
            },
            "reasoning": {
                "type": "string",
                "description": "One or two sentences explaining the sentiment assessment.",
            },
        },
        "required": ["sentiment", "reasoning"],
    },
}


def get_news_sentiment(ticker: str, api_key: str) -> dict:
    """Fetch recent news and ask Claude to classify sentiment.

    Returns
    -------
    dict with keys:
        ``sentiment`` (int)  — -1 bearish, 0 neutral, 1 bullish.
        ``reasoning`` (str)  — Claude's one-sentence explanation.
        ``articles``  (list) — Raw articles fetched (for reuse in run_rag_analysis).
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "RAG requires the anthropic package. Install it with: pip install anthropic"
        ) from exc

    articles = fetch_news(ticker)
    if not articles:
        return {"sentiment": 0, "reasoning": "No recent news found.", "articles": []}

    headlines = "\n".join(
        f"- {a['title']}: {a['summary'] or a['text'][:120]}" for a in articles
    )
    prompt = (
        f"Below are recent news headlines and summaries for {ticker}. "
        f"Classify the overall market sentiment.\n\n{headlines}"
    )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=256,
        tools=[_SENTIMENT_TOOL],
        tool_choice={"type": "tool", "name": "report_sentiment"},
        messages=[{"role": "user", "content": prompt}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "report_sentiment":
            return {
                "sentiment": block.input["sentiment"],
                "reasoning": block.input["reasoning"],
                "articles": articles,
            }

    # Fallback if tool use block is missing (shouldn't happen with tool_choice forced)
    return {"sentiment": 0, "reasoning": "Could not determine sentiment.", "articles": articles}


# ── Prompt construction ───────────────────────────────────────────────────────

def _build_prompt(
    ticker: str,
    current_price: float,
    simulated_end_price: float,
    sim_change_pct: float,
    sim_high: float,
    sim_low: float,
    current_state_label: str,
    next_state_label: str,
    horizon: int,
    articles: list[dict[str, str]],
    monte_carlo: dict | None = None,
    svm_probs: list | None = None,
    state_labels: list | None = None,
) -> str:
    news_block = "\n\n".join(
        f"[{i+1}] {a['title']}\n{a['summary'] or a['text']}"
        for i, a in enumerate(articles)
    ) or "No recent news available."

    mc_block = ""
    if monte_carlo:
        drift_note = (
            f"drift source: {monte_carlo.get('drift_source', 'OLS regression')} "
            f"({monte_carlo['drift_daily'] * 100:+.4f}%/day)"
        )
        if monte_carlo.get("drift_source") == "SVM-conditioned":
            drift_note += (
                f"; OLS baseline was {monte_carlo['drift_ols'] * 100:+.4f}%/day"
            )
        mc_block = f"""
## Monte Carlo Simulation Output for {ticker} ({monte_carlo['n_simulations']} paths, GBM)

- {drift_note}
- Historical daily volatility: {monte_carlo['sigma_daily'] * 100:.4f}%
- Median end price after {horizon} days: ${monte_carlo['median_end']:.2f}
- Mean end price: ${monte_carlo['mean_end']:.2f}
- Pessimistic scenario (10th percentile): ${monte_carlo['p10_end']:.2f}
- Optimistic scenario (90th percentile): ${monte_carlo['p90_end']:.2f}
"""

    svm_block = ""
    if svm_probs is not None and state_labels is not None:
        top_state = state_labels[int(svm_probs.argmax())]
        top_prob = float(svm_probs.max())
        prob_lines = "\n".join(
            f"- {state_labels[i]}: {p * 100:.1f}%"
            for i, p in enumerate(svm_probs)
        )
        svm_block = f"""
## SVM RBF Model — Predicted Next-State Probabilities for {ticker}

- Most likely next state: {top_state} ({top_prob * 100:.1f}% probability)
{prob_lines}
"""

    has_mc = bool(monte_carlo)
    has_svm = bool(svm_probs is not None)

    if has_mc and has_svm:
        compare_instruction = (
            "2. Compare all three models (Markov, Monte Carlo, SVM) — "
            "note where they agree or diverge and what that implies about confidence and uncertainty."
        )
    elif has_mc:
        compare_instruction = (
            "2. Compare the Markov chain and Monte Carlo outputs — "
            "note where they agree or diverge and what that means for uncertainty."
        )
    elif has_svm:
        compare_instruction = (
            "2. Compare the Markov chain and SVM outputs — "
            "note where they agree or diverge and what that implies."
        )
    else:
        compare_instruction = "2. Note the key uncertainties in the model output."

    return f"""You are a financial analyst assistant. Your job is to combine quantitative model output with recent news to produce a balanced, educational analysis for a retail investor. Do NOT give buy/sell advice.

## Markov Chain Model Output for {ticker}

- Current price: ${current_price:.2f}
- Simulated price after {horizon} days: ${simulated_end_price:.2f} ({sim_change_pct:+.1f}%)
- Simulated range: ${sim_low:.2f} – ${sim_high:.2f}
- Current market state: {current_state_label}
- Most likely next state: {next_state_label}
{mc_block}{svm_block}
## Recent News Headlines & Summaries

{news_block}

## Task

Write a concise analysis (3–5 paragraphs) that:
1. Summarises what the Markov chain model suggests about near-term price behaviour.
{compare_instruction}
3. Highlights the most relevant news themes and how they may align with or contradict the model output.
4. Notes key risks or uncertainties none of the models can capture.
5. Ends with a one-sentence disclaimer reminding the reader this is educational, not financial advice.

Be direct, factual, and avoid speculation beyond what the data supports."""


# ── Main entry point ──────────────────────────────────────────────────────────

def run_rag_analysis(
    *,
    ticker: str,
    api_key: str,
    current_price: float,
    simulated_end_price: float,
    sim_change_pct: float,
    sim_high: float,
    sim_low: float,
    current_state_label: str,
    next_state_label: str,
    horizon: int,
    articles: list | None = None,
    monte_carlo: dict | None = None,
    svm_probs=None,
    state_labels: list | None = None,
) -> dict:
    """Fetch news (or reuse pre-fetched articles), call Claude, return analysis + sources.

    Pass ``articles`` from a prior ``get_news_sentiment`` call to avoid a second
    network round-trip to Yahoo Finance.

    Returns
    -------
    dict with keys:
        ``analysis`` (str)  — Analysis text produced by Claude.
        ``sources``  (list) — Articles passed to Claude, each with
                              keys ``title``, ``url``, ``text``.
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "RAG requires the anthropic package. Install it with: pip install anthropic"
        ) from exc

    if articles is None:
        articles = fetch_news(ticker)
    if not articles:
        return {
            "analysis": (
                "No recent news articles were found for this ticker. "
                "The Markov model output is shown above."
            ),
            "sources": [],
        }

    prompt = _build_prompt(
        ticker=ticker,
        current_price=current_price,
        simulated_end_price=simulated_end_price,
        sim_change_pct=sim_change_pct,
        sim_high=sim_high,
        sim_low=sim_low,
        current_state_label=current_state_label,
        next_state_label=next_state_label,
        horizon=horizon,
        articles=articles,
        monte_carlo=monte_carlo,
        svm_probs=svm_probs,
        state_labels=state_labels,
    )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=[
            {
                "type": "text",
                "text": (
                    "You are a helpful financial education assistant. "
                    "Always remind users that your output is for educational purposes only "
                    "and does not constitute financial advice."
                ),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": prompt}],
    )

    text_parts = [block.text for block in response.content if block.type == "text"]
    analysis = "\n\n".join(text_parts).strip()

    # Deduplicate by URL before returning
    seen: set = set()
    sources: list = []
    for a in articles:
        key = a["url"] or a["title"]
        if key and key not in seen:
            seen.add(key)
            sources.append(a)

    return {"analysis": analysis, "sources": sources}
