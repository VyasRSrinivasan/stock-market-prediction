"""
rag.py — News-augmented analysis using the Claude API
======================================================

Pipeline:
1. fetch_news       — Pull recent news headlines + summaries via yfinance
2. run_rag_analysis — Pass all articles + model output to Claude, return analysis + sources
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
) -> str:
    news_block = "\n\n".join(
        f"[{i+1}] {a['title']}\n{a['summary'] or a['text']}"
        for i, a in enumerate(articles)
    ) or "No recent news available."

    return f"""You are a financial analyst assistant. Your job is to combine quantitative model output with recent news to produce a balanced, educational analysis for a retail investor. Do NOT give buy/sell advice.

## Markov Chain Model Output for {ticker}

- Current price: ${current_price:.2f}
- Simulated price after {horizon} days: ${simulated_end_price:.2f} ({sim_change_pct:+.1f}%)
- Simulated range: ${sim_low:.2f} – ${sim_high:.2f}
- Current market state: {current_state_label}
- Most likely next state: {next_state_label}

## Recent News Headlines & Summaries

{news_block}

## Task

Write a concise analysis (3–5 paragraphs) that:
1. Summarises what the Markov model suggests about near-term price behaviour.
2. Highlights the most relevant news themes and how they may align with or contradict the model output.
3. Notes key risks or uncertainties the model cannot capture.
4. Ends with a one-sentence disclaimer reminding the reader this is educational, not financial advice.

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
) -> dict:
    """Fetch news, call Claude, return analysis text and sources.

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
