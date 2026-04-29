"""
rag.py — News-augmented analysis using multiple LLM providers
=============================================================

Pipeline:
1. fetch_news          — Pull recent news headlines + summaries via yfinance
2. get_news_sentiment  — Ask the chosen LLM to classify sentiment (-1/0/1)
3. run_rag_analysis    — Pass all articles + model output to the LLM, return analysis + sources

Supported providers: "anthropic" | "openai" | "gemini" | "deepseek"
"""

from __future__ import annotations

import json
from typing import Any

import yfinance as yf


# ── Provider configuration ────────────────────────────────────────────────────

_PROVIDER_CONFIG: dict[str, dict[str, str | None]] = {
    "anthropic": {"model": "claude-opus-4-6",          "base_url": None},
    "openai":    {"model": "gpt-4o",                   "base_url": None},
    "gemini":    {"model": "gemini-2.0-flash",         "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"},
    "deepseek":  {"model": "deepseek-chat",            "base_url": "https://api.deepseek.com"},
    "groq":      {"model": "llama-3.3-70b-versatile",  "base_url": "https://api.groq.com/openai/v1"},
}

_SYSTEM_PROMPT = (
    "You are a helpful financial education assistant. "
    "Always remind users that your output is for educational purposes only "
    "and does not constitute financial advice."
)


class InsufficientBalanceError(Exception):
    """Raised when a provider rejects the request due to billing/quota limits."""


def _is_balance_error(exc: Exception) -> bool:
    # HTTP status codes: 402 Payment Required, 429 Too Many Requests / quota
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in (402, 429):
        return True
    # SDK exception class names (e.g. openai.RateLimitError, anthropic.APIStatusError)
    type_name = type(exc).__name__.lower()
    if any(k in type_name for k in ("ratelimit", "quota", "billing", "payment", "insufficient")):
        return True
    # Fall back to message string scan
    msg = str(exc).lower()
    return any(k in msg for k in (
        "insufficient_quota", "insufficient_balance", "insufficient balance",
        "billing", "quota exceeded", "rate limit", "rate_limit",
        "payment required", "you exceeded", "credit", "funds", "402", "429",
    ))


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


# ── Low-level LLM helpers ─────────────────────────────────────────────────────

_SENTIMENT_TOOL_ANTHROPIC = {
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


def _sentiment_anthropic(api_key: str, prompt: str) -> dict:
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError("Install the anthropic package: pip install anthropic") from exc

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=_PROVIDER_CONFIG["anthropic"]["model"],
            max_tokens=256,
            tools=[_SENTIMENT_TOOL_ANTHROPIC],
            tool_choice={"type": "tool", "name": "report_sentiment"},
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        if _is_balance_error(exc):
            raise InsufficientBalanceError(str(exc)) from exc
        raise
    for block in response.content:
        if block.type == "tool_use" and block.name == "report_sentiment":
            return {
                "sentiment": block.input["sentiment"],
                "reasoning": block.input["reasoning"],
            }
    return {"sentiment": 0, "reasoning": "Could not determine sentiment."}


def _sentiment_openai_compat(api_key: str, prompt: str, provider: str) -> dict:
    try:
        import openai
    except ImportError as exc:
        raise ImportError("Install the openai package: pip install openai") from exc

    cfg = _PROVIDER_CONFIG[provider]
    kwargs: dict[str, Any] = {"api_key": api_key}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    client = openai.OpenAI(**kwargs)

    full_prompt = (
        prompt
        + '\n\nRespond with JSON only, no other text: {"sentiment": <integer -1, 0, or 1>, "reasoning": "<one or two sentences>"}'
    )
    try:
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=256,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        if _is_balance_error(exc):
            raise InsufficientBalanceError(str(exc)) from exc
        raise
    raw = response.choices[0].message.content or "{}"
    data = json.loads(raw)
    return {
        "sentiment": int(data.get("sentiment", 0)),
        "reasoning": data.get("reasoning", ""),
    }


def _analysis_anthropic(api_key: str, prompt: str) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError("Install the anthropic package: pip install anthropic") from exc

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model=_PROVIDER_CONFIG["anthropic"]["model"],
            max_tokens=1024,
            thinking={"type": "adaptive"},
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        if _is_balance_error(exc):
            raise InsufficientBalanceError(str(exc)) from exc
        raise
    text_parts = [block.text for block in response.content if block.type == "text"]
    return "\n\n".join(text_parts).strip()


def _analysis_openai_compat(api_key: str, prompt: str, provider: str) -> str:
    try:
        import openai
    except ImportError as exc:
        raise ImportError("Install the openai package: pip install openai") from exc

    cfg = _PROVIDER_CONFIG[provider]
    kwargs: dict[str, Any] = {"api_key": api_key}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    client = openai.OpenAI(**kwargs)

    try:
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024,
        )
    except Exception as exc:
        if _is_balance_error(exc):
            raise InsufficientBalanceError(str(exc)) from exc
        raise
    return (response.choices[0].message.content or "").strip()


# ── Source summarization ──────────────────────────────────────────────────────

def _summarize_sources_llm(
    articles: list[dict[str, str]],
    ticker: str,
    current_price: float,
    simulated_end_price: float,
    sim_change_pct: float,
    horizon: int,
    api_key: str,
    provider: str,
) -> list[str]:
    """Return one relevance sentence per article (same order as *articles*).

    Makes a single batched LLM call and parses the JSON response.
    Falls back to empty strings if parsing fails.
    """
    context = (
        f"{ticker} — current price ${current_price:.2f}, "
        f"simulated {horizon}-day end ${simulated_end_price:.2f} ({sim_change_pct:+.1f}%)"
    )
    numbered = "\n\n".join(
        f"[{i+1}] {a['title']}\n{a['summary'] or a['text'][:300]}"
        for i, a in enumerate(articles)
    )
    prompt = (
        f"Simulation context: {context}\n\n"
        f"For each article below, write exactly one sentence explaining how it is relevant "
        f"(or not) to the simulation result and {ticker}'s near-term outlook.\n\n"
        f"{numbered}\n\n"
        f'Respond with JSON only: {{"summaries": ["<sentence 1>", "<sentence 2>", ...]}}'
    )

    raw = ""
    try:
        if provider == "anthropic":
            try:
                import anthropic
            except ImportError as exc:
                raise ImportError("Install anthropic: pip install anthropic") from exc
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=_PROVIDER_CONFIG["anthropic"]["model"],
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = "".join(b.text for b in response.content if b.type == "text")
        else:
            try:
                import openai
            except ImportError as exc:
                raise ImportError("Install openai: pip install openai") from exc
            cfg = _PROVIDER_CONFIG[provider]
            kwargs: dict[str, Any] = {"api_key": api_key}
            if cfg["base_url"]:
                kwargs["base_url"] = cfg["base_url"]
            client = openai.OpenAI(**kwargs)
            response = client.chat.completions.create(
                model=cfg["model"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""

        data = json.loads(raw)
        summaries = data.get("summaries", [])
        # Pad or trim to match article count
        result = [str(s) for s in summaries]
        while len(result) < len(articles):
            result.append("")
        return result[: len(articles)]
    except Exception:
        return [""] * len(articles)


# ── Public API ────────────────────────────────────────────────────────────────

def get_news_sentiment(ticker: str, api_key: str, *, provider: str = "anthropic") -> dict:
    """Fetch recent news and ask the chosen LLM to classify sentiment.

    Returns
    -------
    dict with keys:
        ``sentiment`` (int)  — -1 bearish, 0 neutral, 1 bullish.
        ``reasoning`` (str)  — Model's one-sentence explanation.
        ``articles``  (list) — Raw articles fetched (for reuse in run_rag_analysis).
    """
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

    if provider == "anthropic":
        result = _sentiment_anthropic(api_key, prompt)
    else:
        result = _sentiment_openai_compat(api_key, prompt, provider)

    return {**result, "articles": articles}


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
    svm_probs=None,
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
## SVM RBF Model -- Predicted Next-State Probabilities for {ticker}

- Most likely next state: {top_state} ({top_prob * 100:.1f}% probability)
{prob_lines}
"""

    has_mc = bool(monte_carlo)
    has_svm = svm_probs is not None

    if has_mc and has_svm:
        compare_instruction = (
            "2. Compare all three models (Markov, Monte Carlo, SVM) -- "
            "note where they agree or diverge and what that implies about confidence and uncertainty."
        )
    elif has_mc:
        compare_instruction = (
            "2. Compare the Markov chain and Monte Carlo outputs -- "
            "note where they agree or diverge and what that means for uncertainty."
        )
    elif has_svm:
        compare_instruction = (
            "2. Compare the Markov chain and SVM outputs -- "
            "note where they agree or diverge and what that implies."
        )
    else:
        compare_instruction = "2. Note the key uncertainties in the model output."

    return f"""You are a financial analyst assistant. Your job is to combine quantitative model output with recent news to produce a balanced, educational analysis for a retail investor. Do NOT give buy/sell advice.

## Markov Chain Model Output for {ticker}

- Current price: ${current_price:.2f}
- Simulated price after {horizon} days: ${simulated_end_price:.2f} ({sim_change_pct:+.1f}%)
- Simulated range: ${sim_low:.2f} - ${sim_high:.2f}
- Current market state: {current_state_label}
- Most likely next state: {next_state_label}
{mc_block}{svm_block}
## Recent News Headlines & Summaries

{news_block}

## Task

Write a concise analysis (3-5 paragraphs) that:
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
    provider: str = "anthropic",
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
    summarize_sources: bool = False,
) -> dict:
    """Fetch news (or reuse pre-fetched articles), call the chosen LLM, return analysis + sources.

    Returns
    -------
    dict with keys:
        ``analysis`` (str)  -- Analysis text.
        ``sources``  (list) -- Articles, each with keys ``title``, ``url``, ``text``,
                               and optionally ``relevance_summary`` if summarize_sources=True.
    """
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

    if provider == "anthropic":
        analysis = _analysis_anthropic(api_key, prompt)
    else:
        analysis = _analysis_openai_compat(api_key, prompt, provider)

    # Deduplicate by URL before returning
    seen: set = set()
    sources: list = []
    for a in articles:
        key = a["url"] or a["title"]
        if key and key not in seen:
            seen.add(key)
            sources.append(dict(a))

    if summarize_sources and sources:
        relevance = _summarize_sources_llm(
            articles=sources,
            ticker=ticker,
            current_price=current_price,
            simulated_end_price=simulated_end_price,
            sim_change_pct=sim_change_pct,
            horizon=horizon,
            api_key=api_key,
            provider=provider,
        )
        for src, summary in zip(sources, relevance):
            if summary:
                src["relevance_summary"] = summary

    return {"analysis": analysis, "sources": sources}
