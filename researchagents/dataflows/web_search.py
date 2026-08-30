"""LLM-native web search: OpenAI Responses API or Claude's server-side tool.

Both providers run the search on their own infrastructure and return a
synthesized, cited answer — no separate search-API key needed beyond the
LLM API key itself. Failures return an error string instead of raising so a
review run degrades gracefully to arXiv-only.
"""

from __future__ import annotations

from typing import Optional

DEFAULT_OPENAI_SEARCH_MODEL = "gpt-5.4"
DEFAULT_ANTHROPIC_SEARCH_MODEL = "claude-opus-5"

_PROMPT = (
    "Search the web for: {query}\n\n"
    "Summarize what you find with an emphasis on recent research papers, "
    "preprints, blog posts from research labs, and open-source projects. "
    "Cite sources with URLs and publication dates."
)


def search_web(
    query: str,
    provider: str = "openai",
    model: Optional[str] = None,
    max_uses: int = 3,
) -> str:
    """Run a web search through the configured LLM provider's native tool."""
    try:
        if provider == "anthropic":
            return _anthropic_web_search(query, model, max_uses)
        if provider == "openai":
            return _openai_web_search(query, model)
        return f"Web search failed: unsupported provider '{provider}'."
    except Exception as exc:  # degrade to arXiv-only rather than kill the run
        return f"Web search failed ({provider}): {exc}"


def _openai_web_search(query: str, model: Optional[str]) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=model or DEFAULT_OPENAI_SEARCH_MODEL,
        tools=[{"type": "web_search"}],
        input=_PROMPT.format(query=query),
    )
    text = response.output_text
    return text if text else f"Web search returned no text for query '{query}'."


def _anthropic_web_search(query: str, model: Optional[str], max_uses: int) -> str:
    import anthropic

    client = anthropic.Anthropic()
    resolved_model = model or DEFAULT_ANTHROPIC_SEARCH_MODEL

    messages = [{"role": "user", "content": _PROMPT.format(query=query)}]

    # Opus 5 / Fable 5: opt into server-side refusal fallbacks so a safety
    # decline reroutes to a fallback model instead of losing the search.
    request_kwargs = {}
    if resolved_model.startswith(("claude-opus-5", "claude-fable-5")):
        request_kwargs = {
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
        }

    response = None
    # Server tools may pause the turn; continue until the turn completes.
    for _ in range(5):
        response = client.beta.messages.create(
            model=resolved_model,
            max_tokens=16000,
            tools=[
                {
                    "type": "web_search_20260209",
                    "name": "web_search",
                    "max_uses": max_uses,
                }
            ],
            messages=messages,
            **request_kwargs,
        )
        if response.stop_reason != "pause_turn":
            break
        messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason == "refusal":
        return f"Web search declined for query '{query}'."

    text = "\n".join(
        block.text for block in response.content if block.type == "text"
    ).strip()
    return text if text else f"Web search returned no text for query '{query}'."
