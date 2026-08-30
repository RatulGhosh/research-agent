from unittest.mock import patch

from researchagents.dataflows.web_search import search_web


def test_openai_provider_routes_to_responses_api():
    with patch(
        "researchagents.dataflows.web_search._openai_web_search",
        return_value="synthesized answer",
    ) as mock_search:
        result = search_web("tool use in small LMs", provider="openai")
    assert result == "synthesized answer"
    mock_search.assert_called_once()


def test_anthropic_provider_routes_to_claude():
    with patch(
        "researchagents.dataflows.web_search._anthropic_web_search",
        return_value="claude answer",
    ) as mock_search:
        result = search_web("tool use in small LMs", provider="anthropic", max_uses=2)
    assert result == "claude answer"
    mock_search.assert_called_once_with("tool use in small LMs", None, 2)


def test_unsupported_provider_returns_error_string():
    result = search_web("query", provider="bing")
    assert "unsupported provider" in result


def test_provider_failure_degrades_to_error_string():
    with patch(
        "researchagents.dataflows.web_search._openai_web_search",
        side_effect=RuntimeError("no API key"),
    ):
        result = search_web("query", provider="openai")
    assert result.startswith("Web search failed (openai)")
