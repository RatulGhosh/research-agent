from unittest.mock import patch

import pytest

from researchagents.default_config import DEFAULT_CONFIG
from researchagents.graph.research_graph import ROLE_TIERS, ResearchAgentsGraph
from researchagents.graph.setup import GraphSetup


class FakeLLM:
    def __init__(self, provider, model, base_url=None):
        self.provider = provider
        self.model = model
        self.base_url = base_url


def make_graph(config_overrides):
    config = {
        **DEFAULT_CONFIG,
        "web_search": {"enabled": False},
        **config_overrides,
    }
    with patch(
        "researchagents.graph.research_graph.create_llm",
        side_effect=lambda provider, model, base_url=None: FakeLLM(provider, model, base_url),
    ):
        return ResearchAgentsGraph(config=config)


def test_role_overrides_resolve_provider_and_model():
    graph = make_graph(
        {
            "role_llms": {
                "Critic": {"provider": "anthropic", "model": "claude-opus-5"},
                "Research Manager": {"provider": "anthropic", "model": "claude-opus-5"},
            }
        }
    )
    assert graph.role_llms["Critic"].provider == "anthropic"
    assert graph.role_llms["Critic"].model == "claude-opus-5"
    # Unlisted roles fall back to tier defaults
    assert "Advocate" not in graph.role_llms
    assert graph.quick_thinking_llm.provider == DEFAULT_CONFIG["llm_provider"]


def test_role_override_model_defaults_to_tier_model():
    graph = make_graph({"role_llms": {"Program Director": {"provider": "anthropic"}}})
    # deep-tier role with no model -> deep_think_llm name under the new provider
    assert graph.role_llms["Program Director"].provider == "anthropic"
    assert graph.role_llms["Program Director"].model == DEFAULT_CONFIG["deep_think_llm"]


def test_identical_specs_share_one_client():
    graph = make_graph(
        {
            "role_llms": {
                "Critic": {"provider": "anthropic", "model": "claude-opus-5"},
                "Advocate": {"provider": "anthropic", "model": "claude-opus-5"},
            }
        }
    )
    assert graph.role_llms["Critic"] is graph.role_llms["Advocate"]


def test_unknown_role_raises():
    with pytest.raises(ValueError, match="Unknown role"):
        make_graph({"role_llms": {"Chief Vibes Officer": {"provider": "openai"}}})


def test_graph_setup_uses_override_for_node():
    quick, deep, custom = object(), object(), object()
    setup = GraphSetup(
        quick_thinking_llm=quick,
        deep_thinking_llm=deep,
        tools=[],
        config=DEFAULT_CONFIG,
        role_llms={"Critic": custom},
    )
    assert setup._llm("Critic") is custom
    assert setup._llm("Advocate") is quick
    assert setup._llm("Research Manager", tier="deep") is deep


def test_role_tiers_cover_all_graph_nodes():
    assert set(ROLE_TIERS) == {
        "Novelty Analyst",
        "Feasibility Analyst",
        "Impact Analyst",
        "Methodology Analyst",
        "Advocate",
        "Critic",
        "Research Manager",
        "Ambitious Scoper",
        "Conservative Scoper",
        "Pragmatic Scoper",
        "Program Director",
    }
