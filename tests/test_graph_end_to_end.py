"""End-to-end graph test with fake LLMs (no network, no API keys)."""

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from researchagents.default_config import DEFAULT_CONFIG
from researchagents.graph.research_graph import create_initial_state
from researchagents.graph.setup import GraphSetup


class FakeLLM:
    """Minimal stand-in supporting .invoke and .bind_tools."""

    def __init__(self):
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, prompt):
        self.calls += 1
        return AIMessage(content=f"fake response {self.calls}")


@tool
def fake_search(query: str) -> str:
    """Fake literature search."""
    return f"results for {query}"


def test_graph_runs_end_to_end():
    config = {**DEFAULT_CONFIG, "max_debate_rounds": 1, "max_scope_rounds": 1}
    graph = GraphSetup(
        quick_thinking_llm=FakeLLM(),
        deep_thinking_llm=FakeLLM(),
        tools=[fake_search],
        config=config,
    ).setup_graph()

    state = graph.invoke(
        create_initial_state(
            "Test a tiny idea.",
            "- Compute: 1x GPU",
            venue="- Venue: NeurIPS 2027\n- Track: main",
        ),
        config={"recursion_limit": config["max_recur_limit"]},
    )

    # Every stage produced output
    assert state["venue_report"]
    assert state["novelty_report"]
    assert state["feasibility_report"]
    assert state["impact_report"]
    assert state["methodology_report"]
    assert state["refined_proposal"]
    assert state["final_recommendation"]

    # Debates ran the configured number of turns
    assert state["proposal_debate_state"]["count"] == 2  # advocate + critic
    assert state["scope_debate_state"]["count"] == 3  # three scopers

    # Both sides actually spoke
    assert "Advocate:" in state["proposal_debate_state"]["history"]
    assert "Critic:" in state["proposal_debate_state"]["history"]
    assert "Ambitious Scoper:" in state["scope_debate_state"]["history"]
    assert "Conservative Scoper:" in state["scope_debate_state"]["history"]
    assert "Pragmatic Scoper:" in state["scope_debate_state"]["history"]


def test_graph_skips_venue_analyst_without_venue():
    config = {**DEFAULT_CONFIG, "max_debate_rounds": 1, "max_scope_rounds": 1}
    llm = FakeLLM()
    graph = GraphSetup(
        quick_thinking_llm=llm,
        deep_thinking_llm=FakeLLM(),
        tools=[fake_search],
        config=config,
    ).setup_graph()

    state = graph.invoke(
        create_initial_state("Test a tiny idea.", "- Compute: 1x GPU"),
        config={"recursion_limit": config["max_recur_limit"]},
    )

    assert state["venue_report"] == ""
    assert state["final_recommendation"]
