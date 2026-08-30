import json
import os
import re
from datetime import datetime
from typing import Dict, Optional, Tuple

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from researchagents.dataflows import search_arxiv
from researchagents.default_config import DEFAULT_CONFIG
from researchagents.llm_clients import create_llm

from .setup import GraphSetup


def create_initial_state(research_idea: str, resources: str) -> Dict:
    """Build the initial graph state for a review run."""
    return {
        "messages": [
            HumanMessage(content=f"Proposed research idea:\n\n{research_idea}")
        ],
        "research_idea": research_idea,
        "resources": resources,
        "novelty_report": "",
        "feasibility_report": "",
        "impact_report": "",
        "methodology_report": "",
        "proposal_debate_state": {
            "advocate_history": "",
            "critic_history": "",
            "history": "",
            "current_response": "",
            "judge_decision": "",
            "count": 0,
        },
        "refined_proposal": "",
        "scope_debate_state": {
            "ambitious_history": "",
            "conservative_history": "",
            "pragmatic_history": "",
            "history": "",
            "latest_speaker": "",
            "current_ambitious_response": "",
            "current_conservative_response": "",
            "current_pragmatic_response": "",
            "judge_decision": "",
            "count": 0,
        },
        "final_recommendation": "",
    }


class ResearchAgentsGraph:
    """Orchestrates the research review board."""

    def __init__(self, debug: bool = False, config: Optional[Dict] = None):
        self.debug = debug
        self.config = {**DEFAULT_CONFIG, **(config or {})}

        os.makedirs(self.config["results_dir"], exist_ok=True)

        self.deep_thinking_llm = create_llm(
            self.config["llm_provider"],
            self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
        )
        self.quick_thinking_llm = create_llm(
            self.config["llm_provider"],
            self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
        )

        arxiv_max_results = self.config["arxiv_max_results"]

        @tool
        def search_literature(query: str) -> str:
            """Search arXiv for papers related to the query.

            Args:
                query: Free-text keywords describing the topic to search for.
            """
            return search_arxiv(query, max_results=arxiv_max_results)

        self.tools = [search_literature]

        self.graph = GraphSetup(
            quick_thinking_llm=self.quick_thinking_llm,
            deep_thinking_llm=self.deep_thinking_llm,
            tools=self.tools,
            config=self.config,
        ).setup_graph()

    def propagate(self, research_idea: str, resources: str) -> Tuple[Dict, str]:
        """Evaluate a research idea against the available resources.

        Args:
            research_idea: High-level description of the proposed research.
            resources: Description of available resources (GPUs, time, data,
                team, budget).

        Returns:
            Tuple of (final graph state, final recommendation text).
        """
        initial_state = create_initial_state(research_idea, resources)

        args = {"recursion_limit": self.config["max_recur_limit"]}

        if self.debug:
            final_state = None
            for chunk in self.graph.stream(initial_state, config=args, stream_mode="values"):
                sender = chunk.get("sender", "")
                if sender:
                    print(f"--- {sender} finished ---")
                final_state = chunk
        else:
            final_state = self.graph.invoke(initial_state, config=args)

        self._save_report(final_state)
        return final_state, final_state["final_recommendation"]

    def _save_report(self, state: Dict) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "-", state["research_idea"][:60].lower()).strip("-")
        report_dir = os.path.join(self.config["results_dir"], f"{timestamp}_{slug}")
        os.makedirs(report_dir, exist_ok=True)

        report = "\n\n---\n\n".join(
            [
                f"# Research Review Board Report\n\nGenerated: {datetime.now().isoformat()}",
                f"## Research Idea\n\n{state['research_idea']}",
                f"## Declared Resources\n\n{state['resources']}",
                f"## Novelty Report\n\n{state['novelty_report']}",
                f"## Feasibility Report\n\n{state['feasibility_report']}",
                f"## Impact Report\n\n{state['impact_report']}",
                f"## Methodology Report\n\n{state['methodology_report']}",
                f"## Advocate vs. Critic Debate\n\n{state['proposal_debate_state']['history']}",
                f"## Research Manager: Refined Proposal\n\n{state['refined_proposal']}",
                f"## Scoping Debate\n\n{state['scope_debate_state']['history']}",
                f"## Program Director: Final Recommendation\n\n{state['final_recommendation']}",
            ]
        )

        report_path = os.path.join(report_dir, "report.md")
        with open(report_path, "w") as f:
            f.write(report)

        # Machine-readable copy of the key outputs
        with open(os.path.join(report_dir, "summary.json"), "w") as f:
            json.dump(
                {
                    "research_idea": state["research_idea"],
                    "resources": state["resources"],
                    "refined_proposal": state["refined_proposal"],
                    "final_recommendation": state["final_recommendation"],
                },
                f,
                indent=2,
            )

        if self.debug:
            print(f"Report saved to {report_path}")
        return report_path
