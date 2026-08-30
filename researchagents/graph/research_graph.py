import json
import os
import re
from datetime import datetime
from typing import Dict, Optional, Tuple

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from researchagents.dataflows import search_arxiv, search_web
from researchagents.default_config import DEFAULT_CONFIG
from researchagents.llm_clients import create_llm

from .setup import GraphSetup

# Which default tier each graph node falls back to when it has no entry in
# config["role_llms"]. The two judges get the deep-thinking model.
ROLE_TIERS = {
    "Venue Analyst": "quick",
    "Novelty Analyst": "quick",
    "Feasibility Analyst": "quick",
    "Impact Analyst": "quick",
    "Methodology Analyst": "quick",
    "Advocate": "quick",
    "Critic": "quick",
    "Research Manager": "deep",
    "Ambitious Scoper": "quick",
    "Conservative Scoper": "quick",
    "Pragmatic Scoper": "quick",
    "Program Director": "deep",
}


def create_initial_state(research_idea: str, resources: str, venue: str = "") -> Dict:
    """Build the initial graph state for a review run."""
    return {
        "messages": [
            HumanMessage(content=f"Proposed research idea:\n\n{research_idea}")
        ],
        "research_idea": research_idea,
        "resources": resources,
        "target_venue": venue or "",
        "venue_report": "",
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

        self._llm_cache: Dict[tuple, object] = {}
        self.deep_thinking_llm = self._get_llm(
            self.config["llm_provider"],
            self.config["deep_think_llm"],
            self.config.get("backend_url"),
        )
        self.quick_thinking_llm = self._get_llm(
            self.config["llm_provider"],
            self.config["quick_think_llm"],
            self.config.get("backend_url"),
        )
        self.role_llms = self._build_role_llms()

        self.tools = self._build_tools()

        self.graph = GraphSetup(
            quick_thinking_llm=self.quick_thinking_llm,
            deep_thinking_llm=self.deep_thinking_llm,
            tools=self.tools,
            config=self.config,
            role_llms=self.role_llms,
        ).setup_graph()

    def _get_llm(self, provider: str, model: str, base_url: Optional[str] = None):
        key = (provider, model, base_url)
        if key not in self._llm_cache:
            self._llm_cache[key] = create_llm(provider, model, base_url=base_url)
        return self._llm_cache[key]

    def _build_role_llms(self) -> Dict[str, object]:
        """Resolve config["role_llms"] entries into LLM instances."""
        role_llms = {}
        for role, spec in (self.config.get("role_llms") or {}).items():
            if role not in ROLE_TIERS:
                raise ValueError(
                    f"Unknown role '{role}' in role_llms. "
                    f"Valid roles: {', '.join(ROLE_TIERS)}"
                )
            provider = spec.get("provider", self.config["llm_provider"])
            tier_default = (
                self.config["deep_think_llm"]
                if ROLE_TIERS[role] == "deep"
                else self.config["quick_think_llm"]
            )
            model = spec.get("model", tier_default)
            base_url = spec.get("base_url")
            if base_url is None and provider == self.config["llm_provider"]:
                base_url = self.config.get("backend_url")
            role_llms[role] = self._get_llm(provider, model, base_url)
        return role_llms

    def _build_tools(self):
        arxiv_max_results = self.config["arxiv_max_results"]

        @tool
        def search_literature(query: str) -> str:
            """Search arXiv for papers related to the query.

            Args:
                query: Free-text keywords describing the topic to search for.
            """
            return search_arxiv(query, max_results=arxiv_max_results)

        tools = [search_literature]

        ws_config = self.config.get("web_search") or {}
        if ws_config.get("enabled"):
            ws_provider = ws_config.get("provider", "openai")
            ws_model = ws_config.get("model")
            ws_max_uses = ws_config.get("max_uses", 3)

            @tool
            def web_search(query: str) -> str:
                """Search the web for recent papers, lab blog posts, and code.

                Complements arXiv search: use it for very recent work, industry
                results, published venue versions, and open-source projects.

                Args:
                    query: Free-text keywords describing what to look for.
                """
                return search_web(
                    query,
                    provider=ws_provider,
                    model=ws_model,
                    max_uses=ws_max_uses,
                )

            tools.append(web_search)

        return tools

    def propagate(
        self, research_idea: str, resources: str, venue: str = ""
    ) -> Tuple[Dict, str]:
        """Evaluate a research idea against the available resources.

        Args:
            research_idea: High-level description of the proposed research.
            resources: Description of available resources (GPUs, time, data,
                team, budget).
            venue: Optional target venue description — conference/workshop
                name, URL, and track (main, findings, workshop, industry, ...).

        Returns:
            Tuple of (final graph state, final recommendation text).
        """
        initial_state = create_initial_state(research_idea, resources, venue)

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
                f"## Target Venue\n\n{state.get('target_venue') or '(none specified)'}",
                f"## Venue Report\n\n{state.get('venue_report') or '(no venue research)'}",
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
                    "target_venue": state.get("target_venue", ""),
                    "refined_proposal": state["refined_proposal"],
                    "final_recommendation": state["final_recommendation"],
                },
                f,
                indent=2,
            )

        if self.debug:
            print(f"Report saved to {report_path}")
        return report_path
