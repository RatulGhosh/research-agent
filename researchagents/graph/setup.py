from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from researchagents.agents.analysts.feasibility_analyst import create_feasibility_analyst
from researchagents.agents.analysts.impact_analyst import create_impact_analyst
from researchagents.agents.analysts.methodology_analyst import create_methodology_analyst
from researchagents.agents.analysts.novelty_analyst import create_novelty_analyst
from researchagents.agents.managers.program_director import create_program_director
from researchagents.agents.managers.research_manager import create_research_manager
from researchagents.agents.researchers.advocate import create_advocate
from researchagents.agents.researchers.critic import create_critic
from researchagents.agents.scoping.ambitious_scoper import create_ambitious_scoper
from researchagents.agents.scoping.conservative_scoper import create_conservative_scoper
from researchagents.agents.scoping.pragmatic_scoper import create_pragmatic_scoper
from researchagents.agents.utils.agent_states import AgentState
from researchagents.graph.conditional_logic import ConditionalLogic


class GraphSetup:
    """Builds the research review board graph."""

    def __init__(self, quick_thinking_llm, deep_thinking_llm, tools, config, role_llms=None):
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tools = tools
        self.config = config
        # Per-role LLM overrides: {node name: llm}. Roles without an entry use
        # the tier default passed to _llm().
        self.role_llms = role_llms or {}
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=config["max_debate_rounds"],
            max_scope_rounds=config["max_scope_rounds"],
        )

    def _llm(self, role, tier="quick"):
        if role in self.role_llms:
            return self.role_llms[role]
        return self.deep_thinking_llm if tier == "deep" else self.quick_thinking_llm

    def setup_graph(self):
        workflow = StateGraph(AgentState)

        # Analysts
        workflow.add_node(
            "Novelty Analyst",
            create_novelty_analyst(
                self._llm("Novelty Analyst"),
                self.tools,
                max_search_calls=self.config["max_lit_search_calls"],
            ),
        )
        workflow.add_node("tools_novelty", ToolNode(self.tools))
        workflow.add_node(
            "Feasibility Analyst", create_feasibility_analyst(self._llm("Feasibility Analyst"))
        )
        workflow.add_node("Impact Analyst", create_impact_analyst(self._llm("Impact Analyst")))
        workflow.add_node(
            "Methodology Analyst", create_methodology_analyst(self._llm("Methodology Analyst"))
        )

        # Advocate/critic debate + judge
        workflow.add_node("Advocate", create_advocate(self._llm("Advocate")))
        workflow.add_node("Critic", create_critic(self._llm("Critic")))
        workflow.add_node(
            "Research Manager", create_research_manager(self._llm("Research Manager", tier="deep"))
        )

        # Scoping debate + final judge
        workflow.add_node("Ambitious Scoper", create_ambitious_scoper(self._llm("Ambitious Scoper")))
        workflow.add_node(
            "Conservative Scoper", create_conservative_scoper(self._llm("Conservative Scoper"))
        )
        workflow.add_node("Pragmatic Scoper", create_pragmatic_scoper(self._llm("Pragmatic Scoper")))
        workflow.add_node(
            "Program Director", create_program_director(self._llm("Program Director", tier="deep"))
        )

        # Edges
        workflow.add_edge(START, "Novelty Analyst")
        workflow.add_conditional_edges(
            "Novelty Analyst",
            self.conditional_logic.should_continue_novelty,
            ["tools_novelty", "Feasibility Analyst"],
        )
        workflow.add_edge("tools_novelty", "Novelty Analyst")
        workflow.add_edge("Feasibility Analyst", "Impact Analyst")
        workflow.add_edge("Impact Analyst", "Methodology Analyst")
        workflow.add_edge("Methodology Analyst", "Advocate")

        workflow.add_conditional_edges(
            "Advocate",
            self.conditional_logic.should_continue_debate,
            ["Critic", "Research Manager"],
        )
        workflow.add_conditional_edges(
            "Critic",
            self.conditional_logic.should_continue_debate,
            ["Advocate", "Research Manager"],
        )
        workflow.add_edge("Research Manager", "Ambitious Scoper")

        workflow.add_conditional_edges(
            "Ambitious Scoper",
            self.conditional_logic.should_continue_scoping,
            ["Conservative Scoper", "Program Director"],
        )
        workflow.add_conditional_edges(
            "Conservative Scoper",
            self.conditional_logic.should_continue_scoping,
            ["Pragmatic Scoper", "Program Director"],
        )
        workflow.add_conditional_edges(
            "Pragmatic Scoper",
            self.conditional_logic.should_continue_scoping,
            ["Ambitious Scoper", "Program Director"],
        )
        workflow.add_edge("Program Director", END)

        return workflow.compile()
