from researchagents.agents.utils.agent_states import AgentState


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds=2, max_scope_rounds=1):
        self.max_debate_rounds = max_debate_rounds
        self.max_scope_rounds = max_scope_rounds

    def should_continue_novelty(self, state: AgentState) -> str:
        """Route the novelty analyst's tool-calling loop."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools_novelty"
        return "Feasibility Analyst"

    def should_continue_debate(self, state: AgentState) -> str:
        """Route the advocate/critic debate."""
        # 2 speakers per round
        if state["proposal_debate_state"]["count"] >= 2 * self.max_debate_rounds:
            return "Research Manager"
        if state["proposal_debate_state"]["current_response"].startswith("Advocate"):
            return "Critic"
        return "Advocate"

    def should_continue_scoping(self, state: AgentState) -> str:
        """Route the ambitious/conservative/pragmatic scoping debate."""
        # 3 speakers per round
        if state["scope_debate_state"]["count"] >= 3 * self.max_scope_rounds:
            return "Program Director"
        latest = state["scope_debate_state"]["latest_speaker"]
        if latest.startswith("Ambitious"):
            return "Conservative Scoper"
        if latest.startswith("Conservative"):
            return "Pragmatic Scoper"
        return "Ambitious Scoper"
