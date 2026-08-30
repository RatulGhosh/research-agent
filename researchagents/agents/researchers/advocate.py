from researchagents.agents.utils.context import format_venue_context


def create_advocate(llm):
    """Argues for pursuing the research idea, engaging the critic's points."""

    def advocate_node(state) -> dict:
        debate_state = state["proposal_debate_state"]
        history = debate_state.get("history", "")
        advocate_history = debate_state.get("advocate_history", "")
        current_response = debate_state.get("current_response", "")

        venue_context = format_venue_context(state)

        prompt = f"""You are the Advocate on a research review board, arguing that this research idea is worth pursuing. Build a strong, evidence-based case grounded in the analyst reports, and directly rebut the Critic's latest points.

Key points to focus on:
- Upside: the strongest version of the contribution and why the field needs it.
- Evidence: cite specifics from the novelty, feasibility, impact, and methodology reports that support pursuing it.
- De-risking: for each weakness the Critic raises, propose a concrete mitigation (a pilot experiment, a descoped variant, an alternative dataset) rather than dismissing it.
- Engagement: respond conversationally and directly to the Critic's latest argument; don't just restate your case.

Proposed research idea:
{state["research_idea"]}

Available resources:
{state["resources"]}
{venue_context}
Novelty report: {state.get("novelty_report", "")}
Feasibility report: {state.get("feasibility_report", "")}
Impact report: {state.get("impact_report", "")}
Methodology report: {state.get("methodology_report", "")}

Debate history so far: {history}
Critic's last argument: {current_response}

Deliver a compelling argument for pursuing this research, refute the Critic's concerns with specifics, and where the Critic has a genuinely strong point, concede it and show how a modified version of the idea survives it."""

        response = llm.invoke(prompt)
        argument = f"Advocate: {response.content}"

        new_debate_state = {
            "history": history + "\n" + argument,
            "advocate_history": advocate_history + "\n" + argument,
            "critic_history": debate_state.get("critic_history", ""),
            "current_response": argument,
            "judge_decision": debate_state.get("judge_decision", ""),
            "count": debate_state["count"] + 1,
        }

        return {"proposal_debate_state": new_debate_state}

    return advocate_node
