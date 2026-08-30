from researchagents.agents.utils.context import format_venue_context


def create_ambitious_scoper(llm):
    """Argues for the biggest defensible version of the refined proposal."""

    def ambitious_node(state) -> dict:
        scope_state = state["scope_debate_state"]
        history = scope_state.get("history", "")
        ambitious_history = scope_state.get("ambitious_history", "")
        current_conservative = scope_state.get("current_conservative_response", "")
        current_pragmatic = scope_state.get("current_pragmatic_response", "")

        venue_context = format_venue_context(state)

        prompt = f"""You are the Ambitious Scoper on a research review board. The refined research problem is settled; the question now is how big to make the project. You argue for the most ambitious defensible scope: aim for the top venue, the strongest claims, the full experimental matrix. Small, safe projects produce forgettable papers — if the resources can plausibly support the big version, fight for it.

Refined research problem:
{state.get("refined_proposal", "")}

Available resources:
{state["resources"]}
{venue_context}
Feasibility report: {state.get("feasibility_report", "")}
Methodology report: {state.get("methodology_report", "")}

Scoping debate so far: {history}
Conservative Scoper's last argument: {current_conservative}
Pragmatic Scoper's last argument: {current_pragmatic}

Argue for the ambitious scope, engaging directly with the other scopers' latest points. Be specific: which extra experiments, model scales, or claims the bigger version adds, and why the resource budget stretches to cover them (e.g., via efficient training tricks, shared baselines, staged spending)."""

        response = llm.invoke(prompt)
        argument = f"Ambitious Scoper: {response.content}"

        new_scope_state = {
            "history": history + "\n" + argument,
            "ambitious_history": ambitious_history + "\n" + argument,
            "conservative_history": scope_state.get("conservative_history", ""),
            "pragmatic_history": scope_state.get("pragmatic_history", ""),
            "latest_speaker": "Ambitious",
            "current_ambitious_response": argument,
            "current_conservative_response": current_conservative,
            "current_pragmatic_response": current_pragmatic,
            "judge_decision": scope_state.get("judge_decision", ""),
            "count": scope_state["count"] + 1,
        }

        return {"scope_debate_state": new_scope_state}

    return ambitious_node
