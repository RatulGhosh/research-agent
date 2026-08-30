from researchagents.agents.utils.context import format_venue_context


def create_pragmatic_scoper(llm):
    """Argues for a staged scope balancing ambition against the budget."""

    def pragmatic_node(state) -> dict:
        scope_state = state["scope_debate_state"]
        history = scope_state.get("history", "")
        pragmatic_history = scope_state.get("pragmatic_history", "")
        current_ambitious = scope_state.get("current_ambitious_response", "")
        current_conservative = scope_state.get("current_conservative_response", "")

        venue_context = format_venue_context(state)

        prompt = f"""You are the Pragmatic Scoper on a research review board. The refined research problem is settled; the question now is how big to make the project. You argue for a staged plan: a cheap pilot with explicit go/no-go criteria, then scale into the ambitious experiments only on a positive signal. Challenge both extremes — the Ambitious Scoper's budget optimism and the Conservative Scoper's tendency to descope the project below publishability.

Refined research problem:
{state.get("refined_proposal", "")}

Available resources:
{state["resources"]}
{venue_context}
Feasibility report: {state.get("feasibility_report", "")}
Methodology report: {state.get("methodology_report", "")}

Scoping debate so far: {history}
Ambitious Scoper's last argument: {current_ambitious}
Conservative Scoper's last argument: {current_conservative}

Argue for the staged scope, engaging directly with both scopers' latest points. Be specific: stage boundaries, the decision criterion at each gate, and how the resource budget splits across stages so that a failed pilot still leaves budget for a pivot."""

        response = llm.invoke(prompt)
        argument = f"Pragmatic Scoper: {response.content}"

        new_scope_state = {
            "history": history + "\n" + argument,
            "ambitious_history": scope_state.get("ambitious_history", ""),
            "conservative_history": scope_state.get("conservative_history", ""),
            "pragmatic_history": pragmatic_history + "\n" + argument,
            "latest_speaker": "Pragmatic",
            "current_ambitious_response": current_ambitious,
            "current_conservative_response": current_conservative,
            "current_pragmatic_response": argument,
            "judge_decision": scope_state.get("judge_decision", ""),
            "count": scope_state["count"] + 1,
        }

        return {"scope_debate_state": new_scope_state}

    return pragmatic_node
