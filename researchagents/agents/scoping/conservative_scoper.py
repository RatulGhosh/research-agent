from researchagents.agents.utils.context import format_venue_context


def create_conservative_scoper(llm):
    """Argues for the minimal scope that protects the resource budget."""

    def conservative_node(state) -> dict:
        scope_state = state["scope_debate_state"]
        history = scope_state.get("history", "")
        conservative_history = scope_state.get("conservative_history", "")
        current_ambitious = scope_state.get("current_ambitious_response", "")
        current_pragmatic = scope_state.get("current_pragmatic_response", "")

        venue_context = format_venue_context(state)

        prompt = f"""You are the Conservative Scoper on a research review board. The refined research problem is settled; the question now is how big to make the project. You argue for the smallest scope that still tests the core hypothesis: minimum viable experiments first, spend nothing on the full matrix until the pilot proves the effect exists. Most research projects fail by burning their budget before learning whether the core idea works.

Refined research problem:
{state.get("refined_proposal", "")}

Available resources:
{state["resources"]}
{venue_context}
Feasibility report: {state.get("feasibility_report", "")}
Methodology report: {state.get("methodology_report", "")}

Scoping debate so far: {history}
Ambitious Scoper's last argument: {current_ambitious}
Pragmatic Scoper's last argument: {current_pragmatic}

Argue for the conservative scope, engaging directly with the other scopers' latest points. Be specific: what the cheapest decisive pilot looks like, what fraction of the GPU budget it needs, what go/no-go signal it produces, and which of the Ambitious Scoper's proposed experiments are premature before that signal exists."""

        response = llm.invoke(prompt)
        argument = f"Conservative Scoper: {response.content}"

        new_scope_state = {
            "history": history + "\n" + argument,
            "ambitious_history": scope_state.get("ambitious_history", ""),
            "conservative_history": conservative_history + "\n" + argument,
            "pragmatic_history": scope_state.get("pragmatic_history", ""),
            "latest_speaker": "Conservative",
            "current_ambitious_response": current_ambitious,
            "current_conservative_response": argument,
            "current_pragmatic_response": current_pragmatic,
            "judge_decision": scope_state.get("judge_decision", ""),
            "count": scope_state["count"] + 1,
        }

        return {"scope_debate_state": new_scope_state}

    return conservative_node
