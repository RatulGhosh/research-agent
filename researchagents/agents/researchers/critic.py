def create_critic(llm):
    """Argues against pursuing the idea as posed, stress-testing every claim."""

    def critic_node(state) -> dict:
        debate_state = state["proposal_debate_state"]
        history = debate_state.get("history", "")
        critic_history = debate_state.get("critic_history", "")
        current_response = debate_state.get("current_response", "")

        prompt = f"""You are the Critic on a research review board, stress-testing this research idea. Your job is to find the reasons this project would fail, waste the resource budget, or produce an unpublishable result — before real time and GPUs are spent on it.

Key points to focus on:
- Novelty risk: is this already done or crowded? Use the literature survey; name the specific papers that threaten the contribution.
- Feasibility risk: where the compute/time/data budget breaks down; call out optimistic assumptions in the Advocate's case.
- Methodology risk: confounds, missing baselines, or evaluation problems that would sink it in review.
- Opportunity cost: what the same resources could achieve elsewhere.
- Engagement: respond conversationally and directly to the Advocate's latest argument, poke holes in its specific claims — don't just list generic risks.

Proposed research idea:
{state["research_idea"]}

Available resources:
{state["resources"]}

Novelty report: {state.get("novelty_report", "")}
Feasibility report: {state.get("feasibility_report", "")}
Impact report: {state.get("impact_report", "")}
Methodology report: {state.get("methodology_report", "")}

Debate history so far: {history}
Advocate's last argument: {current_response}

Deliver a rigorous critique. Be adversarial but fair: if a part of the idea genuinely holds up, acknowledge it and focus your attack where the real weaknesses are. If you believe the idea is salvageable only in a modified form, say exactly what modification would satisfy you."""

        response = llm.invoke(prompt)
        argument = f"Critic: {response.content}"

        new_debate_state = {
            "history": history + "\n" + argument,
            "advocate_history": debate_state.get("advocate_history", ""),
            "critic_history": critic_history + "\n" + argument,
            "current_response": argument,
            "judge_decision": debate_state.get("judge_decision", ""),
            "count": debate_state["count"] + 1,
        }

        return {"proposal_debate_state": new_debate_state}

    return critic_node
