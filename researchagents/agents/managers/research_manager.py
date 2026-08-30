def create_research_manager(llm):
    """Judges the advocate/critic debate and drafts the refined research problem."""

    def research_manager_node(state) -> dict:
        debate_state = state["proposal_debate_state"]
        history = debate_state.get("history", "")

        prompt = f"""You are the Research Manager chairing a research review board. The Advocate and Critic have finished debating the proposed idea in light of the analyst reports. Your job is to weigh the debate on its merits and produce a refined version of the research problem that keeps the defensible core and fixes what the Critic legitimately broke. Do not default to a middle position — side with whoever argued better on each specific point.

Proposed research idea:
{state["research_idea"]}

Available resources:
{state["resources"]}

Analyst reports:
Novelty: {state.get("novelty_report", "")}
Feasibility: {state.get("feasibility_report", "")}
Impact: {state.get("impact_report", "")}
Methodology: {state.get("methodology_report", "")}

Debate history:
{history}

Produce, in Markdown:
## Debate Verdict
Which side won on each contested point and why, in plain language.

## Refined Research Problem
A rewritten problem statement (2-3 paragraphs) that a strong PhD student could execute: precise hypothesis, scope, and the novel contribution stated explicitly. This should incorporate the modifications the debate showed to be necessary — narrowed scope, different angle, added baselines, whatever survived scrutiny. If the debate showed the idea has no defensible core, say so explicitly instead of inventing one.

## Key Changes From Original
Bulleted list of what you changed and which debate point motivated each change."""

        response = llm.invoke(prompt)

        new_debate_state = dict(debate_state)
        new_debate_state["judge_decision"] = response.content

        return {
            "proposal_debate_state": new_debate_state,
            "refined_proposal": response.content,
            "sender": "Research Manager",
        }

    return research_manager_node
