from researchagents.agents.utils.context import format_venue_context


def create_program_director(llm):
    """Issues the final recommendation after the scoping debate."""

    def program_director_node(state) -> dict:
        scope_state = state["scope_debate_state"]
        history = scope_state.get("history", "")

        venue_context = format_venue_context(state)

        prompt = f"""You are the Program Director with final authority over this research review. The board has produced analyst reports, an advocate/critic debate verdict with a refined research problem, and a scoping debate. Issue the final decision.

Original research idea:
{state["research_idea"]}

Available resources:
{state["resources"]}
{venue_context}
Analyst reports:
Novelty: {state.get("novelty_report", "")}
Feasibility: {state.get("feasibility_report", "")}
Impact: {state.get("impact_report", "")}
Methodology: {state.get("methodology_report", "")}

Research Manager's verdict and refined problem:
{state.get("refined_proposal", "")}

Scoping debate:
{history}

Produce, in Markdown:

## Recommendation
Exactly one of: **PURSUE**, **PURSUE WITH MODIFICATIONS**, or **DO NOT PURSUE** — followed by a confidence (LOW/MEDIUM/HIGH) and a one-paragraph rationale. Commit to a decision; do not hedge between categories. Choose DO NOT PURSUE when the evidence warrants it — a clear rejection with reasons is more valuable than a lukewarm endorsement.

## Final Research Problem
The definitive problem statement to execute (restate the refined problem, adjusted for your chosen scope). If the recommendation is DO NOT PURSUE, instead state what adjacent problem, if any, would be worth pursuing with these resources.

## Execution Plan
- Stages with go/no-go criteria (adopt whichever scoping position won, or your own synthesis), aligned to the venue deadline if one is targeted
- Resource allocation per stage against the declared budget (GPU-hours, time)
- First two weeks: the concrete first experiments to run
- Top three risks and the pre-planned response to each

## Why Not the Alternatives
One short paragraph each on why you rejected the other two recommendation categories."""

        response = llm.invoke(prompt)

        new_scope_state = dict(scope_state)
        new_scope_state["judge_decision"] = response.content

        return {
            "scope_debate_state": new_scope_state,
            "final_recommendation": response.content,
            "sender": "Program Director",
        }

    return program_director_node
