from researchagents.agents.utils.context import format_venue_context


def create_impact_analyst(llm):
    """Assesses scientific and practical impact if the research succeeds."""

    def impact_analyst_node(state) -> dict:
        research_idea = state["research_idea"]
        novelty_report = state.get("novelty_report", "")
        venue_context = format_venue_context(state)

        prompt = f"""You are an Impact Analyst on a research review board. Your job is to assess how much the proposed research would matter if it succeeds.

Proposed research idea:
{research_idea}

Novelty analyst's literature survey:
{novelty_report}
{venue_context}
Write a report in Markdown covering:
- Scientific impact: what open question does this answer, who in the community cares, and what follow-up work would it enable.
- Practical impact: real-world applications, users, or systems that would benefit.
- Publication potential: if a target venue/track is specified, assess fit against that venue and what result strength that track requires; otherwise suggest realistic target venues.
- Failure value: if the hypothesis turns out false, is the negative result still informative/publishable, or is the effort wasted?
- Timing: is this the right moment for the idea (enabling tools/datasets just appeared, community attention rising) or is it too early/late?
- An impact rating: LOW / MEDIUM / HIGH, with justification.

End with a section titled '## Impact Verdict'."""

        response = llm.invoke(prompt)

        return {
            "impact_report": response.content,
            "sender": "Impact Analyst",
        }

    return impact_analyst_node
