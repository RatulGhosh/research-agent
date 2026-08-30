from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


def create_novelty_analyst(llm, tools, max_search_calls: int = 4):
    """Novelty analyst with access to arXiv search tools.

    Runs a tool-calling loop (via the graph's conditional edges) to survey
    related work before writing its report. Once the search budget is spent,
    the LLM is invoked without tools so the loop always terminates.
    """

    def novelty_analyst_node(state) -> dict:
        research_idea = state["research_idea"]

        system_prompt = (
            "You are a Novelty Analyst on a research review board. Your job is to "
            "assess how novel a proposed research idea is relative to existing "
            "literature.\n\n"
            "Use the search_arxiv tool to survey related work. Issue 2-4 targeted "
            "queries with different phrasings and subtopics before concluding — do "
            "not rely on a single query. When you have enough evidence, write a "
            "report in Markdown covering:\n"
            "- Closest prior work (cite the specific papers you found, with links)\n"
            "- What is genuinely new in this idea vs. incremental\n"
            "- Whether the idea appears already done, crowded, or open\n"
            "- Concrete suggestions to sharpen the novel angle\n"
            "- A novelty rating: LOW / MEDIUM / HIGH, with justification\n\n"
            "Be honest: if the literature suggests the idea is already well covered, "
            "say so plainly. End the final report with a section titled "
            "'## Novelty Verdict'."
        )

        messages = state.get("messages") or [
            HumanMessage(content=f"Proposed research idea:\n\n{research_idea}")
        ]

        searches_used = sum(1 for m in messages if isinstance(m, ToolMessage))
        if searches_used >= max_search_calls:
            chain = llm
            messages = list(messages) + [
                HumanMessage(
                    content="Your literature search budget is exhausted. Write your "
                    "final novelty report now based on the evidence gathered so far."
                )
            ]
        else:
            chain = llm.bind_tools(tools)

        result = chain.invoke([SystemMessage(content=system_prompt)] + list(messages))

        report = state.get("novelty_report", "")
        if not getattr(result, "tool_calls", None):
            report = result.content if isinstance(result.content, str) else str(result.content)

        return {
            "messages": [result],
            "novelty_report": report,
            "sender": "Novelty Analyst",
        }

    return novelty_analyst_node
