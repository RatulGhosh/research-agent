from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


def create_venue_analyst(llm, tools, max_search_calls: int = 4):
    """Researches the target venue/track and its fit for the idea.

    Skipped (empty report) when no target venue was provided. Uses the web
    search tool to pull the CFP, deadlines, scope, and acceptance expectations
    for the specific track; the tool loop terminates once the search budget is
    spent.
    """

    def venue_analyst_node(state) -> dict:
        target_venue = (state.get("target_venue") or "").strip()
        if not target_venue:
            return {"venue_report": "", "sender": "Venue Analyst"}

        research_idea = state["research_idea"]

        system_prompt = (
            "You are a Venue Analyst on a research review board. The team is "
            "considering submitting the proposed research to a specific venue and "
            "track. Your job is to research that venue and assess the fit.\n\n"
            "Use the web_search tool (and search_literature if useful) to find the "
            "venue's call for papers, important dates, and track-specific "
            "requirements — include the venue name and URL in your queries. If a "
            "URL was provided, search for its contents. Issue a few targeted "
            "queries before concluding.\n\n"
            "When you have enough evidence, write a report in Markdown covering:\n"
            "- Venue overview: what the venue/workshop is about, its standing, and "
            "typical acceptance rate if findable\n"
            "- Track requirements: what the named track (main, findings, workshop, "
            "industry, demo, ...) expects — result strength, page limits, "
            "archival status, review process (e.g. findings accepts solid work "
            "below the main-track novelty bar; industry tracks want deployment "
            "evidence; workshops accept preliminary results)\n"
            "- Key dates: submission deadline, notification, camera-ready — and "
            "whether the declared timeline can hit them\n"
            "- Scope fit: does the idea match the CFP topics? Quote the relevant "
            "CFP topics\n"
            "- Bar assessment: what result strength this submission would need at "
            "this track, and the biggest fit risks\n"
            "- A venue-fit rating: POOR / FAIR / STRONG, with justification\n\n"
            "If searches fail or return nothing, say what you could not verify and "
            "fall back to what you know about the venue — clearly marked as "
            "unverified. End with a section titled '## Venue Verdict'."
        )

        messages = state.get("messages") or [
            HumanMessage(
                content=f"Target venue/track:\n{target_venue}\n\n"
                f"Proposed research idea:\n\n{research_idea}"
            )
        ]

        searches_used = sum(1 for m in messages if isinstance(m, ToolMessage))
        if searches_used >= max_search_calls or not tools:
            chain = llm
            if searches_used >= max_search_calls:
                messages = list(messages) + [
                    HumanMessage(
                        content="Your search budget is exhausted. Write your final "
                        "venue report now based on the evidence gathered so far."
                    )
                ]
        else:
            chain = llm.bind_tools(tools)

        result = chain.invoke([SystemMessage(content=system_prompt)] + list(messages))

        report = state.get("venue_report", "")
        if not getattr(result, "tool_calls", None):
            report = result.content if isinstance(result.content, str) else str(result.content)

        return {
            "messages": [result],
            "venue_report": report,
            "sender": "Venue Analyst",
        }

    return venue_analyst_node
