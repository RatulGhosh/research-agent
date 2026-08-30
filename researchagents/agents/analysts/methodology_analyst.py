def create_methodology_analyst(llm):
    """Sketches the experimental design and probes its weak points."""

    def methodology_analyst_node(state) -> dict:
        research_idea = state["research_idea"]
        resources = state["resources"]
        novelty_report = state.get("novelty_report", "")
        feasibility_report = state.get("feasibility_report", "")

        prompt = f"""You are a Methodology Analyst on a research review board. Your job is to sketch how the proposed research would actually be tested, and to find the methodological weak points before a reviewer does.

Proposed research idea:
{research_idea}

Declared available resources:
{resources}

Novelty analyst's literature survey:
{novelty_report}

Feasibility analyst's report:
{feasibility_report}

Write a report in Markdown covering:
- Core hypothesis: state the idea as one or more falsifiable hypotheses.
- Experimental design: datasets, baselines, metrics, and the key comparisons needed to support each hypothesis.
- Confounds and threats to validity: what could make results look good without the hypothesis being true, and how to control for it.
- Ablations: which components must be ablated for the claims to be credible.
- Statistical rigor: seeds, variance reporting, significance — what this area's reviewers expect.
- Anticipated reviewer objections: the three hardest questions a skeptical reviewer would ask, and whether the design can answer them.
- A rigor rating for the idea as posed: WEAK / ADEQUATE / STRONG, with justification.

End with a section titled '## Methodology Verdict'."""

        response = llm.invoke(prompt)

        return {
            "methodology_report": response.content,
            "sender": "Methodology Analyst",
        }

    return methodology_analyst_node
