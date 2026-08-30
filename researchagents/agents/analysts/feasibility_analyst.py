def create_feasibility_analyst(llm):
    """Assesses whether the idea fits the declared compute/resource budget."""

    def feasibility_analyst_node(state) -> dict:
        research_idea = state["research_idea"]
        resources = state["resources"]
        novelty_report = state.get("novelty_report", "")

        prompt = f"""You are a Feasibility Analyst on a research review board. Your job is to assess whether the proposed research is executable with the resources actually available, and where the resource bottlenecks are.

Proposed research idea:
{research_idea}

Declared available resources:
{resources}

Novelty analyst's literature survey (use it to calibrate typical compute in this area):
{novelty_report}

Write a report in Markdown covering:
- Compute budget analysis: estimate GPU-hours / memory needs for the experiments the idea implies (training runs, ablations, baselines, hyperparameter sweeps) and compare against the declared budget. Show your arithmetic and state your assumptions (model sizes, dataset scale, number of runs).
- Data requirements: what datasets are needed, are they public/accessible, licensing or collection burden.
- Engineering burden: infrastructure, frameworks, and implementation effort relative to team size and skills.
- Timeline realism given the declared time budget.
- Descope options: the cheapest experiment that would still test the core hypothesis (minimum viable experiment), and what to cut first if resources run short.
- A feasibility rating: INFEASIBLE / TIGHT / COMFORTABLE, with justification.

Be concrete and quantitative wherever possible. End with a section titled '## Feasibility Verdict'."""

        response = llm.invoke(prompt)

        return {
            "feasibility_report": response.content,
            "sender": "Feasibility Analyst",
        }

    return feasibility_analyst_node
