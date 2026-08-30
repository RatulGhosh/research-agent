from typing import Annotated

from typing_extensions import TypedDict
from langgraph.graph import MessagesState


# Advocate/critic debate over the merit of the idea
class ProposalDebateState(TypedDict):
    advocate_history: Annotated[str, "Advocate conversation history"]
    critic_history: Annotated[str, "Critic conversation history"]
    history: Annotated[str, "Full conversation history"]
    current_response: Annotated[str, "Latest response"]
    judge_decision: Annotated[str, "Research Manager's synthesis"]
    count: Annotated[int, "Number of debate turns so far"]


# Scoping team debate over how big the project should be
class ScopeDebateState(TypedDict):
    ambitious_history: Annotated[str, "Ambitious scoper's conversation history"]
    conservative_history: Annotated[str, "Conservative scoper's conversation history"]
    pragmatic_history: Annotated[str, "Pragmatic scoper's conversation history"]
    history: Annotated[str, "Full conversation history"]
    latest_speaker: Annotated[str, "Scoper that spoke last"]
    current_ambitious_response: Annotated[str, "Latest ambitious response"]
    current_conservative_response: Annotated[str, "Latest conservative response"]
    current_pragmatic_response: Annotated[str, "Latest pragmatic response"]
    judge_decision: Annotated[str, "Program Director's decision"]
    count: Annotated[int, "Number of debate turns so far"]


class AgentState(MessagesState):
    research_idea: Annotated[str, "High-level research idea under evaluation"]
    resources: Annotated[str, "Available resources: GPUs, time, data, people, budget"]
    target_venue: Annotated[
        str, "Target venue: conference/workshop name, URL, and track (main, findings, ...)"
    ]

    sender: Annotated[str, "Agent that sent the last message"]

    # analyst reports
    venue_report: Annotated[str, "Venue/CFP research from the Venue Analyst"]
    novelty_report: Annotated[str, "Literature/novelty assessment from the Novelty Analyst"]
    feasibility_report: Annotated[str, "Compute/resource feasibility assessment"]
    impact_report: Annotated[str, "Scientific and practical impact assessment"]
    methodology_report: Annotated[str, "Experimental design and evaluation plan assessment"]

    # advocate/critic debate
    proposal_debate_state: Annotated[
        ProposalDebateState, "State of the debate on whether the idea has merit"
    ]
    refined_proposal: Annotated[str, "Refined research problem from the Research Manager"]

    # scoping debate
    scope_debate_state: Annotated[
        ScopeDebateState, "State of the debate on how the project should be scoped"
    ]
    final_recommendation: Annotated[str, "Final decision from the Program Director"]
