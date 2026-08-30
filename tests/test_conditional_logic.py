from researchagents.graph.conditional_logic import ConditionalLogic


def make_debate_state(count, current_response):
    return {
        "proposal_debate_state": {
            "advocate_history": "",
            "critic_history": "",
            "history": "",
            "current_response": current_response,
            "judge_decision": "",
            "count": count,
        }
    }


def make_scope_state(count, latest_speaker):
    return {
        "scope_debate_state": {
            "ambitious_history": "",
            "conservative_history": "",
            "pragmatic_history": "",
            "history": "",
            "latest_speaker": latest_speaker,
            "current_ambitious_response": "",
            "current_conservative_response": "",
            "current_pragmatic_response": "",
            "judge_decision": "",
            "count": count,
        }
    }


def test_debate_alternates_between_advocate_and_critic():
    logic = ConditionalLogic(max_debate_rounds=2)
    assert logic.should_continue_debate(make_debate_state(1, "Advocate: ...")) == "Critic"
    assert logic.should_continue_debate(make_debate_state(2, "Critic: ...")) == "Advocate"


def test_debate_ends_at_research_manager():
    logic = ConditionalLogic(max_debate_rounds=2)
    assert (
        logic.should_continue_debate(make_debate_state(4, "Critic: ..."))
        == "Research Manager"
    )


def test_scoping_rotates_through_three_speakers():
    logic = ConditionalLogic(max_scope_rounds=1)
    assert (
        logic.should_continue_scoping(make_scope_state(1, "Ambitious"))
        == "Conservative Scoper"
    )
    assert (
        logic.should_continue_scoping(make_scope_state(2, "Conservative"))
        == "Pragmatic Scoper"
    )


def test_scoping_ends_at_program_director():
    logic = ConditionalLogic(max_scope_rounds=1)
    assert (
        logic.should_continue_scoping(make_scope_state(3, "Pragmatic"))
        == "Program Director"
    )
