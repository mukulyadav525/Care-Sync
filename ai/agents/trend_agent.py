from ai.models.state import AgentState


def trend_node(state: AgentState):

    if state["intent"] != "trend":
        return state

    state["analysis"]["weekly_summary"] = {
        "stress": "Higher than last week",
        "sleep": "Slightly reduced",
        "recovery": "Moderate"
    }

    return state