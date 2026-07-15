from ai.models.state import AgentState


def planner_node(state: AgentState):

    question = state["messages"][-1].content.lower()

    intent = "general"

    stress_keywords = [
        "stress",
        "anxiety",
        "tense",
        "overwhelmed"
    ]

    sleep_keywords = [
        "sleep",
        "nap",
        "rest"
    ]

    hrv_keywords = [
        "hrv",
        "rmssd",
        "sdnn",
        "heart rate variability"
    ]

    trend_keywords = [
        "trend",
        "history",
        "weekly",
        "monthly",
        "compare"
    ]

    if any(word in question for word in stress_keywords):
        intent = "stress"

    elif any(word in question for word in sleep_keywords):
        intent = "sleep"

    elif any(word in question for word in hrv_keywords):
        intent = "hrv"

    elif any(word in question for word in trend_keywords):
        intent = "trend"

    state["intent"] = intent

    return state