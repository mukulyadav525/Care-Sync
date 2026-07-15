from ai.models.state import AgentState


def medical_reasoning_node(state: AgentState):

    intent = state["intent"]

    analysis = state["analysis"]

    reasoning = ""

    if intent == "stress":

        reasoning = (
            "Stress appears elevated because RMSSD is lower "
            "than expected and heart rate is relatively high."
        )

    elif intent == "sleep":

        reasoning = (
            "Sleep duration is below the recommended "
            "7-9 hour range."
        )

    elif intent == "hrv":

        reasoning = (
            "HRV values appear within the normal range "
            "for moderate recovery."
        )

    elif intent == "trend":

        reasoning = (
            "Recent physiological trends indicate gradual "
            "increase in stress."
        )

    else:

        reasoning = (
            "Unable to determine a specific health concern."
        )

    state["analysis"]["reasoning"] = reasoning

    return state