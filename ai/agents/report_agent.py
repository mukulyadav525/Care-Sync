import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ai.models.state import AgentState
from ai.tools.report_tool import create_report
from ai.services.llm_service import get_llm
from ai.services.session_memory import append_turn

logger = logging.getLogger(__name__)

RECOMMENDATIONS = {
    "stress": "Practice relaxation exercises, hydrate well, and sleep before 11 PM.",
    "sleep": "Aim for 7-8 hours of sleep.",
    "hrv": "Continue tracking HRV daily.",
}

SYSTEM_PROMPT = (
    "You are CareSync's health analysis assistant. Answer using only the "
    "measured PPG/HRV/stress data provided below — never invent numbers. "
    "Be concise, practical, and mention the detected intent's recommendation "
    "if relevant."
)


def _fallback_report(state: AgentState) -> str:
    """Deterministic template used if the LLM call fails (e.g. no/invalid
    XAI_API_KEY or the provider is unreachable) so /chat still responds."""
    report = [create_report(state["analysis"]), ""]
    recommendation = RECOMMENDATIONS.get(state["intent"])
    if recommendation:
        report.append(f"Recommendation:\n{recommendation}")
    return "\n".join(report)


def _build_prompt(state: AgentState, summary: str) -> str:
    history = state.get("history") or []
    if history:
        history_block = "\n\n".join(
            f"User: {turn['question']}\nAssistant: {turn['response']}"
            for turn in history
        )
    else:
        history_block = "No prior conversation."

    reasoning = state["analysis"].get("reasoning", "")
    weekly_summary = state["analysis"].get("weekly_summary")

    return (
        f"Conversation history:\n{history_block}\n\n"
        f"Measured data:\n{summary}\n\n"
        f"Detected intent: {state['intent']}\n"
        + (f"Medical reasoning: {reasoning}\n" if reasoning else "")
        + (f"Weekly trend summary: {weekly_summary}\n" if weekly_summary else "")
        + f"User question: {state['messages'][-1].content}"
    )


def report_node(state: AgentState):
    summary = create_report(state["analysis"])
    prompt = _build_prompt(state, summary)

    try:
        result = get_llm().invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        response_text = result.content
    except Exception:
        logger.exception("Grok call failed, falling back to templated report")
        response_text = _fallback_report(state)

    state["response"] = response_text

    session_id = state.get("session_id")
    if session_id:
        append_turn(session_id, state["messages"][-1].content, response_text)

    return state
