from langgraph.graph import StateGraph, END

from ai.models.state import AgentState

from ai.agents.planner import planner_node
from ai.agents.signal_agent import signal_node
from ai.agents.medical_agent import medical_reasoning_node
from ai.agents.trend_agent import trend_node
from ai.agents.report_agent import report_node

builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("signal", signal_node)
builder.add_node("medical", medical_reasoning_node)
builder.add_node("trend", trend_node)
builder.add_node("report", report_node)

builder.set_entry_point("planner")

builder.add_edge("planner", "signal")
builder.add_edge("signal", "medical")
builder.add_edge("medical", "trend")
builder.add_edge("trend", "report")
builder.add_edge("report", END)

graph = builder.compile()
