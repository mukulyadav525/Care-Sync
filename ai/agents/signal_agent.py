from ai.models.state import AgentState
from ai.analysis.pipeline import HealthPipeline

def signal_node(state: AgentState):

    pipeline = HealthPipeline()

    result = pipeline.analyze_csv("ai/data/sample_ppg.csv")

    print("RESULT =", result)

    state["analysis"] = result

    print("STATE =", state)

    return state