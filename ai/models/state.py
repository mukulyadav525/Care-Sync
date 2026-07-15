from typing import TypedDict
from typing import List
from typing import Dict
from typing import Any

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: List[BaseMessage]

    session_id: str

    history: List[Dict[str, str]]

    intent: str

    analysis: Dict[str, Any]

    response: str
