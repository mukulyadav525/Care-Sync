import os
from dotenv import load_dotenv
from langchain_xai import ChatXAI

load_dotenv()

_llm: ChatXAI | None = None


def get_llm() -> ChatXAI:
    """Lazily construct the Grok client. Deferred (rather than built at
    import time) so a missing/invalid XAI_API_KEY only fails the /chat call
    that needs it, instead of crashing the whole service at startup."""
    global _llm
    if _llm is None:
        _llm = ChatXAI(
            model="grok-4",
            xai_api_key=os.getenv("XAI_API_KEY"),
            temperature=0.2,
        )
    return _llm
