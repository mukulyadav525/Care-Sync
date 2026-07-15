"""In-memory, per-process chat history so the /chat endpoint stays
context-aware across turns within the same session. Resets on restart —
there's no persistence layer (that's what services/vector_store.py would be
for, if it grows beyond a simple recent-turns window).
"""
from collections import defaultdict
from threading import Lock

MAX_TURNS = 10

_lock = Lock()
_sessions: dict[str, list[dict[str, str]]] = defaultdict(list)


def get_history(session_id: str) -> list[dict[str, str]]:
    with _lock:
        return list(_sessions[session_id])


def append_turn(session_id: str, question: str, response: str) -> None:
    with _lock:
        history = _sessions[session_id]
        history.append({"question": question, "response": response})
        del history[:-MAX_TURNS]
