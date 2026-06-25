from __future__ import annotations

from typing import Any

from config import settings
from core.ports.session_store import SessionStorePort


class ConversationService:
    def __init__(self, store: SessionStorePort, graph: Any) -> None:
        self.store = store
        self.graph = graph

    async def run(
        self,
        session_id: str,
        user_type: str,
        message: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> dict[str, Any]:
        await self.store.get_or_create_session(session_id, user_type)
        history = await self.store.get_history(
            session_id, last_n=settings.session_history_max_turns * 2
        )
        session_history = [{"role": m.role, "content": m.content} for m in history]

        initial_state: dict[str, Any] = {
            "session_id": session_id,
            "user_type": user_type,
            "message": message,
            "temperature": temperature,
            "top_p": top_p,
            "session_history": session_history,
            "tool_calls": [],
            "retrieved_chunks": [],
            "expanded_queries": [],
            "citations": [],
            "input_guard_passed": True,
            "reflexion_attempts": 0,
            "reflexion_exhausted": False,
            "output_guard_passed": False,
            "holiday_context": None,
            "retrieval_confidence": 0.0,
            "timing": {},
        }

        final_state = await self.graph.ainvoke(initial_state)

        answer = final_state.get("answer", "")
        timing = final_state.get("timing", {})
        await self.store.append_message(session_id, "user", message)
        await self.store.append_message(
            session_id,
            "assistant",
            answer,
            citations=final_state.get("citations", []),
            tool_calls=final_state.get("tool_calls", []),
            timing_ms=timing,
        )

        return final_state
