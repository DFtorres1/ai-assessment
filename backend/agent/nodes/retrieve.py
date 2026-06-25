from __future__ import annotations

import time
from typing import Any

from agent.state import AgentState
from config import settings

INTENT_TAGS: dict[str, list[str]] = {
    "account_lockout": ["lockout", "suspend", "unlock"],
    "password_reset": ["password", "reset"],
    "mfa_issue": ["mfa", "verification", "2fa"],
    "remember_device": ["remember_me", "trusted_device"],
    "username_recovery": ["username"],
    "account_setup": ["signup", "setup", "enrollment"],
    "phone_banking": ["phone_banking", "ivr", "unlock"],
    "holiday_timing": [],
}


def make_retrieve_node(vector_store: Any):
    async def retrieve_node(state: AgentState) -> AgentState:
        t0 = time.monotonic()
        intent = state.get("intent", "")
        tag_list = INTENT_TAGS.get(intent, [])
        tags = tag_list if tag_list else None

        queries = [state["message"]] + list(state.get("expanded_queries", []))
        chunks = await vector_store.search_multi(queries=queries, tags=tags, k=5)

        max_score = max((c.score for c in chunks), default=0.0)
        timing = {**state.get("timing", {}), "retrieval": time.monotonic() - t0}

        return {
            **state,
            "retrieved_chunks": chunks,
            "retrieval_confidence": max_score,
            "route_to_fallback": max_score < settings.retrieval_routing_threshold,
            "timing": timing,
        }

    return retrieve_node
