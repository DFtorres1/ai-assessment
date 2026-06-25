from __future__ import annotations

from typing import Any

from agent.schemas import FallbackResult
from agent.state import AgentState
from core.ports.llm import ChatMessage
from services.retry import llm_retry

# Module-level LLM client — monkeypatched in tests
fallback_llm: Any = None

_FALLBACK_SYSTEM = """You are Blossom's banking helper. The knowledge base did not return a
confident match for this question. Do NOT guess or fabricate an answer.

Your ONLY job: ask one warm, targeted clarifying question that will help
identify exactly what the user needs. Keep it to one sentence."""

_REJECTION_MESSAGE = (
    "I'm here to help with Blossom login and security questions — things like "
    "password resets, account lockouts, MFA setup, and phone banking. "
    "For account balances, transfers, or other banking services, please use "
    "the main app or reach out to our support team. Happy to help with anything "
    "login or security related!"
)


@llm_retry
async def fallback_node(state: AgentState) -> AgentState:
    message = state.get("message", "")
    intent = state.get("intent", "")
    history = state.get("session_history", [])
    history_str = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:])
    prompt = (
        f"{_FALLBACK_SYSTEM}\n\n"
        + (f"Conversation so far:\n{history_str}\n\n" if history_str else "")
        + f"User: {message}\n"
        + f"Detected intent (low confidence): {intent}"
    )
    result = await fallback_llm.structured(
        [ChatMessage(role="human", content=prompt)], FallbackResult
    )
    return {
        **state,
        "answer": result.clarifying_question,
        "citations": [],
        "output_guard_passed": True,
        "reflexion_attempts": 0,
        "reflexion_exhausted": False,
    }


async def rejection_node(state: AgentState) -> AgentState:
    return {
        **state,
        "answer": _REJECTION_MESSAGE,
        "citations": [],
        "output_guard_passed": True,
        "reflexion_attempts": 0,
        "reflexion_exhausted": False,
    }
