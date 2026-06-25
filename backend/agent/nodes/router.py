from __future__ import annotations

import time
from typing import Any

import numpy as np

from agent.schemas import QueryExpansionResult, RouterResult
from agent.state import AgentState
from core.ports.llm import ChatMessage
from services.retry import llm_retry

# Module-level LLM clients — monkeypatched in tests
router_llm: Any = None
expander_llm: Any = None

# Active-Prompt: pre-embedded example bank — set by create_graph, None disables
_example_bank: Any = None

# ── Active-Prompt example bank ────────────────────────────────────────────────

# 10 assessment prompts + 10 edge cases
_EXAMPLE_BANK_ENTRIES: list[tuple[str, str]] = [
    # Assessment prompts
    ("I got locked out after entering the wrong password. Can I unlock myself?", "account_lockout"),
    ("What are the password rules? Can you list them quickly?", "password_reset"),
    ("Why do I keep getting verification codes when I log in?", "mfa_issue"),
    ("How often does 'remember this device' expire?", "remember_device"),
    ("I forgot my username — how do I recover it?", "username_recovery"),
    ("I changed phones and now my codes don't work. What should I do?", "mfa_issue"),
    ("Please help me reset my password safely.", "password_reset"),
    ("Can I unlock a phone-banking user without calling support?", "phone_banking"),
    ("I signed up, but I'm stuck — where do I finish my setup?", "account_setup"),
    (
        "If I start a password reset on a federal holiday, when should I expect the next step?",
        "holiday_timing",
    ),
    # Edge cases
    ("What about on a holiday?", "holiday_timing"),
    ("My account has been blocked after too many wrong attempts.", "account_lockout"),
    ("I can't remember what username I used to register.", "username_recovery"),
    ("The verification code from my authenticator app is not working.", "mfa_issue"),
    ("Will my reset email arrive on Christmas Day?", "holiday_timing"),
    ("I need to enroll in online banking for the first time.", "account_setup"),
    ("How do I stop being asked for a code every single time I log in?", "remember_device"),
    ("As a staff member, how do I reset a member's phone banking PIN?", "phone_banking"),
    ("The reset link I received has already expired.", "password_reset"),
    ("Can I send money to my sister?", "out_of_scope"),
]


class ExampleBank:
    """Pre-embedded example bank for Active-Prompt router selection."""

    def __init__(self, entries: list[tuple[str, str]], embed_fn: Any) -> None:
        self._embed_fn = embed_fn
        self.texts = [e[0] for e in entries]
        self.intents = [e[1] for e in entries]
        raw = embed_fn(self.texts)
        embs = np.array(raw, dtype=float)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        self._embeddings = embs / np.maximum(norms, 1e-8)

    def top_k(self, message: str, k: int = 4) -> list[tuple[str, str]]:
        raw = self._embed_fn([message])
        qv = np.array(raw[0], dtype=float)
        norm = np.linalg.norm(qv)
        qv = qv / max(norm, 1e-8)
        scores = self._embeddings @ qv
        idxs = np.argsort(scores)[::-1][:k]
        return [(self.texts[i], self.intents[i]) for i in idxs]


# ── Router prompts ────────────────────────────────────────────────────────────

_ROUTER_BASE = """You are Blossom's intent router. Classify the user's login & security message
into exactly one intent. Be decisive — pick the closest match.

If a "Previous assistant turn" is provided, use it to resolve ambiguous follow-ups:
• "what about on a holiday?" after a password reset answer → holiday_timing
• "can you give me the step-by-step?" after any in-scope answer → same intent as prior turn
• "tell me more" → same intent as prior turn

Intent definitions:
account_lockout    — locked out, too many failed attempts, account suspended/blocked
password_reset     — forgot password, change password, reset link not arriving
mfa_issue          — verification codes not arriving, authenticator app problems, 2FA setup
remember_device    — "remember this device", trusted device expiry, stay signed in
username_recovery  — forgot username, username not recognized at login
account_setup      — setting up online/digital banking for the first time, enrollment
phone_banking      — phone banking or IVR user: create, lock, unlock, reset
holiday_timing     — question about timing of a login/security action around a holiday
out_of_scope       — balances, transfers, loans, rates, investments, fraud disputes,
                     anything unrelated to accessing or securing a Blossom account"""

_STATIC_EXAMPLES = """Examples:
"I entered the wrong password 5 times and now I'm locked out"
→ intent=account_lockout, confidence=0.97

"Can I send money to my friend?"
→ intent=out_of_scope, confidence=0.99

"The verification code keeps arriving even after I log in successfully"
→ intent=mfa_issue, confidence=0.91

"If I reset my password on Christmas will I have to wait for the next business day?"
→ intent=holiday_timing, confidence=0.94

"I forgot my username but I remember my password"
→ intent=username_recovery, confidence=0.96

"As a staff member, how do I unlock a member's phone banking account?"
→ intent=phone_banking, confidence=0.95

"I set up my account last week but I still can't log in for the first time"
→ intent=account_setup, confidence=0.88

"Can you give me the step-by-step?" (after prior answer about password reset)
→ intent=password_reset, confidence=0.91

"What about on a holiday?" (after prior answer about any reset flow)
→ intent=holiday_timing, confidence=0.93"""

# Legacy alias used by tests that import _ROUTER_SYSTEM directly
_ROUTER_SYSTEM = f"{_ROUTER_BASE}\n\n{_STATIC_EXAMPLES}"

_EXPANDER_SYSTEM = """You are a retrieval query optimizer for a banking training knowledge base.
Generate exactly 3 semantically distinct search queries that together cover the user's intent."""


def _build_router_prompt(
    message: str,
    examples: list[tuple[str, str]],
    prior_turn: str | None,
) -> str:
    example_str = "\n\n".join(f'"{text}"\n→ intent={intent}' for text, intent in examples)
    prior = f"\nPrevious assistant turn: {prior_turn}" if prior_turn else ""
    return (
        f"{_ROUTER_BASE}\n\nExamples (selected for this message):\n{example_str}"
        f"{prior}\nCurrent user message: {message}"
    )


# ── Nodes ─────────────────────────────────────────────────────────────────────


@llm_retry
async def router_node(state: AgentState) -> AgentState:
    t0 = time.monotonic()
    message = state["message"]
    session_history = state.get("session_history", [])
    prior_turn = session_history[-1]["content"] if session_history else None

    if _example_bank is not None:
        try:
            examples = _example_bank.top_k(message, k=4)
            prompt = _build_router_prompt(message, examples, prior_turn)
        except Exception:
            # Fail open: fall back to static prompt
            history_str = "\n".join(f"{m['role']}: {m['content']}" for m in session_history[-4:])
            prompt = "\n".join(
                filter(
                    None,
                    [
                        _ROUTER_SYSTEM,
                        history_str and f"\nRecent conversation:\n{history_str}",
                        f"\nCurrent user message: {message}",
                    ],
                )
            )
    else:
        history_str = "\n".join(f"{m['role']}: {m['content']}" for m in session_history[-4:])
        parts = [_ROUTER_SYSTEM]
        if history_str:
            parts.append(f"\nRecent conversation:\n{history_str}")
        parts.append(f"\nCurrent user message: {message}")
        prompt = "\n".join(parts)

    result = await router_llm.structured([ChatMessage(role="human", content=prompt)], RouterResult)
    timing = {**state.get("timing", {}), "routing": time.monotonic() - t0}
    return {
        **state,
        "intent": result.intent,
        "intent_confidence": result.confidence,
        "timing": timing,
    }


@llm_retry
async def query_expander_node(state: AgentState) -> AgentState:
    t0 = time.monotonic()
    prompt = (
        f"{_EXPANDER_SYSTEM}\n\n"
        f"User intent: {state['intent']}\n"
        f"Original message: {state['message']}"
    )
    result = await expander_llm.structured(
        [ChatMessage(role="human", content=prompt)], QueryExpansionResult
    )
    timing = {**state.get("timing", {}), "query_expansion": time.monotonic() - t0}
    return {**state, "expanded_queries": list(result.queries), "timing": timing}
