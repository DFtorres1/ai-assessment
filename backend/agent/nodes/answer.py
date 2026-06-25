from __future__ import annotations

import logging
import time
from typing import Any

from agent.schemas import AnswerResult, GroundingResult
from agent.state import AgentState
from core.ports.llm import ChatMessage
from services.retry import llm_retry

_log = logging.getLogger(__name__)

# Module-level LLM clients
# answer_llm: test override — takes priority when not None (monkeypatched in tests)
# haiku_llm / sonnet_llm: set by create_graph for production meta-prompting
answer_llm: object = None
haiku_llm: object = None
sonnet_llm: object = None

# Embedding function for cosine pre-check — set by create_graph, None disables the check
_embed_fn: Any = None

_CONFIDENCE_REFLEXION_THRESHOLD = 0.70
_MAX_REFLEXION_ATTEMPTS = 2

_GROUNDING_SYSTEM = """You are a strict compliance reviewer for a banking assistant's responses.
You have two responsibilities: factual grounding and data safety.

── RESPONSIBILITY 1: FACTUAL GROUNDING ──
Every factual claim in the answer must be traceable to the provided source chunks.
A claim is GROUNDED if you can point to the exact sentence or phrase in a chunk
that supports it. A claim is UNSUPPORTED if it cannot be verified in the chunks.

Unsupported claims to remove:
  • Specific numbers not in the chunks (lockout durations, attempt counts, etc.)
  • Process steps not described in the chunks
  • Phone numbers, URLs, or contact information not in the chunks
  • Policy details that contradict the chunks

Do NOT remove:
  • Transitional phrases and conversational tone ("Great news!", "You're almost there!")
  • Offers to explain further ("Want the step-by-step?")
  • Safe next steps when unsure ("You can contact support for this")

── RESPONSIBILITY 2: DATA SAFETY ──
The answer must contain ZERO personal data — not masked, not hinted at,
not partially shown. If any of the following appear, the sentence must be
completely rewritten or removed:

  • Any digit sequence that could be a member's SSN, account number,
    card number, routing number, or PIN
  • Any value that looks like it came from a member's record
    (e.g., "your account 4521..." even as last-4)
  • Any password, security question answer, or authentication secret
  • Full or partial dates of birth combined with a name
  • Government ID numbers

The rewritten answer must convey the same helpful guidance WITHOUT referencing
any personal data. "Your account ending in 4521 is locked" becomes
"Your account is locked". There is no masking — the data simply does not appear.

── RULES ──
• revised_answer must be self-contained — do not reference removed content
• Preserve warmth and tone in the rewrite
• If all claims are grounded and no PII is present, revised_answer = original answer
• pii_description is for internal logging only — never surfaces to the client"""

_SYSTEM = """You are Blossom's friendly banking helper — warm, encouraging, and confidence-boosting.
You speak to members and staff who need fast, reliable help with login & security.

YOUR PROCESS (fill each field strictly in order):
1. sub_questions — break the question into 1-3 sub-questions
2. thought       — which chunks answer each sub-question
3. evidence_quotes — verbatim quotes from chunks
4. gaps          — sub-questions with no chunk evidence
5. answer        — warm, ≤4 sentences, grounded only in evidence_quotes
6. citations     — doc + page for each evidence source
7. step_by_step_offered — true if answer offers step-by-step
8. confidence    — honest 0.0-1.0 self-assessment

STRICT RULES:
• answer contains ZERO information not in evidence_quotes
• Staff (user_type=staff): may include back-office / admin action guidance
• Member (user_type=member): self-service guidance only"""


def _build_prompt(state: AgentState) -> str:
    chunks = state.get("retrieved_chunks", [])
    chunk_context = "\n\n".join(f"[{c.doc_name}, p.{c.page}] {c.text}" for c in chunks)
    holiday_ctx = state.get("holiday_context") or ""
    history = state.get("session_history", [])
    history_str = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    user_type = state.get("user_type", "member")

    holiday_section = f"Holiday context:\n{holiday_ctx}\n\n" if holiday_ctx else ""
    return (
        f"{_SYSTEM}\n\n"
        f"Context chunks:\n{chunk_context or '(none)'}\n\n"
        f"{holiday_section}"
        f"Session history (last 3 turns):\n{history_str or '(new session)'}\n\n"
        f"User type: {user_type}\n"
        f"User: {state['message']}"
    )


def select_answer_llm(confidence: float) -> Any:
    """Meta-prompting: Haiku for high-confidence (≥0.90), Sonnet otherwise."""
    if answer_llm is not None:
        return answer_llm  # test override — skip model selection
    return haiku_llm if confidence >= 0.90 else sonnet_llm


def _check_embedding_grounding(answer: str, chunks: list[Any]) -> bool:
    """Return True if answer is well-grounded (cosine ≥ threshold), False triggers reflexion."""
    if _embed_fn is None or not chunks:
        return True  # skip when embedder unavailable or no context
    try:
        import numpy as np

        from config import settings

        chunk_texts = [c.text for c in chunks]
        embeddings = _embed_fn([answer] + chunk_texts)
        answer_emb = np.array(embeddings[0])
        chunk_embs = [np.array(e) for e in embeddings[1:]]

        scores = []
        for c_emb in chunk_embs:
            norm = np.linalg.norm(answer_emb) * np.linalg.norm(c_emb)
            if norm > 1e-8:
                scores.append(float(np.dot(answer_emb, c_emb) / norm))

        max_score = max(scores) if scores else 1.0
        return max_score >= settings.reflexion_embedding_threshold
    except Exception:
        return True  # fail open — don't block on embedding errors


async def _run_reflexion(
    state: AgentState, answer_str: str, chunks: list[Any]
) -> tuple[str, int, list[str]]:
    from agent.guards.output_guard import grounding_llm

    chunk_context = "\n\n".join(f"[{c.doc_name}, p.{c.page}] {c.text}" for c in chunks)
    best_answer = answer_str
    accumulated_issues: list[str] = []

    for attempt in range(_MAX_REFLEXION_ATTEMPTS):
        try:
            prompt = (
                f"{_GROUNDING_SYSTEM}\n\n"
                f"Source chunks:\n{chunk_context}\n\n"
                f"Original answer to review:\n{best_answer}"
            )
            grounding = await grounding_llm.structured(
                [ChatMessage(role="human", content=prompt)], GroundingResult
            )

            all_grounded = bool(getattr(grounding, "all_claims_grounded", False))
            has_pii = bool(getattr(grounding, "pii_present", True))
            issues = list(getattr(grounding, "issues_found", []) or [])
            accumulated_issues.extend(issues)

            if all_grounded and not has_pii:
                revised = getattr(grounding, "revised_answer", None)
                if isinstance(revised, str) and revised:
                    best_answer = revised
                return best_answer, attempt + 1, accumulated_issues

            revised = getattr(grounding, "revised_answer", None)
            if isinstance(revised, str) and revised:
                best_answer = revised
        except Exception:
            break

    hedged = best_answer.rstrip() + " If you need more details, our support team is happy to help."
    return hedged, _MAX_REFLEXION_ATTEMPTS, accumulated_issues


@llm_retry
async def answer_node(state: AgentState) -> AgentState:
    t_llm = time.monotonic()

    intent_confidence = state.get("intent_confidence", 0.0)
    llm = select_answer_llm(intent_confidence)

    prompt = _build_prompt(state)
    result = await llm.structured([ChatMessage(role="human", content=prompt)], AnswerResult)

    llm_duration = time.monotonic() - t_llm

    citations = [
        {"doc_name": c.doc_name, "page": c.page, "section": c.section} for c in result.citations
    ]

    answer_str = result.answer
    reflexion_attempts = 0

    t_guard = time.monotonic()
    chunks = state.get("retrieved_chunks", [])

    # Strip LLM-hallucinated citations not backed by any retrieved chunk
    valid_doc_pages = {(c.doc_name, c.page) for c in chunks}
    valid_citations = [c for c in citations if (c["doc_name"], c["page"]) in valid_doc_pages]
    if len(valid_citations) < len(citations):
        _log.warning(
            "session=%s stripped %d unsupported citation(s)",
            state.get("session_id"),
            len(citations) - len(valid_citations),
        )
    citations = valid_citations

    from config import settings

    confidence_passes = result.confidence >= settings.reflexion_confidence_threshold
    embedding_passes = _check_embedding_grounding(answer_str, chunks)
    output_guard_passed = confidence_passes and embedding_passes

    hallucination_issues: list[str] = []
    if not output_guard_passed:
        answer_str, reflexion_attempts, hallucination_issues = await _run_reflexion(
            state, answer_str, chunks
        )
        output_guard_passed = reflexion_attempts < _MAX_REFLEXION_ATTEMPTS

    output_guard_duration = time.monotonic() - t_guard

    timing = {
        **state.get("timing", {}),
        "llm_answer": llm_duration,
        "output_guard": output_guard_duration,
    }

    return {
        **state,
        "answer": answer_str,
        "citations": citations,
        "confidence": result.confidence,
        "output_guard_passed": output_guard_passed,
        "reflexion_attempts": reflexion_attempts,
        "reflexion_exhausted": reflexion_attempts >= _MAX_REFLEXION_ATTEMPTS,
        "hallucination_issues": hallucination_issues,
        "timing": timing,
    }
