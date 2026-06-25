from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from typing_extensions import TypedDict


@dataclass
class Chunk:
    doc_name: str
    page: int
    section: str
    text: str
    score: float
    tags: list[str] = field(default_factory=list)


@dataclass
class Citation:
    doc_name: str
    page: int
    section: str


class AgentState(TypedDict):
    # ── Request fields ──────────────────────────────────────────────────────
    session_id: str
    user_type: Literal["member", "staff"]
    message: str
    temperature: float
    top_p: float

    # ── Session continuity ──────────────────────────────────────────────────
    session_history: list[dict[str, str]]  # [{"role": ..., "content": ...}]

    # ── Intent classification ───────────────────────────────────────────────
    intent: str
    intent_confidence: float
    route_to_fallback: bool

    # ── Retrieval ───────────────────────────────────────────────────────────
    expanded_queries: list[str]
    retrieved_chunks: list[Any]
    retrieval_confidence: float

    # ── Holiday context ─────────────────────────────────────────────────────
    holiday_context: str | None

    # ── Answer ──────────────────────────────────────────────────────────────
    answer: str
    citations: list[Any]
    tool_calls: list[Any]
    confidence: float

    # ── Guards ──────────────────────────────────────────────────────────────
    input_guard_passed: bool
    output_guard_passed: bool
    hallucination_issues: list[str]
    reflexion_attempts: int
    reflexion_exhausted: bool

    # ── Observability ───────────────────────────────────────────────────────
    timing: dict[str, float]
