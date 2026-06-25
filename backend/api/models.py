from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=2000)
    user_type: Literal["member", "staff"] = "member"
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


class Citation(BaseModel):
    doc_name: str
    page: int
    section: str


class ToolCall(BaseModel):
    tool: str
    input: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    duration_ms: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    timing_ms: dict[str, int] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str


class IngestResponse(BaseModel):
    chunks_indexed: int
    documents_processed: list[str]
    duration_ms: int


class HealthCheck(BaseModel):
    status: str
    checks: dict[str, str]
