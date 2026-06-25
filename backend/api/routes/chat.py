from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from typing import Any, Literal

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from api.models import ChatRequest, ChatResponse, Citation, ToolCall

router = APIRouter()
_log = structlog.get_logger()

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


def _to_api_citations(raw: list[Any]) -> list[Citation]:
    out = []
    for c in raw:
        if isinstance(c, dict):
            out.append(
                Citation(
                    doc_name=c.get("doc_name", ""),
                    page=c.get("page", 0),
                    section=c.get("section", ""),
                )
            )
        else:
            out.append(
                Citation(
                    doc_name=getattr(c, "doc_name", ""),
                    page=getattr(c, "page", 0),
                    section=getattr(c, "section", ""),
                )
            )
    return out


def _to_api_tool_calls(raw: list[Any]) -> list[ToolCall]:
    out = []
    for tc in raw:
        d = tc if isinstance(tc, dict) else vars(tc)
        out.append(
            ToolCall(
                tool=d.get("tool", "unknown"),
                input={k: v for k, v in d.items() if k != "tool"},
                result=None,
                duration_ms=0.0,
            )
        )
    return out


@router.post("/chat", response_model=ChatResponse)
async def post_chat(req: ChatRequest, request: Request) -> ChatResponse | JSONResponse:
    t_start = time.monotonic()

    from agent.guards.input_guard import InputGuard

    service = request.app.state.conversation
    history = await service.store.get_history(req.session_id, last_n=4)
    session_history = [{"role": m.role, "content": m.content} for m in history]

    t_guard = time.monotonic()
    guard = InputGuard()
    guard_result = await guard.check(req.message, session_history)
    guard_ms = round((time.monotonic() - t_guard) * 1000)

    if guard_result.blocked:
        return JSONResponse(
            status_code=400,
            content={
                "error": "BLOCKED_INPUT",
                "message": guard_result.user_message or "Your message cannot be processed.",
                "request_id": req.session_id,
            },
        )
    final_state = await service.run(
        session_id=req.session_id,
        user_type=req.user_type,
        message=req.message,
        temperature=req.temperature,
        top_p=req.top_p,
    )

    raw_timing = final_state.get("timing", {})
    timing_ms: dict[str, int] = {k: round(v * 1000) for k, v in raw_timing.items()}
    timing_ms["input_guard"] = guard_ms
    timing_ms["total"] = round((time.monotonic() - t_start) * 1000)

    _log.info(
        "chat_request",
        session_id=req.session_id,
        intent=final_state.get("intent"),
        intent_confidence=final_state.get("intent_confidence"),
        retrieval_hits=[
            {"doc": c.doc_name, "page": c.page, "score": c.score}
            for c in final_state.get("retrieved_chunks", [])
        ],
        tool_calls=[
            tc.get("tool") if isinstance(tc, dict) else getattr(tc, "tool", None)
            for tc in final_state.get("tool_calls", [])
        ],
        output_guard_passed=final_state.get("output_guard_passed"),
        reflexion_attempts=final_state.get("reflexion_attempts"),
        timing_ms=timing_ms,
    )

    return ChatResponse(
        answer=final_state.get("answer", ""),
        citations=_to_api_citations(final_state.get("citations", [])),
        tool_calls=_to_api_tool_calls(final_state.get("tool_calls", [])),
        timing_ms=timing_ms,
    )


@router.get("/chat/stream")
async def stream_chat(
    request: Request,
    session_id: str,
    message: str,
    user_type: Literal["member", "staff"] = "member",
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> StreamingResponse:
    from agent.guards.input_guard import InputGuard

    service = request.app.state.conversation
    history = await service.store.get_history(session_id, last_n=4)
    session_history = [{"role": m.role, "content": m.content} for m in history]

    guard = InputGuard()
    guard_result = await guard.check(message, session_history)

    if guard_result.blocked:
        rejection_text = (
            guard_result.user_message
            or "I can only help with login and security topics for Blossom Banking."
        )

        async def _rejection_stream() -> AsyncGenerator[str, None]:
            yield f"data: {json.dumps({'type': 'citations', 'citations': []})}\n\n"
            for word in rejection_text.split():
                yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
                await asyncio.sleep(0.02)
            yield f"data: {json.dumps({'type': 'done', 'tool_calls': []})}\n\n"

        return StreamingResponse(
            _rejection_stream(), media_type="text/event-stream", headers=_SSE_HEADERS
        )

    service = request.app.state.conversation

    async def _event_stream() -> AsyncGenerator[str, None]:
        _evt: dict[str, Any] = {"type": "status", "content": "Searching knowledge base..."}
        yield f"data: {json.dumps(_evt)}\n\n"

        t_start_stream = time.monotonic()
        final_state = await service.run(session_id, user_type, message, temperature, top_p)
        answer = final_state.get("answer", "")
        citations = _to_api_citations(final_state.get("citations", []))
        raw_tool_calls = final_state.get("tool_calls", [])
        tool_calls = _to_api_tool_calls(raw_tool_calls)

        # Emit tool events for any tools that were called
        for tc in raw_tool_calls:
            tool_name = (
                tc.get("tool", "unknown")
                if isinstance(tc, dict)
                else getattr(tc, "tool", "unknown")
            )
            duration_ms = tc.get("duration_ms", 0) if isinstance(tc, dict) else 0
            tool_result = (
                final_state.get("holiday_context") if tool_name == "holidays_api" else None
            )
            tc_input = (
                {k: v for k, v in tc.items() if k not in ("tool", "duration_ms")}
                if isinstance(tc, dict)
                else {}
            )
            _evt = {"type": "tool_start", "tool": tool_name, "input": tc_input}
            yield f"data: {json.dumps(_evt)}\n\n"
            _evt = {
                "type": "tool_end",
                "tool": tool_name,
                "result": tool_result,
                "duration_ms": duration_ms,
            }
            yield f"data: {json.dumps(_evt)}\n\n"

        _evt = {
            "type": "citations",
            "citations": [c.model_dump() for c in citations],
        }
        yield f"data: {json.dumps(_evt)}\n\n"
        for word in answer.split():
            yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
            await asyncio.sleep(0.02)

        raw_timing = final_state.get("timing", {})
        timing_ms: dict[str, int] = {k: round(v * 1000) for k, v in raw_timing.items()}
        timing_ms["total"] = round((time.monotonic() - t_start_stream) * 1000)

        _log.info(
            "chat_request",
            session_id=session_id,
            intent=final_state.get("intent"),
            intent_confidence=final_state.get("intent_confidence"),
            retrieval_hits=[
                {"doc": c.doc_name, "page": c.page, "score": c.score}
                for c in final_state.get("retrieved_chunks", [])
            ],
            tool_calls=[
                tc.get("tool") if isinstance(tc, dict) else getattr(tc, "tool", None)
                for tc in final_state.get("tool_calls", [])
            ],
            output_guard_passed=final_state.get("output_guard_passed"),
            reflexion_attempts=final_state.get("reflexion_attempts"),
            timing_ms=timing_ms,
        )

        _evt = {
            "type": "done",
            "tool_calls": [tc.model_dump() for tc in tool_calls],
            "timing_ms": timing_ms,
        }
        yield f"data: {json.dumps(_evt)}\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)
