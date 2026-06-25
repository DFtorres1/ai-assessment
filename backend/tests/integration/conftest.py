from __future__ import annotations

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from agent.schemas import (
    AnswerCitation,
    AnswerResult,
    FallbackResult,
    QueryExpansionResult,
    RouterResult,
)


@pytest.fixture
async def async_http_client(
    chroma_test_client: Any,
    mock_anthropic: Any,
    mock_holidays_api: Any,
    test_db: Any,
) -> Any:
    """
    Integration test client with mocked LLMs, holidays API, and test ChromaDB/DB.

    Sets sensible defaults on the LLM mocks so tests can selectively override.
    """
    from agent.graph import create_graph
    from api.main import app

    mock_anthropic.router_llm.structured.return_value = RouterResult(
        intent="password_reset",
        confidence=0.95,
        reasoning="integration test default",
    )
    mock_anthropic.expander_llm.structured.return_value = QueryExpansionResult(
        queries=[
            "password reset",
            "forgot password link",
            "how to reset online banking password",
        ]
    )
    mock_anthropic.answer_llm.structured.return_value = AnswerResult(
        thought="chunk covers this",
        evidence_quotes=["To reset your password, click Forgot Password on the login page."],
        answer="To reset your password, click the Forgot Password link on the login page.",
        citations=[
            AnswerCitation(
                doc_name="Login — Security items",
                page=4,
                section="Password Reset",
                supporting_quote="click Forgot Password",
            )
        ],
        step_by_step_offered=False,
        confidence=0.92,
    )

    from agent.schemas import InputGuardResult
    from services.conversation import ConversationService

    mock_anthropic.fallback_llm.structured.return_value = FallbackResult(
        clarifying_question="Could you give me more details so I can help you better?",
        suggested_intent=None,
    )

    # Input guard passes by default in integration tests
    mock_anthropic.input_guard_llm.structured.return_value = InputGuardResult(
        in_scope=True,
        pii_detected=False,
        pii_type=None,
        is_jailbreak=False,
        jailbreak_category=None,
        block=False,
        block_reason=None,
        user_message=None,
    )

    graph = await create_graph(chroma_client=chroma_test_client)

    # create_graph sets module-level LLM vars to real API clients.
    # Re-apply the mocks so integration tests stay offline.
    import agent.guards.input_guard as _input_guard_mod
    import agent.guards.output_guard as _output_guard_mod
    import agent.nodes.answer as _answer_mod
    import agent.nodes.fallback as _fallback_mod
    import agent.nodes.router as _router_mod

    _router_mod.router_llm = mock_anthropic.router_llm
    _router_mod.expander_llm = mock_anthropic.expander_llm
    _answer_mod.answer_llm = mock_anthropic.answer_llm
    _fallback_mod.fallback_llm = mock_anthropic.fallback_llm
    _output_guard_mod.grounding_llm = mock_anthropic.grounding_llm
    _input_guard_mod.input_guard_llm = mock_anthropic.input_guard_llm

    conversation = ConversationService(store=test_db, graph=graph)

    app.state.chroma = chroma_test_client
    app.state.graph = graph
    app.state.db = test_db
    app.state.conversation = conversation

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

    for attr in ("graph", "db", "chroma", "conversation"):
        if hasattr(app.state, attr):
            delattr(app.state, attr)
