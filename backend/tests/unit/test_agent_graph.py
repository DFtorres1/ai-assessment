from __future__ import annotations

import pytest

from agent.graph import create_graph
from agent.schemas import (
    AnswerCitation,
    AnswerResult,
    FallbackResult,
    QueryExpansionResult,
    RouterResult,
)
from agent.state import AgentState


@pytest.fixture
def mock_router_result():
    def _make(intent, confidence=0.95):
        return RouterResult(intent=intent, confidence=confidence, reasoning="test fixture")

    return _make


@pytest.fixture
def mock_answer_result():
    def _make(answer_text, doc="Login — Security items", page=4):
        return AnswerResult(
            sub_questions=["What is the process?"],
            thought="Test fixture thought",
            evidence_quotes=["Source text from chunk"],
            gaps=None,
            answer=answer_text,
            citations=[
                AnswerCitation(
                    doc_name=doc,
                    page=page,
                    section="Test",
                    supporting_quote="Source text",
                )
            ],
            step_by_step_offered=True,
            confidence=0.92,
        )

    return _make


@pytest.fixture
async def graph(chroma_test_client, mock_anthropic, mock_holidays_api):
    g = await create_graph(chroma_client=chroma_test_client)
    # create_graph overwrites module-level LLMs with real clients; re-apply mocks.
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
    return g


# ── Full flow ──────────────────────────────────────────────────────────────


async def test_full_flow_password_reset(
    graph, mock_anthropic, mock_router_result, mock_answer_result
):
    mock_anthropic.router_llm.structured.return_value = mock_router_result("password_reset")
    mock_anthropic.expander_llm.structured.return_value = QueryExpansionResult(
        queries=[
            "password reset",
            "forgot password reset link",
            "how to change password online banking",
        ]
    )
    mock_anthropic.answer_llm.structured.return_value = mock_answer_result(
        "You can reset your password using the Forgot Password link. Want the step-by-step?"
    )
    state = await graph.ainvoke(
        AgentState(
            session_id="test-1",
            user_type="member",
            message="I forgot my password, how do I reset it?",
            temperature=0.2,
            top_p=0.9,
        )
    )
    assert state["intent"] == "password_reset"
    assert state["output_guard_passed"] is True
    assert len(state["citations"]) > 0
    assert state["answer"]


async def test_answer_result_fields_written_to_state(
    graph, mock_anthropic, mock_router_result, mock_answer_result
):
    mock_anthropic.router_llm.structured.return_value = mock_router_result("account_lockout")
    mock_anthropic.expander_llm.structured.return_value = QueryExpansionResult(
        queries=[
            "account locked out",
            "too many password attempts lockout",
            "how to unlock account",
        ]
    )
    answer = mock_answer_result(
        "After 5 failed attempts your account locks for 30 minutes.",
        page=7,  # account lockout chunk is on page 7
    )
    mock_anthropic.answer_llm.structured.return_value = answer
    state = await graph.ainvoke(
        AgentState(
            session_id="test-fields",
            user_type="member",
            message="I got locked out",
            temperature=0.2,
            top_p=0.9,
        )
    )
    assert state["intent"] == "account_lockout"
    assert state["answer"] == answer.answer
    assert state["citations"][0]["doc_name"] == "Login — Security items"
    assert state["retrieval_confidence"] > 0


async def test_holiday_timing_calls_holidays_tool(
    graph,
    mock_anthropic,
    mock_router_result,
    mock_answer_result,
    mock_holidays_api,
):
    mock_anthropic.router_llm.structured.return_value = mock_router_result("holiday_timing")
    mock_anthropic.expander_llm.structured.return_value = QueryExpansionResult(
        queries=[
            "password reset holiday timing",
            "reset email business day",
            "holiday wait time reset",
        ]
    )
    mock_anthropic.answer_llm.structured.return_value = mock_answer_result(
        "Christmas is a federal holiday, so your reset email will arrive the next business day."
    )
    state = await graph.ainvoke(
        AgentState(
            session_id="test-3",
            user_type="member",
            message="If I start a password reset on Christmas, when should I expect the next step?",
            temperature=0.2,
            top_p=0.9,
        )
    )
    assert any(tc["tool"] == "holidays_api" for tc in state["tool_calls"])
    assert state["holiday_context"] is not None


async def test_out_of_scope_routes_to_rejection_no_answer(
    graph, mock_anthropic, mock_router_result
):
    mock_anthropic.router_llm.structured.return_value = mock_router_result(
        "out_of_scope", confidence=0.99
    )
    state = await graph.ainvoke(
        AgentState(
            session_id="test-4",
            user_type="member",
            message="What's my current account balance?",
            temperature=0.2,
            top_p=0.9,
        )
    )
    assert state["intent"] == "out_of_scope"
    mock_anthropic.answer_llm.structured.assert_not_called()
    assert state["answer"]  # rejection node provides a polite message


async def test_staff_type_passed_to_answer_node(
    graph, mock_anthropic, mock_router_result, mock_answer_result
):
    mock_anthropic.router_llm.structured.return_value = mock_router_result("phone_banking")
    mock_anthropic.expander_llm.structured.return_value = QueryExpansionResult(
        queries=[
            "phone banking unlock",
            "IVR user unlock admin",
            "back office unlock phone banking user",
        ]
    )
    mock_anthropic.answer_llm.structured.return_value = mock_answer_result(
        "As a staff member, you can unlock a phone banking user from the back office admin panel."
    )
    await graph.ainvoke(
        AgentState(
            session_id="test-5",
            user_type="staff",
            message="How do I unlock a phone banking user?",
            temperature=0.2,
            top_p=0.9,
        )
    )
    call_args = mock_anthropic.answer_llm.structured.call_args
    assert "staff" in str(call_args)


async def test_low_confidence_retrieval_routes_to_fallback(
    graph, mock_anthropic, mock_router_result
):
    mock_anthropic.router_llm.structured.return_value = mock_router_result(
        "mfa_issue", confidence=0.60
    )
    mock_anthropic.expander_llm.structured.return_value = QueryExpansionResult(
        queries=["mfa issue", "verification code problem", "2fa not working"]
    )
    mock_anthropic.fallback_llm.structured.return_value = FallbackResult(
        clarifying_question="Happy to help! Are you referring to a code sent by text, or from an authenticator app?",
        suggested_intent="mfa_issue",
    )
    state = await graph.ainvoke(
        AgentState(
            session_id="test-6",
            user_type="member",
            message="something is wrong with my code thing",
            temperature=0.2,
            top_p=0.9,
        )
    )
    mock_anthropic.answer_llm.structured.assert_not_called()
    mock_anthropic.fallback_llm.structured.assert_called_once()
    assert "clarif" in state["answer"].lower() or "?" in state["answer"]


async def test_session_history_preserved_across_turns(
    graph, mock_anthropic, mock_router_result, mock_answer_result, test_db
):
    from services.conversation import ConversationService

    service = ConversationService(store=test_db, graph=graph)
    session_id = "test-session-persist"

    mock_anthropic.router_llm.structured.return_value = mock_router_result("password_reset")
    mock_anthropic.expander_llm.structured.return_value = QueryExpansionResult(
        queries=["q1", "q2", "q3"]
    )
    mock_anthropic.answer_llm.structured.return_value = mock_answer_result(
        "Use the Forgot Password link."
    )
    await service.run(session_id, "member", "How do I reset my password?")

    mock_anthropic.router_llm.structured.return_value = mock_router_result("holiday_timing")
    state2 = await service.run(session_id, "member", "What about on a holiday?")

    # After two turns, session_history fed into turn 2 should have 2 messages
    assert len(state2["session_history"]) >= 2


async def test_answer_confidence_below_threshold_triggers_reflexion(
    graph, mock_anthropic, mock_router_result
):
    mock_anthropic.router_llm.structured.return_value = mock_router_result("password_reset")
    mock_anthropic.expander_llm.structured.return_value = QueryExpansionResult(
        queries=["q1", "q2", "q3"]
    )
    low_confidence_answer = AnswerResult(
        thought="Not sure",
        evidence_quotes=[],
        gaps="Chunks don't cover this",
        answer="You might be able to reset your password somewhere.",
        citations=[],
        step_by_step_offered=False,
        confidence=0.45,
    )
    mock_anthropic.answer_llm.structured.return_value = low_confidence_answer
    state = await graph.ainvoke(
        AgentState(
            session_id="test-reflexion",
            user_type="member",
            message="how do i reset?",
            temperature=0.2,
            top_p=0.9,
        )
    )
    assert state["reflexion_attempts"] >= 1
    assert "clarif" in state["answer"].lower() or "help" in state["answer"].lower()


# ── Per-stage timing ──────────────────────────────────────────────────────────


async def test_per_stage_timing_populated(
    graph, mock_anthropic, mock_router_result, mock_answer_result
):
    mock_anthropic.router_llm.structured.return_value = mock_router_result("password_reset")
    mock_anthropic.expander_llm.structured.return_value = QueryExpansionResult(
        queries=["password reset", "forgot password", "reset link"]
    )
    mock_anthropic.answer_llm.structured.return_value = mock_answer_result(
        "Click Forgot Password on the login page."
    )
    state = await graph.ainvoke(
        AgentState(
            session_id="test-timing",
            user_type="member",
            message="I forgot my password",
            temperature=0.2,
            top_p=0.9,
        )
    )
    timing = state.get("timing", {})
    assert "routing" in timing
    assert "query_expansion" in timing
    assert "retrieval" in timing
    assert "llm_answer" in timing
    assert "output_guard" in timing
    assert all(isinstance(v, float) and v >= 0 for v in timing.values())


async def test_holiday_timing_includes_holidays_tool_key(
    graph, mock_anthropic, mock_router_result, mock_answer_result, mock_holidays_api
):
    mock_anthropic.router_llm.structured.return_value = mock_router_result("holiday_timing")
    mock_anthropic.expander_llm.structured.return_value = QueryExpansionResult(
        queries=["holiday reset timing", "christmas password reset", "business day reset"]
    )
    mock_anthropic.answer_llm.structured.return_value = mock_answer_result(
        "Christmas is a federal holiday — your reset email will arrive the next business day."
    )
    state = await graph.ainvoke(
        AgentState(
            session_id="test-timing-holiday",
            user_type="member",
            message="If I reset my password on Christmas, when will the email arrive?",
            temperature=0.2,
            top_p=0.9,
        )
    )
    assert "holidays_tool" in state.get("timing", {})


# ── Meta-prompting ────────────────────────────────────────────────────────────


def test_select_answer_llm_uses_override_when_set():
    """select_answer_llm returns the module-level mock override in test mode."""
    from unittest.mock import MagicMock

    import agent.nodes.answer as ans_mod

    sentinel = MagicMock()
    original = ans_mod.answer_llm
    try:
        ans_mod.answer_llm = sentinel
        from agent.nodes.answer import select_answer_llm

        result = select_answer_llm(confidence=0.95)
        assert result is sentinel
    finally:
        ans_mod.answer_llm = original


def test_select_answer_llm_picks_haiku_for_high_confidence(monkeypatch):
    """select_answer_llm selects haiku model when confidence ≥ 0.90."""
    from unittest.mock import MagicMock

    import agent.nodes.answer as ans_mod

    monkeypatch.setattr(ans_mod, "answer_llm", None)
    haiku_mock = MagicMock()
    haiku_mock._model = "haiku"
    sonnet_mock = MagicMock()
    sonnet_mock._model = "sonnet"
    monkeypatch.setattr(ans_mod, "haiku_llm", haiku_mock)
    monkeypatch.setattr(ans_mod, "sonnet_llm", sonnet_mock)

    from agent.nodes.answer import select_answer_llm

    result = select_answer_llm(confidence=0.95)
    assert result is haiku_mock


def test_select_answer_llm_picks_sonnet_for_low_confidence(monkeypatch):
    """select_answer_llm selects sonnet model when confidence < 0.90."""
    from unittest.mock import MagicMock

    import agent.nodes.answer as ans_mod

    monkeypatch.setattr(ans_mod, "answer_llm", None)
    haiku_mock = MagicMock()
    haiku_mock._model = "haiku"
    sonnet_mock = MagicMock()
    sonnet_mock._model = "sonnet"
    monkeypatch.setattr(ans_mod, "haiku_llm", haiku_mock)
    monkeypatch.setattr(ans_mod, "sonnet_llm", sonnet_mock)

    from agent.nodes.answer import select_answer_llm

    result = select_answer_llm(confidence=0.75)
    assert result is sonnet_mock


# ── Output guard embedding cosine pre-check ───────────────────────────────────


def test_check_embedding_grounding_passes_when_no_embed_fn():
    """Embedding check is a no-op (returns True) when _embed_fn is None."""
    from unittest.mock import MagicMock

    import agent.nodes.answer as ans_mod

    original = ans_mod._embed_fn
    try:
        ans_mod._embed_fn = None
        from agent.nodes.answer import _check_embedding_grounding

        chunk = MagicMock()
        chunk.text = "password reset information"
        result = _check_embedding_grounding("reset my password", [chunk])
        assert result is True
    finally:
        ans_mod._embed_fn = original


def test_check_embedding_grounding_low_similarity_returns_false(monkeypatch):
    """Embedding check returns False when cosine similarity is below threshold."""
    from unittest.mock import MagicMock

    import numpy as np

    import agent.nodes.answer as ans_mod

    # Two orthogonal vectors → cosine similarity = 0.0
    answer_emb = np.array([1.0, 0.0, 0.0])
    chunk_emb = np.array([0.0, 1.0, 0.0])

    monkeypatch.setattr(
        ans_mod,
        "_embed_fn",
        lambda texts: [answer_emb.tolist()] + [chunk_emb.tolist()] * (len(texts) - 1),
    )

    chunk = MagicMock()
    chunk.text = "irrelevant content"
    from agent.nodes.answer import _check_embedding_grounding

    result = _check_embedding_grounding("completely different topic", [chunk])
    assert result is False


async def test_low_embedding_score_triggers_reflexion(
    graph, mock_anthropic, mock_router_result, monkeypatch
):
    """When embedding similarity is below threshold, reflexion runs even with high confidence."""
    import agent.nodes.answer as ans_mod

    # Force embedding to always return orthogonal vectors (similarity = 0.0)
    monkeypatch.setattr(
        ans_mod,
        "_embed_fn",
        lambda texts: [[1.0, 0.0] if i == 0 else [0.0, 1.0] for i in range(len(texts))],
    )

    mock_anthropic.router_llm.structured.return_value = mock_router_result("password_reset")
    mock_anthropic.expander_llm.structured.return_value = QueryExpansionResult(
        queries=["q1", "q2", "q3"]
    )
    high_confidence_answer = AnswerResult(
        thought="Covered",
        evidence_quotes=["Click Forgot Password"],
        answer="Click Forgot Password on the login page.",
        citations=[],
        step_by_step_offered=False,
        confidence=0.92,  # high confidence — would normally skip reflexion
    )
    mock_anthropic.answer_llm.structured.return_value = high_confidence_answer

    state = await graph.ainvoke(
        AgentState(
            session_id="test-embed-check",
            user_type="member",
            message="how do i reset?",
            temperature=0.2,
            top_p=0.9,
        )
    )
    # Embedding check failed (similarity 0.0 < 0.60) → reflexion must run
    assert state["reflexion_attempts"] >= 1


def test_check_embedding_grounding_fails_open_on_exception(monkeypatch):
    """_check_embedding_grounding returns True (fail open) when embed_fn raises."""
    from unittest.mock import MagicMock

    import agent.nodes.answer as ans_mod

    def _bad_embed(texts):
        raise RuntimeError("embedding service unavailable")

    monkeypatch.setattr(ans_mod, "_embed_fn", _bad_embed)
    chunk = MagicMock()
    chunk.text = "some text"
    from agent.nodes.answer import _check_embedding_grounding

    assert _check_embedding_grounding("answer text", [chunk]) is True


async def test_run_reflexion_returns_early_when_all_grounded(monkeypatch):
    """_run_reflexion returns the revised answer on the first attempt when grounding passes."""
    from unittest.mock import AsyncMock, MagicMock

    import agent.guards.output_guard as og_mod

    grounding_result = MagicMock()
    grounding_result.all_claims_grounded = True
    grounding_result.pii_present = False
    grounding_result.issues_found = ["minor phrasing issue"]
    grounding_result.revised_answer = "Well-grounded revised answer."

    grounding_llm_mock = AsyncMock()
    grounding_llm_mock.structured = AsyncMock(return_value=grounding_result)
    monkeypatch.setattr(og_mod, "grounding_llm", grounding_llm_mock)

    chunk = MagicMock()
    chunk.doc_name = "Test Doc"
    chunk.page = 1
    chunk.text = "Source text about password reset."

    from agent.nodes.answer import _run_reflexion

    answer, attempts, issues = await _run_reflexion({}, "Original answer.", [chunk])

    assert answer == "Well-grounded revised answer."
    assert attempts == 1
    assert "minor phrasing issue" in issues


async def test_run_reflexion_revises_and_hedges_when_not_grounded(monkeypatch):
    """_run_reflexion applies revised_answer each attempt, then hedges when exhausted."""
    from unittest.mock import AsyncMock, MagicMock

    import agent.guards.output_guard as og_mod

    grounding_result = MagicMock()
    grounding_result.all_claims_grounded = False  # never passes → all attempts exhausted
    grounding_result.pii_present = False
    grounding_result.issues_found = ["unsupported claim about lockout duration"]
    grounding_result.revised_answer = "Partially revised answer."

    grounding_llm_mock = AsyncMock()
    grounding_llm_mock.structured = AsyncMock(return_value=grounding_result)
    monkeypatch.setattr(og_mod, "grounding_llm", grounding_llm_mock)

    chunk = MagicMock()
    chunk.doc_name = "Test Doc"
    chunk.page = 1
    chunk.text = "Source text."

    from agent.nodes.answer import _MAX_REFLEXION_ATTEMPTS, _run_reflexion

    answer, attempts, issues = await _run_reflexion({}, "Ungrounded answer.", [chunk])

    assert "If you need more details" in answer
    assert attempts == _MAX_REFLEXION_ATTEMPTS
    assert "unsupported claim about lockout duration" in issues


# ── Router node — _example_bank branches ─────────────────────────────────────


async def test_router_node_without_example_bank(monkeypatch):
    """router_node uses static prompt when _example_bank is None."""
    from unittest.mock import AsyncMock

    import agent.nodes.router as router_mod
    from agent.schemas import RouterResult

    monkeypatch.setattr(router_mod, "_example_bank", None)
    mock_llm = AsyncMock()
    mock_llm.structured.return_value = RouterResult(
        intent="password_reset", confidence=0.95, reasoning="test"
    )
    monkeypatch.setattr(router_mod, "router_llm", mock_llm)

    from agent.nodes.router import router_node

    result = await router_node(
        {
            "session_id": "test-no-bank",
            "message": "I forgot my password",
            "session_history": [],
            "timing": {},
            "user_type": "member",
            "temperature": 0.2,
            "top_p": 0.9,
        }
    )
    assert result["intent"] == "password_reset"
    mock_llm.structured.assert_called_once()


async def test_router_node_without_example_bank_with_history(monkeypatch):
    """router_node static prompt includes session history in the else branch."""
    from unittest.mock import AsyncMock

    import agent.nodes.router as router_mod
    from agent.schemas import RouterResult

    monkeypatch.setattr(router_mod, "_example_bank", None)

    captured: list[str] = []

    async def _capture(messages, schema):
        # messages is a list of ChatMessage; extract content from the first one
        captured.append(messages[0].content)
        return RouterResult(intent="mfa_issue", confidence=0.85, reasoning="test")

    mock_llm = AsyncMock()
    mock_llm.structured.side_effect = _capture
    monkeypatch.setattr(router_mod, "router_llm", mock_llm)

    from agent.nodes.router import router_node

    await router_node(
        {
            "session_id": "test-no-bank-history",
            "message": "tell me more about that",
            "session_history": [
                {"role": "user", "content": "How do I reset my password?"},
                {"role": "assistant", "content": "Click Forgot Password."},
            ],
            "timing": {},
            "user_type": "member",
            "temperature": 0.2,
            "top_p": 0.9,
        }
    )
    assert captured and "Recent conversation" in captured[0]


async def test_router_node_example_bank_fallback_on_top_k_error(monkeypatch):
    """router_node falls back to static prompt when _example_bank.top_k raises."""
    from unittest.mock import AsyncMock, MagicMock

    import agent.nodes.router as router_mod
    from agent.schemas import RouterResult

    bad_bank = MagicMock()
    bad_bank.top_k.side_effect = RuntimeError("embed service down")
    monkeypatch.setattr(router_mod, "_example_bank", bad_bank)

    mock_llm = AsyncMock()
    mock_llm.structured.return_value = RouterResult(
        intent="account_lockout", confidence=0.90, reasoning="test"
    )
    monkeypatch.setattr(router_mod, "router_llm", mock_llm)

    from agent.nodes.router import router_node

    result = await router_node(
        {
            "session_id": "test-bad-bank",
            "message": "I'm locked out",
            "session_history": [],
            "timing": {},
            "user_type": "member",
            "temperature": 0.2,
            "top_p": 0.9,
        }
    )
    assert result["intent"] == "account_lockout"
    mock_llm.structured.assert_called_once()
