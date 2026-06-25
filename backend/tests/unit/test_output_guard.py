from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agent.guards.output_guard import OutputGuard
from agent.state import Chunk, Citation


@pytest.fixture
def guard(mock_anthropic):
    return OutputGuard(llm_client=mock_anthropic)


@pytest.fixture
def grounded_chunks() -> list[Chunk]:
    return [
        Chunk(
            doc_name="Login — Security items",
            page=4,
            section="Password Lockout Policy",
            text=(
                "After 5 failed login attempts, the account is locked for 30 minutes. "
                "Members can self-unlock via the 'Forgot Password' link."
            ),
            score=0.91,
            tags=["lockout", "password"],
        )
    ]


def _mock_grounding(
    all_claims_grounded, issues_found, revised_answer, pii_present, pii_description=None
):
    mock = AsyncMock()
    mock.all_claims_grounded = all_claims_grounded
    mock.issues_found = issues_found
    mock.revised_answer = revised_answer
    mock.pii_present = pii_present
    mock.pii_description = pii_description
    return mock


# ── Grounding ─────────────────────────────────────────────────────────────


async def test_passes_fully_grounded_clean_answer(guard, grounded_chunks, mock_anthropic):
    answer = "After 5 failed attempts your account locks for 30 minutes. Use the Forgot Password link to unlock yourself."
    mock_anthropic.structured.return_value = _mock_grounding(True, [], answer, False)
    result = await guard.check(answer=answer, chunks=grounded_chunks, citations=[])
    assert result.passed is True
    assert result.revised_answer == answer


async def test_flags_hallucinated_attempt_count(guard, grounded_chunks, mock_anthropic):
    answer = "Your account locks after 3 attempts and stays locked for 24 hours."
    mock_anthropic.structured.return_value = _mock_grounding(
        False,
        [
            "'3 attempts' not supported — chunks say 5",
            "'24 hours' not supported — chunks say 30 minutes",
        ],
        "Your account locks after 5 failed attempts. You can unlock via the Forgot Password link.",
        False,
    )
    result = await guard.check(answer=answer, chunks=grounded_chunks, citations=[])
    assert result.passed is False
    assert len(result.issues_found) > 0
    assert "3 attempts" in str(result.issues_found) or "24 hours" in str(result.issues_found)


async def test_reflexion_rewrites_unsupported_code(guard, grounded_chunks, mock_anthropic):
    answer = "Your account locks after 3 attempts. Use code UNLOCK22 to regain access."
    mock_anthropic.structured.return_value = _mock_grounding(
        False,
        ["'UNLOCK22' not in source chunks", "'3 attempts' contradicts chunks (5)"],
        "Your account locks after 5 failed attempts. Use the Forgot Password link to unlock.",
        False,
    )
    result = await guard.check(answer=answer, chunks=grounded_chunks, citations=[])
    assert "UNLOCK22" not in result.revised_answer
    assert "5" in result.revised_answer


# ── PII elimination (not masking) ─────────────────────────────────────────


async def test_ssn_in_answer_triggers_rewrite_not_masking(guard, grounded_chunks, mock_anthropic):
    answer = "Your account associated with SSN 123-45-6789 is locked."
    mock_anthropic.structured.return_value = _mock_grounding(
        True,
        [],
        "Your account is locked.",
        True,
        "SSN '123-45-6789' found in sentence 1",
    )
    result = await guard.check(answer=answer, chunks=grounded_chunks, citations=[])
    assert "123-45-6789" not in result.revised_answer
    assert "***" not in result.revised_answer
    assert "[REDACTED]" not in result.revised_answer
    assert result.passed is False  # PII presence = not passed


async def test_account_number_in_answer_rewritten_not_masked(
    guard, grounded_chunks, mock_anthropic
):
    answer = "Account 1234567890 has been unlocked successfully."
    mock_anthropic.structured.return_value = _mock_grounding(
        True,
        [],
        "Your account has been unlocked successfully.",
        True,
    )
    result = await guard.check(answer=answer, chunks=grounded_chunks, citations=[])
    assert "1234567890" not in result.revised_answer
    assert "[ACCOUNT REDACTED]" not in result.revised_answer
    assert "unlocked" in result.revised_answer


async def test_last_four_digits_also_removed(guard, grounded_chunks, mock_anthropic):
    answer = "Your account ending in 4521 is locked."
    mock_anthropic.structured.return_value = _mock_grounding(
        True,
        [],
        "Your account is locked.",
        True,
        "Account reference 'ending in 4521' found",
    )
    result = await guard.check(answer=answer, chunks=grounded_chunks, citations=[])
    assert "4521" not in result.revised_answer
    assert "locked" in result.revised_answer


async def test_clean_answer_with_no_pii_passes_unchanged(guard, grounded_chunks, mock_anthropic):
    answer = "After 5 failed attempts your account is locked. Use the Forgot Password link to unlock yourself."
    mock_anthropic.structured.return_value = _mock_grounding(True, [], answer, False)
    result = await guard.check(answer=answer, chunks=grounded_chunks, citations=[])
    assert result.passed is True
    assert result.revised_answer == answer


async def test_pii_description_not_propagated_to_client(guard, grounded_chunks, mock_anthropic):
    answer = "Account 1234567890 is locked."
    mock_anthropic.structured.return_value = _mock_grounding(
        True,
        [],
        "Your account is locked.",
        True,
        "Account number '1234567890' in sentence 1",
    )
    result = await guard.check(answer=answer, chunks=grounded_chunks, citations=[])
    assert not hasattr(result, "pii_description") or result.pii_description is None


# ── Citation validation ─────────────────────────────────────────────────────


async def test_invalid_citation_stripped(guard, grounded_chunks, mock_anthropic):
    mock_anthropic.structured.return_value = _mock_grounding(
        True, [], "Your account is locked.", False
    )
    bad = Citation(doc_name="Fake Doc", page=999, section="Nonexistent")
    result = await guard.check(
        answer="Your account is locked.", chunks=grounded_chunks, citations=[bad]
    )
    assert bad not in result.valid_citations


async def test_valid_citation_preserved(guard, grounded_chunks, mock_anthropic):
    mock_anthropic.structured.return_value = _mock_grounding(
        True, [], "After 5 attempts the account locks.", False
    )
    good = Citation(doc_name="Login — Security items", page=4, section="Password Lockout Policy")
    result = await guard.check(
        answer="After 5 attempts the account locks.", chunks=grounded_chunks, citations=[good]
    )
    assert good in result.valid_citations
