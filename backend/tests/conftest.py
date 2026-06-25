from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# Point HF cache to a local writable dir (system .cache is root-owned in CI/dev)
_LOCAL_CACHE = str(Path(__file__).parent.parent / ".cache" / "huggingface")
os.environ.setdefault("HF_HOME", _LOCAL_CACHE)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", _LOCAL_CACHE)

import chromadb
import pytest
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from httpx import ASGITransport, AsyncClient

# ── Session-scoped embedding function (model loaded once per run) ─────────────


@pytest.fixture(scope="session")
def embedding_function() -> SentenceTransformerEmbeddingFunction:
    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


# ── ChromaDB in-memory client with 3 seeded knowledge chunks ─────────────────


@pytest.fixture(scope="session")
def chroma_test_client(
    embedding_function: SentenceTransformerEmbeddingFunction,
) -> chromadb.ClientAPI:
    """
    Ephemeral (in-memory) ChromaDB seeded with 5 login/security knowledge chunks.
    Session-scoped — created once and shared across all tests in a run.
    """
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="blossom_knowledge",
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        documents=[
            (
                "Forgot your password or need to reset it? Click 'Forgot Password' on the login page. "
                "We'll send a secure password reset link to your verified email address. "
                "The reset link expires in 24 hours — request a new one if it has expired. "
                "After you reset your password, sign in normally with your new credentials."
            ),
            (
                "If you got locked out of your account, this happens after 5 consecutive failed "
                "login attempts for security. Your account is locked to protect you from unauthorized access. "
                "To unlock your locked account, request a password reset via your verified email. "
                "You can also contact Blossom support to unlock your account immediately."
            ),
            (
                "Verification codes are sent via SMS or email when Blossom detects an "
                "unrecognized device. This is part of multi-factor authentication (MFA) "
                "to protect your account. If your device is remembered and trusted, you won't be "
                "asked for a verification code each time you sign in. "
                "MFA codes expire after 10 minutes — request a new code if yours has expired."
            ),
            (
                "If you forgot your username and cannot sign in, you can recover it by visiting "
                "the login page and clicking 'Forgot Username'. Enter your registered email address "
                "and we will send your username to that email. Username recovery is available 24/7."
            ),
            (
                "Phone banking users can be unlocked by a back-office staff member in the admin portal. "
                "To unlock a phone banking account, go to Member Admin, search for the member, "
                "and select 'Unlock Phone Banking'. Staff can also reset the phone banking PIN "
                "or IVR credentials from this admin screen."
            ),
        ],
        metadatas=[
            {
                "doc_name": "Login — Security items",
                "page": 4,
                "section": "Password Reset",
                "tags": "password,reset",
            },
            {
                "doc_name": "Login — Security items",
                "page": 7,
                "section": "Account Lockout",
                "tags": "lockout,password,unlock",
            },
            {
                "doc_name": "Personal Banking Training",
                "page": 12,
                "section": "MFA Verification",
                "tags": "mfa,verification,remember_me",
            },
            {
                "doc_name": "Login — Security items",
                "page": 11,
                "section": "Username Recovery",
                "tags": "username,recovery",
            },
            {
                "doc_name": "Magic Training — Back Office",
                "page": 8,
                "section": "Phone Banking Admin",
                "tags": "phone_banking,unlock,staff",
            },
        ],
        ids=["chunk-001", "chunk-002", "chunk-003", "chunk-004", "chunk-005"],
    )
    return client


# ── In-memory SQLite session store (per-test isolation) ──────────────────────


@pytest.fixture
async def test_db() -> Any:
    """
    Fresh in-memory SQLite SessionStore for each test.
    Uses ':memory:' so tests never touch the filesystem and never share state.
    """
    from services.sessions import SessionStore

    store = SessionStore(db_path=":memory:")
    await store.initialize()
    yield store
    await store.close()


# ── Mocked Anthropic LLM clients ─────────────────────────────────────────────


@pytest.fixture
def mock_anthropic(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """
    Patches all module-level LLM client references used by graph nodes and guards.

    Tests set return values on the individual AsyncMocks to control LLM behavior
    without making real API calls:

        mock_anthropic.router_llm.ainvoke.return_value = RouterResult(...)
        mock_anthropic.answer_llm.ainvoke.return_value = AnswerResult(...)
    """
    m = MagicMock()
    m.router_llm = AsyncMock()
    m.expander_llm = AsyncMock()
    m.answer_llm = AsyncMock()
    m.fallback_llm = AsyncMock()
    m.grounding_llm = AsyncMock()
    m.input_guard_llm = AsyncMock()

    # Port-style structured() method on the top-level mock (used by output guard tests
    # which pass mock_anthropic directly as llm_client)
    m.structured = AsyncMock()

    # LangChain with_structured_output pattern (kept for backward compat)
    m.with_structured_output = MagicMock()
    m.with_structured_output.return_value.ainvoke = AsyncMock()

    # Raw Anthropic SDK-style mock (used by input guard internally)
    m.messages = MagicMock()
    m.messages.create = AsyncMock()

    monkeypatch.setattr("agent.nodes.router.router_llm", m.router_llm, raising=False)
    monkeypatch.setattr("agent.nodes.router.expander_llm", m.expander_llm, raising=False)
    monkeypatch.setattr("agent.nodes.answer.answer_llm", m.answer_llm, raising=False)
    monkeypatch.setattr("agent.nodes.fallback.fallback_llm", m.fallback_llm, raising=False)
    monkeypatch.setattr("agent.guards.output_guard.grounding_llm", m.grounding_llm, raising=False)
    monkeypatch.setattr(
        "agent.guards.input_guard.input_guard_llm", m.input_guard_llm, raising=False
    )

    return m


# ── Mocked Nager.Date holidays API ──────────────────────────────────────────


@pytest.fixture
def mock_holidays_api(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """
    Mocks all httpx GET calls with a fixed list of 2025 US federal holidays.

    Tests can override .return_value to simulate errors:
        mock_holidays_api.side_effect = httpx.TimeoutException("timeout")
    """
    fixed_holidays = [
        {
            "date": "2025-01-01",
            "localName": "New Year's Day",
            "name": "New Year's Day",
            "countryCode": "US",
        },
        {
            "date": "2025-01-20",
            "localName": "Martin Luther King Jr. Day",
            "name": "Martin Luther King Jr. Day",
            "countryCode": "US",
        },
        {
            "date": "2025-02-17",
            "localName": "Presidents' Day",
            "name": "Washington's Birthday",
            "countryCode": "US",
        },
        {
            "date": "2025-05-26",
            "localName": "Memorial Day",
            "name": "Memorial Day",
            "countryCode": "US",
        },
        {
            "date": "2025-06-19",
            "localName": "Juneteenth",
            "name": "Juneteenth National Independence Day",
            "countryCode": "US",
        },
        {
            "date": "2025-07-04",
            "localName": "Independence Day",
            "name": "Independence Day",
            "countryCode": "US",
        },
        {
            "date": "2025-09-01",
            "localName": "Labor Day",
            "name": "Labour Day",
            "countryCode": "US",
        },
        {
            "date": "2025-10-13",
            "localName": "Columbus Day",
            "name": "Columbus Day",
            "countryCode": "US",
        },
        {
            "date": "2025-11-11",
            "localName": "Veterans Day",
            "name": "Veterans Day",
            "countryCode": "US",
        },
        {
            "date": "2025-11-27",
            "localName": "Thanksgiving Day",
            "name": "Thanksgiving Day",
            "countryCode": "US",
        },
        {
            "date": "2025-12-25",
            "localName": "Christmas Day",
            "name": "Christmas Day",
            "countryCode": "US",
        },
    ]

    mock_response = MagicMock()
    mock_response.json.return_value = fixed_holidays
    mock_response.raise_for_status = MagicMock()
    mock_response.status_code = 200

    mock_get = AsyncMock(return_value=mock_response)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.get = mock_get

    import adapters.holidays.nager as _nager_mod

    mock_httpx = MagicMock()
    mock_httpx.AsyncClient = MagicMock(return_value=mock_client)
    monkeypatch.setattr(_nager_mod, "httpx", mock_httpx)
    return mock_get


# ── FastAPI test client ───────────────────────────────────────────────────────


@pytest.fixture
async def async_http_client() -> Any:
    """httpx.AsyncClient pointed at the test FastAPI application."""
    from api.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


# ── Sample chunks for output guard grounding tests ───────────────────────────


@pytest.fixture
def sample_chunks() -> list[dict[str, Any]]:
    """
    Three representative RetrievedChunk-like dicts for use in output guard
    and retriever unit tests. Covers the three primary intent categories.
    """
    return [
        {
            "id": "chunk-001",
            "text": (
                "To reset your password, click 'Forgot Password' on the login page. "
                "We'll send a secure reset link to your verified email address."
            ),
            "metadata": {
                "doc_name": "Login — Security items",
                "page": 4,
                "section": "Password Reset",
                "tags": "password,reset",
            },
            "score": 0.91,
        },
        {
            "id": "chunk-002",
            "text": (
                "Your account is locked after 5 consecutive failed login attempts. "
                "Request a password reset via your verified email to unlock it."
            ),
            "metadata": {
                "doc_name": "Login — Security items",
                "page": 7,
                "section": "Account Lockout",
                "tags": "lockout,password,unlock",
            },
            "score": 0.85,
        },
        {
            "id": "chunk-003",
            "text": (
                "Verification codes are sent when Blossom detects an unrecognized device. "
                "This is part of multi-factor authentication to protect your account."
            ),
            "metadata": {
                "doc_name": "Personal Banking Training",
                "page": 12,
                "section": "MFA Verification",
                "tags": "mfa,verification,remember_me",
            },
            "score": 0.78,
        },
    ]
