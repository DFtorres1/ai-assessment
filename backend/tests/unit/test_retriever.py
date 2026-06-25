from __future__ import annotations

import pytest

from knowledge.retriever import Retriever


@pytest.fixture
async def retriever(chroma_test_client, embedding_function):
    return Retriever(chroma_client=chroma_test_client, embed_fn=embedding_function)


async def test_returns_chunks_for_password_reset(retriever):
    chunks = await retriever.search(query="how do I reset my password?", tags=["password", "reset"])
    assert len(chunks) > 0
    assert any("password" in c.text.lower() for c in chunks)


async def test_chunks_have_required_metadata(retriever):
    chunks = await retriever.search(query="account lockout", tags=["lockout"])
    for chunk in chunks:
        assert chunk.doc_name
        assert chunk.page > 0
        assert chunk.section
        assert 0 <= chunk.score <= 1


async def test_returns_top_5_max(retriever):
    chunks = await retriever.search(query="login security", tags=[])
    assert len(chunks) <= 5


async def test_expanded_queries_improve_recall(retriever):
    single = await retriever.search(query="code not working", tags=["mfa"])
    expanded = await retriever.search_multi(
        queries=["code not working", "MFA verification code issue", "authenticator app problem"],
        tags=["mfa"],
    )
    assert len(expanded) >= len(single)


async def test_sample_prompt_1_has_hits(retriever):
    chunks = await retriever.search(
        "I got locked out after entering the wrong password", tags=["lockout"]
    )
    assert len(chunks) > 0
    assert chunks[0].score >= 0.5


async def test_sample_prompt_2_has_hits(retriever):
    chunks = await retriever.search("What are the password rules?", tags=["password"])
    assert len(chunks) > 0


async def test_sample_prompt_5_has_hits(retriever):
    chunks = await retriever.search("I forgot my username how do I recover it?", tags=["username"])
    assert len(chunks) > 0


async def test_low_score_below_threshold_triggers_fallback(retriever):
    chunks = await retriever.search("xyzzy nonsense query blorp", tags=[])
    assert all(c.score < 0.5 for c in chunks) or len(chunks) == 0
