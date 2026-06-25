from __future__ import annotations

import pytest

from services.retry import llm_retry


async def test_llm_retry_returns_result_on_success():
    call_count = 0

    @llm_retry
    async def succeeds():
        nonlocal call_count
        call_count += 1
        return "ok"

    result = await succeeds()
    assert result == "ok"
    assert call_count == 1


async def test_llm_retry_reraises_non_retryable_error():
    @llm_retry
    async def raises():
        raise ValueError("not retryable")

    with pytest.raises(ValueError, match="not retryable"):
        await raises()
