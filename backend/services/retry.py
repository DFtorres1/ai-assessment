from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import anthropic
import httpx
import structlog
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_log = structlog.get_logger()

_F = TypeVar("_F", bound=Callable[..., Any])


def _log_retry_attempt(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    _log.warning(
        "llm_retry",
        attempt=retry_state.attempt_number,
        exception=type(exc).__name__ if exc else None,
        message=str(exc) if exc else None,
    )


def llm_retry(func: _F) -> _F:
    """Exponential backoff decorator for LLM and external API calls."""
    decorated = retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((httpx.TimeoutException, anthropic.RateLimitError)),
        before_sleep=_log_retry_attempt,
        reraise=True,
    )(func)
    return decorated
