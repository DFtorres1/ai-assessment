from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SessionStorePort(Protocol):
    async def initialize(self) -> None: ...
    async def close(self) -> None: ...
    async def get_or_create_session(self, session_id: str, user_type: str) -> Any: ...
    async def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: list[Any] | None = None,
        tool_calls: list[Any] | None = None,
        timing_ms: dict[str, Any] | None = None,
    ) -> None: ...
    async def get_history(self, session_id: str, last_n: int = 6) -> list[Any]: ...
    async def delete_session(self, session_id: str) -> None: ...
