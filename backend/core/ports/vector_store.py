from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VectorStorePort(Protocol):
    async def search(self, query: str, k: int = 5, tags: list[str] | None = None) -> list[Any]: ...

    async def search_multi(
        self, queries: list[str], k: int = 5, tags: list[str] | None = None
    ) -> list[Any]: ...
