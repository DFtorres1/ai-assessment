from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class ChatMessage:
    role: str  # "system" | "human" | "ai"
    content: str


@runtime_checkable
class LLMPort(Protocol):
    async def chat(self, messages: list[ChatMessage]) -> str: ...
    async def structured(self, messages: list[ChatMessage], schema: type[Any]) -> Any: ...
