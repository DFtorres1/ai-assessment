from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.ports.llm import ChatMessage

_ROLE_MAP = {
    "system": SystemMessage,
    "human": HumanMessage,
    "ai": AIMessage,
}


def _to_lc(messages: list[ChatMessage]) -> list:
    return [_ROLE_MAP[m.role](content=m.content) for m in messages]


class AnthropicAdapter:
    """Secondary adapter: wraps ChatAnthropic behind LLMPort."""

    def __init__(self, model: str, temperature: float = 0.2, max_tokens: int = 1024) -> None:
        self._llm = ChatAnthropic(model=model, temperature=temperature, max_tokens=max_tokens)

    async def chat(self, messages: list[ChatMessage]) -> str:
        result = await self._llm.ainvoke(_to_lc(messages))
        return str(result.content)

    async def structured(self, messages: list[ChatMessage], schema: type[Any]) -> Any:
        return await self._llm.with_structured_output(schema).ainvoke(_to_lc(messages))
