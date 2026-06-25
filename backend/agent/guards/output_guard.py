from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent.schemas import GroundingResult
from agent.state import Citation
from core.ports.llm import ChatMessage

# Module-level LLM client — monkeypatched in tests, replaced in production
grounding_llm: Any = None


@dataclass
class OutputGuardResult:
    passed: bool
    revised_answer: str
    issues_found: list[str]
    valid_citations: list[Citation] = field(default_factory=list)


class OutputGuard:
    def __init__(self, llm_client: Any = None) -> None:
        self.llm_client = llm_client or grounding_llm

    async def check(
        self,
        answer: str,
        chunks: list[Any],
        citations: list[Citation],
    ) -> OutputGuardResult:
        chunk_context = "\n\n".join(f"[{c.doc_name}, p.{c.page}] {c.text}" for c in chunks)
        prompt = (
            f"Retrieved knowledge:\n{chunk_context}\n\n"
            f"Assistant answer:\n{answer}\n\n"
            "Check: (1) Is every factual claim supported by the retrieved knowledge? "
            "(2) Does the answer contain any PII (SSN, account numbers, card numbers)? "
            "If PII is found, rewrite the answer to remove it entirely "
            "— do NOT mask with *** or [REDACTED]. "
            "Return a GroundingResult."
        )
        grounding = await self.llm_client.structured(
            [ChatMessage(role="human", content=prompt)], GroundingResult
        )

        passed = grounding.all_claims_grounded and not grounding.pii_present
        valid_citations = [c for c in citations if self._is_valid(c, chunks)]

        return OutputGuardResult(
            passed=passed,
            revised_answer=grounding.revised_answer,
            issues_found=list(grounding.issues_found),
            valid_citations=valid_citations,
        )

    def _is_valid(self, citation: Citation, chunks: list[Any]) -> bool:
        return any(c.doc_name == citation.doc_name and c.page == citation.page for c in chunks)
