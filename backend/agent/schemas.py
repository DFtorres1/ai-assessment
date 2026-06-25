from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class InputGuardResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    in_scope: bool
    pii_detected: bool
    pii_type: (
        Literal[
            "ssn",
            "card_number",
            "account_number",
            "routing_number",
            "password",
            "pin",
            "government_id",
            "dob_with_name",
        ]
        | None
    ) = None
    is_jailbreak: bool
    jailbreak_category: (
        Literal[
            "instruction_override",
            "identity_manipulation",
            "fiction_framing",
            "prompt_injection",
            "authority_impersonation",
            "capability_probing",
        ]
        | None
    ) = None
    # `blocked` is the canonical name; `block` is accepted from raw LLM JSON
    blocked: bool = Field(alias="block")
    block_reason: Literal["OUT_OF_SCOPE", "PII_DETECTED", "JAILBREAK_DETECTED"] | None = None
    user_message: str | None = None


class RouterResult(BaseModel):
    intent: Literal[
        "account_lockout",
        "password_reset",
        "mfa_issue",
        "remember_device",
        "username_recovery",
        "account_setup",
        "phone_banking",
        "holiday_timing",
        "out_of_scope",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class QueryExpansionResult(BaseModel):
    queries: list[str] = Field(min_length=3, max_length=3)


class AnswerCitation(BaseModel):
    doc_name: str
    page: int
    section: str
    supporting_quote: str


class AnswerResult(BaseModel):
    sub_questions: list[str] = Field(default_factory=list)
    thought: str
    evidence_quotes: list[str]
    gaps: str | None = None
    answer: str
    citations: list[AnswerCitation]
    step_by_step_offered: bool
    confidence: float = Field(ge=0.0, le=1.0)


class FallbackResult(BaseModel):
    clarifying_question: str
    suggested_intent: (
        Literal[
            "account_lockout",
            "password_reset",
            "mfa_issue",
            "remember_device",
            "username_recovery",
            "account_setup",
            "phone_banking",
            "holiday_timing",
        ]
        | None
    ) = None


class GroundingResult(BaseModel):
    all_claims_grounded: bool
    issues_found: list[str]
    revised_answer: str
    pii_present: bool
    pii_description: str | None = None
