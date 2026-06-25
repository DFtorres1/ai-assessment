from __future__ import annotations

import re

from knowledge.chunker import Chunk

# Keyword → tags mapping (tag = what lands in the `tags` metadata field)
_TAG_RULES: list[tuple[list[str], str]] = [
    (["password", "passphrase", "forgot password", "reset password"], "password"),
    (["reset link", "reset your password", "password reset"], "reset"),
    (
        [
            "locked out",
            "lock out",
            "account locked",
            "account lock",
            "lockout",
            "lock-out",
            "failed login",
        ],
        "lockout",
    ),
    (["unlock", "unlocked"], "unlock"),
    (["mfa", "multi-factor", "multifactor", "two-factor", "2fa", "authenticator"], "mfa"),
    (["verification code", "verify", "verification"], "verification"),
    (["2fa", "two factor", "two-factor"], "2fa"),
    (
        ["remember this device", "remember device", "trusted device", "remembered device"],
        "remember_me",
    ),
    (["trusted device", "trusted_device"], "trusted_device"),
    (["username", "user name", "forgot username"], "username"),
    (["sign up", "signup", "register", "create account", "new account"], "signup"),
    (["setup", "set up", "set-up", "onboarding"], "setup"),
    (["enrollment", "enroll", "enrol"], "enrollment"),
    (["phone banking", "phone-banking", "phonebanking", "ivr"], "phone_banking"),
    (["ivr", "interactive voice"], "ivr"),
]


def tag_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Assign comma-separated tags to each chunk based on text keyword matching."""
    for chunk in chunks:
        text_lower = chunk.text.lower()
        tags: list[str] = []
        seen: set[str] = set()
        for keywords, tag in _TAG_RULES:
            if tag in seen:
                continue
            if any(re.search(r"\b" + re.escape(kw) + r"\b", text_lower) for kw in keywords):
                tags.append(tag)
                seen.add(tag)
        chunk.metadata["tags"] = ",".join(tags)
    return chunks
