"""
scripts/eval.py — Blossom Banking Helper evaluation harness

Runs all 10 assessment prompts against the live /chat endpoint and reports:
  - Latency per prompt (ms)
  - Citations returned (doc + page + section) as retrieval-hit proxy
  - Tool calls fired
  - P95 latency across all prompts
  - SLA breaches (> 5 000 ms)

Usage:
    python scripts/eval.py                          # default: http://localhost:8000
    python scripts/eval.py --url http://host:8000
    python scripts/eval.py --url http://host:8000 --timeout 30
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

# ── Assessment prompts ────────────────────────────────────────────────────────

EVAL_PROMPTS: list[tuple[str, str]] = [
    ("member", "I got locked out after entering the wrong password. Can I unlock myself?"),
    ("member", "What are the password rules? Can you list them quickly?"),
    ("member", "Why do I keep getting verification codes when I log in?"),
    ("member", "How often does 'remember this device' expire?"),
    ("member", "I forgot my username — how do I recover it?"),
    ("member", "I changed phones and now my codes don't work. What should I do?"),
    ("member", "Please help me reset my password safely."),
    ("staff", "Can I unlock a phone-banking user without calling support?"),
    ("member", "I signed up, but I'm stuck — where do I finish my setup?"),
    (
        "member",
        "If I start a password reset on a federal holiday, when should I expect the next step?",
    ),
]

SLA_MS = 5_000  # p95 target per assessment spec

# ── Result model ──────────────────────────────────────────────────────────────


@dataclass
class PromptResult:
    index: int
    user_type: str
    message: str
    latency_ms: float
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    status_code: int = 200
    error: str | None = None

    @property
    def sla_breach(self) -> bool:
        return self.latency_ms > SLA_MS

    @property
    def has_citations(self) -> bool:
        return len(self.citations) > 0


# ── Runner ────────────────────────────────────────────────────────────────────


async def run_prompt(
    client: httpx.AsyncClient,
    index: int,
    user_type: str,
    message: str,
    session_prefix: str,
) -> PromptResult:
    session_id = f"{session_prefix}-{index}"
    payload = {
        "session_id": session_id,
        "message": message,
        "user_type": user_type,
    }

    t0 = time.perf_counter()
    try:
        resp = await client.post("/chat", json=payload)
        latency_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code != 200:
            return PromptResult(
                index=index,
                user_type=user_type,
                message=message,
                latency_ms=latency_ms,
                answer="",
                status_code=resp.status_code,
                error=resp.text[:200],
            )
        body = resp.json()
        return PromptResult(
            index=index,
            user_type=user_type,
            message=message,
            latency_ms=latency_ms,
            answer=body.get("answer", ""),
            citations=body.get("citations", []),
            tool_calls=body.get("tool_calls", []),
            status_code=resp.status_code,
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000
        return PromptResult(
            index=index,
            user_type=user_type,
            message=message,
            latency_ms=latency_ms,
            answer="",
            error=str(exc),
        )


async def run_eval(base_url: str, timeout: float, session_prefix: str) -> list[PromptResult]:
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        # Verify the server is up before running prompts
        try:
            health = await client.get("/health")
            health.raise_for_status()
        except Exception as exc:
            print(f"ERROR  Server at {base_url} is not reachable: {exc}", file=sys.stderr)
            sys.exit(1)

        results: list[PromptResult] = []
        for i, (user_type, message) in enumerate(EVAL_PROMPTS, start=1):
            print(f"  [{i:02d}/10] Running... ", end="", flush=True)
            result = await run_prompt(client, i, user_type, message, session_prefix)
            results.append(result)
            status = "BREACH" if result.sla_breach else ("ERROR" if result.error else "OK")
            print(f"{status}  ({result.latency_ms:.0f} ms)")

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────


def _truncate(text: str, n: int = 90) -> str:
    return text[:n] + "…" if len(text) > n else text


def print_report(results: list[PromptResult], base_url: str) -> None:
    divider = "─" * 110
    thick = "═" * 110

    print()
    print(thick)
    print("  BLOSSOM BANKING HELPER — EVAL REPORT")
    print(f"  Endpoint : {base_url}")
    print(f"  Prompts  : {len(results)}")
    print(thick)

    for r in results:
        breach_flag = "  ⚠ SLA BREACH" if r.sla_breach else ""
        error_flag = f"  ✖ ERROR: {r.error}" if r.error else ""
        tools_fired = ", ".join(tc["tool"] for tc in r.tool_calls) if r.tool_calls else "none"
        user_badge = f"[{r.user_type}]"

        print()
        print(f"  Prompt {r.index:02d}  {user_badge}")
        print(f"  Q: {_truncate(r.message)}")
        print(f"  A: {_truncate(r.answer) if r.answer else '(no answer)'}")
        print(f"  Latency   : {r.latency_ms:,.0f} ms{breach_flag}{error_flag}")
        print(f"  Tools     : {tools_fired}")

        if r.citations:
            print(f"  Citations : {len(r.citations)} hit(s)")
            for c in r.citations:
                print(f"             • {c['doc_name']}  p.{c['page']}  §{c['section']}")
        else:
            print("  Citations : none")

        print(divider)

    # ── Summary statistics ────────────────────────────────────────────────────
    latencies = [r.latency_ms for r in results if not r.error]
    breaches = [r for r in results if r.sla_breach]
    errors = [r for r in results if r.error]
    with_cites = [r for r in results if r.has_citations]
    with_tools = [r for r in results if r.tool_calls]

    print()
    print(thick)
    print("  SUMMARY")
    print(thick)

    if latencies:
        sorted_lats = sorted(latencies)
        p50 = statistics.median(latencies)
        p95_idx = max(0, int(len(sorted_lats) * 0.95) - 1)
        p95 = sorted_lats[p95_idx]
        p99_idx = max(0, int(len(sorted_lats) * 0.99) - 1)
        p99 = sorted_lats[p99_idx]

        print(f"  Latency   p50 = {p50:,.0f} ms  |  p95 = {p95:,.0f} ms  |  p99 = {p99:,.0f} ms")
        print(f"  SLA target   ≤ {SLA_MS:,} ms")
        sla_status = "✓ PASS" if p95 <= SLA_MS else "✗ FAIL"
        print(f"  SLA (p95)    {sla_status}  ({p95:,.0f} ms)")
    else:
        print("  Latency   N/A (all prompts errored)")

    print(
        f"  Prompts   {len(results)} total  |  "
        f"{len(breaches)} breach(es)  |  "
        f"{len(errors)} error(s)  |  "
        f"{len(with_cites)} with citations  |  "
        f"{len(with_tools)} with tool calls"
    )

    if breaches:
        print()
        print("  ⚠  SLA BREACH DETAIL:")
        for r in breaches:
            print(f"     Prompt {r.index:02d}: {r.latency_ms:,.0f} ms — {_truncate(r.message, 60)}")

    if errors:
        print()
        print("  ✖  ERRORS:")
        for r in errors:
            print(f"     Prompt {r.index:02d}: {r.error}")

    print()
    print(thick)
    print()

    # Exit non-zero if any SLA breach or error so CI can catch it
    if breaches or errors:
        sys.exit(1)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the 10 Blossom assessment prompts and report latency + retrieval hits."
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the running API server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "--session-prefix",
        default="eval",
        help="Prefix for session IDs to avoid collisions with real sessions (default: eval)",
    )
    args = parser.parse_args()

    print("\nBlossom Banking Helper — Eval")
    print(f"Target: {args.url}")
    print(f"Running {len(EVAL_PROMPTS)} prompts...\n")

    results = asyncio.run(run_eval(args.url, args.timeout, args.session_prefix))
    print_report(results, args.url)


if __name__ == "__main__":
    main()
