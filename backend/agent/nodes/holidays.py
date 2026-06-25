from __future__ import annotations

import datetime
import time
from typing import Any

from agent.state import AgentState

# HolidaysPort instance — set by create_graph, monkeypatched in tests
_tool: Any = None

_NAMED_DATES: dict[str, tuple[int, int]] = {
    "christmas": (12, 25),
    "new year": (1, 1),
    "thanksgiving": (11, 27),
    "july 4": (7, 4),
    "independence day": (7, 4),
    "labor day": (9, 1),
    "memorial day": (5, 26),
    "juneteenth": (6, 19),
    "veterans day": (11, 11),
    "mlk": (1, 20),
    "martin luther king": (1, 20),
    "presidents day": (2, 17),
    "columbus day": (10, 13),
}


def _parse_date_ref(message: str) -> datetime.date:
    import re

    today = datetime.date.today()
    msg_lower = message.lower()

    for keyword, (month, day) in _NAMED_DATES.items():
        if keyword in msg_lower:
            return datetime.date(today.year, month, day)

    match = re.search(r"\d{4}-\d{2}-\d{2}", message)
    if match:
        return datetime.date.fromisoformat(match.group())

    return today


async def holidays_tool_node(state: AgentState) -> AgentState:
    t0 = time.monotonic()
    message = state.get("message", "")
    ref_date = _parse_date_ref(message)

    await _tool.get_holidays(ref_date.year, "US")
    is_biz = await _tool.is_business_day(ref_date.isoformat())
    next_biz = await _tool.next_business_day(ref_date.isoformat())

    if is_biz.is_business_day:
        day_desc = f"{ref_date.isoformat()} is a regular business day."
    else:
        day_desc = f"{ref_date.isoformat()} is not a business day ({is_biz.reason})."

    holiday_context = f"{day_desc} " f"Next business day: {next_biz.next_business_day}."

    duration = time.monotonic() - t0
    tool_calls = list(state.get("tool_calls", []))
    tool_calls.append(
        {
            "tool": "holidays_api",
            "date": ref_date.isoformat(),
            "year": ref_date.year,
            "duration_ms": round(duration * 1000),
        }
    )

    timing = {**state.get("timing", {}), "holidays_tool": duration}
    return {
        **state,
        "holiday_context": holiday_context,
        "tool_calls": tool_calls,
        "timing": timing,
    }
