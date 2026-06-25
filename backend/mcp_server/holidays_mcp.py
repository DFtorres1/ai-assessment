from __future__ import annotations

import os
from typing import Any, Literal, cast

from mcp.server.fastmcp import FastMCP

from adapters.holidays.nager import NagerHolidaysAdapter

_transport = cast(
    Literal["stdio", "sse", "streamable-http"],
    os.getenv("MCP_TRANSPORT", "stdio"),
)
_port = int(os.getenv("MCP_PORT", "8001"))

mcp = FastMCP(
    "blossom-holidays",
    host="0.0.0.0",  # nosec B104 # noqa: S104 — intentional for containerised deployment
    port=_port,
)
_tool = NagerHolidaysAdapter()


@mcp.tool()
async def get_federal_holidays(year: int, country_code: str = "US") -> list[dict[str, Any]]:
    """Return all US federal holidays for a given year."""
    return await _tool.get_holidays(year=year, country=country_code)


@mcp.tool()
async def is_business_day(date_str: str) -> dict[str, Any]:
    """Check if a date (YYYY-MM-DD) is a US business day (not a holiday or weekend)."""
    result = await _tool.is_business_day(date_str)
    return {"is_business_day": result.is_business_day, "reason": result.reason}


@mcp.tool()
async def next_business_day(from_date: str, days: int = 1) -> dict[str, Any]:
    """Return the next N business day(s) after a given date, skipping holidays and weekends."""
    result = await _tool.next_business_day(from_date)
    return {"next_business_day": result.next_business_day}


if __name__ == "__main__":
    mcp.run(transport=_transport)
