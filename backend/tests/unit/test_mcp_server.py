from __future__ import annotations


async def test_mcp_exposes_get_federal_holidays():
    from mcp_server.holidays_mcp import mcp

    tool_names = [t.name for t in await mcp.list_tools()]
    assert "get_federal_holidays" in tool_names
    assert "is_business_day" in tool_names
    assert "next_business_day" in tool_names


async def test_mcp_get_federal_holidays_returns_list(mock_holidays_api):
    from mcp_server.holidays_mcp import get_federal_holidays

    result = await get_federal_holidays(year=2025, country_code="US")
    assert isinstance(result, list)
    assert len(result) > 0


async def test_mcp_is_business_day_christmas(mock_holidays_api):
    from mcp_server.holidays_mcp import is_business_day

    result = await is_business_day(date_str="2025-12-25")
    assert result["is_business_day"] is False


async def test_mcp_next_business_day_after_christmas(mock_holidays_api):
    from mcp_server.holidays_mcp import next_business_day

    result = await next_business_day(from_date="2025-12-25")
    assert "next_business_day" in result
    assert result["next_business_day"] > "2025-12-25"
