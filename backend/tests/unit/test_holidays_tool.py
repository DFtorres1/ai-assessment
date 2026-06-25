from __future__ import annotations

import pytest

from adapters.holidays.nager import NagerHolidaysAdapter


@pytest.fixture
def tool(mock_holidays_api):
    return NagerHolidaysAdapter()


async def test_returns_holidays_for_current_year(tool):
    holidays = await tool.get_holidays(year=2025, country="US")
    assert len(holidays) > 0
    assert any(h["name"] == "Christmas Day" for h in holidays)


async def test_identifies_christmas_as_holiday(tool):
    result = await tool.is_business_day("2025-12-25")
    assert result.is_business_day is False
    assert result.reason == "Federal Holiday: Christmas Day"


async def test_identifies_regular_weekday_as_business_day(tool):
    result = await tool.is_business_day("2025-06-23")  # Monday, no holiday
    assert result.is_business_day is True


async def test_next_business_day_skips_weekend(tool):
    result = await tool.next_business_day(from_date="2025-12-26")  # Friday before NYE weekend
    assert result.next_business_day in ("2025-12-29", "2025-12-30")


async def test_next_business_day_skips_holiday(tool):
    # 2025-12-25 (Thursday) is Christmas; next business day is Friday 2025-12-26
    result = await tool.next_business_day(from_date="2025-12-24")
    assert result.next_business_day >= "2025-12-25"  # must land after the holiday


async def test_handles_api_timeout_with_backoff(tool, mock_holidays_api):
    mock_holidays_api.side_effect = Exception("API timeout")
    with pytest.raises(Exception):
        await tool.get_holidays(year=2025, country="US")
    # Should have retried 3 times (stop_after_attempt(3))
    assert mock_holidays_api.call_count == 3


async def test_caches_holiday_response(tool):
    await tool.get_holidays(year=2025, country="US")
    await tool.get_holidays(year=2025, country="US")  # second call
    # API should only be called once (cached)
    assert tool._api_call_count == 1


async def test_is_business_day_returns_false_for_weekend(tool):
    """is_business_day returns False with reason='Weekend' for Saturday/Sunday."""
    result = await tool.is_business_day("2025-12-27")  # Saturday
    assert result.is_business_day is False
    assert result.reason == "Weekend"


def test_parse_date_ref_iso_format():
    """_parse_date_ref extracts an ISO date from the message text."""
    import datetime

    from agent.nodes.holidays import _parse_date_ref

    result = _parse_date_ref("My date is 2025-03-15 and I need help")
    assert result == datetime.date(2025, 3, 15)


async def test_holidays_tool_node_non_business_day_context(mock_holidays_api, monkeypatch):
    """holidays_tool_node builds the 'not a business day' description for weekends."""
    import agent.nodes.holidays as holidays_mod
    from agent.nodes.holidays import holidays_tool_node
    from agent.state import AgentState

    # Inject a fresh adapter with the mock already in place
    monkeypatch.setattr(holidays_mod, "_tool", NagerHolidaysAdapter())

    # Saturday 2025-12-27 — weekend → is_business_day=False reason="Weekend"
    state = AgentState(
        session_id="test-hol-weekend",
        user_type="member",
        message="What happens on 2025-12-27?",
        temperature=0.2,
        top_p=0.9,
    )
    result = await holidays_tool_node(state)
    assert "not a business day" in result["holiday_context"]
    assert "Weekend" in result["holiday_context"]
