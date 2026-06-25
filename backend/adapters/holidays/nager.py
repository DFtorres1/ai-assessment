from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_none


@dataclass
class BusinessDayResult:
    is_business_day: bool
    reason: str | None = None


@dataclass
class NextBusinessDayResult:
    next_business_day: str


class NagerHolidaysAdapter:
    """Secondary adapter: Nager.Date public API behind HolidaysPort."""

    _BASE_URL = "https://date.nager.at/api/v3/PublicHolidays/{year}/{country}"

    def __init__(self) -> None:
        self._cache: dict[tuple[int, str], list[dict[str, Any]]] = {}
        self._api_call_count = 0

    async def get_holidays(self, year: int, country: str = "US") -> list[dict[str, Any]]:
        key = (year, country)
        if key in self._cache:
            return self._cache[key]
        result = await self._fetch(year, country)
        self._cache[key] = result
        return result

    @retry(stop=stop_after_attempt(3), wait=wait_none(), reraise=True)
    async def _fetch(self, year: int, country: str) -> list[dict[str, Any]]:
        self._api_call_count += 1
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(self._BASE_URL.format(year=year, country=country))
            response.raise_for_status()
            result: list[dict[str, Any]] = response.json()
            return result

    async def is_business_day(self, date_str: str) -> BusinessDayResult:
        date = datetime.date.fromisoformat(date_str)
        if date.weekday() >= 5:
            return BusinessDayResult(is_business_day=False, reason="Weekend")
        holidays = await self.get_holidays(date.year, "US")
        for h in holidays:
            if h["date"] == date_str:
                return BusinessDayResult(
                    is_business_day=False,
                    reason=f"Federal Holiday: {h['name']}",
                )
        return BusinessDayResult(is_business_day=True)

    async def next_business_day(self, from_date: str) -> NextBusinessDayResult:
        date = datetime.date.fromisoformat(from_date)
        holidays = await self.get_holidays(date.year, "US")
        holiday_dates = {h["date"] for h in holidays}
        next_year_holidays = await self.get_holidays(date.year + 1, "US")
        holiday_dates |= {h["date"] for h in next_year_holidays}
        candidate = date + datetime.timedelta(days=1)
        while True:
            s = candidate.isoformat()
            if candidate.weekday() < 5 and s not in holiday_dates:
                return NextBusinessDayResult(next_business_day=s)
            candidate += datetime.timedelta(days=1)
