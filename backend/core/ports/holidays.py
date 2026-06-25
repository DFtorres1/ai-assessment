from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class HolidaysPort(Protocol):
    async def get_holidays(self, year: int, country: str = "US") -> list[dict]: ...
    async def is_business_day(self, date_str: str) -> Any: ...
    async def next_business_day(self, from_date: str) -> Any: ...
