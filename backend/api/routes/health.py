from __future__ import annotations

from fastapi import APIRouter, Request

from api.models import HealthCheck
from config import settings

router = APIRouter()


@router.get("/health", response_model=HealthCheck)
async def health(request: Request) -> HealthCheck:
    checks: dict[str, str] = {}

    try:
        db = getattr(request.app.state, "db", None)
        if db is not None:
            await db.get_or_create_session("_healthcheck", "member")
        checks["db"] = "ok"
    except Exception:
        checks["db"] = "error"

    try:
        graph = getattr(request.app.state, "graph", None)
        checks["chroma"] = "ok" if graph is not None else "not_initialized"
    except Exception:
        checks["chroma"] = "error"

    checks["anthropic_api_key"] = "ok" if settings.anthropic_api_key else "missing"

    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return HealthCheck(status=overall, checks=checks)
