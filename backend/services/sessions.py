from __future__ import annotations

# Backward-compatibility shim — implementation moved to adapters/session_store/sqlite.py
from adapters.session_store.sqlite import Message, Session
from adapters.session_store.sqlite import SQLiteSessionStore as SessionStore

__all__ = ["Message", "Session", "SessionStore"]
