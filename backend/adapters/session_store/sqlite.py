from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import aiosqlite


@dataclass
class Message:
    role: str
    content: str
    citations: list[Any] = field(default_factory=list)
    tool_calls: list[Any] = field(default_factory=list)
    timing_ms: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    session_id: str
    user_type: str


_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    user_type   TEXT NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(session_id),
    role        TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    citations   TEXT,
    tool_calls  TEXT,
    timing_ms   TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, created_at DESC);
"""


class SQLiteSessionStore:
    """Secondary adapter: SQLite-backed session store behind SessionStorePort."""

    def __init__(self, db_path: str = "./data/sessions.db") -> None:
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_DDL)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def get_or_create_session(self, session_id: str, user_type: str) -> Session:
        if self._conn is None:
            raise RuntimeError("Session store not initialized — call initialize() first")
        await self._conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, user_type) VALUES (?, ?)",
            (session_id, user_type),
        )
        await self._conn.commit()
        return Session(session_id=session_id, user_type=user_type)

    async def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        citations: list[Any] | None = None,
        tool_calls: list[Any] | None = None,
        timing_ms: dict[str, Any] | None = None,
    ) -> None:
        if self._conn is None:
            raise RuntimeError("Session store not initialized — call initialize() first")
        await self._conn.execute(
            """
            INSERT INTO messages (session_id, role, content, citations, tool_calls, timing_ms)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                role,
                content,
                json.dumps(citations or []),
                json.dumps(tool_calls or []),
                json.dumps(timing_ms or {}),
            ),
        )
        await self._conn.commit()

    async def get_history(self, session_id: str, last_n: int = 6) -> list[Message]:
        if self._conn is None:
            raise RuntimeError("Session store not initialized — call initialize() first")
        async with self._conn.execute(
            """
            SELECT role, content, citations, tool_calls, timing_ms
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (session_id, last_n),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            Message(
                role=row["role"],
                content=row["content"],
                citations=json.loads(row["citations"] or "[]"),
                tool_calls=json.loads(row["tool_calls"] or "[]"),
                timing_ms=json.loads(row["timing_ms"] or "{}"),
            )
            for row in reversed(rows)
        ]

    async def delete_session(self, session_id: str) -> None:
        if self._conn is None:
            raise RuntimeError("Session store not initialized — call initialize() first")
        await self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        await self._conn.commit()
