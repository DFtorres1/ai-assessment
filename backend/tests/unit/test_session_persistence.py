from __future__ import annotations

# ── Spec tests (from BACKEND_SPEC.md §9) ─────────────────────────────────────


async def test_session_created_on_first_message(test_db):
    session = await test_db.get_or_create_session("new-session", "member")
    assert session.session_id == "new-session"


async def test_message_appended_and_retrieved(test_db):
    await test_db.get_or_create_session("s1", "member")
    await test_db.append_message("s1", "user", "Hello", citations=[], tool_calls=[], timing_ms={})
    history = await test_db.get_history("s1", last_n=6)
    assert len(history) == 1
    assert history[0].content == "Hello"


async def test_history_limited_to_last_n(test_db):
    await test_db.get_or_create_session("s2", "member")
    for i in range(10):
        await test_db.append_message("s2", "user", f"msg {i}", [], [], {})
    history = await test_db.get_history("s2", last_n=6)
    assert len(history) == 6


async def test_session_idempotent_create(test_db):
    s1 = await test_db.get_or_create_session("dup", "member")
    s2 = await test_db.get_or_create_session("dup", "member")
    assert s1.session_id == s2.session_id


# ── Additional coverage ───────────────────────────────────────────────────────


async def test_history_is_chronological(test_db):
    """get_history returns messages oldest-first so router sees prior turn last."""
    await test_db.get_or_create_session("chrono", "member")
    for msg in ["first", "second", "third"]:
        await test_db.append_message("chrono", "user", msg, [], [], {})
    history = await test_db.get_history("chrono", last_n=6)
    assert [m.content for m in history] == ["first", "second", "third"]


async def test_delete_session_removes_all_data(test_db):
    await test_db.get_or_create_session("to-delete", "staff")
    await test_db.append_message("to-delete", "user", "bye", [], [], {})
    await test_db.delete_session("to-delete")
    history = await test_db.get_history("to-delete", last_n=6)
    assert history == []
