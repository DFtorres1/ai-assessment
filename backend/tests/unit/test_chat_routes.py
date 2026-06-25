from __future__ import annotations


def test_to_api_citations_handles_object_citations():
    """_to_api_citations falls back to getattr when citation is not a dict."""
    from api.models import Citation
    from api.routes.chat import _to_api_citations

    class CitationObj:
        doc_name = "Login — Security items"
        page = 4
        section = "Password Reset"

    result = _to_api_citations([CitationObj()])
    assert result == [Citation(doc_name="Login — Security items", page=4, section="Password Reset")]


def test_to_api_citations_handles_dict_citations():
    """_to_api_citations accepts plain dicts directly."""
    from api.models import Citation
    from api.routes.chat import _to_api_citations

    raw = [{"doc_name": "Test Doc", "page": 2, "section": "Intro"}]
    result = _to_api_citations(raw)
    assert result == [Citation(doc_name="Test Doc", page=2, section="Intro")]


def test_to_api_tool_calls_handles_dict():
    """_to_api_tool_calls normalises dict tool-call entries."""
    from api.routes.chat import _to_api_tool_calls

    raw = [{"tool": "holidays_api", "year": 2025, "duration_ms": 120}]
    result = _to_api_tool_calls(raw)
    assert result[0].tool == "holidays_api"
    assert result[0].result is None
    assert result[0].duration_ms == 0.0
