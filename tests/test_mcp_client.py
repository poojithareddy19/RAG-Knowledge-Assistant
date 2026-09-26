"""The chat's side of the Model Context Protocol, over a real session.

These go through the protocol rather than calling the tool functions: a client
session is opened against the server, the initialize handshake runs, a tool is
called by name and its JSON comes back. None of them needs a database.
"""

import pytest

import src.mcp.server as mcp_server
from src.mcp.client import ToolError, call_tool


def test_a_tool_is_called_over_the_protocol_and_returns_its_json():
    result = call_tool("describe_schema")

    assert "catalog" in result
    assert "measurements" in result["catalog"]


def test_a_tool_failure_arrives_as_an_error_not_as_a_result():
    """The server wraps a failure in its error envelope; the client must turn
    that back into an exception rather than hand the envelope on as data."""
    with pytest.raises(ToolError, match="float_id must be a number"):
        call_tool("get_profile", {"float_id": "not-a-number"})


def test_nearest_floats_rejects_an_impossible_position_before_the_database():
    with pytest.raises(ValueError, match="not a position"):
        mcp_server.nearest_floats(latitude=95.0, longitude=65.0)


def test_nearest_floats_caps_the_number_of_rows(monkeypatch):
    seen = {}

    def fake_fetch(sql, params, readonly):
        seen.update(params)
        seen["readonly"] = readonly
        return ["float_id", "distance_km"], []

    monkeypatch.setattr(mcp_server, "fetch_all", fake_fetch)

    mcp_server.nearest_floats(latitude=10.0, longitude=65.0, limit=10_000)

    assert seen["limit"] == mcp_server.MAX_NEAREST
    assert seen["readonly"] is True


def test_nearest_floats_is_parameterised_not_formatted():
    """The position is bound as a parameter, never pasted into the SQL."""
    assert "%(latitude)s" in mcp_server.NEAREST_FLOATS_SQL
    assert "%(longitude)s" in mcp_server.NEAREST_FLOATS_SQL


def test_the_pipeline_answers_a_tool_question_from_the_tools_rows(monkeypatch):
    import src.utils.pipeline as pipeline

    monkeypatch.setattr(
        pipeline,
        "call_mcp_tool",
        lambda name, arguments: {
            "columns": ["float_id", "distance_km", "latitude", "longitude", "obs_time", "region"],
            "rows": [
                [1901393, 42.3, 10.2, 65.4, "2021-11-18", "Arabian Sea"],
                [1900083, 88.0, 11.0, 64.7, "2019-03-02", "Arabian Sea"],
            ],
        },
    )

    service = object.__new__(pipeline.RAGService)
    result = service.answer_from_tool("nearest_floats", {"latitude": 10.5, "longitude": 65.2})

    assert result["mcp_tool"] == "nearest_floats"
    assert result["generated_sql"] is None
    assert result["row_count"] == 2
    # The sentence is read off the first row, so it cannot claim anything the
    # table does not show.
    assert "1901393" in result["answer"]
    assert "42.3 km" in result["answer"]
    # Positions came back, so it is drawn as a map.
    assert result["chart_kind"] == "trajectory"


def test_an_unreachable_tool_falls_back_rather_than_losing_the_question(monkeypatch):
    import src.utils.pipeline as pipeline

    def down(name, arguments):
        raise ConnectionError("server unavailable")

    monkeypatch.setattr(pipeline, "call_mcp_tool", down)

    service = object.__new__(pipeline.RAGService)

    assert service.answer_from_tool("nearest_floats", {"latitude": 1.0, "longitude": 2.0}) is None


def test_a_missing_location_is_a_refusal_that_says_what_to_do():
    import src.utils.pipeline as pipeline

    service = object.__new__(pipeline.RAGService)
    result = service.answer_from_tool(pipeline.NEEDS_LOCATION, {})

    assert result["refused"] is True
    assert "10.5N 65.2E" in result["answer"]


def test_the_server_describes_every_tool_it_dispatches():
    """A handler without a listed tool is callable but invisible to clients."""
    listed = {tool.name for tool in mcp_server.TOOLS}

    assert set(mcp_server.HANDLERS) == listed
