"""The MCP tool surface.

The test that matters here is the one asserting what is absent. The four safety
layers constrain generated SQL before it reaches the database, and a tool
taking a statement from an external client would route around all of them, so
the absence of one is pinned down rather than left to a code review.

No database and no network: these check the declared surface and the dispatch,
not the queries behind it.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from src.mcp import server as mcp_server
from src.mcp.server import FORBIDDEN_ARGUMENTS, TOOLS, build_server, call

# Pinned, so a tool can only be added on purpose. nearest_floats joined for the
# problem statement's third example; it takes a position, never a statement.
EXPECTED = {
    "query_argo",
    "list_floats",
    "get_profile",
    "nearest_floats",
    "describe_schema",
}


def _schema(tool):
    return tool.input_schema


def test_exactly_the_registered_tools_exist():
    assert {tool.name for tool in TOOLS} == EXPECTED


def test_every_tool_has_a_description():
    for tool in TOOLS:
        assert tool.description and len(tool.description) > 20


def test_no_tool_accepts_free_text_sql():
    for tool in TOOLS:
        for name in _schema(tool).get("properties", {}):
            assert name.lower() not in FORBIDDEN_ARGUMENTS, (
                f"{tool.name} exposes a {name} argument"
            )


def test_no_tool_is_named_as_a_sql_runner():
    for tool in TOOLS:
        assert "sql" not in tool.name.lower()


def test_no_tool_takes_an_unconstrained_extra_argument():
    """additionalProperties must be closed, or sql arrives under any name."""
    for tool in TOOLS:
        assert _schema(tool).get("additionalProperties") is False


def test_query_argo_takes_a_question():
    tool = next(t for t in TOOLS if t.name == "query_argo")
    schema = _schema(tool)

    assert schema["properties"]["question"]["type"] == "string"
    assert schema["required"] == ["question"]


def test_list_floats_takes_an_optional_region_and_year():
    tool = next(t for t in TOOLS if t.name == "list_floats")
    schema = _schema(tool)

    assert schema["properties"]["region"]["type"] == "string"
    assert schema["properties"]["year"]["type"] == "integer"
    assert not schema.get("required")


def test_get_profile_requires_a_float_and_takes_an_optional_cycle():
    tool = next(t for t in TOOLS if t.name == "get_profile")
    schema = _schema(tool)

    assert schema["required"] == ["float_id"]
    assert schema["properties"]["cycle"]["type"] == "integer"


def test_describe_schema_takes_nothing():
    tool = next(t for t in TOOLS if t.name == "describe_schema")

    assert _schema(tool)["properties"] == {}


def test_the_fixed_lookups_are_parameterised_not_interpolated():
    """Hand written SQL, with placeholders rather than formatting."""
    for sql in (mcp_server.LIST_FLOATS_SQL, mcp_server.GET_PROFILE_SQL):
        assert "%(" in sql
        assert "format(" not in sql.lower()
        # No Python interpolation of any kind: these are constants, and a
        # placeholder here would mean a value reaching the database as text.
        assert "{" not in sql


def test_both_fixed_lookups_are_capped():
    assert "LIMIT %(limit)s" in mcp_server.LIST_FLOATS_SQL
    assert "LIMIT %(limit)s" in mcp_server.GET_PROFILE_SQL


def test_an_unknown_tool_is_rejected():
    with pytest.raises(ValueError, match="unknown tool"):
        call("run_sql", {"sql": "SELECT 1"})


def test_a_non_numeric_float_id_is_rejected_before_the_database(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("must not reach the database")

    monkeypatch.setattr(mcp_server, "fetch_all", explode)

    with pytest.raises(ValueError, match="float_id must be a number"):
        mcp_server.get_profile("1900083; DROP TABLE floats")


def test_describe_schema_returns_the_catalog():
    out = mcp_server.describe_schema()

    assert "measurements" in out["catalog"]


def test_the_server_lists_its_tools_over_the_protocol_handler():
    """The same list the client sees, through the registered handler."""
    result = asyncio.run(mcp_server._on_list_tools(None, None))

    assert {tool.name for tool in result.tools} == EXPECTED


def test_a_failing_tool_returns_an_error_result_not_an_exception(monkeypatch):
    def explode(name, arguments):
        raise RuntimeError("database unreachable")

    monkeypatch.setattr(mcp_server, "call", explode)

    class Params:
        name = "list_floats"
        arguments = {}

    result = asyncio.run(mcp_server._on_call_tool(None, Params()))

    assert result.is_error
    assert "database unreachable" in json.loads(result.content[0].text)["error"]


def test_the_server_builds():
    server = build_server()

    assert server.name == "argo"
