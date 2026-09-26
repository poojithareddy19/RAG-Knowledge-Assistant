"""Expose the assistant over the Model Context Protocol, on stdio.

Five tools, each a thin wrapper over something that already exists: the routed
service for natural language, hand written parameterised queries for the three
lookups whose shape never varies, and the column catalog as text.

The chat on the Streamlit page is a client of this server as well as external
ones are: ``src/mcp/client.py`` calls these tools over the protocol for the
questions that have one fixed shape, such as the floats nearest a point.

**There is deliberately no tool that takes SQL.** The four safety layers, the
validator, the read only role, the session settings and the capped LIMIT, all
exist so that generated SQL is constrained before it reaches the database. A
``run_sql`` tool would hand an external client a way around every one of them,
and an MCP client is exactly the caller you cannot vouch for. That absence is
the design, not an omission to be filled in later.

The two lookups do not go through the model either. Their shape is fixed, so
generating SQL for them would spend a model call and add a failure mode to
answer a question that a parameterised query answers exactly.

Registering it with a client:

    {
      "mcpServers": {
        "argo": {
          "command": "python",
          "args": ["-m", "src.mcp.server"],
          "cwd": "/path/to/RAG_Assistant"
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import json

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from src.sqlgen.schema_context import load_catalog
from src.utils.db import fetch_all

SERVER_NAME = "argo"

# Names that would mean "hand me a statement". Asserted against in the tests,
# so a later tool cannot quietly reintroduce the hole this server closes.
FORBIDDEN_ARGUMENTS = (
    "sql",
    "statement",
    "raw_sql",
    "query_sql",
    "expression",
)

# A cap on every row-returning tool. An MCP client pulls the whole result into
# a model's context, so an uncapped query is a context window overrun rather
# than a slow response.
MAX_FLOATS = 200
MAX_LEVELS = 5000
MAX_NEAREST = 50


TOOLS = [
    types.Tool(
        name="query_argo",
        description=(
            "Ask a natural language question about the ARGO measurements. "
            "Returns the answer, the SQL that was generated, and the rows it "
            "produced. The SQL is validated and executed as a read only user."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "A question in plain English about the data.",
                }
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    ),
    types.Tool(
        name="list_floats",
        description=(
            "List floats with their profile counts and date ranges, "
            "optionally restricted to one region or one year."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": (
                        "Arabian Sea, Bay of Bengal or Southern Indian Ocean."
                    ),
                },
                "year": {
                    "type": "integer",
                    "description": "Calendar year of the profiles to count.",
                },
            },
            "additionalProperties": False,
        },
    ),
    types.Tool(
        name="get_profile",
        description=(
            "Return the depth levels of one float's profiles: pressure, "
            "temperature, salinity and their QC flags. Give a cycle number to "
            "get a single profile."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "float_id": {
                    "type": "string",
                    "description": "The float's WMO identifier, for example 1900083.",
                },
                "cycle": {
                    "type": "integer",
                    "description": "One cycle number. Omit for every cycle.",
                },
            },
            "required": ["float_id"],
            "additionalProperties": False,
        },
    ),
    types.Tool(
        name="nearest_floats",
        description=(
            "The floats that have profiled closest to a point, nearest first, "
            "with the distance in kilometres and the position and time of each "
            "float's closest profile."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "number",
                    "minimum": -90,
                    "maximum": 90,
                    "description": "Degrees north; south is negative.",
                },
                "longitude": {
                    "type": "number",
                    "minimum": -180,
                    "maximum": 180,
                    "description": "Degrees east; west is negative.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_NEAREST,
                    "description": "How many floats to return. Defaults to 10.",
                },
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False,
        },
    ),
    types.Tool(
        name="describe_schema",
        description=(
            "The column catalog as text: tables, columns, units, QC "
            "conventions and the join path."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
]


LIST_FLOATS_SQL = """
SELECT
    p.float_id,
    count(*) AS profiles,
    min(p.obs_time)::date AS first_profile,
    max(p.obs_time)::date AS last_profile
FROM profiles p
WHERE (%(region)s::text IS NULL OR p.region = %(region)s)
  AND (
      %(year)s::int IS NULL
      OR (
          p.obs_time >= make_timestamptz(%(year)s::int, 1, 1, 0, 0, 0)
          AND p.obs_time < make_timestamptz(%(year)s::int + 1, 1, 1, 0, 0, 0)
      )
  )
GROUP BY p.float_id
ORDER BY count(*) DESC, p.float_id
LIMIT %(limit)s
"""

GET_PROFILE_SQL = """
SELECT
    p.cycle_number,
    p.obs_time,
    p.latitude,
    p.longitude,
    p.region,
    p.data_mode,
    m.pressure_dbar,
    m.temperature_c,
    m.salinity_psu,
    m.qc_flag
FROM measurements m
JOIN profiles p
    ON p.profile_id = m.profile_id
WHERE p.float_id = %(float_id)s
  AND (%(cycle)s::int IS NULL OR p.cycle_number = %(cycle)s)
ORDER BY p.cycle_number, m.pressure_dbar
LIMIT %(limit)s
"""


# The problem statement's third example, "what are the nearest ARGO floats to
# this location", is a fixed shape: one point in, floats ranked by distance out.
# So it is a parameterised query rather than generated SQL, for the same reason
# as the two lookups above it: generating it would spend a model call and add a
# failure mode to a question that one statement answers exactly.
#
# Per float, the single profile closest to the point, not every profile. A float
# that has cycled past the same spot forty times is one float near here, not
# forty. geography rather than geometry, so the distance is metres on the
# sphere rather than degrees, which mean different things at different
# latitudes. The GiST index on profiles.geom serves the ORDER BY.
NEAREST_FLOATS_SQL = """
WITH here AS (
    SELECT ST_SetSRID(
        ST_MakePoint(%(longitude)s, %(latitude)s), 4326
    )::geography AS point
),
closest AS (
    SELECT DISTINCT ON (p.float_id)
        p.float_id,
        ST_Distance(p.geom, here.point) / 1000.0 AS distance_km,
        p.latitude,
        p.longitude,
        p.obs_time,
        p.region
    FROM profiles p
    CROSS JOIN here
    WHERE p.geom IS NOT NULL
    ORDER BY p.float_id, p.geom <-> here.point
)
SELECT
    float_id,
    -- Back to float8 after rounding. round() on numeric returns a Decimal, the
    -- tool's JSON encoder writes a Decimal as a string, and the first run
    -- against the real archive returned '342.3': text that sorts and plots as
    -- text.
    round(distance_km::numeric, 1)::float8 AS distance_km,
    latitude,
    longitude,
    obs_time,
    region
FROM closest
ORDER BY distance_km, float_id
LIMIT %(limit)s
"""


def nearest_floats(
    latitude: float,
    longitude: float,
    limit: int | None = None,
) -> dict:
    """Floats ranked by their closest profile to a point, by hand written SQL."""
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError) as exc:
        raise ValueError("latitude and longitude must be numbers") from exc

    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError(
            f"({lat}, {lon}) is not a position: latitude runs -90 to 90 and "
            "longitude -180 to 180"
        )

    count = max(1, min(int(limit or 10), MAX_NEAREST))

    columns, rows = fetch_all(
        NEAREST_FLOATS_SQL,
        {"latitude": lat, "longitude": lon, "limit": count},
        readonly=True,
    )

    return _table(columns, rows)


def query_argo(question: str) -> dict:
    """Route one natural language question through the shared service."""
    from src.utils.pipeline import RAGService

    result = _service(RAGService).answer_from_data(question)

    return {
        "answer": result.get("answer"),
        "generated_sql": result.get("generated_sql"),
        "columns": result.get("columns"),
        "rows": result.get("rows"),
        "row_count": result.get("row_count"),
        "refused": bool(result.get("refused")),
    }


def list_floats(region: str | None = None, year: int | None = None) -> dict:
    """Floats with profile counts and date ranges, by hand written SQL."""
    columns, rows = fetch_all(
        LIST_FLOATS_SQL,
        {
            "region": region,
            "year": int(year) if year is not None else None,
            "limit": MAX_FLOATS,
        },
        readonly=True,
    )

    return _table(columns, rows)


def get_profile(float_id: str, cycle: int | None = None) -> dict:
    """One float's levels, by hand written SQL."""
    try:
        identifier = int(str(float_id).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"float_id must be a number, got {float_id!r}") from exc

    columns, rows = fetch_all(
        GET_PROFILE_SQL,
        {
            "float_id": identifier,
            "cycle": int(cycle) if cycle is not None else None,
            "limit": MAX_LEVELS,
        },
        readonly=True,
    )

    return _table(columns, rows)


def describe_schema() -> dict:
    """The column catalog, as the text the SQL generator itself is given."""
    return {"catalog": load_catalog()}


HANDLERS = {
    "query_argo": query_argo,
    "list_floats": list_floats,
    "get_profile": get_profile,
    "nearest_floats": nearest_floats,
    "describe_schema": describe_schema,
}


def call(name: str, arguments: dict | None) -> dict:
    """Dispatch one tool call by name. Raises for an unknown tool."""
    handler = HANDLERS.get(name)

    if handler is None:
        raise ValueError(f"unknown tool: {name}")

    return handler(**(arguments or {}))


def _table(columns, rows) -> dict:
    return {
        "columns": list(columns),
        "rows": [list(row) for row in rows],
        "row_count": len(rows),
    }


def _service(factory):
    """One service per process, built on first use rather than at import."""
    if _service.instance is None:
        _service.instance = factory()

    return _service.instance


_service.instance = None


async def _on_list_tools(context, params) -> types.ListToolsResult:
    return types.ListToolsResult(tools=TOOLS)


async def _on_call_tool(context, params) -> types.CallToolResult:
    """Run a tool and hand back its result as JSON text.

    A failure comes back as an error result rather than an exception, because
    a client should be told the tool failed and why, not dropped.
    """
    try:
        payload = await asyncio.to_thread(call, params.name, params.arguments)
        failed = False
    except Exception as exc:
        payload = {"error": f"{type(exc).__name__}: {exc}"}
        failed = True

    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=json.dumps(payload, default=str, indent=2),
            )
        ],
        is_error=failed,
    )


def build_server() -> Server:
    """The configured server, separated from running it so tests can build one."""
    return Server(
        SERVER_NAME,
        version="0.2.0",
        instructions=(
            "Ocean data from ARGO floats. Ask query_argo a question in plain "
            "English, or use list_floats, get_profile and nearest_floats for "
            "fixed lookups. There is no tool that accepts SQL."
        ),
        on_list_tools=_on_list_tools,
        on_call_tool=_on_call_tool,
    )


async def serve() -> None:
    server = build_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
