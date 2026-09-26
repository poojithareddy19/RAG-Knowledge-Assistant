"""Call this project's MCP server over the protocol, from inside the app.

The problem statement asks for the LLM layer to use the Model Context
Protocol. Until this module the server in ``src/mcp/server.py`` was a separate
door: an external client such as Claude Desktop could use it, and the chat on
the Streamlit page never did. The tools and the conversation were two systems
that happened to share a database.

This is the chat's side of that protocol. ``call_tool`` opens a real MCP
session against the server, performs the initialize handshake, lists nothing it
does not need, calls one tool by name and reads the JSON the tool returned. It
is the same request an external client makes; the only difference is that the
transport is a pair of in-memory streams instead of stdin and stdout, so there
is no subprocess to start and no port to open per question.

Going through the protocol rather than importing the tool functions is the
point, not a formality. The server's argument schemas, its refusal to accept
SQL and its error envelope are then enforced on the chat exactly as they are on
anyone else, so there is one tool layer with one set of rules rather than a
trusted internal path beside an untrusted external one.
"""

from __future__ import annotations

import asyncio
import json

from mcp.client import Client

from src.mcp.server import build_server


class ToolError(RuntimeError):
    """The tool ran and reported a failure through the protocol."""


async def _call(name: str, arguments: dict | None) -> dict:
    async with Client(build_server()) as client:
        result = await client.call_tool(name, arguments or {})

    text = "".join(
        getattr(part, "text", "") or "" for part in (result.content or [])
    )

    payload = json.loads(text) if text.strip() else {}

    if getattr(result, "is_error", False):
        raise ToolError(
            payload.get("error", f"{name} failed")
            if isinstance(payload, dict)
            else f"{name} failed"
        )

    return payload


def call_tool(name: str, arguments: dict | None = None) -> dict:
    """Call one tool on the project's MCP server and return its JSON result.

    Synchronous, because both callers are: a Streamlit script run and a
    FastAPI ``def`` endpoint each execute on a thread with no event loop, which
    is what ``asyncio.run`` needs.
    """
    return asyncio.run(_call(name, arguments))
