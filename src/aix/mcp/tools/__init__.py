"""
``aix.mcp.tools`` — MCP tool packages.

Each module in this package exposes a ``register(mcp: FastMCP)`` function
that decorates one or more functions with ``@mcp.tool`` and adds them to
the shared FastMCP instance built by ``aix.mcp.server.build_mcp_server``.

Conventions
-----------
* Tool names use a ``namespace.verb`` pattern (e.g. ``kg.search``,
  ``media.search_youtube``) so MCP clients can group them visually.
  FastMCP does not enforce a separator — the dot is purely for human UX.
* Tools that touch I/O (Neo4j, HTTP APIs, LLMs) are ``async def``.
* Tool docstrings are surfaced verbatim to the LLM via the MCP
  ``listTools`` response — write them as if for a model, not a human.
"""
