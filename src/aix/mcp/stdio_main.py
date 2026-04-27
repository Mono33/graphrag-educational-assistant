"""
Stdio entry point for the Aix MCP server.

Run via:

    python -m aix.mcp.stdio_main

This is the entry point that local MCP clients (Claude Desktop, Cursor IDE,
ChatGPT desktop, MCP Inspector) talk to over the stdio transport. The MCP
spec defines stdio as the local-process transport — there is no network
exposure and no auth (the OS process boundary is the trust boundary).

For remote / production access, use the Streamable HTTP transport mounted
inside the FastAPI app at ``/mcp`` (Phase 5 of CORE 5 #20). Both transports
share the same tool definitions because they share the same FastMCP
instance from ``aix.mcp.server``.

Example: register this entry point in Claude Desktop's
``claude_desktop_config.json``::

    {
      "mcpServers": {
        "aix-graphrag": {
          "command": "python",
          "args": ["-m", "aix.mcp.stdio_main"],
          "cwd": "C:/Users/louis/KBRAGold/graphaixlearning",
          "env": {
            "PYTHONPATH": "C:/Users/louis/KBRAGold/graphaixlearning/src",
            "NEO4J_URI": "...",
            "NEO4J_USERNAME": "...",
            "NEO4J_PASSWORD": "..."
          }
        }
      }
    }

See ``docs/integrations/MCP_Setup.md`` for the canonical config snippets.
"""

from __future__ import annotations

import logging
import os
import sys


def _configure_logging() -> None:
    """Send all logs to stderr.

    CRITICAL: stdout is reserved for the MCP JSON-RPC protocol over stdio.
    Any stray ``print`` or ``logging.StreamHandler(sys.stdout)`` will break
    the client connection because Claude Desktop / Cursor parse stdout as
    JSON-RPC frames. Logging must go to stderr only.
    """
    level_name = os.environ.get("AIX_MCP_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root = logging.getLogger()
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    root.setLevel(level)


def main() -> None:
    _configure_logging()
    logger = logging.getLogger(__name__)

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    from aix.mcp.server import build_mcp_server

    mcp = build_mcp_server()

    logger.info(
        "[MCP-stdio] Booting Aix MCP server '%s' on stdio transport — "
        "send Ctrl+C in the host (Claude Desktop / Cursor) to disconnect.",
        mcp.name,
    )

    mcp.run()


if __name__ == "__main__":
    main()
