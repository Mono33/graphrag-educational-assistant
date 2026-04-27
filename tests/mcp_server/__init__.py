"""
CORE 5 #20 Phase 6 — MCP integration test suite.

NOTE on directory naming: this package is intentionally called ``mcp_server``
(NOT ``mcp``). Pytest prepends the test root to ``sys.path``, so a directory
named ``tests/mcp/`` would shadow the third-party ``mcp`` SDK that
``fastmcp`` imports as ``from mcp import McpError``. Same hygiene rule we
enforced on the source side in Phase 5 — never let a local package name
collide with the upstream MCP SDK.

These tests lock in the contract shipped by Phases 1-5:

* ``test_mcp_surface``           — tools / resources / prompts inventory
* ``test_mcp_http_auth``         — JWT Bearer auth gate on /mcp/
* ``test_mcp_kg_tools``          — cheap tool calls (no Neo4j needed)
* ``test_mcp_agent_tool_contract`` — agent.run_lesson_plan response shape
* ``test_mcp_openapi_regression``  — strictly-additive REST surface guard

Tests run in-process via FastAPI's ``TestClient`` and FastMCP's in-memory
``Client`` transport. They do NOT require a running uvicorn, a live Neo4j
instance, or any external API key. Total runtime: ~5-10 seconds.

The suite is the local mirror of what a CI job would run on every push.
"""
