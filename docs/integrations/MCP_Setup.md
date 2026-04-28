# Aix MCP Server — Client Setup Guide

> CORE 5 #20 — **DONE** (Option A path). Last updated: 2026-04-27.
> All 7 phases landed: full MCP surface (10 tools / 4 resources / 2 prompts)
> reachable over both **stdio** and **Streamable HTTP**, regression-locked
> by a 19-test pytest suite (`tests/mcp_server/`). Live integration smokes
> against Claude Desktop / Cursor IDE / MCP Inspector are queued as a
> follow-up (deferred — see "Live integration follow-up" section below).

This document is the **canonical onboarding guide** for any client that wants
to talk to the Aix Knowledge Graph and lesson-planning agent over the
[Model Context Protocol (MCP)](https://modelcontextprotocol.io/).

It covers:

1. [What's available right now](#whats-available-right-now)
2. [Quick smoke test (no MCP client needed)](#1-quick-smoke-test-no-mcp-client-needed)
3. [Streamable HTTP transport (remote / production)](#2-streamable-http-transport-remote--production)
4. [Claude Desktop integration (stdio, local)](#3-claude-desktop-integration-stdio-local)
5. [Cursor IDE integration (stdio, local)](#4-cursor-ide-integration-stdio-local)
6. [MCP Inspector — the "DevTools" for MCP](#5-mcp-inspector--the-devtools-for-mcp)
7. [Roadmap (Phase 6-7)](#roadmap)
8. [Troubleshooting](#troubleshooting)

---

## What's available right now

### Tools (10)

| Phase | Name | What it does |
|---|---|---|
| 1 | `kg.search` | Free-text natural-language query (Italian or English) → ranked KG nodes, relationships, methodology recommendations. The same pipeline the webui's Retriever agent uses. |
| 1 | `kg.get_context` | Returns the same shape `POST /api/v1/context` produces (educational ranking + media counts). Drop-in replacement for the REST endpoint. |
| 1 | `kg.list_concepts` | Lists concept names available in a domain. Cheap (file-backed, no Neo4j round-trip). Useful as a discovery pre-step. |
| 1 | `kg.get_schema` | Returns node-label categories, similarity threshold, and methodology categories for a domain. Lets a client craft a precise query. |
| 3 | `media.lookup_curated` | Look up Aix's offline curated media (videos, articles, OER) for a list of concepts. No external API calls. |
| 3 | `media.search_youtube` | Live YouTube Data API v3 search with educational-quality scoring. Falls back to a search URL when `YOUTUBE_API_KEY` is unset. |
| 3 | `media.search_academic` | Live Semantic Scholar search with rate-limit handling and exponential backoff. Returns ranked papers (year, citations, open-access). |
| 3 | `media.search_oer` | Live OER textbook search across DOAB, Open Textbook Library, and BC Campus. Open-access only. |
| 3 | `media.generate_diagram` | LLM-generated Mermaid diagram (mindmap / flowchart / sequence / timeline / hierarchy / comparison / process) rendered to SVG/PNG via mermaid.ink. |
| 4 | `agent.run_lesson_plan` | Runs the **full Aix multi-agent pipeline** (Planner → Retriever → Writer → Critic) and returns a complete lesson plan with the same shape as `POST /api/v1/agent/run`. Streams MCP progress notifications during the 30-120s run. |

### Resources (4)

| URI | Type | Returns |
|---|---|---|
| `kg://schema` | Static | Schema for both domains (node labels, similarity threshold, methodology categories). |
| `kg://concepts/{domain}` | Template | Concept catalogue for the requested domain. |
| `methodology://list` | Static | Pedagogical methodology catalogue (per domain). |
| `media://stats` | Static | Curated media coverage statistics. |

### Prompts (2)

| Name | Purpose |
|---|---|
| `educational-query` | One-shot helper that turns a free-text topic into an educational query suitable for `kg.search`. |
| `lesson-plan-request` | Pre-fills `agent.run_lesson_plan` with topic / duration / methodology / level. |

Domains supported: `neuro` (default) and `udl`.

Transports:

* **stdio** — local, used by Claude Desktop and Cursor IDE.
* **Streamable HTTP** — remote, mounted at `https://.../mcp/` inside the public FastAPI app, gated by the same JWT Bearer secret used by `/api/v1/agent/*`.

---

## 1. Quick smoke test (no MCP client needed)

Validate everything works in-process before wiring it into a real client.

### List the registered tools / resources / prompts

```powershell
cd c:\Users\louis\KBRAGold\graphaixlearning
python scripts/diagnostic/mcp_smoke.py
```

You should see the full surface: 10 tools, 4 resources, 2 prompts.

### Call individual tools end-to-end (stdio, in-process)

```powershell
# Phase 1 — KG (cheap, no Neo4j round-trip):
python scripts/diagnostic/mcp_smoke.py --call kg.list_concepts --domain neuro --limit 8
# Phase 1 — KG (full pipeline ~5-30s):
python scripts/diagnostic/mcp_smoke.py --call kg.search --query "motivazione intrinseca" --domain neuro
python scripts/diagnostic/mcp_smoke.py --call kg.get_schema --domain neuro

# Phase 2 — resources + prompts:
python scripts/diagnostic/mcp_smoke.py --read-resource kg://schema
python scripts/diagnostic/mcp_smoke.py --read-resource kg://concepts/neuro
python scripts/diagnostic/mcp_smoke.py --render-prompt educational-query --topic "fotosintesi"
python scripts/diagnostic/mcp_smoke.py --render-prompt lesson-plan-request --topic "respirazione cellulare" --duration 45
python scripts/diagnostic/mcp_smoke.py --phase2-verify

# Phase 3 — media:
python scripts/diagnostic/mcp_smoke.py --phase3-verify
python scripts/diagnostic/mcp_smoke.py --call media.lookup_curated --concepts "Selective Attention,Cognitive Control" --domain neuro
python scripts/diagnostic/mcp_smoke.py --call media.search_youtube --query "metacognition" --max-results 3

# Phase 4 — full agent pipeline (~60-120s, runs the lesson-plan agent):
python scripts/diagnostic/mcp_smoke.py --phase4-verify
```

### Smoke-test the **HTTP** transport (Phase 5)

After starting the API:

```powershell
uvicorn aix.api.main:app --port 8765 --app-dir src
```

Run the end-to-end Phase 5 verification (asserts 401 without token, logs in, lists tools, calls `kg.list_concepts` over HTTP):

```powershell
python scripts/diagnostic/mcp_smoke.py --phase5-verify --email "your-teacher@email.it" --password "your-pass"
# or via env vars:
$env:AIX_TEST_USER_EMAIL="your-teacher@email.it"
$env:AIX_TEST_USER_PASSWORD="your-pass"
python scripts/diagnostic/mcp_smoke.py --phase5-verify
```

Plus a credentials-free sanity probe (auth gating, /docs, /health):

```powershell
python scripts/diagnostic/probe_mcp_endpoint.py
```

If those work, the server is functional on both transports. The remaining
sections plug it into your favourite MCP client.

---

## 2. Streamable HTTP transport (remote / production)

The MCP server is mounted at **`/mcp/`** inside the same FastAPI app that
serves `/api/v1/*`, `/webui/*`, and `/auth/jwt/*`. There is no separate
process to deploy: the existing uvicorn (or gunicorn-uvicorn worker in
prod) is enough.

### Endpoint

| URL | Purpose |
|---|---|
| `https://<host>/mcp/` | Streamable HTTP RPC (POST for requests, GET for SSE notifications, DELETE to close session) |

The transport is the canonical [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports#streamable-http) defined by the spec — full session lifecycle including SSE-based server notifications and resumable sessions.

### Authentication

The endpoint validates **the same JWT Bearer tokens** that `/api/v1/agent/*`
accepts. A single `POST /auth/jwt/login` call gives a token usable on both
surfaces interchangeably.

```bash
# 1. Mint a token (form-encoded login).
curl -s -X POST 'https://<host>/auth/jwt/login' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=teacher@example.it&password=...' | jq -r .access_token

# 2. Use the token on /mcp/.
curl -X POST 'https://<host>/mcp/' \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}'
```

Auth verifier configuration (FastMCP `JWTVerifier`):

* algorithm: `HS256`
* audience: `fastapi-users:auth`
* secret: `WEBUI_AUTH_SECRET` env var (same secret used by the cookie + Bearer backends)

> **Dev escape hatch (NEVER use in production):** set `AIX_MCP_REQUIRE_AUTH=0`
> to disable the verifier and expose `/mcp/` unauthenticated. The startup
> log will print a loud warning when this is on.

### Connecting from Python (FastMCP `Client`)

```python
import asyncio
from fastmcp import Client

async def main():
    token = "eyJhbGc..."  # from POST /auth/jwt/login
    async with Client("https://<host>/mcp/", auth=token) as mcp:
        tools = await mcp.list_tools()
        print(f"{len(tools)} tools available")
        result = await mcp.call_tool("kg.list_concepts", {"domain": "neuro", "limit": 5})
        print(result.structured_content)

asyncio.run(main())
```

### Connecting from MCP Inspector (browser-based)

```powershell
npx @modelcontextprotocol/inspector
```

In the inspector UI:

* Transport: **Streamable HTTP**
* URL: `https://<host>/mcp/`
* Authentication: `Bearer <token>` (paste the JWT from `/auth/jwt/login`)

### Which clients can use this

| Client | Supported? | Notes |
|---|---|---|
| MCP Inspector (browser) | ✅ Yes | Use Bearer auth in the inspector's auth panel. |
| FastMCP `Client` (Python) | ✅ Yes | `Client(url, auth=token)` — see snippet above. |
| Lovable apps | ✅ Yes (via /api/v1/agent/* OR /mcp/) | The simpler `POST /api/v1/agent/run` may be enough; MCP is overkill unless you need the discovery semantics. |
| Partner LangGraph cloud agents | ✅ Yes | Use FastMCP's `Client` or the official `mcp` SDK with Bearer auth. |
| Cursor IDE / Claude Desktop (remote mode) | ✅ Yes | Both have a "remote MCP" config that accepts `url` + `headers`. |
| ChatGPT Desktop's remote MCP feature | ⚠️ Requires OAuth 2.1 + PKCE | Phase 5 only ships JWT Bearer; OAuth is a separate optional follow-up ("Phase 5b"). |

### CORS

The parent FastAPI app's CORS middleware applies to `/mcp/` automatically.
For browser-based MCP clients, set `WEBUI_CORS_ALLOW_ORIGINS` to the
allowed origins (comma-separated; defaults to `*` for backward-compat).

---

## 3. Claude Desktop integration (stdio, local)

### Where the config lives

| OS | Path |
|---|---|
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

### Config snippet (Windows)

```json
{
  "mcpServers": {
    "aix-graphrag": {
      "command": "C:\\Users\\louis\\KBRAGold\\venv\\Scripts\\python.exe",
      "args": ["-m", "aix.mcp.stdio_main"],
      "cwd": "C:\\Users\\louis\\KBRAGold\\graphaixlearning",
      "env": {
        "PYTHONPATH": "C:\\Users\\louis\\KBRAGold\\graphaixlearning\\src",
        "AIX_MCP_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

> **Important — use the venv's `python.exe`, not just `python`.**
> Claude Desktop launches the subprocess outside any shell, so it does
> not see your `(venv)` activation. Pointing `command` at the venv's
> `python.exe` ensures `fastmcp`, `neo4j`, `langchain`, etc. are
> resolvable.

### Config snippet (macOS / Linux)

```json
{
  "mcpServers": {
    "aix-graphrag": {
      "command": "/path/to/your/venv/bin/python",
      "args": ["-m", "aix.mcp.stdio_main"],
      "cwd": "/path/to/graphaixlearning",
      "env": {
        "PYTHONPATH": "/path/to/graphaixlearning/src",
        "AIX_MCP_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Environment variables to forward

The MCP subprocess runs **without** your shell's environment, so any
secret your tools need must be passed via the `env` block above.

| Variable | Used by | Required? |
|---|---|---|
| `NEO4J_URI` | `kg.*`, `agent.*` | Yes (or have `.env` in `cwd`) |
| `NEO4J_USERNAME` | same | Yes |
| `NEO4J_PASSWORD` | same | Yes |
| `OPENROUTER_API_KEY` (or `OPENAI_API_KEY`) | Cypher gen + planner / writer / critic LLMs | Yes |
| `YOUTUBE_API_KEY` | `media.search_youtube` (live mode) | No (falls back to URL) |
| `AIX_MCP_LOG_LEVEL` | The MCP subprocess log verbosity | No (default `INFO`) |

The subprocess also calls `dotenv.load_dotenv()` at startup, so if your
secrets live in `c:/Users/louis/KBRAGold/graphaixlearning/.env` and you
keep `cwd` pointed there, you can omit the secret-environment entries.

### After editing the config

1. Save `claude_desktop_config.json`.
2. **Fully quit** Claude Desktop (system tray → Quit, not just close
   the window — Claude keeps a background process).
3. Re-open it. The server appears as a *plug icon* in the prompt bar.
4. Type a question that uses the KG, e.g.:

   > Usa lo strumento `kg.search` per trovare strategie didattiche
   > legate alla motivazione intrinseca nel dominio `neuro`.

---

## 4. Cursor IDE integration (stdio, local)

Cursor reads `~/.cursor/mcp.json` on startup. Add an entry that mirrors
the Claude Desktop one:

```json
{
  "mcpServers": {
    "aix-graphrag": {
      "command": "C:\\Users\\louis\\KBRAGold\\venv\\Scripts\\python.exe",
      "args": ["-m", "aix.mcp.stdio_main"],
      "cwd": "C:\\Users\\louis\\KBRAGold\\graphaixlearning",
      "env": {
        "PYTHONPATH": "C:\\Users\\louis\\KBRAGold\\graphaixlearning\\src"
      }
    }
  }
}
```

Restart Cursor (Cmd/Ctrl+Shift+P → *"Developer: Reload Window"*). The
agent will auto-discover the full tool surface and call them when
relevant.

---

## 5. MCP Inspector — the "DevTools" for MCP

The official Inspector is invaluable when something doesn't work.

### Against the stdio entry (local)

```powershell
npx @modelcontextprotocol/inspector C:\Users\louis\KBRAGold\venv\Scripts\python.exe -m aix.mcp.stdio_main
```

### Against the HTTP mount (remote / Phase 5)

```powershell
npx @modelcontextprotocol/inspector
```

Then in the UI:

* Transport: **Streamable HTTP**
* URL: `http://127.0.0.1:8765/mcp/` (dev) or `https://<host>/mcp/` (prod)
* Authentication: paste your `Bearer <token>` from `/auth/jwt/login`

The inspector lets you:

* See the negotiated MCP version + server identity
* Call any tool by name with a JSON form for arguments
* Read any resource / render any prompt
* Inspect the raw JSON-RPC traffic (request + response)
* Read the `instructions` field that the server advertises to the LLM

If the Inspector connects but a real client doesn't, the bug is in the
client's config; if the Inspector also fails, the bug is in our code.

---

## Roadmap

| Phase | Status | Scope |
|---|---|---|
| 1 | ✅ Done | 4 `kg.*` tools + stdio entry + smoke script + this doc |
| 2 | ✅ Done | 4 `kg://...` / `methodology://list` / `media://stats` resources + 2 prompt templates (`educational-query`, `lesson-plan-request`) |
| 3 | ✅ Done | 5 `media.*` tools (lookup curated, search YouTube, search Semantic Scholar, search OER, generate Mermaid diagram) |
| 4 | ✅ Done | 1 `agent.run_lesson_plan` tool — wraps the same `stream_agent_events` helper that backs `POST /api/v1/agent/run` from CORE 2 #7. MCP progress notifications stream the Planner / Retriever / Writer / Critic phases. |
| 5 | ✅ Done | Streamable HTTP transport mounted at `/mcp/` inside the FastAPI app, gated by the same JWT Bearer secret as `/api/v1/agent/*`. |
| 6 | ✅ Done | `tests/mcp_server/` integration suite (19 tests) + OpenAPI strictly-additive regression baseline (`data/diagnostic/openapi_before_p20.txt`). All green in ~64s. |
| 7 | ✅ Done | Final docs polish + ClickUp #20 DONE entry. **Live integration smokes deferred** — see the next section. |

Each phase landed behind its own smoke test, no breaking changes to the
existing `/api/v1/*`, `/webui/*`, or `/auth/*` surfaces. See
`docs/product/ClickUp_Agentic_GraphRAG_Update.md` → Subtask 20 for the
master tracker.

---

## Live integration follow-up (deferred from Phase 7)

The 19-test regression suite proves the MCP surface is correct, the JWT
auth gate works, and the REST API is strictly additive. **What it does
not prove** is that a real GUI MCP client (Claude Desktop / Cursor IDE)
discovers and calls our tools when wired up — that's a manual
click-test, not a code change.

The following 30-minute smoke sequence is queued as a follow-up. None
of it requires new code; the configs are already in this document.

| Smoke | What it proves | Where to find the config |
|---|---|---|
| Cursor IDE (stdio) — wire `~/.cursor/mcp.json`, reload window, see 10 tools in MCP panel, call one in chat | The stdio transport works against the most common dev-time MCP host | [§4 — Cursor IDE integration](#4-cursor-ide-integration-stdio-local) |
| Claude Desktop (stdio) — wire `claude_desktop_config.json`, fully quit + relaunch, see plug icon, ask a question that uses `kg.search` | The stdio transport works against the canonical MCP reference client | [§3 — Claude Desktop integration](#3-claude-desktop-integration-stdio-local) |
| MCP Inspector (HTTP + JWT) — `npx @modelcontextprotocol/inspector`, paste a token from `/auth/jwt/login`, list tools, call one | The Streamable HTTP transport works against any third-party MCP client (Lovable connect, partner LangGraph agents, browser playgrounds) | [§5 — MCP Inspector](#5-mcp-inspector--the-devtools-for-mcp) |

The pre-commit pytest suite already locks every contract those clients
would exercise (tool names, prompt arguments, JWT verifier behaviour,
resource shapes). If the smokes ever fail, the diff is in the *client*
config, not in our server code.

### What's *out* of this follow-up (and why)

| Concern | Where it actually belongs |
|---|---|
| Lovable apps connecting via MCP | Lovable doesn't have a native MCP-client UI today (April 2026). Lovable-built frontends should consume our agent via the **REST endpoint** (`POST /api/v1/agent/run` from CORE 2 #7), which is fully documented in `/docs` (Swagger). MCP is overkill. |
| ChatGPT Desktop's remote MCP feature | OpenAI's remote MCP currently requires **OAuth 2.1 + PKCE**; we ship JWT Bearer. A separate optional follow-up ("Phase 5b") would add an OAuth provider; defer until a real customer needs ChatGPT integration. |
| `kg.search` returning *correct* concepts | Data-quality / Neo4j review — separate from MCP wiring. |
| Agent generating *good* lesson plans | Subjective benchmark, tracked under #11 / #13 / #17 (eval harness). |
| Production load testing | Ops task, post-deploy. Out of scope for #20. |

---

## Production deployment notes

When promoting `/mcp/` from `127.0.0.1:8765` to a public host, the
following invariants must hold:

| Concern | Required setting |
|---|---|
| Transport security | TLS-only — terminate at the reverse proxy (nginx / Caddy / Cloud Run / Fly). The MCP spec assumes encrypted transport for any non-localhost deployment. |
| `WEBUI_AUTH_SECRET` | Must be ≥ 32 random bytes, identical for the API and the MCP verifier (they share it via env), rotated only with a forced re-login window. |
| `AIX_MCP_REQUIRE_AUTH` | Leave **unset** (or set to `1`). The `=0` escape hatch is dev-only and prints a loud warning at startup. |
| `WEBUI_CORS_ALLOW_ORIGINS` | Set to the explicit list of allowed origins (no `*`) for any browser-based MCP client. |
| Session affinity | Streamable HTTP keeps state across requests. If you scale horizontally, configure sticky sessions on the load balancer (the session ID is in the `Mcp-Session-Id` request header). |
| Cold-start budget | First call per worker pays ~30s for `Text2Cypher` schema cache + Node2Vec model load. Use a warm-up probe in your readiness check or rely on the pre-built singletons (`kg_tools._GRAPHRAG_TOOL`, `media_tools._MEDIA_LOOKUP`). |
| OpenAPI surface | `/mcp/` is a Starlette mount, **not** an OpenAPI route. It will not appear in `/openapi.json` and that is enforced by `tests/mcp_server/test_mcp_openapi_regression.py::test_mcp_path_is_not_in_openapi`. |
| Future: OAuth 2.1 + PKCE | Required for ChatGPT Desktop's remote MCP feature. Implement as a separate `/oauth/*` mount and pass the resolved JWT to `JWTVerifier`. Out of scope for #20. |

---

## Troubleshooting

### *"The Aix server doesn't show up in Claude Desktop"*

1. Confirm the JSON parses: paste the file into <https://jsonlint.com/>.
2. Check Claude Desktop's logs:
   * Windows: `%APPDATA%\Claude\logs\`
   * macOS: `~/Library/Logs/Claude/`
3. Re-launch Claude Desktop after **fully quitting** (tray icon → Quit).
4. Run the MCP Inspector against the same command/args/env to isolate
   whether the bug is in the server or the client.

### *"`Failed to load module aix.mcp.stdio_main`"*

The subprocess can't find our package. Verify:

* `command` points at the **venv's** `python.exe` (not the system one).
* `cwd` is the repo root (the directory containing `requirements.txt`).
* `env.PYTHONPATH` includes `<repo>/src`.

A fast Windows sanity check from PowerShell:

```powershell
$env:PYTHONPATH = "C:\Users\louis\KBRAGold\graphaixlearning\src"
& "C:\Users\louis\KBRAGold\venv\Scripts\python.exe" -c "from aix.mcp import build_mcp_server; m = build_mcp_server(); print(m.name)"
# Expected: aix-graphrag
```

### *"`POST /mcp/` returns 401 even with my token"*

* Confirm the token came from `POST /auth/jwt/login` (not `/auth/login`,
  which mints a cookie, not a Bearer token).
* The Bearer header must be exactly `Authorization: Bearer <token>`.
* Audience check: the verifier expects audience `fastapi-users:auth` and
  algorithm `HS256`. Tokens issued by anything other than our
  `JWTStrategy` (e.g. an OAuth provider) won't validate.
* If you rebuilt the server with a new `WEBUI_AUTH_SECRET`, old tokens
  signed with the previous secret are invalidated — log in again.

### *"`kg.search` returns 0 nodes"*

Either the KG genuinely has no data on that topic, or Neo4j credentials
aren't reaching the MCP subprocess. Verify the credentials are in `.env`
(loaded via `dotenv.load_dotenv()` at startup) or in the `env` block of
the MCP config. The smoke script reuses the same `.env` so if the
smoke works but the MCP client doesn't, it's a config-passthrough bug.

### *"My logs are corrupting the protocol"*

The stdio entry point sends **all** logs to stderr precisely so the JSON-RPC
protocol on stdout stays clean. If you import a third-party module that
writes to stdout (`print(...)`, `logging.StreamHandler(sys.stdout)`), it
will break the connection. Run the offending tool through MCP Inspector
to see the malformed frames.

### *"Cold-start takes ~30 seconds"*

That's the one-time `Text2Cypher` schema cache build + Node2Vec model
load. Subsequent calls in the same MCP session are sub-second because
we cache `GraphRAGTool` instances per domain in module scope. If you
hit cold start every call, your client is restarting the subprocess —
check that the MCP host keeps the connection alive between turns
(Claude Desktop and Cursor both do by default).

### *"`circular import` at uvicorn cold start"*

Symptom: `cannot import name 'mcp' from partially initialized module 'aix.mcp.server'`.
Root cause: a stale `sys.path.insert(0, src/aix)` in `aix/api/main.py` — that
made our internal `aix.mcp` package also resolvable as plain `mcp`, colliding
with the official Anthropic `mcp` SDK that `fastmcp` imports during its own
logging setup. **Fixed** by inserting `src/` (the project source root) instead
of `src/aix/`. If you see this again after editing `main.py`, double-check the
`sys.path.insert` line at the top.
