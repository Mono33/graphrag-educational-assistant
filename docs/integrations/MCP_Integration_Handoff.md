# MCP Integration Handoff — Design / Dev Guide

> Last updated: 2026-05-03.
> Companion to: `docs/integrations/MCP_Setup.md` (full technical reference).

---

## What we ship today

The Aix Agentic GraphRAG platform exposes a **full MCP surface**:

| Primitive | Count | Examples |
|-----------|-------|----------|
| **Tools** | 10 | `kg.search`, `kg.list_concepts`, `media.search_youtube`, `agent.run_lesson_plan` |
| **Resources** | 4 | `kg://schema`, `kg://concepts/{domain}`, `methodology://list`, `media://stats` |
| **Prompts** | 2 | `educational-query`, `lesson-plan-request` |

**Two transports are live:**

| Transport | Use case | Auth |
|-----------|----------|------|
| **stdio** (local subprocess) | Claude Desktop, MCP Inspector (local) | None needed (OS process boundary) |
| **Streamable HTTP** (`/mcp/`) | Remote clients, partner agents, browser-based Inspector | JWT Bearer (same token as `/api/v1/*`) |

---

## 1. Claude Desktop

**How it works:** Claude Desktop spawns the MCP server as a local subprocess (stdio).

**Setup:**
1. Copy `mcp-configs/claude-desktop.example.json` into:
   - **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
   - **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
   - **Linux:** `~/.config/Claude/claude_desktop_config.json`
2. Replace `<VENV>` and `<REPO>` with your actual paths
3. **Fully quit** Claude Desktop (system tray → Quit, not just close window)
4. Relaunch — look for the plug icon in the prompt bar
5. Test: "Usa lo strumento `kg.search` per trovare strategie sulla motivazione intrinseca nel dominio `neuro`."

**Critical:** Use the **venv's** `python.exe`, not bare `python`. Claude Desktop launches outside any shell.

**Env vars:** Either keep them in `graphaixlearning/.env` (auto-loaded) or duplicate in the `"env"` block.

---

## 2. Slack

**Slack is NOT an MCP client.** There is no "plug MCP into Slack" button. Integration options:

### Option A — REST API (recommended, simplest)

Build a Slack bot (Bolt SDK) that calls our existing REST endpoints:

| Endpoint | Purpose |
|----------|---------|
| `POST /api/v1/agent/run` | Full lesson plan (same as web UI) |
| `POST /api/v1/context` | KG context lookup |
| `GET /api/v1/health` | Health check |

**Auth flow:** Bot service calls `POST /auth/jwt/login` once → caches Bearer token → passes it on every request.

**Streaming:** `POST /api/v1/agent/run` supports SSE (`Accept: text/event-stream`). The bot can post incremental updates to a Slack thread as planner/writer/critic phases complete.

**This is the same API the web UI uses.** No new server code needed.

### Option B — MCP over HTTP (only if discovery matters)

A backend service (Python/Node) connects to `https://<host>/mcp/` with a Bearer token and uses the MCP protocol for tool discovery + invocation. Useful if the Slack bot is actually an **agent** that needs to discover tools dynamically.

```python
from fastmcp import Client

async with Client("https://<host>/mcp/", auth=token) as mcp:
    tools = await mcp.list_tools()
    result = await mcp.call_tool("kg.search", {"query": "motivazione", "domain": "neuro"})
```

### When to pick A vs B

| Criteria | Option A (REST) | Option B (MCP HTTP) |
|----------|-----------------|---------------------|
| Slack bot sends known requests | Best fit | Overkill |
| Bot is an agent that picks tools dynamically | Not designed for this | Good fit |
| Dev effort | ~1 day (Bolt + fetch) | ~2-3 days (MCP client + agent logic) |
| Dependencies | `httpx` or `requests` | `fastmcp` SDK |

**Recommendation:** Start with **Option A**. Move to B only if you build an agentic Slack assistant.

---

## 3. Lovable

**Lovable does NOT have a native MCP client UI.** Lovable-built frontends should consume the agent via the **REST API**.

### Integration pattern

```
Lovable app  →  POST /api/v1/agent/run  →  Aix backend  →  lesson plan JSON
```

**What to give Lovable devs:**

1. **Base URL** of the deployed Aix API
2. **Auth:** `POST /auth/jwt/login` (form-encoded `username` + `password`) → returns `{ "access_token": "..." }`
3. **Main endpoint:** `POST /api/v1/agent/run` with body:
   ```json
   {
     "query": "Crea una lezione sull'attenzione selettiva",
     "domain": "neuro",
     "language": "it",
     "max_revisions": 1
   }
   ```
4. **Response shape:** `{ "lesson_plan_md": "...", "meta": {...}, "planner": {...}, "retriever": {...} }`
5. **OpenAPI docs:** `https://<host>/docs` (Swagger) — full schema, try-it-out panel
6. **SSE streaming** (optional): set `Accept: text/event-stream` header to get phase-by-phase progress

**MCP is NOT needed for Lovable** unless they build a service that needs tool discovery semantics.

---

## 4. Partner / LangGraph agents (remote MCP)

For any Python-based partner agent that wants tool discovery over the network:

1. Copy `mcp-configs/remote-http.example.json` for the connection shape
2. Obtain JWT: `POST /auth/jwt/login`
3. Connect: `Client("https://<host>/mcp/", auth=token)`
4. Full MCP protocol: `list_tools()`, `call_tool()`, `read_resource()`, `render_prompt()`

**Session affinity required** if the API runs behind a load balancer (sticky on `Mcp-Session-Id` header).

---

## Quick reference: which client uses which transport

| Client | Transport | Auth | Config template |
|--------|-----------|------|-----------------|
| **Claude Desktop** | stdio (local) | None (OS boundary) | `mcp-configs/claude-desktop.example.json` |
| **MCP Inspector** | stdio or HTTP | None / Bearer JWT | See `MCP_Setup.md` §5 |
| **Slack bot** | REST (`/api/v1/*`) | Bearer JWT | No MCP config needed |
| **Lovable app** | REST (`/api/v1/*`) | Bearer JWT | No MCP config needed |
| **Partner agents** | Streamable HTTP (`/mcp/`) | Bearer JWT | `mcp-configs/remote-http.example.json` |
| **ChatGPT Desktop** | Streamable HTTP | OAuth 2.1 + PKCE | Not yet supported (Phase 5b) |

---

## Governance (all channels)

Regardless of transport, these are **platform concerns** enforced server-side:

- **Scopes:** Today all authenticated users see all 10 tools. Role-based tool filtering is a future enhancement.
- **Audit:** All tool calls are logged (structured logging, `session_id` correlation).
- **Rate limits:** Inherit from the FastAPI app's middleware. LLM-heavy tools (`agent.run_lesson_plan`) are naturally throttled by upstream API costs.
- **Least privilege:** The MCP server exposes a **curated** tool set — no raw shell, no raw DB access, no file system. Every tool wraps a validated business operation.
- **Auth in production:** `AIX_MCP_REQUIRE_AUTH` must be `1` (default). The `=0` escape hatch is dev-only.

---

## Troubleshooting

See `docs/integrations/MCP_Setup.md` → "Troubleshooting" section for:
- Server not showing up in Claude Desktop
- `Failed to load module` errors
- 401 on `/mcp/` with valid token
- Cold-start latency (~30s first call)
- Circular import issues

---

## Appendix — MCP Integration Checklist

Before shipping any MCP-connected feature, the team should be able to answer **yes** to every question below. Use this as a review gate during design reviews and PR sign-offs.

### A. Do you actually need MCP?

- [ ] **Is the client an MCP host?** (Claude Desktop, MCP Inspector, a LangGraph agent with an MCP client SDK.) If not — e.g. a Slack bot, a Lovable frontend, a mobile app — use the REST API (`/api/v1/*`) instead. MCP adds protocol overhead with no benefit for plain HTTP consumers.
- [ ] **Does the client need tool discovery?** If it always calls the same 1-2 endpoints, REST is simpler. MCP shines when the client (or its LLM) picks tools dynamically.

### B. Auth and security

- [ ] **Is `AIX_MCP_REQUIRE_AUTH` set to `1` (or unset) in every non-local environment?** The `=0` escape hatch must never reach staging or production.
- [ ] **Are JWT tokens short-lived and scoped?** Tokens from `/auth/jwt/login` carry the `fastapi-users:auth` audience. Never hardcode tokens in client configs that get committed to git.
- [ ] **Is the transport encrypted?** Any non-localhost `/mcp/` deployment must be behind TLS. Stdio (local) is exempt — the OS process boundary is the trust boundary.

### C. Tool surface and context budget

- [ ] **Are you exposing only the tools the client actually needs?** Exposing all 10 tools to a client that only calls `kg.search` wastes context tokens. Future: role-based tool filtering; today: document which tools each integration uses.
- [ ] **Have you measured the token cost of your tool catalogue?** Each tool's JSON Schema (name + description + parameter definitions) is injected into the LLM's context window on every turn. More tools = higher cost and slower responses.
- [ ] **Are tool descriptions concise?** Verbose descriptions inflate context. If you add a tool, keep the first line under ~120 characters.

### D. Operational readiness

- [ ] **Does the integration handle cold-start?** The first MCP call per process pays ~30s for KG schema cache + model load. Clients should show a loading state, not time out.
- [ ] **Is there a health check?** For HTTP integrations, `GET /api/v1/health` confirms the backend is alive. For stdio, MCP Inspector is the diagnostic tool.
- [ ] **Are errors surfaced, not swallowed?** MCP tool errors return `isError: true` with a message. The client must display or log this, not silently fail.
- [ ] **Is session affinity configured?** If running multiple API workers behind a load balancer, sticky sessions on the `Mcp-Session-Id` header are required for Streamable HTTP.

### E. Testing

- [ ] **Can you list all 10 tools from the client?** This is the minimum acceptance test for any MCP integration.
- [ ] **Can you call at least one cheap tool (`kg.list_concepts`)?** This validates end-to-end: config, env vars, Neo4j connection, MCP protocol.
- [ ] **Have you tested with MCP Inspector first?** If Inspector works but your client doesn't, the bug is in the client config, not our server.
