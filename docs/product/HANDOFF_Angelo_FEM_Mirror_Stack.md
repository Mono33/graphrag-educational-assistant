# Handoff — FEM "Mirror Stack" Frontend + Public API + MCP Tool Servers

> **For:** Angelo
> **From:** Louis
> **Date:** 2026-04-27
> **Scope:** Everything in this branch that hasn't been pushed yet — three large, additive landings (a teacher-facing webui, a public agent API, and an MCP server) plus the regression test scaffolding that locks them in.

---

## 0. TL;DR — what's in this changeset

This branch ships **three independent, additive features** behind the same FastAPI app, plus the safety net (tests + diagnostic scripts + docs) that proves they work and that nothing existing was broken.

| Theme | CORE# | Status | What it is |
|---|---|---|---|
| **Mirror Stack — teacher webui** (`/webui/*`) | CORE 2 #6.6 P0–P3 | ✅ Done | FastAPI + htmx + WebAwesome + Tailwind chat workspace at `/webui/lesson/{id}`. Three-pane layout (educational profile sidebar / agent chat / media sidebar). End-to-end teacher login → profile → granular streaming agent → lesson plan with file-upload context injection. Replaces the Streamlit prototype (which now shows a retirement banner). |
| **Public Agent API** (`/api/v1/agent/*`) | CORE 2 #7 | ✅ Done | `POST /api/v1/agent/run` (synchronous JSON) and `POST /api/v1/agent/stream` (SSE). JWT Bearer auth in parallel to cookie auth. Swagger UI Minimal/Rich examples dropdown. Strictly additive vs. the pre-existing `/api/v1/context` GraphRAG endpoint. |
| **MCP Tool Servers** (`/mcp/` + stdio) | CORE 5 #20 | ✅ Done (Option A) | FastMCP 3.x server: 10 tools, 4 resources, 2 prompts. Two transports: **stdio** (Claude Desktop / Cursor IDE) and **Streamable HTTP** (mounted at `/mcp/`, gated by the same JWT secret as the public API). Regression-locked by a 19-test pytest suite. |

**Nothing that worked before is broken.** Verified by: (a) before/after OpenAPI diffs (`data/diagnostic/openapi_before_p7.txt`, `openapi_before_p20.txt`), (b) pytest regression suite, (c) end-to-end smokes per phase.

**One thing left for you (Angelo):** a 30-minute manual smoke against real GUI MCP clients (Cursor IDE / Claude Desktop / MCP Inspector). No code changes — configs are already in `MCP_Setup.md`. Section 6 below has the click-by-click sequence.

---

## 1. Read these docs in this order

You don't need to read everything. The first three give you 90% of the context. The rest is reference.

### Tier 1 — start here (≈ 90 minutes total)

1. **`docs/product/ClickUp_Agentic_GraphRAG_Update.md`** — **the master tracker.** Top-level status table at the top; Subtask sections below it. Look for **Subtask 6.5, 6.6, 7, and 20** — those are everything in this branch. Each Acceptance Criteria checkbox tells you exactly what landed in each phase.
2. **`docs/architecture/Frontend_Platform_Evaluation.md`** — the "why FastAPI + htmx + WebAwesome instead of just keeping Streamlit?" decision doc. Read this before touching anything in `src/aix/webui/`. Lays out the three options we evaluated and why we picked the middle path.
3. **`docs/integrations/MCP_Setup.md`** — the canonical MCP onboarding guide. Has working configs for Claude Desktop and Cursor IDE, the Streamable HTTP / JWT auth flow, MCP Inspector instructions, the deferred live-integration follow-up (your Option B), and a full troubleshooting section.

### Tier 2 — when you actually touch code (≈ 30 minutes)

4. **`docs/architecture/Agentic_GraphRAG_Architecture_Analysis.md`** — the existing system architecture. Useful background if you've never touched the agent pipeline.
5. **`docs/api/API_INTEGRATION_GUIDE.md`** — pre-existing GraphRAG API guide. Now extended in spirit by the `/api/v1/agent/*` endpoints (visible in `/docs`).
6. **`docs/architecture/Agentic_GraphRAG_BestPractices_Validation.md`** — the validation doc that audits CORE 1–6 against best practices. Useful to understand *why* certain things were built.

### Tier 3 — reference only (read on demand)

| Doc | When you need it |
|---|---|
| `docs/runbooks/QUICKSTART.md` | First-time local setup (env vars, Neo4j, virtualenv) |
| `docs/runbooks/GraphRAG_Data_Pipeline_Guide.md` | If you need to ingest / reset KG data |
| `docs/product/REPO_REORG_MIGRATION_GUIDE.md` | The `src/aix/*` layout cheat sheet — old→new path mappings |
| `docs/architecture/Agent_Domain_Prompt_Integration.md` | The 3 options for connecting domain prompts; we picked Option 2 |
| `docs/api/Explainability_API_Guide_for_Frontend.md` | Pre-existing explainability endpoints |
| `docs/reports/FUTURE_FIXES.md`, `docs/reports/NEXT_SESSION.md` | Backlog notes |

---

## 2. Changeset map — every uncommitted file, mapped to its CORE# and the doc that explains it

This is the output of `git status --short`, annotated. **All of this is staged for the next commit / PR.**

### 2.1 Modified files (16)

| File | CORE# | What changed | Read this doc to understand it |
|---|---|---|---|
| `.env.example` | CORE 2 #6.6, #7, #20 | Added `WEBUI_AUTH_SECRET`, `WEBUI_CORS_ALLOW_ORIGINS`, `AIX_MCP_REQUIRE_AUTH`, `AIX_TEST_USER_EMAIL/PASSWORD` | `MCP_Setup.md` § "Production deployment notes" |
| `.gitignore` | (housekeeping) | Excluded `data/diagnostic/chat_input_rendered.html`, ad-hoc artifacts | — |
| `apps/streamlit/main.py` | CORE 2 #6.5 | Streamlit retirement banner pointing teachers at `/webui/` | ClickUp #6.5 |
| `docs/product/ClickUp_Agentic_GraphRAG_Update.md` | (the tracker itself) | Updated phases #6.5/#6.6/#7/#20 to ✅, added "Last Updated" stamps after each phase | (this is the doc) |
| `requirements.txt` | CORE 2 #6.6, #7, #20 | Pinned: `fastapi-users[sqlalchemy]`, `aiosqlite`, `jinja2`, `python-multipart`, `pypdf`, `sse-starlette`, `fastmcp>=3.0.0,<4.0.0` | `MCP_Setup.md` + ClickUp #6.6 / #7 |
| `src/aix/agent/agents/writer_agent.py` | CORE 2 #6.6 P3 | Wires `teacher_provided_context` into the writer prompt when an attachment is present | ClickUp #6.6 P3 |
| `src/aix/agent/graph/lesson_planner_graph.py` | CORE 2 #6.6 | Passes the new context fields through the LangGraph nodes | ClickUp #6.6 |
| `src/aix/agent/graph/nodes.py` | CORE 2 #6.6 | Granular `StreamEvent`s (planner → retriever → writer → critic phase boundaries) | ClickUp #6.6 P2 |
| `src/aix/agent/graph/state.py` | CORE 2 #6.6 P3 | New optional field `teacher_provided_context: str` on `AgentState` | ClickUp #6.6 P3 |
| `src/aix/agent/orchestrator.py` | CORE 2 #6.6 | Streams the granular `StreamEvent`s to the UI | ClickUp #6.6 P2 |
| `src/aix/agent/prompts/writer_prompt.py` | CORE 2 #6.6 P3 | Prompt template now includes `{teacher_provided_context}` block when truthy | ClickUp #6.6 P3 |
| `src/aix/api/main.py` | CORE 2 #6.6, #7, #20 | Mounts `/webui/*`, `/api/v1/agent/*`, `/auth/jwt/*` (Bearer), `/mcp/` (lifespan-combined). Adds env-driven CORS. Fixed `sys.path` to point at `src/` (not `src/aix/`) — see "Lessons learned" in ClickUp #20. | ClickUp #6.6/#7/#20 + `MCP_Setup.md` |
| `src/aix/api/routes/__init__.py` | CORE 2 #7 | Re-exports `agent_router` | ClickUp #7 |
| `src/aix/api/routes/context.py` | CORE 2 #7 | Docstring update — removed the "forthcoming `/api/v1/agent`" note (now landed) | ClickUp #7 |
| `src/aix/api/schemas/__init__.py` | CORE 2 #7 | Re-exports the new agent schemas | ClickUp #7 |
| `src/aix/api/schemas/models.py` | CORE 2 #7 | Minor consolidation — no behaviour change | ClickUp #7 |

### 2.2 New files / new directories (untracked)

#### Frontend / webui (CORE 2 #6.6 P0–P3)

| Path | What it is |
|---|---|
| `src/aix/webui/__init__.py` | Package marker |
| `src/aix/webui/db.py` | aiosqlite + SQLAlchemy async setup |
| `src/aix/webui/routes.py` | Top-level `/webui/*` routes (home, redirect to lesson) |
| `src/aix/webui/auth/*.py` (5 files) | `fastapi-users` integration: cookie + Bearer transports, JWT strategy (HS256, audience `fastapi-users:auth`), shared `WEBUI_AUTH_SECRET`. **Critical**: this is the same auth surface the public API and `/mcp/` reuse — one login, three surfaces. |
| `src/aix/webui/lessons/*.py` | Lesson model + routes + schemas + `uploads.py` (PDF text extraction via `pypdf` → injected as `teacher_provided_context`, **not** ingested into the KG) |
| `src/aix/webui/agent/service.py` | **Two helpers**: `run_agent_stream` (webui — DB-persists results) and `stream_agent_events` (DB-less — used by the public API and the MCP `agent.run_lesson_plan` tool). Same agent pipeline; different persistence boundary. |
| `src/aix/webui/templates/_base.html` | Shared layout (Tailwind via CDN, WebAwesome 3.x, htmx 2, Alpine.js) |
| `src/aix/webui/templates/pages/*.html` (4) | `home.html`, `auth_login.html`, `auth_register.html`, `lesson_new.html`, `lesson_show.html` (the chat workspace) |
| `src/aix/webui/templates/partials/*.html` (13) | htmx fragments — chat cards (planner / retriever / writer / critic / lesson / error), input form, profile sidebar (view + edit), media panel, attachments, navbar |

→ Read: `Frontend_Platform_Evaluation.md` (decision rationale) + ClickUp #6.6 P0/P1/P2/P3 (each phase's acceptance criteria).

#### Public Agent API (CORE 2 #7)

| Path | What it is |
|---|---|
| `src/aix/api/routes/agent.py` | `POST /api/v1/agent/run` (returns the full `AgentRunResponse`) + `POST /api/v1/agent/stream` (SSE). Both use `_AGENT_REQUEST_OPENAPI_EXAMPLES` for the Swagger Minimal/Rich dropdown. |
| `src/aix/api/schemas/agent.py` | `AgentRunRequest`, `AgentRunResponse`, `AgentStreamEvent` Pydantic schemas. The request example is enriched (Rich profile shape — mirrors what `/api/v1/context` shows). |
| `src/aix/api/schemas/educational_profile.py` | Shared profile schemas used by both webui and the public API |
| `tests/api/__init__.py`, `tests/api/test_agent_routes.py` | Integration tests for `/api/v1/agent/run` and `/api/v1/agent/stream` (mocked `stream_agent_events`) |

→ Read: ClickUp #7 + visit `http://127.0.0.1:8765/docs` once the server is up — the Swagger page is self-documenting.

#### MCP Tool Servers (CORE 5 #20)

| Path | What it is |
|---|---|
| `src/aix/mcp/__init__.py`, `server.py` | Composition root — `build_mcp_server()` (idempotent) registers all tools/resources/prompts |
| `src/aix/mcp/stdio_main.py` | Local entry point for Claude Desktop / Cursor IDE — logs to stderr only |
| `src/aix/mcp/http_app.py` | The Streamable HTTP factory; sets `JWTVerifier` (HS256, audience `fastapi-users:auth`, secret = `WEBUI_AUTH_SECRET`) |
| `src/aix/mcp/tools/kg_tools.py` | 4 tools: `kg.search`, `kg.get_context`, `kg.list_concepts`, `kg.get_schema` |
| `src/aix/mcp/tools/media_tools.py` | 5 tools: `media.lookup_curated`, `media.search_youtube`, `media.search_academic`, `media.search_oer`, `media.generate_diagram` |
| `src/aix/mcp/tools/agent_tools.py` | 1 tool: `agent.run_lesson_plan` (wraps `stream_agent_events`, emits MCP progress notifications) |
| `src/aix/mcp/resources/kg_resources.py` | 4 resources: `kg://schema`, `kg://concepts/{domain}`, `methodology://list`, `media://stats` |
| `src/aix/mcp/prompts/educational_prompts.py` | 2 prompts: `educational-query`, `lesson-plan-request` |
| `tests/mcp_server/conftest.py` + 5 test files | 19 regression tests — surface inventory, JWT auth gate, KG tool shapes, agent-tool contract (mocked), OpenAPI strictly-additive guard |

→ Read: `MCP_Setup.md` (operational) + ClickUp #20 § "Key engineering notes / lessons learned" (all the FastMCP 3.x quirks we found and fixed — circular import, prompt arg-string requirement, `tests/mcp/` shadowing, etc.).

#### Diagnostics & baselines

| Path | What it is | When to use it |
|---|---|---|
| `scripts/diagnostic/mcp_smoke.py` | The Swiss-army knife. Lists tools/resources/prompts, calls any tool, has `--phase2-verify` / `--phase3-verify` / `--phase4-verify` / `--phase5-verify` modes for end-to-end checks | Daily — when working on MCP |
| `scripts/diagnostic/probe_mcp_endpoint.py` | Credentials-free probe: hits `/health`, `/docs`, `POST /mcp/` (expects 401) | Smoke after restarting uvicorn |
| `scripts/diagnostic/inspect_mcp_mount.py` | Verifies `/mcp/` is mounted in the FastAPI app | When `/mcp/` returns 404 |
| `scripts/diagnostic/list_openapi_paths.py` | Dumps live OpenAPI paths to stdout | Quick visual diff |
| `scripts/diagnostic/capture_openapi_baseline.py` | Re-snapshots the OpenAPI baseline file | Before a planned API change |
| `scripts/diagnostic/inspect_chat_input.py` | Server-side fetch + parse of the chat-input partial | Used to debug the paperclip-icon regression in P3 |
| `data/diagnostic/openapi_before_p7.txt`, `openapi_after_p7.txt` | OpenAPI snapshots that prove #7 was strictly additive | Reference; locked into the regression suite |
| `data/diagnostic/openapi_before_p20.txt` | OpenAPI snapshot that proves #20 was strictly additive (`/mcp/` is a Starlette mount, not in the spec) | Reference; locked into `tests/mcp_server/test_mcp_openapi_regression.py` |

#### Newly added docs

| Path | Purpose |
|---|---|
| `docs/architecture/Frontend_Platform_Evaluation.md` | "Why FastAPI + htmx" decision doc (P0 of #6.6) |
| `docs/architecture/mockup front end.png` | UI mockup that drove the P2 chat workspace design |
| `docs/integrations/MCP_Setup.md` | Canonical MCP onboarding (Phases 1–5 surface, Phase 6 regression note, Phase 7 deployment + live-smoke follow-up) |
| `docs/integrations/Screenshot 2026-04-26 233735.png` | "What Phase 6 IS / IS NOT" reference image |

---

## 3. Quick start (15 minutes from a fresh clone)

Once this branch is merged or pulled:

```powershell
# 1. Install deps
cd C:\Users\louis\KBRAGold\graphaixlearning
& C:\Users\louis\KBRAGold\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Copy .env.example → .env and fill in:
#    NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD
#    OPENROUTER_API_KEY (or OPENAI_API_KEY)
#    WEBUI_AUTH_SECRET (≥ 32 random bytes — generate with: python -c "import secrets; print(secrets.token_hex(32))")
#    YOUTUBE_API_KEY (optional — falls back to a search URL)

# 3. Start the API + webui + MCP HTTP mount (single process)
uvicorn aix.api.main:app --port 8765 --app-dir src --reload
```

Then in a separate shell:

```powershell
# Run the full regression suite (should take ~60-90s, all green)
python -m pytest tests/api tests/mcp_server -v --tb=short

# Smoke the MCP server in-process (no uvicorn needed)
python scripts/diagnostic/mcp_smoke.py
python scripts/diagnostic/mcp_smoke.py --phase2-verify
python scripts/diagnostic/mcp_smoke.py --phase3-verify

# Smoke the MCP HTTP transport (uvicorn must be running on 8765)
python scripts/diagnostic/probe_mcp_endpoint.py
python scripts/diagnostic/mcp_smoke.py --phase5-verify --email "<your-email>" --password "<your-password>"
```

Open in browser:

| URL | What you'll see |
|---|---|
| `http://127.0.0.1:8765/docs` | Swagger UI — both `/api/v1/context` (existing GraphRAG) and `/api/v1/agent/run` + `/api/v1/agent/stream` (new) with Minimal/Rich example dropdowns |
| `http://127.0.0.1:8765/webui/` | The teacher webui — register a user, fill out the educational profile, generate a lesson |
| `http://127.0.0.1:8765/openapi.json` | Raw OpenAPI spec — diff vs. `data/diagnostic/openapi_before_p20.txt` should show only additive changes |

---

## 4. The regression safety net (what's locked in)

| Test file | What it locks |
|---|---|
| `tests/api/test_agent_routes.py` | `/api/v1/agent/run` and `/api/v1/agent/stream` contract (mocked agent stream) |
| `tests/mcp_server/test_mcp_surface.py` (5) | 10 tools / 3 static + 1 templated resources / 2 prompts / server identity / instructions text |
| `tests/mcp_server/test_mcp_http_auth.py` (4) | 401 unauth, 401 wrong-secret, 2xx valid JWT, `/api/v1/health` stays public |
| `tests/mcp_server/test_mcp_kg_tools.py` (5) | `kg.list_concepts` + `kg.get_schema` shapes for both domains, `limit` validation |
| `tests/mcp_server/test_mcp_agent_tool_contract.py` (3) | `agent.run_lesson_plan` happy path (mocked stream), input validation, error propagation |
| `tests/mcp_server/test_mcp_openapi_regression.py` (2) | REST surface strictly additive vs. baseline; `/mcp/` not in OpenAPI |
| `tests/integration/*.py` (pre-existing, untouched) | Phase 1/2/3 media + intent/translation tests |

**Total: 19 new tests + the pre-existing test suite. All green in ~64s.**

---

## 5. Suggested commit / PR breakdown

Because these are three independent landings, I'd propose splitting the merge into **three PRs** for cleaner review:

| PR | Scope | Files |
|---|---|---|
| **PR 1 — Mirror Stack webui (#6.5 + #6.6 P0–P3)** | Streamlit retirement banner + the entire `src/aix/webui/` package + `src/aix/api/main.py` webui mount + frontend evaluation doc + mockup | `apps/streamlit/main.py`, `src/aix/webui/**`, `docs/architecture/Frontend_Platform_Evaluation.md`, `docs/architecture/mockup front end.png`, `src/aix/api/schemas/educational_profile.py`, agent state + writer prompt context-injection edits |
| **PR 2 — Public Agent API (#7)** | `/api/v1/agent/*` endpoints + JWT Bearer + tests + OpenAPI baselines | `src/aix/api/routes/agent.py`, `src/aix/api/schemas/agent.py`, `tests/api/`, `data/diagnostic/openapi_*p7*.txt`, `scripts/diagnostic/list_openapi_paths.py`, the `/api/v1/agent` mount in `main.py` |
| **PR 3 — MCP Tool Servers (#20)** | Full `src/aix/mcp/` package + tests + smoke scripts + `MCP_Setup.md` + the `/mcp/` mount in `main.py` | `src/aix/mcp/**`, `tests/mcp_server/`, `scripts/diagnostic/mcp_smoke.py`, `scripts/diagnostic/probe_mcp_endpoint.py`, `scripts/diagnostic/inspect_mcp_mount.py`, `scripts/diagnostic/capture_openapi_baseline.py`, `data/diagnostic/openapi_before_p20.txt`, `docs/integrations/**`, `requirements.txt` (`fastmcp` pin) |

The ClickUp tracker doc and `.env.example` / `.gitignore` updates can ride along in PR 1 (or be a tiny standalone PR 0).

The skill at `C:\Users\louis\.cursor\skills-cursor\split-to-prs\SKILL.md` automates this split if you'd like.

---

## 6. What's pending — Option B live integration smoke (your task)

This is the **only** thing not done. It's pure manual click-testing — **zero code changes**. The configs are already in `MCP_Setup.md`.

### 6.1 Cursor IDE smoke (~5 min)

Add to `~/.cursor/mcp.json` (user-level, not project-level — project-level was tried and is finicky on Windows):

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

Then `Cmd/Ctrl+Shift+P → Developer: Reload Window`. Open the MCP panel and confirm 10 tools listed. Ask Cursor's chat: *"List the concepts in the neuro domain using the aix-graphrag MCP server"* and verify it calls `kg.list_concepts`.

### 6.2 Claude Desktop smoke (~10 min)

Same JSON, different file: `%APPDATA%\Claude\claude_desktop_config.json`. **Fully quit** Claude Desktop (system tray → Quit), relaunch, look for the plug icon. Ask: *"Usa lo strumento `kg.search` per trovare strategie didattiche legate alla motivazione intrinseca nel dominio `neuro`."*

Full troubleshooting in `MCP_Setup.md` § "Troubleshooting" if it doesn't appear.

### 6.3 MCP Inspector smoke against the HTTP mount (~10 min)

```powershell
# In one shell (uvicorn must be running):
uvicorn aix.api.main:app --port 8765 --app-dir src

# In another shell:
npx @modelcontextprotocol/inspector
```

In the inspector UI:
- Transport: **Streamable HTTP**
- URL: `http://127.0.0.1:8765/mcp/`
- Auth: paste a JWT from `POST /auth/jwt/login` (form-encoded `username=...&password=...`)

Confirm: 10 tools listed → call `kg.list_concepts` with `{"domain":"neuro","limit":5}` → see structured response. This is the same flow Lovable / partner LangGraph cloud agents / browser MCP playgrounds will use.

### 6.4 What to update after the smoke

- Tick the three boxes in `MCP_Setup.md` § "Live integration follow-up" (or add a "Verified by Angelo on `<date>`" stamp).
- Drop a one-liner in `docs/product/ClickUp_Agentic_GraphRAG_Update.md` Subtask 20 (it already says ✅ DONE; just confirm the live smokes too).

---

## 7. Known issues & deferred work (transparency)

These were intentionally **not** fixed in this branch. They're tracked elsewhere.

| Issue | Tracked in | Why deferred |
|---|---|---|
| Planner / Critic occasionally hit `JSON parse failure` from OpenRouter empty bodies | ClickUp #11a (Agent JSON Parse Hardening) | Pre-existing agent-layer bug, not caused by anything in this branch. Agent recovers gracefully. |
| `kg.search`'s `cypher_query` field returns empty in some result shapes | Cosmetic — flagged to be a 1-line fix in `aix.agent.tools.graphrag_tool` | Not user-visible; result data is correct, only the debug field is missing |
| `media.search_youtube` falls back to a URL if `YOUTUBE_API_KEY` is unset | Documented as expected behaviour in `MCP_Setup.md` | By design — graceful degradation |
| ChatGPT Desktop's remote MCP feature requires OAuth 2.1 + PKCE; we ship JWT Bearer | "Phase 5b" (optional follow-up) in `MCP_Setup.md` | Defer until a real customer needs ChatGPT integration |
| Several deprecation warnings during pytest (Pydantic V2 in fastapi-users, authlib in fastmcp.JWTVerifier) | Upstream migrations | Wait for upstream — none affect runtime behaviour |

---

## 8. If you get stuck — a 60-second triage

| Symptom | Look here |
|---|---|
| `uvicorn` won't start, "circular import" mentioning `aix.mcp.server` | `MCP_Setup.md` § "Troubleshooting" — last entry. Verify `sys.path.insert` in `src/aix/api/main.py` points at `src/`, not `src/aix/`. |
| `/mcp/` returns 404 | Run `python scripts/diagnostic/inspect_mcp_mount.py` — confirms whether the mount survived app startup |
| `/mcp/` returns 401 even with a token | `MCP_Setup.md` § "Troubleshooting" — confirm token came from `/auth/jwt/login` (Bearer), not `/auth/login` (cookie) |
| pytest fails with `ImportError: cannot import name 'McpError' from 'mcp'` | The `tests/mcp_server/` package was renamed from `tests/mcp/` to avoid shadowing the third-party `mcp` SDK. If you see `tests/mcp/` reappearing somewhere, delete it. |
| Webui chat input has no paperclip icon | Hard refresh (`Ctrl+Shift+R`). The icon is a plain HTML `<button>` with `title="..."` — not a `<wa-tooltip>` (those misbehaved). |
| Slow cold start (~30s) on first `kg.*` call | Expected — Text2Cypher schema cache + Node2Vec model load. Subsequent calls in the same MCP session are sub-second. |

---

## 9. One-line summary for ClickUp

> Branch lands the FEM Mirror Stack frontend (#6.6 P0–P3), the public Agent JSON+SSE API (#7), and the MCP Tool Servers (#20) — all strictly additive, regression-locked by 19 new pytest cases, fully documented in `docs/integrations/MCP_Setup.md` and the master tracker. Live GUI-client smoke (Cursor IDE / Claude Desktop / MCP Inspector — ~30 min, no code) handed to Angelo for final sign-off.

---

*If anything in this doc is stale or wrong, ping me — I'd rather fix it once than have you waste an hour figuring out the mismatch.*
