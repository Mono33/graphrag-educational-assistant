"""
GraphRAG Educational API - FastAPI Application

This API provides educational context from the GraphRAG knowledge graph.
It is designed to be integrated with the FEM AixLearning agent.

Usage:
    uvicorn aix.api.main:app --reload --port 8000

Or run directly:
    python -m aix.api.main

Endpoints:
    POST /api/v1/context     - Get educational context for a query
    GET  /api/v1/context/domains - List available domains
    GET  /api/v1/health      - Health check
    GET  /docs               - Swagger UI documentation
    GET  /redoc              - ReDoc documentation
"""

import asyncio
import logging
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load the repo-local .env before any submodules read os.environ at import
# time (auth manager, Sentry, CORS, MCP auth, etc.).
_REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_REPO_ROOT / ".env")

# Add the project source root (``src/``) to sys.path so ``aix.api``,
# ``aix.mcp``, ``aix.webui`` … all resolve via their canonical names when
# the file is invoked directly (``python src/aix/api/main.py``).
#
# CRITICAL: we deliberately insert ``src/`` (parent of ``aix``), NOT
# ``src/aix``. Adding ``src/aix`` would make our internal ``aix.mcp``
# package ALSO importable as plain ``mcp``, which collides with the
# official Anthropic ``mcp`` SDK (a transitive dep of ``fastmcp``).
# When ``fastmcp.utilities.logging`` does ``import mcp`` it would land
# on our package, triggering a circular import while ``aix.mcp.server``
# is mid-loading. See CORE 5 #20 Phase 5 for the bug we hit and fixed.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aix.api import __version__
from aix.api.routes import context_router
from aix.api.schemas import HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Phase B (#37/#39 readiness) — Windows + Postgres event-loop guard.
# psycopg's async driver (which backs LangGraph's AsyncPostgresSaver) requires
# the Selector event loop; Windows' default ProactorEventLoop raises at connect
# time. We switch the policy ONLY when (a) we're on Windows AND (b) a Postgres
# backend is actually configured — so the existing SQLite dev path keeps the
# default Proactor loop completely untouched. No-op on Linux/prod, where the
# default loop already supports psycopg async. Must run at import time, before
# uvicorn creates the event loop.
def _maybe_set_windows_selector_loop() -> None:
    if sys.platform != "win32":
        return
    _pg_schemes = ("postgres://", "postgresql://", "postgres+", "postgresql+")
    pg_configured = any(
        (os.getenv(var) or "").startswith(_pg_schemes)
        for var in (
            "LANGGRAPH_DATABASE_URL",
            "LANGGRAPH_CHECKPOINTER_URL",
            "WEBUI_DATABASE_URL",
        )
    )
    if not pg_configured:
        return
    selector_policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if selector_policy is None:  # pragma: no cover - non-Windows safety net
        return
    if isinstance(asyncio.get_event_loop_policy(), selector_policy):
        return
    asyncio.set_event_loop_policy(selector_policy())
    logger.info(
        "🪟 Windows + Postgres detected → WindowsSelectorEventLoopPolicy active "
        "(required by psycopg async; no-op on Linux/prod)."
    )


_maybe_set_windows_selector_loop()


# GlitchTip / Sentry error monitoring
import sentry_sdk

_sentry_dsn = os.getenv("SENTRY_DSN", "")
if _sentry_dsn:
    sentry_sdk.init(
        dsn=_sentry_dsn,
        traces_sample_rate=0.2,
        release=__version__,
        environment=os.getenv("ENVIRONMENT", "production"),
    )
    logger.info("✅ GlitchTip error monitoring enabled")
else:
    logger.info("ℹ️ GlitchTip disabled (no SENTRY_DSN configured)")


def _warm_schema(domain: str) -> None:
    """Pre-populate Text2CypherConverter schema cache for a domain.

    Runs in a thread-pool executor so it doesn't block the event loop.
    After this call, the first real request for the domain skips the 60+
    Neo4j schema-extraction queries and hits the cache directly.
    """
    try:
        from aix.retrieval.multilingual_text2cypher import MultilingualText2Cypher
        from aix.retrieval.text2cypher import Text2CypherConverter

        if domain in Text2CypherConverter._schema_cache:
            logger.info(f"ℹ️ Schema cache already warm for domain='{domain}' (skip)")
            return

        t = MultilingualText2Cypher()
        schema = t.pipeline.converter.schema_extractor.extract_schema(domain=domain)
        Text2CypherConverter._schema_cache[domain] = schema
        t.pipeline.converter.schema_info = schema
        Text2CypherConverter._prompt_cache[domain] = t.pipeline.converter._create_prompt_template(
            domain=domain
        )
        logger.info(f"✅ Schema cache warmed for domain='{domain}'")
    except Exception as e:
        logger.warning(f"⚠️ Schema cache warm-up failed for domain='{domain}': {e}")


# CORE 5 #20 Phase 5 — Build the MCP Streamable HTTP sub-app once, at import
# time. ``build_mcp_http_app()`` is idempotent and side-effect-free at the
# Neo4j level (tool/resource/prompt registration only); the actual MCP
# session manager starts later inside ``lifespan`` via
# ``mcp_app.lifespan(app)``. We assign to a module-level variable so both
# ``lifespan`` (closure lookup) and the ``app.mount(...)`` call below see
# the same instance. Wrapped in try/except so an MCP build failure cannot
# block the public /api/v1 surface from coming up.
_mcp_http_app = None
try:
    from aix.mcp.http_app import MCP_MOUNT_PATH, build_mcp_http_app

    _mcp_http_app = build_mcp_http_app()
    if _mcp_http_app is None:
        logger.info("ℹ️ MCP HTTP app not built (build_mcp_http_app returned None)")
except Exception as exc:  # noqa: BLE001 - any import/build failure must be soft
    logger.warning("⚠️ MCP HTTP app could not be loaded: %s", exc, exc_info=True)
    MCP_MOUNT_PATH = "/mcp"  # noqa: F811 - default path for the log line below


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting GraphRAG Educational API...")
    logger.info(f"📦 Version: {__version__}")

    # Test Neo4j connection
    try:
        from neo4j import GraphDatabase

        from aix.core.config import config

        driver_kwargs = {
            "auth": (config.neo4j.user, config.neo4j.password),
        }
        if not config.neo4j.encrypted:
            driver_kwargs["encrypted"] = config.neo4j.encrypted
        driver = GraphDatabase.driver(config.neo4j.uri, **driver_kwargs)
        driver.verify_connectivity()
        driver.close()
        logger.info("✅ Neo4j connection verified")
    except Exception as e:
        logger.warning(f"⚠️ Neo4j connection check failed: {e}")

    # Check domain configs
    try:
        from aix.domains import get_domain_config

        neuro_config = get_domain_config("neuro")
        udl_config = get_domain_config("udl")
        loaded = []
        if neuro_config:
            loaded.append("neuro")
        if udl_config:
            loaded.append("udl")
        logger.info(f"✅ Domain configs loaded: {loaded}")
    except Exception as e:
        logger.warning(f"⚠️ Domain config check failed: {e}")

    # Warm Text2Cypher schema cache in background threads so the first real
    # request doesn't pay the 60+ Neo4j schema-extraction queries.
    try:
        loop = asyncio.get_event_loop()
        for _domain in ["udl", "neuro"]:
            loop.run_in_executor(None, _warm_schema, _domain)
        logger.info("🔥 Schema cache warm-up started for: udl, neuro")
    except Exception as e:
        logger.warning(f"⚠️ Schema cache warm-up scheduling failed: {e}")

    # Pre-warm GraphRAGTool for both domains so the first lesson request
    # doesn't pay the lazy-init cost (EnhancedMultilingualText2Cypher import
    # + Node2Vec model load + Neo4j driver creation ≈ 2-5 s).
    def _warm_graphrag_tool(domain: str) -> None:
        try:
            from aix.agent.tools.graphrag_tool import GraphRAGTool

            GraphRAGTool(domain=domain)._ensure_initialized()
            logger.info(f"🔥 GraphRAGTool warm for domain='{domain}'")
        except Exception as e:
            logger.warning(f"⚠️ GraphRAGTool warm-up failed for domain='{domain}': {e}")

    try:
        loop = asyncio.get_event_loop()
        for _domain in ["neuro", "udl"]:
            loop.run_in_executor(None, _warm_graphrag_tool, _domain)
        logger.info("🔥 GraphRAGTool warm-up started for: neuro, udl")
    except Exception as e:
        logger.warning(f"⚠️ GraphRAGTool warm-up scheduling failed: {e}")

    # Path C webui — initialise the SQLite (or Postgres in CORE 6) schema
    # used by /webui auth + lessons. Idempotent: create_all is a no-op if
    # tables already exist. Wrapped so a webui DB problem never blocks the
    # public /api/v1 surface from coming up.
    try:
        from aix.webui.db import init_db as _webui_init_db

        await _webui_init_db()
    except Exception as e:
        logger.warning(f"⚠️ WebUI DB init failed (auth + lessons disabled): {e}")

    # CORE 2 #11a follow-up — LLM connectivity probe.
    # Issues a one-shot ``GET /models`` against the configured LLM
    # endpoint and emits a single, actionable log line that distinguishes
    # TLS failures, auth failures, and outages from each other (the
    # OpenAI SDK collapses all three into 'Connection error.' otherwise).
    # Default-on; flip ``AIX_LLM_PROBE_ENABLED=false`` to disable. Runs in
    # the background — never blocks startup, never fails the API.
    try:
        from aix.core.connectivity_probe import schedule_startup_probe

        await schedule_startup_probe()
    except Exception as e:  # noqa: BLE001
        logger.debug("⚠️ LLM connectivity probe scheduling failed: %s", e)

    # CORE 5 #20 Phase 5 — MCP Streamable HTTP sub-app lifespan.
    # FastMCP's http_app() Starlette app initialises an internal
    # ``StreamableHTTPSessionManager`` inside its own lifespan. If we don't
    # enter that lifespan from the parent app, the very first request to
    # ``/mcp/`` raises "Task group is not initialized". We use an
    # AsyncExitStack so that any failure in MCP startup never prevents the
    # main API from coming up.
    async with AsyncExitStack() as _mcp_stack:
        if _mcp_http_app is not None:
            try:
                await _mcp_stack.enter_async_context(_mcp_http_app.lifespan(app))
                logger.info("✅ MCP HTTP sub-app lifespan started (mounted at /mcp/)")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "⚠️ MCP HTTP sub-app lifespan failed to start (skipping): %s",
                    exc,
                )

        logger.info("✅ API ready to serve requests")

        yield  # Server is running

    # Shutdown
    logger.info("👋 Shutting down GraphRAG Educational API...")

    # Phase B readiness — release the LangGraph checkpointer cleanly so the
    # Postgres connection pool (prod) / SQLite connection (dev) is torn down on
    # graceful shutdown instead of being abandoned. Best-effort and idempotent:
    # close_checkpointer() is a no-op if the checkpointer was never initialised
    # (e.g. no lesson run happened), and never raises out of this block.
    try:
        from aix.agent.graph.checkpointer import close_checkpointer

        await close_checkpointer()
    except Exception as e:  # noqa: BLE001
        logger.warning("⚠️ Checkpointer shutdown failed: %s", e)


# Create FastAPI app
app = FastAPI(
    title="GraphRAG Educational API",
    description="""
## 🎓 GraphRAG Educational Knowledge Graph API

This API provides educational context from a neuroscience and pedagogy knowledge graph.

### Purpose
- Provide structured educational context for the FEM AixLearning agent
- Return methodologies, evidence, and recommendations based on teacher queries

### Integration
DEV team can call `POST /api/v1/context` with a query and receive structured data
ready to inject into their prompt templates.

### Response Structure
The response includes:
- **query_info**: Translation and Cypher query details
- **context**: Structured data with methodologies, evidence, confidence levels
- **formatted_prompt_section**: Pre-formatted text ready for prompt injection
- **metrics**: Retrieval statistics

### Example Usage
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/context",
    json={
        "query": "Quali strategie per studenti con ADHD?",
        "domain": "neuro",
        "language": "it"
    }
)

context = response.json()
# Use context["formatted_prompt_section"] directly in prompts
# Or access context["context"]["primary_methodologies"] for structured data
```
    """,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — env-driven allow-list (CORE 2 #7).
#
# Backward-compat: ``WEBUI_CORS_ALLOW_ORIGINS`` defaults to ``*`` so existing
# .env files keep working unchanged. To tighten in deploy, set e.g.
#    WEBUI_CORS_ALLOW_ORIGINS=https://aixlearning.it,https://app.aixlearning.it
# Comma-separated. We also support a single ``*`` for "allow any origin"
# explicitly, since the FastAPI middleware treats ``["*"]`` differently
# from a populated allow-list.
# ---------------------------------------------------------------------------
_cors_raw = os.getenv("WEBUI_CORS_ALLOW_ORIGINS", "*").strip()
if _cors_raw == "*" or not _cors_raw:
    _cors_origins: list[str] = ["*"]
else:
    _cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("✅ CORS configured: origins=%s", _cors_origins)

# Include routers
app.include_router(context_router, prefix="/api/v1")

# Public Agent API (CORE 2 #7) — JSON+SSE contract for the multi-agent
# pipeline. Wrapped in try/except so an import failure here can never
# break the existing /api/v1/context surface from coming up.
try:
    from aix.api.routes import agent_router

    app.include_router(agent_router, prefix="/api/v1")
    logger.info("✅ Agent JSON+SSE API mounted at /api/v1/agent/*")
except Exception as exc:  # noqa: BLE001
    logger.warning("⚠️ Agent JSON+SSE API not loaded (skipping): %s", exc)

# Path C webui — internal HTML+SSE surface for agent end-to-end testing.
# Mounts at /webui/* (not /api/v1/*), so the public JSON contract used by the
# DEV team is unaffected. See docs/architecture/Frontend_Platform_Evaluation.md
# (ADR-0001) and CORE 2 subtask #6.6 in ClickUp_Agentic_GraphRAG_Update.md.
try:
    from aix.webui import router as webui_router

    app.include_router(webui_router)
    logger.info("✅ Path C webui mounted at /webui/")
except ImportError as exc:
    logger.warning(f"⚠️ Path C webui not loaded (skipping): {exc}")

# CORE 2 #6.6 P5 (brand refresh) — serve the WebUI's static assets at /static/.
# This carries the warm-academic brand CSS (aix-brand.css) and any future
# locally-served fonts / SVG icons. Lives under aix.webui.static so the
# package keeps owning its own assets; the mount path /static is conventional
# and unused elsewhere in the API. Wrapped in try/except like every other
# optional mount so a missing folder cannot block the public /api/v1 surface.
try:
    from fastapi.staticfiles import StaticFiles

    _WEBUI_STATIC_DIR = Path(__file__).resolve().parents[1] / "webui" / "static"
    if _WEBUI_STATIC_DIR.is_dir():
        app.mount(
            "/static",
            StaticFiles(directory=str(_WEBUI_STATIC_DIR)),
            name="static",
        )
        logger.info("✅ WebUI static mounted at /static (from %s)", _WEBUI_STATIC_DIR)
    else:
        logger.info(
            "ℹ️ WebUI static directory not present (skipping /static mount): %s",
            _WEBUI_STATIC_DIR,
        )
except Exception as exc:  # noqa: BLE001
    logger.warning("⚠️ /static mount failed (skipping): %s", exc)

# JSON Bearer auth router (CORE 2 #7).
#
# Mounts ``POST /auth/jwt/login`` and ``POST /auth/jwt/logout``. These are
# **separate from** the existing HTML cookie endpoints at /auth/login etc.
# (different paths, no collision). Their job is to give the public JSON
# API a way to mint Bearer tokens, which Swagger UI's "Authorize" dialog
# can then send on every /api/v1/agent/* call so /docs becomes a live
# test bench for the agent — same UX you used for /api/v1/context.
#
# Wrapped in try/except so an import failure can never block the existing
# /webui/* and /api/v1/context surfaces.
try:
    from aix.webui.auth import bearer_backend, fastapi_users

    app.include_router(
        fastapi_users.get_auth_router(bearer_backend),
        prefix="/auth/jwt",
        tags=["api-auth"],
    )
    logger.info("✅ JSON Bearer auth router mounted at /auth/jwt/*")
except Exception as exc:  # noqa: BLE001
    logger.warning("⚠️ JSON Bearer auth router not loaded (skipping): %s", exc)


# CORE 5 #20 Phase 5 — Mount the Streamable HTTP MCP endpoint at /mcp/.
#
# Same JWT Bearer secret as /api/v1/agent/* (see aix.mcp.http_app for the
# JWTVerifier configuration), so a teacher can POST /auth/jwt/login once and
# use the resulting token against /api/v1/agent/run AND /mcp/ interchange-
# ably. To temporarily disable auth for local dev (e.g. while testing with
# the MCP Inspector without minting a token first), set
# ``AIX_MCP_REQUIRE_AUTH=0`` in .env — NEVER in production.
#
# We mount unconditionally only when ``_mcp_http_app`` was built above; if
# it wasn't (e.g. fastmcp not installed in this environment), we just skip
# silently — the rest of the API surface is unaffected.
if _mcp_http_app is not None:
    try:
        app.mount(MCP_MOUNT_PATH, _mcp_http_app)
        logger.info(
            "✅ MCP Streamable HTTP endpoint mounted at %s/ (Phase 5)",
            MCP_MOUNT_PATH,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("⚠️ MCP HTTP endpoint mount failed (skipping): %s", exc)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "GraphRAG Educational API",
        "version": __version__,
        "description": "Knowledge graph context for educational queries",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """
    Health check endpoint

    Returns the status of the API and its dependencies
    """
    neo4j_connected = False
    domain_configs = []

    # Check Neo4j
    try:
        from neo4j import GraphDatabase

        from aix.core.config import config

        driver = GraphDatabase.driver(
            config.neo4j.uri, auth=(config.neo4j.user, config.neo4j.password)
        )
        driver.verify_connectivity()
        driver.close()
        neo4j_connected = True
    except Exception:
        pass

    # Check domain configs
    try:
        from aix.domains import get_domain_config

        if get_domain_config("neuro"):
            domain_configs.append("neuro")
        if get_domain_config("udl"):
            domain_configs.append("udl")
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if neo4j_connected else "degraded",
        neo4j_connected=neo4j_connected,
        version=__version__,
        domain_configs_loaded=domain_configs,
    )


# Run with: python -m api.main
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
