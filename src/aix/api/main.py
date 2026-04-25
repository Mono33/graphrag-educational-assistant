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
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from aix.api import __version__
from aix.api.routes import context_router
from aix.api.schemas import HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        Text2CypherConverter._prompt_cache[domain] = t.pipeline.converter._create_prompt_template(domain=domain)
        logger.info(f"✅ Schema cache warmed for domain='{domain}'")
    except Exception as e:
        logger.warning(f"⚠️ Schema cache warm-up failed for domain='{domain}': {e}")


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting GraphRAG Educational API...")
    logger.info(f"📦 Version: {__version__}")
    
    # Test Neo4j connection
    try:
        from aix.core.config import config
        from neo4j import GraphDatabase
        driver_kwargs = {
            "auth": (config.neo4j.user, config.neo4j.password),
        }
        if not config.neo4j.encrypted:
            driver_kwargs["encrypted"] = config.neo4j.encrypted
        driver = GraphDatabase.driver(
            config.neo4j.uri,
            **driver_kwargs
        )
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

    logger.info("✅ API ready to serve requests")

    yield  # Server is running
    
    # Shutdown
    logger.info("👋 Shutting down GraphRAG Educational API...")


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
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(context_router, prefix="/api/v1")


@app.get("/", tags=["root"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "GraphRAG Educational API",
        "version": __version__,
        "description": "Knowledge graph context for educational queries",
        "docs": "/docs",
        "health": "/api/v1/health"
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
        from aix.core.config import config
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            config.neo4j.uri,
            auth=(config.neo4j.user, config.neo4j.password)
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
        domain_configs_loaded=domain_configs
    )


# Run with: python -m api.main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

