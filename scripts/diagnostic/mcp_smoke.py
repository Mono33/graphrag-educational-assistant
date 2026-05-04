"""
MCP smoke script — CORE 5 #20.

Modes
-----

* Default ("listing" mode): build the FastMCP server in-process and dump
  the tools / resources / prompts it advertises. Cheap (no Neo4j round-trip).
  Catches registration regressions instantly.

* ``--call kg.search --query "..."`` ("tool call" mode): invoke a single
  tool through the same in-memory server (no stdio, no HTTP) and pretty-print
  the JSON result. Useful for end-to-end validation that the wrapper code
  in ``aix.mcp.tools.kg_tools`` correctly translates ``GraphRAGTool``
  output into the public Pydantic schema.

* ``--read-resource kg://schema`` ("resource read" mode, Phase 2): read a
  resource by URI through ``mcp.read_resource(...)`` and dump its JSON
  payload. Templates accept the ``{var}`` segment as part of the URI
  (e.g. ``kg://concepts/neuro``).

* ``--render-prompt educational-query --topic "fotosintesi"`` ("prompt
  render" mode, Phase 2): call ``mcp.get_prompt(name, arguments)`` and
  pretty-print the rendered MCP message envelope.

* ``--phase3-verify`` (Phase 3): one-shot end-to-end smoke for the
  curated media lookup. **Offline only** — does NOT hit YouTube /
  Semantic Scholar / OpenAI to keep CI runs deterministic and quota-free.
  The four live-API tools (search_youtube, search_academic, search_oer,
  generate_diagram) can still be exercised one-off via ``--call``.

* ``--phase5-verify`` (Phase 5, requires running uvicorn): smoke-test the
  Streamable HTTP mount at ``/mcp/``. Confirms 401 without a token, then
  logs in via ``POST /auth/jwt/login`` and hits the same endpoint with a
  Bearer token to list tools and call ``kg.list_concepts`` end-to-end.
  Set ``AIX_TEST_USER_EMAIL`` / ``AIX_TEST_USER_PASSWORD`` (or pass
  ``--email`` / ``--password``) to point at a real account.

Examples (run from repo root)::

    python scripts/diagnostic/mcp_smoke.py
    python scripts/diagnostic/mcp_smoke.py --call kg.list_concepts --domain neuro
    python scripts/diagnostic/mcp_smoke.py --call kg.search --query "ADHD"
    python scripts/diagnostic/mcp_smoke.py --read-resource kg://schema
    python scripts/diagnostic/mcp_smoke.py --read-resource kg://concepts/neuro
    python scripts/diagnostic/mcp_smoke.py --read-resource methodology://list
    python scripts/diagnostic/mcp_smoke.py --read-resource media://stats
    python scripts/diagnostic/mcp_smoke.py --render-prompt educational-query --topic "fotosintesi"
    python scripts/diagnostic/mcp_smoke.py --render-prompt lesson-plan-request --topic "respirazione cellulare" --duration 45
    python scripts/diagnostic/mcp_smoke.py --phase3-verify
    python scripts/diagnostic/mcp_smoke.py --call media.lookup_curated --concepts "Selective Attention,Cognitive Control" --domain neuro
    python scripts/diagnostic/mcp_smoke.py --call media.search_youtube --query "metacognition" --max-results 3
    python scripts/diagnostic/mcp_smoke.py --call media.search_academic --query "spaced retrieval" --max-results 3 --year-from 2018
    python scripts/diagnostic/mcp_smoke.py --call media.search_oer --query "cognitive psychology" --max-results 3
    python scripts/diagnostic/mcp_smoke.py --call media.generate_diagram --concept "Working Memory" --diagram-type mindmap
    python scripts/diagnostic/mcp_smoke.py --phase4-verify
    python scripts/diagnostic/mcp_smoke.py --call agent.run_lesson_plan --query "Crea una lezione sull'attenzione" --domain neuro --max-revisions 0
    python scripts/diagnostic/mcp_smoke.py --phase5-verify --base-url http://127.0.0.1:8765 --email teacher@aix.it --password ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _ensure_pythonpath() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _configure_logging() -> None:
    level_name = os.environ.get("AIX_MCP_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def _print_listing() -> None:
    from aix.mcp import build_mcp_server

    mcp = build_mcp_server()

    print("=" * 72)
    print(f" Aix MCP server — '{mcp.name}' v{mcp.version}")
    print("=" * 72)

    tools = await mcp.list_tools()
    print(f"\nTools registered: {len(tools)}")
    for tool in tools:
        desc = (tool.description or "").splitlines()[0] if tool.description else ""
        if len(desc) > 100:
            desc = desc[:97] + "..."
        line = f"  - {tool.name:<22} | {desc}"
        print(line.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8", errors="replace"))

    try:
        resources = await mcp.list_resources()
        print(f"\nResources registered (static): {len(resources)}")
        for res in resources:
            print(f"  - {res.uri}")
    except Exception as exc:
        print(f"\nResources: unavailable ({exc})")

    try:
        templates = await mcp.list_resource_templates()
        print(f"\nResource templates registered: {len(templates)}")
        for tpl in templates:
            uri = getattr(tpl, "uriTemplate", None) or getattr(tpl, "uri_template", None) or getattr(tpl, "uri", "?")
            print(f"  - {uri}")
    except Exception as exc:
        print(f"\nResource templates: unavailable ({exc})")

    try:
        prompts = await mcp.list_prompts()
        print(f"\nPrompts registered: {len(prompts)}")
        for p in prompts:
            arg_names = [getattr(a, "name", "?") for a in (getattr(p, "arguments", None) or [])]
            arg_label = f" args=[{', '.join(arg_names)}]" if arg_names else ""
            print(f"  - {p.name}{arg_label}")
    except Exception as exc:
        print(f"\nPrompts: unavailable ({exc})")

    print()


async def _call_tool(name: str, arguments: Dict[str, Any]) -> None:
    from aix.mcp import build_mcp_server

    mcp = build_mcp_server()

    print(f"--> Calling tool: {name}")
    print(f"--> Arguments: {json.dumps(arguments, ensure_ascii=False)}")
    print("-" * 72)

    result = await mcp.call_tool(name, arguments)

    payload: Any
    if hasattr(result, "structured_content") and result.structured_content is not None:
        payload = result.structured_content
    elif hasattr(result, "data") and result.data is not None:
        payload = result.data
    elif hasattr(result, "content"):
        payload = [
            getattr(c, "text", str(c)) for c in (result.content or [])
        ]
    else:
        payload = str(result)

    try:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    except TypeError:
        print(repr(payload))


async def _phase2_verify() -> None:
    """End-to-end smoke for Phase 2 (resources + prompts) in a single process.

    Reads each resource (showing structure + counts but capping arrays so the
    output stays human-readable) and renders each prompt template, all on one
    shared FastMCP instance. Useful as a single command to confirm Phase 2
    health without paying Python cold-start cost six times.
    """
    from aix.mcp import build_mcp_server

    mcp = build_mcp_server()

    def _payload_from(result: Any) -> Any:
        # FastMCP's read_resource returns a ResourceResult with `.contents`
        # (list of ResourceContent). Each chunk has `.text` or `.content`.
        chunks_attr = getattr(result, "contents", None)
        if chunks_attr is not None:
            chunks = list(chunks_attr)
        elif isinstance(result, list):
            chunks = result
        else:
            chunks = [result]

        out: List[Any] = []
        for chunk in chunks:
            text = getattr(chunk, "text", None) or getattr(chunk, "content", None)
            if text is None:
                text = str(chunk)
            try:
                out.append(json.loads(text))
            except (TypeError, ValueError):
                out.append(text)
        return out[0] if len(out) == 1 else out

    def _summarise(label: str, payload: Any) -> None:
        print(f"\n{'-' * 72}")
        print(f"  {label}")
        print("-" * 72)
        if isinstance(payload, dict):
            keys = list(payload.keys())
            print(f"  shape: dict, top-level keys: {keys}")
            if "domains" in payload and isinstance(payload["domains"], list):
                for d in payload["domains"]:
                    if not isinstance(d, dict):
                        continue
                    name = d.get("domain", "?")
                    extras: List[str] = []
                    if "label_categories" in d:
                        extras.append(f"label_categories={len(d.get('label_categories') or {})}")
                    if "methodology_categories" in d:
                        extras.append(f"methodology_categories={len(d.get('methodology_categories') or {})}")
                    if "concepts_with_media" in d:
                        extras.append(f"concepts_with_media={d.get('concepts_with_media')}")
                    if "by_kind" in d:
                        extras.append(f"by_kind={d.get('by_kind')}")
                    print(f"   - {name}: {', '.join(extras) if extras else d}")
            elif "concepts" in payload and isinstance(payload["concepts"], list):
                print(f"  domain={payload.get('domain')!r}, count={payload.get('count')}, "
                      f"first 5 concepts={payload['concepts'][:5]}")
        else:
            preview = json.dumps(payload, ensure_ascii=False, default=str)
            if len(preview) > 400:
                preview = preview[:400] + "..."
            print(f"  preview: {preview}")

    print("=" * 72)
    print(" Phase 2 verification — Aix MCP server")
    print("=" * 72)

    # 1) kg://schema
    raw = await mcp.read_resource("kg://schema")
    _summarise("kg://schema", _payload_from(raw))

    # 2) kg://concepts/neuro (template)
    raw = await mcp.read_resource("kg://concepts/neuro")
    _summarise("kg://concepts/neuro (template)", _payload_from(raw))

    # 3) methodology://list
    raw = await mcp.read_resource("methodology://list")
    _summarise("methodology://list", _payload_from(raw))

    # 4) media://stats
    raw = await mcp.read_resource("media://stats")
    _summarise("media://stats", _payload_from(raw))

    # 5) educational-query prompt
    print(f"\n{'-' * 72}")
    print("  prompt: educational-query")
    print("-" * 72)
    p = await mcp.render_prompt(
        "educational-query",
        {
            "topic": "fotosintesi",
            "student_profile": "scuola primaria, classe 5a, 1 con dislessia",
            "domain": "neuro",
        },
    )
    msgs = getattr(p, "messages", []) or []
    print(f"  messages: {len(msgs)}")
    for i, m in enumerate(msgs):
        dumped = m.model_dump() if hasattr(m, "model_dump") else m
        role = dumped.get("role") if isinstance(dumped, dict) else "?"
        text = ""
        if isinstance(dumped, dict):
            content = dumped.get("content")
            if isinstance(content, dict):
                text = content.get("text", str(content))
            else:
                text = str(content)
        text_preview = text if len(text) <= 220 else text[:217] + "..."
        print(f"   [{i}] role={role}: {text_preview}")

    # 6) lesson-plan-request prompt
    # MCP requires prompt arguments to be strings — note the str-form duration.
    print(f"\n{'-' * 72}")
    print("  prompt: lesson-plan-request")
    print("-" * 72)
    p = await mcp.render_prompt(
        "lesson-plan-request",
        {
            "topic": "respirazione cellulare",
            "duration_minutes": "45",
            "methodology": "spaced retrieval + active recall",
            "level": "secondaria di primo grado",
            "domain": "neuro",
        },
    )
    msgs = getattr(p, "messages", []) or []
    print(f"  messages: {len(msgs)}")
    for i, m in enumerate(msgs):
        dumped = m.model_dump() if hasattr(m, "model_dump") else m
        role = dumped.get("role") if isinstance(dumped, dict) else "?"
        text = ""
        if isinstance(dumped, dict):
            content = dumped.get("content")
            if isinstance(content, dict):
                text = content.get("text", str(content))
            else:
                text = str(content)
        text_preview = text if len(text) <= 220 else text[:217] + "..."
        print(f"   [{i}] role={role}: {text_preview}")

    print(f"\n{'=' * 72}")
    print(" Phase 2 verification: PASS")
    print("=" * 72)


async def _phase3_verify() -> None:
    """End-to-end smoke for Phase 3 (media tools) — offline only.

    Exercises ``media.lookup_curated`` against a small basket of concepts we
    know exist in both domains, and verifies the four live-API tools are
    *registered* (correct schema + descriptors) without hitting YouTube /
    Semantic Scholar / OpenAI. To exercise the live tools, use ``--call``
    with explicit arguments — that path is intentionally opt-in to keep
    quota costs deterministic.
    """
    from aix.mcp import build_mcp_server

    mcp = build_mcp_server()

    print("=" * 72)
    print(" Phase 3 verification — Aix MCP server (media tools, offline)")
    print("=" * 72)

    expected_tool_names = {
        "media.lookup_curated",
        "media.search_youtube",
        "media.search_academic",
        "media.search_oer",
        "media.generate_diagram",
    }
    tools = await mcp.list_tools()
    seen = {t.name for t in tools}
    missing = expected_tool_names - seen
    print(
        f"\nRegistered tool surface: {len(seen)} total, "
        f"{len(expected_tool_names & seen)}/5 media.* tools present"
    )
    for name in sorted(expected_tool_names):
        marker = "OK" if name in seen else "MISSING"
        print(f"  [{marker}] {name}")
    if missing:
        print(f"\nERROR: missing media tools: {sorted(missing)}")
        sys.exit(2)

    # Exercise the offline curated lookup against known concepts in both
    # domains. We picked these from the head of the curated mapping JSON.
    cases = [
        ("neuro", ["Selective Attention", "Cognitive Control", "Working Memory"]),
        ("udl", ["Engagement", "Representation", "Action and Expression"]),
    ]

    for domain, concepts in cases:
        print(f"\n{'-' * 72}")
        print(f"  media.lookup_curated  domain={domain}  concepts={concepts}")
        print("-" * 72)
        try:
            result = await mcp.call_tool(
                "media.lookup_curated",
                {"concepts": concepts, "domain": domain},
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            sys.exit(2)

        payload: Any
        if hasattr(result, "structured_content") and result.structured_content is not None:
            payload = result.structured_content
        elif hasattr(result, "data") and result.data is not None:
            payload = result.data
        else:
            payload = result

        # Coerce to dict for inspection.
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()

        if not isinstance(payload, dict):
            print(f"  unexpected payload type: {type(payload).__name__}")
            sys.exit(2)

        requested = payload.get("requested")
        matched = payload.get("matched")
        by_concept = payload.get("by_concept") or []
        print(f"  requested={requested}  matched={matched}")
        for entry in by_concept:
            if not isinstance(entry, dict):
                continue
            counts = entry.get("counts") or {}
            print(
                f"   - {entry.get('concept_name'):<28} "
                f"found={entry.get('found')}  "
                f"counts={counts}"
            )

        if not matched:
            print(
                "  WARNING: zero matches in this domain — verify the curated "
                "mapping is in sync with the chosen concept list."
            )

    print(f"\n{'=' * 72}")
    print(" Phase 3 verification: PASS  (live-API tools registered, opt-in via --call)")
    print("=" * 72)


async def _phase4_verify() -> None:
    """End-to-end smoke for Phase 4 (agent.run_lesson_plan).

    Runs ONE short lesson with ``max_revisions=0`` (skip critic loop for
    speed) and validates the structured response has all expected fields.
    Costs ~3 LLM calls (planner + writer + critic) — same as a single
    /api/v1/agent/run hit. Uses the 'neuro' domain because we know it has
    KG content for the chosen query.
    """
    from aix.mcp import build_mcp_server

    mcp = build_mcp_server()

    print("=" * 72)
    print(" Phase 4 verification — Aix MCP server (agent.run_lesson_plan)")
    print("=" * 72)

    expected_tool = "agent.run_lesson_plan"
    tools = await mcp.list_tools()
    seen = {t.name for t in tools}
    if expected_tool not in seen:
        print(f"\nERROR: tool {expected_tool!r} is not registered")
        sys.exit(2)
    print(f"\nTool surface: {len(seen)} total, agent.run_lesson_plan present (OK)")

    query = "Crea una breve lezione sull'attenzione selettiva per la scuola primaria"
    print(f"\nRunning agent (max_revisions=0, domain=neuro):")
    print(f"   query={query!r}")
    print("   This will take ~30-90 seconds (LLM round-trips).")
    print("-" * 72)

    try:
        result = await mcp.call_tool(
            "agent.run_lesson_plan",
            {
                "query": query,
                "domain": "neuro",
                "language": "it",
                "max_revisions": 0,
            },
        )
    except Exception as exc:
        print(f"  ERROR raised by tool: {exc}")
        sys.exit(2)

    payload: Any
    if hasattr(result, "structured_content") and result.structured_content is not None:
        payload = result.structured_content
    elif hasattr(result, "data") and result.data is not None:
        payload = result.data
    else:
        payload = result

    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()

    if not isinstance(payload, dict):
        print(f"  ERROR: unexpected payload type {type(payload).__name__}")
        sys.exit(2)

    lesson_md = payload.get("lesson_plan_md") or ""
    meta = payload.get("meta") or {}
    planner = payload.get("planner") or {}
    retriever = payload.get("retriever") or {}

    print(f"\n  lesson_plan_md: {len(lesson_md)} chars")
    if lesson_md:
        first_lines = "\n    ".join(lesson_md.splitlines()[:3])
        print(f"    preview:\n    {first_lines}")
    print(f"\n  meta:")
    print(f"    duration_seconds      = {meta.get('duration_seconds')}")
    print(f"    approved              = {meta.get('approved')}")
    print(f"    revision_count        = {meta.get('revision_count')}")
    print(f"    nodes_count           = {meta.get('nodes_count')}")
    print(f"    recommendations_count = {meta.get('recommendations_count')}")
    print(f"    media_counts          = {meta.get('media_counts')}")
    print(f"    search_queries_count  = {meta.get('search_queries_count')}")
    print(f"\n  planner:")
    print(f"    intent_label   = {planner.get('intent_label')!r}")
    print(f"    scope_label    = {planner.get('scope_label')!r}")
    print(f"    key_concepts   = {planner.get('key_concepts')}")
    print(f"    search_queries = {len(planner.get('search_queries') or [])} queries")
    print(f"\n  retriever:")
    print(f"    nodes_count           = {retriever.get('nodes_count')}")
    print(f"    relationships_count   = {retriever.get('relationships_count')}")
    print(f"    recommendations_count = {retriever.get('recommendations_count')}")
    print(f"    top_concepts          = {retriever.get('top_concepts')}")
    print(f"    retrieval_confidence  = {retriever.get('retrieval_confidence')!r}")

    if not lesson_md.strip():
        print("\n  ERROR: lesson_plan_md is empty — agent failed silently")
        sys.exit(2)
    if meta.get("duration_seconds", 0) <= 0:
        print("\n  ERROR: duration_seconds is missing/zero")
        sys.exit(2)

    print(f"\n{'=' * 72}")
    print(" Phase 4 verification: PASS  (full agent pipeline reachable via MCP)")
    print("=" * 72)


async def _phase5_verify(
    base_url: str,
    email: Optional[str],
    password: Optional[str],
) -> None:
    """End-to-end smoke for Phase 5 (Streamable HTTP mount at /mcp/).

    Pipeline:
        1. Hit ``GET /api/v1/health`` to confirm the server is up.
        2. POST ``/auth/jwt/login`` with form-encoded creds → Bearer token.
        3. Connect a FastMCP ``Client`` over Streamable HTTP to ``/mcp/``
           using ``auth=<token>``.
        4. ``client.list_tools()`` and assert ``agent.run_lesson_plan`` is
           among the registered tools (10 minimum after Phases 1-4).
        5. Call ``kg.list_concepts`` (cheap KG round-trip) to confirm a
           tool actually executes through the HTTP transport, not just
           via the in-process listing.

    Requires:
        * uvicorn aix.api.main:app running at ``base_url``.
        * A valid teacher account (env ``AIX_TEST_USER_EMAIL`` /
          ``AIX_TEST_USER_PASSWORD``, or pass ``--email`` / ``--password``).
    """
    import httpx
    from fastmcp import Client

    base_url = base_url.rstrip("/")

    print("=" * 72)
    print(" Phase 5 verification — Aix MCP Streamable HTTP mount")
    print(f" Target: {base_url}/mcp/")
    print("=" * 72)

    email = email or os.getenv("AIX_TEST_USER_EMAIL")
    password = password or os.getenv("AIX_TEST_USER_PASSWORD")
    if not email or not password:
        print(
            "\nERROR: missing credentials. Pass --email/--password or set "
            "AIX_TEST_USER_EMAIL / AIX_TEST_USER_PASSWORD."
        )
        sys.exit(2)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Health probe
        try:
            r = await client.get(f"{base_url}/api/v1/health")
            r.raise_for_status()
        except Exception as exc:
            print(f"\nERROR: health probe failed at {base_url}: {exc}")
            print("       Is uvicorn running? Try:")
            print("         uvicorn aix.api.main:app --port 8765")
            sys.exit(2)
        print(f"\n[1/5] Health: {r.json().get('status', 'unknown')}")

        # 2. Confirm /mcp/ rejects unauthenticated requests.
        # MCP spec: clients send POST with JSON-RPC initialize. We just want
        # to see a 401, so a minimal POST is sufficient.
        try:
            r = await client.post(
                f"{base_url}/mcp/",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {},
                },
                headers={"Accept": "application/json, text/event-stream"},
            )
        except Exception as exc:
            print(f"\nERROR: unauthenticated probe to /mcp/ failed: {exc}")
            sys.exit(2)
        if r.status_code != 401:
            print(
                f"\nWARNING: expected 401 from unauthenticated /mcp/, got "
                f"{r.status_code}. Is AIX_MCP_REQUIRE_AUTH disabled? "
                f"Body: {r.text[:160]}"
            )
        else:
            print(f"[2/5] Unauth /mcp/: 401 (auth correctly enforced)")

        # 3. Login via /auth/jwt/login (form-encoded).
        try:
            r = await client.post(
                f"{base_url}/auth/jwt/login",
                data={"username": email, "password": password},
            )
            r.raise_for_status()
            token = r.json().get("access_token")
        except Exception as exc:
            print(f"\nERROR: login failed for {email}: {exc}")
            sys.exit(2)
        if not token:
            print(f"\nERROR: login response missing access_token: {r.text[:200]}")
            sys.exit(2)
        print(f"[3/5] Login: token acquired ({len(token)} chars)")

    # 4. Connect via FastMCP Client over Streamable HTTP.
    mcp_url = f"{base_url}/mcp/"
    try:
        async with Client(transport=mcp_url, auth=token) as mcp_client:
            tools = await mcp_client.list_tools()
            tool_names = {t.name for t in tools}
            print(f"[4/5] MCP list_tools: {len(tool_names)} tools registered")
            for name in sorted(tool_names):
                print(f"        - {name}")

            expected = {
                "kg.search",
                "kg.get_context",
                "kg.list_concepts",
                "kg.get_schema",
                "media.lookup_curated",
                "agent.run_lesson_plan",
            }
            missing = expected - tool_names
            if missing:
                print(f"\nERROR: missing expected tools: {missing}")
                sys.exit(2)

            # 5. End-to-end tool call over HTTP.
            try:
                result = await mcp_client.call_tool(
                    "kg.list_concepts",
                    {"domain": "neuro", "limit": 5},
                )
            except Exception as exc:
                print(f"\nERROR: kg.list_concepts call failed over HTTP: {exc}")
                sys.exit(2)

            payload: Any
            if hasattr(result, "structured_content") and result.structured_content is not None:
                payload = result.structured_content
            elif hasattr(result, "data") and result.data is not None:
                payload = result.data
            else:
                payload = result
            if hasattr(payload, "model_dump"):
                payload = payload.model_dump()

            concepts = (
                (payload or {}).get("concepts", [])
                if isinstance(payload, dict)
                else []
            )
            print(
                f"[5/5] kg.list_concepts(domain=neuro, limit=5): "
                f"{len(concepts)} concepts returned"
            )
            for c in concepts[:5]:
                print(f"        - {c}")
    except Exception as exc:
        print(f"\nERROR: MCP HTTP client session failed: {exc}")
        sys.exit(2)

    print(f"\n{'=' * 72}")
    print(" Phase 5 verification: PASS  (Bearer-protected /mcp/ end-to-end)")
    print("=" * 72)


async def _read_resource(uri: str) -> None:
    """Read an MCP resource by URI and pretty-print the JSON payload."""
    from aix.mcp import build_mcp_server

    mcp = build_mcp_server()

    print(f"--> Reading resource: {uri}")
    print("-" * 72)

    result = await mcp.read_resource(uri)

    # FastMCP returns a ResourceResult with `.contents` (list of
    # ResourceContent). Each item exposes a `.text` (or `.content`) string,
    # which for our resources is JSON.
    chunks_attr = getattr(result, "contents", None)
    if chunks_attr is not None:
        chunks = list(chunks_attr)
    elif isinstance(result, list):
        chunks = result
    else:
        chunks = [result]

    for idx, chunk in enumerate(chunks):
        text = getattr(chunk, "text", None) or getattr(chunk, "content", None)
        if text is None:
            text = str(chunk)
        if len(chunks) > 1:
            print(f"# chunk {idx + 1}/{len(chunks)}")
        try:
            parsed = json.loads(text)
            print(json.dumps(parsed, indent=2, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            print(text)


async def _render_prompt(name: str, arguments: Dict[str, Any]) -> None:
    """Render an MCP prompt template and pretty-print the message envelope."""
    from aix.mcp import build_mcp_server

    mcp = build_mcp_server()

    print(f"--> Rendering prompt: {name}")
    print(f"--> Arguments: {json.dumps(arguments, ensure_ascii=False)}")
    print("-" * 72)

    # IMPORTANT: FastMCP 3.x exposes TWO methods on the server:
    #   * mcp.get_prompt(name, version)      → fetches the Prompt definition
    #   * mcp.render_prompt(name, arguments) → renders the prompt with args
    # We want the second — the first treats `arguments` as `version` and
    # crashes inside the version-matching logic with a misleading error.
    result = await mcp.render_prompt(name, arguments)

    messages = getattr(result, "messages", None)
    if messages is None and hasattr(result, "model_dump"):
        dumped = result.model_dump()
        messages = dumped.get("messages")

    if messages is None:
        print(repr(result))
        return

    serialisable: List[Dict[str, Any]] = []
    for m in messages:
        if hasattr(m, "model_dump"):
            serialisable.append(m.model_dump())
        elif isinstance(m, dict):
            serialisable.append(m)
        else:
            serialisable.append({"repr": repr(m)})

    print(json.dumps(serialisable, indent=2, ensure_ascii=False, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the Aix MCP server in-process."
    )
    parser.add_argument(
        "--call",
        metavar="TOOL_NAME",
        help="Invoke a single tool by name (e.g. kg.list_concepts).",
    )
    parser.add_argument(
        "--read-resource",
        dest="read_resource",
        metavar="URI",
        help="Read an MCP resource by URI (e.g. kg://schema, kg://concepts/neuro).",
    )
    parser.add_argument(
        "--render-prompt",
        dest="render_prompt",
        metavar="NAME",
        help="Render an MCP prompt template (e.g. educational-query).",
    )
    parser.add_argument(
        "--phase2-verify",
        dest="phase2_verify",
        action="store_true",
        help="One-shot end-to-end smoke for Phase 2 (4 resources + 2 prompts).",
    )
    parser.add_argument(
        "--phase3-verify",
        dest="phase3_verify",
        action="store_true",
        help=(
            "One-shot end-to-end smoke for Phase 3 (media.lookup_curated, "
            "offline). Live-API media tools are schema-checked but not invoked."
        ),
    )
    parser.add_argument(
        "--phase4-verify",
        dest="phase4_verify",
        action="store_true",
        help=(
            "One-shot end-to-end smoke for Phase 4 (agent.run_lesson_plan). "
            "Runs ONE short lesson with max_revisions=0 (~3 LLM calls)."
        ),
    )
    parser.add_argument(
        "--phase5-verify",
        dest="phase5_verify",
        action="store_true",
        help=(
            "End-to-end smoke for Phase 5 (Streamable HTTP mount at /mcp/). "
            "Requires uvicorn running. Logs in, asserts 401 without token, "
            "lists tools and calls kg.list_concepts over HTTP."
        ),
    )
    parser.add_argument(
        "--base-url",
        dest="base_url",
        default="http://127.0.0.1:8765",
        help="Base URL of the running aix.api server (Phase 5).",
    )
    parser.add_argument(
        "--email",
        default=None,
        help=(
            "Teacher account email for Phase 5 login. Falls back to "
            "AIX_TEST_USER_EMAIL env var."
        ),
    )
    parser.add_argument(
        "--password",
        default=None,
        help=(
            "Teacher account password for Phase 5 login. Falls back to "
            "AIX_TEST_USER_PASSWORD env var."
        ),
    )
    parser.add_argument("--domain", default="neuro", help="Domain (neuro|udl).")
    parser.add_argument("--query", default=None, help="Query string for kg.search.")
    parser.add_argument("--limit", type=int, default=20, help="Limit for list tools.")
    parser.add_argument(
        "--concepts",
        default=None,
        help="Comma-separated concept list for media.lookup_curated.",
    )
    parser.add_argument(
        "--max-results",
        dest="max_results",
        type=int,
        default=5,
        help="Max results cap for media.search_* tools (default 5).",
    )
    parser.add_argument(
        "--year-from",
        dest="year_from",
        type=int,
        default=None,
        help="Earliest publication year (media.search_academic).",
    )
    parser.add_argument(
        "--open-access-only",
        dest="open_access_only",
        action="store_true",
        help="Filter to open-access papers only (media.search_academic).",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language hint for media.search_youtube / media.search_oer.",
    )
    parser.add_argument(
        "--concept",
        default=None,
        help="Central concept for media.generate_diagram.",
    )
    parser.add_argument(
        "--diagram-type",
        dest="diagram_type",
        default="mindmap",
        help=(
            "Mermaid diagram type: mindmap, flowchart, sequence, timeline, "
            "hierarchy, comparison, process."
        ),
    )
    parser.add_argument(
        "--related-concepts",
        dest="related_concepts",
        default=None,
        help="Comma-separated related concepts for media.generate_diagram.",
    )
    parser.add_argument(
        "--no-validate",
        dest="no_validate",
        action="store_true",
        help="Skip mermaid.ink render validation (media.generate_diagram).",
    )
    parser.add_argument(
        "--max-revisions",
        dest="max_revisions",
        type=int,
        default=None,
        help="Critic revision cap for agent.run_lesson_plan (0..4).",
    )
    parser.add_argument(
        "--session-id",
        dest="session_id",
        default=None,
        help="Optional session id for agent.run_lesson_plan log tracing.",
    )
    parser.add_argument(
        "--topic", default=None, help="Topic for prompt rendering."
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Optional student profile (educational-query prompt).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Lesson duration in minutes (lesson-plan-request prompt).",
    )
    parser.add_argument(
        "--methodology",
        default=None,
        help="Methodology hint (lesson-plan-request prompt).",
    )
    parser.add_argument(
        "--level",
        default=None,
        help="Educational level cue (lesson-plan-request prompt).",
    )
    args = parser.parse_args()

    _ensure_pythonpath()
    _configure_logging()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass

    selected = sum(
        1
        for x in (
            args.call,
            args.read_resource,
            args.render_prompt,
            args.phase2_verify,
            args.phase3_verify,
            args.phase4_verify,
            args.phase5_verify,
        )
        if x
    )
    if selected > 1:
        print(
            "ERROR: choose at most one of --call / --read-resource / "
            "--render-prompt / --phase2-verify / --phase3-verify / "
            "--phase4-verify / --phase5-verify"
        )
        sys.exit(2)

    if args.phase2_verify:
        asyncio.run(_phase2_verify())
        return

    if args.phase3_verify:
        asyncio.run(_phase3_verify())
        return

    if args.phase4_verify:
        asyncio.run(_phase4_verify())
        return

    if args.phase5_verify:
        asyncio.run(
            _phase5_verify(
                base_url=args.base_url,
                email=args.email,
                password=args.password,
            )
        )
        return

    if args.read_resource:
        asyncio.run(_read_resource(args.read_resource))
        return

    if args.render_prompt:
        prompt_args: Dict[str, Any] = {}
        if args.topic is not None:
            prompt_args["topic"] = args.topic
        if args.render_prompt == "educational-query":
            if args.profile:
                prompt_args["student_profile"] = args.profile
            prompt_args["domain"] = args.domain
        elif args.render_prompt == "lesson-plan-request":
            # MCP spec: prompt arguments must be passed as strings.
            prompt_args["duration_minutes"] = str(args.duration)
            if args.methodology:
                prompt_args["methodology"] = args.methodology
            if args.level:
                prompt_args["level"] = args.level
            prompt_args["domain"] = args.domain
        asyncio.run(_render_prompt(args.render_prompt, prompt_args))
        return

    if not args.call:
        asyncio.run(_print_listing())
        return

    arguments: Dict[str, Any]
    if args.call.startswith("media."):
        arguments = {}
        if args.call == "media.lookup_curated":
            if not args.concepts:
                print("ERROR: --concepts is required for media.lookup_curated")
                sys.exit(2)
            concept_list = [c.strip() for c in args.concepts.split(",") if c.strip()]
            arguments["concepts"] = concept_list
            arguments["domain"] = args.domain
        elif args.call == "media.search_youtube":
            if not args.query:
                print("ERROR: --query is required for media.search_youtube")
                sys.exit(2)
            arguments["query"] = args.query
            arguments["max_results"] = args.max_results
            arguments["language"] = args.language
        elif args.call == "media.search_academic":
            if not args.query:
                print("ERROR: --query is required for media.search_academic")
                sys.exit(2)
            arguments["query"] = args.query
            arguments["max_results"] = args.max_results
            if args.year_from is not None:
                arguments["year_from"] = args.year_from
            arguments["open_access_only"] = bool(args.open_access_only)
        elif args.call == "media.search_oer":
            if not args.query:
                print("ERROR: --query is required for media.search_oer")
                sys.exit(2)
            arguments["query"] = args.query
            arguments["max_results"] = args.max_results
            arguments["language"] = args.language
        elif args.call == "media.generate_diagram":
            concept = args.concept or args.query
            if not concept:
                print(
                    "ERROR: --concept (or --query) is required for "
                    "media.generate_diagram"
                )
                sys.exit(2)
            arguments["concept"] = concept
            arguments["diagram_type"] = args.diagram_type
            if args.related_concepts:
                arguments["related_concepts"] = [
                    c.strip()
                    for c in args.related_concepts.split(",")
                    if c.strip()
                ]
            arguments["validate"] = not args.no_validate
        else:
            print(f"ERROR: unknown media tool '{args.call}'")
            sys.exit(2)
    elif args.call.startswith("agent."):
        arguments = {}
        if args.call == "agent.run_lesson_plan":
            if not args.query:
                print("ERROR: --query is required for agent.run_lesson_plan")
                sys.exit(2)
            arguments["query"] = args.query
            arguments["domain"] = args.domain
            if args.max_revisions is not None:
                arguments["max_revisions"] = args.max_revisions
            if args.session_id:
                arguments["session_id"] = args.session_id
        else:
            print(f"ERROR: unknown agent tool '{args.call}'")
            sys.exit(2)
    else:
        arguments = {"domain": args.domain}
        if args.call in {"kg.search", "kg.get_context"}:
            if not args.query:
                print("ERROR: --query is required for kg.search / kg.get_context")
                sys.exit(2)
            arguments["query"] = args.query
        elif args.call == "kg.list_concepts":
            arguments["limit"] = args.limit
        elif args.call == "kg.get_schema":
            pass

    asyncio.run(_call_tool(args.call, arguments))


if __name__ == "__main__":
    main()
