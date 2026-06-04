"""
LLM connectivity probe (CORE 2 #11a follow-up — observability).

When the agent runs fail with the SDK's generic ``Connection error.``
after retries, the cause is almost always one of three things:

* the host can't complete a TLS handshake to the LLM endpoint (corp
  firewall doing SSL inspection, missing CA bundle on Windows, broken
  ``certifi`` install),
* the API key is missing / typo'd / revoked,
* the upstream provider is having an outage.

The OpenAI Python SDK swallows all three into the same opaque message,
which made the recent ``Connection error.`` failures look like a code
regression of #9 / #11a. They were not — but distinguishing those cases
required reading scattered logs.

This module solves that problem at startup: it issues a single tiny call
(``GET /models?limit=1``) against the configured ``OpenAIConfig.base_url``
and emits one log line that says **exactly** what's wrong. The probe is:

* fully optional — disable with ``AIX_LLM_PROBE_ENABLED=false``,
* never blocking — it runs as a background task with a 5s timeout and
  cannot delay startup or fail the API,
* no-op for the agent runtime — only logging side-effects,
* zero behavioural change for production traffic — agents still pull the
  same async client from :func:`OpenAIConfig.get_async_client`.

Designed to ship behind a default-ON flag because the cost is essentially
free (~50–500 ms one-shot call) and the diagnostic signal is high.
"""

from __future__ import annotations

import asyncio
import logging
import os
import ssl
from typing import Final

logger = logging.getLogger(__name__)


_PROBE_TIMEOUT_S: Final[float] = 5.0


def _log_tls_remediation(base_url: str, exc: BaseException) -> None:
    """Single source of truth for the TLS-failure remediation message.
    Both the dedicated ``ssl.SSLCertVerificationError`` branch and the
    ``httpx.ConnectError`` sniffer call this so the operator-facing log
    line is identical regardless of which exception type bubbled up."""
    logger.error(
        "[llm_probe] ❌ TLS certificate verification FAILED against %s (%s). "
        "This is the actual root cause behind 'Connection error.' in agent runs. "
        "Likely fixes — try in order: "
        "(1) `pip install --force-reinstall certifi` inside the venv, "
        "(2) export SSL_CERT_FILE=$(python -m certifi) before launching uvicorn, "
        "(3) if a corporate firewall / antivirus is doing SSL inspection, install "
        "its root CA into the venv's certifi bundle, "
        "(4) on Windows, run `python -m pip install --upgrade pip-system-certs` to "
        "let Python trust the OS certificate store.",
        base_url,
        exc,
    )


def _probe_enabled() -> bool:
    return (os.getenv("AIX_LLM_PROBE_ENABLED") or "true").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _redact(api_key: str) -> str:
    """Show only the first 8 + last 4 chars of the API key for the log."""
    if not api_key:
        return "<empty>"
    if len(api_key) <= 14:
        return "<redacted>"
    return f"{api_key[:8]}…{api_key[-4:]}"


async def _probe_once() -> None:
    """One probe round-trip. Catches every conceivable error and turns
    it into a single human-actionable log line. Never re-raises."""
    try:
        from aix.core.config import config as app_config
    except Exception as exc:  # noqa: BLE001
        logger.warning("[llm_probe] config import failed: %s — skipping probe", exc)
        return

    base_url = (app_config.openai.base_url or "").rstrip("/")
    api_key = app_config.openai.api_key or ""
    model = app_config.openai.model or "<unset>"

    if not base_url:
        logger.warning("[llm_probe] base_url not configured — skipping probe")
        return
    if not api_key:
        logger.error(
            "[llm_probe] ❌ API key is empty (checked OPENROUTER_API_KEY then OPENAI_API_KEY). "
            "Agent runs WILL fail with 'Connection error.' — set the key in .env and restart."
        )
        return

    # ``httpx`` is already a transitive dep of ``openai>=1`` and ``fastapi``.
    # Importing here keeps the module import-light when the probe is disabled.
    try:
        import httpx
    except Exception as exc:  # noqa: BLE001
        logger.warning("[llm_probe] httpx unavailable (%s) — skipping probe", exc)
        return

    headers = {"Authorization": f"Bearer {api_key}"}
    # OpenRouter recommends an HTTP-Referer + X-Title for free routing;
    # set them only when we look like we're hitting OpenRouter so the
    # header set stays minimal for plain OpenAI.
    if "openrouter" in base_url:
        headers.setdefault("HTTP-Referer", "https://aix.local")
        headers.setdefault("X-Title", "aix-graphrag-startup-probe")

    url = f"{base_url}/models"
    started = asyncio.get_event_loop().time()
    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT_S) as client:
            resp = await client.get(url, headers=headers)
        elapsed_ms = int((asyncio.get_event_loop().time() - started) * 1000)
    except ssl.SSLCertVerificationError as exc:
        _log_tls_remediation(base_url, exc)
        return
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        # ``httpx.ConnectError`` wraps the underlying SSL error from
        # ``ssl.SSLCertVerificationError`` instead of letting it propagate,
        # so we sniff the message to surface the same actionable remediation
        # advice as the dedicated branch above. This is the path that fires
        # in practice on Windows hosts with a stale ``certifi`` bundle or
        # corporate-firewall SSL inspection.
        msg = str(exc)
        if "CERTIFICATE_VERIFY_FAILED" in msg or "SSL" in msg:
            _log_tls_remediation(base_url, exc)
            return
        logger.error(
            "[llm_probe] ❌ Cannot connect to LLM endpoint %s (%s: %s). "
            "Check internet/VPN/firewall. Agent runs will fail with 'Connection error.'",
            base_url,
            exc.__class__.__name__,
            exc,
        )
        return
    except httpx.ReadTimeout as exc:
        logger.error(
            "[llm_probe] ❌ Read timeout against %s after %.1fs (%s). "
            "Endpoint reachable but unresponsive — likely upstream incident.",
            base_url,
            _PROBE_TIMEOUT_S,
            exc,
        )
        return
    except Exception as exc:  # noqa: BLE001 — defensive catch-all
        logger.error(
            "[llm_probe] ❌ Unexpected probe failure against %s (%s: %s). "
            "Falling back to opaque 'Connection error.' if agent runs are attempted.",
            base_url,
            exc.__class__.__name__,
            exc,
        )
        return

    if resp.status_code == 200:
        logger.info(
            "[llm_probe] ✅ LLM endpoint reachable: %s status=200 elapsed=%dms model=%r api_key=%s",
            base_url,
            elapsed_ms,
            model,
            _redact(api_key),
        )
        return

    if resp.status_code in (401, 403):
        logger.error(
            "[llm_probe] ❌ Authentication failed (HTTP %d) against %s — API key %s is "
            "rejected. Agent runs will fail. Verify OPENROUTER_API_KEY / OPENAI_API_KEY "
            "in .env (key may be revoked, expired, or scoped to a different project).",
            resp.status_code,
            base_url,
            _redact(api_key),
        )
        return

    # Any other 4xx/5xx — log the body preview so the cause is obvious.
    body_preview = (resp.text or "")[:240]
    logger.warning(
        "[llm_probe] ⚠️ LLM endpoint returned HTTP %d against %s elapsed=%dms — body preview: %r",
        resp.status_code,
        base_url,
        elapsed_ms,
        body_preview,
    )


async def schedule_startup_probe() -> None:
    """Schedule the probe in the background — never blocks startup.

    Designed to be called from within FastAPI's ``lifespan`` startup
    section. Returns immediately; the probe runs concurrently with the
    rest of startup and the first few requests.
    """
    if not _probe_enabled():
        logger.debug("[llm_probe] disabled via AIX_LLM_PROBE_ENABLED")
        return
    try:
        asyncio.create_task(_probe_once())
    except Exception as exc:  # noqa: BLE001
        logger.warning("[llm_probe] failed to schedule probe: %s", exc)
