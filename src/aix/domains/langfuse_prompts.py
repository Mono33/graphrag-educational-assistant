"""
Langfuse prompt client for domain prompts.

Domain prompts (system_prompt, writer_prompt, response_template) live in Langfuse
so they can be edited from the Langfuse UI without code deploys.

Cache TTL is 60 seconds — Langfuse edits take effect within one minute,
no server restart needed.

Required env vars:
    LANGFUSE_SECRET_KEY              sk-lf-...
    LANGFUSE_PUBLIC_KEY              pk-lf-...
    LANGFUSE_HOST or LANGFUSE_BASE_URL   optional, defaults to https://cloud.langfuse.com
"""

import logging
import os

from langfuse import Langfuse

logger = logging.getLogger(__name__)
_client: "Langfuse | None" = None


def _get_client() -> Langfuse:
    global _client
    if _client is None:
        _client = Langfuse(
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            host=(
                os.environ.get("LANGFUSE_HOST")
                or os.environ.get("LANGFUSE_BASE_URL")
                or "https://cloud.langfuse.com"
            ),
        )
    return _client


def fetch_prompt(name: str, cache_ttl_seconds: int = 60) -> str:
    """Fetch a text prompt from Langfuse by name.

    Caches the result for `cache_ttl_seconds` (default 60 s).
    Raises RuntimeError with a clear message if the prompt is unavailable.
    """
    try:
        return _get_client().get_prompt(name, cache_ttl_seconds=cache_ttl_seconds).compile()
    except Exception as e:
        logger.error("Langfuse prompt '%s' unavailable: %s", name, e)
        raise RuntimeError(
            f"Prompt '{name}' not found in Langfuse. "
            "Check LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY and that the prompt exists "
            "in Langfuse with the label 'production'."
        ) from e
