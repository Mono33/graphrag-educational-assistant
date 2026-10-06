"""Provider prompt caching: the stable head of a prompt is billed at a fraction.

A call that repeats a long, identical prefix (the 100k-character schema of
text2cypher, the instructions of the citation judge) pays full price for it every
time unless the prompt marks where the prefix ends. OpenRouter forwards that mark
(``cache_control``) to Anthropic and Google; measured on 2026-10-06:

* Anthropic (Sonnet 5): a cached read costs about 12% of the plain call;
* Google (Gemini flash-lite): about 10%. Implicit caching does **not** happen
  there -- an identical 22k-token prefix came back with 0 cached tokens -- so
  without the mark nothing is saved.

Only the wording of the request changes: the text is the same, split in blocks.
Other providers keep plain strings, so they are untouched. Off with
``AIX_PROMPT_CACHE_ENABLED=false``. Nothing here may fail a run.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: OpenRouter honours ``cache_control`` for these providers.
_SUPPORTED_PREFIXES = ("anthropic/", "google/")


def enabled() -> bool:
    raw = os.getenv("AIX_PROMPT_CACHE_ENABLED")
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() in ("1", "true", "yes", "on")


def supports(model: Optional[str]) -> bool:
    """True when ``model`` is served by a provider that reads ``cache_control``."""
    return bool(model) and str(model).lower().startswith(_SUPPORTED_PREFIXES)


def _mark(ttl: Optional[str]) -> dict[str, Any]:
    mark: dict[str, Any] = {"type": "ephemeral"}
    if ttl:
        mark["ttl"] = ttl
    return mark


def _block(text: str, *, cache: bool, ttl: Optional[str]) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "text", "text": text}
    if cache:
        block["cache_control"] = _mark(ttl)
    return block


def cacheable(
    messages: list[dict[str, Any]],
    *,
    model: Optional[str],
    stable_user_prefix: Optional[str] = None,
    ttl: Optional[str] = None,
) -> list[dict[str, Any]]:
    """``messages`` with the stable head marked for caching; a copy, never in place.

    The system message becomes one marked block. When ``stable_user_prefix`` is
    given and the user message starts with it, the user message is split in a
    marked block (the prefix) and a plain one (the rest), so a turn that changes
    only the tail still reads the head from the cache. A message that is not a
    plain string, or a model that does not read the mark, is returned as it is.
    """
    if not enabled() or not supports(model):
        return messages
    try:
        out: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, str) or not content:
                out.append(message)
                continue
            role = message.get("role")
            if role == "system":
                out.append({**message, "content": [_block(content, cache=True, ttl=ttl)]})
            elif (
                role == "user"
                and stable_user_prefix
                and content.startswith(stable_user_prefix)
                and len(content) > len(stable_user_prefix)
            ):
                rest = content[len(stable_user_prefix) :]
                out.append(
                    {
                        **message,
                        "content": [
                            _block(stable_user_prefix, cache=True, ttl=ttl),
                            _block(rest, cache=False, ttl=None),
                        ],
                    }
                )
            else:
                out.append(message)
        return out
    except Exception:  # noqa: BLE001 — an unmarked prompt beats a failed call
        logger.warning("[prompt-cache] could not mark the prompt", exc_info=True)
        return messages


def cacheable_lc(
    messages: list[Any], *, model: Optional[str], ttl: Optional[str] = None
) -> list[Any]:
    """The same for LangChain messages: each ``SystemMessage`` becomes a marked block."""
    if not enabled() or not supports(model):
        return messages
    try:
        from langchain_core.messages import SystemMessage

        out: list[Any] = []
        for message in messages:
            if isinstance(message, SystemMessage) and isinstance(message.content, str):
                out.append(SystemMessage(content=[_block(message.content, cache=True, ttl=ttl)]))
            else:
                out.append(message)
        return out
    except Exception:  # noqa: BLE001
        logger.warning("[prompt-cache] could not mark the prompt", exc_info=True)
        return messages
