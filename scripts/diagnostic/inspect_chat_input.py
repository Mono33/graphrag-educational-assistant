"""
Diagnostic — fetch a lesson page server-side and analyse the rendered HTML.

Used to debug the "paperclip icon doesn't show in the chat input" bug
without browser DevTools (CORE 2 #6.6 P3, 2026-04-26).

What it does
------------
1. Prompts for email + password (no defaults; nothing logged to disk besides
   the rendered HTML for the page you're inspecting).
2. POSTs to ``/auth/login`` to obtain the JWT-in-HttpOnly-cookie session.
3. Picks a lesson:
       - If you pass ``--lesson-id``, uses that one.
       - Otherwise hits ``/webui/`` and uses the most recently created
         lesson visible to the logged-in user. (Falls back to asking
         interactively if it can't find one.)
4. GETs ``/webui/lesson/{id}`` with the auth cookie and saves the FULL
   HTML to ``data/diagnostic/chat_input_rendered.html`` for further
   inspection.
5. Prints a structured "is the markup actually there?" report so we know
   immediately whether the bug is server-side (markup missing) or
   client-side (markup present but hidden by CSS / web-component).

Run from the repo root:
    python scripts/diagnostic/inspect_chat_input.py
    # or with a specific lesson id:
    python scripts/diagnostic/inspect_chat_input.py --lesson-id <uuid>

Requires the dev server already running:
    python -m uvicorn aix.api.main:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import argparse
import getpass
import re
import sys
from pathlib import Path

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8765"

# ── Output target ────────────────────────────────────────────────────────────
# Saved beside the curated KG/media data so it sits next to other
# diagnostic / generated artifacts; gitignored via data/ .gitignore patterns.
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "diagnostic"
OUT_FILE = OUT_DIR / "chat_input_rendered.html"


def _login(client: httpx.Client, email: str, password: str) -> None:
    """Authenticate against /auth/login and verify the cookie was set."""
    print(f"  → POST /auth/login as {email!r}…", flush=True)
    r = client.post(
        "/auth/login",
        # The login route uses Form(...) fields ``email`` + ``password``
        # (see src/aix/webui/auth/routes.py::login_post). NOT the OAuth2
        # ``username`` convention — fastapi-users' default form schema was
        # overridden so the form key matches the visible field label.
        data={"email": email, "password": password},
        # /auth/login responds 303 on success → redirect to /webui/. We don't
        # want httpx to follow it because we just need the Set-Cookie header.
        follow_redirects=False,
    )
    if r.status_code not in (200, 303, 302):
        sys.exit(
            f"❌ Login failed: HTTP {r.status_code}\n"
            f"   Response: {r.text[:200]}"
        )

    # httpx.Cookies stores entries as http.cookiejar.Cookie objects; iterating
    # ``client.cookies`` yields names (strings), iterating ``client.cookies.jar``
    # yields Cookie objects (need ``.name``). We use the former for safety.
    cookie_names = [str(name) for name in client.cookies]
    if not any("auth" in name.lower() for name in cookie_names):
        # Exact cookie name depends on the FastAPI-Users config. Don't fail
        # here — the next authenticated GET will redirect to /auth/login if
        # the cookie really wasn't set, and we catch that downstream.
        print(
            f"  ⚠️  Login responded OK but no auth-looking cookie in jar "
            f"(saw: {cookie_names!r}). Will verify on first authenticated GET."
        )
    else:
        print(f"  ✅ Logged in (cookies attached: {cookie_names!r}).")


def _pick_lesson_id(client: httpx.Client, override: str | None) -> str:
    if override:
        return override
    print("  → GET /webui/  (looking for most recent lesson)…", flush=True)
    r = client.get("/webui/", follow_redirects=False)
    if r.status_code in (302, 303):
        sys.exit(
            "❌ /webui/ redirected — auth cookie probably wasn't kept. "
            "Try passing --lesson-id explicitly."
        )
    matches = re.findall(r"/webui/lesson/([0-9a-fA-F-]{36})", r.text)
    if matches:
        most_recent = matches[0]
        print(f"  ✅ Found lesson id on home page: {most_recent}")
        return most_recent
    return input(
        "  Could not auto-detect a lesson on /webui/. "
        "Paste a lesson id (UUID): "
    ).strip()


def _fetch_lesson(client: httpx.Client, lesson_id: str) -> str:
    url = f"/webui/lesson/{lesson_id}"
    print(f"  → GET {url}…", flush=True)
    r = client.get(url, follow_redirects=False)
    if r.status_code in (302, 303):
        sys.exit(
            f"❌ {url} → {r.status_code} {r.headers.get('location', '?')}\n"
            "   Auth cookie didn't survive — re-run and double-check the "
            "password."
        )
    if r.status_code != 200:
        sys.exit(f"❌ {url} → HTTP {r.status_code}\n   {r.text[:300]}")
    print(f"  ✅ Got {len(r.text):,} bytes of HTML.")
    return r.text


# ── Inspection rules ─────────────────────────────────────────────────────────
# Each rule is a (label, predicate, hint) tuple. Predicate is either a literal
# substring or a compiled regex. Hint is what to read into the result if the
# predicate doesn't match.
def _inspect(html: str) -> None:
    print("\n──────────── HTML inspection ────────────")

    checks: list[tuple[str, str | re.Pattern, str]] = [
        (
            "Paperclip <button> markup is in the response",
            re.compile(r'<button[^>]*aria-label="Allega documento"', re.I),
            "→ The plain <button> isn't being rendered — server-side issue. "
            "Check that the lesson.status is 'draft' and that "
            "partials/chat_input.html includes the draft branch.",
        ),
        (
            "Paperclip wa-icon is in the response",
            "<wa-icon name=\"paperclip\"",
            "→ The icon element itself is missing — likely a templating bug.",
        ),
        (
            "Hidden file input is in the response",
            'id="chat-upload-input"',
            "→ The file input piece of the upload pipeline is missing.",
        ),
        (
            "wa-tooltip wraps the chat-input action buttons",
            re.compile(r'<wa-tooltip[^>]*>\s*<button', re.I | re.S),
            "→ The tooltip wrapper isn't there. That would be unusual.",
        ),
        (
            "Send button (<wa-button … type=\"submit\">) is rendered",
            re.compile(r'<wa-button[^>]*type="submit"', re.I),
            "→ Send button missing too — bigger problem than just the paperclip.",
        ),
        (
            "Lesson is in 'draft' status (chat input branch)",
            # The status badge is rendered as <wa-tag>draft</wa-tag> by
            # lesson_show.html. The previous regex looked for <span>draft</span>
            # which the codebase never emits — false negative even on a
            # genuine draft page (cf. ClickUp #6.6 changelog 10.4).
            re.compile(
                r"<wa-tag[^>]*>\s*draft\s*</wa-tag>"
                r"|chat-upload-input"  # implies draft-branch markup is present
                r"|hx-post=\"/webui/lesson/[^\"]+/run\"",  # chat-input form on draft
                re.I,
            ),
            "→ Status isn't 'draft' on this page; the chat-input file-picker "
            "branch only renders for draft lessons.",
        ),
    ]

    for label, predicate, hint in checks:
        if isinstance(predicate, re.Pattern):
            ok = bool(predicate.search(html))
        else:
            ok = predicate in html
        marker = "✅" if ok else "❌"
        print(f"  {marker}  {label}")
        if not ok:
            print(f"       {hint}")

    # Surface the first occurrence of the chat-input action-button block so
    # we can eyeball the actual markup without opening the file.
    print("\n──────────── Chat-input action block (excerpt) ────────────")
    m = re.search(
        r'(<input type="file"[^>]*id="chat-upload-input".*?</wa-tooltip>'
        r'\s*<wa-tooltip[^>]*>\s*<button[^>]*disabled[^>]*>.*?</wa-tooltip>)',
        html,
        flags=re.S | re.I,
    )
    if m:
        excerpt = m.group(1)
        # Trim the typical file-input attribute soup so the relevant bit fits.
        for line in excerpt.splitlines():
            print(f"    {line.rstrip()}")
    else:
        print(
            "    (could not locate the action-block region — meaning the "
            "paperclip / file-input markup probably isn't in the HTML at "
            "all; bug is server-side, not browser-side)"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect rendered chat input.")
    ap.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"WebUI base URL (default: {DEFAULT_BASE_URL})",
    )
    ap.add_argument(
        "--lesson-id",
        default=None,
        help="Specific lesson UUID to inspect (otherwise auto-detect).",
    )
    ap.add_argument(
        "--email",
        default=None,
        help="Login email (default: prompt interactively).",
    )
    args = ap.parse_args()

    print(f"🔍 WebUI chat-input diagnostic — target {args.base_url}")
    email = args.email or input("  Email: ").strip()
    password = getpass.getpass("  Password (input hidden): ")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(base_url=args.base_url, timeout=15.0) as client:
        _login(client, email, password)
        lesson_id = _pick_lesson_id(client, args.lesson_id)
        html = _fetch_lesson(client, lesson_id)

    OUT_FILE.write_text(html, encoding="utf-8")
    print(f"\n💾 Full HTML saved to: {OUT_FILE.relative_to(REPO_ROOT)}")
    print(
        "   (you can share that file's contents with the assistant — it'll "
        "read it directly via the Read tool to confirm what's being served)."
    )

    _inspect(html)


if __name__ == "__main__":
    main()
