"""Server-side HTML sanitisation for rendered lesson content (Fase 0 · PR 1).

The lesson body reaches the browser as HTML, marked ``| safe`` in four
templates, having been produced by ``markdown.markdown``. python-markdown
passes raw HTML through by design — it removed ``safe_mode`` in 3.0 and its own
documentation says to sanitise the output instead — so anything that looks like
a tag in the Markdown arrives in the page as a tag.

That would be fine if the Markdown were ours. It is not. It is written by an
LLM, from a corpus, from the teacher's own typing, and from the text extracted
out of files they upload. Every one of those is a path by which
``<img src=x onerror=...>`` can reach the page, and the session cookie is
sitting right there. The upload path is the one that matters most: a PDF a
teacher was sent by someone else, opened in a tool that renders its extract.

So the HTML is filtered against an allowlist on the way out, in the one place
that produces it. The list below is what a lesson plan actually uses —
headings, emphasis, lists, tables, code, links, images — and nothing else.
Anything outside it is dropped rather than escaped, because a lesson containing
a literal ``<script>`` in its prose is not what the teacher meant either.

nh3 rather than bleach: bleach is in maintenance mode and pulls in html5lib,
while nh3 is the maintained binding to ammonia and needs nothing else. Both
would do; this one is smaller and faster to review.
"""

from __future__ import annotations

import nh3

#: What a lesson plan is made of. Deliberately smaller than any default: an
#: element earns its place by appearing in output the writer actually produces.
ALLOWED_TAGS: set[str] = {
    # structure
    "p",
    "br",
    "hr",
    "div",
    "span",
    "blockquote",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    # emphasis
    "strong",
    "b",
    "em",
    "i",
    "u",
    "s",
    "del",
    "ins",
    "mark",
    "sub",
    "sup",
    # lists
    "ul",
    "ol",
    "li",
    "dl",
    "dt",
    "dd",
    # tables — the writer emits them for assessment rubrics
    "table",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "th",
    "td",
    "caption",
    # code
    "pre",
    "code",
    "kbd",
    "samp",
    "var",
    # links and images
    "a",
    "img",
    "figure",
    "figcaption",
    # footnotes, which the markdown extension set enables
    "section",
    "abbr",
    "cite",
}

#: Attributes, per tag. ``class`` is allowed only where the markdown extensions
#: emit one — code highlighting and footnote anchors — so a crafted document
#: cannot reach for arbitrary styling hooks in the application's own CSS.
ALLOWED_ATTRIBUTES: dict[str, set[str]] = {
    "a": {"href", "title", "id"},
    "img": {"src", "alt", "title", "width", "height"},
    "th": {"align", "colspan", "rowspan", "scope"},
    "td": {"align", "colspan", "rowspan"},
    "ol": {"start"},
    "li": {"id"},
    "code": {"class"},
    "pre": {"class"},
    "div": {"class"},
    "span": {"class"},
    "section": {"class", "id"},
    "sup": {"id"},
    "abbr": {"title"},
}

#: Schemes a link or an image may use. ``data:`` is absent deliberately: a
#: ``data:text/html`` URL is a script delivery mechanism, and no legitimate
#: lesson needs one. ``mailto`` stays because a lesson may cite a contact.
ALLOWED_SCHEMES: set[str] = {"http", "https", "mailto"}


def sanitize_html(html: str) -> str:
    """Filter rendered HTML down to what a lesson is allowed to contain.

    Returns ``""`` for empty input rather than ``None``, so callers can drop the
    result straight into a template without a guard.

    Link targets get ``rel="noopener noreferrer"`` and open in a new tab: a
    lesson can cite an external resource, and a cited page must not be able to
    reach back into the tab it was opened from through ``window.opener``.
    """
    if not html:
        return ""

    return nh3.clean(
        html,
        tags=ALLOWED_TAGS,
        attributes={tag: set(attrs) for tag, attrs in ALLOWED_ATTRIBUTES.items()},
        url_schemes=ALLOWED_SCHEMES,
        link_rel="noopener noreferrer",
        strip_comments=True,
    )
