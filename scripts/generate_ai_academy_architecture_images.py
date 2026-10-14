"""Generate presentation-ready LIGHT and STRONG Academy architecture diagrams."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH = 1920
HEIGHT = 1080
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "docs" / "product"

BG = "#F7F6F2"
WHITE = "#FFFFFF"
INK = "#171717"
MUTED = "#5B5B58"
LINE = "#C9C6BE"
GREEN = "#00A878"
GREEN_PALE = "#E8F7F0"
YELLOW = "#F0B52D"
YELLOW_PALE = "#FFF3D2"
BLUE = "#3478B8"
BLUE_PALE = "#EAF2FA"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        [Path(r"C:\Windows\Fonts\segoeuib.ttf"), Path(r"C:\Windows\Fonts\arialbd.ttf")]
        if bold
        else [Path(r"C:\Windows\Fonts\segoeui.ttf"), Path(r"C:\Windows\Fonts\arial.ttf")]
    )
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


F_EYEBROW = font(23, bold=True)
F_TITLE = font(56, bold=True)
F_SUBTITLE = font(27)
F_CARD_TITLE = font(29, bold=True)
F_CARD_BODY = font(25)
F_CARD_BODY_SMALL = font(22)
F_NUMBER = font(21, bold=True)
F_TECH_LABEL = font(24, bold=True)
F_TECH = font(23)
F_TRUST = font(24, bold=True)
F_FOOTER = font(20)


def text_width(draw: ImageDraw.ImageDraw, text: str, selected_font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=selected_font)
    return box[2] - box[0]


def wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    selected_font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_width(draw, candidate, selected_font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    selected_font: ImageFont.ImageFont,
    fill: str,
    max_width: int,
    line_gap: int = 7,
) -> int:
    x, y = xy
    line_height = selected_font.size + line_gap
    for line in wrap_text(draw, text, selected_font, max_width):
        draw.text((x, y), line, font=selected_font, fill=fill)
        y += line_height
    return y


def draw_header(
    draw: ImageDraw.ImageDraw,
    model: str,
    descriptor: str,
    subtitle: str,
    accent: str,
) -> None:
    draw.text((88, 62), "AI-POWERED ACADEMY", font=F_EYEBROW, fill=MUTED)
    model_text = f"{model} | "
    draw.text((88, 101), model_text, font=F_TITLE, fill=accent)
    x_descriptor = 88 + text_width(draw, model_text, F_TITLE)
    draw.text((x_descriptor, 101), descriptor, font=F_TITLE, fill=INK)
    draw.text((90, 178), subtitle, font=F_SUBTITLE, fill=MUTED)
    draw.line((90, 224, 1830, 224), fill=LINE, width=2)


def draw_arrow(draw: ImageDraw.ImageDraw, x1: int, x2: int, y: int) -> None:
    draw.line((x1, y, x2 - 13, y), fill=INK, width=4)
    draw.polygon([(x2 - 13, y - 9), (x2, y), (x2 - 13, y + 9)], fill=INK)


def draw_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    number: str,
    title: str,
    bullets: Iterable[str],
    accent: str,
    pale: str,
    body_font: ImageFont.ImageFont = F_CARD_BODY,
) -> None:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=22, fill=WHITE, outline=LINE, width=2)
    draw.rounded_rectangle((x1, y1, x2, y1 + 12), radius=8, fill=accent)
    draw.rounded_rectangle((x1 + 24, y1 + 31, x1 + 69, y1 + 72), radius=13, fill=pale)
    number_x = x1 + 46 - text_width(draw, number, F_NUMBER) // 2
    draw.text((number_x, y1 + 39), number, font=F_NUMBER, fill=accent)
    title_x = x1 + 82
    title_y = y1 + 34
    title_lines = wrap_text(draw, title, F_CARD_TITLE, x2 - title_x - 22)
    for title_line in title_lines:
        draw.text((title_x, title_y), title_line, font=F_CARD_TITLE, fill=INK)
        title_y += 34

    y = max(y1 + 102, title_y + 13)
    for bullet in bullets:
        draw.ellipse((x1 + 27, y + 10, x1 + 35, y + 18), fill=accent)
        y = draw_wrapped(
            draw,
            (x1 + 49, y),
            bullet,
            body_font,
            INK,
            x2 - x1 - 75,
            line_gap=6,
        )
        y += 13


def draw_trust_band(draw: ImageDraw.ImageDraw, y1: int, y2: int, text: str) -> None:
    draw.rounded_rectangle((90, y1, 1830, y2), radius=18, fill=INK)
    label = "TRUST & GOVERNANCE"
    draw.text((120, y1 + 19), label, font=F_TRUST, fill=YELLOW)
    x = 120 + text_width(draw, label, F_TRUST) + 35
    draw.text((x, y1 + 19), text, font=F_TRUST, fill=WHITE)


def draw_tech_row(
    draw: ImageDraw.ImageDraw,
    y1: int,
    label: str,
    text: str,
    label_fill: str,
    row_fill: str,
    tech_font: ImageFont.ImageFont = F_TECH,
) -> None:
    y2 = y1 + 86
    draw.rounded_rectangle((90, y1, 1830, y2), radius=18, fill=row_fill, outline=LINE, width=2)
    draw.rounded_rectangle((106, y1 + 14, 395, y2 - 14), radius=14, fill=label_fill)
    label_x = 250 - text_width(draw, label, F_TECH_LABEL) // 2
    draw.text((label_x, y1 + 28), label, font=F_TECH_LABEL, fill=WHITE)
    lines = wrap_text(draw, text, tech_font, 1380)
    start_y = y1 + 15 if len(lines) > 1 else y1 + 28
    for line in lines[:2]:
        draw.text((430, start_y), line, font=tech_font, fill=INK)
        start_y += tech_font.size + 7


def draw_footer(draw: ImageDraw.ImageDraw) -> None:
    note = (
        "Illustrative reference architecture — final technologies depend on discovery, "
        "client infrastructure, data sensitivity, corpus and measurable use case."
    )
    draw.text((90, 1031), note, font=F_FOOTER, fill=MUTED)
    tag = "FEM AI Team"
    draw.text((1830 - text_width(draw, tag, F_FOOTER), 1031), tag, font=F_FOOTER, fill=MUTED)


def render_light() -> Path:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw_header(
        draw,
        "LIGHT",
        "Focused Pilot",
        "Selected corpus • One priority use case • One or two FEM tools",
        GREEN,
    )

    card_y1, card_y2 = 266, 640
    card_width, gap, x = 390, 40, 120
    cards = [
        (
            "01",
            "Selected corporate sources",
            ["Approved documents", "LMS or Drive/SharePoint folder", "One approved API if needed"],
            GREEN,
            GREEN_PALE,
        ),
        (
            "02",
            "Secure knowledge processing",
            [
                "Validate and quarantine",
                "Parse documents and basic OCR",
                "Metadata, permissions and versions",
            ],
            YELLOW,
            YELLOW_PALE,
        ),
        (
            "03",
            "Configurable AI engine",
            [
                "Governed search or RAG when useful",
                "Grounded answers with citations",
                "Model-independent orchestration",
                "Human review",
            ],
            GREEN,
            GREEN_PALE,
        ),
        (
            "04",
            "FEM applications & tools",
            [
                "Knowledge Tester",
                "Learning Assistant / chatbot",
                "AI for Learning or versioned API",
            ],
            YELLOW,
            YELLOW_PALE,
        ),
    ]
    for index, (number, title, bullets, accent, pale) in enumerate(cards):
        x1 = x + index * (card_width + gap)
        draw_card(
            draw,
            (x1, card_y1, x1 + card_width, card_y2),
            number,
            title,
            bullets,
            accent,
            pale,
        )
        if index < len(cards) - 1:
            draw_arrow(draw, x1 + card_width + 9, x1 + card_width + gap - 9, 452)

    draw_trust_band(
        draw,
        675,
        740,
        "Private corpus • Access controlled • Traceable sources • Human approval",
    )
    draw_tech_row(
        draw,
        770,
        "PORTABLE / SELF-HOSTED",
        "FastAPI • Docling / Tesseract • PostgreSQL + pgvector • LangGraph • "
        "SeaweedFS or Garage • Client IdP / Keycloak",
        GREEN,
        GREEN_PALE,
    )
    draw_tech_row(
        draw,
        875,
        "GOOGLE CLOUD MANAGED",
        "Cloud Run • Cloud Storage • Document AI • Cloud SQL + pgvector • "
        "Cloud Tasks • Vertex AI • IAM / IAP",
        BLUE,
        BLUE_PALE,
    )
    draw_footer(draw)

    output = OUTPUT_DIR / "AI_Academy_LIGHT_Architecture.png"
    image.save(output, format="PNG", optimize=True)
    return output


def render_strong() -> Path:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw_header(
        draw,
        "STRONG",
        "Governed Academy",
        "Multiple governed sources • Enterprise controls • Content lifecycle • Multiple tools",
        YELLOW,
    )

    card_y1, card_y2 = 258, 650
    card_width, gap, x = 320, 25, 110
    cards = [
        (
            "01",
            "Corporate ecosystem",
            [
                "LMS / LXP and HRIS",
                "CMS / DAM and intranet",
                "Drive / SharePoint",
                "Approved APIs and events",
            ],
            GREEN,
            GREEN_PALE,
        ),
        (
            "02",
            "Connect & govern",
            [
                "ACL-aware connectors",
                "Quarantine and classification",
                "Ownership and versioning",
                "Lineage and retention",
            ],
            YELLOW,
            YELLOW_PALE,
        ),
        (
            "03",
            "Enterprise knowledge layer",
            [
                "Tenant-scoped hybrid RAG",
                "Citations and provenance",
                "Optional graph after validation",
                "Continuous content lifecycle",
            ],
            GREEN,
            GREEN_PALE,
        ),
        (
            "04",
            "Agent & tool orchestration",
            [
                "Plan, retrieve, generate, review",
                "Policy gateway and MCP tools",
                "Human approval and evaluation",
                "Model-independent routing",
            ],
            YELLOW,
            YELLOW_PALE,
        ),
        (
            "05",
            "Academy apps & integrations",
            [
                "AI for Learning / Academy portal",
                "Tester and learning assistant",
                "LMS / HRIS / Workspace",
                "Versioned APIs and events",
            ],
            GREEN,
            GREEN_PALE,
        ),
    ]
    for index, (number, title, bullets, accent, pale) in enumerate(cards):
        x1 = x + index * (card_width + gap)
        draw_card(
            draw,
            (x1, card_y1, x1 + card_width, card_y2),
            number,
            title,
            bullets,
            accent,
            pale,
            body_font=F_CARD_BODY_SMALL,
        )
        if index < len(cards) - 1:
            draw_arrow(draw, x1 + card_width + 5, x1 + card_width + gap - 5, 454)

    draw_trust_band(
        draw,
        680,
        745,
        "SSO • Tenant isolation • RBAC / ABAC • Encryption • Audit • SLO • Backup / DR",
    )
    draw_tech_row(
        draw,
        775,
        "PORTABLE / SELF-HOSTED",
        "FastAPI • LangGraph / FastMCP • PostgreSQL + pgvector • optional Neo4j / AGE • "
        "SeaweedFS / Ceph • Keycloak / OpenBao • OpenTelemetry / Grafana",
        GREEN,
        GREEN_PALE,
        tech_font=font(21),
    )
    draw_tech_row(
        draw,
        880,
        "GOOGLE CLOUD MANAGED",
        "Cloud Run / GKE • Cloud Storage • Document AI • Cloud SQL / AlloyDB • Vertex AI • "
        "optional Vector Search / Spanner Graph • IAM / KMS / Model Armor / Cloud Operations",
        BLUE,
        BLUE_PALE,
        tech_font=font(21),
    )
    draw_footer(draw)

    output = OUTPUT_DIR / "AI_Academy_STRONG_Architecture.png"
    image.save(output, format="PNG", optimize=True)
    return output


if __name__ == "__main__":
    for generated in (render_light(), render_strong()):
        print(generated)
