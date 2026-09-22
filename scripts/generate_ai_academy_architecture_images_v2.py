"""Generate infographic-style LIGHT and STRONG Academy architecture diagrams."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


W, H = 1920, 1080
OUT = Path(__file__).resolve().parents[1] / "docs" / "product"

BG = "#F7F7F4"
WHITE = "#FFFFFF"
INK = "#1A1B1D"
MUTED = "#60646B"
LINE = "#C8CBC8"
GREEN = "#00A878"
GREEN_DARK = "#087A5B"
GREEN_PALE = "#E9F8F2"
GOLD = "#F2B632"
GOLD_DARK = "#A86D00"
GOLD_PALE = "#FFF4D6"
BLUE = "#397DB8"
BLUE_DARK = "#265C87"
BLUE_PALE = "#EAF3FA"
PURPLE = "#7A5AA6"
PURPLE_PALE = "#F1EBF8"
RED_PALE = "#FCEDEB"


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    for name in names:
        path = Path(r"C:\Windows\Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


F_EYE = get_font(21, True)
F_H1 = get_font(52, True)
F_SUB = get_font(25)
F_GROUP = get_font(25, True)
F_NODE = get_font(22, True)
F_BODY = get_font(20)
F_SMALL = get_font(18)
F_TINY = get_font(16)
F_STACK_HEAD = get_font(21, True)
F_STACK = get_font(19)


def width(draw: ImageDraw.ImageDraw, text: str, selected_font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=selected_font)
    return box[2] - box[0]


def wrap(
    draw: ImageDraw.ImageDraw,
    text: str,
    selected_font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = word if not current else f"{current} {word}"
        if width(draw, candidate, selected_font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def center_text(
    draw: ImageDraw.ImageDraw,
    cx: int,
    y: int,
    text: str,
    selected_font: ImageFont.ImageFont,
    fill: str,
) -> None:
    draw.text((cx - width(draw, text, selected_font) // 2, y), text, font=selected_font, fill=fill)


def header(
    draw: ImageDraw.ImageDraw,
    model: str,
    descriptor: str,
    subtitle: str,
    accent: str,
) -> None:
    draw.text((72, 43), "AI-POWERED ACADEMY  /  REFERENCE ARCHITECTURE", font=F_EYE, fill=MUTED)
    left = f"{model} | "
    draw.text((72, 78), left, font=F_H1, fill=accent)
    draw.text((72 + width(draw, left, F_H1), 78), descriptor, font=F_H1, fill=INK)
    draw.text((74, 149), subtitle, font=F_SUB, fill=MUTED)
    draw.line((72, 190, 1848, 190), fill=LINE, width=2)


def rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str,
    outline: str = LINE,
    radius: int = 18,
    line_width: int = 2,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=line_width)


def arrow(
    draw: ImageDraw.ImageDraw,
    points: Sequence[tuple[int, int]],
    color: str = INK,
    line_width: int = 4,
    dashed: bool = False,
    both: bool = False,
) -> None:
    if dashed:
        for start, end in zip(points, points[1:]):
            x1, y1 = start
            x2, y2 = end
            length = max(abs(x2 - x1), abs(y2 - y1))
            if length == 0:
                continue
            for pos in range(0, length, 18):
                end_pos = min(pos + 10, length)
                sx = int(x1 + (x2 - x1) * pos / length)
                sy = int(y1 + (y2 - y1) * pos / length)
                ex = int(x1 + (x2 - x1) * end_pos / length)
                ey = int(y1 + (y2 - y1) * end_pos / length)
                draw.line((sx, sy, ex, ey), fill=color, width=line_width)
    else:
        draw.line(points, fill=color, width=line_width, joint="curve")

    def head(previous: tuple[int, int], tip: tuple[int, int]) -> None:
        px, py = previous
        tx, ty = tip
        if abs(tx - px) >= abs(ty - py):
            direction = 1 if tx > px else -1
            draw.polygon(
                [(tx, ty), (tx - 13 * direction, ty - 8), (tx - 13 * direction, ty + 8)],
                fill=color,
            )
        else:
            direction = 1 if ty > py else -1
            draw.polygon(
                [(tx, ty), (tx - 8, ty - 13 * direction), (tx + 8, ty - 13 * direction)],
                fill=color,
            )

    if len(points) >= 2:
        head(points[-2], points[-1])
        if both:
            head(points[1], points[0])


def icon(draw: ImageDraw.ImageDraw, kind: str, cx: int, cy: int, color: str, scale: int = 34) -> None:
    r = scale // 2
    if kind == "person":
        draw.ellipse((cx - r // 2, cy - r, cx + r // 2, cy), fill=color)
        draw.rounded_rectangle((cx - r, cy + 3, cx + r, cy + r + 13), radius=10, fill=color)
    elif kind == "files":
        for offset in (8, 0):
            draw.rounded_rectangle(
                (cx - r + offset, cy - r - offset, cx + r + offset, cy + r - offset),
                radius=4,
                outline=color,
                width=3,
            )
            draw.line((cx - r + 8 + offset, cy - 3 - offset, cx + r - 7 + offset, cy - 3 - offset), fill=color, width=3)
            draw.line((cx - r + 8 + offset, cy + 7 - offset, cx + r - 11 + offset, cy + 7 - offset), fill=color, width=3)
    elif kind == "shield":
        draw.polygon(
            [(cx, cy - r - 4), (cx + r, cy - r // 2), (cx + r - 4, cy + r // 2), (cx, cy + r + 7), (cx - r + 4, cy + r // 2), (cx - r, cy - r // 2)],
            outline=color,
            fill=None,
        )
        draw.line((cx - 7, cy, cx - 1, cy + 7, cx + 11, cy - 8), fill=color, width=4)
    elif kind in {"database", "memory"}:
        draw.ellipse((cx - r, cy - r, cx + r, cy - r // 2), outline=color, width=3)
        draw.rectangle((cx - r, cy - 3 * r // 4, cx + r, cy + 3 * r // 4), outline=color, width=3)
        draw.arc((cx - r, cy + r // 2, cx + r, cy + r), 0, 180, fill=color, width=3)
        if kind == "memory":
            draw.line((cx - r + 7, cy, cx + r - 7, cy), fill=color, width=2)
    elif kind == "chip":
        draw.rounded_rectangle((cx - r, cy - r, cx + r, cy + r), radius=7, outline=color, width=3)
        draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=color)
        for delta in (-11, 0, 11):
            draw.line((cx - r - 7, cy + delta, cx - r, cy + delta), fill=color, width=3)
            draw.line((cx + r, cy + delta, cx + r + 7, cy + delta), fill=color, width=3)
            draw.line((cx + delta, cy - r - 7, cx + delta, cy - r), fill=color, width=3)
            draw.line((cx + delta, cy + r, cx + delta, cy + r + 7), fill=color, width=3)
    elif kind == "apps":
        for dx in (-12, 12):
            for dy in (-12, 12):
                draw.rounded_rectangle((cx + dx - 8, cy + dy - 8, cx + dx + 8, cy + dy + 8), radius=4, fill=color)
    elif kind == "bot":
        draw.rounded_rectangle((cx - r, cy - r + 4, cx + r, cy + r), radius=9, outline=color, width=3)
        draw.ellipse((cx - 11, cy - 4, cx - 5, cy + 2), fill=color)
        draw.ellipse((cx + 5, cy - 4, cx + 11, cy + 2), fill=color)
        draw.line((cx - 10, cy + 11, cx + 10, cy + 11), fill=color, width=3)
        draw.line((cx, cy - r + 4, cx, cy - r - 8), fill=color, width=3)
        draw.ellipse((cx - 4, cy - r - 13, cx + 4, cy - r - 5), fill=color)
    elif kind == "graph":
        pts = [(cx - 16, cy + 13), (cx - 5, cy - 15), (cx + 17, cy - 3), (cx + 11, cy + 18)]
        draw.line((*pts[0], *pts[1], *pts[2], *pts[3], *pts[0]), fill=color, width=3)
        for px, py in pts:
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color)
    elif kind == "plan":
        draw.rounded_rectangle((cx - r, cy - r, cx + r, cy + r), radius=5, outline=color, width=3)
        for dy in (-11, 0, 11):
            draw.line((cx - 9, cy + dy, cx - 4, cy + dy + 4, cx + 3, cy + dy - 5), fill=color, width=2)
            draw.line((cx + 8, cy + dy, cx + r - 5, cy + dy), fill=color, width=2)
    elif kind == "tools":
        draw.line((cx - 17, cy + 17, cx + 12, cy - 12), fill=color, width=7)
        draw.arc((cx + 2, cy - 22, cx + 24, cy), 35, 230, fill=color, width=5)
        draw.ellipse((cx - 22, cy + 7, cx - 8, cy + 21), outline=color, width=4)
    elif kind == "cloud":
        draw.ellipse((cx - 25, cy - 5, cx + 5, cy + 20), outline=color, width=3)
        draw.ellipse((cx - 7, cy - 18, cx + 24, cy + 18), outline=color, width=3)
        draw.line((cx - 22, cy + 19, cx + 24, cy + 19), fill=color, width=3)


def titled_group(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    accent: str,
    pale: str,
    icon_kind: str,
) -> None:
    x1, y1, x2, _ = box
    rounded(draw, box, WHITE, accent, radius=20, line_width=3)
    draw.rounded_rectangle((x1, y1, x2, y1 + 54), radius=17, fill=pale)
    draw.rectangle((x1, y1 + 35, x2, y1 + 54), fill=pale)
    icon(draw, icon_kind, x1 + 34, y1 + 27, accent, scale=25)
    draw.text((x1 + 64, y1 + 14), title, font=F_GROUP, fill=INK)


def compact_node(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    accent: str,
    icon_kind: str,
    fill: str = WHITE,
) -> None:
    x1, y1, x2, y2 = box
    rounded(draw, box, fill, LINE, radius=15)
    icon(draw, icon_kind, x1 + 31, (y1 + y2) // 2, accent, scale=25)
    draw.text((x1 + 62, y1 + 12), title, font=F_NODE, fill=INK)
    if subtitle:
        lines = wrap(draw, subtitle, F_SMALL, x2 - x1 - 78)
        sy = y1 + 43
        for line in lines[:2]:
            draw.text((x1 + 62, sy), line, font=F_SMALL, fill=MUTED)
            sy += 22


def stack_profiles(
    draw: ImageDraw.ImageDraw,
    portable_lines: Sequence[str],
    cloud_lines: Sequence[str],
) -> None:
    draw.text((72, 853), "IMPLEMENTATION PROFILES", font=F_EYE, fill=MUTED)
    cards = [
        ((72, 885, 945, 1014), "PORTABLE / SELF-HOSTED", portable_lines, GREEN, GREEN_PALE, "database"),
        ((975, 885, 1848, 1014), "GOOGLE CLOUD MANAGED", cloud_lines, BLUE, BLUE_PALE, "cloud"),
    ]
    for box, label, lines, accent, pale, icon_kind in cards:
        x1, y1, x2, _ = box
        rounded(draw, box, pale, LINE, radius=18)
        icon(draw, icon_kind, x1 + 38, y1 + 36, accent, scale=24)
        draw.text((x1 + 72, y1 + 20), label, font=F_STACK_HEAD, fill=accent)
        y = y1 + 58
        for line in lines:
            draw.text((x1 + 28, y), line, font=F_STACK, fill=INK)
            y += 27


def footer(draw: ImageDraw.ImageDraw) -> None:
    text = "Illustrative architecture. Final scope and technologies are selected after client discovery."
    draw.text((72, 1040), text, font=F_TINY, fill=MUTED)
    tag = "FEM AI Team"
    draw.text((1848 - width(draw, tag, F_TINY), 1040), tag, font=F_TINY, fill=MUTED)


def light_v2() -> Path:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    header(
        draw,
        "LIGHT",
        "Focused AI Pilot",
        "A selected corporate corpus transformed into one measurable AI learning use case",
        GREEN,
    )

    rounded(draw, (72, 213, 1848, 825), WHITE, GREEN, radius=28, line_width=3)
    draw.rounded_rectangle((98, 196, 350, 232), radius=12, fill=GREEN)
    center_text(draw, 224, 202, "LIGHT ARCHITECTURE", F_EYE, WHITE)

    # User / learner lane
    titled_group(draw, (100, 294, 326, 696), "User", GREEN, GREEN_PALE, "person")
    icon(draw, "person", 213, 396, GREEN_DARK, scale=50)
    center_text(draw, 213, 456, "Employee / learner", F_NODE, INK)
    center_text(draw, 213, 493, "asks a question", F_BODY, MUTED)
    center_text(draw, 213, 523, "or starts an activity", F_BODY, MUTED)
    rounded(draw, (127, 578, 299, 642), GREEN_PALE, GREEN, radius=14)
    center_text(draw, 213, 588, "Grounded", F_NODE, GREEN_DARK)
    center_text(draw, 213, 615, "response", F_NODE, GREEN_DARK)

    # Selected data sources
    titled_group(draw, (390, 245, 1410, 355), "Selected corporate data sources", BLUE, BLUE_PALE, "files")
    compact_node(draw, (420, 305, 715, 340), "Documents", "", BLUE, "files", BLUE_PALE)
    compact_node(draw, (745, 305, 1050, 340), "LMS / Drive folder", "", BLUE, "database", BLUE_PALE)
    compact_node(draw, (1080, 305, 1380, 340), "Approved API", "", BLUE, "tools", BLUE_PALE)

    # Secure processing
    titled_group(draw, (430, 395, 1370, 510), "Secure knowledge processing", GOLD, GOLD_PALE, "shield")
    compact_node(draw, (458, 456, 730, 494), "Validate & quarantine", "", GOLD_DARK, "shield", GOLD_PALE)
    compact_node(draw, (764, 456, 1037, 494), "Parse / OCR", "", GOLD_DARK, "files", GOLD_PALE)
    compact_node(draw, (1070, 456, 1342, 494), "Metadata & ACL", "", GOLD_DARK, "plan", GOLD_PALE)

    # Private knowledge workspace
    titled_group(draw, (470, 550, 1330, 663), "Private knowledge workspace", BLUE, BLUE_PALE, "database")
    compact_node(draw, (498, 610, 750, 648), "Versioned originals", "", BLUE_DARK, "files", BLUE_PALE)
    compact_node(draw, (774, 610, 1050, 648), "Private search index", "", BLUE_DARK, "database", BLUE_PALE)
    compact_node(draw, (1074, 610, 1302, 648), "Source citations", "", BLUE_DARK, "graph", BLUE_PALE)

    # AI engine
    titled_group(draw, (430, 708, 1370, 796), "Configurable AI learning engine", GREEN, GREEN_PALE, "chip")
    draw.text(
        (477, 767),
        "Governed search / RAG when useful  •  Model gateway  •  Human review",
        font=F_BODY,
        fill=INK,
    )

    # FEM tools output
    titled_group(draw, (1480, 292, 1818, 696), "FEM tools", GOLD, GOLD_PALE, "apps")
    compact_node(draw, (1512, 372, 1786, 452), "Knowledge Tester", "Practice and assessment", GOLD_DARK, "plan", GOLD_PALE)
    compact_node(draw, (1512, 474, 1786, 554), "Learning Assistant", "Grounded dialogue", GOLD_DARK, "bot", GOLD_PALE)
    compact_node(draw, (1512, 576, 1786, 656), "Delivery", "AI for Learning / API", GOLD_DARK, "apps", GOLD_PALE)

    # Flow
    arrow(draw, [(900, 355), (900, 395)], GREEN_DARK)
    arrow(draw, [(900, 510), (900, 550)], GREEN_DARK)
    arrow(draw, [(900, 663), (900, 708)], GREEN_DARK, both=True)
    arrow(draw, [(326, 540), (375, 540), (375, 749), (430, 749)], GREEN_DARK)
    draw.text((382, 717), "query", font=F_SMALL, fill=MUTED)
    arrow(draw, [(1370, 749), (1440, 749), (1440, 616), (1480, 616)], GREEN_DARK)
    draw.text((1383, 717), "result", font=F_SMALL, fill=MUTED)
    arrow(draw, [(1480, 660), (1430, 660), (1430, 685), (326, 685)], GOLD_DARK, dashed=True)

    draw.rounded_rectangle((470, 672, 1330, 700), radius=10, fill=INK)
    center_text(
        draw,
        900,
        676,
        "PRIVATE  •  ACCESS-CONTROLLED  •  AUDITABLE",
        F_SMALL,
        WHITE,
    )

    stack_profiles(
        draw,
        [
            "FastAPI • Docling / Tesseract • PostgreSQL + pgvector",
            "LangGraph • SeaweedFS / Garage • Client IdP / Keycloak",
        ],
        [
            "Cloud Run • Cloud Storage • Document AI • Cloud SQL + pgvector",
            "Cloud Tasks • Vertex AI • IAM / IAP",
        ],
    )
    footer(draw)
    target = OUT / "AI_Academy_LIGHT_Architecture_v2.png"
    image.save(target, "PNG", optimize=True)
    return target


def agent_node(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    accent: str = PURPLE,
) -> None:
    x1, y1, x2, y2 = box
    rounded(draw, box, PURPLE_PALE, accent, radius=15, line_width=2)
    icon(draw, "bot", x1 + 34, (y1 + y2) // 2, accent, scale=23)
    draw.text((x1 + 67, y1 + 21), title, font=F_NODE, fill=INK)


def strong_v2() -> Path:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    header(
        draw,
        "STRONG",
        "Governed Agentic Academy",
        "A governed knowledge and agent platform connecting multiple corporate systems and learning tools",
        GOLD,
    )

    rounded(draw, (72, 213, 1848, 825), WHITE, PURPLE, radius=28, line_width=3)
    draw.rounded_rectangle((98, 196, 372, 232), radius=12, fill=PURPLE)
    center_text(draw, 235, 202, "STRONG ARCHITECTURE", F_EYE, WHITE)

    # Users / Academy
    titled_group(draw, (96, 312, 326, 681), "Academy", GREEN, GREEN_PALE, "person")
    icon(draw, "person", 211, 404, GREEN_DARK, scale=48)
    center_text(draw, 211, 463, "Learners", F_NODE, INK)
    center_text(draw, 211, 493, "Content owners", F_BODY, MUTED)
    center_text(draw, 211, 521, "Academy teams", F_BODY, MUTED)
    rounded(draw, (120, 579, 302, 642), GREEN_PALE, GREEN, radius=14)
    center_text(draw, 211, 590, "Personalized", F_NODE, GREEN_DARK)
    center_text(draw, 211, 617, "experience", F_NODE, GREEN_DARK)

    # Cognitive controls
    compact_node(draw, (480, 251, 740, 339), "Memory", "Session and governed state", PURPLE, "memory", PURPLE_PALE)
    compact_node(draw, (776, 251, 1036, 339), "Planning", "Goals, policy and routing", PURPLE, "plan", PURPLE_PALE)

    # Orchestrator
    titled_group(draw, (560, 377, 1000, 490), "Academy orchestrator", PURPLE, PURPLE_PALE, "chip")
    center_text(draw, 780, 441, "Coordinates agents, models and tools", F_BODY, INK)

    # Agents
    agent_node(draw, (400, 534, 675, 604), "Retrieval agent")
    agent_node(draw, (697, 534, 972, 604), "Learning-design agent")
    agent_node(draw, (400, 625, 675, 695), "Evaluation agent")
    agent_node(draw, (697, 625, 972, 695), "Integration agent")

    # Tool hub
    titled_group(draw, (1070, 283, 1388, 604), "Tool & policy hub", GOLD, GOLD_PALE, "tools")
    compact_node(draw, (1100, 357, 1358, 421), "MCP / tool registry", "Controlled actions", GOLD_DARK, "tools", GOLD_PALE)
    compact_node(draw, (1100, 440, 1358, 504), "Guardrails", "Approval and policy", GOLD_DARK, "shield", GOLD_PALE)
    compact_node(draw, (1100, 523, 1358, 587), "Evaluations", "Quality and evidence", GOLD_DARK, "plan", GOLD_PALE)

    # Corporate sources
    titled_group(draw, (1460, 252, 1819, 604), "Corporate systems", BLUE, BLUE_PALE, "files")
    compact_node(draw, (1490, 327, 1789, 391), "Learning", "LMS / LXP • HRIS", BLUE_DARK, "apps", BLUE_PALE)
    compact_node(draw, (1490, 410, 1789, 474), "Knowledge", "CMS / DAM • intranet", BLUE_DARK, "database", BLUE_PALE)
    compact_node(draw, (1490, 493, 1789, 557), "Content", "Drive / SharePoint", BLUE_DARK, "files", BLUE_PALE)

    # Knowledge platform
    titled_group(draw, (1040, 635, 1819, 754), "Governed enterprise knowledge", BLUE, BLUE_PALE, "database")
    compact_node(draw, (1070, 695, 1295, 738), "Content store", "", BLUE_DARK, "files", BLUE_PALE)
    compact_node(draw, (1313, 695, 1552, 738), "Hybrid RAG", "", BLUE_DARK, "database", BLUE_PALE)
    compact_node(draw, (1570, 695, 1789, 738), "Optional graph", "", BLUE_DARK, "graph", BLUE_PALE)

    # Flows inspired by the supplied agentic/multi-agent example
    arrow(draw, [(326, 433), (560, 433)], PURPLE)
    draw.text((357, 400), "goal / query", font=F_SMALL, fill=MUTED)
    arrow(draw, [(560, 471), (340, 471), (340, 610), (326, 610)], GREEN_DARK, dashed=True)
    draw.text((353, 574), "governed output", font=F_SMALL, fill=MUTED)
    arrow(draw, [(610, 339), (610, 377)], PURPLE, both=True)
    arrow(draw, [(906, 339), (906, 377)], PURPLE, both=True)
    arrow(draw, [(680, 490), (555, 534)], PURPLE)
    arrow(draw, [(755, 490), (835, 534)], PURPLE)
    arrow(draw, [(820, 604), (835, 625)], PURPLE, both=True)
    arrow(draw, [(555, 604), (537, 625)], PURPLE, both=True)
    arrow(draw, [(1000, 433), (1070, 433)], GOLD_DARK, both=True)
    arrow(draw, [(1388, 405), (1460, 405)], BLUE_DARK, both=True)
    arrow(draw, [(1388, 548), (1420, 548), (1420, 694), (1460, 694)], BLUE_DARK, dashed=True)
    arrow(draw, [(1070, 695), (1008, 695), (1008, 660), (972, 660)], PURPLE, both=True)

    # Cross-cutting control plane
    draw.rounded_rectangle((96, 774, 1819, 819), radius=13, fill=INK)
    draw.text((122, 786), "CONTROL PLANE", font=F_SMALL, fill=GOLD)
    draw.text(
        (300, 786),
        "SSO • tenant isolation • RBAC / ABAC • encryption • secrets • audit • observability • SLO • backup / DR",
        font=F_SMALL,
        fill=WHITE,
    )

    stack_profiles(
        draw,
        [
            "FastAPI • LangGraph / FastMCP • PostgreSQL + pgvector • optional Neo4j / AGE",
            "SeaweedFS / Ceph • Keycloak / OpenBao • OpenTelemetry / Grafana",
        ],
        [
            "Cloud Run / GKE • Cloud Storage • Document AI • Cloud SQL / AlloyDB • Vertex AI",
            "Optional Vector Search / Spanner Graph • IAM / KMS • Model Armor • Cloud Operations",
        ],
    )
    footer(draw)
    target = OUT / "AI_Academy_STRONG_Architecture_v2.png"
    image.save(target, "PNG", optimize=True)
    return target


def strong_v3(v2_path: Path) -> Path:
    """Create the Qdrant/open-source profile while preserving the v2 layout."""
    image = Image.open(v2_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    box = (72, 885, 945, 1014)
    x1, y1, _, _ = box
    rounded(draw, box, GREEN_PALE, LINE, radius=18)
    icon(draw, "database", x1 + 38, y1 + 36, GREEN, scale=24)
    draw.text((x1 + 72, y1 + 20), "OPEN-SOURCE / SELF-HOSTED", font=F_STACK_HEAD, fill=GREEN)
    draw.text(
        (x1 + 28, y1 + 58),
        "FastAPI • LangGraph / FastMCP • PostgreSQL • Qdrant • optional Neo4j / AGE",
        font=F_STACK,
        fill=INK,
    )
    draw.text(
        (x1 + 28, y1 + 85),
        "SeaweedFS / Ceph • Keycloak / OpenBao • OpenTelemetry / Grafana",
        font=F_STACK,
        fill=INK,
    )

    target = OUT / "AI_Academy_STRONG_Architecture_v3.png"
    image.save(target, "PNG", optimize=True)
    return target


def light_v3(v2_path: Path) -> Path:
    """Rename the LIGHT portable profile while preserving the v2 layout."""
    image = Image.open(v2_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    box = (72, 885, 945, 1014)
    x1, y1, _, _ = box
    rounded(draw, box, GREEN_PALE, LINE, radius=18)
    icon(draw, "database", x1 + 38, y1 + 36, GREEN, scale=24)
    draw.text((x1 + 72, y1 + 20), "OPEN-SOURCE / SELF-HOSTED", font=F_STACK_HEAD, fill=GREEN)
    draw.text(
        (x1 + 28, y1 + 58),
        "FastAPI • Docling / Tesseract • PostgreSQL + pgvector",
        font=F_STACK,
        fill=INK,
    )
    draw.text(
        (x1 + 28, y1 + 85),
        "LangGraph • SeaweedFS / Garage • Client IdP / Keycloak",
        font=F_STACK,
        fill=INK,
    )

    target = OUT / "AI_Academy_LIGHT_Architecture_v3.png"
    image.save(target, "PNG", optimize=True)
    return target


if __name__ == "__main__":
    light_path = light_v2()
    strong_path = strong_v2()
    for path in (light_path, light_v3(light_path), strong_path, strong_v3(strong_path)):
        print(path)
