"""
Pool entry dataclasses and JSON serialization for the verified media pool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import date


@dataclass
class VideoEntry:
    title: str
    video_id: str
    url: str
    embed_url: str
    channel: str
    rights_status: str  # "youtube_embed" | "youtube_cc"
    verified_date: str  # ISO YYYY-MM-DD
    language: str = "en"
    duration_hint: Optional[str] = None
    graph_context: Optional[str] = None
    # Engagement fields — populated by 04_enrich_pool.py
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    duration_seconds: Optional[int] = None
    trusted_channel: bool = False
    quality_score: Optional[float] = None


@dataclass
class CitationEntry:
    title: str
    authors: List[str]
    doi: str
    doi_url: str
    rights_status: str  # "open_access_paper"
    verified_date: str
    year: Optional[int] = None
    journal: Optional[str] = None
    open_access_pdf: Optional[str] = None
    graph_context: Optional[str] = None


@dataclass
class WikipediaEntry:
    title: str
    url: str
    rights_status: str  # "cc_by_sa"
    verified_date: str
    language: str = "en"


@dataclass
class ConceptEntry:
    concept_name: str
    domain: str
    videos: List[VideoEntry] = field(default_factory=list)
    citations: List[CitationEntry] = field(default_factory=list)
    wikipedia: Optional[WikipediaEntry] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "concept_name": self.concept_name,
            "domain": self.domain,
            "videos": [asdict(v) for v in self.videos],
            "citations": [asdict(c) for c in self.citations],
            "wikipedia": asdict(self.wikipedia) if self.wikipedia else None,
        }
        return d

    def has_content(self) -> bool:
        return bool(self.videos or self.citations or self.wikipedia)


def load_pool(path: str) -> Dict[str, Dict]:
    """Load existing pool from disk. Returns empty dict if file doesn't exist."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("entries", {})
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_pool(path: str, domain: str, model: str, entries: Dict[str, Dict]) -> None:
    """Write the full pool dict to disk atomically via temp file."""
    import os
    import tempfile

    payload = {
        "domain": domain,
        "generated_by": model,
        "generated_date": date.today().isoformat(),
        "entries": entries,
    }
    dir_ = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", dir=dir_, suffix=".tmp", delete=False, encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp_path = f.name
    os.replace(tmp_path, path)


def load_checkpoint(path: str) -> Dict[str, str]:
    """Load checkpoint dict {concept_name: 'done'}. Returns empty dict if missing."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_checkpoint(path: str, checkpoint: Dict[str, str]) -> None:
    import os
    import tempfile

    dir_ = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", dir=dir_, suffix=".tmp", delete=False, encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        tmp_path = f.name
    os.replace(tmp_path, path)
