"""Quick inspector: dump a lesson's stored educational_profile_json from SQLite.

Usage: python scripts/diagnostic/inspect_lesson_profile.py <lesson_id>
       python scripts/diagnostic/inspect_lesson_profile.py        # dumps all lessons
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

DB = Path(__file__).resolve().parents[2] / "data" / "webui" / "webui.db"


def main() -> int:
    if not DB.exists():
        print(f"DB not found: {DB}")
        return 1

    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    print(f"Tables in DB: {tables}")
    candidates = [t for t in tables if "lesson" in t.lower()]
    if not candidates:
        print("No 'lesson' table found.")
        return 1
    table = candidates[0]
    print(f"Using table: {table}\n")

    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    print(f"Columns: {cols}\n")

    if len(sys.argv) > 1:
        rows = conn.execute(
            f"SELECT * FROM {table} WHERE id = ?",
            (sys.argv[1],),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 10"
        ).fetchall()

    if not rows:
        print("No lessons found.")
        return 0

    for r in rows:
        print("=" * 80)
        d = dict(r)
        for k, v in d.items():
            if k == "educational_profile_json":
                continue
            print(f"{k:30s}: {v}")
        try:
            profile = json.loads(d.get("educational_profile_json") or "{}")
        except Exception as exc:  # noqa: BLE001
            print(f"PROFILE  : <unparseable JSON: {exc}>")
            continue
        group = profile.get("group") or {}
        room = profile.get("classroom") or {}
        print("--- profile.* ---")
        print("subject_area             :", profile.get("subject_area"))
        print("specific_topic           :", profile.get("specific_topic"))
        print("time_available_minutes   :", profile.get("time_available_minutes"))
        print("--- group.* ---")
        print("group.title              :", group.get("title"))
        print("group.students_number    :", group.get("students_number"))
        print("group.grade              :", group.get("grade"))
        print("group.disabilities       :", group.get("disabilities"))
        print("group.class_features     :", group.get("class_features"))
        print("group.student_attributes :", group.get("student_attributes"))
        print("--- classroom.* ---")
        print("room.title               :", room.get("title"))
        print("room.forniture_mobility  :", room.get("forniture_mobility"))
        print("room.own_device          :", room.get("own_device"))
        print("room.has_lim/wifi/suite  :", room.get("has_lim"), room.get("has_wifi"), room.get("has_suite"), room.get("pc_station"))
        print("--- raw JSON ---")
        print(json.dumps(profile, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
