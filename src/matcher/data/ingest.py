from __future__ import annotations

import json
from pathlib import Path

from matcher.data.schema import JobDescription, Resume


def load_jobs(path: Path) -> list[JobDescription]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        JobDescription(
            job_id=r["job_id"],
            title=r["title"],
            seniority=r.get("seniority"),
            raw_text=r["text"],
        )
        for r in rows
    ]


def load_resumes(path: Path) -> list[Resume]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        Resume(
            candidate_id=r["candidate_id"],
            raw_text=r["text"],
            summary=r.get("summary"),
        )
        for r in rows
    ]
