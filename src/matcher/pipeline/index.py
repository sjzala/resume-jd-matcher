from __future__ import annotations

from pathlib import Path

from matcher.data.ingest import load_jobs
from matcher.models.bi_encoder import BiEncoderMatcher

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    jobs = load_jobs(ROOT / "data" / "processed" / "jobs.json")
    docs = {j.job_id: j.raw_text for j in jobs}
    matcher = BiEncoderMatcher()
    matcher.build_index(docs)
    matcher.save(ROOT / "data" / "index")
    print(f"indexed {len(docs)} jobs")


if __name__ == "__main__":
    main()
