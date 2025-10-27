from __future__ import annotations

from pathlib import Path

from matcher.data.ingest import load_jobs
from matcher.features.skill_extract import skill_overlap
from matcher.models.bi_encoder import BiEncoderMatcher
from matcher.models.cross_encoder import CrossEncoderReranker

ROOT = Path(__file__).resolve().parents[3]


def score_resume(
    resume_text: str,
    top_k: int = 5,
    use_rerank: bool = True,
) -> list[dict]:
    jobs = load_jobs(ROOT / "data" / "processed" / "jobs.json")
    job_docs = {j.job_id: j.raw_text for j in jobs}
    job_by_id = {j.job_id: j for j in jobs}

    bi = BiEncoderMatcher()
    bi.build_index(job_docs)
    candidates = bi.search(resume_text, k=max(top_k * 4, 20))

    if use_rerank:
        cross = CrossEncoderReranker()
        pairs = [(jid, job_docs[jid]) for jid, _ in candidates]
        candidates = cross.rerank(resume_text, pairs, top_k=top_k)
    else:
        candidates = candidates[:top_k]

    results = []
    for jid, score in candidates:
        matched, missing = skill_overlap(resume_text, job_docs[jid])
        results.append(
            {
                "job_id": jid,
                "title": job_by_id[jid].title,
                "score": float(score),
                "matched_skills": matched,
                "missing_skills": missing,
            }
        )
    return results
