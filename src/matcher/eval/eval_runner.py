from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from matcher.data.ingest import load_jobs, load_resumes
from matcher.eval.metrics import mean_reciprocal_rank, ndcg_at_k, precision_at_k
from matcher.models.bi_encoder import BiEncoderMatcher
from matcher.models.cross_encoder import CrossEncoderReranker
from matcher.models.rerank import LLMReranker
from matcher.models.tfidf_baseline import TfidfMatcher

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed"


@dataclass
class StageResult:
    name: str
    mrr: float | None
    ndcg_at_5: float | None
    p_at_5: float | None
    note: str = ""


def _binary_relevant(eval_set: dict[str, dict[str, float]]) -> dict[str, set[str]]:
    return {cid: {jid for jid, g in jobs.items() if g >= 2} for cid, jobs in eval_set.items()}


def _graded(eval_set: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {cid: dict(jobs) for cid, jobs in eval_set.items()}


def _score_stage(
    name: str,
    retrieved_per_q: list[list[str]],
    eval_set: dict[str, dict[str, float]],
    note: str = "",
) -> StageResult:
    binary = _binary_relevant(eval_set)
    graded = _graded(eval_set)
    candidate_ids = list(eval_set.keys())
    relevant_binary = [binary[cid] for cid in candidate_ids]
    mrr = mean_reciprocal_rank(retrieved_per_q, relevant_binary)
    ndcgs = [
        ndcg_at_k(retrieved_per_q[i], graded[cid], k=5)
        for i, cid in enumerate(candidate_ids)
    ]
    ps = [
        precision_at_k(retrieved_per_q[i], binary[cid], k=5)
        for i, cid in enumerate(candidate_ids)
    ]
    return StageResult(
        name=name,
        mrr=mrr,
        ndcg_at_5=sum(ndcgs) / len(ndcgs),
        p_at_5=sum(ps) / len(ps),
        note=note,
    )


def run() -> list[StageResult]:
    jobs = load_jobs(DATA / "jobs.json")
    resumes = load_resumes(DATA / "resumes.json")
    eval_set = json.loads((DATA / "eval_set.json").read_text(encoding="utf-8"))

    job_docs = {j.job_id: j.raw_text for j in jobs}
    resume_by_id = {r.candidate_id: r for r in resumes}

    results: list[StageResult] = []

    tfidf = TfidfMatcher()
    tfidf.fit(job_docs)
    tfidf_retrieved = [
        [jid for jid, _ in tfidf.search(resume_by_id[cid].raw_text, k=len(job_docs))]
        for cid in eval_set
    ]
    results.append(_score_stage("TF-IDF baseline", tfidf_retrieved, eval_set))

    bi = BiEncoderMatcher()
    bi.build_index(job_docs)
    bi_retrieved = [
        [jid for jid, _ in bi.search(resume_by_id[cid].raw_text, k=len(job_docs))]
        for cid in eval_set
    ]
    results.append(_score_stage("Bi-encoder (mpnet-base-v2)", bi_retrieved, eval_set))

    cross = CrossEncoderReranker()
    cross_retrieved: list[list[str]] = []
    for cid, top in zip(eval_set.keys(), bi_retrieved, strict=False):
        top_n = top[:20]
        candidates = [(jid, job_docs[jid]) for jid in top_n]
        reranked = cross.rerank(resume_by_id[cid].raw_text, candidates, top_k=len(top_n))
        cross_retrieved.append([jid for jid, _ in reranked])
    results.append(_score_stage("+ Cross-encoder rerank", cross_retrieved, eval_set))

    llm = LLMReranker()
    if llm.available:
        llm_retrieved: list[list[str]] = []
        for cid, top in zip(eval_set.keys(), cross_retrieved, strict=False):
            top_n = top[:10]
            scored = [
                llm.score_pair(resume_by_id[cid].raw_text, job_docs[jid], jid)
                for jid in top_n
            ]
            ranked = sorted(scored, key=lambda v: v.score, reverse=True)
            llm_retrieved.append([v.job_id for v in ranked])
        results.append(_score_stage("+ LLM rerank (Claude Haiku)", llm_retrieved, eval_set))
    else:
        results.append(
            StageResult(
                name="+ LLM rerank (Claude Haiku)",
                mrr=None,
                ndcg_at_5=None,
                p_at_5=None,
                note="pending - requires ANTHROPIC_API_KEY",
            )
        )

    return results


def print_table(results: list[StageResult]) -> str:
    lines = [
        "| System                          |  MRR  | NDCG@5 |  P@5  |",
        "|---------------------------------|:-----:|:------:|:-----:|",
    ]
    for r in results:
        if r.mrr is None:
            lines.append(f"| {r.name:<31} | {r.note} |")
        else:
            lines.append(
                f"| {r.name:<31} | {r.mrr:.3f} | {r.ndcg_at_5:.3f} |  {r.p_at_5:.3f} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    results = run()
    table = print_table(results)
    print(table)
    (ROOT / "data" / "processed" / "eval_results.md").write_text(table + "\n", encoding="utf-8")
