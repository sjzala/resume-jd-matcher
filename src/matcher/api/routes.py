from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from matcher.pipeline.score import score_resume

router = APIRouter()


class MatchRequest(BaseModel):
    resume_text: str
    top_k: int = 5
    use_rerank: bool = True


class MatchHit(BaseModel):
    job_id: str
    title: str
    score: float
    matched_skills: list[str]
    missing_skills: list[str]


class MatchResponse(BaseModel):
    hits: list[MatchHit]


@router.post("/match", response_model=MatchResponse)
def match(req: MatchRequest) -> MatchResponse:
    if not req.resume_text.strip():
        raise HTTPException(status_code=422, detail="resume_text must not be empty")
    hits = score_resume(req.resume_text, top_k=req.top_k, use_rerank=req.use_rerank)
    return MatchResponse(hits=[MatchHit(**h) for h in hits])
