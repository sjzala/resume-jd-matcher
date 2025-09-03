from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class Experience(BaseModel):
    company: str
    title: str
    start: date | None = None
    end: date | None = None
    bullets: list[str] = Field(default_factory=list)


class Education(BaseModel):
    institution: str
    degree: str | None = None
    field: str | None = None
    graduation: date | None = None


class Resume(BaseModel):
    candidate_id: str
    raw_text: str
    summary: str | None = None
    skills: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)


Seniority = Literal["intern", "junior", "mid", "senior", "staff", "principal"]


class JobDescription(BaseModel):
    job_id: str
    title: str
    company: str | None = None
    location: str | None = None
    seniority: Seniority | None = None
    raw_text: str
    required_skills: list[str] = Field(default_factory=list)
    nice_to_have: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    job_id: str
    candidate_id: str
    score: float = Field(ge=0.0, le=1.0)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    rationale: str | None = None
