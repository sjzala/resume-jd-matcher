from __future__ import annotations

import json
import os
from dataclasses import dataclass

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


@dataclass
class LLMVerdict:
    job_id: str
    score: float
    rationale: str


class LLMReranker:
    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        if Anthropic is None or not os.environ.get("ANTHROPIC_API_KEY"):
            self._client = None
        else:
            self._client = Anthropic()
        self.model = model

    @property
    def available(self) -> bool:
        return self._client is not None

    def score_pair(self, resume_text: str, jd_text: str, job_id: str) -> LLMVerdict:
        if self._client is None:
            raise RuntimeError("LLMReranker unavailable: install anthropic and set ANTHROPIC_API_KEY")
        prompt = (
            "Rate the fit between this candidate's resume and the job description on a 0.0-1.0 scale. "
            "Return strict JSON: {\"score\": <float>, \"rationale\": <one-sentence string>}.\n\n"
            f"RESUME:\n{resume_text}\n\nJOB DESCRIPTION:\n{jd_text}"
        )
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        start = text.find("{")
        end = text.rfind("}")
        payload = json.loads(text[start : end + 1])
        return LLMVerdict(
            job_id=job_id,
            score=float(payload["score"]),
            rationale=str(payload["rationale"]),
        )
