from __future__ import annotations

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        pairs = [(query, doc) for _, doc in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(
            zip([cid for cid, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(cid, float(score)) for cid, score in ranked[:top_k]]
