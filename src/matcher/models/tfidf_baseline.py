from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfMatcher:
    def __init__(self, ngram_range: tuple[int, int] = (1, 2), min_df: int = 1) -> None:
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words="english",
            lowercase=True,
        )
        self._doc_ids: list[str] = []
        self._matrix = None

    def fit(self, docs: dict[str, str]) -> None:
        self._doc_ids = list(docs.keys())
        texts = [docs[i] for i in self._doc_ids]
        self._matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if self._matrix is None:
            raise RuntimeError("call fit before search")
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        ranked = sorted(zip(self._doc_ids, sims), key=lambda x: x[1], reverse=True)
        return [(doc_id, float(score)) for doc_id, score in ranked[:k]]
