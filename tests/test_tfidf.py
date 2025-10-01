from __future__ import annotations

from matcher.models.tfidf_baseline import TfidfMatcher


def test_tfidf_returns_ranked() -> None:
    docs = {
        "a": "machine learning engineer python pytorch",
        "b": "graphic designer photoshop illustrator",
        "c": "data scientist sql pandas",
    }
    m = TfidfMatcher()
    m.fit(docs)
    hits = m.search("python ml engineer", k=2)
    assert len(hits) == 2
    assert hits[0][0] == "a"
