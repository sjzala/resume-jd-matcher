from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(
    retrieved_per_query: Sequence[Sequence[str]],
    relevant_per_query: Sequence[set[str]],
) -> float:
    if len(retrieved_per_query) != len(relevant_per_query):
        raise ValueError("mismatched query counts")
    if not retrieved_per_query:
        return 0.0
    rrs = [
        reciprocal_rank(r, g)
        for r, g in zip(retrieved_per_query, relevant_per_query, strict=False)
    ]
    return float(np.mean(rrs))


def ndcg_at_k(
    retrieved: Sequence[str],
    relevance: dict[str, float],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    gains = [relevance.get(item, 0.0) for item in retrieved[:k]]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(np.asarray(gains) * discounts))

    ideal = sorted(relevance.values(), reverse=True)[:k]
    idiscounts = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
    idcg = float(np.sum(np.asarray(ideal) * idiscounts))

    return dcg / idcg if idcg > 0 else 0.0
