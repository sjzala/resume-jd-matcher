from __future__ import annotations

import pytest

from matcher.eval.metrics import mean_reciprocal_rank, ndcg_at_k, precision_at_k, reciprocal_rank


def test_precision_at_k_perfect() -> None:
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


def test_precision_at_k_none() -> None:
    assert precision_at_k(["x", "y", "z"], {"a"}, k=3) == 0.0


def test_precision_at_k_partial() -> None:
    assert precision_at_k(["a", "x", "b"], {"a", "b"}, k=3) == pytest.approx(2 / 3)


def test_reciprocal_rank_first() -> None:
    assert reciprocal_rank(["a", "b"], {"a"}) == 1.0


def test_reciprocal_rank_second() -> None:
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5


def test_reciprocal_rank_miss() -> None:
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_mean_reciprocal_rank() -> None:
    rr = mean_reciprocal_rank([["a", "b"], ["x", "a"]], [{"a"}, {"a"}])
    assert rr == pytest.approx((1.0 + 0.5) / 2)


def test_ndcg_at_k_ideal_order() -> None:
    rel = {"a": 3.0, "b": 2.0, "c": 1.0}
    assert ndcg_at_k(["a", "b", "c"], rel, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_reversed() -> None:
    rel = {"a": 3.0, "b": 2.0, "c": 1.0}
    score = ndcg_at_k(["c", "b", "a"], rel, k=3)
    assert 0 < score < 1


def test_invalid_k() -> None:
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, k=0)
