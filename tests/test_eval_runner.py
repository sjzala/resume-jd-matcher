from __future__ import annotations

from matcher.eval.eval_runner import StageResult, print_table


def test_print_table_with_pending() -> None:
    results = [
        StageResult(name="Baseline", mrr=0.4, ndcg_at_5=0.5, p_at_5=0.3),
        StageResult(name="LLM rerank", mrr=None, ndcg_at_5=None, p_at_5=None, note="pending"),
    ]
    table = print_table(results)
    assert "Baseline" in table
    assert "pending" in table
