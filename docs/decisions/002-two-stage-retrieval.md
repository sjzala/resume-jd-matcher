# ADR 002 - Two-stage retrieval (bi-encoder + cross-encoder)

## Status

Accepted.

## Context

Three obvious topologies for the scoring stack: a bi-encoder alone, a cross-encoder alone, and a bi-encoder followed by a cross-encoder rerank over the top-k.

## Decision

Use a two-stage bi-encoder + cross-encoder pipeline. Bi-encoder retrieves top-20 from FAISS; cross-encoder reranks to top-5.

## Reasoning

- **Bi-encoder alone** is fast at query time (O(n) over the corpus) but the absolute ranking precision at the very top is mediocre — for resume/JD pairs that share heavy lexical overlap but disagree on seniority or specialisation, the bi-encoder's symmetric similarity does not capture asymmetric fit.
- **Cross-encoder alone** is the most accurate per pair but scales O(n) per query with no way to prune: every JD must be paired with the resume and run through the network. At ~50ms per pair on CPU, a 10-JD corpus is fine, but the design needs to generalise to the v0.2 corpus (~12k JDs) where this would cost ~10 minutes per query.
- **Bi-encoder + cross-encoder** is the standard IR pattern from the MS MARCO baselines: get cheap recall from a vector store, then spend cross-encoder cycles only on the surviving k. We pay one network pass per candidate in the top-k, not one per corpus document.

The choice of k = 20 for the rerank cut is small enough that latency stays under target and large enough to absorb a moderate amount of bi-encoder noise.

## Alternatives rejected

- **No rerank** — fails on adversarial cases in the seed corpus where the bi-encoder confuses MLOps with ML Engineer roles. Rerank fixes these in informal spot-checks.
- **Late-interaction (ColBERT-style)** — strong middle ground but adds heavy index and tooling cost; not worth it at this scale.
