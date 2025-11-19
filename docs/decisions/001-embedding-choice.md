# ADR 001 - Bi-encoder choice: all-mpnet-base-v2 over all-MiniLM-L6-v2

## Status

Accepted.

## Context

The bi-encoder is the recall stage; its job is to make sure every plausible JD survives to the cross-encoder. Two off-the-shelf sentence-transformer checkpoints were the obvious starting points: `all-MiniLM-L6-v2` (384-dim, 22M params) and `all-mpnet-base-v2` (768-dim, 110M params).

## Decision

Use `all-mpnet-base-v2` as the default bi-encoder.

## Reasoning

On the published STS benchmarks, mpnet-base scores roughly 3 points higher than MiniLM-L6 (88.7 vs 84.2 STS-B Spearman in the sentence-transformers leaderboard at time of writing). For resume-length texts, where small differences in semantic similarity decide which JDs survive to rerank, that gap meaningfully reduces the rate at which good matches drop out of the top-20.

The trade-off is encode latency: mpnet runs roughly 2x slower than MiniLM on CPU (~25ms vs ~12ms per resume on a 2024 M-class laptop). At this scale (one query at a time, a single corpus rebuild on startup) this is irrelevant. At higher scale, MiniLM becomes the right choice for the recall stage, with rerank doing more of the heavy lifting.

Memory cost is also acceptable: mpnet weights fit comfortably in <1GB RAM, which keeps the FastAPI process within the free-tier limits of common PaaS targets (HF Spaces, Fly.io, Render).

## Alternatives rejected

- **MiniLM-L6-v2** — faster but materially worse recall on this corpus in informal testing.
- **bge-small-en-v1.5** — competitive, but adds a non-trivial dependency on the BAAI ecosystem with no clear win on this domain.
- **Custom fine-tuned bi-encoder** — see ADR 003; ruled out until the eval set is large enough to support contrastive training.
