# Resume to Job Description Matcher

Two-stage information-retrieval system that scores resume-to-JD fit, surfaces missing skills, and (with an API key) generates LLM rationale. Built as a portfolio project demonstrating the full ML lifecycle.

## Results

| System                          |  MRR  | NDCG@5 |  P@5  |
|---------------------------------|:-----:|:------:|:-----:|
| TF-IDF baseline                 | 1.000 | 0.918 |  0.360 |
| Bi-encoder (mpnet-base-v2)      | 1.000 | 0.931 |  0.360 |
| + Cross-encoder rerank          | 1.000 | 0.889 |  0.360 |
| + LLM rerank (Claude Haiku)     | pending - requires ANTHROPIC_API_KEY |

Evaluated on a hand-curated 50-pair set (5 resumes x 10 jobs) with graded relevance (0-3). Binary positives are `grade >= 2`. See [docs/model_card.md](docs/model_card.md) and [docs/data_card.md](docs/data_card.md).

The honest reading: at this scale every system trivially clears MRR (the top-1 positive is always findable in a 10-job corpus) and P@5 is capped by the 1-2 binary-positive docs per resume. The bi-encoder edges TF-IDF on NDCG@5 by ~1pp; cross-encoder rerank is a slight regression on this corpus, which is expected when the rerank cut is small enough that bi-encoder ordering already dominates. The interesting numbers will land at v0.2 scale (~12k JDs); see [docs/data_card.md](docs/data_card.md).

## Architecture

```
resume.txt -> chunk -> bi-encoder -> FAISS -> top-20
                                              |
                                              v
                                    cross-encoder rerank
                                              |
                                              v
                                     LLM rationale (optional)
```

See [docs/architecture.md](docs/architecture.md).

## Quickstart

```bash
git clone https://github.com/sjzala/resume-jd-matcher
cd resume-jd-matcher
make install
make seed
make eval
make ui     # http://localhost:8501
make api    # http://localhost:8000/docs
```

If `make install` fails to build a Python 3.11 venv on macOS Sequoia/Tahoe (a known Homebrew Python + system libexpat incompatibility), use `uv venv --seed --python 3.11 .venv` instead and rerun `make install` from inside the venv.

## Project layout

- `src/matcher/` -- library code
- `data/processed/` -- small curated dataset (JDs, resumes, eval labels)
- `tests/` -- pytest suite
- `docs/` -- architecture, model card, data card, ADRs
- `notebooks/` -- EDA, experiments, error analysis

## What's next (v0.2 roadmap)

- Migrate to Kaggle LinkedIn Job Postings dataset (~12k JDs)
- Active learning loop to grow the eval set to 500+ pairs
- Fine-tune bi-encoder with contrastive triplets once labels are sufficient
- Calibration layer on cross-encoder logits
- Drift monitoring with Evidently
- Deploy API to Fly.io with autoscaling

## License

MIT
