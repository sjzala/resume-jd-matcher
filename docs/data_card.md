# Data Card

## Source

The bundled dataset is hand-authored synthetic data, written directly into `scripts/seed_data.py` (Appendix A of the project spec). It contains 10 job descriptions and 5 resumes, plus a graded 50-pair relevance set in `data/processed/eval_set.json` (5 resumes x 10 jobs, each cell scored 0-3).

No real candidate or employer text was used. The corpus was designed to span common ML/IR archetypes (search relevance, recommendations, NLP/RAG, MLOps, CV, growth analytics, backend, junior analyst, bioinformatics) so the retrieval stack has both clear positives and adversarial near-misses.

## Schema

- `jobs.json`: list of `{job_id, title, seniority, text}`.
- `resumes.json`: list of `{candidate_id, summary, text}`.
- `eval_set.json`: dict of `candidate_id -> {job_id -> grade}` where grade in `{0, 1, 2, 3}`.

Grade semantics: 0 = irrelevant, 1 = weak fit, 2 = good fit, 3 = ideal fit. Binary positives in the eval runner are `grade >= 2`.

## Known limitations

- Very small (10 jobs / 5 resumes / 50 graded pairs). Eval noise is high; differences below ~3pp on NDCG@5 should not be treated as significant.
- English only.
- Tech-industry biased; vocabulary skews to Python/ML and will not generalise to non-tech roles.
- Labels are authored by the project author; no inter-annotator agreement.

## Licence

MIT-compatible; all text is original to this project.

## Roadmap (v0.2)

Migrate to the public Kaggle "LinkedIn Job Postings" dataset (~12k JDs) plus a derived resume corpus seeded from public profiles with the author's own permission, with relevance labels grown by active learning rather than a single pass. Until then this card stays honest about being a portfolio-scale demo, not a benchmark.
