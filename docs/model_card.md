# Model Card

## Components

- **Bi-encoder retriever** — `sentence-transformers/all-mpnet-base-v2`. 768-dim embeddings, L2-normalised, scored with cosine similarity via a FAISS `IndexFlatIP`. Used to fetch the top-20 candidates per resume.
- **Cross-encoder reranker** — `cross-encoder/ms-marco-MiniLM-L-6-v2`. Operates on the bi-encoder's top-20 (resume, jd) pairs and returns the top-5.
- **LLM judge (optional)** — Anthropic `claude-haiku-4-5-20251001` used in `LLMReranker` to produce a 0-1 fit score and a one-sentence rationale per surviving candidate. Activated only when `ANTHROPIC_API_KEY` is set; the eval pipeline degrades gracefully when it is not.
- **Skill extractor** — rule-based; a regex-backed pass over a curated seed list in `features/skill_extract.py`. Surfaces `matched_skills` and `missing_skills` in the API response. Not part of the ranking score.

## Intended use

A portfolio demo of two-stage information retrieval applied to resume-to-JD matching. It is suitable for interactive exploration and reading code as a reference implementation.

## Out-of-scope use

It is **not** suitable for any automated hiring decision, including screening, rejection, or ranking of real candidates. The training corpora of all three models reflect tech-industry English and will discriminate against non-traditional resumes and non-English speakers. There is no demographic auditing, no fairness analysis, and no calibration of scores against real outcomes.

## Known biases and failure modes

- **Tech-vocabulary bias** — both encoders were trained on web/QA pairs (MS MARCO, etc.). They will under-rank resumes that describe technical skills in non-canonical phrasings (e.g. "neural nets" vs "PyTorch").
- **Length bias** — longer JDs accumulate more lexical overlap and tend to dominate TF-IDF. The bi-encoder is less length-sensitive but not immune.
- **Seniority leakage** — seniority is part of `JobDescription` but not used in scoring; a senior resume can still match a junior role and vice versa.
- **Skill aliasing** — the seed list in `skill_extract.py` is small (~50 entries); misses many real-world skill names.

## Calibration

The current cross-encoder logits are passed through as-is; they are not isotonic-fitted or Platt-scaled to a probability. The Streamlit UI displays the raw score and explicitly labels it as a ranking signal, not a probability. ADR 003 captures the plan to add calibration once a meaningful labelled set exists.

## Latency / cost

On a 2024 M-class laptop, indexing the 10-JD corpus takes ~1.5s on first encode (model cold start dominates), and a single query through the full bi+cross stack returns in <200ms once warm. The LLM stage adds ~600ms-1s per candidate at Haiku pricing.
