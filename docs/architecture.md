# Architecture

The matcher is a two-stage retrieval system that scores a single resume against a catalog of job descriptions. Stages are composed deliberately so each one operates over the candidate set its predecessor produced, trading off coverage for accuracy at every step.

```
              resume.txt
                  |
                  v
            +-----------+
            |  parse +  |    pdf / docx / txt -> raw_text
            |  chunk    |    optional fixed-width chunking
            +-----------+
                  |
                  v
        +-------------------+
        | bi-encoder embed  |    SentenceTransformer (mpnet-base-v2)
        +-------------------+    L2-normalised, batch=32
                  |
                  v
            +-----------+
            |  FAISS    |    IndexFlatIP, cosine via inner product
            +-----------+    return top-20 job ids
                  |
                  v
       +----------------------+
       | cross-encoder rerank |   ms-marco-MiniLM-L-6-v2
       +----------------------+   scores each (resume, jd) pair
                  |
                  v
              top-5 hits
                  |
                  v
       +----------------------+   optional, requires
       |  LLM rationale       |   ANTHROPIC_API_KEY
       +----------------------+   Claude Haiku 4.5
```

The bi-encoder gives O(n) retrieval at query time; the cross-encoder pays O(k) per query but rescues the top-k against subtle skill aliasing the bi-encoder misses (the classic MS MARCO pattern). The optional LLM stage exists for explainability, not ranking lift — it adds rationale and a calibrated 0-1 fit score that the UI surfaces to the user.

The target on a single resume against a 10-JD catalog on a 2024 M-class laptop is p50 <200ms without the LLM stage. The FAISS index is rebuilt from `data/processed/jobs.json` on demand; for production this becomes a persistent on-disk index loaded once at process start (see `pipeline/index.py`).

Skill extraction is a parallel side-channel — a rule-based pass over a curated seed list — that the API returns as `matched_skills` / `missing_skills` to give the user actionable gap feedback independent of the ranking score.
