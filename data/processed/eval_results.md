| System                          |  MRR  | NDCG@5 |  P@5  |
|---------------------------------|:-----:|:------:|:-----:|
| TF-IDF baseline                 | 1.000 | 0.918 |  0.360 |
| Bi-encoder (mpnet-base-v2)      | 1.000 | 0.931 |  0.360 |
| + Cross-encoder rerank          | 1.000 | 0.889 |  0.360 |
| + LLM rerank (Claude Haiku)     | pending - requires ANTHROPIC_API_KEY |
