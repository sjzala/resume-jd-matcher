from __future__ import annotations

import re

_SKILL_SEED = {
    "python", "pytorch", "tensorflow", "jax", "scikit-learn", "sklearn",
    "pandas", "numpy", "spark", "sql", "fastapi", "flask", "django",
    "docker", "kubernetes", "k8s", "aws", "gcp", "azure", "terraform",
    "airflow", "kubeflow", "mlflow", "feast", "faiss", "milvus", "qdrant",
    "huggingface", "transformers", "langchain", "openai", "anthropic",
    "ranking", "retrieval", "rag", "llm", "nlp", "cv", "computer vision",
    "a/b testing", "causal inference", "tableau", "looker", "dbt",
}


def extract_skills(text: str) -> list[str]:
    lower = text.lower()
    found: set[str] = set()
    for skill in _SKILL_SEED:
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, lower):
            found.add(skill)
    return sorted(found)


def skill_overlap(resume_text: str, jd_text: str) -> tuple[list[str], list[str]]:
    r = set(extract_skills(resume_text))
    j = set(extract_skills(jd_text))
    matched = sorted(r & j)
    missing = sorted(j - r)
    return matched, missing
