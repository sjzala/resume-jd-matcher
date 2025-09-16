from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"

JOBS = [
    {"job_id": "j01", "title": "Senior Machine Learning Engineer", "seniority": "senior",
     "text": "We are hiring a Senior ML Engineer to build production recommendation systems. Strong Python, PyTorch, and experience deploying models on Kubernetes. Familiarity with feature stores and vector search."},
    {"job_id": "j02", "title": "NLP Research Engineer", "seniority": "mid",
     "text": "NLP Research Engineer to work on retrieval-augmented generation. PhD or strong publications preferred. PyTorch, HuggingFace Transformers, and experience with evaluation of LLMs."},
    {"job_id": "j03", "title": "Data Scientist - Growth", "seniority": "mid",
     "text": "Data Scientist on the Growth team. A/B testing, causal inference, and SQL-heavy analytics. Python, pandas, statsmodels. Communicate findings to product and exec stakeholders."},
    {"job_id": "j04", "title": "Computer Vision Engineer", "seniority": "mid",
     "text": "Computer Vision Engineer for medical imaging. Train and evaluate CNN and ViT models. Knowledge of DICOM, image augmentation, and model calibration. PyTorch required."},
    {"job_id": "j05", "title": "MLOps Engineer", "seniority": "senior",
     "text": "MLOps Engineer to own the ML platform. Kubeflow, MLflow, Airflow. CI/CD for ML pipelines. Terraform and AWS. Strong Python and Go."},
    {"job_id": "j06", "title": "Backend Engineer - Python", "seniority": "mid",
     "text": "Backend Engineer building APIs in FastAPI and Postgres. Docker, Kubernetes, and event-driven systems with Kafka. No ML required."},
    {"job_id": "j07", "title": "Applied Scientist - Search", "seniority": "senior",
     "text": "Applied Scientist on Search Relevance. Learning-to-rank, bi-encoder and cross-encoder retrieval, and large-scale evaluation. Python and PyTorch."},
    {"job_id": "j08", "title": "Junior Data Analyst", "seniority": "junior",
     "text": "Junior Data Analyst. SQL, Tableau, and Excel. Build dashboards for the marketing team. Bachelor's degree in a quantitative field."},
    {"job_id": "j09", "title": "ML Engineer - Recommendations", "seniority": "mid",
     "text": "ML Engineer on the Recommendations team. Build candidate generation and ranking models. PyTorch, Spark, and SQL. Productionize models with low-latency serving."},
    {"job_id": "j10", "title": "Bioinformatics Scientist", "seniority": "mid",
     "text": "Bioinformatics Scientist. Single-cell RNA-seq analysis, Python, R, and Nextflow pipelines. Collaborate with wet-lab scientists."},
]

RESUMES = [
    {"candidate_id": "r01", "summary": "ML Engineer with 4 years building search and recommendation systems.",
     "text": "Machine Learning Engineer with 4 years of experience. Built large-scale recommendation systems at a marketplace company. Strong Python and PyTorch. Deployed models on Kubernetes via TorchServe. Worked with FAISS vector indexes and a Feast feature store. Recent project: cross-encoder reranker lifted NDCG@5 by 14%."},
    {"candidate_id": "r02", "summary": "NLP researcher focused on retrieval-augmented generation.",
     "text": "NLP researcher with publications at ACL and EMNLP. Three years of experience with HuggingFace Transformers and PyTorch. Built a RAG system for legal document QA with bi-encoder retrieval and a reranker. Evaluated LLM outputs with both human and LLM-as-judge protocols."},
    {"candidate_id": "r03", "summary": "Growth data scientist, A/B testing and causal inference.",
     "text": "Data Scientist on a Growth team for 3 years. Designed and analyzed 150+ A/B tests. Strong SQL and Python. Stakeholder communication with product and exec. Built attribution models using causal inference techniques (DoWhy, EconML)."},
    {"candidate_id": "r04", "summary": "Computer vision engineer, medical imaging.",
     "text": "Computer Vision Engineer with 2 years in medical imaging. Trained ResNet and Swin Transformer models on DICOM data. Built calibration and OOD detection layers. PyTorch, MONAI. Familiar with HIPAA constraints."},
    {"candidate_id": "r05", "summary": "Backend engineer transitioning into ML.",
     "text": "Backend Engineer with 5 years of Python and FastAPI experience. Built APIs serving 10k QPS on Kubernetes. Recently completed a self-study program in machine learning. Built a sentiment classification side project. Looking to move into ML Engineering roles."},
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "jobs.json").write_text(json.dumps(JOBS, indent=2), encoding="utf-8")
    (OUT / "resumes.json").write_text(json.dumps(RESUMES, indent=2), encoding="utf-8")
    print(f"wrote {len(JOBS)} jobs and {len(RESUMES)} resumes to {OUT}")


if __name__ == "__main__":
    main()
