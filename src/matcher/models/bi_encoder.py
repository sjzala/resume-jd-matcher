from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class IndexBundle:
    index: faiss.Index
    ids: list[str]


class BiEncoderMatcher:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._bundle: IndexBundle | None = None

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)

    def build_index(self, docs: dict[str, str]) -> None:
        ids = list(docs.keys())
        texts = [docs[i] for i in ids]
        vecs = self.embed(texts)
        index = faiss.IndexFlatIP(self.dim)
        index.add(vecs)
        self._bundle = IndexBundle(index=index, ids=ids)

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if self._bundle is None:
            raise RuntimeError("index not built; call build_index first")
        q = self.embed([query])
        scores, idx = self._bundle.index.search(q, k)
        return [
            (self._bundle.ids[i], float(s))
            for s, i in zip(scores[0], idx[0])
            if i != -1
        ]

    def save(self, dir_path: Path) -> None:
        if self._bundle is None:
            raise RuntimeError("nothing to save; build index first")
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._bundle.index, str(dir_path / "index.faiss"))
        (dir_path / "ids.txt").write_text("\n".join(self._bundle.ids), encoding="utf-8")

    def load(self, dir_path: Path) -> None:
        index = faiss.read_index(str(dir_path / "index.faiss"))
        ids = (dir_path / "ids.txt").read_text(encoding="utf-8").splitlines()
        self._bundle = IndexBundle(index=index, ids=ids)
