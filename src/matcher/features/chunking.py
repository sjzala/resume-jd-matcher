from __future__ import annotations


def chunk_text(text: str, max_chars: int = 800, overlap: int = 100) -> list[str]:
    if max_chars <= overlap:
        raise ValueError("max_chars must be greater than overlap")
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return chunks
