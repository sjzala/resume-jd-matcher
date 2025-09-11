from __future__ import annotations

import re
from pathlib import Path

import pdfplumber
from docx import Document

_SECTION_HEADERS = re.compile(
    r"^(experience|work experience|education|skills|projects|summary|objective)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")
    raise ValueError(f"unsupported file type: {suffix}")


def _extract_pdf(path: Path) -> str:
    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def _extract_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs).strip()


def split_sections(text: str) -> dict[str, str]:
    matches = list(_SECTION_HEADERS.finditer(text))
    if not matches:
        return {"body": text}
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        key = m.group(1).lower().strip()
        sections[key] = text[start:end].strip()
    return sections
