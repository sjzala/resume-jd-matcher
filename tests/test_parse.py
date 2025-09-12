from __future__ import annotations

from pathlib import Path

from matcher.data.parse import extract_text, split_sections


def test_extract_text_txt(tmp_path: Path) -> None:
    p = tmp_path / "r.txt"
    p.write_text("hello world", encoding="utf-8")
    assert extract_text(p) == "hello world"


def test_split_sections_no_headers() -> None:
    sections = split_sections("just some prose")
    assert sections == {"body": "just some prose"}


def test_split_sections_with_headers() -> None:
    text = "Summary\nfoo\n\nSkills\nbar\nbaz\n"
    sections = split_sections(text)
    assert "summary" in sections
    assert "skills" in sections
    assert "bar" in sections["skills"]
