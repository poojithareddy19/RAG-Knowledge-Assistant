"""Document loading.

Turns a raw uploaded file into a list of ``(page_number, text)`` tuples. Page
numbers are preserved so that citations can point at an exact page. Non-paginated
formats (txt, md, docx) report page 0 for the whole body — the chunker still
assigns them stable ids.

Supported: .pdf .docx .txt .md .markdown
"""
from __future__ import annotations

from pathlib import Path

Segment = tuple[int, str]  # (page_number, text)


def load_document(path: str | Path) -> list[Segment]:
    """Dispatch on file extension. Raises ValueError on unsupported types."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)
    if suffix in {".txt", ".md", ".markdown"}:
        return _load_text(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _load_pdf(path: Path) -> list[Segment]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    segments: list[Segment] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            segments.append((i, text))
    return segments


def _load_docx(path: Path) -> list[Segment]:
    import docx  # python-docx

    document = docx.Document(str(path))
    body = "\n".join(p.text for p in document.paragraphs if p.text.strip())
    return [(0, body)] if body.strip() else []


def _load_text(path: Path) -> list[Segment]:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return [(0, text)] if text else []


SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md", ".markdown")
