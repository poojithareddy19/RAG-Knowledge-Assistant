"""Chunking.

Splits loaded document segments into overlapping chunks while preserving page
attribution. We chunk *per page* so a chunk never straddles a page boundary —
that keeps citations honest (a chunk is always on exactly one page).

We use LangChain's ``RecursiveCharacterTextSplitter`` because it splits on a
hierarchy of separators (paragraph -> line -> sentence -> word) and is a well
tested, stable primitive. The rest of the system is framework-agnostic; only
this file depends on it, so swapping in another splitter is a one-file change.
"""
from __future__ import annotations

import re
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.loader import Segment
from src.utils.config import get_config
from src.utils.schemas import Chunk


def _slug(doc_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", doc_name).strip("_")


def chunk_segments(doc_name: str, segments: List[Segment]) -> List[Chunk]:
    """Convert (page, text) segments into a flat list of Chunk objects."""
    cfg = get_config().ingestion
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    slug = _slug(doc_name)
    chunks: List[Chunk] = []
    running_index = 0

    for page, text in segments:
        text = text.strip()
        if not text:
            continue
        if len(text) <= cfg.min_chunk_chars:
            pieces = [text]
        else:
            pieces = splitter.split_text(text)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            chunk_id = f"{slug}::p{page}::c{running_index}"
            chunks.append(
                Chunk(chunk_id=chunk_id, doc_name=doc_name, page=page, text=piece)
            )
            running_index += 1
    return chunks
