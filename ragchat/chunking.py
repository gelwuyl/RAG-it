"""Chunking per config.yaml (PRD §5, F16).

Splitter choices:
- recursive: structure-aware recursive splitter (paragraph -> sentence -> word)
- markdown_header: split on Markdown headers first, then recursive within
- semantic: same splitter family in v1; kept as a distinct enum value so the
  fingerprint changes if a true embedding-based semantic splitter is added
"""
from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .config import PipelineConfig


@dataclass
class Chunk:
    text: str
    ref: str  # human-readable location reference for citations


def _recursive_splitter(cfg: PipelineConfig) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=max(cfg.chunk_size, 1),
        chunk_overlap=max(min(cfg.chunk_overlap, cfg.chunk_size - 1), 0),
    )


def split_document(text: str, title: str, cfg: PipelineConfig) -> list[Chunk]:
    splitter = _recursive_splitter(cfg)

    if cfg.splitter == "markdown_header":
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "H1"), ("##", "H2"), ("###", "H3"), ("####", "H4"),
            ]
        )
        chunks: list[Chunk] = []
        for section in header_splitter.split_text(text):
            header_path = " > ".join(section.metadata.values())
            ref = f"{header_path}" if header_path else None
            for piece in splitter.split_text(section.page_content):
                chunks.append(Chunk(text=piece, ref=ref or ""))
        return chunks

    # recursive and semantic (v1)
    pieces = splitter.split_text(text)
    return [Chunk(text=p, ref="") for p in pieces]


def refine_refs(chunks: list[Chunk], text: str) -> list[Chunk]:
    """Attach approximate position references ('~N% through document') to
    chunks that have no structural reference, for citation display."""
    if not chunks:
        return chunks
    total = len(text)
    for ch in chunks:
        if not ch.ref and total > 0:
            pos = text.find(ch.text[:80])
            if pos >= 0:
                pct = round(pos * 100 / total)
                ch.ref = f"~{pct}% of document"
            else:
                ch.ref = ""
    return chunks
