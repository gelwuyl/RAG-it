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


# A chunk covering this much of its document is, for a reader's purposes, the
# document. Below it, where the passage sits is worth saying; at or above it,
# saying "~0% through" about the whole file is just wrong.
WHOLE_DOCUMENT_COVERAGE = 0.9


def refine_refs(chunks: list[Chunk], text: str) -> list[Chunk]:
    """Attach an approximate position ('~N% through') to chunks with no
    structural reference, for citation display.

    The number is where the passage STARTS, and the wording has to say so. It
    used to read "~N% of document", which is a quantity rather than a position —
    so a short file that produces a single chunk showed "~0% of document" beside
    an excerpt containing all of it. Nought percent of the document, and it was
    the whole thing.
    """
    if not chunks:
        return chunks
    total = len(text)
    for ch in chunks:
        if not ch.ref and total > 0:
            # Locate it FIRST. Checking coverage before confirming the passage
            # is even in this document labels an unrelated chunk "whole
            # document" whenever it happens to be longer than the file — a
            # confident claim about a passage that is not there at all.
            pos = text.find(ch.text[:80])
            if pos < 0:
                ch.ref = ""
            elif len(ch.text) / total >= WHOLE_DOCUMENT_COVERAGE:
                # Nothing to locate: there is only one place it can be.
                ch.ref = "whole document"
            else:
                ch.ref = f"~{round(pos * 100 / total)}% through"
    return chunks
