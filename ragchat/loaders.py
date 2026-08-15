"""Raw text extraction for the supported formats (PRD F4, F5).

PDF (text layer), Markdown/text variants, and HTML (fetched pages or .html
files). OCR for scanned PDFs is out of scope (PRD §2).
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

TEXT_EXTENSIONS = {
    ".md", ".txt", ".markdown", ".rst", ".csv", ".json", ".yaml", ".yml", ".log",
}
HTML_EXTENSIONS = {".html", ".htm"}
PDF_EXTENSIONS = {".pdf"}


def load_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def load_html(data: bytes, url: str | None = None) -> str:
    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else ""
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if title:
        text = f"{title}\n\n{text}"
    return text


def load_bytes(filename: str, data: bytes, url: str | None = None) -> str:
    ext = Path(filename).suffix.lower()
    if ext in PDF_EXTENSIONS:
        return load_pdf(data)
    if ext in HTML_EXTENSIONS or url is not None:
        return load_html(data, url)
    if ext in TEXT_EXTENSIONS or ext == "":
        return data.decode("utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {ext or filename}")


def fetch_url(url: str) -> tuple[str, bytes]:
    """Fetch a web page, following redirects. Returns (final_url, body)."""
    resp = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "ragchat/1.0 (+class project)"},
        allow_redirects=True,
    )
    resp.raise_for_status()
    return resp.url, resp.content


def page_title(url: str, body: bytes) -> str:
    soup = BeautifulSoup(body, "html.parser")
    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)[:200]
    return url[:200]
