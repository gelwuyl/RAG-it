"""App settings (env vars) + hot-reloadable pipeline config (config.yaml).

Secrets live in environment variables only (PRD F15). Pipeline knobs live in
config.yaml and are re-read on every request so experiments need no restart
(PRD F16); index-affecting knobs (chunking + embedding model) are recorded
into a fingerprint stored with each chunk (PRD F18).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
DB_PATH = DATA_DIR / "app.db"
UPLOAD_DIR = DATA_DIR / "uploads"
EVAL_DIR = ROOT / "eval"


class Settings:
    def __init__(self) -> None:
        self.proxy_base_url = os.environ.get(
            "PROXY_BASE_URL", "https://llmproxy.mrchloep.com/v1"
        )
        self.api_key = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        self.db_url = os.environ.get("DATABASE_URL", f"sqlite:///{DB_PATH}")
        self.session_secret = os.environ.get("SESSION_SECRET", "dev-session-secret")
        self.allowed_root = Path(
            os.environ.get("RAG_ALLOWED_ROOT", str(Path.home()))
        ).resolve()
        # Google OAuth (PRD F1). Optional: when unset, only the password
        # fallback auth is available (documented risk fallback in PRD §9).
        self.google_client_id = os.environ.get("GOOGLE_CLIENT_ID", "")
        self.google_client_secret = os.environ.get("GOOGLE_CLIENT_SECRET", "")
        self.google_redirect_uri = os.environ.get("GOOGLE_REDIRECT_URI", "")
        # Default generation model; overridden by config.yaml
        self.default_llm_model = os.environ.get("RAG_LLM_MODEL", "qwen3.8-max")
        self.default_embedding_model = os.environ.get(
            "RAG_EMBEDDING_MODEL", "text-embedding-005"
        )


settings = Settings()


@dataclass(frozen=True)
class PipelineConfig:
    chunk_size: int
    chunk_overlap: int
    splitter: str
    top_k: int
    candidate_k: int
    similarity_threshold: float
    hybrid_search: bool
    reranker: bool
    query_rewrite: bool
    llm_model: str
    temperature: float
    embedding_model: str

    def fingerprint(self) -> str:
        """Hash of the index-affecting settings (PRD F18)."""
        raw = f"{self.chunk_size}|{self.chunk_overlap}|{self.splitter}|{self.embedding_model}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]


def load_config() -> PipelineConfig:
    """Re-read config.yaml on every call so tuning needs no restart."""
    data: dict = {}
    if CONFIG_PATH.exists():
        data = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    c = data.get("chunking", {})
    r = data.get("retrieval", {})
    q = data.get("query", {})
    g = data.get("generation", {})
    e = data.get("embedding", {})
    return PipelineConfig(
        chunk_size=int(c.get("chunk_size", 512)),
        chunk_overlap=int(c.get("chunk_overlap", 75)),
        splitter=str(c.get("splitter", "recursive")),
        top_k=int(r.get("top_k", 4)),
        candidate_k=int(r.get("candidate_k", 20)),
        similarity_threshold=float(r.get("similarity_threshold", 0.0)),
        hybrid_search=bool(r.get("hybrid_search", False)),
        reranker=bool(r.get("reranker", False)),
        query_rewrite=bool(q.get("query_rewrite", True)),
        llm_model=str(g.get("llm_model", settings.default_llm_model)),
        temperature=float(g.get("temperature", 0.0)),
        embedding_model=str(e.get("model", settings.default_embedding_model)),
    )
