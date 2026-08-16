"""SQLite-backed application database (PRD T4): users, documents, folders, chats."""
from __future__ import annotations

import time
import uuid
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from .config import DATA_DIR, settings

DATA_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.db_url,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if "sqlite" in settings.db_url else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
Base = declarative_base()


def now() -> float:
    return time.time()


def new_id() -> str:
    return uuid.uuid4().hex


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=new_id)
    provider = Column(String, default="password")  # password | google
    sub = Column(String)  # OAuth subject id or username
    email = Column(String)
    name = Column(String)
    password_hash = Column(String, nullable=True)
    created_at = Column(Float, default=now)
    documents = relationship("Document", back_populates="user")
    folder_sources = relationship("FolderSource", back_populates="user")
    conversations = relationship("Conversation", back_populates="user")


class Document(Base):
    __tablename__ = "documents"
    id = Column(String, primary_key=True, default=new_id)
    user_id = Column(String, ForeignKey("users.id"), index=True)
    source_type = Column(String)  # upload | url | folder
    title = Column(String)
    path_or_url = Column(String)  # server path for folder docs, URL for url docs
    content_hash = Column(String, index=True)  # change detection for folder sync
    config_fingerprint = Column(String)  # chunking config it was indexed under (F18)
    status = Column(String, default="pending")  # pending | indexing | ready | failed
    error = Column(Text, nullable=True)
    n_chunks = Column(Integer, default=0)
    created_at = Column(Float, default=now)
    user = relationship("User", back_populates="documents")


class FolderSource(Base):
    __tablename__ = "folder_sources"
    id = Column(String, primary_key=True, default=new_id)
    user_id = Column(String, ForeignKey("users.id"), index=True)
    path = Column(String)
    created_at = Column(Float, default=now)
    last_scan_at = Column(Float, nullable=True)
    user = relationship("User", back_populates="folder_sources")


class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True, default=new_id)
    user_id = Column(String, ForeignKey("users.id"), index=True)
    title = Column(String, default="New chat")
    created_at = Column(Float, default=now)
    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message", back_populates="conversation", order_by="Message.created_at"
    )


class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, default=new_id)
    conversation_id = Column(String, ForeignKey("conversations.id"), index=True)
    role = Column(String)  # user | assistant
    content = Column(Text)
    citations = Column(Text, nullable=True)  # JSON list of {number, doc_id, title, ref, excerpt}
    created_at = Column(Float, default=now)
    conversation = relationship("Conversation", back_populates="messages")


class ConfigOverride(Base):
    """Single-row store for the live pipeline config (PRD F16).

    On Vercel the repo (and thus config.yaml) is read-only, so tuning knobs
    are persisted here (Neon/Postgres, writable) instead of the file. When a
    row exists it fully replaces config.yaml as the config source.
    """

    __tablename__ = "config_overrides"
    key = Column(String, primary_key=True)
    value = Column(Text)  # JSON-encoded full config dict
    updated_at = Column(Float, default=now)


def get_config_override(key: str):
    s = SessionLocal()
    try:
        return s.get(ConfigOverride, key)
    finally:
        s.close()


def set_config_override(key: str, value: str) -> None:
    s = SessionLocal()
    try:
        row = s.get(ConfigOverride, key)
        if row is None:
            row = ConfigOverride(key=key)
            s.add(row)
        row.value = value
        row.updated_at = now()
        s.commit()
    finally:
        s.close()


def init_db() -> None:
    Base.metadata.create_all(engine)


def get_session():
    """FastAPI dependency yielding a database session."""
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


def get_user(db, user_id: Optional[str]) -> Optional[User]:
    if not user_id:
        return None
    return db.get(User, user_id)
