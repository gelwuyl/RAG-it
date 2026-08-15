"""Authentication: Google OAuth (PRD F1) with a username/password fallback.

Sessions are signed cookies (itsdangerous). The password fallback exists so
the app is usable before Google OAuth credentials are configured (PRD §9).
"""
from __future__ import annotations

import secrets
from typing import Optional

import bcrypt
import httpx
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from itsdangerous import BadSignature, URLSafeSerializer
from sqlalchemy.orm import Session

from .config import settings
from .db import User, get_session

_serializer = URLSafeSerializer(settings.session_secret, salt="session")
SESSION_COOKIE = "ragchat_session"
_basic = HTTPBasic(auto_error=False)


def hash_password(pw: str) -> str:
    # bcrypt accepts at most 72 bytes; truncate longer inputs
    return bcrypt.hashpw(pw.encode("utf-8")[:72], bcrypt.gensalt()).decode()


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode("utf-8")[:72], hashed.encode())
    except ValueError:
        return False


def encode_session(user_id: str) -> str:
    return _serializer.dumps({"uid": user_id})


def decode_session(token: str) -> Optional[str]:
    try:
        data = _serializer.loads(token)
        return data.get("uid")
    except BadSignature:
        return None


def get_current_user(request: Request, db: Session = Depends(get_session)) -> User:
    token = request.cookies.get(SESSION_COOKIE)
    uid = decode_session(token) if token else None
    if not uid:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = db.get(User, uid)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def oauth_configured() -> bool:
    return bool(settings.google_client_id and settings.google_client_secret)


def google_auth_url(state: str) -> str:
    params = {
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    qs = httpx.QueryParams(params)
    return f"https://accounts.google.com/o/oauth2/v2/auth?{qs}"


async def google_exchange_code(code: str) -> dict:
    """Exchange an authorization code for an ID token, returning user info."""
    async with httpx.AsyncClient(timeout=30) as client:
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        token_resp.raise_for_status()
        tokens = token_resp.json()
        info_resp = await client.get(
            "https://openidconnect.googleapis.com/v1/userinfo",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        info_resp.raise_for_status()
        return info_resp.json()


def find_or_create_google_user(db: Session, info: dict) -> User:
    sub = info.get("sub")
    user = (
        db.query(User).filter(User.provider == "google", User.sub == sub).first()
    )
    if not user:
        user = User(
            provider="google",
            sub=sub,
            email=info.get("email"),
            name=info.get("name") or info.get("email"),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def find_or_create_password_user(db: Session, username: str, password: str) -> tuple[User, Optional[str]]:
    """Create a password user; if it exists, verify. Returns (user, error)."""
    user = (
        db.query(User)
        .filter(User.provider == "password", User.sub == username)
        .first()
    )
    if user:
        if not user.password_hash or not verify_password(password, user.password_hash):
            raise ValueError("Invalid username or password")
        return user, None
    user = User(
        provider="password",
        sub=username,
        name=username,
        password_hash=hash_password(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user, None


def issue_state() -> str:
    return secrets.token_urlsafe(24)
