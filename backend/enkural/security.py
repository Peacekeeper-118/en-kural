"""Local-app security: per-installation access token + origin check.

The server binds to 127.0.0.1 by default.  Every ``/api`` request must carry
the token (``Authorization: Bearer …``), which the UI receives once via the
start-up URL and keeps in sessionStorage.  This stops other local users,
processes and *web pages open in the same browser* from calling the API.
"""
from __future__ import annotations

import os
import secrets
from typing import Optional

from fastapi import Header, HTTPException, Request

from . import config

_token: Optional[str] = None


def token_file():
    return config.DATA_DIR / ".token"


def get_token() -> Optional[str]:
    """Return the access token (None when auth is explicitly disabled)."""
    global _token
    if _token is not None:
        return _token or None
    if os.environ.get("ENKURAL_NO_AUTH") == "1":
        _token = ""
        return None
    env = os.environ.get("ENKURAL_TOKEN")
    if env:
        _token = env
        return _token
    tf = token_file()
    if tf.exists():
        _token = tf.read_text().strip()
    if not _token:
        _token = secrets.token_urlsafe(24)
        tf.parent.mkdir(parents=True, exist_ok=True)
        tf.write_text(_token)
        try:
            os.chmod(tf, 0o600)
        except OSError:
            pass
    return _token


def _own_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    if not origin:
        return True  # same-origin GETs and non-browser clients carry no Origin
    host = request.headers.get("host", "")
    return origin.split("://", 1)[-1].rstrip("/") == host


async def require_token(request: Request, authorization: Optional[str] = Header(default=None)) -> None:
    token = get_token()
    if not _own_origin(request):
        raise HTTPException(status_code=403, detail="Cross-origin requests are not allowed.")
    if token is None:
        return
    supplied = None
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if supplied is None:
        supplied = request.query_params.get("token")
    if not supplied or not secrets.compare_digest(supplied, token):
        raise HTTPException(status_code=401, detail="Missing or invalid access token.")
