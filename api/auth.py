"""Authentication helpers for the local KANCHA API."""

from __future__ import annotations

import secrets
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import Header, HTTPException, status

_TOKEN_PATH = Path.home() / ".kancha" / "auth_token"


def get_auth_token() -> str:
    """Return the installation token, creating it with private permissions."""
    _TOKEN_PATH.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        _TOKEN_PATH.chmod(0o600)
    except FileNotFoundError:
        pass

    if _TOKEN_PATH.exists():
        token = _TOKEN_PATH.read_text(encoding="utf-8").strip()
        if token:
            return token

    token = secrets.token_urlsafe(32)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=_TOKEN_PATH.parent,
        prefix="auth_token.",
        delete=False,
    ) as temporary:
        temporary.write(token + "\n")
        temporary_path = Path(temporary.name)
    temporary_path.chmod(0o600)
    temporary_path.replace(_TOKEN_PATH)
    return token


async def verify_token(authorization: str | None = Header(default=None)) -> None:
    """Validate the bearer token required by every REST endpoint."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")

    token = authorization.removeprefix("Bearer ").strip()
    if not token or not secrets.compare_digest(get_auth_token(), token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
