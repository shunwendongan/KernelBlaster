from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from ..config import config


def worker_authorization_header() -> dict[str, str]:
    return {"Authorization": f"Bearer {config.WORKER_TOKEN}"}


async def require_worker_token(authorization: str | None = Header(default=None)) -> None:
    expected = f"Bearer {config.WORKER_TOKEN}"
    if authorization is None or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Worker authentication required")


__all__ = ["require_worker_token", "worker_authorization_header"]
