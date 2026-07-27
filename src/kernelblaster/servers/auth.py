
"""Validate distinct bearer-token audiences across trusted services."""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from ..config import config


def _configured_token(attribute: str, audience: str) -> str:
    token = str(getattr(config, attribute, "") or "").strip()
    if not token:
        raise RuntimeError(f"KERNELBLASTER_{audience}_TOKEN must be configured")
    return token


def validate_worker_token() -> str:
    return _configured_token("WORKER_TOKEN", "WORKER")


def validate_control_token() -> str:
    return _configured_token("CONTROL_TOKEN", "CONTROL")


def validate_supervisor_token() -> str:
    return _configured_token("SUPERVISOR_TOKEN", "SUPERVISOR")


def validate_profiler_token() -> str:
    return _configured_token("PROFILER_TOKEN", "PROFILER")


def validate_token_boundaries() -> None:
    control_token = validate_control_token()
    worker_token = validate_worker_token()
    supervisor_token = validate_supervisor_token()
    profiler_token = validate_profiler_token()
    tokens = (control_token, worker_token, supervisor_token, profiler_token)
    if any(hmac.compare_digest(left, right) for index, left in enumerate(tokens) for right in tokens[index + 1 :]):
        raise RuntimeError("Control, worker, and supervisor tokens must be different values")


def worker_authorization_header() -> dict[str, str]:
    return {"Authorization": f"Bearer {validate_worker_token()}"}


def control_authorization_header() -> dict[str, str]:
    return {"Authorization": f"Bearer {validate_control_token()}"}


def supervisor_authorization_header() -> dict[str, str]:
    return {"Authorization": f"Bearer {validate_supervisor_token()}"}


def profiler_authorization_header() -> dict[str, str]:
    return {"Authorization": f"Bearer {validate_profiler_token()}"}


def _require_token(
    authorization: str | None,
    *,
    token: str,
    audience: str,
) -> None:
    expected = f"Bearer {token}"
    if authorization is None or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail=f"{audience} authentication required")


async def require_worker_token(authorization: str | None = Header(default=None)) -> None:
    _require_token(
        authorization,
        token=validate_worker_token(),
        audience="Worker",
    )


async def require_control_token(authorization: str | None = Header(default=None)) -> None:
    _require_token(
        authorization,
        token=validate_control_token(),
        audience="Control",
    )


async def require_supervisor_token(authorization: str | None = Header(default=None)) -> None:
    _require_token(
        authorization,
        token=validate_supervisor_token(),
        audience="Supervisor",
    )


async def require_profiler_token(authorization: str | None = Header(default=None)) -> None:
    _require_token(
        authorization,
        token=validate_profiler_token(),
        audience="Profiler",
    )


__all__ = [
    "control_authorization_header",
    "profiler_authorization_header",
    "require_control_token",
    "require_profiler_token",
    "require_supervisor_token",
    "require_worker_token",
    "validate_control_token",
    "validate_profiler_token",
    "validate_token_boundaries",
    "validate_supervisor_token",
    "validate_worker_token",
    "supervisor_authorization_header",
    "worker_authorization_header",
]
