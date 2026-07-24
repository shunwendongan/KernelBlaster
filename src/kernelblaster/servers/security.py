from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException


SECRET_ENVIRONMENT_MARKERS = (
    "API_KEY",
    "AUTHORIZATION",
    "CREDENTIAL",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)


def sanitized_worker_environment(
    source: dict[str, str] | None = None,
) -> dict[str, str]:
    """Copy process settings without forwarding control-plane credentials."""

    source = source or os.environ
    return {
        str(key): str(value)
        for key, value in source.items()
        if not any(marker in str(key).upper() for marker in SECRET_ENVIRONMENT_MARKERS)
    }


def allowed_source_path(path: str, *, cwd: Path | None = None) -> Path:
    resolved = Path(path).resolve(strict=False)
    configured = os.getenv("KERNELBLASTER_ALLOWED_SOURCE_ROOTS")
    roots = (
        [Path(item).resolve() for item in configured.split(os.pathsep) if item]
        if configured
        else [(cwd or Path.cwd()).resolve()]
    )
    if not any(resolved.is_relative_to(root) for root in roots):
        raise HTTPException(status_code=400, detail="Source path escapes allowed roots")
    return resolved


__all__ = ["allowed_source_path", "sanitized_worker_environment"]
