from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException


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


__all__ = ["allowed_source_path"]
