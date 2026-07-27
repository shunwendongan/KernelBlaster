
"""集中定义 Worker 子进程允许继承的环境变量和源码路径边界。"""

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
    """
    复制进程设置而不转发控制平面凭据。

    参数:
        source: 待分析或转换的源码文本。
    """

    source = source or os.environ
    return {
        str(key): str(value)
        for key, value in source.items()
        if not any(marker in str(key).upper() for marker in SECRET_ENVIRONMENT_MARKERS)
    }


def validate_worker_environment(source: dict[str, str] | None = None) -> None:
    """Fail closed if a worker process receives a control-plane credential.

    The trusted Supervisor may own its inbound submit token and its outbound
    worker-callback token. The check intentionally matches credential classes
    rather than values, so it also catches newly introduced provider secrets.
    """

    source = source if source is not None else os.environ
    violations = sorted(
        str(key)
        for key in source
        if str(key).upper()
        not in {"KERNELBLASTER_WORKER_TOKEN", "KERNELBLASTER_SUPERVISOR_TOKEN"}
        and any(marker in str(key).upper() for marker in SECRET_ENVIRONMENT_MARKERS)
    )
    if violations:
        names = ", ".join(violations)
        raise RuntimeError(f"Worker environment contains control-plane credentials: {names}")


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


__all__ = [
    "allowed_source_path",
    "sanitized_worker_environment",
    "validate_worker_environment",
]
