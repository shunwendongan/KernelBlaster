
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

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """

    source = source or os.environ
    return {
        str(key): str(value)
        for key, value in source.items()
        if not any(marker in str(key).upper() for marker in SECRET_ENVIRONMENT_MARKERS)
    }


def allowed_source_path(path: str, *, cwd: Path | None = None) -> Path:
    """
    处理 `allowed_source_path` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        path: 待读取、写入或校验的文件系统路径。
        cwd: 调用方提供的 `cwd` 参数。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。

    异常:
        HTTPException: 输入、外部调用或状态不满足执行要求时抛出。
    """
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
