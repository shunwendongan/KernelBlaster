
"""生成并校验 Worker 服务使用的 Bearer Token。"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from ..config import config


def worker_authorization_header() -> dict[str, str]:
    """
    处理 `worker_authorization_header` 对应的领域操作，并返回调用方所需的标准化结果。

    返回:
        当前操作产生的结果；具体类型由返回注解和调用约定确定。
    """
    return {"Authorization": f"Bearer {config.WORKER_TOKEN}"}


async def require_worker_token(authorization: str | None = Header(default=None)) -> None:
    """
    处理 `require_worker_token` 对应的领域操作，并返回调用方所需的标准化结果。

    参数:
        authorization: 调用方提供的 `authorization` 参数。

    异常:
        HTTPException: 输入、外部调用或状态不满足执行要求时抛出。
    """
    expected = f"Bearer {config.WORKER_TOKEN}"
    if authorization is None or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Worker authentication required")


__all__ = ["require_worker_token", "worker_authorization_header"]
