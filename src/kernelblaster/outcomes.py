
"""定义工作流终态及跨模块传递的标准运行结果。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RunStatus(str, Enum):
    """一次 KernelBlaster 工作流调用的标准终态。"""

    IMPROVED = "improved"
    NO_IMPROVEMENT = "no_improvement"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class RunOutcome:
    """保存工作流终态、诊断原因、性能指标和可选成功产物。"""

    status: RunStatus
    artifact_path: Path | None = None
    reason: str | None = None
    profiling_mode: str = "ncu"
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """
        处理 `success` 对应的领域操作，并返回调用方所需的标准化结果。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        return bool(
            self.status is RunStatus.IMPROVED
            and self.artifact_path is not None
            and self.artifact_path.is_file()
        )

    def to_dict(self) -> dict[str, Any]:
        """
        处理 `to_dict` 对应的领域操作，并返回调用方所需的标准化结果。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["artifact_path"] = (
            str(self.artifact_path) if self.artifact_path is not None else None
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunOutcome":
        """
        处理 `from_dict` 对应的领域操作，并返回调用方所需的标准化结果。

        参数:
            payload: 跨接口传递的序列化载荷。

        返回:
            当前操作产生的结果；具体类型由返回注解和调用约定确定。
        """
        artifact = payload.get("artifact_path")
        return cls(
            status=RunStatus(payload["status"]),
            artifact_path=Path(artifact) if artifact else None,
            reason=payload.get("reason"),
            profiling_mode=payload.get("profiling_mode", "ncu"),
            metrics=dict(payload.get("metrics") or {}),
        )


__all__ = ["RunOutcome", "RunStatus"]
