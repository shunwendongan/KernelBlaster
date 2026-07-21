from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RunStatus(str, Enum):
    """Terminal status for one KernelBlaster workflow invocation."""

    IMPROVED = "improved"
    NO_IMPROVEMENT = "no_improvement"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class RunOutcome:
    """Structured result that keeps diagnostic artifacts separate from success."""

    status: RunStatus
    artifact_path: Path | None = None
    reason: str | None = None
    profiling_mode: str = "ncu"
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return bool(
            self.status is RunStatus.IMPROVED
            and self.artifact_path is not None
            and self.artifact_path.is_file()
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["artifact_path"] = (
            str(self.artifact_path) if self.artifact_path is not None else None
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunOutcome":
        artifact = payload.get("artifact_path")
        return cls(
            status=RunStatus(payload["status"]),
            artifact_path=Path(artifact) if artifact else None,
            reason=payload.get("reason"),
            profiling_mode=payload.get("profiling_mode", "ncu"),
            metrics=dict(payload.get("metrics") or {}),
        )


__all__ = ["RunOutcome", "RunStatus"]
