"""Trusted adapter registry and compatibility metadata for legacy Drivers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from .contracts import AdapterKind, TaskSpec


@dataclass(frozen=True)
class AdapterDescriptor:
    id: str
    version: str
    kind: AdapterKind
    task_ids: frozenset[str]
    plugin_digest: str | None = None


class AdapterRegistry:
    """An immutable allowlist. Requests select IDs, never code or paths."""

    def __init__(self, descriptors: tuple[AdapterDescriptor, ...]) -> None:
        indexed: dict[tuple[str, str], AdapterDescriptor] = {}
        for descriptor in descriptors:
            key = (descriptor.id, descriptor.version)
            if key in indexed:
                raise ValueError("duplicate adapter ID/version")
            indexed[key] = descriptor
        self._descriptors = indexed

    def resolve(self, task: TaskSpec) -> AdapterDescriptor:
        descriptor = self._descriptors.get((task.adapter_id, task.adapter_version))
        if descriptor is None or task.id not in descriptor.task_ids:
            raise KeyError("adapter_not_allowlisted")
        return descriptor


@dataclass(frozen=True)
class LegacyDriverAdapter:
    """Compatibility identity for an existing per-task driver.cpp.

    The original source remains unchanged. The trusted runner interprets the
    exact legacy stdout token and emits the versioned correctness result.
    """

    task_id: str
    path: Path

    def validate(self) -> None:
        if self.path.name != "driver.cpp" or not self.path.is_file():
            raise ValueError("legacy adapter requires an existing driver.cpp")
        source = self.path.read_text(encoding="utf-8")
        if "launch_gpu_implementation" not in source or "int main(" not in source:
            raise ValueError("legacy driver is missing its required ABI")

    @property
    def digest(self) -> str:
        self.validate()
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    @staticmethod
    def passed(returncode: int, stdout: bytes | str) -> bool:
        text = stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else stdout
        tokens = [line.strip().lower() for line in text.splitlines() if line.strip()]
        return returncode == 0 and tokens == ["passed"]
