"""Public, versioned contracts for portable standalone runs.

The portability layer deliberately keeps deployment facts as data.  A bundle
records what was observed on a machine; it never assumes a particular AutoDL
image, GPU product name, CUDA release, or filesystem layout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping


INSTANCE_IDENTITY_SCHEMA = "instance-identity/v1"
HARDWARE_IDENTITY_SCHEMA = "hardware-identity/v1"
RUN_BUNDLE_SCHEMA = "run-bundle/v1"
AGGREGATE_REPORT_SCHEMA = "aggregate-report/v1"


def canonical_bytes(value: Mapping[str, Any] | list[Any]) -> bytes:
    """Encode public contract data in its one canonical JSON representation."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256(value: bytes | Mapping[str, Any] | list[Any]) -> str:
    """Return a lowercase SHA-256 hex digest for bytes or canonical JSON data."""
    payload = value if isinstance(value, bytes) else canonical_bytes(value)
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class InstanceIdentity:
    """Stable logical identity for one independently managed instance."""

    instance_id: str
    created_at: str
    schema_version: str = INSTANCE_IDENTITY_SCHEMA

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class HardwareIdentity:
    """Observed device/runtime facts plus separate audit and comparison hashes."""

    logical_id: str
    name: str
    compute_capability: str
    target_arch: str
    total_memory_bytes: int | None
    driver_version: str | None
    cuda_version: str | None
    image_digest: str | None
    gpu_uuid: str | None = None
    runtime: str | None = None
    schema_version: str = HARDWARE_IDENTITY_SCHEMA

    def audit_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "logical_id": self.logical_id,
            "name": self.name,
            "compute_capability": self.compute_capability,
            "target_arch": self.target_arch,
            "total_memory_bytes": self.total_memory_bytes,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "image_digest": self.image_digest,
            "gpu_uuid": self.gpu_uuid,
            "runtime": self.runtime,
        }

    def comparison_payload(self) -> dict[str, Any]:
        """Stable comparison fields; volatile free memory and UUID are excluded."""
        driver_major = (self.driver_version or "unknown").split(".", 1)[0]
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "compute_capability": self.compute_capability,
            "target_arch": self.target_arch,
            "total_memory_bytes": self.total_memory_bytes,
            "driver_major": driver_major,
            "cuda_version": self.cuda_version,
            "image_digest": self.image_digest,
            "runtime": self.runtime,
        }

    @property
    def audit_fingerprint(self) -> str:
        return "sha256:" + sha256(self.audit_payload())

    @property
    def comparison_group(self) -> str:
        return "sha256:" + sha256(self.comparison_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.audit_payload(),
            "audit_fingerprint": self.audit_fingerprint,
            "comparison_group": self.comparison_group,
        }
