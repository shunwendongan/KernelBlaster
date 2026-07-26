"""Instance and GPU identity helpers with capability-first detection."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import uuid
from typing import Callable

from .contracts import HardwareIdentity, InstanceIdentity


CommandRunner = Callable[[list[str]], bytes]
_INSTANCE_FILE = "instance-identity.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run(command: list[str]) -> bytes:
    return subprocess.check_output(command, stderr=subprocess.STDOUT)


def load_or_create_instance_identity(state_dir: str | Path) -> InstanceIdentity:
    """Create an atomic local identity once; callers rotate it explicitly."""
    root = Path(state_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / _INSTANCE_FILE
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return InstanceIdentity(
            instance_id=str(payload["instance_id"]),
            created_at=str(payload["created_at"]),
            schema_version=str(payload.get("schema_version") or "instance-identity/v1"),
        )
    identity = InstanceIdentity(instance_id=uuid.uuid4().hex, created_at=_utc_now())
    temporary = root / f".{_INSTANCE_FILE}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(json.dumps(identity.to_dict(), sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return identity


def rotate_instance_identity(state_dir: str | Path) -> InstanceIdentity:
    """Explicitly replace an identity after a copied state disk is adopted."""
    root = Path(state_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = InstanceIdentity(instance_id=uuid.uuid4().hex, created_at=_utc_now())
    temporary = root / f".{_INSTANCE_FILE}.{uuid.uuid4().hex}.tmp"
    path = root / _INSTANCE_FILE
    try:
        temporary.write_text(json.dumps(identity.to_dict(), sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return identity


def _cuda_version(runner: CommandRunner) -> str | None:
    try:
        output = runner(["nvcc", "--version"]).decode("utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"release\s+([0-9]+\.[0-9]+)", output)
    return match.group(1) if match else None


def detect_hardware_identity(
    *,
    runner: CommandRunner = _run,
    environment: dict[str, str] | None = None,
) -> HardwareIdentity:
    """Probe the selected GPU without product-name enums or free-memory input."""
    environment = environment or os.environ
    logical_id = environment.get("KERNELBLASTER_GPU_DEVICE", "0").strip()
    query = runner(
        [
            "nvidia-smi",
            f"--id={logical_id}",
            "--query-gpu=name,uuid,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    ).decode("utf-8", errors="replace").strip()
    fields = [field.strip() for field in query.split(",")]
    if len(fields) != 5:
        raise RuntimeError("nvidia-smi did not return name,uuid,memory,driver,compute capability")
    name, gpu_uuid, memory_mib, driver_version, compute_capability = fields
    if not re.fullmatch(r"[0-9]+\.[0-9]+", compute_capability):
        raise RuntimeError("GPU compute capability could not be detected")
    return HardwareIdentity(
        logical_id=logical_id,
        name=name,
        gpu_uuid=gpu_uuid or None,
        compute_capability=compute_capability,
        target_arch="sm_" + compute_capability.replace(".", ""),
        total_memory_bytes=int(memory_mib) * 1024 * 1024 if memory_mib.isdigit() else None,
        driver_version=driver_version or None,
        cuda_version=_cuda_version(runner),
        image_digest=environment.get("KERNELBLASTER_SUPERVISOR_IMAGE_DIGEST") or None,
        runtime=environment.get("KERNELBLASTER_RUNTIME_ID") or None,
    )
