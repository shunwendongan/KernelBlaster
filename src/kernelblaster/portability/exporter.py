"""Export terminal standalone runs as deterministic, verifiable bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..storage.state import StateStore
from .archive import file_sha256, write_deterministic_tar
from .contracts import RUN_BUNDLE_SCHEMA, canonical_bytes, sha256


@dataclass(frozen=True)
class ExportResult:
    bundle_path: Path
    bundle_hash: str
    content_hash: str
    payload_count: int


def export_run(
    store: StateStore,
    run_id: str,
    destination: str | Path,
    *,
    allow_incomplete: bool = False,
    gzip_compress: bool = False,
    maximum_payload_bytes: int = 4 * 1024**3,
    maximum_members: int = 10_000,
) -> ExportResult:
    """Export a run closure without exporting unrelated CAS content."""
    snapshot = store.repository.snapshot_run(run_id)
    status = str(snapshot["run"].get("status") or "")
    if not allow_incomplete and status in {"pending", "running", "leased"}:
        raise ValueError("only terminal runs may be exported without --allow-incomplete")
    artifacts = sorted(snapshot.get("artifacts") or [], key=lambda item: str(item["digest"]))
    if len(artifacts) + 1 > maximum_members:
        raise ValueError("run bundle exceeds configured member limit")
    payloads: list[dict[str, Any]] = []
    members: list[tuple[str, Path | bytes]] = []
    total_bytes = 0
    for artifact in artifacts:
        digest = str(artifact["digest"])
        verified = store.cas.verify(digest)
        if verified.size_bytes != int(artifact["size_bytes"]):
            raise ValueError(f"artifact metadata size mismatch for {digest}")
        total_bytes += verified.size_bytes
        if total_bytes > maximum_payload_bytes:
            raise ValueError("run bundle exceeds configured payload limit")
        member = f"payloads/{digest}"
        payloads.append({"digest": digest, "path": member, "sha256": digest, "size_bytes": verified.size_bytes})
        members.append((member, store.cas.get_path(digest)))
    content_hash = sha256(snapshot)
    manifest = {
        "schema_version": RUN_BUNDLE_SCHEMA,
        "content_hash": content_hash,
        "snapshot": snapshot,
        "payloads": payloads,
        "incomplete": status in {"pending", "running", "leased"},
    }
    destination_path = Path(destination).expanduser()
    members.append(("manifest.json", canonical_bytes(manifest) + b"\n"))
    write_deterministic_tar(destination_path, members, gzip_compress=gzip_compress)
    bundle_hash = file_sha256(destination_path)
    return ExportResult(
        bundle_path=destination_path,
        bundle_hash=bundle_hash,
        content_hash=content_hash,
        payload_count=len(payloads),
    )
