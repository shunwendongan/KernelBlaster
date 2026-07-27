"""Strict staging importer for untrusted portable run bundles."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
from typing import Any

from ..storage.cas import ArtifactMetadata
from ..storage.state import StateStore
from .archive import file_sha256, safe_member_name
from .contracts import RUN_BUNDLE_SCHEMA, sha256


@dataclass(frozen=True)
class ImportResult:
    run_id: str
    bundle_hash: str
    content_hash: str
    idempotent: bool


def _checked_members(
    archive: tarfile.TarFile,
    *,
    maximum_members: int,
    maximum_member_bytes: int,
    maximum_total_bytes: int,
) -> dict[str, tarfile.TarInfo]:
    result: dict[str, tarfile.TarInfo] = {}
    total = 0
    for member in archive:
        name = safe_member_name(member.name)
        if name in result:
            raise ValueError("run bundle contains duplicate member paths")
        if not member.isfile() or member.issym() or member.islnk() or member.isdev() or member.isfifo():
            raise ValueError("run bundle may contain regular files only")
        if member.size < 0 or member.size > maximum_member_bytes:
            raise ValueError("run bundle member exceeds configured size limit")
        total += member.size
        if total > maximum_total_bytes:
            raise ValueError("run bundle exceeds configured expanded size limit")
        result[name] = member
        if len(result) > maximum_members:
            raise ValueError("run bundle exceeds configured member limit")
    return result


def _read_member(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    source = archive.extractfile(member)
    if source is None:
        raise ValueError("run bundle member cannot be read")
    return source.read()


def import_run(
    store: StateStore,
    bundle: str | Path,
    *,
    maximum_members: int = 10_000,
    maximum_member_bytes: int = 2 * 1024**3,
    maximum_total_bytes: int = 4 * 1024**3,
) -> ImportResult:
    """Import an untrusted bundle after full validation and CAS staging."""
    bundle_path = Path(bundle).expanduser()
    if not bundle_path.is_file() or bundle_path.is_symlink():
        raise ValueError("run bundle must be a regular file")
    bundle_hash = file_sha256(bundle_path)
    compressed_size = max(bundle_path.stat().st_size, 1)
    with tarfile.open(bundle_path, mode="r:*") as archive:
        members = _checked_members(
            archive,
            maximum_members=maximum_members,
            maximum_member_bytes=maximum_member_bytes,
            maximum_total_bytes=maximum_total_bytes,
        )
        if "manifest.json" not in members:
            raise ValueError("run bundle has no manifest.json")
        manifest = json.loads(_read_member(archive, members["manifest.json"]).decode("utf-8"))
        if manifest.get("schema_version") != RUN_BUNDLE_SCHEMA:
            raise ValueError("unsupported run bundle schema")
        snapshot = dict(manifest.get("snapshot") or {})
        content_hash = str(manifest.get("content_hash") or "")
        if not content_hash or content_hash != sha256(snapshot):
            raise ValueError("run bundle content hash mismatch")
        payloads = list(manifest.get("payloads") or [])
        expected_names = {"manifest.json"}
        for payload in payloads:
            digest = str(payload.get("digest") or "")
            path = safe_member_name(str(payload.get("path") or ""))
            if path != f"payloads/{digest}" or len(digest) != 64:
                raise ValueError("run bundle payload manifest is invalid")
            if str(payload.get("sha256") or "") != digest:
                raise ValueError("run bundle payload digest declaration is invalid")
            expected_names.add(path)
        if set(members) != expected_names:
            raise ValueError("run bundle contains unrecognized members")
        total_declared = sum(int(item.get("size_bytes") or -1) for item in payloads)
        if total_declared < 0 or total_declared > maximum_total_bytes:
            raise ValueError("run bundle payload size declaration is invalid")
        if total_declared > compressed_size * 1000:
            raise ValueError("run bundle compression ratio exceeds safety limit")
        stage_root = Path(tempfile.mkdtemp(prefix=".bundle-import-", dir=store.state_dir))
        try:
            staged: dict[str, Path] = {}
            for payload in payloads:
                digest = str(payload["digest"])
                member = members[f"payloads/{digest}"]
                if member.size != int(payload["size_bytes"]):
                    raise ValueError("run bundle payload size mismatch")
                target = stage_root / digest
                source = archive.extractfile(member)
                assert source is not None
                actual = hashlib.sha256()
                with target.open("wb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        actual.update(chunk)
                        output.write(chunk)
                if actual.hexdigest() != digest:
                    raise ValueError("run bundle payload hash mismatch")
                staged[digest] = target
            artifact_metadata = {str(item["digest"]): item for item in snapshot.get("artifacts") or []}
            if set(artifact_metadata) != set(staged):
                raise ValueError("run bundle artifact index and payload set differ")
            for digest, source in staged.items():
                source_metadata = artifact_metadata[digest]
                stored = store.cas.put_file(
                    source,
                    media_type=str(source_metadata.get("media_type") or "application/octet-stream"),
                    producer=source_metadata.get("producer"),
                    source_digest=source_metadata.get("source_digest"),
                    schema=source_metadata.get("schema_name"),
                )
                if stored.digest != digest:
                    raise ValueError("CAS import digest mismatch")
                store.repository.register_artifact(stored)
            imported = store.repository.import_portable_snapshot(
                snapshot, bundle_hash=bundle_hash, content_hash=content_hash
            )
        finally:
            shutil.rmtree(stage_root, ignore_errors=True)
    return ImportResult(
        run_id=str(imported["run"]["id"]),
        bundle_hash=bundle_hash,
        content_hash=content_hash,
        idempotent=bool(imported["idempotent"]),
    )
