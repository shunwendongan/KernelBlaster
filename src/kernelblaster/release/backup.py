"""SQLite-first state backup and explicit restore primitives for release rollback."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sqlite(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        result = connection.execute("PRAGMA integrity_check").fetchone()
    if result is None or result[0] != "ok":
        raise ValueError("SQLite integrity check failed")


def create_state_backup(state_dir: str | Path, destination: str | Path) -> Path:
    """Create a consistent SQLite snapshot and a small immutable manifest."""
    source_root = Path(state_dir).expanduser().resolve()
    database = source_root / "control.sqlite3"
    if not database.is_file():
        raise FileNotFoundError("control.sqlite3 does not exist in state directory")
    root = Path(destination).expanduser().resolve() / f"kernelblaster-state-{_timestamp()}"
    root.mkdir(parents=True, exist_ok=False)
    target = root / "control.sqlite3"
    source_uri = f"file:{database.as_posix()}?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as source, sqlite3.connect(target) as snapshot:
        source.backup(snapshot)
    _validate_sqlite(target)
    payload: dict[str, Any] = {
        "schema_version": "state-backup/v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_state_dir": "redacted",
        "sqlite_file": target.name,
        "sqlite_size_bytes": target.stat().st_size,
        "sqlite_sha256": _sha256(target),
        "cas_is_immutable_and_not_copied": True,
    }
    (root / "manifest.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return root


def restore_state_backup(backup_dir: str | Path, state_dir: str | Path, *, confirm: bool = False) -> Path:
    """Restore only when the caller explicitly confirms the destructive replacement."""
    if not confirm:
        raise ValueError("restore requires explicit confirmation")
    root = Path(backup_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    source = root / "control.sqlite3"
    target = Path(state_dir).expanduser().resolve() / "control.sqlite3"
    if not source.is_file():
        raise FileNotFoundError("backup control.sqlite3 is missing")
    if not manifest_path.is_file():
        raise FileNotFoundError("backup manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "state-backup/v1" or manifest.get("sqlite_file") != source.name:
        raise ValueError("backup manifest is not a state-backup/v1 manifest")
    if manifest.get("sqlite_sha256") != _sha256(source):
        raise ValueError("backup SQLite digest does not match its manifest")
    _validate_sqlite(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".restore.tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite stale restore file: {temporary}")
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            for chunk in iter(lambda: input_stream.read(1024 * 1024), b""):
                output_stream.write(chunk)
        _validate_sqlite(temporary)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target
