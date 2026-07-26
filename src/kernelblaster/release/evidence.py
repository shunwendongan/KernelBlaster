"""Sanitized, hash-indexed release evidence generation and verification."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping


_SENSITIVE = re.compile(r"api[_-]?key|authorization|credential|password|secret|token|private|seed", re.I)
_PATH_KEYS = re.compile(r"(?:^|_)(?:path|home|hostname|host|instance|state_dir)(?:$|_)", re.I)
_SCHEMA_ROOT = Path(__file__).resolve().parents[3] / "docs" / "schemas"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _sanitize(value: Any, key: str = "") -> Any:
    if _SENSITIVE.search(key):
        return "[redacted]"
    if _PATH_KEYS.search(key) and isinstance(value, str):
        return "[redacted]"
    if isinstance(value, dict):
        return {str(child): _sanitize(item, str(child)) for child, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item, key) for item in value]
    return value


def _slug(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return result or "unknown-gpu"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_commit() -> str:
    """Return the checked-out commit without making evidence generation depend on Git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "unavailable"


def _schema_digests() -> dict[str, str]:
    return {
        schema.name: _sha256(schema)
        for schema in sorted(_SCHEMA_ROOT.glob("*-v1.schema.json"))
        if schema.is_file()
    }


def build_release_manifest(*, scope: str, profile: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    sanitized_profile = _sanitize(dict(profile))
    return _sanitize({
        "schema_version": "release-evidence/v1",
        "scope": scope,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "commit": _git_commit(),
        "profile": sanitized_profile,
        "profile_sha256": _canonical_sha256(sanitized_profile),
        "schema_digests": _schema_digests(),
        "evidence": dict(evidence),
    })


def _copy_sanitized_json_source(source: Path, destination: Path) -> None:
    """Keep release evidence public: raw profiler/log artifacts stay outside Git."""
    if source.suffix.lower() != ".json":
        raise ValueError("release evidence sources must be sanitized JSON summaries")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("release evidence source must be valid UTF-8 JSON") from error
    destination.write_text(
        json.dumps(_sanitize(payload), sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_release_evidence(
    root: str | Path,
    *,
    scope: str,
    profile: Mapping[str, Any],
    evidence: Mapping[str, Any],
    source_files: Mapping[str, str | Path] | None = None,
) -> Path:
    """Write redacted manifests and a whole-tree SHA256 index."""
    destination = Path(root).expanduser().resolve()
    scope_dir = destination / scope
    scope_dir.mkdir(parents=True, exist_ok=False)
    schema_dir = destination / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    for schema_name in (
        "release-evidence-v1.schema.json",
        "run-bundle-v1.schema.json",
        "aggregate-report-v1.schema.json",
    ):
        schema = _SCHEMA_ROOT / schema_name
        if schema.is_file():
            shutil.copyfile(schema, schema_dir / schema_name)
    manifest = build_release_manifest(scope=scope, profile=profile, evidence=evidence)
    (scope_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    for name, source in (source_files or {}).items():
        safe_name = Path(name).name
        source_path = Path(source)
        if not source_path.is_file() or source_path.is_symlink():
            raise ValueError("release evidence source must be a regular file")
        _copy_sanitized_json_source(source_path, scope_dir / safe_name)
    index: dict[str, str] = {}
    for path in sorted(candidate for candidate in destination.rglob("*") if candidate.is_file() and candidate.name != "SHA256SUMS.json"):
        index[path.relative_to(destination).as_posix()] = _sha256(path)
    (destination / "SHA256SUMS.json").write_text(json.dumps(index, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return scope_dir


def verify_release_evidence(root: str | Path) -> dict[str, Any]:
    destination = Path(root).expanduser().resolve()
    index_path = destination / "SHA256SUMS.json"
    if not index_path.is_file():
        raise FileNotFoundError("release evidence SHA256SUMS.json is missing")
    expected = json.loads(index_path.read_text(encoding="utf-8"))
    failures = [name for name, digest in expected.items() if not (destination / name).is_file() or _sha256(destination / name) != digest]
    actual = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.json"
    }
    unexpected = sorted(actual.difference(expected))
    return {
        "valid": not failures and not unexpected,
        "file_count": len(expected),
        "failures": failures,
        "unexpected_files": unexpected,
    }
