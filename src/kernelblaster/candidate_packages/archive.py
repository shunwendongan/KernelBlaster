"""Minimal deterministic archive helpers for untrusted candidate artifacts.

This module deliberately has no dependency on the Supervisor or Docker SDK so
the fixed GPU Job runtime can validate packages without importing privileged
control-plane code.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path, PurePosixPath
import tarfile
from typing import Mapping


def _safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("candidate archive members must be safe relative POSIX paths")
    return path.as_posix()


def build_archive(files: Mapping[str, bytes]) -> bytes:
    """Build a byte-stable USTAR archive containing regular files only."""
    if not files:
        raise ValueError("candidate archive may not be empty")
    output = BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for raw_name, payload in sorted(files.items()):
            name = _safe_name(raw_name)
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o444
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            archive.addfile(info, BytesIO(payload))
    return output.getvalue()


def archive_files(
    payload: bytes, *, maximum_bytes: int = 64 * 1024 * 1024
) -> dict[str, bytes]:
    """Safely read a bounded archive and reject links, devices and duplicates."""
    if len(payload) > maximum_bytes:
        raise ValueError("candidate archive exceeds the fixed size limit")
    result: dict[str, bytes] = {}
    with tarfile.open(fileobj=BytesIO(payload), mode="r:*") as archive:
        for member in archive.getmembers():
            name = _safe_name(member.name)
            if name in result:
                raise ValueError("candidate archive contains a duplicate member")
            if not member.isfile() or member.issym() or member.islnk() or member.isdev():
                raise ValueError("candidate archive may contain regular files only")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("candidate archive member is unreadable")
            result[name] = stream.read()
    if not result:
        raise ValueError("candidate archive may not be empty")
    return result


def extract_archive(payload: bytes, destination: str | Path) -> tuple[Path, ...]:
    """Extract validated regular files without using tarfile.extract()."""
    files = archive_files(payload)
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    for name, contents in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.parent.resolve().is_relative_to(root):
            raise ValueError("candidate archive extraction escapes the staging root")
        target.write_bytes(contents)
        extracted.append(target)
    return tuple(extracted)


__all__ = ["archive_files", "build_archive", "extract_archive"]
