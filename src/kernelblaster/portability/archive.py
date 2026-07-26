"""Deterministic, bounded archive primitives for portable run bundles."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path, PurePosixPath
import tarfile
from typing import BinaryIO, Iterable


def safe_member_name(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("bundle members must be safe relative POSIX paths")
    return path.as_posix()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(safe_member_name(name))
    info.size = size
    info.mode = 0o444
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    return info


def write_deterministic_tar(
    destination: str | Path,
    members: Iterable[tuple[str, Path | bytes]],
    *,
    gzip_compress: bool = False,
) -> None:
    """Write sorted regular-file members with stable tar metadata.

    ``.tar.gz`` is deterministic as well: its mtime and original filename are
    blanked rather than inherited from the destination path.
    """
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(((safe_member_name(name), payload) for name, payload in members), key=lambda item: item[0])
    with target.open("wb") as raw:
        outer: BinaryIO | gzip.GzipFile
        if gzip_compress:
            outer = gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="")
        else:
            outer = raw
        try:
            with tarfile.open(fileobj=outer, mode="w|") as archive:
                for name, payload in ordered:
                    if isinstance(payload, bytes):
                        archive.addfile(_tar_info(name, len(payload)), fileobj=_BytesReader(payload))
                    else:
                        source = Path(payload)
                        if not source.is_file() or source.is_symlink():
                            raise ValueError("bundle payload must be a regular non-symbolic-link file")
                        with source.open("rb") as stream:
                            archive.addfile(_tar_info(name, source.stat().st_size), fileobj=stream)
        finally:
            if gzip_compress:
                outer.close()


class _BytesReader:
    """Minimal file-like reader that keeps tarfile streaming for byte members."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self.payload) - self.offset
        result = self.payload[self.offset : self.offset + size]
        self.offset += len(result)
        return result
