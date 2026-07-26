#!/usr/bin/env python3
"""Create offline keys and verify signed trusted Adapter plugin bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kernelblaster.harness.contracts import AdapterKind  # noqa: E402
from src.kernelblaster.harness.plugins import (  # noqa: E402
    PluginAdapter,
    build_signed_plugin,
    verify_signed_plugin,
)


def _keygen(private_path: Path, public_path: Path) -> None:
    if private_path.exists() or public_path.exists():
        raise FileExistsError("refusing to overwrite an Adapter signing key")
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    key = Ed25519PrivateKey.generate()
    private_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
    )
    private_path.chmod(0o600)


def _build(descriptor_path: Path, payload_dir: Path, private_path: Path, output: Path) -> None:
    private_path = private_path.expanduser().resolve()
    if private_path.is_relative_to(ROOT):
        raise RuntimeError("Adapter private keys must remain outside the repository")
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    allowed = {"plugin_id", "version", "key_id", "adapters"}
    if not isinstance(descriptor, dict) or set(descriptor) != allowed:
        raise ValueError("descriptor must contain only plugin_id/version/key_id/adapters")
    adapters = tuple(
        PluginAdapter(
            id=item["id"],
            version=item["version"],
            kind=AdapterKind(item["kind"]),
            task_ids=tuple(item["task_ids"]),
        )
        for item in descriptor["adapters"]
    )
    files: dict[str, bytes] = {}
    root = payload_dir.resolve()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("plugin payload may not contain symlinks")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    bundle = build_signed_plugin(
        files,
        plugin_id=descriptor["plugin_id"],
        version=descriptor["version"],
        key_id=descriptor["key_id"],
        adapters=adapters,
        private_key=private_path.read_bytes(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bundle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    keygen = subparsers.add_parser("keygen")
    keygen.add_argument("--private-key", type=Path, required=True)
    keygen.add_argument("--public-key", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--key-id", required=True)
    verify.add_argument("--public-key", type=Path, required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--descriptor", type=Path, required=True)
    build.add_argument("--payload-dir", type=Path, required=True)
    build.add_argument("--private-key", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "keygen":
        _keygen(args.private_key, args.public_key)
        return 0
    if args.command == "build":
        _build(args.descriptor, args.payload_dir, args.private_key, args.output)
        return 0
    manifest = verify_signed_plugin(
        args.bundle.read_bytes(),
        trusted_public_keys={args.key_id: args.public_key.read_bytes()},
    )
    print(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
