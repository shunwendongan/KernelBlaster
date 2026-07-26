"""Offline Ed25519 signing for immutable trusted Adapter plugin bundles."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
import tarfile
from typing import Mapping, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from pydantic import Field, model_validator

from ..gpu_jobs.bundles import build_deterministic_bundle, validate_bundle
from .contracts import AdapterKind, StrictModel


class PluginFile(StrictModel):
    path: str = Field(pattern=r"^[A-Za-z0-9_.][A-Za-z0-9_./-]{0,254}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size: int = Field(ge=0, le=64 * 1024 * 1024)


class PluginAdapter(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
    version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    kind: AdapterKind
    task_ids: tuple[str, ...]


class AdapterPluginManifest(StrictModel):
    schema_version: Literal["adapter-plugin/v1"] = "adapter-plugin/v1"
    plugin_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
    version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    key_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
    adapters: tuple[PluginAdapter, ...]
    files: tuple[PluginFile, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> "AdapterPluginManifest":
        paths = [item.path for item in self.files]
        if not paths or len(paths) != len(set(paths)):
            raise ValueError("plugin files must be non-empty and unique")
        if any(path.startswith("/") or ".." in path.split("/") for path in paths):
            raise ValueError("plugin files must use safe relative paths")
        adapters = [(item.id, item.version) for item in self.adapters]
        if not adapters or len(adapters) != len(set(adapters)):
            raise ValueError("plugin adapters must be non-empty and unique")
        if any(not adapter.task_ids for adapter in self.adapters):
            raise ValueError("plugin adapters require task IDs")
        return self

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")


class TrustedAdapterKey(StrictModel):
    key_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
    public_key_base64: str

    def raw_key(self) -> bytes:
        try:
            value = base64.b64decode(self.public_key_base64, validate=True)
        except ValueError as error:
            raise ValueError("trusted Adapter public key is not base64") from error
        if len(value) != 32:
            raise ValueError("trusted Adapter public keys must contain 32 raw bytes")
        return value


class TrustedAdapterKeys(StrictModel):
    schema_version: Literal["adapter-trusted-keys/v1"] = "adapter-trusted-keys/v1"
    keys: tuple[TrustedAdapterKey, ...] = ()

    @model_validator(mode="after")
    def unique_keys(self) -> "TrustedAdapterKeys":
        identifiers = [item.key_id for item in self.keys]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("trusted Adapter key IDs must be unique")
        for item in self.keys:
            item.raw_key()
        return self

    def mapping(self) -> dict[str, bytes]:
        return {item.key_id: item.raw_key() for item in self.keys}


class AllowedAdapterPlugin(StrictModel):
    plugin_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
    version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    key_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    adapter_ids: tuple[str, ...]


class AdapterPluginAllowlist(StrictModel):
    schema_version: Literal["adapter-plugin-allowlist/v1"] = "adapter-plugin-allowlist/v1"
    plugins: tuple[AllowedAdapterPlugin, ...] = ()

    @model_validator(mode="after")
    def unique_plugins(self) -> "AdapterPluginAllowlist":
        identities = [(item.plugin_id, item.version) for item in self.plugins]
        if len(identities) != len(set(identities)):
            raise ValueError("allowlisted Adapter plugins must be unique")
        return self


def _private_key(raw: bytes) -> Ed25519PrivateKey:
    if len(raw) != 32:
        raise ValueError("Ed25519 private keys must contain 32 raw bytes")
    return Ed25519PrivateKey.from_private_bytes(raw)


def _public_key(raw: bytes) -> Ed25519PublicKey:
    if len(raw) != 32:
        raise ValueError("Ed25519 public keys must contain 32 raw bytes")
    return Ed25519PublicKey.from_public_bytes(raw)


def build_signed_plugin(
    files: Mapping[str, bytes],
    *,
    plugin_id: str,
    version: str,
    key_id: str,
    adapters: tuple[PluginAdapter, ...],
    private_key: bytes,
) -> bytes:
    """Build a byte-stable tar whose manifest authenticates every payload."""
    if any(name in {"plugin-manifest.json", "plugin-signature.ed25519"} for name in files):
        raise ValueError("plugin payload uses a reserved filename")
    entries = tuple(
        PluginFile(path=name, sha256=hashlib.sha256(payload).hexdigest(), size=len(payload))
        for name, payload in sorted(files.items())
    )
    manifest = AdapterPluginManifest(
        plugin_id=plugin_id,
        version=version,
        key_id=key_id,
        adapters=adapters,
        files=entries,
    )
    canonical = manifest.canonical_bytes()
    signature = _private_key(private_key).sign(canonical)
    return build_deterministic_bundle(
        {
            **files,
            "plugin-manifest.json": canonical,
            "plugin-signature.ed25519": base64.b64encode(signature),
        }
    )


def verify_signed_plugin(
    payload: bytes, *, trusted_public_keys: Mapping[str, bytes]
) -> AdapterPluginManifest:
    """Verify archive shape, publisher identity, signature, size and file hashes."""
    names = validate_bundle(payload)
    required = {"plugin-manifest.json", "plugin-signature.ed25519"}
    if not required.issubset(names):
        raise ValueError("signed plugin is missing its manifest or signature")
    with tarfile.open(fileobj=BytesIO(payload), mode="r:*") as archive:
        members: dict[str, bytes] = {}
        for name in names:
            stream = archive.extractfile(name)
            assert stream is not None
            members[name] = stream.read()
    manifest = AdapterPluginManifest.model_validate_json(members["plugin-manifest.json"])
    if members["plugin-manifest.json"] != manifest.canonical_bytes():
        raise ValueError("plugin manifest is not canonically encoded")
    raw_public = trusted_public_keys.get(manifest.key_id)
    if raw_public is None:
        raise ValueError("plugin signing key is not trusted")
    try:
        signature = base64.b64decode(members["plugin-signature.ed25519"], validate=True)
        _public_key(raw_public).verify(signature, manifest.canonical_bytes())
    except (ValueError, InvalidSignature) as error:
        raise ValueError("plugin signature is invalid") from error
    expected_names = {item.path for item in manifest.files} | required
    if set(names) != expected_names:
        raise ValueError("plugin archive does not match its manifest")
    for item in manifest.files:
        content = members[item.path]
        if len(content) != item.size or hashlib.sha256(content).hexdigest() != item.sha256:
            raise ValueError("plugin payload hash mismatch")
    return manifest


def verify_allowlisted_plugin(
    payload: bytes,
    *,
    trusted_keys: TrustedAdapterKeys,
    allowlist: AdapterPluginAllowlist,
) -> AdapterPluginManifest:
    digest = hashlib.sha256(payload).hexdigest()
    allowed = next((item for item in allowlist.plugins if item.sha256 == digest), None)
    if allowed is None:
        raise ValueError("plugin digest is not allowlisted")
    manifest = verify_signed_plugin(payload, trusted_public_keys=trusted_keys.mapping())
    if (manifest.plugin_id, manifest.version, manifest.key_id) != (
        allowed.plugin_id,
        allowed.version,
        allowed.key_id,
    ):
        raise ValueError("plugin identity does not match the allowlist")
    if {item.id for item in manifest.adapters} != set(allowed.adapter_ids):
        raise ValueError("plugin Adapter IDs do not match the allowlist")
    return manifest


__all__ = [
    "AdapterPluginManifest",
    "AdapterPluginAllowlist",
    "AllowedAdapterPlugin",
    "PluginAdapter",
    "PluginFile",
    "TrustedAdapterKey",
    "TrustedAdapterKeys",
    "build_signed_plugin",
    "verify_signed_plugin",
    "verify_allowlisted_plugin",
]
