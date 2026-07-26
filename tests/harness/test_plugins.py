from __future__ import annotations

import base64
import hashlib

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from src.kernelblaster.harness.plugins import (
    AdapterPluginAllowlist,
    AllowedAdapterPlugin,
    PluginAdapter,
    TrustedAdapterKey,
    TrustedAdapterKeys,
    build_signed_plugin,
    verify_allowlisted_plugin,
    verify_signed_plugin,
)


def _keys() -> tuple[bytes, bytes]:
    key = Ed25519PrivateKey.generate()
    private = key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    return private, public


def test_signed_adapter_plugin_is_deterministic_and_verifies_publisher_and_files():
    private, public = _keys()
    kwargs = {
        "plugin_id": "example.project",
        "version": "1.0.0",
        "key_id": "local-owner",
        "adapters": (
            PluginAdapter(
                id="example.elementwise",
                version="1.0.0",
                kind="declarative",
                task_ids=("example.relu.forward",),
            ),
        ),
        "private_key": private,
    }
    first = build_signed_plugin({"task.json": b"{}", "adapter.py": b"# trusted\n"}, **kwargs)
    second = build_signed_plugin({"adapter.py": b"# trusted\n", "task.json": b"{}"}, **kwargs)
    assert first == second
    manifest = verify_signed_plugin(first, trusted_public_keys={"local-owner": public})
    assert manifest.plugin_id == "example.project"
    assert {item.path for item in manifest.files} == {"adapter.py", "task.json"}


def test_signed_plugin_rejects_unknown_or_wrong_key():
    private, public = _keys()
    other_private, other_public = _keys()
    plugin = build_signed_plugin(
        {"task.json": b"{}"},
        plugin_id="example.project",
        version="1",
        key_id="owner",
        adapters=(
            PluginAdapter(
                id="example.adapter",
                version="1",
                kind="trusted_code",
                task_ids=("example.task.forward",),
            ),
        ),
        private_key=private,
    )
    with pytest.raises(ValueError, match="not trusted"):
        verify_signed_plugin(plugin, trusted_public_keys={})
    with pytest.raises(ValueError, match="signature is invalid"):
        verify_signed_plugin(plugin, trusted_public_keys={"owner": other_public})
    assert public != other_public and private != other_private


def test_image_build_policy_requires_digest_key_identity_and_adapter_allowlist():
    private, public = _keys()
    plugin = build_signed_plugin(
        {"adapter.py": b"# trusted\n"},
        plugin_id="example.project",
        version="1",
        key_id="owner",
        adapters=(
            PluginAdapter(
                id="example.adapter",
                version="1",
                kind="trusted_code",
                task_ids=("example.task.forward",),
            ),
        ),
        private_key=private,
    )
    keys = TrustedAdapterKeys(
        keys=(
            TrustedAdapterKey(
                key_id="owner",
                public_key_base64=base64.b64encode(public).decode("ascii"),
            ),
        )
    )
    allowed = AllowedAdapterPlugin(
        plugin_id="example.project",
        version="1",
        key_id="owner",
        sha256=hashlib.sha256(plugin).hexdigest(),
        adapter_ids=("example.adapter",),
    )
    manifest = verify_allowlisted_plugin(
        plugin,
        trusted_keys=keys,
        allowlist=AdapterPluginAllowlist(plugins=(allowed,)),
    )
    assert manifest.key_id == "owner"
    with pytest.raises(ValueError, match="not allowlisted"):
        verify_allowlisted_plugin(
            plugin,
            trusted_keys=keys,
            allowlist=AdapterPluginAllowlist(),
        )
