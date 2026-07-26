#!/usr/bin/env python3
"""Verify an external Adapter bundle and build it into an immutable Job image."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kernelblaster.harness import (  # noqa: E402
    AdapterPluginAllowlist,
    TrustedAdapterKeys,
    verify_allowlisted_plugin,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--trusted-keys", type=Path, required=True)
    parser.add_argument("--allowlist", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    payload = args.bundle.read_bytes()
    keys = TrustedAdapterKeys.model_validate_json(args.trusted_keys.read_bytes())
    allowlist = AdapterPluginAllowlist.model_validate_json(args.allowlist.read_bytes())
    manifest = verify_allowlisted_plugin(payload, trusted_keys=keys, allowlist=allowlist)
    digest = hashlib.sha256(payload).hexdigest()
    with tempfile.TemporaryDirectory(prefix="kernelblaster-adapter-build-") as temporary:
        context = Path(temporary)
        (context / "docker" / "adapter-plugins").mkdir(parents=True)
        (context / "src" / "kernelblaster" / "gpu_jobs").mkdir(parents=True)
        (context / "src" / "kernelblaster" / "harness").mkdir(parents=True)
        shutil.copy2(ROOT / "docker" / "Dockerfile", context / "docker" / "Dockerfile")
        shutil.copy2(
            ROOT / "src" / "kernelblaster" / "gpu_jobs" / "sandbox_runner.py",
            context / "src" / "kernelblaster" / "gpu_jobs" / "sandbox_runner.py",
        )
        shutil.copy2(
            ROOT / "src" / "kernelblaster" / "harness" / "contracts.py",
            context / "src" / "kernelblaster" / "harness" / "contracts.py",
        )
        shutil.copy2(args.trusted_keys, context / "docker" / "trusted-adapter-keys.json")
        shutil.copy2(args.allowlist, context / "docker" / "adapter-plugin-allowlist.json")
        (context / "docker" / "adapter-plugins" / f"{digest}.tar").write_bytes(payload)
        subprocess.run(
            [
                "docker",
                "build",
                "--target",
                "gpu-job",
                "--tag",
                args.tag,
                "--file",
                str(context / "docker" / "Dockerfile"),
                str(context),
            ],
            check=True,
        )
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", args.tag], text=True
    ).strip()
    print(
        json.dumps(
            {
                "adapter_bundle_digest": digest,
                "image": args.tag,
                "image_digest": image_id,
                "plugin_id": manifest.plugin_id,
                "plugin_version": manifest.version,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
