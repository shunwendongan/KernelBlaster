"""Explicit, non-scheduling SSH targets for independently run instances."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import tomllib


_SAFE_ALIAS = re.compile(r"^[A-Za-z0-9_.@:-]+$")
_SAFE_WORKDIR = re.compile(r"^/[A-Za-z0-9._/-]*$")
_SENSITIVE_PARTS = ("password", "token", "secret", "private_key", "identity_file")
_OPERATIONS = {
    "preflight": "scripts/run_preflight.py",
    "smoke": "scripts/compose_smoke.py",
    "export": "scripts/export_run.py",
}


@dataclass(frozen=True)
class SshTarget:
    target_id: str
    ssh_alias: str
    workdir: str
    profile: str | None = None

    def command(self, operation: str, arguments: list[str] | None = None) -> list[str]:
        if operation not in _OPERATIONS:
            raise ValueError(f"unsupported explicit target operation: {operation}")
        if (
            not _SAFE_ALIAS.fullmatch(self.ssh_alias)
            or not _SAFE_WORKDIR.fullmatch(self.workdir)
            or ".." in Path(self.workdir).parts
        ):
            raise ValueError("target alias or workdir contains unsafe characters")
        safe_arguments = arguments or []
        if any(not re.fullmatch(r"[A-Za-z0-9_./:=+,-]+", value) for value in safe_arguments):
            raise ValueError("target argument contains unsupported characters")
        command = ["cd", "--", self.workdir, "&&", "python", _OPERATIONS[operation], *safe_arguments]
        if self.profile:
            command.extend(["--profile", self.profile])
        return ["ssh", self.ssh_alias, "--", " ".join(command)]


def load_targets(path: str | Path) -> dict[str, SshTarget]:
    """Load a deliberately small, secret-free targets.toml schema."""
    with Path(path).expanduser().open("rb") as stream:
        payload = tomllib.load(stream)
    configured = dict(payload.get("targets") or {})
    result: dict[str, SshTarget] = {}
    for target_id, raw in configured.items():
        values = dict(raw or {})
        lowered = " ".join(values).lower()
        if any(part in lowered for part in _SENSITIVE_PARTS):
            raise ValueError("targets.toml may not contain credentials or private key material")
        if set(values) - {"ssh_alias", "workdir", "profile"}:
            raise ValueError("targets.toml contains unsupported fields")
        target = SshTarget(
            target_id=str(target_id),
            ssh_alias=str(values.get("ssh_alias") or ""),
            workdir=str(values.get("workdir") or ""),
            profile=str(values["profile"]) if values.get("profile") else None,
        )
        target.command("preflight")
        result[target.target_id] = target
    return result


def run_explicit_target(target: SshTarget, operation: str, arguments: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run only a whitelisted operation on the user-selected target."""
    return subprocess.run(target.command(operation, arguments), check=False, text=True)
