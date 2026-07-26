from __future__ import annotations

from pathlib import Path

import pytest

from src.kernelblaster.portability import (
    build_aggregate_report,
    detect_hardware_identity,
    load_or_create_instance_identity,
    load_targets,
    rotate_instance_identity,
)
from src.kernelblaster.storage import StateStore


def test_instance_identity_is_stable_until_explicit_rotation(tmp_path: Path) -> None:
    first = load_or_create_instance_identity(tmp_path)
    second = load_or_create_instance_identity(tmp_path)
    rotated = rotate_instance_identity(tmp_path)
    assert first.instance_id == second.instance_id
    assert rotated.instance_id != first.instance_id


def test_hardware_identity_separates_audit_and_comparison_identity() -> None:
    def runner(command: list[str]) -> bytes:
        if command[0] == "nvidia-smi":
            return b"Future GPU, GPU-a, 24576, 600.1, 9.9\n"
        return b"Cuda compilation tools, release 13.0, V13.0.1\n"

    first = detect_hardware_identity(runner=runner, environment={})

    def other_uuid(command: list[str]) -> bytes:
        if command[0] == "nvidia-smi":
            return b"Future GPU, GPU-b, 24576, 600.1, 9.9\n"
        return b"Cuda compilation tools, release 13.0, V13.0.1\n"

    second = detect_hardware_identity(runner=other_uuid, environment={})
    assert first.audit_fingerprint != second.audit_fingerprint
    assert first.comparison_group == second.comparison_group
    assert first.target_arch == "sm_99"


def test_targets_reject_secrets_and_only_build_whitelisted_commands(tmp_path: Path) -> None:
    configured = tmp_path / "targets.toml"
    configured.write_text(
        "[targets.demo]\nssh_alias = 'host-alias'\nworkdir = '/workspace/kernelblaster'\n",
        encoding="utf-8",
    )
    target = load_targets(configured)["demo"]
    assert target.command("preflight")[:3] == ["ssh", "host-alias", "--"]
    with pytest.raises(ValueError, match="unsupported"):
        target.command("shell")

    configured.write_text(
        "[targets.bad]\nssh_alias = 'host'\nworkdir = '/work'\ntoken = 'nope'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="credentials"):
        load_targets(configured)


def test_aggregate_report_never_combines_comparison_groups(tmp_path: Path) -> None:
    store = StateStore(state_dir=tmp_path / "state")
    for run_id, group in (("a", "sha256:one"), ("b", "sha256:two")):
        store.repository.create_run(run_id)
        store.repository.bind_run_portability(
            run_id=run_id,
            source_instance_id=store.instance_identity.instance_id,
            target_id="local",
            comparison_group=group,
        )
        store.repository.finish_run(run_id, "completed")
    report = build_aggregate_report(store.repository)
    assert len(report["comparison_groups"]) == 2
    assert {row["performance_ranking"] for row in report["comparison_groups"]} == {
        "incomparable_across_groups"
    }
