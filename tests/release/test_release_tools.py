from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest
from scripts import e2e_release

from src.kernelblaster.release import (
    build_e2e_plan,
    create_state_backup,
    fault_plan,
    load_release_profile,
    restore_state_backup,
    verify_release_evidence,
    write_release_evidence,
)
from src.kernelblaster.release.e2e import ReleaseProfile
from src.kernelblaster.storage import StateStore


def test_release_profile_and_e2e_plan_are_configurable(tmp_path: Path, monkeypatch) -> None:
    profile = tmp_path / "release.toml"
    profile.write_text("[portability]\nmodel = 'profile-model'\nrollouts = 4\n", encoding="utf-8")
    monkeypatch.setenv("KERNELBLASTER_PROFILE_STEPS", "3")
    selected = load_release_profile(profile, overrides={"gpu": "detected-gpu"})
    plan = build_e2e_plan(selected, tmp_path / "e2e")
    assert selected.model == "profile-model"
    assert selected.rollouts == 4
    assert selected.steps == 3
    assert selected.gpu_label == "detected-gpu"
    assert plan[1]["command"][plan[1]["command"].index("--max-completion-tokens") + 1] == "64"
    agent = plan[2]
    assert "${CAPABILITY_REPORT_DIGEST}" in agent["command"]
    assert agent["environment"]["LLM_MAX_REQUESTS"] == "32"


def test_all_faults_default_to_simulated() -> None:
    safe = {entry["name"]: entry for entry in fault_plan()}
    real = {entry["name"]: entry for entry in fault_plan(allow_real_faults=True)}
    assert safe["oom"]["mode"] == "simulated"
    assert safe["illegal_access"]["mode"] == "simulated"
    assert real["oom"]["mode"] == "real"
    assert safe["ncu_tool_missing"]["mode"] == "simulated"


def test_e2e_executor_binds_agent_to_preflight_digest(tmp_path: Path, monkeypatch) -> None:
    plan = build_e2e_plan(ReleaseProfile(model="release-model"), tmp_path)
    calls: list[tuple[list[str], dict[str, str]]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs["env"]))
        stdout = 'KERNELBLASTER_PREFLIGHT_JSON {"report_digest": "capability-digest"}\n' if len(calls) == 1 else ""
        return CompletedProcess(command, 0, stdout=stdout)

    monkeypatch.setattr(e2e_release.subprocess, "run", fake_run)
    assert e2e_release.execute_e2e_plan(plan, tmp_path) == 0
    agent_command, agent_environment = calls[2]
    assert "capability-digest" in agent_command
    assert agent_environment["LLM_MAX_TOTAL_TOKENS"] == "250000"
    assert json.loads((tmp_path / "e2e-result.json").read_text(encoding="utf-8"))["stages"][2]["status"] == "passed"


def test_release_evidence_redacts_and_hashes_all_files(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"safe": True, "api_key": "must-not-copy"}), encoding="utf-8")
    root = tmp_path / "release-evidence"
    written = write_release_evidence(
        root,
        scope="local-future-gpu",
        profile={"model": "configured", "state_dir": "/secret/state"},
        evidence={"hostname": "host", "api_key": "never-publish", "status": "passed"},
        source_files={"result.json": source},
    )
    manifest = json.loads((written / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["profile"]["state_dir"] == "[redacted]"
    assert manifest["evidence"]["api_key"] == "[redacted]"
    assert len(manifest["profile_sha256"]) == 64
    assert json.loads((written / "result.json").read_text(encoding="utf-8"))["api_key"] == "[redacted]"
    assert verify_release_evidence(root) == {"valid": True, "file_count": 5, "failures": [], "unexpected_files": []}
    (written / "result.json").write_text("{}", encoding="utf-8")
    assert verify_release_evidence(root)["valid"] is False


def test_state_backup_restore_requires_confirmation(tmp_path: Path) -> None:
    store = StateStore(state_dir=tmp_path / "state")
    store.repository.create_run("backup-run")
    backup = create_state_backup(store.state_dir, tmp_path / "backups")
    with pytest.raises(ValueError, match="confirmation"):
        restore_state_backup(backup, store.state_dir)
    restored = restore_state_backup(backup, store.state_dir, confirm=True)
    assert restored.is_file()
