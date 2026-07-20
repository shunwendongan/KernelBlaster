from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "sync_portfolio_docs", ROOT / "scripts" / "sync_portfolio_docs.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _copy_documentation_root(tmp_path: Path) -> Path:
    for name in ("README.md", "README.zh-CN.md"):
        shutil.copy2(ROOT / name, tmp_path / name)
    shutil.copytree(ROOT / "docs", tmp_path / "docs")
    shutil.copytree(ROOT / "artifacts", tmp_path / "artifacts")
    (tmp_path / "portfolio").mkdir()
    shutil.copy2(ROOT / "portfolio" / "status.json", tmp_path / "portfolio" / "status.json")
    shutil.copytree(
        ROOT / "portfolio" / "case_studies",
        tmp_path / "portfolio" / "case_studies",
    )
    return tmp_path


def test_context_derives_published_nine_and_ten_task_metrics():
    context = MODULE.load_context(ROOT)
    assert context["new9"]["attempted_upstream"] == pytest.approx(5.0199631675)
    assert context["new9"]["strict_upstream"] == pytest.approx(3.3017281925)
    assert context["new9"]["attempted_pytorch"] == pytest.approx(1.4154586790)
    assert context["new9"]["strict_pytorch"] == pytest.approx(0.9309749235)
    assert context["new9"]["verified"] == 3
    assert context["all10"]["verified"] == 4


def test_sync_is_idempotent(tmp_path):
    root = _copy_documentation_root(tmp_path)
    path = root / "README.md"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "This fork has completed", "STALE: This fork has completed", 1
        ),
        encoding="utf-8",
    )
    changed = MODULE.synchronize(root=root, write=True)
    assert Path("README.md") in changed
    assert MODULE.synchronize(root=root, write=True) == []
    assert MODULE.synchronize(root=root, write=False) == []


def test_generated_status_blocks_are_localized_and_evidence_is_labeled():
    context = MODULE.load_context(ROOT)
    chinese = MODULE._root_block(context, chinese=True)
    english = MODULE._root_block(context, chinese=False)

    assert "52 项通过" in chinese
    assert "10/10 个候选通过" in chinese
    assert "阻塞：HTTP 401 invalid_api_key；不重试" in chinese
    assert "未运行（Day 11–14 不在本阶段范围）" in chinese
    assert "10/10 candidates passed" not in chinese
    assert "Full Chinese report" in english
    assert "English full report" not in english


def test_replace_block_rejects_missing_markers():
    with pytest.raises(MODULE.DocumentationSyncError, match="exactly one"):
        MODULE.replace_block("no markers", "STATUS", "body")


def test_replace_block_rejects_duplicate_markers():
    text = (
        "<!-- STATUS:START -->\na\n<!-- STATUS:END -->\n"
        "<!-- STATUS:START -->\nb\n<!-- STATUS:END -->\n"
    )
    with pytest.raises(MODULE.DocumentationSyncError, match="exactly one"):
        MODULE.replace_block(text, "STATUS", "body")


def test_invalid_comparison_schema_fails_loudly(tmp_path):
    root = _copy_documentation_root(tmp_path)
    comparison = root / "artifacts/portfolio-v1.0/results/core10_rtx3080_comparison.json"
    payload = json.loads(comparison.read_text(encoding="utf-8"))
    payload["schema_version"] = "broken"
    comparison.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MODULE.DocumentationSyncError, match="schema_version"):
        MODULE.load_context(root)


def test_missing_status_source_fails_loudly(tmp_path):
    root = _copy_documentation_root(tmp_path)
    status_path = root / "portfolio/status.json"
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    payload["sources"]["comparison"] = "artifacts/missing.json"
    status_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MODULE.DocumentationSyncError, match="does not exist"):
        MODULE.load_context(root)


def test_artifact_hash_mismatch_is_rejected(tmp_path):
    root = _copy_documentation_root(tmp_path)
    context = MODULE.load_context(root)
    readme = root / "artifacts/portfolio-v1.0/README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(MODULE.DocumentationSyncError, match="SHA256"):
        MODULE.validate_artifact_hashes(root, context)


def test_change_policy_requires_documentation_for_results():
    with pytest.raises(MODULE.DocumentationSyncError, match="require README/docs"):
        MODULE.validate_change_policy(["portfolio/case_studies/core10/new.cu"])
    MODULE.validate_change_policy(
        ["portfolio/case_studies/core10/new.cu", "docs/portfolio/validation.md"]
    )


def test_broken_relative_markdown_link_is_rejected(tmp_path):
    with pytest.raises(MODULE.DocumentationSyncError, match="Broken link"):
        MODULE.validate_links(
            tmp_path,
            {Path("README.md"): "[missing](docs/not-there.md)"},
        )


@pytest.mark.parametrize(
    "machine_path",
    (
        "/home/example/src/KernelBlaster/out/results.json",
        "C:\\Users\\example\\KernelBlaster\\out\\results.json",
        "\\\\" + "wsl.localhost\\Ubuntu\\home\\example\\KernelBlaster",
    ),
)
def test_machine_specific_absolute_path_is_rejected(tmp_path, machine_path):
    with pytest.raises(MODULE.DocumentationSyncError, match="Machine-specific"):
        MODULE.validate_links(tmp_path, {Path("README.md"): machine_path})
