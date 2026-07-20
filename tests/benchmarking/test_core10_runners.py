from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PYTORCH = _load("benchmark_pytorch", "benchmark_pytorch.py")
CANDIDATES = _load("benchmark_candidates", "benchmark_candidates.py")
ANALYZE = _load("analyze_core10_comparison", "analyze_core10_comparison.py")


def test_pytorch_matrix_covers_every_core10_task():
    expected = {"004", "007", "019", "023", "026", "036", "040", "047", "088", "095"}
    assert set(PYTORCH.CORE10_TASKS) == expected
    assert set(PYTORCH.PYTORCH_METHODS) == expected
    assert all(PYTORCH.PYTORCH_METHODS[task_id] for task_id in expected)
    assert "pytorch_fused_gelu_tanh" in PYTORCH.PYTORCH_METHODS["088"]
    assert "pytorch_preallocated_out" in PYTORCH.PYTORCH_METHODS["019"]


def test_candidate_manifest_resolves_relative_sources(tmp_path):
    source = tmp_path / "candidate.cu"
    driver = tmp_path / "edge.cpp"
    source.write_text("// candidate\n", encoding="utf-8")
    driver.write_text("// driver\n", encoding="utf-8")
    manifest = tmp_path / "candidates.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "tasks": [
                    {
                        "id": "004",
                        "name": "candidate",
                        "source": source.name,
                        "extra_correctness_drivers": [driver.name],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    loaded = CANDIDATES.load_candidates(manifest)
    assert loaded["004"]["source_path"] == source.resolve()
    assert loaded["004"]["extra_driver_paths"] == [driver.resolve()]


def test_candidate_manifest_rejects_duplicate_ids(tmp_path):
    source = tmp_path / "candidate.cu"
    source.write_text("// candidate\n", encoding="utf-8")
    manifest = tmp_path / "candidates.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "tasks": [
                    {"id": "004", "name": "a", "source": source.name},
                    {"id": "004", "name": "b", "source": source.name},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        CANDIDATES.load_candidates(manifest)


def test_comparison_selects_only_verified_improvements():
    candidate_results = []
    pytorch_results = []
    for task_id in ANALYZE.TASK_IDS:
        improved = task_id != "088"
        candidate_results.append(
            {
                "task_id": task_id,
                "kernel": f"kernel-{task_id}",
                "candidate": f"candidate-{task_id}",
                "source": f"{task_id}.cu",
                "correct": True,
                "stable": True,
                "performance_claim_allowed": True,
                "all_sessions_not_slower": improved,
                "baseline_median_us": 20.0,
                "candidate_median_us": 10.0 if improved else 21.0,
                "speedup": 2.0 if improved else 20.0 / 21.0,
                "session_speedups": [2.0, 2.0, 2.0] if improved else [0.95] * 3,
            }
        )
        eager_method = (
            "pytorch_driver_formula" if task_id == "088" else "pytorch_eager"
        )
        pytorch_results.append(
            {
                "task_id": task_id,
                "method": eager_method,
                "allocation_mode": "framework_allocated",
                "median_us": 15.0,
                "correct": True,
                "stable": True,
            }
        )
        if task_id == "088":
            pytorch_results.append(
                {
                    "task_id": task_id,
                    "method": "pytorch_fused_gelu_tanh",
                    "allocation_mode": "framework_allocated_equivalent",
                    "median_us": 8.0,
                    "correct": True,
                    "stable": True,
                }
            )
    rows = ANALYZE.build_comparison_rows(
        {"results": candidate_results}, {"results": pytorch_results}
    )
    assert len(rows) == 10
    failed = next(row for row in rows if row["task_id"] == "088")
    assert failed["selected_variant"] == "upstream_baseline"
    assert failed["portfolio_speedup"] == 1.0
    assert failed["pytorch_best_method"] == "pytorch_fused_gelu_tanh"
    assert ANALYZE.summarize(rows)["verified_improved_tasks"] == 9
