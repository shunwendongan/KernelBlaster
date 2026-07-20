#!/usr/bin/env python3
"""Run correctness-first CUDA Events comparisons for Core 10 candidates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from src.kernelblaster.portfolio import load_suite  # noqa: E402


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_candidates(path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0":
        raise ValueError("Candidate manifest must use schema_version 1.0.")
    candidates: dict[str, dict[str, Any]] = {}
    for item in manifest.get("tasks", []):
        task_id = str(item.get("id", ""))
        if not task_id or task_id in candidates:
            raise ValueError(f"Missing or duplicate candidate task ID: {task_id!r}")
        source = (path.parent / item["source"]).resolve()
        if not source.is_file():
            raise ValueError(f"Candidate source does not exist: {source}")
        extra_drivers = [
            (path.parent / driver).resolve()
            for driver in item.get("extra_correctness_drivers", [])
        ]
        if not all(driver.is_file() for driver in extra_drivers):
            raise ValueError(f"Candidate {task_id} has a missing correctness Driver.")
        candidates[task_id] = {
            **item,
            "source_path": source,
            "extra_driver_paths": extra_drivers,
        }
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Core 10 candidates against upstream init.cu."
    )
    parser.add_argument("--suite", default="core10")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT_DIR
        / "portfolio"
        / "case_studies"
        / "core10"
        / "candidates.json",
    )
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--cooldown-seconds", type=float, default=60.0)
    parser.add_argument("--max-session-spread-percent", type=float, default=5.0)
    parser.add_argument("--ncu", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "out" / "portfolio" / "candidates" / _timestamp(),
    )
    args = parser.parse_args()

    suite = load_suite(args.suite, ROOT_DIR)
    manifest_path = args.manifest.resolve()
    candidates = load_candidates(manifest_path)
    selected_ids = args.task_id or [task.task_id for task in suite.tasks]
    suite_ids = {task.task_id for task in suite.tasks}
    unknown = sorted(set(selected_ids) - suite_ids)
    missing = sorted(set(selected_ids) - set(candidates))
    if unknown:
        parser.error(f"Task IDs are not in suite {suite.name}: {unknown}")
    if missing:
        parser.error(f"Task IDs have no candidate in {manifest_path.name}: {missing}")

    output_dir = args.output_dir.resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Refusing to overwrite output directory: {output_dir}")

    task_map = {task.task_id: task for task in suite.tasks}
    rows: list[dict[str, Any]] = []
    for task_id in selected_ids:
        task = task_map[task_id]
        candidate = candidates[task_id]
        task_output = output_dir / task_id
        command = [
            sys.executable,
            str(SCRIPT_DIR / "benchmark_cuda.py"),
            "--task-dir",
            str((ROOT_DIR / task.path).resolve()),
            "--task-id",
            task_id,
            "--kernel",
            task.name,
            "--candidate",
            str(candidate["source_path"]),
            "--candidate-name",
            candidate["name"],
            "--warmup",
            str(args.warmup),
            "--repetitions",
            str(args.repetitions),
            "--sessions",
            str(args.sessions),
            "--cooldown-seconds",
            str(args.cooldown_seconds),
            "--max-session-spread-percent",
            str(args.max_session_spread_percent),
            "--output-dir",
            str(task_output),
        ]
        for extra_driver in candidate["extra_driver_paths"]:
            command.extend(["--extra-correctness-driver", str(extra_driver)])
        if args.ncu:
            command.append("--ncu")
        completed = subprocess.run(
            command,
            cwd=ROOT_DIR,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
        )
        log_path = output_dir / f"launcher-{task_id}.log"
        log_path.write_text(
            "COMMAND\n"
            + json.dumps(command)
            + "\n\nSTDOUT\n"
            + completed.stdout
            + "\n\nSTDERR\n"
            + completed.stderr,
            encoding="utf-8",
        )
        summary_path = task_output / "summary.json"
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.is_file()
            else None
        )
        comparison = summary.get("comparison") if summary else None
        rows.append(
            {
                "task_id": task_id,
                "kernel": task.name,
                "candidate": candidate["name"],
                "source": candidate["source"],
                "returncode": completed.returncode,
                "status": "completed" if completed.returncode == 0 else "failed",
                # benchmark_cuda emits summary only after every original and
                # normalized correctness executable has passed.
                "correct": bool(summary and comparison),
                "stable": summary.get("stable") if summary else None,
                "performance_claim_allowed": summary.get(
                    "performance_claim_allowed"
                )
                if summary
                else False,
                "baseline_median_us": comparison.get("baseline_median_us")
                if comparison
                else None,
                "candidate_median_us": comparison.get("candidate_median_us")
                if comparison
                else None,
                "speedup": comparison.get("speedup") if comparison else None,
                "session_speedups": comparison.get("session_speedups")
                if comparison
                else None,
                "all_sessions_not_slower": comparison.get(
                    "all_sessions_not_slower"
                )
                if comparison
                else None,
                "summary": str(summary_path.relative_to(output_dir))
                if summary
                else None,
                "log": log_path.name,
            }
        )
        _atomic_json(
            output_dir / "suite_summary.json",
            {
                "schema_version": "1.0",
                "suite": suite.name,
                "complete": False,
                "results": rows,
            },
        )

    payload = {
        "schema_version": "1.0",
        "suite": suite.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.relative_to(ROOT_DIR)),
        "protocol": {
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "sessions": args.sessions,
            "cooldown_seconds": args.cooldown_seconds,
            "max_session_spread_percent": args.max_session_spread_percent,
            "ncu": args.ncu,
        },
        "results": rows,
        "complete": len(rows) == len(selected_ids),
        "completed_tasks": sum(row["status"] == "completed" for row in rows),
        "failed_tasks": sum(row["status"] == "failed" for row in rows),
    }
    _atomic_json(output_dir / "suite_summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not payload["failed_tasks"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
