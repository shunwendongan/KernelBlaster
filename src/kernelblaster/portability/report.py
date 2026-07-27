"""Hardware-aware aggregate reports for independently imported runs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..storage.repository import JobRepository
from .contracts import AGGREGATE_REPORT_SCHEMA, canonical_bytes, sha256


def build_aggregate_report(repository: JobRepository) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in repository.list_runs():
        portability = repository.get_run_portability(str(run["id"])) or {}
        group = str(portability.get("comparison_group") or "legacy_unknown")
        groups.setdefault(group, []).append(
            {
                "run_id": run["id"],
                "status": run["status"],
                "created_at": run["created_at"],
                "instance_id": portability.get("source_instance_id") or "legacy_unknown",
                "target_id": portability.get("target_id") or "legacy_unknown",
                "target_arch": portability.get("target_arch") or "unknown",
                "audit_fingerprint": portability.get("audit_fingerprint") or "legacy_unknown",
            }
        )
    for items in groups.values():
        items.sort(key=lambda item: (item["created_at"], item["run_id"]))
    report = {
        "schema_version": AGGREGATE_REPORT_SCHEMA,
        "comparison_groups": [
            {"comparison_group": group, "runs": runs, "performance_ranking": "incomparable_across_groups"}
            for group, runs in sorted(groups.items())
        ],
        "correctness_portability": [
            {"run_id": item["run_id"], "comparison_group": group, "status": item["status"]}
            for group, runs in sorted(groups.items())
            for item in runs
        ],
    }
    report["report_hash"] = sha256(report)
    return report


def write_aggregate_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "aggregate-report.json"
    csv_path = root / "aggregate-runs.csv"
    english_path = root / "aggregate-report.en.md"
    chinese_path = root / "aggregate-report.zh-CN.md"
    json_path.write_bytes(canonical_bytes(report) + b"\n")
    rows = [item for group in report["comparison_groups"] for item in group["runs"]]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run_id", "status", "created_at", "instance_id", "target_id", "target_arch", "audit_fingerprint"])
        writer.writeheader()
        writer.writerows(rows)
    summary = f"# Aggregate report\n\nComparison groups: {len(report['comparison_groups'])}\n\nPerformance is never ranked across comparison groups.\n"
    english_path.write_text(summary, encoding="utf-8")
    chinese_path.write_text("# 聚合报告\n\n比较组数量：" + str(len(report["comparison_groups"])) + "\n\n性能不会跨比较组自动排名。\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "english": english_path, "chinese": chinese_path}
