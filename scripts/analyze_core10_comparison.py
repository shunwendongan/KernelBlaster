#!/usr/bin/env python3
"""Join formal CUDA candidate and same-GPU PyTorch Core 10 results."""
from __future__ import annotations

import argparse
import csv
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
from typing import Any


TASK_IDS = ("004", "007", "019", "023", "026", "036", "040", "047", "088", "095")
MIN_MATERIAL_SPEEDUP = 1.01


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("Geometric mean requires positive values.")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _eager_method(task_id: str) -> str:
    return "pytorch_driver_formula" if task_id == "088" else "pytorch_eager"


def augment_candidate_details(
    candidate_payload: dict[str, Any], base_dir: Path
) -> None:
    """Join p10/p90/session medians from each append-only task summary."""
    for row in candidate_payload.get("results", []):
        relative = row.get("summary")
        if not relative:
            continue
        path = (base_dir / relative).resolve()
        detail = _load(path)
        variants = detail.get("summaries", {})
        for prefix, variant in (("baseline", "baseline"), ("candidate", row["candidate"])):
            selected = variants.get(variant)
            if not selected:
                continue
            row[f"{prefix}_p10_us"] = selected["all_samples"]["p10_us"]
            row[f"{prefix}_p90_us"] = selected["all_samples"]["p90_us"]
            row[f"{prefix}_session_medians_us"] = selected["session_medians"]


def build_comparison_rows(
    candidate_payload: dict[str, Any], pytorch_payload: dict[str, Any]
) -> list[dict[str, Any]]:
    candidate_rows = {
        str(row["task_id"]): row for row in candidate_payload.get("results", [])
    }
    pytorch_rows: dict[str, list[dict[str, Any]]] = {}
    for row in pytorch_payload.get("results", []):
        pytorch_rows.setdefault(str(row["task_id"]), []).append(row)
    missing_candidates = sorted(set(TASK_IDS) - set(candidate_rows))
    missing_pytorch = sorted(set(TASK_IDS) - set(pytorch_rows))
    if missing_candidates or missing_pytorch:
        raise ValueError(
            f"Incomplete Core 10 inputs; candidate={missing_candidates}, pytorch={missing_pytorch}"
        )

    rows: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        candidate = candidate_rows[task_id]
        torch_methods = pytorch_rows[task_id]
        eager_name = _eager_method(task_id)
        eager = next((row for row in torch_methods if row["method"] == eager_name), None)
        if eager is None:
            raise ValueError(f"Task {task_id} is missing {eager_name}.")
        correct_methods = [row for row in torch_methods if row.get("correct")]
        if not correct_methods:
            raise ValueError(f"Task {task_id} has no correct PyTorch method.")
        best = min(correct_methods, key=lambda row: float(row["median_us"]))

        baseline_us = float(candidate["baseline_median_us"])
        candidate_us = float(candidate["candidate_median_us"])
        attempted_speedup = float(candidate["speedup"])
        verified_improvement = bool(
            candidate.get("correct")
            and candidate.get("stable")
            and candidate.get("performance_claim_allowed")
            and candidate.get("all_sessions_not_slower")
            and attempted_speedup >= MIN_MATERIAL_SPEEDUP
        )
        selected_us = candidate_us if verified_improvement else baseline_us
        selected_variant = candidate["candidate"] if verified_improvement else "upstream_baseline"
        eager_us = float(eager["median_us"])
        best_us = float(best["median_us"])
        rows.append(
            {
                "task_id": task_id,
                "kernel": candidate["kernel"],
                "candidate": candidate["candidate"],
                "source": candidate["source"],
                "correct": bool(candidate.get("correct")),
                "stable": bool(candidate.get("stable")),
                "all_sessions_not_slower": bool(candidate.get("all_sessions_not_slower")),
                "baseline_median_us": baseline_us,
                "baseline_p10_us": candidate.get("baseline_p10_us"),
                "baseline_p90_us": candidate.get("baseline_p90_us"),
                "baseline_session_medians_us": candidate.get(
                    "baseline_session_medians_us"
                ),
                "candidate_median_us": candidate_us,
                "candidate_p10_us": candidate.get("candidate_p10_us"),
                "candidate_p90_us": candidate.get("candidate_p90_us"),
                "candidate_session_medians_us": candidate.get(
                    "candidate_session_medians_us"
                ),
                "attempted_speedup": attempted_speedup,
                "session_speedups": candidate.get("session_speedups"),
                "verified_improvement": verified_improvement,
                "selected_variant": selected_variant,
                "selected_median_us": selected_us,
                "portfolio_speedup": baseline_us / selected_us,
                "pytorch_eager_method": eager_name,
                "pytorch_eager_median_us": eager_us,
                "pytorch_eager_stable": bool(eager.get("stable")),
                "pytorch_eager_p10_us": eager.get("p10_us"),
                "pytorch_eager_p90_us": eager.get("p90_us"),
                "candidate_vs_pytorch_eager": eager_us / candidate_us,
                "selected_vs_pytorch_eager": eager_us / selected_us,
                "pytorch_best_method": best["method"],
                "pytorch_best_allocation_mode": best["allocation_mode"],
                "pytorch_best_median_us": best_us,
                "pytorch_best_stable": bool(best.get("stable")),
                "pytorch_best_p10_us": best.get("p10_us"),
                "pytorch_best_p90_us": best.get("p90_us"),
                "candidate_vs_pytorch_best": best_us / candidate_us,
                "selected_vs_pytorch_best": best_us / selected_us,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = [float(row["attempted_speedup"]) for row in rows]
    selected = [float(row["portfolio_speedup"]) for row in rows]
    library = [float(row["selected_vs_pytorch_best"]) for row in rows]
    attempted_library = [float(row["candidate_vs_pytorch_best"]) for row in rows]
    return {
        "denominator": len(rows),
        "material_speedup_threshold": MIN_MATERIAL_SPEEDUP,
        "verified_improved_tasks": sum(row["verified_improvement"] for row in rows),
        "correct_tasks": sum(row["correct"] for row in rows),
        "stable_tasks": sum(row["stable"] for row in rows),
        "attempted_candidate_geomean_speedup": _geomean(attempted),
        "all10_selected_portfolio_geomean_speedup": _geomean(selected),
        "selected_vs_pytorch_best_geomean": _geomean(library),
        "attempted_candidate_vs_pytorch_best_geomean": _geomean(
            attempted_library
        ),
        "custom_faster_than_pytorch_best_tasks": sum(value > 1.0 for value in library),
        "pytorch_best_faster_tasks": sum(value < 1.0 for value in library),
        "attempted_candidate_faster_than_pytorch_best_tasks": sum(
            value > 1.0 for value in attempted_library
        ),
        "pytorch_best_stable_tasks": sum(row["pytorch_best_stable"] for row in rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    serializable = [
        {
            key: json.dumps(value, separators=(",", ":"))
            if isinstance(value, (list, dict))
            else value
            for key, value in row.items()
        }
        for row in rows
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(serializable[0]))
        writer.writeheader()
        writer.writerows(serializable)


def render_svg(rows: list[dict[str, Any]], path: Path) -> None:
    width, left, plot_width = 1120, 250, 650
    top, row_height = 82, 48
    height = top + row_height * len(rows) + 70
    low, high = -2.0, 8.0

    def x_for(ratio: float) -> float:
        value = min(high, max(low, math.log2(max(ratio, 2**low))))
        return left + (value - low) / (high - low) * plot_width

    one_x = x_for(1.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:22px;font-weight:700}.label{font-size:13px}.small{font-size:11px;fill:#526079}</style>',
        '<text x="24" y="32" class="title">Core 10 — upstream, selected custom, and same-RTX-3080 PyTorch</text>',
        '<text x="24" y="55" class="small">Log2 ratio; right of 1.0 means the custom result is faster. Blue: upstream/candidate attempt. Green/orange: selected/PyTorch best.</text>',
    ]
    for tick in (-2, 0, 2, 4, 6, 8):
        x = left + (tick - low) / (high - low) * plot_width
        parts.append(f'<line x1="{x:.1f}" y1="66" x2="{x:.1f}" y2="{height - 35}" stroke="#d7dce5" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 14}" text-anchor="middle" class="small">{2**tick:g}×</text>')
    parts.append(f'<line x1="{one_x:.1f}" y1="66" x2="{one_x:.1f}" y2="{height - 35}" stroke="#172033" stroke-width="2"/>')
    for index, row in enumerate(rows):
        y = top + index * row_height
        attempted = float(row["attempted_speedup"])
        library = float(row["selected_vs_pytorch_best"])
        attempted_x = x_for(attempted)
        library_x = x_for(library)
        parts.append(f'<text x="24" y="{y + 18}" class="label">{escape(row["task_id"] + " " + row["kernel"])}</text>')
        parts.append(f'<line x1="{min(one_x, attempted_x):.1f}" y1="{y + 10}" x2="{max(one_x, attempted_x):.1f}" y2="{y + 10}" stroke="#3578d4" stroke-width="8"/>')
        color = "#248a5b" if library >= 1.0 else "#d07428"
        parts.append(f'<line x1="{min(one_x, library_x):.1f}" y1="{y + 28}" x2="{max(one_x, library_x):.1f}" y2="{y + 28}" stroke="{color}" stroke-width="8"/>')
        parts.append(f'<text x="920" y="{y + 14}" class="small">candidate {attempted:.3f}×</text>')
        parts.append(f'<text x="920" y="{y + 32}" class="small">vs PyTorch {library:.3f}×</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--pytorch-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    candidate_payload = _load(args.candidate_summary.resolve())
    pytorch_payload = _load(args.pytorch_summary.resolve())
    augment_candidate_details(candidate_payload, args.candidate_summary.resolve().parent)
    rows = build_comparison_rows(candidate_payload, pytorch_payload)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_csv = output_dir / "core10_rtx3080_comparison.csv"
    comparison_json = output_dir / "core10_rtx3080_comparison.json"
    figure = output_dir / "core10_rtx3080_comparison.svg"
    _write_csv(comparison_csv, rows)
    _atomic_json(
        comparison_json,
        {
            "schema_version": "1.0",
            "hardware": "NVIDIA GeForce RTX 3080 (sm_86)",
            "candidate_protocol": candidate_payload.get("protocol"),
            "pytorch_protocol": pytorch_payload.get("protocol"),
            "summary": summarize(rows),
            "results": rows,
        },
    )
    render_svg(rows, figure)
    files = [comparison_csv, comparison_json, figure]
    _atomic_json(
        output_dir / "core10_rtx3080_SHA256SUMS.json",
        {path.name: _sha256(path) for path in files},
    )
    print(json.dumps(summarize(rows), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
