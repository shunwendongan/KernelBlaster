#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))

from src.kernelblaster.benchmarking import (  # noqa: E402
    BENCHMARK_MARKER,
    BENCHMARK_SCHEMA_VERSION,
    comparison_validity,
    instrument_profiler_driver,
    instrument_driver,
    latency_summary,
    ncu_metric_names,
    normalize_cuda_source,
    session_spread_percent,
    sha256_text,
    write_compilation_units,
)


NCU_SECTIONS = (
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "Occupancy",
    "SchedulerStats",
    "LaunchStats",
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _default_output_dir() -> Path:
    return ROOT_DIR / "out" / "portfolio" / "benchmark" / _timestamp()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: float,
    log_path: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    log_path.write_text(
        "COMMAND\n"
        + json.dumps(command)
        + "\n\nSTDOUT\n"
        + completed.stdout
        + "\n\nSTDERR\n"
        + completed.stderr,
        encoding="utf-8",
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}; see {log_path.name}."
        )
    return completed


def _compile(
    source_dir: Path,
    *,
    sm: str,
    timeout: float,
) -> Path:
    template = ROOT_DIR / "src" / "kernelblaster" / "servers" / "cuda_env" / "CMakeLists.txt"
    shutil.copy2(template, source_dir / "CMakeLists.txt")
    build_dir = source_dir / "build"
    try:
        import torch
        from torch.utils import cmake_prefix_path
    except ImportError as error:
        raise RuntimeError(
            "benchmark_cuda.py must run in the KernelBlaster container with PyTorch."
        ) from error

    _run_command(
        [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DGPU_ARCH_VERSION={sm.removeprefix('sm_')}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ],
        cwd=source_dir,
        timeout=timeout,
        log_path=source_dir / "cmake-configure.log",
    )
    _run_command(
        ["cmake", "--build", str(build_dir), "--parallel"],
        cwd=source_dir,
        timeout=timeout,
        log_path=source_dir / "cmake-build.log",
    )
    executable = build_dir / "main"
    if not executable.is_file():
        raise RuntimeError("CMake completed without producing build/main.")
    return executable


def _compile_program(
    root: Path,
    *,
    label: str,
    mode: str,
    driver: str,
    source: str,
    sm: str,
    timeout: float,
) -> Path:
    directory = root / "build" / label / mode
    write_compilation_units(directory, driver, source)
    return _compile(directory, sm=sm, timeout=timeout)


def _run_correctness(
    executable: Path,
    *,
    output_dir: Path,
    label: str,
    mode: str,
    timeout: float,
) -> dict[str, Any]:
    completed = _run_command(
        [str(executable)],
        cwd=executable.parent,
        timeout=timeout,
        log_path=output_dir / f"correctness-{label}-{mode}.log",
        check=False,
    )
    passed = completed.returncode == 0 and "passed" in completed.stdout.lower()
    return {
        "mode": mode,
        "returncode": completed.returncode,
        "passed": passed,
    }


def _gpu_telemetry() -> dict[str, Any]:
    fields = (
        "name,uuid,driver_version,temperature.gpu,power.draw,"
        "clocks.sm,clocks.mem,utilization.gpu,memory.total"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"available": False, "error_type": type(error).__name__}
    values = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
    names = fields.split(",")
    return {"available": True, **dict(zip(names, values, strict=False))}


def _parse_benchmark_output(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.startswith(BENCHMARK_MARKER)]
    if len(lines) != 1:
        raise RuntimeError("Benchmark output did not contain exactly one result marker.")
    return json.loads(lines[0][len(BENCHMARK_MARKER) :])


def _run_ncu(
    executable: Path,
    *,
    output_dir: Path,
    label: str,
    timeout: float,
) -> dict[str, Any]:
    report_base = output_dir / f"ncu-{label}"
    command = ["ncu"]
    for section in NCU_SECTIONS:
        command.extend(["--section", section])
    command.extend(
        [
            "--profile-from-start",
            "off",
            "--export",
            str(report_base),
            "--force-overwrite",
            str(executable),
        ]
    )
    try:
        completed = _run_command(
            command,
            cwd=executable.parent,
            timeout=timeout,
            log_path=output_dir / f"ncu-{label}.log",
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "failed", "error_type": type(error).__name__}
    if completed.returncode != 0:
        error_name = (
            "ERR_NVGPUCTRPERM"
            if "ERR_NVGPUCTRPERM" in completed.stderr + completed.stdout
            else "NCUCommandFailed"
        )
        return {
            "status": "failed",
            "error_type": error_name,
            "returncode": completed.returncode,
        }

    report_path = report_base.with_suffix(".ncu-rep")
    csv_result = _run_command(
        ["ncu", "--import", str(report_path), "--csv", "--page", "raw"],
        cwd=executable.parent,
        timeout=timeout,
        log_path=output_dir / f"ncu-{label}-export.log",
        check=False,
    )
    csv_path = output_dir / f"ncu-{label}.csv"
    csv_path.write_text(csv_result.stdout, encoding="utf-8")
    metric_names = ncu_metric_names(csv_result.stdout)
    _atomic_json(
        output_dir / f"ncu-{label}-metrics.json",
        {"sections": list(NCU_SECTIONS), "metric_names": metric_names},
    )
    if csv_result.returncode != 0 or not metric_names:
        return {
            "status": "failed",
            "error_type": "NCUCSVExportFailed",
            "returncode": csv_result.returncode,
            "report": report_path.name,
            "csv": csv_path.name,
        }
    return {
        "status": "completed",
        "report": report_path.name,
        "csv": csv_path.name,
        "sections": list(NCU_SECTIONS),
        "metric_names": metric_names,
    }


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() or None


def _version(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return {"status": "unavailable", "error_type": type(error).__name__}
    return {
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "output": (completed.stdout + completed.stderr).strip()[:10_000],
    }


def _summarize_records(
    records: list[dict[str, Any]], labels: list[str]
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for label in labels:
        selected = [record for record in records if record["variant"] == label]
        all_samples = [
            sample for record in selected for sample in record["samples_us"]
        ]
        session_medians = [record["latency"]["median_us"] for record in selected]
        summaries[label] = {
            "all_samples": latency_summary(all_samples),
            "session_medians": session_medians,
            "session_medians_summary": latency_summary(session_medians),
            "session_spread_percent": session_spread_percent(session_medians),
        }
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correctness-first CUDA Events benchmark for KernelBench-CUDA."
    )
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--candidate", type=Path, default=None)
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument(
        "--extra-correctness-driver",
        type=Path,
        action="append",
        default=[],
        help="Additional Driver with the same launcher signature; may be repeated.",
    )
    parser.add_argument("--gpu", default="RTX 3080")
    parser.add_argument("--sm", default="sm_86")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--shape", default="driver-defined")
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument(
        "--inner-loops",
        type=int,
        default=0,
        help="Zero enables calibration to approximately one millisecond per sample.",
    )
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--max-session-spread-percent", type=float, default=5.0)
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=60.0,
        help="Cooldown before the single automatic retest of unstable sessions.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=1200)
    parser.add_argument("--ncu", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    args = parser.parse_args()

    task_dir = args.task_dir.resolve()
    driver_path = task_dir / "driver.cpp"
    upstream_baseline_path = (task_dir / "init.cu").resolve()
    baseline_path = (args.baseline or upstream_baseline_path).resolve()
    candidate_path = args.candidate.resolve() if args.candidate else None
    for path in (driver_path, baseline_path):
        if not path.is_file():
            parser.error(f"Required input does not exist: {path}")
    if candidate_path is not None and not candidate_path.is_file():
        parser.error(f"Candidate does not exist: {candidate_path}")
    extra_driver_paths = [path.resolve() for path in args.extra_correctness_driver]
    for path in extra_driver_paths:
        if not path.is_file():
            parser.error(f"Extra correctness Driver does not exist: {path}")
    if min(args.warmup, args.repetitions, args.sessions) < 1:
        parser.error("Warmup, repetitions, and sessions must be positive.")
    if args.max_session_spread_percent <= 0 or args.cooldown_seconds < 0:
        parser.error("Session spread must be positive and cooldown non-negative.")

    output_dir = args.output_dir.resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Refusing to overwrite output directory: {output_dir}")

    driver = driver_path.read_text(encoding="utf-8")
    instrumented_driver = instrument_driver(
        driver,
        seed=args.seed,
        warmup=args.warmup,
        repetitions=args.repetitions,
        inner_loops=args.inner_loops,
    )
    profiler_driver = instrument_profiler_driver(driver)
    variant_paths = {"baseline": baseline_path}
    if candidate_path is not None:
        variant_paths[args.candidate_name] = candidate_path

    manifest: dict[str, Any] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "run_id": output_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_id": args.task_id,
        "kernel": args.kernel,
        "git_commit": _git_commit(),
        "gpu": args.gpu,
        "sm": args.sm,
        "dtype": args.dtype,
        "shape": args.shape,
        "seed": args.seed,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "sessions": args.sessions,
        "max_session_spread_percent": args.max_session_spread_percent,
        "cooldown_seconds": args.cooldown_seconds,
        "container_image": os.getenv("KERNELBLASTER_CONTAINER_IMAGE"),
        "container_digest": os.getenv("KERNELBLASTER_CONTAINER_DIGEST"),
        "driver_sha256": sha256_text(driver),
        "extra_correctness_drivers": [
            {"name": path.name, "sha256": sha256_text(path.read_text(encoding="utf-8"))}
            for path in extra_driver_paths
        ],
        "environment": {
            "nvidia_smi": _version(["nvidia-smi"]),
            "nvcc": _version(["nvcc", "--version"]),
            "ncu": _version(["ncu", "--version"]),
        },
        "variants": {},
        "baseline_scope": (
            "upstream_baseline"
            if baseline_path == upstream_baseline_path
            else "variant_head_to_head"
        ),
    }

    executables: dict[str, Path] = {}
    for label, source_path in variant_paths.items():
        original = source_path.read_text(encoding="utf-8")
        normalized, transformations = normalize_cuda_source(original)
        correctness_results: list[dict[str, Any]] = []
        normalized_executable: Path | None = None
        correctness_drivers = [("official", driver)] + [
            (f"extra-{index}-{path.stem}", path.read_text(encoding="utf-8"))
            for index, path in enumerate(extra_driver_paths)
        ]
        for driver_label, correctness_driver in correctness_drivers:
            for source_mode, correctness_source in (
                ("original", original),
                ("normalized", normalized),
            ):
                executable = _compile_program(
                    output_dir,
                    label=label,
                    mode=f"{source_mode}-correctness-{driver_label}",
                    driver=correctness_driver,
                    source=correctness_source,
                    sm=args.sm,
                    timeout=args.timeout_seconds,
                )
                result = _run_correctness(
                    executable,
                    output_dir=output_dir,
                    label=label,
                    mode=f"{source_mode}-{driver_label}",
                    timeout=args.timeout_seconds,
                )
                result["driver"] = driver_label
                correctness_results.append(result)
                if source_mode == "normalized" and driver_label == "official":
                    normalized_executable = executable
        if not all(result["passed"] for result in correctness_results):
            raise RuntimeError(f"Correctness failed for {label}; timing was not started.")
        if normalized_executable is None:
            raise RuntimeError("Official normalized correctness executable is missing.")

        benchmark_executable = _compile_program(
            output_dir,
            label=label,
            mode="benchmark",
            driver=instrumented_driver,
            source=normalized,
            sm=args.sm,
            timeout=args.timeout_seconds,
        )
        executables[label] = benchmark_executable
        manifest["variants"][label] = {
            "source_name": source_path.name,
            "source_sha256": sha256_text(original),
            "normalized_sha256": sha256_text(normalized),
            "transformations": transformations,
            "correctness": correctness_results,
        }
        if args.ncu:
            profiler_executable = _compile_program(
                output_dir,
                label=label,
                mode="ncu-profile",
                driver=profiler_driver,
                source=normalized,
                sm=args.sm,
                timeout=args.timeout_seconds,
            )
            manifest["variants"][label]["ncu"] = _run_ncu(
                profiler_executable,
                output_dir=output_dir,
                label=label,
                timeout=args.timeout_seconds,
            )
        _atomic_json(output_dir / "run_manifest.json", manifest)

    records: list[dict[str, Any]] = []
    jsonl_path = output_dir / "measurements.jsonl"
    labels = list(executables)
    round_summaries: list[dict[str, Any]] = []

    def run_round(round_index: int) -> dict[str, Any]:
        round_records: list[dict[str, Any]] = []
        for session in range(args.sessions):
            order = labels if session % 2 == 0 else list(reversed(labels))
            for position, label in enumerate(order):
                before = _gpu_telemetry()
                completed = _run_command(
                    [str(executables[label])],
                    cwd=executables[label].parent,
                    timeout=args.timeout_seconds,
                    log_path=output_dir
                    / f"round-{round_index}-session-{session}-{position}-{label}.log",
                    check=False,
                )
                if (
                    completed.returncode != 0
                    or "passed" not in completed.stdout.lower()
                ):
                    raise RuntimeError(
                        f"Benchmark execution failed correctness for {label}."
                    )
                parsed = _parse_benchmark_output(completed.stdout)
                samples = [float(value) for value in parsed["samples_us"]]
                after = _gpu_telemetry()
                record = {
                    "schema_version": BENCHMARK_SCHEMA_VERSION,
                    "run_id": output_dir.name,
                    "task_id": args.task_id,
                    "kernel": args.kernel,
                    "variant": label,
                    "round": round_index,
                    "session": session,
                    "order": "AB" if order == labels else "BA",
                    "order_position": position,
                    "seed": args.seed,
                    "warmup": args.warmup,
                    "repetitions": args.repetitions,
                    "inner_loops": int(parsed["inner_loops"]),
                    "correct": True,
                    "samples_us": samples,
                    "latency": latency_summary(samples),
                    "telemetry_before": before,
                    "telemetry_after": after,
                    "source_sha256": manifest["variants"][label]["source_sha256"],
                    "normalized_sha256": manifest["variants"][label][
                        "normalized_sha256"
                    ],
                    "driver_sha256": manifest["driver_sha256"],
                    "git_commit": manifest["git_commit"],
                    "container_digest": manifest["container_digest"],
                    "gpu": args.gpu,
                    "sm": args.sm,
                    "dtype": args.dtype,
                    "shape": args.shape,
                }
                records.append(record)
                round_records.append(record)
                with jsonl_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
        return _summarize_records(round_records, labels)

    summaries = run_round(0)
    round_summaries.append(summaries)
    unstable = any(
        summary["session_spread_percent"] > args.max_session_spread_percent
        for summary in summaries.values()
    )
    if unstable:
        if args.cooldown_seconds:
            time.sleep(args.cooldown_seconds)
        summaries = run_round(1)
        round_summaries.append(summaries)

    comparison = None
    if len(labels) == 2:
        baseline_median = summaries["baseline"]["session_medians_summary"][
            "median_us"
        ]
        candidate_label = next(label for label in labels if label != "baseline")
        candidate_median = summaries[candidate_label]["session_medians_summary"][
            "median_us"
        ]
        session_speedups = [
            baseline / candidate
            for baseline, candidate in zip(
                summaries["baseline"]["session_medians"],
                summaries[candidate_label]["session_medians"],
                strict=True,
            )
        ]
        comparison = {
            "candidate": candidate_label,
            "baseline_median_us": baseline_median,
            "candidate_median_us": candidate_median,
            "speedup": baseline_median / candidate_median,
            "session_speedups": session_speedups,
            "all_sessions_not_slower": all(value >= 1.0 for value in session_speedups),
            "comparison_scope": manifest["baseline_scope"],
        }
        comparison.update(
            comparison_validity(
                baseline_source_sha256=manifest["variants"]["baseline"][
                    "source_sha256"
                ],
                candidate_source_sha256=manifest["variants"][candidate_label][
                    "source_sha256"
                ],
                baseline_session_medians=summaries["baseline"]["session_medians"],
                candidate_session_medians=summaries[candidate_label][
                    "session_medians"
                ],
                speedup=comparison["speedup"],
                max_session_spread_percent=args.max_session_spread_percent,
            )
        )
        with (output_dir / "comparison.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            csv_comparison = {
                key: json.dumps(value) if isinstance(value, list) else value
                for key, value in comparison.items()
            }
            writer = csv.DictWriter(stream, fieldnames=list(csv_comparison))
            writer.writeheader()
            writer.writerow(csv_comparison)

    stable = all(
        summary["session_spread_percent"] <= args.max_session_spread_percent
        for summary in summaries.values()
    )
    ncu_attribution_valid = None
    if args.ncu:
        ncu_attribution_valid = all(
            variant.get("ncu", {}).get("status") == "completed"
            and bool(variant.get("ncu", {}).get("metric_names"))
            for variant in manifest["variants"].values()
        )
    performance_claim_allowed = stable and (
        comparison is None or comparison["formal_valid"]
    )

    summary = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "run_id": output_dir.name,
        "task_id": args.task_id,
        "kernel": args.kernel,
        "summaries": summaries,
        "round_summaries": round_summaries,
        "retest_triggered": len(round_summaries) == 2,
        "active_round": len(round_summaries) - 1,
        "stable": stable,
        "comparison": comparison,
        "ncu_attribution_valid": ncu_attribution_valid,
        "performance_claim_allowed": performance_claim_allowed,
    }
    _atomic_json(output_dir / "summary.json", summary)
    _atomic_json(output_dir / "run_manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.ncu and not ncu_attribution_valid:
        return 4
    if not performance_claim_allowed:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
