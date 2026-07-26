#!/usr/bin/env python3
"""Run fixed NSYS/NCU plans against a correctness-gated replay capsule."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from src.kernelblaster.candidate_packages import (
    SanitizerPlan,
    build_candidate_capsule,
    build_fixed_cuda_candidate,
    build_profiler_replay_capsule,
    validate_candidate_package,
)
from src.kernelblaster.candidate_packages.qualification import run_sanitizer
from src.kernelblaster.candidate_packages.archive import archive_files
from src.kernelblaster.harness import (
    CaseBundle,
    build_development_case_bundle,
    core10_task_specs,
)
from src.kernelblaster.profiler_jobs.contracts import ProfilePlanId, ProfileRequest
from src.kernelblaster.profiler_jobs.worker import (
    FixedToolRunner,
    _materialize_profiler_target,
    parse_profile_csv,
)


async def _run(target: Path, root: Path, plan: ProfilePlanId) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    request = ProfileRequest(
        artifact_digest="0" * 64,
        plan_id=plan,
        kernel_filter="kb019_relu",
        deadline=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    execution = await FixedToolRunner().run(
        request,
        target,
        "candidate-capsule-events-v1",
        root,
        600,
    )
    kernel, metrics, reason = parse_profile_csv(
        plan, execution.csv_output, request.kernel_filter
    )
    combined = execution.stdout + b"\n" + execution.stderr
    if b"ERR_NVGPUCTRPERM" in combined:
        reason_value = "permission_denied"
    elif execution.returncode != 0:
        reason_value = "execution_failed"
    else:
        reason_value = reason.value
    return {
        "plan_id": plan.value,
        "tool": execution.tool,
        "tool_version": execution.version,
        "returncode": execution.returncode,
        "reason_code": reason_value,
        "kernel": kernel,
        "metrics": [item.model_dump(mode="json") for item in metrics],
        "report_sha256": (
            hashlib.sha256(execution.report).hexdigest() if execution.report else None
        ),
        "csv_sha256": (
            hashlib.sha256(execution.csv_output).hexdigest()
            if execution.csv_output
            else None
        ),
        "nsys_timestamp_workaround": execution.used_timestamp_workaround,
        "diagnostic_tail": (
            combined.decode("utf-8", errors="replace")[-2000:]
            if reason_value != "none"
            else None
        ),
    }


async def _run_plans(target: Path, root: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for plan in (ProfilePlanId.NSYS_TIMELINE_V1, ProfilePlanId.NCU_TRIAGE_V1):
        results.append(await _run(target, root / plan.value, plan))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-arch", default="sm_86")
    args = parser.parse_args()
    task = next(item for item in core10_task_specs() if item.id.endswith("019.forward"))
    package_payload = build_fixed_cuda_candidate(task)
    package = validate_candidate_package(package_payload, task=task)
    source = archive_files(package_payload)["candidate.cu"]
    full_cases = build_development_case_bundle(task)
    cases = CaseBundle(
        task_spec_digest=full_cases.task_spec_digest,
        cases=(full_cases.cases[2],),
    )
    with tempfile.TemporaryDirectory(prefix="kernelblaster-profiler-smoke-") as temporary:
        root = Path(temporary)
        source_path = root / "candidate.cu"
        module_path = root / "module.cubin"
        source_path.write_bytes(source)
        subprocess.run(
            [
                "nvcc",
                "-O3",
                "-std=c++17",
                "--cubin",
                f"-arch={args.target_arch}",
                str(source_path),
                "-o",
                str(module_path),
            ],
            check=True,
            timeout=180,
        )
        candidate_capsule = build_candidate_capsule(
            package,
            module=module_path.read_bytes(),
            target_arch=args.target_arch,
            compiler_id="nvcc:profiler-smoke",
        )
        replay = build_profiler_replay_capsule(
            candidate_capsule,
            task_payload=task.canonical_bytes(),
            case_payload=cases.canonical_bytes(),
        )
        target = _materialize_profiler_target(
            replay,
            artifact_kind="candidate_profiler_capsule",
            root=root,
            request_digest=hashlib.sha256(replay).hexdigest(),
        )
        sanitizer_root = root / "sanitizer"
        sanitizer_root.mkdir()
        sanitizers = [
            run_sanitizer(
                plan,
                replay_executable=target.executable,
                capsule_path=root / "replay" / "candidate.capsule",
                root=sanitizer_root,
            )
            for plan in SanitizerPlan
        ]
        results = asyncio.run(_run_plans(target.executable, root))
    payload = {
        "schema_version": "profiler-capsule-smoke/v1",
        "candidate_capsule_sha256": hashlib.sha256(candidate_capsule).hexdigest(),
        "profiler_replay_sha256": hashlib.sha256(replay).hexdigest(),
        "results": results,
        "sanitizers": [
            {
                "plan": item.plan.value,
                "status": item.status,
                "reason_code": item.reason_code,
                "log_sha256": hashlib.sha256(item.log).hexdigest(),
            }
            for item in sanitizers
        ],
    }
    print(json.dumps(payload, sort_keys=True))
    return (
        0
        if all(item["reason_code"] == "none" for item in results)
        and all(item.status == "succeeded" for item in sanitizers)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
