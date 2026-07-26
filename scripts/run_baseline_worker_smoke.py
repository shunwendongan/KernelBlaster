#!/usr/bin/env python3
"""Run the independent PyTorch baseline and strict gate on a real CUDA GPU."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kernelblaster.baseline_jobs import (  # noqa: E402
    BaselineCapabilities,
    BaselineProvider,
    BaselineRequest,
    PairedWorkload,
    evaluate_multi_workload_gate,
)
from src.kernelblaster.baseline_jobs.providers import PyTorchEagerProviderRuntime  # noqa: E402
from src.kernelblaster.baseline_jobs.worker import BaselineWorker  # noqa: E402
from src.kernelblaster.gpu_jobs.bundles import build_deterministic_bundle  # noqa: E402
from src.kernelblaster.harness import build_development_case_bundle, core10_task_specs  # noqa: E402


async def _run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    task = next(item for item in core10_task_specs() if item.id.endswith("019.forward"))
    cases = build_development_case_bundle(task)
    evaluation_bundle = build_deterministic_bundle(
        {
            "task-spec.json": task.canonical_bytes(),
            "case-bundle.json": cases.canonical_bytes(),
            "init.cu": (
                ROOT / "data" / "kernelbench-cuda" / "level1" / "019_ReLU" / "init.cu"
            ).read_bytes(),
        }
    )
    payloads = {
        task.canonical_sha256(): task.canonical_bytes(),
        cases.canonical_sha256(): cases.canonical_bytes(),
        hashlib.sha256(evaluation_bundle).hexdigest(): evaluation_bundle,
    }

    class Control:
        async def download(self, digest):
            return payloads[digest]

        async def upload(self, payload, *, schema):
            assert schema == "baseline-result/v1"
            return {"digest": hashlib.sha256(payload).hexdigest()}

    properties = torch.cuda.get_device_properties(0)
    hardware_payload = {
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory": properties.total_memory,
    }
    hardware_fingerprint = hashlib.sha256(
        json.dumps(hardware_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    image_digest = args.image_digest
    worker = BaselineWorker(
        Control(),
        BaselineCapabilities(
            image_digest=image_digest,
            hardware_fingerprint=hardware_fingerprint,
            target_arch=f"sm_{properties.major}{properties.minor}",
            providers=tuple(BaselineProvider),
        ),
        providers={BaselineProvider.PYTORCH_EAGER: PyTorchEagerProviderRuntime()},
    )
    request = BaselineRequest(
        request_id="019:forward:pytorch-eager:smoke",
        task_id=task.id,
        task_spec_digest=task.canonical_sha256(),
        case_bundle_digest=cases.canonical_sha256(),
        evaluation_bundle_digest=hashlib.sha256(evaluation_bundle).hexdigest(),
        provider=BaselineProvider.PYTORCH_EAGER,
        hardware_fingerprint=hardware_fingerprint,
        target_arch=f"sm_{properties.major}{properties.minor}",
        protocol_digest=hashlib.sha256(b"baseline-events-v1").hexdigest(),
        deadline=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    result = await worker.evaluate(request)
    if not result.comparable:
        raise RuntimeError(f"PyTorch eager baseline was not comparable: {result.reason_code}")
    paired = tuple(
        PairedWorkload(
            workload_id=item.workload_id,
            weight=item.weight,
            core=item.core,
            baseline_device_us=item.device_samples_us,
            candidate_device_us=tuple(value * 0.90 for value in item.device_samples_us),
            baseline_host_us=item.host_samples_us,
            candidate_host_us=tuple(value * 1.10 for value in item.host_samples_us),
        )
        for item in result.workloads
    )
    gate = evaluate_multi_workload_gate(paired)
    if not gate.qualified:
        raise RuntimeError("strict multi-workload gate fixture failed")
    return {
        "schema_version": "baseline-worker-smoke/v1",
        "hardware": hardware_payload,
        "hardware_fingerprint": hardware_fingerprint,
        "image_digest": image_digest,
        "task_spec_digest": task.canonical_sha256(),
        "case_bundle_digest": cases.canonical_sha256(),
        "baseline": result.model_dump(mode="json"),
        "gate_fixture": {
            "kind": "fixed_candidate_10_percent_faster_device_time",
            **asdict(gate),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-digest", default="sha256:" + "0" * 64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = asyncio.run(_run(args))
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
