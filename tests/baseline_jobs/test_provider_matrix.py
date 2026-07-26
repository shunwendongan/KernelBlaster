from __future__ import annotations

import asyncio

from src.kernelblaster.baseline_jobs import BaselineCoordinator, BaselineProvider


def test_optional_provider_transport_failure_does_not_hide_other_columns():
    class Control:
        async def baseline(self, request):
            if request["provider"] == "cutlass":
                raise RuntimeError("quota")
            comparable = request["provider"] in {"upstream_cuda", "pytorch_eager"}
            return {
                "schema_version": "baseline-result/v1",
                "request_id": request["request_id"],
                "status": "succeeded" if comparable else "unavailable",
                "reason_code": "none" if comparable else "not_applicable",
                "correctness_passed": comparable,
                "comparable": comparable,
                "cache_key": "a" * 64,
                "workloads": (
                    [
                        {
                            "workload_id": "hot",
                            "cache_mode": "hot",
                            "weight": 1,
                            "core": True,
                            "device_samples_us": [10] * 5,
                            "host_samples_us": [12] * 5,
                        }
                    ]
                    if comparable
                    else []
                ),
                "provenance": {
                    "provider": request["provider"],
                    "provider_version": "test",
                    "image_digest": "sha256:" + "b" * 64,
                    "hardware_fingerprint": "gpu-a",
                    "target_arch": "sm_86",
                    "task_spec_digest": "c" * 64,
                    "case_bundle_digest": "d" * 64,
                    "protocol_digest": "e" * 64,
                },
                "artifact_roles": {},
            }

    matrix = asyncio.run(
        BaselineCoordinator(
            Control(),
            task_id="example.task.forward",
            task_spec_digest="c" * 64,
            case_bundle_digest="d" * 64,
            evaluation_bundle_digest="f" * 64,
            protocol_digest="e" * 64,
            hardware_fingerprint="gpu-a",
            target_arch="sm_86",
        ).evaluate_all()
    )
    assert matrix.formal_baseline_ready
    assert matrix.comparable_references == (BaselineProvider.PYTORCH_EAGER,)
    assert matrix.results[BaselineProvider.CUTLASS].status.value == "blocked"
