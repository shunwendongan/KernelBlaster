from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib

from src.kernelblaster.baseline_jobs.contracts import (
    BaselineCapabilities,
    BaselineProvider,
    BaselineReasonCode,
    BaselineRequest,
    BaselineStatus,
    BaselineWorkloadMeasurement,
)
from src.kernelblaster.baseline_jobs.worker import BaselineWorker, ProviderExecution
from src.kernelblaster.gpu_jobs.bundles import build_deterministic_bundle
from src.kernelblaster.harness import build_development_case_bundle, core10_task_specs


def _request(task, cases, bundle, **updates):
    payload = {
        "request_id": "019:forward:pytorch-eager",
        "task_id": task.id,
        "task_spec_digest": hashlib.sha256(task.canonical_bytes()).hexdigest(),
        "case_bundle_digest": hashlib.sha256(cases.canonical_bytes()).hexdigest(),
        "evaluation_bundle_digest": hashlib.sha256(bundle).hexdigest(),
        "provider": "pytorch_eager",
        "hardware_fingerprint": "gpu-a",
        "target_arch": "sm_86",
        "protocol_digest": "d" * 64,
        "deadline": datetime.now(timezone.utc) + timedelta(minutes=5),
    }
    payload.update(updates)
    return BaselineRequest.model_validate(payload)


class Runtime:
    def __init__(self):
        self.calls = 0

    async def execute(self, request, task, cases, evaluation_bundle):
        self.calls += 1
        assert request.provider is BaselineProvider.PYTORCH_EAGER
        assert cases.task_spec_digest == task.canonical_sha256()
        assert evaluation_bundle
        return ProviderExecution(
            status=BaselineStatus.SUCCEEDED,
            reason_code=BaselineReasonCode.NONE,
            correctness_passed=True,
            provider_version="torch:test",
            workloads=(
                BaselineWorkloadMeasurement(
                    workload_id="canonical-hot",
                    cache_mode="hot",
                    weight=1,
                    core=True,
                    device_samples_us=(10, 10, 10, 10, 10),
                    host_samples_us=(12, 12, 12, 12, 12),
                ),
            ),
        )


def test_worker_binds_every_cache_dimension_and_reuses_identical_result():
    task = next(item for item in core10_task_specs() if item.id.endswith("019.forward"))
    cases = build_development_case_bundle(task)
    bundle = build_deterministic_bundle(
        {"task-spec.json": task.canonical_bytes(), "case-bundle.json": cases.canonical_bytes()}
    )
    payloads = {
        hashlib.sha256(task.canonical_bytes()).hexdigest(): task.canonical_bytes(),
        hashlib.sha256(cases.canonical_bytes()).hexdigest(): cases.canonical_bytes(),
        hashlib.sha256(bundle).hexdigest(): bundle,
    }

    class Control:
        def __init__(self):
            self.downloads = 0
            self.uploads = 0

        async def download(self, digest):
            self.downloads += 1
            return payloads[digest]

        async def upload(self, payload, *, schema):
            self.uploads += 1
            assert schema == "baseline-result/v1"
            return {"digest": hashlib.sha256(payload).hexdigest()}

    runtime = Runtime()
    control = Control()
    worker = BaselineWorker(
        control,
        BaselineCapabilities(
            image_digest="sha256:" + "a" * 64,
            hardware_fingerprint="gpu-a",
            target_arch="sm_86",
            providers=tuple(BaselineProvider),
        ),
        providers={BaselineProvider.PYTORCH_EAGER: runtime},
    )
    request = _request(task, cases, bundle)
    first = asyncio.run(worker.evaluate(request))
    second = asyncio.run(worker.evaluate(request))
    assert first == second and first.comparable
    assert runtime.calls == 1 and control.downloads == 3 and control.uploads == 1
    assert set(first.artifact_roles.values()) == {"baseline_result"}
    assert request.cache_key(image_digest="sha256:" + "a" * 64) != request.cache_key(
        image_digest="sha256:" + "b" * 64
    )


def test_not_applicable_provider_is_unavailable_without_blocking_or_downloading():
    task = next(item for item in core10_task_specs() if item.id.endswith("019.forward"))
    cases = build_development_case_bundle(task)
    bundle = build_deterministic_bundle({"task-spec.json": task.canonical_bytes()})

    class Control:
        async def download(self, digest):
            raise AssertionError("not-applicable providers must not download artifacts")

        async def upload(self, payload, *, schema):
            return {"digest": hashlib.sha256(payload).hexdigest()}

    worker = BaselineWorker(
        Control(),
        BaselineCapabilities(
            image_digest="sha256:" + "a" * 64,
            hardware_fingerprint="gpu-a",
            target_arch="sm_86",
            providers=tuple(BaselineProvider),
        ),
    )
    result = asyncio.run(
        worker.evaluate(_request(task, cases, bundle, provider="triton"))
    )
    assert result.status is BaselineStatus.UNAVAILABLE
    assert result.reason_code is BaselineReasonCode.NOT_APPLICABLE
    assert not result.comparable


def test_hardware_mismatch_is_a_noncomparable_result():
    task = next(item for item in core10_task_specs() if item.id.endswith("019.forward"))
    cases = build_development_case_bundle(task)
    bundle = build_deterministic_bundle({"task-spec.json": task.canonical_bytes()})

    class Control:
        async def upload(self, payload, *, schema):
            return {"digest": hashlib.sha256(payload).hexdigest()}

    worker = BaselineWorker(
        Control(),
        BaselineCapabilities(
            image_digest="sha256:" + "a" * 64,
            hardware_fingerprint="gpu-a",
            target_arch="sm_86",
            providers=tuple(BaselineProvider),
        ),
    )
    result = asyncio.run(
        worker.evaluate(_request(task, cases, bundle, hardware_fingerprint="gpu-b"))
    )
    assert result.reason_code is BaselineReasonCode.HARDWARE_MISMATCH
    assert not result.comparable
