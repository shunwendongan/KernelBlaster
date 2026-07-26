"""Independent immutable Baseline Worker; never executes generated candidates."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
import re
import subprocess
from typing import Protocol

from fastapi import Depends, FastAPI
import uvicorn

from ..gpu_jobs.bundles import validate_bundle
from ..harness.contracts import CaseBundle, TaskSpec
from ..servers.auth import require_baseline_token, validate_baseline_token
from .client import ControlBaselineClient
from .contracts import (
    BaselineCapabilities,
    BaselineProvider,
    BaselineProvenance,
    BaselineReasonCode,
    BaselineRequest,
    BaselineResult,
    BaselineStatus,
    BaselineWorkloadMeasurement,
)


_IMAGE = re.compile(r"^sha256:[0-9a-f]{64}$")


_APPLICABILITY: dict[BaselineProvider, frozenset[str] | None] = {
    BaselineProvider.UPSTREAM_CUDA: None,
    BaselineProvider.PYTORCH_EAGER: None,
    BaselineProvider.PYTORCH_COMPILE: None,
    BaselineProvider.TRITON: frozenset({"007", "026", "036", "047"}),
    BaselineProvider.CUBLAS: frozenset({"004", "007"}),
    BaselineProvider.CUDNN: frozenset({"036", "040"}),
    BaselineProvider.CUTLASS: frozenset({"004", "007"}),
}


@dataclass(frozen=True)
class ProviderExecution:
    status: BaselineStatus
    reason_code: BaselineReasonCode
    correctness_passed: bool
    provider_version: str
    workloads: tuple[BaselineWorkloadMeasurement, ...] = ()


class ProviderRuntime(Protocol):
    async def execute(
        self,
        request: BaselineRequest,
        task: TaskSpec,
        cases: CaseBundle,
        evaluation_bundle: bytes,
    ) -> ProviderExecution: ...


class UnconfiguredProviderRuntime:
    """Fail explicitly until an image registers a fixed provider implementation."""

    async def execute(
        self,
        request: BaselineRequest,
        task: TaskSpec,
        cases: CaseBundle,
        evaluation_bundle: bytes,
    ) -> ProviderExecution:
        del task, cases, evaluation_bundle
        return ProviderExecution(
            status=BaselineStatus.UNAVAILABLE,
            reason_code=BaselineReasonCode.PROVIDER_UNAVAILABLE,
            correctness_passed=False,
            provider_version=f"{request.provider.value}:unconfigured",
        )


class BaselineWorker:
    def __init__(
        self,
        control: ControlBaselineClient,
        capabilities: BaselineCapabilities,
        *,
        providers: dict[BaselineProvider, ProviderRuntime] | None = None,
    ) -> None:
        self.control = control
        self.capabilities = capabilities
        self.providers = providers or {}
        self._cache: dict[str, BaselineResult] = {}
        self._lock = asyncio.Lock()

    async def _download(self, digest: str) -> bytes:
        payload = await self.control.download(digest)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("artifact_hash_mismatch")
        return payload

    async def evaluate(self, request: BaselineRequest) -> BaselineResult:
        cache_key = request.cache_key(image_digest=self.capabilities.image_digest)
        async with self._lock:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
            result = await self._evaluate_uncached(request, cache_key)
            encoded = json.dumps(
                result.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            uploaded = await self.control.upload(encoded, schema="baseline-result/v1")
            digest = hashlib.sha256(encoded).hexdigest()
            if uploaded.get("digest") != digest:
                raise ValueError("artifact_hash_mismatch")
            result = result.model_copy(update={"artifact_roles": {digest: "baseline_result"}})
            self._cache[cache_key] = result
            return result

    async def _evaluate_uncached(self, request: BaselineRequest, cache_key: str) -> BaselineResult:
        provenance = BaselineProvenance(
            provider=request.provider,
            provider_version="unavailable",
            image_digest=self.capabilities.image_digest,
            hardware_fingerprint=self.capabilities.hardware_fingerprint,
            target_arch=self.capabilities.target_arch,
            task_spec_digest=request.task_spec_digest,
            case_bundle_digest=request.case_bundle_digest,
            protocol_digest=request.protocol_digest,
        )
        if (
            request.hardware_fingerprint != self.capabilities.hardware_fingerprint
            or request.target_arch != self.capabilities.target_arch
        ):
            return BaselineResult(
                request_id=request.request_id,
                status=BaselineStatus.FAILED,
                reason_code=BaselineReasonCode.HARDWARE_MISMATCH,
                cache_key=cache_key,
                provenance=provenance,
            )
        parts = request.task_id.split(".")
        number = parts[2] if len(parts) > 2 else ""
        applicable = _APPLICABILITY[request.provider]
        if applicable is not None and number not in applicable:
            return BaselineResult(
                request_id=request.request_id,
                status=BaselineStatus.UNAVAILABLE,
                reason_code=BaselineReasonCode.NOT_APPLICABLE,
                cache_key=cache_key,
                provenance=provenance,
            )
        try:
            task_payload, case_payload, evaluation_bundle = await asyncio.gather(
                self._download(request.task_spec_digest),
                self._download(request.case_bundle_digest),
                self._download(request.evaluation_bundle_digest),
            )
            task = TaskSpec.model_validate_json(task_payload)
            cases = CaseBundle.model_validate_json(case_payload)
            if task.canonical_bytes() != task_payload or cases.canonical_bytes() != case_payload:
                raise ValueError("artifact_hash_mismatch")
            if task.id != request.task_id:
                raise ValueError("artifact_hash_mismatch")
            cases.validate_for(task)
            validate_bundle(evaluation_bundle)
        except ValueError:
            return BaselineResult(
                request_id=request.request_id,
                status=BaselineStatus.FAILED,
                reason_code=BaselineReasonCode.ARTIFACT_HASH_MISMATCH,
                cache_key=cache_key,
                provenance=provenance,
            )
        runtime = self.providers.get(request.provider, UnconfiguredProviderRuntime())
        try:
            execution = await runtime.execute(request, task, cases, evaluation_bundle)
        except Exception:
            execution = ProviderExecution(
                status=BaselineStatus.FAILED,
                reason_code=BaselineReasonCode.EXECUTION_FAILED,
                correctness_passed=False,
                provider_version="error",
            )
        provenance = provenance.model_copy(
            update={"provider_version": execution.provider_version}
        )
        return BaselineResult(
            request_id=request.request_id,
            status=execution.status,
            reason_code=execution.reason_code,
            correctness_passed=execution.correctness_passed,
            comparable=(
                execution.status is BaselineStatus.SUCCEEDED
                and execution.reason_code is BaselineReasonCode.NONE
                and execution.correctness_passed
                and bool(execution.workloads)
            ),
            cache_key=cache_key,
            workloads=execution.workloads,
            provenance=provenance,
        )


def capabilities_from_environment() -> BaselineCapabilities:
    image = os.getenv("KERNELBLASTER_BASELINE_IMAGE_DIGEST", "")
    if not _IMAGE.fullmatch(image):
        raise RuntimeError("Baseline Worker requires an immutable sha256 image digest")
    hardware = os.getenv("KERNELBLASTER_HARDWARE_FINGERPRINT", "").strip()
    target_arch = os.getenv("KERNELBLASTER_TARGET_ARCH", "").strip()
    if not hardware or not target_arch:
        try:
            detected = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,name,compute_cap,driver_version",
                    "--format=csv,noheader,nounits",
                    "--id=0",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            uuid, name, capability, driver = [item.strip() for item in detected.split(",")]
            hardware = hashlib.sha256(
                json.dumps(
                    {"uuid": uuid, "name": name, "compute_capability": capability, "driver": driver},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            target_arch = f"sm_{capability.replace('.', '')}"
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise RuntimeError("Baseline Worker cannot bind its GPU hardware") from error
    if not re.fullmatch(r"sm_[0-9]{2,3}", target_arch):
        raise RuntimeError("Baseline Worker requires a valid target arch")
    return BaselineCapabilities(
        image_digest=image,
        hardware_fingerprint=hardware,
        target_arch=target_arch,
        providers=tuple(BaselineProvider),
    )


APP = FastAPI(title="KernelBlaster Baseline Worker")


def _worker() -> BaselineWorker:
    worker = getattr(APP.state, "worker", None)
    if worker is None:
        token = validate_baseline_token()
        from .providers import built_in_provider_runtimes

        worker = BaselineWorker(
            ControlBaselineClient(
                os.getenv("KERNELBLASTER_CONTROL_URL", "http://control:8000"), token
            ),
            capabilities_from_environment(),
            providers=built_in_provider_runtimes(),
        )
        APP.state.worker = worker
    return worker


@APP.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "baseline-worker"}


@APP.get("/ready")
async def ready(_authorized: None = Depends(require_baseline_token)) -> dict[str, str]:
    _worker()
    return {"status": "ready", "service": "baseline-worker"}


@APP.get("/v1/capabilities")
async def capabilities(
    _authorized: None = Depends(require_baseline_token),
) -> dict[str, object]:
    return _worker().capabilities.model_dump(mode="json")


@APP.post("/v1/baselines")
async def evaluate(
    request: BaselineRequest,
    _authorized: None = Depends(require_baseline_token),
) -> dict[str, object]:
    return (await _worker().evaluate(request)).model_dump(mode="json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=2004)
    args = parser.parse_args()
    validate_baseline_token()
    _worker()
    uvicorn.run(APP, host=args.host, port=args.port)


if __name__ == "__main__":
    main()


__all__ = [
    "APP",
    "BaselineWorker",
    "ProviderExecution",
    "ProviderRuntime",
    "UnconfiguredProviderRuntime",
    "capabilities_from_environment",
]
