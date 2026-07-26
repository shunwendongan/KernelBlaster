"""Run the complete provider matrix without letting optional failures block CUDA."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
import uuid

from ..observability import record_event
from .contracts import BaselineProvider, BaselineRequest, BaselineResult


@dataclass(frozen=True)
class BaselineMatrix:
    task_id: str
    results: dict[BaselineProvider, BaselineResult]

    @property
    def upstream_cuda(self) -> BaselineResult | None:
        return self.results.get(BaselineProvider.UPSTREAM_CUDA)

    @property
    def formal_baseline_ready(self) -> bool:
        result = self.upstream_cuda
        return bool(result and result.comparable)

    @property
    def comparable_references(self) -> tuple[BaselineProvider, ...]:
        return tuple(
            provider
            for provider, result in self.results.items()
            if provider is not BaselineProvider.UPSTREAM_CUDA and result.comparable
        )


class BaselineCoordinator:
    def __init__(
        self,
        control: Any,
        *,
        task_id: str,
        task_spec_digest: str,
        case_bundle_digest: str,
        evaluation_bundle_digest: str,
        protocol_digest: str,
        hardware_fingerprint: str,
        target_arch: str,
        objective: str = "latency",
        timeout_seconds: int = 30 * 60,
    ) -> None:
        self.control = control
        self.task_id = task_id
        self.task_spec_digest = task_spec_digest
        self.case_bundle_digest = case_bundle_digest
        self.evaluation_bundle_digest = evaluation_bundle_digest
        self.protocol_digest = protocol_digest
        self.hardware_fingerprint = hardware_fingerprint
        self.target_arch = target_arch
        self.objective = objective
        self.timeout_seconds = timeout_seconds

    async def evaluate_all(self) -> BaselineMatrix:
        results: dict[BaselineProvider, BaselineResult] = {}
        for provider in BaselineProvider:
            request = BaselineRequest(
                request_id=f"{self.task_id}:{provider.value}:{uuid.uuid4().hex[:12]}",
                task_id=self.task_id,
                task_spec_digest=self.task_spec_digest,
                case_bundle_digest=self.case_bundle_digest,
                evaluation_bundle_digest=self.evaluation_bundle_digest,
                provider=provider,
                hardware_fingerprint=self.hardware_fingerprint,
                target_arch=self.target_arch,
                protocol_digest=self.protocol_digest,
                objective=self.objective,
                deadline=datetime.now(timezone.utc) + timedelta(seconds=self.timeout_seconds),
            )
            try:
                payload = await self.control.baseline(request.model_dump(mode="json"))
                result = BaselineResult.model_validate(payload)
            except Exception:
                # Transport/quota interruption is recoverable and distinct from
                # an implementation's convergence or correctness state.
                from .contracts import (
                    BaselineProvenance,
                    BaselineReasonCode,
                    BaselineStatus,
                )

                result = BaselineResult(
                    request_id=request.request_id,
                    status=BaselineStatus.BLOCKED,
                    reason_code=BaselineReasonCode.QUOTA_BLOCKED,
                    cache_key=request.cache_key(image_digest="sha256:" + "0" * 64),
                    provenance=BaselineProvenance(
                        provider=provider,
                        provider_version="unavailable",
                        image_digest="sha256:" + "0" * 64,
                        hardware_fingerprint=self.hardware_fingerprint,
                        target_arch=self.target_arch,
                        task_spec_digest=self.task_spec_digest,
                        case_bundle_digest=self.case_bundle_digest,
                        protocol_digest=self.protocol_digest,
                    ),
                )
            results[provider] = result
            record_event(
                "baseline_provider_completed",
                status="ok" if result.comparable else "error",
                data={
                    "task_id": self.task_id,
                    "provider": provider.value,
                    "status": result.status.value,
                    "reason_code": result.reason_code.value,
                    "comparable": result.comparable,
                },
            )
        return BaselineMatrix(task_id=self.task_id, results=results)


__all__ = ["BaselineCoordinator", "BaselineMatrix"]
