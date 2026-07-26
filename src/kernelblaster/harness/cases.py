"""Deterministic public fixtures; deployments may replace them with external bundles."""

from __future__ import annotations

from .contracts import CaseBundle, CaseSpec, CaseTier, TaskSpec


def _shape(task: TaskSpec, selector: str) -> dict[str, int]:
    selected: dict[str, int] = {}
    for dimension in task.dimensions:
        values = dimension.values or (dimension.minimum, dimension.maximum)
        if selector == "minimum":
            value = values[0]
        elif selector == "odd":
            value = next((item for item in values if item % 2 == 1), values[0])
        elif selector == "middle":
            value = values[len(values) // 2]
        elif selector == "near-maximum":
            value = values[-2] if len(values) > 1 else values[-1]
        else:
            value = values[-1]
        selected[dimension.name] = value
    return selected


def build_development_case_bundle(task: TaskSpec) -> CaseBundle:
    """Create five disclosed cases spanning the task's bounded generator."""
    cases = (
        CaseSpec(
            id="dev-canonical",
            tier=CaseTier.DEV,
            shape=_shape(task, "maximum"),
            seed=0,
            distribution="normal",
        ),
        CaseSpec(
            id="feedback-minimum",
            tier=CaseTier.FEEDBACK,
            shape=_shape(task, "minimum"),
            seed=42,
            distribution="zero",
        ),
        CaseSpec(
            id="feedback-odd",
            tier=CaseTier.FEEDBACK,
            shape=_shape(task, "odd"),
            seed=20260721,
            distribution="alternating",
        ),
        CaseSpec(
            id="audit-extreme",
            tier=CaseTier.AUDIT,
            shape=_shape(task, "near-maximum"),
            seed=20260722,
            distribution="extreme",
        ),
        CaseSpec(
            id="audit-constant",
            tier=CaseTier.AUDIT,
            shape=_shape(task, "middle"),
            seed=20260723,
            distribution="constant",
        ),
    )
    return CaseBundle(task_spec_digest=task.canonical_sha256(), cases=cases)


__all__ = ["build_development_case_bundle"]
