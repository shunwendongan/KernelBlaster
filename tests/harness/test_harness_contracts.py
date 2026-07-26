from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from src.kernelblaster.harness import (
    CaseBundle,
    CorrectnessResultV2,
    Direction,
    build_development_case_bundle,
    core10_task_specs,
    parse_correctness_stdout,
)


def _case(*, passed: bool = True) -> dict[str, object]:
    mismatch = 0 if passed else 1
    return {
        "case_id": "boundary-33",
        "tier": "feedback",
        "shape": {"B": 1, "D": 33},
        "seed": 42,
        "distribution": "extreme",
        "passed": passed,
        "deterministic": True,
        "cuda_error": None,
        "tensors": [
            {
                "tensor": "output",
                "count": 33,
                "mismatch_count": mismatch,
                "nonfinite_count": 0,
                "max_abs_error": 0.01,
                "p99_abs_error": 0.005,
                "max_normalized_error": 2.0 if mismatch else 0.5,
            }
        ],
    }


def _result(*, passed: bool = True) -> dict[str, object]:
    return {
        "schema_version": "correctness-result/v2",
        "protocol_id": "generated-correctness-v2",
        "task_spec_digest": "a" * 64,
        "direction": "forward",
        "passed": passed,
        "case_count": 1,
        "cases": [_case(passed=passed)],
    }


def test_core10_has_forward_and_backward_contracts_for_every_task():
    specs = core10_task_specs()
    assert len(specs) == 20
    for task_id in ("004", "007", "019", "023", "026", "036", "040", "047", "088", "095"):
        selected = [item for item in specs if f".{task_id}." in item.id]
        assert {item.direction for item in selected} == {Direction.FORWARD, Direction.BACKWARD}
        backward = next(item for item in selected if item.direction is Direction.BACKWARD)
        differentiable = {
            tensor.name
            for tensor in backward.tensors
            if tensor.differentiable and tensor.gradient_of is None
        }
        assert set(backward.gradient_targets) == differentiable
        assert {tensor.gradient_of for tensor in backward.tensors if tensor.gradient_of} == (
            differentiable
        )
        assert {item.cache_mode.value for item in backward.workloads} == {"hot", "rotating"}


def test_only_four_forward_contracts_advertise_triton():
    advertised = {
        item.id
        for item in core10_task_specs()
        if "triton" in item.candidate_backends
    }
    assert advertised == {
        "kernelbench.level1.007.forward",
        "kernelbench.level1.026.forward",
        "kernelbench.level1.036.forward",
        "kernelbench.level1.047.forward",
    }


def test_task_digest_is_canonical_and_workloads_bind_every_dimension():
    task = core10_task_specs()[0]
    assert task.canonical_sha256() == hashlib.sha256(task.canonical_bytes()).hexdigest()
    payload = task.model_dump(mode="json")
    payload["workloads"][0]["shape"].pop("M")
    with pytest.raises(ValidationError, match="bind every declared dimension"):
        type(task).model_validate(payload)


def test_correctness_v2_requires_one_unambiguous_json_line_and_consistent_flags():
    encoded = json.dumps(_result(), sort_keys=True).encode()
    parsed = parse_correctness_stdout(encoded + b"\n")
    assert parsed.passed and parsed.case_count == 1
    assert parsed.public_feedback()["cases"][0]["seed"] == 42
    with pytest.raises(ValueError, match="exactly one JSON line"):
        parse_correctness_stdout(b"log line\n" + encoded)
    contradictory = _result()
    contradictory["passed"] = False
    with pytest.raises(ValidationError, match="contradicts"):
        CorrectnessResultV2.model_validate(contradictory)


def test_failed_correctness_is_valid_evidence_but_not_a_pass():
    parsed = CorrectnessResultV2.model_validate(_result(passed=False))
    assert parsed.passed is False
    assert parsed.cases[0].tensors[0].mismatch_count == 1


def test_case_bundle_is_adaptive_disclosed_and_content_addressed():
    payload = {
        "task_spec_digest": "b" * 64,
        "cases": [
            {
                "id": "audit-1",
                "tier": "audit",
                "shape": {"B": 1},
                "seed": 7,
                "distribution": "normal",
            }
        ],
    }
    first = CaseBundle.model_validate(payload)
    second = CaseBundle.model_validate(payload)
    assert first.disclosure == "adaptive_disclosed"
    assert first.canonical_sha256() == second.canonical_sha256()


def test_every_core10_direction_has_dev_feedback_and_audit_cases():
    for task in core10_task_specs():
        bundle = build_development_case_bundle(task)
        assert bundle.task_spec_digest == task.canonical_sha256()
        assert {case.tier.value for case in bundle.cases} == {"dev", "feedback", "audit"}
        dimensions = {item.name: item for item in task.dimensions}
        for case in bundle.cases:
            assert set(case.shape) == set(dimensions)
            for name, value in case.shape.items():
                dimension = dimensions[name]
                assert dimension.minimum <= value <= dimension.maximum
