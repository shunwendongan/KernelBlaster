from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from src.kernelblaster.harness import (  # noqa: E402
    CandidateRun,
    CaseBundle,
    CaseSpec,
    CaseTier,
    CorrectnessHarness,
    PyTorchAutogradAdapter,
    core10_task_specs,
)
from src.kernelblaster.harness.reference import torch_dtype  # noqa: E402


def _minimum_bundle(task):
    shape = {item.name: (item.values or (item.minimum,))[0] for item in task.dimensions}
    return CaseBundle(
        task_spec_digest=task.canonical_sha256(),
        cases=(
            CaseSpec(
                id="dev-minimum",
                tier=CaseTier.DEV,
                shape=shape,
                seed=7,
                distribution="normal",
            ),
        ),
    )


def _valid_candidate(task, adapter):
    expected_specs = {item.name: item for item in task.tensors if item.mutable}

    def candidate(inputs, scalars, context):
        reference = adapter.reference(task, inputs, scalars)
        return {
            name: value.to(torch_dtype(expected_specs[name].dtype))
            for name, value in reference.items()
        }

    return candidate


def test_core10_forward_and_backward_use_one_trusted_runtime():
    adapter = PyTorchAutogradAdapter()
    harness = CorrectnessHarness(device="cpu")
    results = [
        harness.evaluate(
            task,
            _minimum_bundle(task),
            adapter=adapter,
            candidate=_valid_candidate(task, adapter),
        )
        for task in core10_task_specs()
    ]
    assert len(results) == 20
    assert all(result.passed for result in results)
    backward = [result for result in results if result.direction.value == "backward"]
    tasks_by_digest = {task.canonical_sha256(): task for task in core10_task_specs()}
    for result in backward:
        task = tasks_by_digest[result.task_spec_digest]
        assert {metric.tensor for metric in result.cases[0].tensors} == {
            f"grad_{name}" for name in task.gradient_targets
        }


def test_runtime_owns_mutation_nonfinite_and_guard_verdicts():
    task = next(item for item in core10_task_specs() if item.id.endswith("019.forward"))
    bundle = _minimum_bundle(task)
    adapter = PyTorchAutogradAdapter()
    harness = CorrectnessHarness(device="cpu")

    def mutant(inputs, scalars, context):
        inputs["input"].add_(1)
        output = torch.full_like(inputs["input"], float("nan"))
        return CandidateRun(
            outputs={"output": output},
            guard_canary_intact=False,
            output_poison_overwritten=False,
        )

    result = harness.evaluate(task, bundle, adapter=adapter, candidate=mutant)
    assert not result.passed
    assert set(result.cases[0].violations) >= {
        "guard_canary",
        "input_mutation",
        "nonfinite_output",
        "output_poison",
    }


def test_case_bundle_rejects_an_out_of_contract_dynamic_shape():
    task = core10_task_specs()[0]
    bundle = _minimum_bundle(task)
    payload = bundle.model_dump(mode="json")
    payload["cases"][0]["shape"]["M"] = 999
    invalid = CaseBundle.model_validate(payload)
    with pytest.raises(ValueError, match="dimension constraints"):
        invalid.validate_for(task)


def test_all_naive_backward_sources_are_present():
    root = Path("portfolio/harness/core10/backward")
    names = {path.name[:3] for path in root.glob("*.cu")}
    assert names == {"004", "007", "019", "023", "026", "036", "040", "047", "088", "095"}
