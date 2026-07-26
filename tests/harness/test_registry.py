from pathlib import Path

import pytest

from src.kernelblaster.harness import (
    AdapterDescriptor,
    AdapterKind,
    AdapterRegistry,
    LegacyDriverAdapter,
    core10_task_specs,
)


def test_registry_is_an_id_allowlist_not_a_runtime_path_loader():
    task = core10_task_specs()[0]
    registry = AdapterRegistry(
        (
            AdapterDescriptor(
                id="kernelbench.legacy-driver",
                version="1.0.0",
                kind=AdapterKind.LEGACY_DRIVER,
                task_ids=frozenset({task.id}),
            ),
        )
    )
    assert registry.resolve(task).kind is AdapterKind.LEGACY_DRIVER
    other = core10_task_specs()[2]
    with pytest.raises(KeyError, match="adapter_not_allowlisted"):
        registry.resolve(other)


def test_legacy_driver_requires_exact_pass_token(tmp_path: Path):
    driver = tmp_path / "driver.cpp"
    driver.write_text(
        "void launch_gpu_implementation();\nint main() { return 0; }\n", encoding="utf-8"
    )
    adapter = LegacyDriverAdapter("task", driver)
    assert len(adapter.digest) == 64
    assert adapter.passed(0, b"passed\n")
    assert not adapter.passed(0, b"failed\n")
    assert not adapter.passed(0, b"log\npassed\n")
    assert not adapter.passed(1, b"passed\n")
