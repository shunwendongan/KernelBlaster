# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
from io import BytesIO
import json
import tarfile
import threading

import pytest
from pydantic import ValidationError

from src.kernelblaster.gpu_jobs.bundles import build_deterministic_bundle
from src.kernelblaster.gpu_jobs.contracts import (
    GpuCapabilities,
    GpuDeviceCapability,
    GpuJobManifest,
    GpuReasonCode,
    GpuJobStage,
    GpuRuntimeCapability,
)
from src.kernelblaster.gpu_jobs.sandbox import (
    DockerSandboxRuntime,
    PrivateEvaluationProfileManifest,
    SandboxConfiguration,
    SandboxExecution,
    SandboxPolicy,
    SandboxStageExecutor,
    _read_output_archive,
    _tar_files,
    public_generated_feedback,
)
from src.kernelblaster.gpu_jobs.sandbox_runner import _command, _validated_correctness
from src.kernelblaster.harness import build_development_case_bundle, core10_task_specs


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest(**updates: object) -> GpuJobManifest:
    payload: dict[str, object] = {
        "job_id": "job-1",
        "run_id": "run-1",
        "idempotency_key": "candidate-1",
        "stage": "compile",
        "source_bundle_digest": "a" * 64,
        "target_arch": "sm_86",
        "benchmark_protocol_id": "generated-v1",
        "deadline": datetime.now(timezone.utc) + timedelta(minutes=10),
        "trusted_bundle_kind": "generated_v1",
        "private_evaluation_profile_id": "private-v1",
    }
    payload.update(updates)
    return GpuJobManifest.model_validate(payload)


def _capabilities() -> GpuCapabilities:
    return GpuCapabilities(
        supervisor_id="test",
        device=GpuDeviceCapability(
            logical_id="0",
            name="test-gpu",
            compute_capability="8.6",
            target_arch="sm_86",
            total_memory_bytes=1024,
        ),
        runtime=GpuRuntimeCapability(cuda_version="12.8", driver_version="test"),
        generated_jobs_enabled=True,
    )


def _profiles(bundle_digest: str) -> PrivateEvaluationProfileManifest:
    return PrivateEvaluationProfileManifest.model_validate(
        {
            "profiles": [
                {"id": "private-v1", "bundle_digest": bundle_digest, "driver_path": "driver.cpp"}
            ]
        }
    )


class _Volume:
    name = "kernelblaster-input-test"

    def __init__(self):
        self.removed = False

    def remove(self, force=True):
        self.removed = force


class _Container:
    def __init__(self, output: bytes | None = None, *, timed_out: bool = False):
        self.output = output
        self.timed_out = timed_out
        self.removed = False
        self.started = False
        self.exec_commands: list[list[str]] = []
        self.attrs = {"State": {"OOMKilled": False}}

    def put_archive(self, path, payload):
        assert path == "/input" and payload
        return True

    def start(self):
        self.started = True

    def wait(self, timeout=None):
        if self.timed_out and timeout != 10:
            raise TimeoutError("timed out")
        return {"StatusCode": 0}

    def exec_run(self, command, **kwargs):
        self.exec_commands.append(command)
        assert command == ["/usr/bin/tar", "-C", "/work", "-cf", "-", "out"]
        assert kwargs == {
            "stdout": True,
            "stderr": False,
            "stream": True,
            "demux": False,
        }
        assert self.output is not None
        return type("ExecResult", (), {"output": iter([self.output])})()

    def reload(self):
        return None

    def kill(self, signal=None):
        return None

    def remove(self, force=True):
        self.removed = force


class _Client:
    def __init__(self, output: bytes, *, timed_out: bool = False):
        self.output = output
        self.volume = _Volume()
        self.created: list[tuple[tuple, dict]] = []
        self.stager = _Container()
        self.job = _Container(output, timed_out=timed_out)
        self.volumes = self
        self.containers = self
        self.images = self

    def ping(self):
        return True

    def get(self, image):
        return type("Image", (), {"id": image, "attrs": {"RepoDigests": []}})()

    def create(self, *args, **kwargs):
        if "name" in kwargs:
            return self.volume
        self.created.append((args, kwargs))
        return self.stager if len(self.created) == 1 else self.job


def test_generated_manifest_requires_a_private_profile_and_hides_driver_digest():
    with pytest.raises(ValidationError, match="private_evaluation_profile_id"):
        _manifest(private_evaluation_profile_id=None)
    with pytest.raises(ValidationError, match="may not accept driver_digest"):
        _manifest(driver_digest="b" * 64)
    manifest = _manifest()
    assert manifest.input_digests() == ("a" * 64,)


def test_sandbox_runtime_uses_pinned_image_private_namespace_and_always_cleans_up():
    output = _tar_files(
        {
            "out/stdout.log": b"ok\n",
            "out/stderr.log": b"",
            "out/result.json": b'{"reason":"none"}',
            "out/candidate": b"binary",
        }
    )
    client = _Client(output)
    policy = SandboxPolicy()
    configuration = SandboxConfiguration(
        image="sha256:" + "1" * 64,
        gpu_device="0",
        profiles=_profiles("b" * 64),
        policy=policy,
    )
    runtime = DockerSandboxRuntime(client, configuration)
    runtime.validate()
    execution = runtime.execute(input_archive=b"input", manifest=_manifest())
    assert execution.reason.value == "none"
    assert execution.outputs["candidate"] == b"binary"
    assert client.job.exec_commands
    assert client.volume.removed and client.stager.removed and client.job.removed
    _args, settings = client.created[1]
    assert settings["network_mode"] == "none"
    assert settings["ipc_mode"] == "private"
    assert settings["read_only"] is True
    assert settings["user"] == "65532:65532"
    assert settings["cap_drop"] == ["ALL"]
    assert settings["security_opt"] == ["no-new-privileges:true"]
    assert settings["pids_limit"] == 64
    assert settings["mem_limit"] == 8 * 1024**3
    assert settings["nano_cpus"] == 2_000_000_000
    assert settings["volumes"] == {client.volume.name: {"bind": "/input", "mode": "ro"}}
    assert "TOKEN" not in " ".join(settings["environment"])


def test_sandbox_rejects_symlink_and_unknown_output_members():
    payload = BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        link = tarfile.TarInfo("out/candidate")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        archive.addfile(link)
    with pytest.raises(ValueError, match="regular files"):
        _read_output_archive(payload.getvalue(), stage=GpuJobStage.COMPILE, policy=SandboxPolicy())


def test_sandbox_timeout_kills_the_job_waits_for_gpu_and_cleans_up(monkeypatch):
    output = _tar_files(
        {
            "out/stdout.log": b"",
            "out/stderr.log": b"",
            "out/result.json": b'{"reason":"none"}',
        }
    )
    client = _Client(output)
    runtime = DockerSandboxRuntime(
        client,
        SandboxConfiguration(
            image="sha256:" + "2" * 64,
            gpu_device="0",
            profiles=_profiles("b" * 64),
            policy=SandboxPolicy(compile_seconds=0),
        ),
    )
    monkeypatch.setattr(runtime, "_gpu_recovered", lambda: True)
    execution = runtime.execute(input_archive=b"input", manifest=_manifest())
    assert execution.reason.value == "stage_timeout"
    assert client.job.removed and client.volume.removed


def test_fixed_policy_has_distinct_stage_timeouts():
    policy = SandboxPolicy()
    assert policy.timeout_for(GpuJobStage.COMPILE) == 180
    assert policy.timeout_for(GpuJobStage.CORRECTNESS) == 60
    assert policy.timeout_for(GpuJobStage.EVENTS) == 90


def test_v2_private_profile_requires_all_harness_bindings():
    payload = {
        "schema_version": "gpu-private-evaluation-profiles/v2",
        "profiles": [
            {
                "id": "core10-v2",
                "bundle_digest": "a" * 64,
                "driver_path": "driver.cpp",
                "task_spec_digest": "b" * 64,
                "case_bundle_digest": "c" * 64,
                "adapter_id": "kernelbench.legacy-driver",
                "adapter_version": "1.0.0",
                "correctness_protocol_id": "generated-correctness-v2",
                "disclosure": "adaptive_disclosed",
            }
        ],
    }
    assert PrivateEvaluationProfileManifest.model_validate(payload).profiles[0].adapter_id
    del payload["profiles"][0]["task_spec_digest"]
    with pytest.raises(ValidationError, match="versioned Harness bindings"):
        PrivateEvaluationProfileManifest.model_validate(payload)


def test_v2_profile_bundle_is_canonically_bound_to_task_cases_and_adapter():
    task = core10_task_specs()[0]
    cases = build_development_case_bundle(task)
    profile = PrivateEvaluationProfileManifest.model_validate(
        {
            "schema_version": "gpu-private-evaluation-profiles/v2",
            "profiles": [
                {
                    "id": "core10-v2",
                    "bundle_digest": "a" * 64,
                    "driver_path": "driver.cpp",
                    "task_spec_digest": task.canonical_sha256(),
                    "case_bundle_digest": cases.canonical_sha256(),
                    "adapter_id": task.adapter_id,
                    "adapter_version": task.adapter_version,
                    "correctness_protocol_id": "generated-correctness-v2",
                    "disclosure": "adaptive_disclosed",
                }
            ],
        }
    ).profiles[0]
    files = {
        "private/task-spec.json": task.canonical_bytes(),
        "private/case-bundle.json": cases.canonical_bytes(),
    }
    SandboxStageExecutor._validate_v2_profile(profile, files)
    files["private/case-bundle.json"] += b"\n"
    with pytest.raises(ValueError, match="canonical encoding"):
        SandboxStageExecutor._validate_v2_profile(profile, files)


def test_runner_requires_v2_correctness_to_match_task_digest_and_full_schema():
    result = {
        "schema_version": "correctness-result/v2",
        "protocol_id": "generated-correctness-v2",
        "task_spec_digest": "a" * 64,
        "direction": "forward",
        "passed": True,
        "case_count": 1,
        "cases": [
            {
                "case_id": "dev-1",
                "tier": "dev",
                "shape": {"N": 33},
                "seed": 7,
                "distribution": "normal",
                "passed": True,
                "deterministic": True,
                "cuda_error": None,
                "tensors": [
                    {
                        "tensor": "output",
                        "count": 33,
                        "mismatch_count": 0,
                        "nonfinite_count": 0,
                        "max_abs_error": 0.0,
                        "p99_abs_error": 0.0,
                        "max_normalized_error": 0.0,
                    }
                ],
            }
        ],
    }
    request = {
        "correctness_protocol_id": "generated-correctness-v2",
        "task_spec_digest": "a" * 64,
    }
    canonical, passed = _validated_correctness(request, json.dumps(result).encode())
    assert passed and json.loads(canonical)["case_count"] == 1
    request["task_spec_digest"] = "b" * 64
    with pytest.raises(ValueError, match="binding mismatch"):
        _validated_correctness(request, json.dumps(result).encode())


def test_sandbox_compile_always_enables_ptxas_verbose(monkeypatch):
    class CandidateRoot:
        def rglob(self, pattern):
            assert pattern == "*.cu"
            return ["/input/candidate/candidate.cu"]

    class Driver:
        def is_file(self):
            return True

        def __str__(self):
            return "/input/private/driver.cpp"

    monkeypatch.setattr(
        "src.kernelblaster.gpu_jobs.sandbox_runner.INPUT",
        type(
            "Input",
            (),
            {
                "__truediv__": lambda self, value: (
                    CandidateRoot() if value == "candidate" else Driver()
                )
            },
        )(),
    )
    command = _command(
        {
            "stage": "compile",
            "target_arch": "sm_86",
            "driver_path": "private/driver.cpp",
        }
    )
    assert "--ptxas-options=-v" in command


def test_stage_executor_imports_only_allowed_outputs_and_public_feedback_has_no_private_data():
    source = build_deterministic_bundle({"candidate.cu": b"// candidate\n"})
    private = build_deterministic_bundle(
        {"driver.cpp": b"// private driver\n", "seeds.txt": b"do-not-prompt"}
    )
    source_digest = _digest(source)
    private_digest = _digest(private)
    manifest = _manifest(source_bundle_digest=source_digest)

    class Control:
        def __init__(self):
            self.payloads = {source_digest: source, private_digest: private}
            self.uploads: list[bytes] = []

        async def download(self, digest):
            return self.payloads[digest]

        async def upload(self, payload, *, media_type, schema):
            self.uploads.append(payload)
            return {"digest": _digest(payload)}

    class Runtime:
        def validate(self):
            return None

        def execute(self, *, input_archive, manifest):
            # The private driver is delivered only to the Job's read-only input,
            # never exposed through the Supervisor's result/feedback API.
            assert b"do-not-prompt" in input_archive
            return SandboxExecution(
                reason=GpuReasonCode.NONE,
                outputs={"candidate": b"binary", "stdout.log": b"ok", "stderr.log": b""},
            )

    control = Control()
    executor = SandboxStageExecutor(control, Runtime(), _profiles(private_digest))
    result = asyncio.run(executor(manifest, _capabilities(), asyncio.Event()))
    assert result.status.value == "succeeded"
    feedback = public_generated_feedback(result)
    assert "private" not in json.dumps(feedback)
    assert b"do-not-prompt" not in b"".join(control.uploads)


def test_stage_executor_cancellation_kills_the_active_docker_job():
    source = build_deterministic_bundle({"candidate.cu": b"// candidate\n"})
    private = build_deterministic_bundle({"driver.cpp": b"// private driver\n"})
    source_digest = _digest(source)
    private_digest = _digest(private)

    class Control:
        async def download(self, digest):
            return {source_digest: source, private_digest: private}[digest]

    class Runtime:
        def __init__(self):
            self.release = threading.Event()
            self.started = threading.Event()
            self.cancelled = False

        def execute(self, *, input_archive, manifest):
            self.started.set()
            self.release.wait(timeout=2)
            return SandboxExecution(reason=GpuReasonCode.NONE, outputs={})

        def cancel_active(self):
            self.cancelled = True
            self.release.set()

    async def scenario():
        runtime = Runtime()
        executor = SandboxStageExecutor(Control(), runtime, _profiles(private_digest))
        task = asyncio.create_task(
            executor(_manifest(source_bundle_digest=source_digest), _capabilities(), asyncio.Event())
        )
        assert await asyncio.to_thread(runtime.started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert runtime.cancelled is True

    asyncio.run(scenario())
