#!/usr/bin/env python3
"""Run the ordered capability preflight and publish capability-report/v1."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kernelblaster.gpu_jobs import build_deterministic_bundle  # noqa: E402
from src.kernelblaster.observability import (  # noqa: E402
    RunRecorder,
    record_event,
    set_run_recorder,
)
from src.kernelblaster.preflight.client import ControlPlaneClient  # noqa: E402
from src.kernelblaster.preflight.contracts import AgentCapabilityMode  # noqa: E402
from src.kernelblaster.preflight.provider import build_provider_auth_probe  # noqa: E402
from src.kernelblaster.preflight.runner import (  # noqa: E402
    PreflightConfiguration,
    PreflightRunner,
)
from src.kernelblaster.portability import load_profile  # noqa: E402


PREFLIGHT_MARKER = "KERNELBLASTER_PREFLIGHT_JSON "
PREFLIGHT_MODEL = "gpt-5.6-sol"  # fallback only; profiles and environment override it


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "out" / "preflight" / timestamp


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


async def _run(args: argparse.Namespace) -> int:
    api_key = os.getenv("KERNELBLASTER_LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("KERNELBLASTER_LLM_API_KEY or OPENAI_API_KEY is required")
    if not args.control_token:
        raise RuntimeError("KERNELBLASTER_CONTROL_TOKEN is required")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite preflight output: {output_dir}")
    output_dir.mkdir(parents=True)

    provider_probe = build_provider_auth_probe(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        timeout_seconds=args.provider_timeout_seconds,
    )
    recorder = RunRecorder(
        output_dir,
        model=args.model,
        provider_config={
            "provider": "openai_compatible",
            "base_url": args.base_url,
            "api_key_configured": True,
            "max_requests": 1,
            "max_total_tokens": 64,
            "max_completion_tokens": 64,
            "max_concurrency": 1,
        },
        suite={"name": "capability-preflight", "requests": 1},
        gpu_target=args.gpu,
        repo_root=ROOT,
    )
    set_run_recorder(recorder)

    def observe(name, check) -> None:
        record_event(
            "capability_check_completed",
            status=("ok" if check.status.value != "unavailable" else "error"),
            stage=name.value,
            data={
                "check": name.value,
                "status": check.status.value,
                "reason_code": check.reason_code.value,
                "duration_ms": check.duration_ms,
            },
        )

    smoke = ROOT / "portfolio" / "trusted_gpu_smoke" / "vector_add.cu"
    source_bundle = build_deterministic_bundle({"vector_add.cu": smoke.read_bytes()})
    try:
        result = await PreflightRunner(
            ControlPlaneClient(args.control_url, args.control_token),
            provider_probe,
            configuration=PreflightConfiguration(
                private_evaluation_profile_id=args.private_evaluation_profile_id,
                benchmark_protocol_id=args.benchmark_protocol_id,
                minimum_free_vram_bytes=args.minimum_free_vram_bytes,
            ),
            observer=observe,
        ).run(source_bundle=source_bundle)
        _atomic_write(output_dir / "capability-report.json", result.report.canonical_bytes())
        public_result = {
            "report_digest": result.report_digest,
            "agent_mode": result.report.agent_mode.value,
            "execution_backend": result.report.execution_backend.value,
            "hardware_fingerprint": result.report.hardware_fingerprint,
            "target_arch": result.report.target_arch,
        }
        print(PREFLIGHT_MARKER + json.dumps(public_result, sort_keys=True))
        passed = bool(
            result.report_digest
            and result.report.agent_mode is not AgentCapabilityMode.UNAVAILABLE
        )
        recorder.close("completed" if passed else "failed")
        return 0 if passed else 2
    finally:
        set_run_recorder(None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--gpu", default=None, help="Evidence label only; use auto or a profile value.")
    parser.add_argument("--control-url", default="http://127.0.0.1:8000")
    parser.add_argument("--control-token", default=None)
    parser.add_argument(
        "--base-url",
        default=None,
    )
    parser.add_argument(
        "--private-evaluation-profile-id",
        default=os.getenv(
            "KERNELBLASTER_PREFLIGHT_PRIVATE_PROFILE_ID",
            "preflight-vector-add-v1",
        ),
    )
    parser.add_argument(
        "--benchmark-protocol-id",
        default="trusted-smoke-v1",
    )
    parser.add_argument(
        "--minimum-free-vram-bytes",
        type=int,
        default=int(
            os.getenv("KERNELBLASTER_PREFLIGHT_MIN_FREE_VRAM_BYTES", str(2 * 1024**3))
        ),
    )
    parser.add_argument("--provider-timeout-seconds", type=float, default=None)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=(
            Path(os.environ["KERNELBLASTER_CONTROL_ENV_FILE"])
            if os.getenv("KERNELBLASTER_CONTROL_ENV_FILE")
            else None
        ),
        help="Optional external Provider environment file; its contents are never logged.",
    )
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    args = parser.parse_args()
    profile = load_profile(args.profile)

    def configured(cli_value, environment_key: str, profile_key: str, fallback):
        if cli_value is not None:
            return cli_value
        if os.getenv(environment_key):
            return os.environ[environment_key]
        return profile.get(profile_key, fallback)

    args.model = configured(args.model, "KERNELBLASTER_PREFLIGHT_MODEL", "model", os.getenv("MODEL", PREFLIGHT_MODEL))
    args.gpu = configured(args.gpu, "KERNELBLASTER_GPU_LABEL", "gpu", "auto")
    args.provider_timeout_seconds = float(
        configured(args.provider_timeout_seconds, "KERNELBLASTER_PREFLIGHT_PROVIDER_TIMEOUT_SECONDS", "provider_timeout_seconds", 180)
    )
    if args.env_file is not None:
        load_dotenv(args.env_file.expanduser(), override=False)
    args.control_token = args.control_token or os.getenv("KERNELBLASTER_CONTROL_TOKEN")
    args.base_url = args.base_url or os.getenv(
        "KERNELBLASTER_LLM_BASE_URL", "https://api.openai.com/v1"
    )
    try:
        return asyncio.run(_run(args))
    except Exception as error:
        parser.exit(2, f"Preflight failed: {type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
