# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import atexit
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import threading
from typing import Any
import uuid


SCHEMA_VERSION = "2.0"
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)
_SAFE_USAGE_KEYS = {
    "api_key_configured",
    "cached_tokens",
    "cache_write_tokens",
    "completion_tokens",
    "input_tokens",
    "max_completion_tokens",
    "max_total_tokens",
    "output_tokens",
    "prompt_tokens",
    "reasoning_tokens",
    "total_tokens",
}
_BEARER_PATTERN = re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/=-]+")
_AUTH_PATTERN = re.compile(
    r"(?i)(\bauthorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;}]+"
)
_QUERY_SECRET_PATTERN = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|secret|password)=)[^&#\s]+"
)
_URL_USERINFO_PATTERN = re.compile(r"(?i)(https?://)[^/@\s]+@")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def redact_secrets(value: Any, key: str = "") -> Any:
    """Recursively redact values whose field names conventionally hold secrets."""
    normalized_key = key.lower()
    if any(part in normalized_key for part in _SENSITIVE_KEY_PARTS):
        if normalized_key in _SAFE_USAGE_KEYS:
            return value
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(k): redact_secrets(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item, key) for item in value]
    if isinstance(value, str):
        redacted = _BEARER_PATTERN.sub(r"\1[REDACTED]", value)
        redacted = _AUTH_PATTERN.sub(r"\1[REDACTED]", redacted)
        redacted = _QUERY_SECRET_PATTERN.sub(r"\1[REDACTED]", redacted)
        return _URL_USERINFO_PATTERN.sub(r"\1[REDACTED]@", redacted)
    return value


def prompt_metadata(messages: list[dict], include_content: bool = False) -> dict[str, Any]:
    canonical = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    metadata: dict[str, Any] = {
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "character_count": sum(
            len(str(message.get("content", ""))) for message in messages
        ),
        "message_count": len(messages),
    }
    if include_content:
        metadata["messages"] = redact_secrets(deepcopy(messages))
    return metadata


def _git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _source_tree_sha256(repo_root: Path) -> str | None:
    """Hash tracked working-tree content, including local modifications."""
    try:
        completed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
        digest = hashlib.sha256()
        for raw_path in completed.stdout.split(b"\0"):
            if not raw_path:
                continue
            relative = raw_path.decode("utf-8", errors="surrogateescape")
            path = repo_root / relative
            if not path.is_file():
                continue
            digest.update(raw_path)
            digest.update(b"\0")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        return digest.hexdigest()
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class RunRecorder:
    """Append-only event recorder with manifest and summary snapshots."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        model: str,
        provider_config: dict[str, Any],
        suite: dict[str, Any] | None = None,
        gpu_target: str | None = None,
        run_id: str | None = None,
        dry_run: bool = False,
        repo_root: str | Path | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "run_manifest.json"
        self.events_path = self.output_dir / "events.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.run_id = run_id or uuid.uuid4().hex
        self._lock = threading.Lock()
        self._sequence = 0
        self._closed = False
        self._started_at = utc_now()
        self._summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": "dry_run" if dry_run else "running",
            "started_at": self._started_at,
            "finished_at": None,
            "llm": {
                "requests_started": 0,
                "requests_completed": 0,
                "requests_failed": 0,
                "retries": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latency_seconds": 0.0,
            },
            "cuda": {
                "compilations": 0,
                "correctness_checks": 0,
                "profiles": 0,
                "validation_status": "NOT RUN",
                "performance_results": "pending",
            },
            "tasks": {
                "total": 0,
                "by_outcome": {},
                "profiling_modes": {},
                "results": [],
            },
            "errors": 0,
        }

        root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_at": self._started_at,
            "status": "dry_run" if dry_run else "running",
            "source": {
                "git_commit": _git_commit(root),
                "tree_sha256": _source_tree_sha256(root),
            },
            "model": model,
            "provider": redact_secrets(provider_config),
            "suite": redact_secrets(suite or {}),
            "target": {"gpu": gpu_target},
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "machine": platform.machine(),
                "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            },
            "validation": {
                "cuda": "NOT RUN",
                "llm_smoke_test": "NOT RUN",
                "performance_results": "pending",
                "gates": {
                    "environment": "NOT RUN",
                    "compile": "NOT RUN",
                    "correctness": "NOT RUN",
                    "events_stability": "NOT RUN",
                    "ncu_permission": "NOT RUN",
                    "api_smoke": "NOT RUN",
                },
            },
            "budget": {
                "limits": {
                    key: provider_config.get(key)
                    for key in (
                        "max_requests",
                        "max_total_tokens",
                        "max_completion_tokens",
                        "max_concurrency",
                    )
                },
                "consumed": {
                    "requests": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            },
            "outcomes": [],
            "profiling_modes": [],
            "failure_classification": {},
        }
        _atomic_json_write(self.manifest_path, manifest)
        self.events_path.touch(exist_ok=True)
        _atomic_json_write(self.summary_path, self._summary)
        atexit.register(self.close)

    def record_event(
        self,
        event_type: str,
        *,
        status: str = "ok",
        task_id: str | int | None = None,
        rollout_id: str | int | None = None,
        stage: str | None = None,
        candidate_id: str | None = None,
        attempt: int | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            self._sequence += 1
            event = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "sequence": self._sequence,
                "timestamp": utc_now(),
                "event_type": event_type,
                "status": status,
                "task_id": task_id,
                "rollout_id": rollout_id,
                "stage": stage,
                "candidate_id": candidate_id,
                "attempt": attempt,
                "data": redact_secrets(data or {}),
            }
            with self.events_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
            self._update_summary(event_type, status, event["data"])
            _atomic_json_write(self.summary_path, self._summary)

    def close(self, status: str | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if status:
                self._summary["status"] = status
            elif self._summary["status"] == "running":
                self._summary["status"] = (
                    "completed_with_errors" if self._summary["errors"] else "completed"
                )
            self._summary["finished_at"] = utc_now()
            cuda_summary = self._summary["cuda"]
            if cuda_summary["correctness_checks"]:
                cuda_summary["validation_status"] = "RUN"
            if cuda_summary["profiles"]:
                cuda_summary["performance_results"] = "collected_pending_analysis"
            _atomic_json_write(self.summary_path, self._summary)

            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = self._summary["status"]
            manifest["finished_at"] = self._summary["finished_at"]
            if cuda_summary["correctness_checks"]:
                manifest["validation"]["cuda"] = "RUN"
            llm_summary = self._summary["llm"]
            if llm_summary["requests_failed"]:
                manifest["validation"]["llm_smoke_test"] = "FAILED"
            elif llm_summary["requests_completed"]:
                manifest["validation"]["llm_smoke_test"] = "RUN"
            if cuda_summary["profiles"]:
                manifest["validation"][
                    "performance_results"
                ] = "collected_pending_analysis"
            tasks = self._summary["tasks"]
            manifest["outcomes"] = tasks["results"]
            manifest["profiling_modes"] = sorted(tasks["profiling_modes"])
            manifest["failure_classification"] = {
                status: count
                for status, count in tasks["by_outcome"].items()
                if status in {"failed", "timeout", "blocked"}
            }
            manifest["budget"]["consumed"] = {
                # A failed HTTP attempt still consumes the request budget. The
                # provider emits one request_started event for every attempt,
                # including retries, so this is the auditable hard-cap count.
                "requests": llm_summary["requests_started"],
                "prompt_tokens": llm_summary["prompt_tokens"],
                "completion_tokens": llm_summary["completion_tokens"],
                "total_tokens": llm_summary["total_tokens"],
            }
            gates = manifest["validation"]["gates"]
            if cuda_summary["compilations"]:
                gates["compile"] = "RUN"
            if cuda_summary["correctness_checks"]:
                gates["correctness"] = "RUN"
            if "events_only" in tasks["profiling_modes"]:
                gates["events_stability"] = "RUN"
            if "ncu" in tasks["profiling_modes"]:
                gates["ncu_permission"] = "RUN"
            if llm_summary["requests_failed"]:
                gates["api_smoke"] = "FAILED"
            elif llm_summary["requests_completed"]:
                gates["api_smoke"] = "RUN"
            _atomic_json_write(self.manifest_path, manifest)

    def _update_summary(
        self,
        event_type: str,
        status: str,
        data: dict[str, Any],
    ) -> None:
        llm = self._summary["llm"]
        if event_type == "llm_request_started":
            llm["requests_started"] += 1
        elif event_type == "llm_request_completed":
            llm["requests_completed"] += 1
            usage = data.get("usage", {})
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                llm[key] += int(usage.get(key, 0) or 0)
            llm["latency_seconds"] += float(data.get("latency_seconds", 0) or 0)
        elif event_type == "llm_request_failed":
            llm["requests_failed"] += 1
        elif event_type == "llm_retry_scheduled":
            llm["retries"] += 1
        elif event_type == "cuda_compile_completed":
            self._summary["cuda"]["compilations"] += 1
        elif event_type == "cuda_correctness_completed":
            self._summary["cuda"]["correctness_checks"] += 1
        elif event_type == "cuda_profile_completed":
            self._summary["cuda"]["profiles"] += 1
        elif event_type == "task_outcome":
            tasks = self._summary["tasks"]
            outcome = str(data.get("outcome", "failed"))
            profiling_mode = str(data.get("profiling_mode", "unknown"))
            tasks["total"] += 1
            tasks["by_outcome"][outcome] = tasks["by_outcome"].get(outcome, 0) + 1
            tasks["profiling_modes"][profiling_mode] = (
                tasks["profiling_modes"].get(profiling_mode, 0) + 1
            )
            tasks["results"].append(
                {
                    "task_id": data.get("task_id"),
                    "outcome": outcome,
                    "profiling_mode": profiling_mode,
                    "reason": data.get("reason"),
                    "metrics": data.get("metrics", {}),
                }
            )

        if status == "error":
            self._summary["errors"] += 1
