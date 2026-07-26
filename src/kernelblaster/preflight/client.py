# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticated Agent-side client for the Control API."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import aiohttp


TERMINAL_JOB_STATUSES = {
    "succeeded",
    "failed",
    "blocked",
    "timed_out",
    "cancelled",
}


class ControlApiError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"Control API HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class ControlPlaneClient:
    def __init__(self, base_url: str, token: str, *, request_timeout: float = 30) -> None:
        if not token.strip():
            raise ValueError("Control token is required")
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}
        self.request_timeout = request_timeout

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
        payload: bytes | None = None,
        media_type: str = "application/json",
    ) -> tuple[bytes, dict[str, str]]:
        headers = {**self.headers, "content-type": media_type}
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                method,
                self.base_url + path,
                json=json_payload,
                data=payload,
                headers=headers,
            ) as response:
                body = await response.read()
                if response.status >= 400:
                    detail = body.decode("utf-8", errors="replace")[:2048]
                    raise ControlApiError(response.status, detail)
                return body, {key.lower(): value for key, value in response.headers.items()}

    async def request_json(
        self,
        method: str,
        path: str,
        *,
        json_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body, _headers = await self._request(
            method,
            path,
            json_payload=json_payload,
        )
        decoded = json.loads(body)
        if not isinstance(decoded, dict):
            raise ValueError("Control API response must be an object")
        return decoded

    async def create_run(self, run_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        return await self.request_json(
            "POST",
            "/v1/runs",
            json_payload={"run_id": run_id, "metadata": metadata},
        )

    async def get_run(self, run_id: str) -> dict[str, Any]:
        return await self.request_json("GET", f"/v1/runs/{run_id}")

    async def upload(
        self,
        payload: bytes,
        *,
        media_type: str,
        schema: str | None = None,
    ) -> dict[str, Any]:
        path = "/v1/control/artifacts"
        headers = {**self.headers, "content-type": media_type}
        if schema:
            headers["x-kernelblaster-schema"] = schema
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.put(
                self.base_url + path,
                data=payload,
                headers=headers,
            ) as response:
                body = await response.read()
                if response.status >= 400:
                    raise ControlApiError(
                        response.status,
                        body.decode("utf-8", errors="replace")[:2048],
                    )
                decoded = json.loads(body)
                if not isinstance(decoded, dict):
                    raise ValueError("artifact upload response must be an object")
                return decoded

    async def download(self, digest: str) -> bytes:
        body, _headers = await self._request("GET", f"/v1/artifacts/{digest}")
        return body

    async def gpu_capabilities(self) -> dict[str, Any]:
        return await self.request_json("GET", "/v1/gpu/capabilities")

    async def profiler_capabilities(self) -> dict[str, Any]:
        return await self.request_json("GET", "/v1/profiler/capabilities")

    async def submit_gpu_job(self, manifest: dict[str, Any]) -> dict[str, Any]:
        return await self.request_json("POST", "/v1/gpu/jobs", json_payload=manifest)

    async def get_gpu_job(self, job_id: str) -> dict[str, Any]:
        return await self.request_json("GET", f"/v1/gpu/jobs/{job_id}")

    async def cancel_gpu_job(self, job_id: str) -> dict[str, Any]:
        return await self.request_json("POST", f"/v1/gpu/jobs/{job_id}/cancel")

    async def wait_gpu_job(
        self,
        job_id: str,
        *,
        timeout_seconds: float,
        poll_seconds: float = 0.25,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            job = await self.get_gpu_job(job_id)
            if str(job.get("status")) in TERMINAL_JOB_STATUSES:
                return job
            await asyncio.sleep(poll_seconds)
        try:
            await self.cancel_gpu_job(job_id)
        finally:
            raise TimeoutError(f"GPU Job {job_id} exceeded the preflight deadline")

    async def profile(self, request: dict[str, Any]) -> dict[str, Any]:
        return await self.request_json("POST", "/v1/profiles", json_payload=request)


__all__ = ["ControlApiError", "ControlPlaneClient", "TERMINAL_JOB_STATUSES"]
