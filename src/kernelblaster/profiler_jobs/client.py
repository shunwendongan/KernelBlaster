# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticated clients for Control/Profiler Worker routing."""

from __future__ import annotations

from typing import Any

import aiohttp

from .contracts import ProfileRequest, ProfileResult, ProfilerCapabilities


class ProfilerClient:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}

    async def capabilities(self) -> ProfilerCapabilities:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/v1/capabilities", headers=self.headers
            ) as response:
                response.raise_for_status()
                return ProfilerCapabilities.model_validate(await response.json())

    async def profile(self, request: ProfileRequest) -> ProfileResult:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/profiles",
                json=request.model_dump(mode="json"),
                headers=self.headers,
            ) as response:
                response.raise_for_status()
                return ProfileResult.model_validate(await response.json())


class ControlProfilerClient:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}

    async def download(self, digest: str) -> tuple[bytes, str, str, str]:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/v1/profiler/artifacts/{digest}", headers=self.headers
            ) as response:
                response.raise_for_status()
                return (
                    await response.read(),
                    response.headers["x-kernelblaster-source-digest"],
                    response.headers["x-kernelblaster-benchmark-protocol-id"],
                    response.headers.get(
                        "x-kernelblaster-artifact-kind", "executable"
                    ),
                )

    async def upload(
        self,
        payload: bytes,
        *,
        media_type: str,
        schema: str,
        source_digest: str,
    ) -> dict[str, Any]:
        headers = {
            **self.headers,
            "content-type": media_type,
            "x-kernelblaster-schema": schema,
            "x-kernelblaster-source-digest": source_digest,
        }
        async with aiohttp.ClientSession() as session:
            async with session.put(
                f"{self.base_url}/v1/profiler/artifacts", data=payload, headers=headers
            ) as response:
                response.raise_for_status()
                return await response.json()


__all__ = ["ControlProfilerClient", "ProfilerClient"]
