"""Authenticated clients for Control/Baseline Worker routing."""

from __future__ import annotations

from typing import Any

import aiohttp

from .contracts import BaselineCapabilities, BaselineRequest, BaselineResult


class BaselineClient:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}

    async def capabilities(self) -> BaselineCapabilities:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/v1/capabilities", headers=self.headers
            ) as response:
                response.raise_for_status()
                return BaselineCapabilities.model_validate(await response.json())

    async def evaluate(self, request: BaselineRequest) -> BaselineResult:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/baselines",
                json=request.model_dump(mode="json"),
                headers=self.headers,
            ) as response:
                response.raise_for_status()
                return BaselineResult.model_validate(await response.json())


class ControlBaselineClient:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}

    async def download(self, digest: str) -> bytes:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/v1/baseline/artifacts/{digest}", headers=self.headers
            ) as response:
                response.raise_for_status()
                return await response.read()

    async def upload(self, payload: bytes, *, schema: str) -> dict[str, Any]:
        headers = {
            **self.headers,
            "content-type": "application/json",
            "x-kernelblaster-schema": schema,
        }
        async with aiohttp.ClientSession() as session:
            async with session.put(
                f"{self.base_url}/v1/baseline/artifacts", data=payload, headers=headers
            ) as response:
                response.raise_for_status()
                return await response.json()


__all__ = ["BaselineClient", "ControlBaselineClient"]
