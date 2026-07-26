# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-request, zero-retry Provider authentication probe."""

from __future__ import annotations

from typing import Any, Awaitable, Callable


def build_provider_auth_probe(
    *,
    base_url: str,
    api_key: str,
    model: str,
    timeout_seconds: float = 180,
) -> Callable[[], Awaitable[dict[str, Any]]]:
    if not api_key:
        raise ValueError("Provider API key is required")

    async def probe() -> dict[str, Any]:
        # Keep the optional OpenAI dependency outside CPU-only preflight imports.
        from ..llm import OpenAICompatibleProvider, OpenAICompatibleSettings

        provider = OpenAICompatibleProvider(
            OpenAICompatibleSettings(
                base_url=base_url,
                api_key=api_key,
                timeout_seconds=timeout_seconds,
                max_concurrency=1,
                max_retries=0,
                max_requests=1,
                max_total_tokens=64,
                max_completion_tokens=64,
                reasoning_effort="none",
                stream=False,
                log_content=False,
            )
        )
        response = await provider.generate(
            [
                {
                    "role": "user",
                    "content": (
                        "Return exactly the single word KERNELBLASTER_OK and no "
                        "additional text."
                    ),
                }
            ],
            model=model,
            n=1,
        )
        if response.response.strip() != "KERNELBLASTER_OK":
            raise RuntimeError("Provider authentication probe returned unexpected content")
        return {
            "provider": response.provider,
            "response_model": (
                response.response_models[0] if response.response_models else response.model
            ),
            "usage": response.usage,
            "attempts": response.attempts,
        }

    return probe


__all__ = ["build_provider_auth_probe"]
