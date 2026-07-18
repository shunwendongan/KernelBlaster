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

import asyncio
from copy import deepcopy
from dataclasses import dataclass
import math
import random
import re
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import openai
from loguru import logger

from .base import LLMBudgetExceeded, LLMConfigurationError, LLMProvider, LLMResponse


@dataclass(frozen=True)
class OpenAICompatibleSettings:
    base_url: str
    api_key: str
    timeout_seconds: float = 1800
    max_concurrency: int = 4
    max_retries: int = 3
    max_requests: int = 0
    max_total_tokens: int = 0
    stream: bool = False

    def validate(self) -> None:
        if not self.api_key:
            raise LLMConfigurationError(
                "No API key configured. Set KERNELBLASTER_LLM_API_KEY or OPENAI_API_KEY."
            )
        if not self.base_url.startswith(("http://", "https://")):
            raise LLMConfigurationError(
                "KERNELBLASTER_LLM_BASE_URL must start with http:// or https://."
            )
        if self.max_concurrency < 1:
            raise LLMConfigurationError("LLM_MAX_CONCURRENCY must be at least 1.")
        if self.max_retries < 0:
            raise LLMConfigurationError("LLM_MAX_RETRIES cannot be negative.")
        if self.max_requests < 0 or self.max_total_tokens < 0:
            raise LLMConfigurationError("LLM budgets cannot be negative.")


@dataclass
class _CandidateResult:
    content: str
    usage: dict[str, Any]
    request_id: str
    attempts: int
    elapsed_time: float
    usage_source: str


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return dict(value)
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def _estimate_tokens(text: str) -> int:
    """Use a deterministic tokenizer-free estimate for gateway responses."""
    return max(1, math.ceil(len(text) / 4)) if text else 0


def _sanitize_base_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    hostname = parsed.hostname or ""
    if parsed.port:
        hostname = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path.rstrip("/"), "", ""))


def _merge_usage(items: list[dict[str, Any]]) -> dict[str, int]:
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {
        key: sum(int(item.get(key, 0) or 0) for item in items)
        for key in keys
    }


class OpenAICompatibleProvider(LLMProvider):
    """Chat Completions provider with client-side candidate fan-out."""

    name = "openai_compatible"

    def __init__(
        self,
        settings: OpenAICompatibleSettings,
        client: openai.AsyncOpenAI | openai.AsyncAzureOpenAI | None = None,
    ) -> None:
        settings.validate()
        self.settings = settings
        self._client = client or openai.AsyncOpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key,
            timeout=settings.timeout_seconds,
            max_retries=0,
        )
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        self._budget_lock = asyncio.Lock()
        self._request_count = 0
        self._total_tokens = 0

    def public_config(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "base_url": _sanitize_base_url(self.settings.base_url),
            "timeout_seconds": self.settings.timeout_seconds,
            "max_concurrency": self.settings.max_concurrency,
            "max_retries": self.settings.max_retries,
            "max_requests": self.settings.max_requests,
            "max_total_tokens": self.settings.max_total_tokens,
            "stream": self.settings.stream,
            "fanout_mode": "client",
            "api_key_configured": bool(self.settings.api_key),
        }

    async def generate(
        self,
        messages: list[dict[str, Any]],
        model: str,
        n: int = 1,
    ) -> LLMResponse:
        if not model:
            raise LLMConfigurationError("A model identifier is required.")
        if n < 1:
            raise ValueError("n must be at least 1.")

        started = time.monotonic()
        tasks = [
            asyncio.create_task(self._generate_candidate(messages, model))
            for _ in range(n)
        ]
        try:
            candidates = await asyncio.gather(*tasks)
        except Exception:
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise

        usage_sources = {candidate.usage_source for candidate in candidates}
        return LLMResponse(
            input_messages=deepcopy(messages),
            generations=[candidate.content for candidate in candidates],
            usage=_merge_usage([candidate.usage for candidate in candidates]),
            model=model,
            num_tasks=n,
            elapsed_time=time.monotonic() - started,
            provider=self.name,
            request_ids=[
                candidate.request_id
                for candidate in candidates
                if candidate.request_id
            ],
            attempts=sum(candidate.attempts for candidate in candidates),
            usage_source=(usage_sources.pop() if len(usage_sources) == 1 else "mixed"),
        )

    async def _generate_candidate(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> _CandidateResult:
        started = time.monotonic()
        last_error: Exception | None = None

        for attempt in range(self.settings.max_retries + 1):
            await self._reserve_request()
            try:
                async with self._semaphore:
                    response = await self._create_completion(messages, model)
                content, usage, request_id, usage_source = self._normalize_response(
                    response, messages
                )
                await self._record_tokens(int(usage.get("total_tokens", 0) or 0))
                return _CandidateResult(
                    content=content,
                    usage=usage,
                    request_id=request_id,
                    attempts=attempt + 1,
                    elapsed_time=time.monotonic() - started,
                    usage_source=usage_source,
                )
            except Exception as error:
                last_error = error
                if attempt >= self.settings.max_retries or not self._is_retryable(error):
                    raise
                delay = min(2**attempt, 30) + random.uniform(0, 1)
                logger.warning(
                    "OpenAI-compatible request failed on attempt {}/{}; retrying in {:.2f}s ({})",
                    attempt + 1,
                    self.settings.max_retries + 1,
                    delay,
                    type(error).__name__,
                )
                await asyncio.sleep(delay)

        raise RuntimeError("OpenAI-compatible request failed") from last_error

    async def _create_completion(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> Any:
        args: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": self.settings.stream,
            "n": 1,
        }
        request_model = model

        if re.search(r"claude-.*-thinking", model.lower()):
            args["extra_body"] = {
                "thinking": {"type": "enabled", "budget_tokens": 16384}
            }
            request_model = model.replace("-thinking", "")
        elif re.search(r"deepseek-r1", model.lower()):
            args["top_p"] = 0.95
            args["max_tokens"] = 12288
            args["extra_body"] = {"enable_thinking": True}
        elif re.search("qwen", model.lower()) or re.search("kevin", model.lower()):
            args["top_p"] = 0.9
            args["max_tokens"] = 12288
            args["extra_body"] = {"enable_thinking": True}
        elif re.search(r".*-nemotron-.*-thinking", model.lower()):
            request_model = model.replace("-thinking", "")
            args["max_tokens"] = 12288

        args["model"] = request_model
        if not self.settings.stream:
            return await self._client.chat.completions.create(**args)

        chunks = []
        stream = await self._client.chat.completions.create(**args)
        async for chunk in stream:
            chunks.append(chunk)
        return chunks

    def _normalize_response(
        self,
        response: Any,
        messages: list[dict[str, Any]],
    ) -> tuple[str, dict[str, Any], str, str]:
        if isinstance(response, list):
            content_parts: list[str] = []
            usage: dict[str, Any] = {}
            request_id = ""
            for chunk in response:
                request_id = request_id or str(getattr(chunk, "id", "") or "")
                chunk_usage = _to_plain_dict(getattr(chunk, "usage", None))
                if chunk_usage:
                    usage = chunk_usage
                for choice in getattr(chunk, "choices", []) or []:
                    delta = getattr(choice, "delta", None)
                    text = getattr(delta, "content", None) if delta else None
                    if text:
                        content_parts.append(text)
            content = "".join(content_parts)
        else:
            choices = getattr(response, "choices", []) or []
            if not choices:
                raise RuntimeError("The provider returned no completion choices.")
            content = getattr(choices[0].message, "content", None) or ""
            usage = _to_plain_dict(getattr(response, "usage", None))
            request_id = str(getattr(response, "id", "") or "")

        if usage:
            usage_source = "provider"
        else:
            prompt_text = "".join(
                str(message.get("content", "")) for message in messages
            )
            prompt_tokens = _estimate_tokens(prompt_text)
            completion_tokens = _estimate_tokens(content)
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            usage_source = "estimated"

        return content, usage, request_id, usage_source

    async def _reserve_request(self) -> None:
        async with self._budget_lock:
            if (
                self.settings.max_requests
                and self._request_count >= self.settings.max_requests
            ):
                raise LLMBudgetExceeded(
                    f"LLM request budget exhausted ({self.settings.max_requests})."
                )
            if (
                self.settings.max_total_tokens
                and self._total_tokens >= self.settings.max_total_tokens
            ):
                raise LLMBudgetExceeded(
                    "LLM token budget exhausted "
                    f"({self.settings.max_total_tokens})."
                )
            self._request_count += 1

    async def _record_tokens(self, total_tokens: int) -> None:
        async with self._budget_lock:
            self._total_tokens += max(0, total_tokens)

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        if isinstance(
            error,
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ),
        ):
            return True
        if isinstance(error, openai.APIStatusError):
            return error.status_code in (408, 409, 429) or error.status_code >= 500
        return False
