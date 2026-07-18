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

from typing import TYPE_CHECKING

from .base import LLMConfigurationError, LLMProvider
from .openai_compatible import OpenAICompatibleProvider, OpenAICompatibleSettings

if TYPE_CHECKING:
    from ..config.config import SystemConfig


_provider: LLMProvider | None = None


def get_llm_provider(config: type[SystemConfig]) -> LLMProvider:
    """Return the process-wide provider configured before program startup."""
    global _provider
    if _provider is not None:
        return _provider

    if config.LLM_PROVIDER != "openai_compatible":
        raise LLMConfigurationError(
            f"Unsupported KERNELBLASTER_LLM_PROVIDER: {config.LLM_PROVIDER}"
        )

    _provider = OpenAICompatibleProvider(
        OpenAICompatibleSettings(
            base_url=config.LLM_BASE_URL,
            api_key=config.API_KEY,
            timeout_seconds=config.LLM_REQUEST_TIMEOUT_SECONDS,
            max_concurrency=config.LLM_MAX_CONCURRENCY,
            max_retries=config.LLM_MAX_RETRIES,
            max_requests=config.LLM_MAX_REQUESTS,
            max_total_tokens=config.LLM_MAX_TOTAL_TOKENS,
            stream=config.STREAM.lower() in ("true", "1", "yes", "y", "on"),
            log_content=config.LLM_LOG_CONTENT,
        )
    )
    return _provider


def reset_llm_provider() -> None:
    """Clear the singleton so future isolated tests can rebuild configuration."""
    global _provider
    _provider = None
