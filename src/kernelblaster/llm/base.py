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

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    """Provider-neutral response consumed by the existing optimization agents."""

    input_messages: list[dict]
    generations: list[str]
    usage: dict
    model: str
    num_tasks: int
    elapsed_time: float
    provider: str = ""
    request_ids: list[str] = field(default_factory=list)
    attempts: int = 0
    usage_source: str = "provider"

    def __str__(self) -> str:
        return str(asdict(self))

    @property
    def response(self) -> str:
        """Return the first generation for legacy single-response callers."""
        return self.generations[0] if self.generations else ""


class LLMProvider(ABC):
    """Minimal asynchronous interface for code-generation providers."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, Any]],
        model: str,
        n: int = 1,
    ) -> LLMResponse:
        """Generate ``n`` candidates for one conversation."""

    @abstractmethod
    def public_config(self) -> dict[str, Any]:
        """Return non-secret configuration suitable for manifests and logs."""


class LLMConfigurationError(RuntimeError):
    """Raised when an LLM provider is missing required configuration."""


class LLMBudgetExceeded(RuntimeError):
    """Raised before a request that would exceed a configured hard budget."""
