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
import os
import secrets
from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
from urllib.parse import urlsplit, urlunsplit

from .gpu_config import GPUType
from typing import Any

load_dotenv()


def _public_url(value: str) -> str:
    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    if parsed.port:
        hostname = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path.rstrip("/"), "", ""))


class ExperimentalFeatures:
    OPT_RL_NCU = os.getenv("KERNELBLASTER_OPT_RL_NCU", "0") == "1"

    @staticmethod
    def dict() -> dict[str, bool | int]:
        # return a dict of the experimental features and their values
        return {
            k: v
            for k, v in ExperimentalFeatures.__dict__.items()
            if not k.startswith("_") and not callable(v)
        }


class SystemConfig:
    # Pluggable remote LLM provider. Secrets are intentionally environment-only.
    LLM_PROVIDER = os.getenv(
        "KERNELBLASTER_LLM_PROVIDER", "openai_compatible"
    ).strip().lower()
    LLM_BASE_URL = os.getenv(
        "KERNELBLASTER_LLM_BASE_URL", "https://api.openai.com/v1"
    ).strip()
    LLM_BASE_URL_PUBLIC = _public_url(LLM_BASE_URL)
    API_KEY = os.getenv("KERNELBLASTER_LLM_API_KEY") or os.getenv(
        "OPENAI_API_KEY", ""
    )
    MODEL = os.getenv("MODEL", "gpt-4.1-20250414")
    STREAM = os.getenv("STREAM", "False")
    LLM_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "4"))
    LLM_REQUEST_TIMEOUT_SECONDS = float(
        os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "1800")
    )
    LLM_MAX_REQUESTS = int(os.getenv("LLM_MAX_REQUESTS", "0"))
    LLM_MAX_TOTAL_TOKENS = int(os.getenv("LLM_MAX_TOTAL_TOKENS", "0"))
    LLM_MAX_COMPLETION_TOKENS = int(
        os.getenv("LLM_MAX_COMPLETION_TOKENS", "12288")
    )
    LLM_REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "").strip().lower()
    LLM_LOG_CONTENT = os.getenv("LLM_LOG_CONTENT", "false").lower() in (
        "true",
        "1",
        "yes",
        "y",
        "on",
    )
    WORKER_TOKEN = os.getenv("KERNELBLASTER_WORKER_TOKEN") or secrets.token_urlsafe(32)
    MAX_GPU_BINARY_BYTES = int(
        os.getenv("KERNELBLASTER_MAX_GPU_BINARY_BYTES", str(256 * 1024 * 1024))
    )
    MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", 300))
    NUM_PARALLEL_GENERATIONS_PER_ATTEMPT = int(
        os.getenv("NUM_PARALLEL_GENERATIONS_PER_ATTEMPT", 4)
    )

    # Server URLs
    COMPILE_SERVER_URL = os.getenv("COMPILE_SERVER_URL", None)
    GPU_SERVER_URLS = {
        gpu: os.getenv(f"GPU_SERVER_URL_{gpu.value.upper()}", None) for gpu in GPUType
    }

    assert (
        COMPILE_SERVER_URL is None or "http" in COMPILE_SERVER_URL
    ), "COMPILE_SERVER_URL must be unset or start with http"
    for __gpu, __url in GPU_SERVER_URLS.items():
        assert (
            __url is None or "http" in __url
        ), f"GPU_SERVER_URL_{__gpu.value.upper()} must be unset or start with http but got {__url}"

    LLM_GATEWAY_CLIENT_ID = os.getenv("LLM_GATEWAY_CLIENT_ID", None)
    LLM_GATEWAY_CLIENT_SECRET = os.getenv("LLM_GATEWAY_CLIENT_SECRET", None)
    LLM_GATEWAY_BASE_URL = os.getenv("LLM_GATEWAY_BASE_URL", None)
    LLM_GATEWAY_TOKEN_URL = os.getenv(
        "LLM_GATEWAY_TOKEN_URL",
        f"{LLM_GATEWAY_BASE_URL}/token" if LLM_GATEWAY_BASE_URL else None
    )

    NVCF_FUNCTION_ID = os.getenv("NVCF_FUNCTION_ID", None)
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

    TEMP_DIRECTORY = "/tmp/kernelblaster"
    EOS_BASE_URL = os.getenv("EOS_BASE_URL", None)

    # Number of speedups the NCU agent must get over the original cuda-c baseline
    # before it's considered a success.
    OPT_NCU_MIN_IMPROVEMENTS = int(os.getenv("KERNELBLASTER_NCU_MIN_IMPROVEMENTS", "3"))

    EXPERIMENTAL_FEATURES = ExperimentalFeatures()

    @staticmethod
    def set_compile_server_url(url: str):
        assert (
            SystemConfig.COMPILE_SERVER_URL is None
        ), "COMPILE_SERVER_URL is already set"
        SystemConfig.COMPILE_SERVER_URL = url

    @staticmethod
    def set_gpu_server_url(gpu: GPUType, url: str):
        if (
            gpu not in SystemConfig.GPU_SERVER_URLS
            or SystemConfig.GPU_SERVER_URLS[gpu] is None
        ):
            SystemConfig.GPU_SERVER_URLS[gpu] = url
        else:
            logger.warning(
                f"GPU_SERVER_URL_{gpu.value.upper()} is already set to {SystemConfig.GPU_SERVER_URLS[gpu]}"
            )

    @staticmethod
    def get_gpu_server_url(gpu: GPUType) -> str:
        assert (
            gpu in SystemConfig.GPU_SERVER_URLS
        ), f"GPU_SERVER_URL_{gpu.value.upper()} is not set"
        return SystemConfig.GPU_SERVER_URLS[gpu]

    @staticmethod
    def get_all_gpu_server_urls() -> dict[GPUType, str]:
        return {
            gpu: url
            for gpu, url in SystemConfig.GPU_SERVER_URLS.items()
            if url is not None
        }

    @staticmethod
    def CUSTOM_LOGGER_FORMAT(record):
        attempt_id = record["extra"].get("attempt_id")
        if attempt_id is None:
            return (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<green>{extra[agent_name]}</green> - <level>{message}</level> "
                "\n{exception}"
            )
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<green>{extra[agent_name]}</green> | "
            "<green>Attempt {extra[attempt_id]} Task {extra[task_id]}</green> - <level>{message}</level>"
            "\n{exception}"
        )

    @classmethod
    def print_config(cls, logger):
        gpu_servers_strs = "\n".join(
            [
                f"    - {gpu.value.upper()}: {url}"
                for gpu, url in cls.GPU_SERVER_URLS.items()
            ]
        )
        """Print the various hyperparameters to the logger."""
        config_str = f"""
- Using {cls.MODEL} for generation
- LLM provider: {cls.LLM_PROVIDER}
- LLM endpoint: {cls.LLM_BASE_URL_PUBLIC}
- LLM API key configured: {bool(cls.API_KEY)}
- LLM max concurrency: {cls.LLM_MAX_CONCURRENCY}
- LLM max retries: {cls.LLM_MAX_RETRIES}
- LLM request budget: {cls.LLM_MAX_REQUESTS or 'unlimited'}
- LLM token budget: {cls.LLM_MAX_TOTAL_TOKENS or 'unlimited'}
- LLM max completion tokens: {cls.LLM_MAX_COMPLETION_TOKENS}
- LLM reasoning effort: {cls.LLM_REASONING_EFFORT or 'provider default'}
- Parallel generations per attempt: {cls.NUM_PARALLEL_GENERATIONS_PER_ATTEMPT}
- Maximum attempts: {cls.MAX_ATTEMPTS}
- Compiler server: {cls.COMPILE_SERVER_URL}
- GPU servers:
{cls.get_all_gpu_server_urls()}
- Experimental features: {cls.EXPERIMENTAL_FEATURES.dict()}
"""
        logger.info(f"Config:\n{config_str}")


config = SystemConfig()


@dataclass
class WorkflowConfig:
    model: str
    run_cuda: bool
    run_cuda_perf: bool
    run_cuda_bench: bool
    run_cuda_perf_bench: bool
    retry_failed: bool
    gpu: GPUType
    # RL optimization parameters
    rl_iterations: int = 10
    rl_rollout_steps: int = 5
    rl_buffer_size: int = 100
    rl_update_frequency: int = 3
    use_baseline_optimization: bool = True
    # Optional shared OptimizationDatabase instance (not serialized)
    shared_optimization_database: Any = None

    def dict(self):
        # Do not use dataclasses.asdict(): it deep-copies values before they can
        # be removed, and the shared database intentionally owns an RLock.
        return {
            name: value
            for name, value in vars(self).items()
            if name != "shared_optimization_database"
        }

    def should_skip_folder(self, folder: Path):
        """Check if we should skip processing this problem based on existing files"""

        if not folder.exists():
            return False

        files = os.listdir(folder)

        def should_skip(name: str, folder_key: str, enabled_field: str):
            # The files are created by the agents themselves
            agent_enabled = getattr(self, enabled_field)
            return (
                not agent_enabled
                or f"final_{name}.cu" in files
                or (
                    f"failed_{name}" in files
                    and (Path(folder) / folder_key / ".finished").exists()
                    and not self.retry_failed
                )
            )

        return should_skip("rl_cuda_perf", "rl_ncu", "run_cuda_perf")


    def validate(self):
        assert isinstance(
            self.gpu, GPUType
        ), f"Please ensure gpu is a GPUType: got {type(self.gpu)}"
