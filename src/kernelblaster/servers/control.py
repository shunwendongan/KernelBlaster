# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal CPU-only control service for the Compose security boundary.

Workflow submission and job execution remain intentionally out of this service
until the dedicated Job API is introduced.  This process proves the control
audience, secret boundary, and worker connectivity without starting CUDA tools.
"""

from __future__ import annotations

import argparse

import uvicorn
from fastapi import Depends, FastAPI

from .auth import require_control_token, validate_token_boundaries


app = FastAPI(title="KernelBlaster Control")


@app.get("/health")
async def health(_authorized: None = Depends(require_control_token)) -> dict[str, str]:
    return {"status": "ok", "service": "control"}


@app.get("/ready")
async def ready(_authorized: None = Depends(require_control_token)) -> dict[str, str]:
    """Report the authenticated readiness boundary without starting a workflow."""
    return {"status": "ready", "service": "control"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CPU-only control boundary")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    validate_token_boundaries()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
