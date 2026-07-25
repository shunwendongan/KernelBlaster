# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.kernelblaster.config import config
from src.kernelblaster.servers import control
from src.kernelblaster.servers.auth import (
    require_control_token,
    require_worker_token,
    validate_token_boundaries,
)
from src.kernelblaster.servers.serve_api import app as legacy_workflow_app


def _set_distinct_tokens(monkeypatch) -> None:
    monkeypatch.setattr(config, "CONTROL_TOKEN", "control-token")
    monkeypatch.setattr(config, "WORKER_TOKEN", "worker-token")


def test_token_configuration_requires_both_distinct_audiences(monkeypatch):
    _set_distinct_tokens(monkeypatch)
    assert validate_token_boundaries() is None

    monkeypatch.setattr(config, "CONTROL_TOKEN", "")
    with pytest.raises(RuntimeError, match="KERNELBLASTER_CONTROL_TOKEN"):
        validate_token_boundaries()

    monkeypatch.setattr(config, "CONTROL_TOKEN", "same-token")
    monkeypatch.setattr(config, "WORKER_TOKEN", "same-token")
    with pytest.raises(RuntimeError, match="different values"):
        validate_token_boundaries()


@pytest.mark.asyncio
async def test_token_audiences_are_not_interchangeable(monkeypatch):
    _set_distinct_tokens(monkeypatch)
    assert await require_control_token("Bearer control-token") is None
    assert await require_worker_token("Bearer worker-token") is None
    with pytest.raises(HTTPException) as control_rejection:
        await require_control_token("Bearer worker-token")
    with pytest.raises(HTTPException) as worker_rejection:
        await require_worker_token("Bearer control-token")
    assert control_rejection.value.status_code == 401
    assert worker_rejection.value.status_code == 401


def test_control_exposes_only_authenticated_health_boundaries(monkeypatch):
    _set_distinct_tokens(monkeypatch)
    client = TestClient(control.app)
    assert client.get("/health").status_code == 401
    assert client.get("/ready", headers={"Authorization": "Bearer worker-token"}).status_code == 401
    response = client.get("/ready", headers={"Authorization": "Bearer control-token"})
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "service": "control"}
    assert {route.path for route in control.app.routes} >= {"/health", "/ready"}
    assert "/submit" not in {route.path for route in control.app.routes}


def test_legacy_workflow_routes_use_the_control_audience():
    protected = {"/submit", "/status/{task_id}", "/cancel/{task_id}"}
    for route in legacy_workflow_app.routes:
        if route.path in protected:
            assert route.dependant.dependencies[0].call is require_control_token
