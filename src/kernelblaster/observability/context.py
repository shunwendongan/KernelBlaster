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

from contextvars import ContextVar

from .recorder import RunRecorder


_active_recorder: ContextVar[RunRecorder | None] = ContextVar(
    "kernelblaster_run_recorder", default=None
)


def get_run_recorder() -> RunRecorder | None:
    return _active_recorder.get()


def set_run_recorder(recorder: RunRecorder | None) -> None:
    _active_recorder.set(recorder)


def record_event(event_type: str, **kwargs) -> None:
    recorder = get_run_recorder()
    if recorder is not None:
        recorder.record_event(event_type, **kwargs)
