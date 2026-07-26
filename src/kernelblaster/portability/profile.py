"""TOML portability profiles with explicit, testable source precedence."""

from __future__ import annotations

import os
from pathlib import Path
import tomllib
from typing import Any, Mapping


PROFILE_ENV_PREFIX = "KERNELBLASTER_PROFILE_"


def load_profile(
    path: str | Path | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load TOML, then apply environment and explicit CLI-style overrides.

    Only public values are returned; secret Provider values intentionally remain
    in their dedicated environment variables and never enter profiles.
    """
    values: dict[str, Any] = {}
    if path is not None:
        with Path(path).expanduser().open("rb") as stream:
            payload = tomllib.load(stream)
        values.update(dict(payload.get("portability") or payload))
    environment = environment or os.environ
    for key, value in environment.items():
        if key.startswith(PROFILE_ENV_PREFIX) and value:
            values[key[len(PROFILE_ENV_PREFIX) :].lower()] = value
    values.update({key: value for key, value in (overrides or {}).items() if value is not None})
    return values
