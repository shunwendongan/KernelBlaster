from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_trusted_pilot", ROOT / "scripts" / "run_trusted_pilot.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_missing_optional_command_is_recorded_instead_of_crashing(tmp_path):
    log_path = tmp_path / "missing.log"

    completed = MODULE._run(
        ["kernelblaster-command-that-does-not-exist"],
        log_path=log_path,
        timeout=1,
    )

    assert completed.returncode == 127
    assert "FileNotFoundError" in completed.stderr
    assert "kernelblaster-command-that-does-not-exist" in log_path.read_text(
        encoding="utf-8"
    )
