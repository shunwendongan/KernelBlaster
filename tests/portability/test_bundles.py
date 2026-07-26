from __future__ import annotations

import io
from pathlib import Path
import tarfile

import pytest

from src.kernelblaster.portability.exporter import export_run
from src.kernelblaster.portability.importer import import_run
from src.kernelblaster.storage import StateStore


def _store(tmp_path: Path, name: str) -> StateStore:
    return StateStore(state_dir=tmp_path / name)


def _completed_run(store: StateStore, run_id: str = "portable-run") -> str:
    store.repository.create_run(run_id, metadata={"suite": "portable-unit"})
    store.repository.bind_run_portability(
        run_id=run_id,
        source_instance_id=store.instance_identity.instance_id,
        target_id="local",
        target_arch="sm_999",
        audit_fingerprint="sha256:audit",
        comparison_group="sha256:group",
    )
    artifact = store.cas.put_bytes(b"portable payload", media_type="text/plain", producer="test")
    store.repository.register_artifact(artifact)
    store.repository.link_run_artifact(run_id=run_id, digest=artifact.digest, role="summary")
    store.repository.finish_run(run_id, "completed")
    return artifact.digest


def test_export_import_is_deduplicated_and_idempotent(tmp_path: Path) -> None:
    source = _store(tmp_path, "source")
    digest = _completed_run(source)
    bundle = tmp_path / "portable-run.tar"

    exported = export_run(source, "portable-run", bundle)
    target = _store(tmp_path, "target")
    first = import_run(target, bundle)
    second = import_run(target, bundle)

    assert exported.payload_count == 1
    assert first.idempotent is False
    assert second.idempotent is True
    assert target.repository.get_run("portable-run")["status"] == "completed"
    assert target.cas.get_bytes(digest) == b"portable payload"
    portability = target.repository.get_run_portability("portable-run")
    assert portability is not None
    assert portability["content_hash"] == first.content_hash


def test_import_rejects_path_traversal_before_state_mutation(tmp_path: Path) -> None:
    target = _store(tmp_path, "target")
    bundle = tmp_path / "malicious.tar"
    with tarfile.open(bundle, "w") as archive:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))

    with pytest.raises(ValueError, match="safe relative"):
        import_run(target, bundle)
    assert target.repository.list_runs() == []


def test_same_run_id_with_different_content_is_rejected(tmp_path: Path) -> None:
    source = _store(tmp_path, "source")
    _completed_run(source)
    original = tmp_path / "original.tar"
    export_run(source, "portable-run", original)
    target = _store(tmp_path, "target")
    import_run(target, original)

    source.cas.put_bytes(b"new payload")
    source.repository.create_run("second")
    source.repository.finish_run("second", "completed")
    # Re-exporting a modified snapshot with the original run id simulates a
    # malicious/conflicting remote run without depending on archive internals.
    with source.repository._transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE runs SET metadata_json = ? WHERE id = ?", ('{"suite":"different"}', "portable-run")
        )
    conflicting = tmp_path / "conflicting.tar"
    export_run(source, "portable-run", conflicting)

    with pytest.raises(ValueError, match="different portable content"):
        import_run(target, conflicting)
