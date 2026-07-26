"""SQLite repository for control-owned runs, jobs, leases, and artifact indexes."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator
import uuid

from .cas import ArtifactMetadata


TERMINAL_JOB_STATUSES = {"succeeded", "failed", "blocked", "timed_out", "cancelled"}
ACTIVE_JOB_STATUSES = {"leased", "running"}
ALL_JOB_STATUSES = {"pending", *ACTIVE_JOB_STATUSES, *TERMINAL_JOB_STATUSES}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime | None = None) -> str:
    return (value or _utc_now()).isoformat().replace("+00:00", "Z")


def _json(value: dict[str, Any] | None) -> str:
    return json.dumps(value or {}, sort_keys=True, separators=(",", ":"))


class JobRepository:
    """A short-lived-connection SQLite repository safe for a single control process."""

    def __init__(self, database_path: str | Path, *, busy_timeout_ms: int = 5_000) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.busy_timeout_ms = busy_timeout_ms
        self.migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=self.busy_timeout_ms / 1000)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        return connection

    @contextmanager
    def _transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def migrate(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )"""
            )
            applied = {
                row["version"] for row in connection.execute("SELECT version FROM schema_migrations")
            }
            if 1 not in applied:
                connection.executescript(
                    """
                    CREATE TABLE runs (
                        id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        finished_at TEXT,
                        metadata_json TEXT NOT NULL
                    );
                    CREATE TABLE jobs (
                        id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                        idempotency_key TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        status TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        terminal_reason TEXT,
                        UNIQUE(run_id, idempotency_key)
                    );
                    CREATE TABLE job_attempts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        ordinal INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        worker_id TEXT,
                        started_at TEXT NOT NULL,
                        finished_at TEXT,
                        result_json TEXT,
                        UNIQUE(job_id, ordinal)
                    );
                    CREATE TABLE leases (
                        id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        attempt_id INTEGER NOT NULL REFERENCES job_attempts(id) ON DELETE CASCADE,
                        worker_id TEXT NOT NULL,
                        acquired_at TEXT NOT NULL,
                        renewed_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        released_at TEXT,
                        release_reason TEXT
                    );
                    CREATE UNIQUE INDEX one_active_lease_per_job
                    ON leases(job_id) WHERE released_at IS NULL;
                    CREATE TABLE artifacts (
                        digest TEXT PRIMARY KEY,
                        media_type TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        producer TEXT,
                        source_digest TEXT,
                        schema_name TEXT
                    );
                    CREATE TABLE job_artifacts (
                        job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                        attempt_id INTEGER REFERENCES job_attempts(id) ON DELETE CASCADE,
                        digest TEXT NOT NULL REFERENCES artifacts(digest),
                        role TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY(job_id, attempt_id, digest, role)
                    );
                    CREATE TABLE run_artifacts (
                        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                        digest TEXT NOT NULL REFERENCES artifacts(digest),
                        role TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY(run_id, digest, role)
                    );
                    """
                )
                connection.execute(
                    "INSERT INTO schema_migrations(version, name, applied_at) VALUES (1, ?, ?)",
                    ("initial_state_store", _timestamp()),
                )
            if 2 not in applied:
                connection.executescript(
                    """
                    CREATE TABLE instances (
                        id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        identity_json TEXT NOT NULL
                    );
                    CREATE TABLE run_portability (
                        run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
                        source_instance_id TEXT REFERENCES instances(id),
                        target_id TEXT NOT NULL,
                        target_arch TEXT,
                        audit_fingerprint TEXT,
                        comparison_group TEXT,
                        content_hash TEXT,
                        imported_bundle_hash TEXT,
                        created_at TEXT NOT NULL
                    );
                    CREATE UNIQUE INDEX one_portable_content_hash
                    ON run_portability(run_id, content_hash) WHERE content_hash IS NOT NULL;
                    CREATE TABLE bundle_imports (
                        bundle_hash TEXT PRIMARY KEY,
                        content_hash TEXT NOT NULL,
                        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                        source_instance_id TEXT,
                        imported_at TEXT NOT NULL,
                        manifest_json TEXT NOT NULL
                    );
                    """
                )
                connection.execute(
                    "INSERT INTO schema_migrations(version, name, applied_at) VALUES (2, ?, ?)",
                    ("portable_runs_and_bundle_imports", _timestamp()),
                )

    def create_run(self, run_id: str, *, status: str = "running", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        now = _timestamp()
        with self._transaction(immediate=True) as connection:
            connection.execute(
                """INSERT INTO runs(id, status, created_at, metadata_json) VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO NOTHING""",
                (run_id, status, now, _json(metadata)),
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"run {run_id} does not exist")
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        return result

    def finish_run(self, run_id: str, status: str) -> dict[str, Any]:
        with self._transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE runs SET status = ?, finished_at = ? WHERE id = ?",
                (status, _timestamp(), run_id),
            )
        return self.get_run(run_id)

    def delete_run_view(self, run_id: str) -> None:
        """Delete DB references only; immutable CAS payloads remain available."""
        with self._transaction(immediate=True) as connection:
            connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))

    def register_instance(self, identity: dict[str, Any]) -> dict[str, Any]:
        """Persist a stable local instance identity without accepting a rewrite."""
        instance_id = str(identity.get("instance_id") or "")
        if not instance_id:
            raise ValueError("instance_id is required")
        payload = _json(identity)
        with self._transaction(immediate=True) as connection:
            connection.execute(
                """INSERT INTO instances(id, created_at, identity_json) VALUES (?, ?, ?)
                ON CONFLICT(id) DO NOTHING""",
                (instance_id, str(identity.get("created_at") or _timestamp()), payload),
            )
            row = connection.execute("SELECT * FROM instances WHERE id = ?", (instance_id,)).fetchone()
        assert row is not None
        result = dict(row)
        result["identity"] = json.loads(result.pop("identity_json"))
        return result

    def bind_run_portability(
        self,
        *,
        run_id: str,
        source_instance_id: str | None,
        target_id: str = "local",
        target_arch: str | None = None,
        audit_fingerprint: str | None = None,
        comparison_group: str | None = None,
        content_hash: str | None = None,
        imported_bundle_hash: str | None = None,
    ) -> dict[str, Any]:
        """Bind a run once; changing a locked target or content identity is rejected."""
        with self._transaction(immediate=True) as connection:
            if connection.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone() is None:
                raise KeyError(f"run {run_id} does not exist")
            if source_instance_id is not None and connection.execute(
                "SELECT 1 FROM instances WHERE id = ?", (source_instance_id,)
            ).fetchone() is None:
                raise KeyError(f"instance {source_instance_id} does not exist")
            existing = connection.execute(
                "SELECT * FROM run_portability WHERE run_id = ?", (run_id,)
            ).fetchone()
            if existing is not None:
                for key, value in {
                    "source_instance_id": source_instance_id,
                    "target_id": target_id,
                    "target_arch": target_arch,
                    "audit_fingerprint": audit_fingerprint,
                    "comparison_group": comparison_group,
                    "content_hash": content_hash,
                    "imported_bundle_hash": imported_bundle_hash,
                }.items():
                    if value is not None and existing[key] is not None and existing[key] != value:
                        raise ValueError(f"portable run {run_id} is already locked to a different {key}")
                row = existing
            else:
                connection.execute(
                    """INSERT INTO run_portability(
                        run_id, source_instance_id, target_id, target_arch, audit_fingerprint,
                        comparison_group, content_hash, imported_bundle_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id,
                        source_instance_id,
                        target_id,
                        target_arch,
                        audit_fingerprint,
                        comparison_group,
                        content_hash,
                        imported_bundle_hash,
                        _timestamp(),
                    ),
                )
                row = connection.execute("SELECT * FROM run_portability WHERE run_id = ?", (run_id,)).fetchone()
        assert row is not None
        return dict(row)

    def get_run_portability(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM run_portability WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row is not None else None

    def list_runs(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM runs ORDER BY created_at, id").fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            result.append(item)
        return result

    def snapshot_run(self, run_id: str) -> dict[str, Any]:
        """Return a JSON-safe, deterministic snapshot of one run and its evidence."""
        run = self.get_run(run_id)
        portability = self.get_run_portability(run_id)
        with self._connect() as connection:
            instance = None
            if portability and portability.get("source_instance_id"):
                row = connection.execute(
                    "SELECT * FROM instances WHERE id = ?", (portability["source_instance_id"],)
                ).fetchone()
                if row is not None:
                    instance = dict(row)
                    instance["identity"] = json.loads(instance.pop("identity_json"))
            jobs = []
            for row in connection.execute(
                "SELECT * FROM jobs WHERE run_id = ? ORDER BY created_at, id", (run_id,)
            ).fetchall():
                job = self._job_row(row)
                attempts = []
                for attempt in connection.execute(
                    "SELECT * FROM job_attempts WHERE job_id = ? ORDER BY ordinal", (job["id"],)
                ).fetchall():
                    item = dict(attempt)
                    item["result"] = json.loads(item.pop("result_json") or "{}")
                    attempts.append(item)
                job["attempts"] = attempts
                jobs.append(job)
            run_artifacts = [
                dict(row)
                for row in connection.execute(
                    "SELECT digest, role, created_at FROM run_artifacts WHERE run_id = ? ORDER BY digest, role",
                    (run_id,),
                ).fetchall()
            ]
            job_artifacts = [
                dict(row)
                for row in connection.execute(
                    """SELECT job_artifacts.job_id, job_attempts.ordinal AS attempt_ordinal,
                              job_artifacts.digest, job_artifacts.role, job_artifacts.created_at
                       FROM job_artifacts
                       JOIN jobs ON jobs.id = job_artifacts.job_id
                       LEFT JOIN job_attempts ON job_attempts.id = job_artifacts.attempt_id
                       WHERE jobs.run_id = ?
                       ORDER BY job_artifacts.job_id, attempt_ordinal, job_artifacts.digest, job_artifacts.role""",
                    (run_id,),
                ).fetchall()
            ]
            digests = sorted({item["digest"] for item in run_artifacts + job_artifacts})
            artifacts = []
            for digest in digests:
                row = connection.execute("SELECT * FROM artifacts WHERE digest = ?", (digest,)).fetchone()
                if row is None:
                    raise KeyError(f"indexed artifact {digest} does not exist")
                artifacts.append(dict(row))
        return {
            "run": run,
            "portability": portability,
            "instance": instance,
            "jobs": jobs,
            "run_artifacts": run_artifacts,
            "job_artifacts": job_artifacts,
            "artifacts": artifacts,
        }

    def import_portable_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        bundle_hash: str,
        content_hash: str,
    ) -> dict[str, Any]:
        """Import an already validated snapshot in one SQLite transaction.

        CAS promotion happens before this call.  If the transaction fails, newly
        written CAS blobs remain unreferenced and therefore cannot appear as a
        partially imported run.
        """
        run = dict(snapshot.get("run") or {})
        run_id = str(run.get("id") or "")
        if not run_id:
            raise ValueError("portable snapshot has no run id")
        portability = dict(snapshot.get("portability") or {})
        source_instance_id = portability.get("source_instance_id")
        instance = snapshot.get("instance")
        with self._transaction(immediate=True) as connection:
            prior_bundle = connection.execute(
                "SELECT run_id, content_hash FROM bundle_imports WHERE bundle_hash = ?", (bundle_hash,)
            ).fetchone()
            if prior_bundle is not None:
                if prior_bundle["content_hash"] != content_hash:
                    raise ValueError("bundle hash is already associated with different content")
                return {"run": self.get_run(prior_bundle["run_id"]), "idempotent": True}
            existing = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if existing is not None:
                prior = connection.execute(
                    "SELECT content_hash FROM run_portability WHERE run_id = ?", (run_id,)
                ).fetchone()
                if prior is None or prior["content_hash"] != content_hash:
                    raise ValueError("run id already exists with different portable content")
                connection.execute(
                    """INSERT INTO bundle_imports(bundle_hash, content_hash, run_id, source_instance_id,
                       imported_at, manifest_json) VALUES (?, ?, ?, ?, ?, ?)""",
                    (bundle_hash, content_hash, run_id, source_instance_id, _timestamp(), _json(snapshot)),
                )
                return {"run": self.get_run(run_id), "idempotent": True}
            if instance and source_instance_id:
                identity = dict(instance.get("identity") or {})
                connection.execute(
                    "INSERT OR IGNORE INTO instances(id, created_at, identity_json) VALUES (?, ?, ?)",
                    (source_instance_id, str(instance.get("created_at") or _timestamp()), _json(identity)),
                )
            for artifact in snapshot.get("artifacts") or []:
                digest = str(artifact.get("digest") or "")
                if connection.execute("SELECT 1 FROM artifacts WHERE digest = ?", (digest,)).fetchone() is None:
                    raise KeyError(f"CAS artifact {digest} must be registered before import")
            connection.execute(
                "INSERT INTO runs(id, status, created_at, finished_at, metadata_json) VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    str(run.get("status") or "completed"),
                    str(run.get("created_at") or _timestamp()),
                    run.get("finished_at"),
                    _json(dict(run.get("metadata") or {})),
                ),
            )
            connection.execute(
                """INSERT INTO run_portability(
                    run_id, source_instance_id, target_id, target_arch, audit_fingerprint,
                    comparison_group, content_hash, imported_bundle_hash, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    source_instance_id,
                    str(portability.get("target_id") or "imported"),
                    portability.get("target_arch"),
                    portability.get("audit_fingerprint"),
                    portability.get("comparison_group"),
                    content_hash,
                    bundle_hash,
                    _timestamp(),
                ),
            )
            job_ids: dict[str, str] = {}
            attempt_ids: dict[tuple[str, int], int] = {}
            for source_job in snapshot.get("jobs") or []:
                source_job_id = str(source_job.get("id") or "")
                if not source_job_id:
                    raise ValueError("portable snapshot contains a job without id")
                job_id = source_job_id
                if connection.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,)).fetchone() is not None:
                    job_id = uuid.uuid4().hex
                job_ids[source_job_id] = job_id
                connection.execute(
                    """INSERT INTO jobs(id, run_id, idempotency_key, kind, status, payload_json,
                       created_at, updated_at, terminal_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        job_id,
                        run_id,
                        str(source_job.get("idempotency_key") or source_job_id),
                        str(source_job.get("kind") or "imported"),
                        str(source_job.get("status") or "succeeded"),
                        _json(dict(source_job.get("payload") or {})),
                        str(source_job.get("created_at") or _timestamp()),
                        str(source_job.get("updated_at") or _timestamp()),
                        source_job.get("terminal_reason"),
                    ),
                )
                for attempt in source_job.get("attempts") or []:
                    cursor = connection.execute(
                        """INSERT INTO job_attempts(job_id, ordinal, status, worker_id, started_at, finished_at, result_json)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            job_id,
                            int(attempt.get("ordinal") or 1),
                            str(attempt.get("status") or "succeeded"),
                            attempt.get("worker_id"),
                            str(attempt.get("started_at") or _timestamp()),
                            attempt.get("finished_at"),
                            _json(dict(attempt.get("result") or {})),
                        ),
                    )
                    attempt_ids[(source_job_id, int(attempt.get("ordinal") or 1))] = int(cursor.lastrowid)
            for item in snapshot.get("run_artifacts") or []:
                connection.execute(
                    "INSERT OR IGNORE INTO run_artifacts(run_id, digest, role, created_at) VALUES (?, ?, ?, ?)",
                    (run_id, item["digest"], item["role"], item.get("created_at") or _timestamp()),
                )
            for item in snapshot.get("job_artifacts") or []:
                source_job_id = str(item["job_id"])
                ordinal = item.get("attempt_ordinal")
                connection.execute(
                    """INSERT OR IGNORE INTO job_artifacts(job_id, attempt_id, digest, role, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        job_ids[source_job_id],
                        attempt_ids.get((source_job_id, int(ordinal))) if ordinal is not None else None,
                        item["digest"],
                        item["role"],
                        item.get("created_at") or _timestamp(),
                    ),
                )
            connection.execute(
                """INSERT INTO bundle_imports(bundle_hash, content_hash, run_id, source_instance_id,
                   imported_at, manifest_json) VALUES (?, ?, ?, ?, ?, ?)""",
                (bundle_hash, content_hash, run_id, source_instance_id, _timestamp(), _json(snapshot)),
            )
        return {"run": self.get_run(run_id), "idempotent": False}

    def submit_job(
        self,
        *,
        run_id: str,
        idempotency_key: str,
        kind: str,
        payload: dict[str, Any] | None = None,
        job_id: str | None = None,
    ) -> dict[str, Any]:
        if not idempotency_key or not kind:
            raise ValueError("idempotency_key and kind are required")
        job_id = job_id or uuid.uuid4().hex
        now = _timestamp()
        with self._transaction(immediate=True) as connection:
            if connection.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone() is None:
                raise KeyError(f"run {run_id} does not exist")
            connection.execute(
                """INSERT INTO jobs(id, run_id, idempotency_key, kind, status, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)
                ON CONFLICT(run_id, idempotency_key) DO NOTHING""",
                (job_id, run_id, idempotency_key, kind, _json(payload), now, now),
            )
            row = connection.execute(
                "SELECT * FROM jobs WHERE run_id = ? AND idempotency_key = ?",
                (run_id, idempotency_key),
            ).fetchone()
        assert row is not None
        return self._job_row(row)

    @staticmethod
    def _job_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        return result

    def _recover_expired_leases(self, connection: sqlite3.Connection, now: str) -> None:
        expired = connection.execute(
            "SELECT id, job_id, attempt_id FROM leases WHERE released_at IS NULL AND expires_at <= ?",
            (now,),
        ).fetchall()
        for lease in expired:
            connection.execute(
                "UPDATE leases SET released_at = ?, release_reason = 'expired' WHERE id = ? AND released_at IS NULL",
                (now, lease["id"]),
            )
            connection.execute(
                "UPDATE job_attempts SET status = 'timed_out', finished_at = ? WHERE id = ? AND finished_at IS NULL",
                (now, lease["attempt_id"]),
            )
            connection.execute(
                "UPDATE jobs SET status = 'pending', updated_at = ? WHERE id = ? AND status IN ('leased', 'running')",
                (now, lease["job_id"]),
            )

    def acquire_lease(self, *, worker_id: str, ttl_seconds: int = 60) -> dict[str, Any] | None:
        if not worker_id or not 1 <= ttl_seconds <= 3_600:
            raise ValueError("worker_id and a TTL between 1 and 3600 seconds are required")
        now_value = _utc_now()
        now = _timestamp(now_value)
        expires_at = _timestamp(now_value + timedelta(seconds=ttl_seconds))
        with self._transaction(immediate=True) as connection:
            self._recover_expired_leases(connection, now)
            job = connection.execute(
                "SELECT * FROM jobs WHERE status = 'pending' ORDER BY created_at, id LIMIT 1"
            ).fetchone()
            if job is None:
                return None
            ordinal = connection.execute(
                "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM job_attempts WHERE job_id = ?",
                (job["id"],),
            ).fetchone()[0]
            attempt = connection.execute(
                """INSERT INTO job_attempts(job_id, ordinal, status, worker_id, started_at)
                VALUES (?, ?, 'leased', ?, ?) RETURNING id""",
                (job["id"], ordinal, worker_id, now),
            ).fetchone()
            lease_id = uuid.uuid4().hex
            connection.execute(
                """INSERT INTO leases(id, job_id, attempt_id, worker_id, acquired_at, renewed_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (lease_id, job["id"], attempt["id"], worker_id, now, now, expires_at),
            )
            connection.execute(
                "UPDATE jobs SET status = 'leased', updated_at = ? WHERE id = ?",
                (now, job["id"]),
            )
            result = self._job_row(job)
            result.update(
                {
                    "lease_id": lease_id,
                    "attempt_id": attempt["id"],
                    "worker_id": worker_id,
                    "expires_at": expires_at,
                    "status": "leased",
                }
            )
            return result

    def acquire_job_lease(
        self, *, job_id: str, worker_id: str, ttl_seconds: int = 60
    ) -> dict[str, Any]:
        if not worker_id or not 1 <= ttl_seconds <= 3_600:
            raise ValueError("worker_id and a TTL between 1 and 3600 seconds are required")
        now_value = _utc_now()
        now = _timestamp(now_value)
        expires_at = _timestamp(now_value + timedelta(seconds=ttl_seconds))
        with self._transaction(immediate=True) as connection:
            self._recover_expired_leases(connection, now)
            job = connection.execute(
                "SELECT * FROM jobs WHERE id = ? AND status = 'pending'", (job_id,)
            ).fetchone()
            if job is None:
                raise KeyError(f"pending job {job_id} does not exist")
            ordinal = connection.execute(
                "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM job_attempts WHERE job_id = ?",
                (job_id,),
            ).fetchone()[0]
            attempt = connection.execute(
                """INSERT INTO job_attempts(job_id, ordinal, status, worker_id, started_at)
                VALUES (?, ?, 'leased', ?, ?) RETURNING id""",
                (job_id, ordinal, worker_id, now),
            ).fetchone()
            lease_id = uuid.uuid4().hex
            connection.execute(
                """INSERT INTO leases(id, job_id, attempt_id, worker_id, acquired_at, renewed_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (lease_id, job_id, attempt["id"], worker_id, now, now, expires_at),
            )
            connection.execute(
                "UPDATE jobs SET status = 'leased', updated_at = ? WHERE id = ?",
                (now, job_id),
            )
        return {
            **self._job_row(job),
            "status": "leased",
            "lease_id": lease_id,
            "attempt_id": attempt["id"],
            "worker_id": worker_id,
            "expires_at": expires_at,
        }

    def cancel_pending_job(self, job_id: str, *, reason: str = "cancelled") -> dict[str, Any]:
        now = _timestamp()
        with self._transaction(immediate=True) as connection:
            job = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(f"job {job_id} does not exist")
            if job["status"] in TERMINAL_JOB_STATUSES:
                return self._job_row(job)
            if job["status"] != "pending":
                raise ValueError("only pending jobs may be cancelled without an active lease")
            connection.execute(
                "UPDATE jobs SET status = 'cancelled', updated_at = ?, terminal_reason = ? WHERE id = ?",
                (now, reason, job_id),
            )
            cancelled = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        assert cancelled is not None
        return self._job_row(cancelled)

    def heartbeat_lease(self, *, lease_id: str, worker_id: str, ttl_seconds: int = 60) -> dict[str, Any]:
        if not worker_id or not 1 <= ttl_seconds <= 3_600:
            raise ValueError("worker_id and a TTL between 1 and 3600 seconds are required")
        now_value = _utc_now()
        now = _timestamp(now_value)
        expires_at = _timestamp(now_value + timedelta(seconds=ttl_seconds))
        with self._transaction(immediate=True) as connection:
            self._recover_expired_leases(connection, now)
            lease = connection.execute(
                "SELECT * FROM leases WHERE id = ? AND worker_id = ? AND released_at IS NULL",
                (lease_id, worker_id),
            ).fetchone()
            if lease is None:
                raise KeyError("active lease does not exist")
            connection.execute(
                "UPDATE leases SET renewed_at = ?, expires_at = ? WHERE id = ?",
                (now, expires_at, lease_id),
            )
            connection.execute(
                "UPDATE jobs SET status = 'running', updated_at = ? WHERE id = ? AND status = 'leased'",
                (now, lease["job_id"]),
            )
        return {"lease_id": lease_id, "expires_at": expires_at, "status": "running"}

    def complete_job(
        self,
        *,
        job_id: str,
        lease_id: str,
        worker_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        reason: str | None = None,
        artifact_roles: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if status not in TERMINAL_JOB_STATUSES:
            raise ValueError(f"invalid terminal status: {status}")
        now = _timestamp()
        with self._transaction(immediate=True) as connection:
            job = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if job is None:
                raise KeyError(f"job {job_id} does not exist")
            if job["status"] in TERMINAL_JOB_STATUSES:
                attempt = connection.execute(
                    "SELECT id FROM job_attempts WHERE job_id = ? ORDER BY ordinal DESC LIMIT 1",
                    (job_id,),
                ).fetchone()
                self._link_job_artifacts(
                    connection,
                    job_id=job_id,
                    attempt_id=attempt["id"] if attempt is not None else None,
                    artifact_roles=artifact_roles or {},
                )
                return {"job": self._job_row(job), "idempotent": True}
            lease = connection.execute(
                """SELECT * FROM leases WHERE id = ? AND job_id = ? AND worker_id = ?
                AND released_at IS NULL AND expires_at > ?""",
                (lease_id, job_id, worker_id, now),
            ).fetchone()
            if lease is None:
                raise KeyError("active lease does not exist")
            connection.execute(
                "UPDATE leases SET released_at = ?, release_reason = 'completed' WHERE id = ?",
                (now, lease_id),
            )
            connection.execute(
                "UPDATE job_attempts SET status = ?, finished_at = ?, result_json = ? WHERE id = ?",
                (status, now, _json(result), lease["attempt_id"]),
            )
            connection.execute(
                "UPDATE jobs SET status = ?, updated_at = ?, terminal_reason = ? WHERE id = ?",
                (status, now, reason, job_id),
            )
            self._link_job_artifacts(
                connection,
                job_id=job_id,
                attempt_id=lease["attempt_id"],
                artifact_roles=artifact_roles or {},
            )
            completed = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        assert completed is not None
        return {"job": self._job_row(completed), "idempotent": False}

    @staticmethod
    def _link_job_artifacts(
        connection: sqlite3.Connection,
        *,
        job_id: str,
        attempt_id: int | None,
        artifact_roles: dict[str, str],
    ) -> None:
        for digest, role in artifact_roles.items():
            if connection.execute(
                "SELECT 1 FROM artifacts WHERE digest = ?", (digest,)
            ).fetchone() is None:
                raise KeyError(f"artifact {digest} does not exist")
            connection.execute(
                """INSERT OR IGNORE INTO job_artifacts(job_id, attempt_id, digest, role, created_at)
                VALUES (?, ?, ?, ?, ?)""",
                (job_id, attempt_id, digest, role, _timestamp()),
            )

    def register_artifact(self, artifact: ArtifactMetadata) -> dict[str, Any]:
        with self._transaction(immediate=True) as connection:
            connection.execute(
                """INSERT INTO artifacts(digest, media_type, size_bytes, created_at, producer, source_digest, schema_name)
                VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(digest) DO NOTHING""",
                (
                    artifact.digest,
                    artifact.media_type,
                    artifact.size_bytes,
                    artifact.created_at,
                    artifact.producer,
                    artifact.source_digest,
                    artifact.schema,
                ),
            )
            row = connection.execute("SELECT * FROM artifacts WHERE digest = ?", (artifact.digest,)).fetchone()
        assert row is not None
        return dict(row)

    def link_job_artifact(self, *, job_id: str, digest: str, role: str, attempt_id: int | None = None) -> None:
        with self._transaction(immediate=True) as connection:
            connection.execute(
                """INSERT OR IGNORE INTO job_artifacts(job_id, attempt_id, digest, role, created_at)
                VALUES (?, ?, ?, ?, ?)""",
                (job_id, attempt_id, digest, role, _timestamp()),
            )

    def link_run_artifact(self, *, run_id: str, digest: str, role: str) -> None:
        with self._transaction(immediate=True) as connection:
            connection.execute(
                """INSERT OR IGNORE INTO run_artifacts(run_id, digest, role, created_at)
                VALUES (?, ?, ?, ?)""",
                (run_id, digest, role, _timestamp()),
            )

    def artifact_references(self, digest: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, role FROM run_artifacts WHERE digest = ? ORDER BY run_id, role", (digest,)
            ).fetchall()
        return [dict(row) for row in rows]

    def list_artifacts(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM artifacts ORDER BY digest").fetchall()
        return [dict(row) for row in rows]

    def profiler_candidate(self, executable_digest: str) -> dict[str, str]:
        """Return provenance only for an executable that passed correctness.

        Profiling is intentionally a new diagnostic job.  This lookup never
        updates the compile/correctness jobs whose terminal state is evidence.
        """
        with self._connect() as connection:
            artifact = connection.execute(
                "SELECT digest, schema_name FROM artifacts WHERE digest = ?",
                (executable_digest,),
            ).fetchone()
            replay_rows = connection.execute(
                """SELECT jobs.payload_json FROM jobs
                JOIN job_artifacts ON job_artifacts.job_id = jobs.id
                WHERE job_artifacts.digest = ? AND job_artifacts.role = 'profiler_replay'
                AND jobs.kind = 'gpu:correctness' AND jobs.status = 'succeeded'""",
                (executable_digest,),
            ).fetchall()
            compile_rows = connection.execute(
                """SELECT jobs.payload_json FROM jobs
                JOIN job_artifacts ON job_artifacts.job_id = jobs.id
                WHERE job_artifacts.digest = ? AND job_artifacts.role = 'executable'
                AND jobs.kind = 'gpu:compile' AND jobs.status = 'succeeded'""",
                (executable_digest,),
            ).fetchall()
            correctness_rows = connection.execute(
                """SELECT payload_json FROM jobs
                WHERE kind = 'gpu:correctness' AND status = 'succeeded'"""
            ).fetchall()
        if artifact is None:
            raise KeyError("profiler candidate executable is not registered")
        if replay_rows:
            payload = json.loads(replay_rows[0]["payload_json"])
            if payload.get("trusted_bundle_kind") != "generated_v2":
                raise ValueError("profiler replay is not a generated-v2 artifact")
            source_digest = str(payload.get("source_bundle_digest") or "")
            benchmark_protocol_id = str(payload.get("benchmark_protocol_id") or "")
            if len(source_digest) != 64 or not benchmark_protocol_id:
                raise ValueError("profiler replay provenance is incomplete")
            return {
                "artifact_digest": executable_digest,
                "source_digest": source_digest,
                "benchmark_protocol_id": benchmark_protocol_id,
                "artifact_kind": "candidate_profiler_capsule",
            }
        if not compile_rows:
            raise KeyError("profiler candidate executable is not registered")
        correctness = [json.loads(row["payload_json"]) for row in correctness_rows]
        if not any(row.get("executable_digest") == executable_digest for row in correctness):
            raise ValueError("profiler candidate has not passed correctness")
        compile_payload = json.loads(compile_rows[0]["payload_json"])
        source_digest = str(compile_payload.get("source_bundle_digest") or "")
        if len(source_digest) != 64:
            raise ValueError("profiler candidate source digest is unavailable")
        benchmark_protocol_id = str(
            compile_payload.get("benchmark_protocol_id") or ""
        )
        if not benchmark_protocol_id:
            raise ValueError("profiler candidate benchmark protocol is unavailable")
        return {
            "artifact_digest": executable_digest,
            "source_digest": source_digest,
            "benchmark_protocol_id": benchmark_protocol_id,
            "artifact_kind": "executable",
        }
