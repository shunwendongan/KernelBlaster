# AutoDL independent instances

KernelBlaster supports standalone AutoDL instances.  Each instance owns its
own Control service, SQLite database, CAS directory, GPU Supervisor, Profiler
Worker, and Baseline Worker.  Instances never share a database, queue, CAS or
service-discovery layer.

## Bootstrap

Copy `deploy/autodl/profile.env.example` outside the repository and set the
state directory, selected GPU device and optional immutable image references.
Run:

```bash
bash scripts/autodl/bootstrap.sh --profile /secure/path/autodl.env --check-only
bash scripts/autodl/bootstrap.sh --profile /secure/path/autodl.env
```

The script probes the platform but never installs, upgrades or replaces the
NVIDIA display/kernel driver.  Configure that driver through the AutoDL image
or platform.  All hardware decisions are made from capability probes; no GPU
product name needs to be added to source code.

Run `python scripts/manage_instance.py show --state-dir <state>` to display
the stable instance identity.  If a data disk was cloned, intentionally create
a new identity with `python scripts/manage_instance.py rotate --yes`.

## Independent execution and migration

Run a normal standalone workload, then export one terminal run:

```bash
python scripts/export_run.py <run-id> /data/run.tar --state-dir <state>
python scripts/import_run.py /data/run.tar --state-dir <local-state>
python scripts/aggregate_runs.py /data/aggregate --state-dir <local-state>
```

Bundles contain only the selected run's public artifact closure.  Credentials,
private evaluation profiles, drivers and seeds are excluded.  Import treats the
archive as untrusted input and rejects links, traversal paths, unknown files,
oversized members and hash mismatches.  Re-importing identical content is safe;
a matching run ID with different content is rejected.

## Optional explicit remote target

`deploy/autodl/targets.toml.example` supports a user-selected SSH target.  It
stores only an SSH alias, remote work directory and profile reference.  SSH
keys remain in the user's SSH agent/config.  KernelBlaster never schedules,
retries on another machine or migrates an active run.
