# Release operations and evidence

[中文版本](release-operations.zh-CN.md)

KernelBlaster releases are evidence-led research releases, not production
deployment claims.  Automatic generated execution remains disabled unless an
operator explicitly configures its existing feature flag and service secrets.

## Plan and validate an E2E run

Start from the public profile, then produce an auditable plan before any paid
Provider request or GPU workload:

```bash
python scripts/e2e_release.py --profile configs/release.example.toml
python scripts/fault_inject.py --output release-evidence/fault-plan.json
```

`--execute` runs the executable preflight, 64-token Provider smoke, and bounded
RMSNorm Agent stages, passing the published preflight capability digest to the
Agent. It assumes Compose services and external secrets were started by the
operator. Hardware, model, token budget, paths and timeouts are profile values,
not source-code GPU/version choices; the trusted smoke keeps its fixed trusted
smoke model.

## Evidence

Create redacted evidence from a public JSON summary and then verify its index:

```bash
python scripts/release_evidence.py local-<gpu-slug> \
  --profile configs/release.example.toml \
  --evidence-json out/release-summary.json
python scripts/verify_release.py release-evidence
```

The writer records the checked-out commit, a canonical profile hash and schema
digests, then removes credentials, tokens, private values, host/instance
details and state paths before adding every evidence file to `SHA256SUMS.json`.
`--include` accepts only public JSON summaries and sanitizes them again; large
raw NSYS/NCU reports remain release artifacts, so commit their summaries and
hashes rather than sensitive or oversized raw files.

## Backup and rollback

Before migration or release, back up the SQLite control state:

```bash
python scripts/backup_state.py /safe/backups --state-dir "$KERNELBLASTER_STATE_DIR"
python scripts/restore_state_backup.py /safe/backups/kernelblaster-state-<timestamp> \
  --state-dir "$KERNELBLASTER_STATE_DIR" --yes
```

The backup uses SQLite's online backup API and records a snapshot SHA-256; CAS
is immutable and intentionally not copied. Stop Control before restoring.
Rollback restores SQLite only after explicit confirmation; it does not attempt
a destructive schema down-migration or delete historical CAS payloads.

See [release checklist](release-checklist.md) before preparing a tag.
