# Release checklist

[中文版本](release-checklist.zh-CN.md)

- [ ] `uv lock --check`, full CPU tests, static checks and Compose config pass.
- [ ] Local hardware evidence records detected capability, environment, image digest and measurement protocol.
- [ ] At least one fresh AutoDL instance completes bootstrap, preflight, smoke, bounded run, export and local import.
- [ ] Fault plan records restart, lease, storage, profiler and secret-boundary results; dangerous cases are explicit.
- [ ] `scripts/verify_release.py release-evidence` reports valid hashes.
- [ ] Backup of the previous SQLite state and previous Compose/image references exists.
- [ ] README and runbooks do not claim production readiness, universal NCU access or cross-GPU performance ranking.
- [ ] Automatic untrusted execution remains disabled by default.
- [ ] Tag and GitHub Release receive a separate final approval.
