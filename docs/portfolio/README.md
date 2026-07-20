# Portfolio validation and development progress

[简体中文](README.zh-CN.md) | **English**

<!-- PORTFOLIO_PROGRESS:START -->
**Last updated: 2026-07-20**

- Days 1–2: provider, recorder, suite validation, dry-run, and CPU tests are complete.
- Days 3–7: WSL2/RTX 3080, container, compilation, correctness, and CUDA Events infrastructure are complete; Agent Pilot/Core 10 did not run because the API smoke returned 401.
- Days 8–10: RMSNorm V0–V3c is complete, with 49.348× in the independent deep run and 52.772× in the unified Core 10 rerun.
- Core 10 follow-up: all nine new candidates are correct and 3/9 pass the strict gate; the full Core 10 strict score is 4.356×.
- PyTorch comparison: the strict full-ten ratio is 0.992×, effectively parity; unstable tasks remain explicitly labeled.

[Full Chinese report](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-comparison.zh-CN.md) · [English summary](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-summary.en.md) · [Per-task JSON](../../artifacts/portfolio-v1.0/results/core10_rtx3080_comparison.json) · [Comparison figure](../../artifacts/portfolio-v1.0/figures/core10_rtx3080_comparison.svg) · [Raw-file hashes](../../artifacts/portfolio-v1.0/manifests/core10_rtx3080_raw_sha256.csv) · [Candidate manifest](../../portfolio/case_studies/core10/candidates.json) · [Draft PR #5](https://github.com/shunwendongan/KernelBlaster/pull/5)
<!-- PORTFOLIO_PROGRESS:END -->

## Documentation map

- [Architecture, provider, recorder, benchmark, and docs-sync boundaries](architecture.md)
- [Validation gates, completed work, evidence, and remaining blockers](validation.md)
- [Measured RMSNorm V0–V3c case study](rmsnorm-case-study.md)
- [Published RTX 3080 artifact bundle](../../artifacts/portfolio-v1.0/README.md)
- [Full Chinese Core 10 and PyTorch analysis](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-comparison.zh-CN.md)
- [Standalone English result summary](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-summary.en.md)

## Update workflow

`portfolio/status.json` records narrative state and points to canonical evidence. Performance values are derived from the checked-in result JSON rather than copied into the manifest. After any benchmark, candidate, analysis, or publication change:

```bash
python scripts/sync_portfolio_docs.py --write
python scripts/sync_portfolio_docs.py --check
```

The GitHub documentation-sync workflow repeats the check and rejects relevant changes that omit README/docs or status updates.
