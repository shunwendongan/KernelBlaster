# Portfolio validation and development progress

[简体中文](README.zh-CN.md) | **English**

<!-- PORTFOLIO_PROGRESS:START -->
**Last updated: 2026-07-21**

- Days 1–2: provider, recorder, suite validation, dry-run, and CPU tests are complete.
- Days 3–7: WSL2/RTX 3080, container, compilation, correctness, and CUDA Events infrastructure are complete; the historical API smoke returned 401 and the current credential has not been revalidated.
- Days 8–10: RMSNorm V0–V3c is complete, with 49.348× in the independent deep run and 52.772× in the unified Core 10 rerun.
- Core 10 follow-up: historical v1 validated all nine new candidates and 3/9 passed its old strict gate; targeted schema v2 confirmed 004/036/040, left 007 inconclusive, and kept 095 exploratory.
- PyTorch comparison: the strict full-ten ratio is 0.992×, effectively parity; unstable tasks remain explicitly labeled.

[Schema-v2 targeted validation](../../artifacts/portfolio-v2.0/reports/rtx3080-targeted-validation.en.md) · [Schema-v2 result JSON](../../artifacts/portfolio-v2.0/results/rtx3080_targeted_validation.json) · [Full Chinese report](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-comparison.zh-CN.md) · [English summary](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-summary.en.md) · [Per-task JSON](../../artifacts/portfolio-v1.0/results/core10_rtx3080_comparison.json) · [Comparison figure](../../artifacts/portfolio-v1.0/figures/core10_rtx3080_comparison.svg) · [Raw-file hashes](../../artifacts/portfolio-v1.0/manifests/core10_rtx3080_raw_sha256.csv) · [Candidate manifest](../../portfolio/case_studies/core10/candidates.json)
<!-- PORTFOLIO_PROGRESS:END -->

## Documentation map

- Architecture, provider, recorder, benchmark, and docs-sync boundaries: [English](architecture.md) · [简体中文](architecture.zh-CN.md)
- Validation gates, completed work, evidence, and remaining blockers: [English](validation.md) · [简体中文](validation.zh-CN.md)
- Measured RMSNorm V0–V3c case study: [English](rmsnorm-case-study.md) · [简体中文](rmsnorm-case-study.zh-CN.md)
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
