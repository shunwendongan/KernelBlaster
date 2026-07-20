# Validation status and benchmark protocol

This document is the living validation record for the portfolio fork. Historical raw runs remain append-only below ignored `out/portfolio/` directories; reviewed evidence is published under `artifacts/portfolio-v1.0/`.

<!-- VALIDATION_STATUS:START -->
| Gate | Current status | Canonical evidence |
| --- | --- | --- |
| Provider/Recorder/Suite CPU tests | PASSED — 52 passed | `tests/` and the published validation report |
| Real gateway smoke | BLOCKED — HTTP 401, no retry | `artifacts/portfolio-v1.0/results/analysis_summary.json` |
| RTX 3080 container and `sm_86` build | PASSED | `artifacts/portfolio-v1.0/environment/environment.json` |
| Official candidate correctness | PASSED — 10/10 | Core 10 comparison JSON and raw SHA256 manifest |
| RMSNorm edge correctness | PASSED | committed `edge_driver.cpp` and deep-case artifacts |
| CUDA Events timing | COMPLETED — 20/100/3 | Core 10 comparison JSON/CSV |
| Same-GPU PyTorch comparison | COMPLETED | `pytorch_core10_rtx3080.csv` |
| NCU hardware counters | BLOCKED — `ERR_NVGPUCTRPERM` | environment manifest and historical validation report |
| Cross-GPU comparison | NOT RUN — deferred Day 11–14 | no performance claim published |

The manual follow-up validates all ten candidates and formally improves 4/10 under the strict gate. Agent-driven rollout search remains separate and unexecuted because the one bounded live API request failed authentication.
<!-- VALIDATION_STATUS:END -->

## Completed development and experiment timeline

1. **Days 1–2 — provider and observability:** implemented the OpenAI-compatible provider boundary, client-side fan-out, retry classification, atomic request/token reservation, usage fallback, Recorder sequence/atomicity/redaction, Suite validation, and offline dry-run artifacts.
2. **Days 3–7 — local GPU validation:** pinned the NGC PyTorch 25.01 container, mapped the RTX 3080 to `sm_86`, validated compiler/GPU servers, added the independent CUDA Events runner, and recorded the baseline Core 10 state. The bounded live API smoke returned HTTP 401, so Agent Pilot/Core 10 rollouts did not run.
3. **Days 8–10 — RMSNorm deep case:** implemented V1–V3c, added odd/tiny/63–65-channel correctness inputs, retained failed variants, and published the 49.348× paired V3c result. NCU counter access remained blocked, so no hardware-counter attribution was published.
4. **Core 10 follow-up:** added nine manual candidates, reran all ten tasks with 20 warmups, 100 samples, three independent process Sessions, AB/BA ordering, and compared them with same-GPU PyTorch eager/out/fused methods.
5. **Publication:** checked in redacted JSON/CSV/SVG reports and SHA256 links for 362 ignored raw JSON/JSONL/CSV/SVG/log files. Merged PR #4 contains the code and artifact publication; Draft PR #5 tracks the living-documentation follow-up.

## Result interpretation

- The diagnostic candidate medians include unstable tasks and are useful for prioritizing follow-up work, not release claims.
- The strict score requires correctness, no more than 5% cross-session spread, every paired Session not slower, and at least 1.01× aggregate speedup. Rejected tasks remain in the denominator as upstream 1.0.
- The nine newly developed candidates score 5.020× diagnostic and 3.302× strict versus upstream; 004, 007, and 040 pass the strict gate.
- Full Core 10, including the existing RMSNorm case, scores 6.351× diagnostic and 4.356× strict versus upstream.
- Against the fastest measured PyTorch method, the nine-candidate diagnostic/strict ratios are 1.415×/0.931×; the full-ten ratios are 1.447×/0.992×.
- Task 007 calls cuBLAS and therefore demonstrates correct mature-library integration rather than a custom GEMM beating cuBLAS.

Canonical evidence:

- [Per-task comparison JSON](../../artifacts/portfolio-v1.0/results/core10_rtx3080_comparison.json)
- [Full Chinese analysis](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-comparison.zh-CN.md)
- [Standalone English summary](../../artifacts/portfolio-v1.0/reports/core10-rtx3080-summary.en.md)
- [Environment manifest](../../artifacts/portfolio-v1.0/environment/environment.json)
- [Raw-artifact SHA256 manifest](../../artifacts/portfolio-v1.0/manifests/core10_rtx3080_raw_sha256.csv)

## Acceptance rules

- Reject a candidate when any official or extra correctness input fails, regardless of timing.
- Preserve original and timing-normalized source SHA256 values in each raw manifest.
- Report GPU, SM target, precision, shape, seed, warmup, repetitions, inner loops, Session/order, median, p10/p90, telemetry, driver/CUDA/container, and source provenance.
- Treat upstream CUDA, candidate CUDA, PyTorch eager/out/fused, and paper-reported results as separate baselines.
- Stop formal performance claims when the automatic cooldown/retest still exceeds the 5% Session gate.
- Do not infer NCU bottlenecks from CUDA Events or code inspection; `ERR_NVGPUCTRPERM` remains an explicit attribution blocker.

## Remaining blockers and deferred work

- Supply a valid external API credential before Agent-driven Pilot/Core 10 search; the single 401 request was not retried or billed as a successful completion.
- Reload the Windows NVIDIA driver after enabling performance counters, then collect and export the required NCU sections.
- Run L40S/A100 matching comparisons only in the deferred Day 11–14 scope.
- Generalize fixed-shape candidates across shapes, dtypes, layouts, streams, graph capture, backward paths, and stricter numerical tolerances before production-library claims.

## Reproduction commands

```bash
python scripts/benchmark_candidates.py \
  --warmup 20 --repetitions 100 --sessions 3 \
  --cooldown-seconds 60 \
  --output-dir out/portfolio/candidates/<run-id>

python scripts/benchmark_pytorch.py \
  --warmup 20 --repetitions 100 --sessions 3 \
  --output-dir out/portfolio/pytorch/<run-id>

python scripts/analyze_core10_comparison.py \
  --candidate-summary out/portfolio/candidates/<run-id>/suite_summary.json \
  --pytorch-summary out/portfolio/pytorch/<run-id>/pytorch_summary.json \
  --output-dir out/portfolio/analysis/<run-id>

python scripts/sync_portfolio_docs.py --write
python scripts/sync_portfolio_docs.py --check
```

The language model remains an external inference service. Rollouts update trajectories and the optimization database; they do not train or fine-tune model weights.
