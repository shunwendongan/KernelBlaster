# Validation status and benchmark protocol

**English** | [简体中文](validation.zh-CN.md)

This document is the living validation record for the portfolio fork. Historical raw runs remain append-only below ignored `out/portfolio/` directories; reviewed evidence is published under `artifacts/portfolio-v1.0/`.

<!-- VALIDATION_STATUS:START -->
| Gate | Current status | Canonical evidence |
| --- | --- | --- |
| Provider/Recorder/Suite CPU tests | PASSED — 98 passed | `tests/` |
| Real gateway smoke | NOT RUN — historical HTTP 401 has not been revalidated | historical `artifacts/portfolio-v1.0/results/analysis_summary.json` |
| RTX 3080 container and `sm_86` build | PASSED | `artifacts/portfolio-v1.0/environment/environment.json` |
| Official candidate correctness | HISTORICAL V1 PASSED — 10/10; schema-v2 targeted 5/5 passed | `artifacts/portfolio-v2.0/results/rtx3080_targeted_validation.json` |
| RMSNorm edge correctness | PASSED | committed `edge_driver.cpp` and deep-case artifacts |
| CUDA Events timing | schema-v2 targeted confirmation: 004/036/040 improved; 007 inconclusive; 095 exploratory | `artifacts/portfolio-v2.0/results/rtx3080_targeted_validation.json` |
| Same-GPU PyTorch comparison | COMPLETED | `pytorch_core10_rtx3080.csv` |
| NCU hardware counters | BLOCKED — `ERR_NVGPUCTRPERM` | environment manifest and historical validation report |
| Cross-GPU comparison | NOT RUN — deferred Day 11–14 | no performance claim published |

The historical manual follow-up validated all ten candidates and improved 4/10 under the old gate. Those claims remain immutable historical evidence. The targeted schema-v2 result covers only five tasks and must not be generalized to a full Core 10 or Agent-search claim.
<!-- VALIDATION_STATUS:END -->

## Completed development and experiment timeline

1. **Days 1–2 — provider and observability:** implemented the OpenAI-compatible provider boundary, client-side fan-out, retry classification, atomic request/token reservation, usage fallback, Recorder sequence/atomicity/redaction, Suite validation, and offline dry-run artifacts.
2. **Days 3–7 — local GPU validation:** pinned the NGC PyTorch 25.01 container, mapped the RTX 3080 to `sm_86`, validated compiler/GPU servers, added the independent CUDA Events runner, and recorded the baseline Core 10 state. The bounded live API smoke returned HTTP 401, so Agent Pilot/Core 10 rollouts did not run.
3. **Days 8–10 — RMSNorm deep case:** implemented V1–V3c, added odd/tiny/63–65-channel correctness inputs, retained failed variants, and published the 49.348× paired V3c result. NCU counter access remained blocked, so no hardware-counter attribution was published.
4. **Core 10 follow-up:** added nine manual candidates, reran all ten tasks with 20 warmups, 100 samples, three independent process Sessions, AB/BA ordering, and compared them with same-GPU PyTorch eager/out/fused methods.
5. **Publication:** checked in redacted JSON/CSV/SVG reports and SHA256 links for 362 ignored raw JSON/JSONL/CSV/SVG/log files. Merged PRs #4 and #5 contain the artifact publication and living-documentation follow-up.
6. **Schema-v2 targeted validation:** reran correctness for 004/007/036/040/095 and five-Session confirmation for 004/007/036/040. Tasks 004/036/040 passed the new bootstrap and stability gate, 007 was inconclusive because the upstream baseline was unstable, and 095 remains exploratory.

## Result interpretation

- Historical v1 diagnostic candidate medians include unstable tasks and are useful for prioritizing follow-up work, not release claims.
- The current schema-v2 gate requires correctness, no more than 5% cross-session spread, five paired Sessions, at least 1.01× median speedup, and a paired-bootstrap 95% lower bound above 1.0.
- The nine newly developed candidates score 5.020× diagnostic and 3.302× strict versus upstream; 004, 007, and 040 pass the strict gate.
- Full Core 10, including the existing RMSNorm case, scores 6.351× diagnostic and 4.356× strict versus upstream.
- Against the fastest measured PyTorch method, the nine-candidate diagnostic/strict ratios are 1.415×/0.931×; the full-ten ratios are 1.447×/0.992×.
- Task 007 calls cuBLAS and therefore demonstrates correct mature-library integration rather than a custom GEMM beating cuBLAS.

Canonical evidence:

- [Schema-v2 targeted validation](../../artifacts/portfolio-v2.0/reports/rtx3080-targeted-validation.en.md)
- [Schema-v2 result JSON](../../artifacts/portfolio-v2.0/results/rtx3080_targeted_validation.json)
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
- Require at least five paired confirmation Sessions, median speedup of 1.01× or higher, and a paired-bootstrap 95% lower bound above 1.0.
- Do not infer NCU bottlenecks from CUDA Events or code inspection; `ERR_NVGPUCTRPERM` remains an explicit attribution blocker.

## Remaining blockers and deferred work

- Supply a valid external API credential before Agent-driven Pilot/Core 10 search; the single 401 request was not retried or billed as a successful completion.
- Run schema-v2 correctness and five-Session confirmation for the other five Core 10 tasks; rerun 007 until both sides are stable and confirm 095 with five Sessions.
- Collect the required NCU sections on an explicitly authorized profiler worker; local Docker/WSL remains `events_only`.
- Run and publish L40S/A100 matching comparisons independently.
- Generalize fixed-shape candidates across shapes, dtypes, layouts, streams, graph capture, backward paths, and stricter numerical tolerances before production-library claims.

## Reproduction commands

```bash
python scripts/benchmark_candidates.py \
  --phase confirmation \
  --warmup 20 --repetitions 100 --sessions 5 \
  --cooldown-seconds 60 \
  --output-dir out/portfolio/candidates/<run-id>

python scripts/benchmark_pytorch.py \
  --warmup 20 --repetitions 100 --sessions 5 \
  --output-dir out/portfolio/pytorch/<run-id>

python scripts/analyze_core10_comparison.py \
  --candidate-summary out/portfolio/candidates/<run-id>/suite_summary.json \
  --pytorch-summary out/portfolio/pytorch/<run-id>/pytorch_summary.json \
  --output-dir out/portfolio/analysis/<run-id>

python scripts/sync_portfolio_docs.py --write
python scripts/sync_portfolio_docs.py --check
```

The language model remains an external inference service. Rollouts update trajectories and the optimization database; they do not train or fine-tune model weights.
