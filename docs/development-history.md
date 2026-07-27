# KernelBlaster development history and branch status

**English** | [简体中文](development-history.zh-CN.md)

This static audit snapshot was built from GitHub commits and PR metadata on 2026-07-27. It explains why the source contains a classic Agent path, secure infrastructure, and interfaces that are not fully connected. GitHub remains the live source for branch status.

## Default-branch baseline

At audit time, `master` points to `60eb264f` (PR #21, runtime capability preflight). README statements about current behavior must therefore follow this commit and its ancestors.

## Stages already on `master`

| Stage | Representative PR | Main result |
| --- | --- | --- |
| Provider and observability | PR #1–2 | OpenAI-compatible Provider, budgets, bilingual docs, and run records |
| Reproducible experiments | PR #3–6 | RMSNorm/Core 10, PyTorch comparison, artifact hashes, and docs sync |
| Measurement/status contract | PR #14 | Separate correctness, timing, diagnostic, and terminal fields |
| Compose boundaries | PR #15 | Isolated Control/GPU services and token audiences |
| State and artifacts | PR #16 | SQLite task state and SHA-256 CAS |
| GPU Jobs | PR #17 | Hardware detection, digest-only manifests, and Supervisor |
| Ephemeral sandbox | PR #18 | Non-root, no-network, fixed-resource generated Job containers |
| Profiler Worker | PR #19–20 | Fixed NSYS/NCU plans, WSL preflight, and CSV parsing |
| Runtime preflight | PR #21 | Ordered Provider→storage→GPU→sandbox→Events→diagnostics report |

## Stacked development line

PRs #22–27 were merged into the next development branch, not successively into `master`:

```text
master / PR-07a
  -> PR-07b Agent candidate funnel
  -> PR-07c generic correctness harness
  -> PR-07d independent baseline providers
  -> PR-07e CUDA/Triton candidate isolation
  -> PR-08 AutoDL portability
  -> PR-09 E2E release hardening
```

GitHub reports these PRs as merged because each entered the next branch. That does not mean the feature entered the default branch.

| PR | Development capability | Effect on mainline documentation |
| --- | --- | --- |
| #22 / PR-07b | Structured CandidateEvaluation, sandbox CandidateEvaluator, discovery/confirmation funnel | Closes the gap between the secure backend and Agent generation loop |
| #23 / PR-07c | Generic forward/backward harness, TaskSpec, and Adapter contracts | Further separates private drivers/seeds from candidate code |
| #24 / PR-07d | Independent Baseline Worker, provider columns, multi-workload ranking | Adds reference columns while upstream CUDA remains the formal baseline |
| #25 / PR-07e | `candidate-package/v2`, CUDA/Triton AOT isolation, replay capsules | Prevents candidate Python/host-launcher execution and strengthens publication gates |
| #26 / PR-08 | AutoDL instances, run-bundle import/export, cross-instance aggregation | Adds multi-instance portability tools |
| #27 / PR-09 | Release orchestration, fault planning, evidence, backup/restore | Defines 0.3.0 release gates; final real hardware acceptance remains deferred |

## Why this boundary belongs in the docs

Without the branch relationship, readers may:

1. misread `RuntimeBackendBundle` fail-closed behavior as a normal configuration error;
2. use harness, baseline, candidate-package, or AutoDL commands that do not exist on `master`;
3. treat development-branch validation from a PR body as current default-branch validation.

The root README therefore summarizes mainline capabilities and labels the later stack as development status. Commands must match files on the branch being documented.

## Documentation checklist after the stack reaches `master`

When PR-07b through PR-09 actually reach the default branch, update together:

- Agent status and default commands in the [quick start](quickstart.md);
- CandidateEvaluator, Harness, Baseline Worker, Candidate Package, and Portability sections in [source architecture](architecture.md);
- the root README feature matrix and source tree;
- generated validation status in `portfolio/status.json`;
- all bilingual pairs and `scripts/sync_portfolio_docs.py --check`.

Until then, development-branch validation is useful evolution evidence, not a substitute for `master` capability claims.
