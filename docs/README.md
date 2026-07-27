# KernelBlaster documentation

**English** | [简体中文](README.zh-CN.md)

These documents are ordered from finding an entry point, through understanding the system, to optimizing an operator. Except for historical GPU evidence explicitly labeled under `docs/portfolio/`, feature descriptions follow the source on the default `master` branch.

## What should I read first?

| Goal | Start here | Outcome |
| --- | --- | --- |
| Understand the project in ten minutes | [Quick start](quickstart.md) | Distinguish no-GPU, manual optimization, and Agent/infrastructure paths |
| Understand the source tree | [Source architecture](architecture.md) | Locate entry points, state, execution planes, storage, and evidence |
| Learn to write fast operators | [High-performance operator guide](operator-development.md) | Iterate from contract and baseline through correctness and performance gates |
| Follow the core call chain | [Core source guide](source-guide.md) | Read `run_RL.py`, the Graph, Agent, and Profiler in call order |
| Understand project history | [Development history and branch status](development-history.md) | Separate `master` capabilities from stacked-PR development work |
| Reproduce results | [Portfolio index](portfolio/README.md) | Locate RTX 3080 results, validation rules, and the RMSNorm case study |
| Interpret measurements | [Measurement and status contract](measurement-status-contract.md) | Interpret correctness, timing, diagnostic, and terminal states |

## Documentation areas

### Usage and learning

- [Quick start](quickstart.md): shortest paths, dependency boundaries, and common mistakes.
- [High-performance operator guide](operator-development.md): a CUDA optimization method that transfers to other operators.
- [Core source guide](source-guide.md): internal call chain and critical invariants.

### Architecture and contracts

- [Source architecture](architecture.md): source directories, execution paths, data, and trust boundaries.
- [Measurement and status contract](measurement-status-contract.md): semantics of machine-readable results.
- [Portfolio architecture](portfolio/architecture.md): experiment suites, runners, and artifact contracts.

### Results and evidence

- [Portfolio status](portfolio/README.md): current validation progress.
- [Validation protocol](portfolio/validation.md): how results are produced and what may be claimed.
- [RMSNorm case study](portfolio/rmsnorm-case-study.md): an end-to-end example from memory mapping to measured candidate.
- `artifacts/portfolio-v*/`: immutable results, reports, figures, and SHA-256 manifests.

## Evidence vocabulary

Always distinguish three kinds of statements in this repository:

1. **Implemented in source**: code, contracts, and test definitions exist on the default branch; this does not claim execution on the reader's machine.
2. **Historically validated**: checked-in artifacts identify hardware, protocol, results, and hashes; conclusions apply only within the declared boundary.
3. **In development or blocked**: code lives off `master`, or still depends on Provider credentials, GPU access, profiler permissions, or cross-GPU resources.

Documentation changes should keep the Chinese pair [README.zh-CN.md](README.zh-CN.md) in sync. Generated Portfolio status blocks are owned by `scripts/sync_portfolio_docs.py`; do not edit their contents by hand.
