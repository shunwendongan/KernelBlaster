# KernelBlaster RTX 3080 validation artifacts

This directory contains the curated, redacted outputs for Days 1-10. Raw logs,
build trees, benchmark samples, and any `.ncu-rep` files remain under ignored
`out/portfolio/` directories.

- `environment/`: sanitized WSL, Docker, CUDA, GPU, Server, API, and NCU status.
- `results/`: machine-readable Core 10, deep-case, failure, usage, and aggregate data.
- `figures/`: SVG generated from the result CSV/JSON files.
- `reports/`: Chinese validation report and standalone English summary.
- `manifests/`: SHA256 links to local raw inputs and generated analysis outputs.

Important interpretation boundaries:

- The single live API request failed with HTTP 401 and was not retried. No Core
  10 Agent search result is claimed.
- All ten Core tasks have a baseline attempt and final state; seven passed the
  session-stability gate and three remain explicitly unstable.
- RMSNorm is a manual Day 8-10 case study. Its CUDA Events result is separate
  from the Agent-only portfolio score.
- Nsight Compute hardware-counter attribution is blocked by
  `ERR_NVGPUCTRPERM` until the Windows driver reloads the enabled setting.
