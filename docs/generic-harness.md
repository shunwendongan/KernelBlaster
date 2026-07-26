# Generic multi-operator Harness

Chinese version: [generic-harness.zh-CN.md](generic-harness.zh-CN.md)

KernelBlaster evaluates operators through versioned contracts instead of a
single RMSNorm ABI. `harness-task/v1` declares direction, tensor/scalar ABI,
bounded dynamic shapes, all differentiable inputs, numeric and determinism
classes, workspace, one caller-owned stream, optional CUDA Graph behavior, and
weighted performance workloads. The first catalog covers Core 10 forward and
backward; RMSNorm is only one normalization example.

The trusted runtime generates Dev, Feedback, and Audit inputs, snapshots all
immutable inputs, and owns `correctness-result/v2`. It checks output poison,
guard canaries reported by a low-level Adapter, input mutation, shape/dtype,
NaN/Inf, CUDA errors, every requested gradient, and repeat stability. A
candidate cannot declare itself correct. Feedback and Audit cases are fully
disclosed, so the protocol is explicitly `adaptive_disclosed`, not held-out.

Ordinary operators use declarative TaskSpecs. Complex references and resource
lifecycle rules use an image-resident trusted Adapter. Existing KernelBench
`driver.cpp` files are preserved behind `LegacyDriverAdapter`; the confirmed
Core 10 bug where a failed comparison returned process status zero is fixed.
Backward reference values come from PyTorch autograd with FP16 inputs and FP32
math. PyTorch is an oracle here, not a mandatory Custom Op gate for external
projects. Ten deterministic, auditable CUDA backward baselines live under
`portfolio/harness/core10/backward/`.

## Private cases and CAS registration

Keep real case files outside the checkout, normally under
`$HOME/secrets/kernelblaster/cases`. The repository contains only deterministic
public fixtures and tooling. With Control running, seed the external directory
when needed and upload each canonical TaskSpec, case bundle, and trusted source
bundle to CAS:

```bash
uv run python scripts/register_core10_harness.py \
  --case-root "$HOME/secrets/kernelblaster/cases" \
  --output "$HOME/secrets/kernelblaster/core10-harness-catalog.json" \
  --public-fixtures
```

Omit `--public-fixtures` for a real evaluation deployment. Every existing case
file is schema-, shape-, TaskSpec-digest-, and SHA-256-bound before upload.

## Signed Adapter plugins

An external project normally adds a TaskSpec plus an Ed25519-signed Adapter
bundle instead of changing KernelBlaster core code. Generate the private key
outside the repository, build a deterministic bundle, then verify it:

```bash
uv run python scripts/adapter_plugin.py keygen \
  --private-key "$HOME/secrets/kernelblaster/adapter.key" \
  --public-key "$HOME/secrets/kernelblaster/adapter.pub"
uv run python scripts/adapter_plugin.py build \
  --descriptor /path/to/plugin-descriptor.json \
  --payload-dir /path/to/plugin-payload \
  --private-key "$HOME/secrets/kernelblaster/adapter.key" \
  --output /path/to/adapter-plugin.tar
uv run python scripts/adapter_plugin.py verify \
  --bundle /path/to/adapter-plugin.tar --key-id project-owner \
  --public-key "$HOME/secrets/kernelblaster/adapter.pub"
```

The image build additionally requires an exact bundle digest, publisher key,
plugin identity, and Adapter-ID allowlist. Only public keys and the signed
bundle enter the temporary Docker build context; the private key never does:

```bash
uv run python scripts/build_adapter_job_image.py \
  --bundle /path/to/adapter-plugin.tar \
  --trusted-keys "$HOME/secrets/kernelblaster/trusted-adapter-keys.json" \
  --allowlist "$HOME/secrets/kernelblaster/adapter-plugin-allowlist.json" \
  --tag local/kernelblaster-gpu-job:adapter-v1
```

Pin the resulting `sha256:` image ID in the Supervisor configuration. A tag is
never accepted as the generated Job trust anchor.

## RTX 3080 smoke

The fixed smoke validates all 20 TaskSpecs, disclosed dynamic cases, the ten
naive CUDA backward baselines on a Harness-owned non-default stream, and a
mutation/NaN/poison/canary adversarial fixture:

```bash
python scripts/run_core10_harness_smoke.py --device cuda --backward-cuda \
  --output /tmp/core10-harness-smoke.json
```

CUDA Events ranking, independent baseline providers, AOT candidate isolation,
and profiler replay are layered on this correctness contract in the following
stacked changes. NSYS and NCU remain diagnostic and never decide correctness.
