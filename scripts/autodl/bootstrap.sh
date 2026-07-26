#!/usr/bin/env bash
# Idempotent AutoDL bootstrap.  It deliberately never installs a display or
# kernel driver; host-driver ownership belongs to the AutoDL image/platform.
set -euo pipefail

check_only=0
install_user_deps=0
profile=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only) check_only=1 ;;
    --install-user-deps) install_user_deps=1 ;;
    --profile) profile="$2"; shift ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

if [[ -n "$profile" ]]; then
  [[ -f "$profile" ]] || { printf 'Profile does not exist: %s\n' "$profile" >&2; exit 2; }
  while IFS='=' read -r key value; do
    [[ -z "$key" || "$key" == \#* ]] && continue
    [[ "$key" =~ ^(KERNELBLASTER|AUTODL|PYTHON_BIN)_[A-Z0-9_]*$ ]] || {
      printf 'Profile key is not allowed: %s\n' "$key" >&2; exit 2;
    }
    export "$key=$value"
  done < "$profile"
fi

python_bin="${PYTHON_BIN:-python3}"
if [[ -z "${KERNELBLASTER_STATE_DIR:-}" ]]; then
  if [[ -n "${AUTODL_TMP:-}" ]]; then
    export KERNELBLASTER_STATE_DIR="$AUTODL_TMP/kernelblaster"
  else
    export KERNELBLASTER_STATE_DIR="$HOME/.local/share/kernelblaster"
  fi
fi

mkdir -p "$KERNELBLASTER_STATE_DIR"
[[ -w "$KERNELBLASTER_STATE_DIR" ]] || { printf 'State directory is not writable: %s\n' "$KERNELBLASTER_STATE_DIR" >&2; exit 1; }

command -v "$python_bin" >/dev/null || { printf 'Python executable unavailable: %s\n' "$python_bin" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { printf 'nvidia-smi is unavailable; bootstrap will not install a driver.\n' >&2; exit 1; }
nvidia-smi --query-gpu=name,uuid,driver_version,compute_cap --format=csv,noheader

if command -v nvcc >/dev/null; then
  nvcc --version | tail -n 1
else
  printf 'nvcc unavailable; configure a compatible toolkit before generated GPU jobs.\n' >&2
fi

if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
  docker info --format '{{.ServerVersion}}'
else
  printf 'Docker/NVIDIA runtime is unavailable; install it through the image/platform, not this script.\n' >&2
fi

df -Pk "$KERNELBLASTER_STATE_DIR" | tail -n 1

if [[ "$check_only" -eq 1 ]]; then
  printf 'AutoDL capability check completed.\n'
  exit 0
fi

if [[ "$install_user_deps" -eq 1 ]]; then
  "$python_bin" -m pip install --user --upgrade uv
  uv sync --frozen
fi

printf 'AutoDL bootstrap completed with state directory: %s\n' "$KERNELBLASTER_STATE_DIR"
