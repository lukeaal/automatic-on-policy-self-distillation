#!/usr/bin/env bash
set -euo pipefail

# Sync this repository to a remote path via the slurm-gpu SSH config entry.
# This forces a non-interactive SSH session so rsync works even if the host
# has RequestTTY/RemoteCommand configured for interactive srun shells.

REMOTE_HOST="${REMOTE_HOST:-slurm-gpu}"
REMOTE_SUBDIR="${1:-automatic-on-policy-self-distillation}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_TARGET="${REMOTE_HOST}:~/${REMOTE_SUBDIR}/"

echo "Syncing ${LOCAL_DIR}/ -> ${REMOTE_TARGET}"

rsync \
  -az \
  --delete \
  --progress \
  --exclude ".venv/" \
  --exclude "__pycache__/" \
  --exclude ".pytest_cache/" \
  --exclude ".mypy_cache/" \
  --exclude ".ruff_cache/" \
  -e "ssh -o RemoteCommand=none -o RequestTTY=no" \
  "${LOCAL_DIR}/" \
  "${REMOTE_TARGET}"

echo "Sync complete."
