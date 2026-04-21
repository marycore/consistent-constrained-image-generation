#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh" "${1:-}"

"${SSH_BASE[@]}" "${RUNPOD_USER}@${RUNPOD_HOST}" "mkdir -p '${RUNPOD_REMOTE_DIR}'"

# Mirror local -> remote (delete stale remote files); avoid ownership/group
# preservation because some mounted RunPod volumes reject chown/chgrp.
rsync -az --delete --info=progress2 --no-owner --no-group \
  -e "${RSYNC_SSH}" \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude ".env/" \
  --exclude "__pycache__/" \
  --exclude ".pytest_cache/" \
  --exclude ".mypy_cache/" \
  "${PROJECT_ROOT}/" \
  "${RUNPOD_USER}@${RUNPOD_HOST}:${RUNPOD_REMOTE_DIR}/"

echo "Sync complete: ${PROJECT_ROOT} -> ${RUNPOD_USER}@${RUNPOD_HOST}:${RUNPOD_REMOTE_DIR}"
