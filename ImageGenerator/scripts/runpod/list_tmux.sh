#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ENV_FILE=""
if [[ "${1:-}" == "--env" ]]; then
  ENV_FILE="${2:-}"
  shift 2
fi

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh" "${ENV_FILE}"

"${SSH_BASE[@]}" "${RUNPOD_USER}@${RUNPOD_HOST}" "cd '${RUNPOD_REMOTE_DIR}' && tmux ls || true"
