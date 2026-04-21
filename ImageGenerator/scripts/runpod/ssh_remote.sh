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

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 [--env /path/to/runpod.env] \"<remote command>\""
  exit 1
fi

REMOTE_CMD="$*"
"${SSH_BASE[@]}" "${RUNPOD_USER}@${RUNPOD_HOST}" "mkdir -p '${RUNPOD_REMOTE_DIR}' && cd '${RUNPOD_REMOTE_DIR}' && ${REMOTE_CMD}"
