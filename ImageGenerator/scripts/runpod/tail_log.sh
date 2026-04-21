#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ENV_FILE=""
if [[ "${1:-}" == "--env" ]]; then
  ENV_FILE="${2:-}"
  shift 2
fi

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 [--env /path/to/runpod.env] <session_name>"
  exit 1
fi

SESSION_NAME="$1"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh" "${ENV_FILE}"

"${SSH_BASE[@]}" -t "${RUNPOD_USER}@${RUNPOD_HOST}" "cd '${RUNPOD_REMOTE_DIR}' && tail -f 'logs/${SESSION_NAME}.log'"
