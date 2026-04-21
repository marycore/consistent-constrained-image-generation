#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ENV_FILE=""
if [[ "${1:-}" == "--env" ]]; then
  ENV_FILE="${2:-}"
  shift 2
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 [--env /path/to/runpod.env] <session_name> \"<command>\""
  exit 1
fi

SESSION_NAME="$1"
shift
REMOTE_CMD="$*"
ENCODED_CMD="$(printf "%s" "${REMOTE_CMD}" | base64 -w0)"

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh" "${ENV_FILE}"

"${SSH_BASE[@]}" "${RUNPOD_USER}@${RUNPOD_HOST}" "cd '${RUNPOD_REMOTE_DIR}' && mkdir -p logs && tmux has-session -t '${SESSION_NAME}' 2>/dev/null && tmux kill-session -t '${SESSION_NAME}' || true && : > logs/${SESSION_NAME}.log && tmux new-session -d -s '${SESSION_NAME}' \"bash -lc 'set -euo pipefail; CMD=\\\$(echo ${ENCODED_CMD} | base64 -d); eval \\\"\\\${CMD}\\\" 2>&1 | tee logs/${SESSION_NAME}.log'\" && tmux ls | grep '${SESSION_NAME}'"

echo "Started tmux session '${SESSION_NAME}' on RunPod."
echo "Attach: ./scripts/runpod/attach_tmux.sh ${SESSION_NAME}"
echo "Logs:   ./scripts/runpod/tail_log.sh ${SESSION_NAME}"
