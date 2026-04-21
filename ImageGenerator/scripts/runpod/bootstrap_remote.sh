#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

ENV_FILE=""
if [[ "${1:-}" == "--env" ]]; then
  ENV_FILE="${2:-}"
fi

"${SCRIPT_DIR}/ssh_remote.sh" ${ENV_FILE:+--env "${ENV_FILE}"} \
  "python3 -m venv .venv && source .venv/bin/activate && python -m pip install --upgrade pip && pip install -r requirements.txt && if ! command -v tmux >/dev/null 2>&1; then apt-get update && apt-get install -y tmux; fi"

echo "Remote bootstrap complete."
