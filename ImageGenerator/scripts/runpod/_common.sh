#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${1:-${SCRIPT_DIR}/runpod.env}"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing env file: ${ENV_FILE}"
  echo "Create it from ${SCRIPT_DIR}/runpod.env.example"
  exit 1
fi

# shellcheck source=/dev/null
source "${ENV_FILE}"

: "${RUNPOD_HOST:?RUNPOD_HOST is required}"
: "${RUNPOD_USER:?RUNPOD_USER is required}"
: "${RUNPOD_PORT:?RUNPOD_PORT is required}"
: "${RUNPOD_SSH_KEY:?RUNPOD_SSH_KEY is required}"
: "${RUNPOD_REMOTE_DIR:?RUNPOD_REMOTE_DIR is required}"

SSH_KEY_EXPANDED="${RUNPOD_SSH_KEY/#\~/$HOME}"
SSH_BASE=(ssh -p "${RUNPOD_PORT}" -i "${SSH_KEY_EXPANDED}" -o ServerAliveInterval=30 -o ServerAliveCountMax=120)
RSYNC_SSH="ssh -p ${RUNPOD_PORT} -i ${SSH_KEY_EXPANDED} -o ServerAliveInterval=30 -o ServerAliveCountMax=120"
