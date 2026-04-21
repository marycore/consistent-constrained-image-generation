#!/usr/bin/env bash
# Pull one folder from RunPod -> local (default: ckpts; merge by default, no --delete).
# Usage:
#   ./scripts/runpod/sync_folder_from_runpod.sh
#   ./scripts/runpod/sync_folder_from_runpod.sh --folder ckpts
#   ./scripts/runpod/sync_folder_from_runpod.sh --folder:outputs
#   ./scripts/runpod/sync_folder_from_runpod.sh --folder outputs --delete
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

DELETE_ARGS=()
ENV_FILE="${SCRIPT_DIR}/runpod.env"
FOLDER="ckpts"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --delete)
      DELETE_ARGS=(--delete)
      shift
      ;;
    --folder)
      FOLDER="${2:?missing value after --folder}"
      shift 2
      ;;
    --folder:*)
      FOLDER="${1#--folder:}"
      shift
      ;;
    --env)
      ENV_FILE="${2:?missing path after --env}"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--env /path/to/runpod.env] [--folder <name>|--folder:<name>] [--delete]" >&2
      exit 1
      ;;
  esac
done

# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh" "${ENV_FILE}"

FOLDER="${FOLDER#/}"
FOLDER="${FOLDER%/}"
if [[ -z "${FOLDER}" ]]; then
  echo "Folder cannot be empty. Use --folder <name> or --folder:<name>." >&2
  exit 1
fi
if [[ "${FOLDER}" == *".."* ]]; then
  echo "Folder cannot contain '..': ${FOLDER}" >&2
  exit 1
fi

mkdir -p "${PROJECT_ROOT}/${FOLDER}"

rsync -az --info=progress2 --no-owner --no-group \
  "${DELETE_ARGS[@]}" \
  -e "${RSYNC_SSH}" \
  "${RUNPOD_USER}@${RUNPOD_HOST}:${RUNPOD_REMOTE_DIR}/${FOLDER}/" \
  "${PROJECT_ROOT}/${FOLDER}/"

echo "Pulled ${FOLDER}/ from ${RUNPOD_USER}@${RUNPOD_HOST}:${RUNPOD_REMOTE_DIR}/${FOLDER}/ -> ${PROJECT_ROOT}/${FOLDER}/"
