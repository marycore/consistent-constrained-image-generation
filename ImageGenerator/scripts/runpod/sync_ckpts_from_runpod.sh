#!/usr/bin/env bash
# Backward-compatible wrapper; use sync_folder_from_runpod.sh directly.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "${SCRIPT_DIR}/sync_folder_from_runpod.sh" "$@"
