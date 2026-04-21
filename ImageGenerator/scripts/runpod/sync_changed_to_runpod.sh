#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/_common.sh" "${1:-}"

"${SSH_BASE[@]}" "${RUNPOD_USER}@${RUNPOD_HOST}" "mkdir -p '${RUNPOD_REMOTE_DIR}'"

tmp_changed="$(mktemp)"
tmp_deleted="$(mktemp)"
cleanup() {
  rm -f "${tmp_changed}" "${tmp_deleted}"
}
trap cleanup EXIT

git_root="$(git -C "${PROJECT_ROOT}" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${git_root}" ]]; then
  echo "Project is not inside a git repository. Use sync_to_runpod.sh for full sync."
  exit 1
fi

rel_prefix=""
if [[ "${PROJECT_ROOT}" != "${git_root}" ]]; then
  rel_prefix="${PROJECT_ROOT#${git_root}/}/"
fi

# Changed tracked + staged + untracked files.
{
  git -C "${git_root}" diff --name-only
  git -C "${git_root}" diff --cached --name-only
  git -C "${git_root}" ls-files --others --exclude-standard
} | if [[ -n "${rel_prefix}" ]]; then
      awk -v p="${rel_prefix}" 'index($0, p) == 1 {print substr($0, length(p)+1)}'
    else
      awk 'NF {print}'
    fi | sort -u > "${tmp_changed}"

# Deleted tracked files (unstaged + staged) to remove on remote.
{
  git -C "${git_root}" diff --name-only --diff-filter=D
  git -C "${git_root}" diff --cached --name-only --diff-filter=D
} | if [[ -n "${rel_prefix}" ]]; then
      awk -v p="${rel_prefix}" 'index($0, p) == 1 {print substr($0, length(p)+1)}'
    else
      awk 'NF {print}'
    fi | sort -u > "${tmp_deleted}"

# Keep exclusions aligned with full sync behavior.
if [[ -s "${tmp_changed}" ]]; then
  filtered_changed="$(mktemp)"
  trap 'rm -f "${tmp_changed}" "${tmp_deleted}" "${filtered_changed}"' EXIT
  exclude_re='^([.]git/|[.]venv/|[.]env/|__pycache__/|[.]pytest_cache/|[.]mypy_cache/)'
  if command -v rg >/dev/null 2>&1; then
    rg -v "${exclude_re}" "${tmp_changed}" > "${filtered_changed}" || true
  else
    # Keep this script usable on hosts without ripgrep.
    awk -v re="${exclude_re}" '!($0 ~ re)' "${tmp_changed}" > "${filtered_changed}" || true
  fi

  if [[ -s "${filtered_changed}" ]]; then
    rsync -az --info=progress2 --no-owner --no-group \
      -e "${RSYNC_SSH}" \
      --files-from="${filtered_changed}" \
      "${PROJECT_ROOT}/" \
      "${RUNPOD_USER}@${RUNPOD_HOST}:${RUNPOD_REMOTE_DIR}/"
    echo "Synced changed files to RunPod."
  else
    echo "No changed files to sync after exclusions."
  fi
else
  echo "No changed files to sync."
fi

if [[ -s "${tmp_deleted}" ]]; then
  while IFS= read -r rel; do
    [[ -z "${rel}" ]] && continue
    case "${rel}" in
      .git/*|.venv/*|.env/*|__pycache__/*|.pytest_cache/*|.mypy_cache/*)
        continue
        ;;
    esac
    "${SSH_BASE[@]}" "${RUNPOD_USER}@${RUNPOD_HOST}" "rm -rf '${RUNPOD_REMOTE_DIR}/${rel}'"
  done < "${tmp_deleted}"
  echo "Removed deleted tracked files on RunPod."
fi
