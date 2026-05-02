#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  printf 'error: python executable not found: %s\n' "${PYTHON_BIN}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_step() {
  local label="$1"
  shift
  printf '\n==> %s\n' "${label}"
  printf '$ %s\n' "$*"
  if [[ "${CI_LOCAL_DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  "$@"
}

printf 'Running local CI parity checks from %s\n' "${REPO_ROOT}"
printf 'Using PYTHONPATH=%s\n' "${PYTHONPATH}"

run_step "Run unit tests" "${PYTHON_BIN}" -m pytest tests/ -x -q
run_step "Check public artifact references" "${PYTHON_BIN}" scripts/check_public_claim_artifacts.py
run_step "Benchmark run matmul-speedup (1/2)" "${PYTHON_BIN}" -m research_engine.cli benchmark-run matmul-speedup
run_step "Benchmark run matmul-speedup (2/2)" "${PYTHON_BIN}" -m research_engine.cli benchmark-run matmul-speedup
run_step "Export matmul history artifacts" "${PYTHON_BIN}" -m research_engine.cli export-history --benchmark-id matmul-speedup --output-dir .noeris/history
run_step "Check exported history regressions" "${PYTHON_BIN}" scripts/check_history_regressions.py --path .noeris/history/history-regressions.json --summary-path .noeris/history/history-summary.json --benchmark-id matmul-speedup --fail-on-missing

printf '\nLocal CI parity checks completed successfully.\n'
