#!/usr/bin/env bash
# =============================================================================
# LongBench v2 evaluation against an SFI-enabled vLLM server.
#
# Prerequisites:
#   1. An SFI server is running (quality settings):
#        K_HEAD=4096 bash scripts/serve_sparse.sh <GPU> <MODEL_PATH> 8000
#   2. The official LongBench v2 harness is cloned somewhere:
#        git clone https://github.com/THUDM/LongBench.git
#      and its config/model2path.json maps <MODEL_NAME> to your local model
#      path (the mapped path doubles as the API model name that vLLM serves).
#
# Usage:
#   LONGBENCH_ROOT=/path/to/LongBench \
#   bash scripts/run_longbench_v2.sh <MODEL_NAME> [N_PROC] [PORT]
#
#   N_PROC  client-side concurrency (= decode batch pressure on the server)
#
# Sub-sampling trick (the harness has no --sample flag): pre-write stub rows
#   {"_id": "<sample-id>", "stub": true}
# into the output file for every sample you want to SKIP; the harness's
# resume mechanism only runs the remaining ones.
#
# Judging: response completeness (no empty/error rows) + accuracy vs the dense
# baseline run of the same model (serve with ENABLE_SPARSE controller removed,
# or compare against published dense numbers).
# =============================================================================
set -euo pipefail

MODEL_NAME="${1:?usage: LONGBENCH_ROOT=... run_longbench_v2.sh <MODEL_NAME> [N_PROC] [PORT]}"
N_PROC="${2:-4}"
PORT="${3:-8000}"
LONGBENCH_ROOT="${LONGBENCH_ROOT:?set LONGBENCH_ROOT to your LongBench v2 checkout}"

if ! curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
  echo "ERROR: no server on :${PORT}. Start scripts/serve_sparse.sh first." >&2
  exit 1
fi
echo "==> server on :${PORT} is up; running LongBench v2 pred (n_proc=${N_PROC})"
echo "    (point the harness's API URL at http://127.0.0.1:${PORT}/v1 if your"
echo "     LongBench checkout reads it from a constant or env in pred.py)"

cd "${LONGBENCH_ROOT}"
python pred.py --model "${MODEL_NAME}" --n_proc "${N_PROC}"
