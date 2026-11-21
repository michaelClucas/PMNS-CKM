#!/bin/sh
# usage: sh run_refits.sh CONFIG... ; runs sequentially, pinned, single-threaded, appends to run.log
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for c in "$@"; do
  echo "== refit $c start $(date -u +%FT%TZ)" >> experiments/20261120-047-skeptic-audit/run.log
  .venv/Scripts/python tools/pin.py .venv/Scripts/python experiments/20261120-047-skeptic-audit/refit_lbfgsb.py "$c" 50 16 >> experiments/20261120-047-skeptic-audit/run.log 2>&1
  echo "== refit $c end $(date -u +%FT%TZ)" >> experiments/20261120-047-skeptic-audit/run.log
done
