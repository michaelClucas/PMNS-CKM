#!/usr/bin/env bash
# Stage 2 (after datasets and ql fits exist). Two lanes, <= 4 cores in total:
#   lane A (2 torch threads): broad theta-NPEs, then broad PMNS predictive flows
#   lane B (1 core): DE-MC posteriors (ql then all), then analyses
# Run from the experiment folder: bash run_stage2.sh   (logs append to run.log)
set -u
PY=../../.venv/Scripts/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
(
  for v in so10_I so10_II; do $PY train_npe.py broad_$v ds_broad_$v.npz $v theta all 11 0; done
  for v in so10_II so10_I; do
    $PY train_npe.py pred_ql_$v ds_broad_$v.npz $v pmns ql 4 0
    $PY train_npe.py pred_mass_$v ds_broad_$v.npz $v pmns mass 4 0
  done
  echo "LANE_A_DONE"
) >> run.log 2>&1 &
(
  for v in so10_I so10_II; do $PY mcmc.py $v ql 20000 3; done
  for v in so10_I so10_II; do $PY mcmc.py $v all 20000 4; done
  echo "LANE_B_DONE"
) >> run.log 2>&1 &
wait
