#!/bin/bash

source ~/veros-env/bin/activate

export MPI4JAX_USE_CUDA_MPI=1
export VEROS_LINEAR_SOLVER=petsc
export VEROS_PETSC_OPTIONS="
    -ksp_type gmres -ksp_rtol 1e-10
    -pc_gamg_agg_nsmooths 1 -pc_gamg_process_eq_limit 10000
"

mpirun -n 16 -- \
    veros run global_01deg/global_01deg.py \
    -b jax --device gpu --force-overwrite
