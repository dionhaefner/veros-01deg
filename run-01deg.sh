#!/bin/bash

source ~/veros-env/bin/activate

export MPI4JAX_USE_CUDA_MPI=1

mpirun -n 16 veros run global_01deg/global_01deg.py --force-overwrite -n 4 4
