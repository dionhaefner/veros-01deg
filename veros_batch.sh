#!/bin/bash -l
#
#SBATCH -p hronn
#SBATCH -A ocean
#SBATCH --job-name=veros_01deg
#SBATCH --ntasks=384
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
##SBATCH --constraint=v3
##SBATCH --mail-type=ALL
##SBATCH --mail-user=mail@dionhaefner.de
##SBATCH --output=slurm.out

conda activate veros-fend
ml load gcc h5py/2.10.0_mpich3.3.2_py3.8.6 petsc4py/3.13.0_mpich3.3.2_py3.8.6 mpi4py/3.0.3_mpich3.3.2_py3.8.6
ml unload python

export VEROS_LINEAR_SOLVER=petsc
export OMP_NUM_THREADS=1

srun --mpi=pmi2 -- \
  veros run global_01deg/global_01deg.py -n 24 16 -b jax --float-type float32 -v debug -s identifier 01deg -b jax --force-overwrite
