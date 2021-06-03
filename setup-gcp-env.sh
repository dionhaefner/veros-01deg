#! /bin/bash

set -e


# install base dependencies
sudo apt-get update
sudo apt-get install git build-essentials libopenblas-dev liblapack-dev


# install Python
sudo apt-get install -y python3 python3-dev python3-pip python3-virtualenv
virtualenv ~/veros-env --python=python3
source ~/veros-env/bin/activate
pip install numpy cython


# install CUDA
sudo wget -O /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get install -y cuda
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export CUDA_ROOT=/usr/local/cuda' >> ~/.bashrc
source ~/.bashrc


# install OpenMPI and mpi4py
cd $HOME
wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.5.tar.gz
tar xvzf openmpi-4.0.5.tar.gz
cd openmpi-4.0.5
./configure --with-cuda=/usr/local/cuda
make -j 12
sudo make install
pip install mpi4py --no-binary mpi4py


# install HDF5 and h5py
cd $HOME
wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.12/hdf5-1.12.0/src/hdf5-1.12.0.tar.gz
tar xvzf hdf5-1.12.0.tar.gz
cd hdf5-1.12.0/
./configure --enable-parallel --enable-shared --enable-build-mode=production
make -j 12
make install
CC=mpicc HDF5_MPI="ON" HDF5_DIR="$HOME/hdf5-1.12.0/hdf5" pip install h5py --no-binary h5py


# install PETSc and petsc4py
cd $HOME
git clone -b v3.15 https://gitlab.com/petsc/petsc.git petsc
cd petsc

./configure --with-debugging=0 \
    COPTFLAGS='-O3 -march=native -mtune=native' \
    CXXOPTFLAGS='-O3 -march=native -mtune=native' \
    FOPTFLAGS='-O3 -march=native -mtune=native' \
    CUDAOPTFLAGS='-O3' \
    --with-cuda --with-precision=double --download-hypre

make PETSC_DIR=`pwd` PETSC_ARCH=arch-linux-c-opt all
PETSC_DIR=`pwd` PETSC_ARCH=arch-linux-c-opt all pip install petsc4py==3.15 --no-binary petsc4py


# install JAX and mpi4jax
pip install --upgrade jax jaxlib==0.1.67+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
pip install mpi4jax


# install Veros
cd $HOME
git clone https://github.com/team-ocean/veros.git -b master  # TODO: replace with release
cd veros
CUDA_COMPUTE_CAPABILITY=80 pip install -e .
