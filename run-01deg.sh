#!/bin/bash

source ~/veros-env/bin/activate

mpirun -n 16 veros run global_01deg/global_01deg.py --force-overwrite -n 4 4
sudo shutdown now
