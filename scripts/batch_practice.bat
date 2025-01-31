#!/bin/bash

#SBATCH --job-name=singlecpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
module load conda/latest
conda activate HeST
./HeST_basic_script.py --num_qps=1000000 --file_path='/work/pi_shertel_umass_edu/quasiparticle_simulation/sweep_2d/h_15_z_1.5_test.pkl' 

