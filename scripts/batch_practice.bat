#!/bin/bash

#SBATCH --job-name=singlecpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
module load conda/latest
conda activate HeST
pwd
./HeST_basic_script.py --num_qps=1000000 --file_path='/work/pi_shertel_umass_edu/many_run_data/more_quasiparticles.pkl'

