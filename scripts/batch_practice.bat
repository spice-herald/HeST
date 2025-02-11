#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
module load conda/latest
conda activate HeST
./HeST_basic_script.py --num_qps=2000000 --file_path='/work/pi_shertel_umass_edu/quasiparticle_simulation/waveform_comparison/Test_2.pkl' --evap_eff=0.2,0.4,0.6

