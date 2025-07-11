#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
module load conda/latest
conda activate HeST

./HeST_review_runs.py --num_qps=6000000 --file_path='/work/pi_shertel_umass_edu/quasiparticle_simulation/debug/understanding_late_central/no_specular_change.pkl' --refl_prob=0.15 --evap_eff=0.0,0.0,0.125,0.0,0.5,0.0, --pos=0.0,1.75,0.5, --diff_prob=0.95
