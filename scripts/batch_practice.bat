#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
module load conda/latest
conda activate HeST

# remaining points to run (0, .948, 0.5): done, (0.84, -0.42, 1.75): done, (-0.84, -0.42, 3.5): done
./HeST_review_runs.py --num_qps=2000000 --file_path='/work/pi_shertel_umass_edu/quasiparticle_simulation/waveform_comparison/review_plots/pretty_high.pkl' --refl_prob=0.15 --evap_eff=0.0,0.0,0.125,0.0,0.5,0.0, --pos=-0.8,-0.42,3.5
