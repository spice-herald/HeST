#!/bin/bash

#SBATCH -N 1 -c 12
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32768 
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ./parameters/sweep_mom_bins.txt)
INPUT_FILE=($LINE)
module load conda/latest
conda activate HeST
# ./HeST_basic_script.py --file_path="${INPUT_FILE[0]}" --num_qps=1000000  --refl_prob="${INPUT_FILE[1]}" --evap_eff="${INPUT_FILE[2]}"
echo "${INPUT_FILE[0]}"
./HeST_basic_script.py --file_path="${INPUT_FILE[0]}" --num_qps=1000000  --refl_prob=0.15 --evap_eff="${INPUT_FILE[1]}"