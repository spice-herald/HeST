#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ./parameters/runs_1000_with_points.txt)
INPUT_FILE=($LINE)
module load conda/latest
conda activate HeST
echo ${INPUT_FILE[1]}
./HeST_basic_script.py --file_path="${INPUT_FILE[0]}" --num_qps="${INPUT_FILE[1]}" --refl_prob=0.45 --evap_eff=0.2,0.0,0.0,0.0,0.0,0.167,