#!/bin/bash

#SBATCH -N 1 -c 12
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ./parameters/height_sweep_0_5.txt)
INPUT_FILE=($LINE)
module load conda/latest
conda activate HeST
./HeST_basic_script.py --file_path="${INPUT_FILE[0]}" --num_qps="${INPUT_FILE[1]}"