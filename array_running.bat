#!/bin/bash

#SBATCH -N 1 -c 12
INPUT_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ./parameters/runs_20.txt)
module load conda/latest
conda activate HeST
./HeST_basic_script.py --num_qps=100000 --file_path="${INPUT_FILE}"