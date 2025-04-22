#!/bin/bash

#SBATCH -N 1 -c 12
INPUT_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ./parameters/sample_1000.txt)
module load conda/latest
conda activate HeST
./scripts/HeST_basic_script.py --num_qps=5000000 --file_path="${INPUT_FILE}" 