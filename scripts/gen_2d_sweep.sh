#!/bin/bash

#SBATCH -N 1 -c 12
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
line_number=$(($SLURM_ARRAY_TASK_ID))
echo $line_number
LINE=$(sed -n "${line_number}p" ./parameters/detector_geometry_sweep_2.txt)

INPUT_FILE=($LINE)
echo "${INPUT_FILE[0]}"
echo "${INPUT_FILE[1]}"
echo "${INPUT_FILE[2]}"
module load conda/latest
conda activate HeST
./optimization_runs.py --file_path="${INPUT_FILE[0]}" --num_qps=2000000  --refl_prob=0.15 --setup="${INPUT_FILE[1]}" --pos="${INPUT_FILE[2]}"
# ./HeST_basic_script.py --file_path="${INPUT_FILE[0]}" --num_qps=5000000  --refl_prob=0.45 --evap_eff=0.2,0.0,0.0,0.0,0.0,0.167, --pos="${INPUTE_FILE[1]}"
# ./HeST_review_runs.py --file_path="${INPUT_FILE[0]}" --num_qps=5000000  --refl_prob=0.45 --evap_eff=0.2,0.0,0.0,0.0,0.0,0.167,  --pos="${INPUT_FILE[1]}"

