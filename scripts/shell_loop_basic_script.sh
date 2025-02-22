#!/bin/bash

#SBATCH -N 1 -c 12
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32768 
task_id=$SLURM_TASK_ID  # Assigning to a variable to ensure it's treated as a number
start=$((task_id * 10))
end=$(((task_id + 1) * 10))
for ((i=$start; i<$end; i++)); do
    echo $i
done
# ./HeST_basic_script.py --file_path="${INPUT_FILE[0]}" --num_qps=1000000  --refl_prob=0.15 --evap_eff="${INPUT_FILE[1]}"