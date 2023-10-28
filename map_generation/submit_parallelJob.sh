#!/bin/bash
#SBATCH --job-name=parallel_job
#SBATCH --ntasks=1       # Total number of tasks/processes
#SBATCH --nodes=1        # Number of nodes to allocate
#SBATCH --time 24:00:00
#SBATCH --account=penningb0

# Load any necessary modules or activate a virtual environment
z_slice=${2}
script=${1}

# Run the Python script
time python -u ${script} ${z_slice}
