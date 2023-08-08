#!/bin/bash
#SBATCH --job-name=parallel_job
#SBATCH --ntasks=30       # Total number of tasks/processes
#SBATCH --nodes=30        # Number of nodes to allocate

# Load any necessary modules or activate a virtual environment

# Run the Python script
srun time python -u ${1}
