#!/bin/bash
# -----------------------------------------------------------------------------------
# Sample Slurm job script
#
#   for TACC Frontera CLX nodes
#
#   *** OpenMP Job in Small Queue ***
# 
# Last revised: July 6, 2021
#
# Notes:
#
#   -- Launch this script by executing
#   -- Copy/edit this script as desired.  Launch by executing
#      "sbatch clx.openmp.slurm" on a Frontera login node.
#
#   -- OpenMP codes run on a single node (upper case N = 1).
#        OpenMP ignores the value of lower case n,
#        but slurm needs a plausible value to schedule the job.
#
#   -- Default value of OMP_NUM_THREADS is 1; be sure to change it!
#
#   -- Increase thread count gradually while looking for optimal setting.
#        If there is sufficient memory available, the optimal setting
#        is often 56 (1 thread per core) but may be higher.

#----------------------------------------------------

#SBATCH -J tclr_model_brute           # Job name
#SBATCH -o slurm_out/%x.o%j       # Name of stdout output file
#SBATCH -e slurm_out/%x.e%j       # Name of stderr error file
#SBATCH -p development           # Queue (partition) name
#SBATCH -N 10             # Total # of nodes (must be 1 for OpenMP)
#SBATCH -n 10               # Total # of mpi tasks (should be 1 for OpenMP)
#SBATCH -c 56		   # # of cpus per task
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=lcford2@ncsu.edu
#SBATCH -A ECS21018        # Project/Allocation name (req'd if you have more than 1)

# load remora module
module load remora/1.8.5

# Any other commands must follow all #SBATCH directives...
module list
cd $WORK/predict-release/python_scripts
pwd
date

# Set thread count (default value is 1)...
export OMP_NUM_THREADS=56   # this is 1 thread/core; may want to start lower

# activate python environment
source ../sry-env/bin/activate

remora ibrun python run_tclr_model.py
