#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=48:22:05   # walltime
#SBATCH --ntasks=1      # limit to one node
#SBATCH --cpus-per-task=1  # number of processor cores (i.e. threads)
#SBATCH --mem-per-cpu=1024M   # memory per CPU core
#SBATCH --mail-user=julius.gonsior@tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH -A p_ml_il
#SBATCH --output /lustre/ssd/ws/s5968580-IL_TD2/slurm_output_exp_init_sampling.txt
#SBATCH --error /lustre/ssd/ws/s5968580-IL_TD2/slurm_error_exp_init_sampling.txt
#SBATCH --array 1-99

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
OUTFILE="" 
MPLCONFIGDIR=/lustre/ssd/ws/s5968580-IL_TD2/cache python3 -m pipenv run python /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/exp_create_init_sampling_methods.py $i /lustre/ssd/ws/s5968580-IL_TD2/datasets/init_sampling
exit 0
