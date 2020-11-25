#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=0:59:59   # walltime
#SBATCH --ntasks=10      # limit to one node
#SBATCH --cpus-per-task=1  # number of processor cores (i.e. threads)
#SBATCH --mem-per-cpu=1024M   # memory per CPU core
#SBATCH --mail-user=julius.gonsior@tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH -A p_ml_il
#SBATCH --output /lustre/ssd/ws/s5968580-IL_TD2/slurm_out_test.txt
#SBATCH --error /lustre/ssd/ws/s5968580-IL_TD2/slurm_error_test.txt

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
TITLE="test"
OUTFILE="/lustre/ssd/ws/s5968580-IL_TD2/slurm_out$TITLE.txt"
MPLCONFIGDIR=/lustre/ssd/ws/s5968580-IL_TD2/cache python3 -m pipenv run python /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/slurm/slurm_test.py
exit 0
