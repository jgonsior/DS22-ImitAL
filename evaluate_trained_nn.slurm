#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=23:59:55   # walltime
#SBATCH --ntasks=1      # limit to one node
#SBATCH --cpus-per-task=1  # number of processor cores (i.e. threads)
#SBATCH --mem-per-cpu=1024M   # memory per CPU core
#SBATCH --mail-user=julius.gonsior@tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH -A p_ml_il
#SBATCH --output /lustre/ssd/ws/s5968580-IL_TD/slurm_output_log.txt
#SBATCH --error /lustre/ssd/ws/s5968580-IL_TD/slurm_error_log.txt
#SBATCH --array 0-40

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
OUTFILE="" #/lustre/ssd/ws/s5968580-IL_TD/out"
for i in $(seq $((1 + 250*$SLURM_ARRAY_TASK_ID -250)) $((250 + 250*$SLURM_ARRAY_TASK_ID -250))); 
do
    MPLCONFIGDIR=/lustre/ssd/ws/s5968580-IL_TD/cache python3 -m pipenv run python /lustre/ssd/ws/s5968580-IL_TD/imitating-weakal/single_al_cycle.py --NN_BINARY ../variable_synthetic_1_50_10/trained_listwise_binary.pickle --OUTPUT_DIRECTORY NN_BINARY --SAMPLING trained_nn --CLUSTER dummy --NR_QUERIES_PER_ITERATION 1 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 50 --RANDOM_SEED $i --N_JOBS 1 
done 0
