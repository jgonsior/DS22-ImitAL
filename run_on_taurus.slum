#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=00:00:05   # walltime
#SBATCH --ntasks=2      # limit to one node
#SBATCH --cpus-per-task=1  # number of processor cores (i.e. threads)
#SBATCH --mem-per-cpu=1024M   # memory per CPU core
#SBATCH --mail-user=julius.gonsior@tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH -A p_ml_il
#SBATCH --output /lustre/ssd/ws/s5968580-IL_TD/imitating-weakal/test_data/output_log.txt
#SBATCH --error /lustre/ssd/ws/s5968580-IL_TD/imitating-weakal/test_data/error_log.txt


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
OUTFILE="" #/lustre/ssd/ws/s5968580-IL_TD/out"
MPLCONFIGDIR=/lustre/ssd/ws/s5968580-IL_TD/cache python3 -m pipenv run python /lustre/ssd/ws/s5968580-IL_TD/imitating-weakal/imit_training.py --DATASETS_PATH ../datasets --OUTPUT_DIRECTORY test_data  --CLUSTER dummy --NR_QUERIES_PER_ITERATION 5 --DATASET_NAME synthetic --START_SET_SIZE 1 --USER_QUERY_BUDGET_LIMIT 100 --RANDOM_SEED -2 --N_JOBS 1 --AMOUNT_OF_PEAKED_OBJECTS 4 --MAX_AMOUNT_OF_WS_PEAKS 0 --AMOUNT_OF_LEARN_ITERATIONS 500000000 --USE_OPTIMAL_ONLY > /lustre/ssd/ws/s5968580-IL_TD/imitating-weakal/test_data/output.txt
exit 0
