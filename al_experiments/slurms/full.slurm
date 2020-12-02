#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=100:22:05   # walltime
#SBATCH --ntasks=100      # limit to one node
#SBATCH --cpus-per-task=1  # number of processor cores (i.e. threads)
#SBATCH --mem-per-cpu=1024M   # memory per CPU core
#SBATCH --mail-user=julius.gonsior@tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH -A p_ml_il
#SBATCH --output /lustre/ssd/ws/s5968580-IL_TD2/slurm_out.txt
#SBATCH --error /lustre/ssd/ws/s5968580-IL_TD2/slurm_error.txt

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
OUTFILE="/lustre/ssd/ws/s5968580-IL_TD2/lstm_hyperout.txt"
MPLCONFIGDIR=/lustre/ssd/ws/s5968580-IL_TD2/cache python3 -m pipenv run python /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/full_experiment.py --TRAIN_STATE_ARGSECOND_PROBAS --TRAIN_STATE_ARGTHIRD_PROBAS --TRAIN_STATE_DISTANCES_LAB --TRAIN_STATE_DISTANCES_UNLAB --OUTPUT_DIRECTORY /lustre/ssd/ws/s5968580-IL_TD2/more_data_20_2/ --BASE_PARAM_STRING MORE_DATA --FINAL_PICTURE /lustre/ssd/ws/s5968580-IL_TD2/tmp_all_states/plots_ALL_STATES/ --USER_QUERY_BUDGET_LIMIT 50 --TRAIN_NR_LEARNING_SAMPLES 10000 --TEST_NR_LEARNING_SAMPLES 30000 --TEST_COMPARISONS random --ONLY_TRAINING_DATA --TRAIN_AMOUNT_OF_PEAKED_SAMPLES 20
exit 0
