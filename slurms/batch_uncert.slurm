#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=9:59:59   # walltime
#SBATCH --ntasks=8      # limit to one node
#SBATCH --cpus-per-task=1  # number of processor cores (i.e. threads)
#SBATCH --mem-per-cpu=1024M   # memory per CPU core
#SBATCH --mail-user=julius.gonsior@tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH -A p_ml_il
#SBATCH --output /lustre/ssd/ws/s5968580-IL_TD2/slurm_out_uncert.txt
#SBATCH --error /lustre/ssd/ws/s5968580-IL_TD2/slurm_error_uncert.txt

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
TITLE="uncertainty"
OUTFILE="/lustre/ssd/ws/s5968580-IL_TD2/slurm_out$TITLE.txt"
MPLCONFIGDIR=/lustre/ssd/ws/s5968580-IL_TD2/cache python3 -m pipenv run python /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/full_experiment.py --TRAIN_STATE_DISTANCES --TRAIN_STATE_UNCERTAINTIES --TRAIN_STATE_PREDICTED_UNITY --BATCH_MODE --INITIAL_BATCH_SAMPLING_METHOD $TITLE --BASE_PARAM_STRING batch_$TITLE --INITIAL_BATCH_SAMPLING_ARG 200 --OUTPUT_DIRECTORY /lustre/ssd/ws/s5968580-IL_TD2/single_vs_batch/ --FINAL_PICTURE /lustre/ssd/ws/s5968580-IL_TD2/single_vs_batch/plots_batch_$TITLE/ --USER_QUERY_BUDGET_LIMIT 50 --TRAIN_NR_LEARNING_SAMPLES 10 --TEST_NR_LEARNING_SAMPLES 10 --TEST_COMPARISONS random uncertainty_max_margin --PLOT_METRIC acc_auc --INCLUDE_OPTIMAL_IN_PLOT --NR_ANN_HYPER_SEARCH_ITERATIONS 3
exit 0
