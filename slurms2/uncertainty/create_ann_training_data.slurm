#!/bin/bash
#SBATCH --time=23:59:59   # walltime
#SBATCH --nodes=1  # number of processor cores (i.e. threads)
#SBATCH --ntasks=1      # limit to one node
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=128 # equals 256 threads
#SBATCH --mem-per-cpu=1972M   # memory per CPU core
#SBATCH -p romeo
#SBATCH --mail-user=julius.gonsior@tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH -A p_ml_il
#SBATCH --output /lustre/ssd/ws/s5968580-IL_TD2/slurm_uncertainty_create_ann_training_data_out.txt
#SBATCH --error /lustre/ssd/ws/s5968580-IL_TD2/slurm_uncertainty_create_ann_training_data_error.txt

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
export JOBLIB_TEMP_FOLDER=/lustre/ssd/ws/s5968580-IL_TD2/tmp
MPLCONFIGDIR=/lustre/ssd/ws/s5968580-IL_TD2/cache python3 -m pipenv run python /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/full_experiment.py --TRAIN_STATE_DISTANCES --TRAIN_STATE_UNCERTAINTIES --TRAIN_STATE_PREDICTED_UNITY --BATCH_MODE --INITIAL_BATCH_SAMPLING_METHOD uncertainty --BASE_PARAM_STRING batch_uncertainty --INITIAL_BATCH_SAMPLING_ARG 200 --OUTPUT_DIRECTORY /lustre/ssd/ws/s5968580-IL_TD2/single_vs_batch/ --USER_QUERY_BUDGET_LIMIT 50 --TRAIN_NR_LEARNING_SAMPLES 1000 --ONLY_TRAINING_DATA 
exit 0
