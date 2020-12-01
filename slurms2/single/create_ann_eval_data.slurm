#!/bin/bash
#SBATCH --time=23:59:59   # walltime
#SBATCH --ntasks=1      # limit to one node
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1  # number of processor cores (i.e. threads)
#SBATCH --mem-per-cpu=1972M   # memory per CPU core
#SBATCH --mail-user=julius.gonsior@tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH -A p_ml_il
#SBATCH --output /lustre/ssd/ws/s5968580-IL_TD2/slurm_single_ann_eval_out.txt
#SBATCH --error /lustre/ssd/ws/s5968580-IL_TD2/slurm_single_ann_eval_error.txt
#SBATCH --array 0-99

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
i=$(( 100000 + $SLURM_ARRAY_TASK_ID * 10 ))

MPLCONFIGDIR=/lustre/ssd/ws/s5968580-IL_TD2/cache python3 -m pipenv run python /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/full_experiment.py --TRAIN_STATE_DISTANCES --TRAIN_STATE_UNCERTAINTIES --TRAIN_STATE_PREDICTED_UNITY  --INITIAL_BATCH_SAMPLING_METHOD furthest --BASE_PARAM_STRING batch_single --INITIAL_BATCH_SAMPLING_ARG 200 --OUTPUT_DIRECTORY /lustre/ssd/ws/s5968580-IL_TD2/single_vs_batch/ --USER_QUERY_BUDGET_LIMIT 50 --TEST_NR_LEARNING_SAMPLES 10 --SKIP_TRAINING_DATA_GENERATION --STOP_AFTER_ANN_EVAL --TEST_PARALLEL_OFFSET $i

exit 0
    