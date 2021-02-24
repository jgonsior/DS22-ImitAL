#!/bin/bash
#SBATCH --time=23:59:59   # walltime
#SBATCH --nodes=1  
#SBATCH --ntasks=1      
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1 
#SBATCH --mem-per-cpu=2583M   # memory per CPU core
#SBATCH --mail-user=julius.gonsior@tu-dresden.de   
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH -A p_ml_il
#SBATCH --output /lustre/ssd/ws/s5968580-IL_TD2/slurm_hybrid-500#0.2_0.0_0.2_0.2_ann_eval_data_out.txt
#SBATCH --error /lustre/ssd/ws/s5968580-IL_TD2/slurm_hybrid-500#0.2_0.0_0.2_0.2_ann_eval_data_error.txt
#SBATCH --array 0-999

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

i=$(( 100000 + $SLURM_ARRAY_TASK_ID * 10 ))

MPLCONFIGDIR=/lustre/ssd/ws/s5968580-IL_TD2/cache python3 -m pipenv run python /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/ann_eval_data.py --TRAIN_STATE_DISTANCES --TRAIN_STATE_UNCERTAINTIES --TRAIN_STATE_PREDICTED_UNITY --BATCH_MODE --INITIAL_BATCH_SAMPLING_METHOD hybrid --BASE_PARAM_STRING batch_hybrid-500#0.2_0.0_0.2_0.2 --INITIAL_BATCH_SAMPLING_ARG 500 --OUTPUT_DIRECTORY /lustre/ssd/ws/s5968580-IL_TD2/single_vs_batch/ --TOTAL_BUDGET 50 --TEST_NR_LEARNING_SAMPLES 10 --INITIAL_BATCH_SAMPLING_HYBRID_UNCERT 0.2 --INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY 0.2 --INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST 0.2 --INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB 0.0 --TEST_RANDOM_ID_OFFSET $i
exit 0
    