#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=23:59:59   # walltime
#SBATCH --ntasks=1      # limit to one node
#SBATCH -n 1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task=24  # number of processor cores (i.e. threads)
#SBATCH --mem-per-cpu=5250M   # memory per CPU core
#SBATCH --mail-user=julius.gonsior@tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH -A p_ml_il
#SBATCH --output /lustre/ssd/ws/s5968580-IL_TD2/slurm_lstm_hyper_output_log.txt
#SBATCH --error /lustre/ssd/ws/s5968580-IL_TD2/slurm_lstm_hyper_error_log.txt

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
OUTFILE="/lustre/ssd/ws/s5968580-IL_TD2/lstm_hyperout.txt"
MPLCONFIGDIR=/lustre/ssd/ws/s5968580-IL_TD2/cache python3 -m pipenv run python /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/train_lstm.py --DATA_PATH /lustre/ssd/ws/s5968580-IL_TD2/single_vs_batch/batch_furthest_lab --STATE_ENCODING listwise --TARGET_ENCODING binary --HYPER_SEARCH --N_ITER 10
exit 0
