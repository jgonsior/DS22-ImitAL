#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=48:22:05   # walltime
#SBATCH --ntasks=4      # limit to one node
#SBATCH --cpus-per-task=1  # number of processor cores (i.e. threads)
#SBATCH --mem-per-cpu=1024M   # memory per CPU core
#SBATCH --mail-user=julius.gonsior@tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH -A p_ml_il_lstm_hyper
#SBATCH --output /lustre/ssd/ws/s5968580-IL_TD/slurm_lstm_hyper_output_log.txt
#SBATCH --error /lustre/ssd/ws/s5968580-IL_TD/slurm_lstm_hyper_error_log.txt

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE
OUTFILE="/lustre/ssd/ws/s5968580-IL_TD/lstm_hyperout.txt"
MPLCONFIGDIR=/lustre/ssd/ws/s5968580-IL_TD/cache python3 -m pipenv run python /lustre/ssd/ws/s5968580-IL_TD/imitating-weakal/train_lstm.py --DATA_PATH ../fixed_synthetic_5_50_20_large --STATE_ENCODING listwise --TARGET_ENCODING binary --HYPER_SEARCH --N_ITER 4
exit 0
