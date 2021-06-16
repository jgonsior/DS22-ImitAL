#!/bin/bash
#SBATCH --time=1:00:00   # walltime
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2583M   # memory per CPU core
#SBATCH --mail-user=julius.gonsior@tu-dresden.de
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH -A p_ml_il
#SBATCH --output /lustre/ssd/ws/s5968580-praticAl/run_how_to_out.txt
#SBATCH --error /lustre/ssd/ws/s5968580-praticAl/run_how_to_error.txt
#SBATCH --array 0-10000

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

#i=$(( 0 + $SLURM_ARRAY_TASK_ID * 1 ))

module load Python/3.8.6
MPLCONFIGPATH=/lustre/ssd/ws/s5968580-praticAl/cache python3 -m pipenv run python /lustre/ssd/ws/s5968580-praticAl/code/run_how_to_combine_ws_and_al_labels_experiment.py --OUTPUT_PATH /lustre/ssd/ws/s5968580-praticAl/exp_results/run_how_to --DATASETS_PATH /lustre/ssd/ws/s5968580-praticAl/datasets --STAGE JOB --JOB_ID $SLURM_ARRAY_TASK_ID
exit 0