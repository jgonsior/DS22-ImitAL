#!/bin/bash
#SBATCH --time=23:59:59   # walltime
#SBATCH --nodes=1  
#SBATCH --ntasks=1      
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2 
#SBATCH --mem-per-cpu=5250M   # memory per CPU core
#SBATCH --mail-user=julius.gonsior@tu-dresden.de   
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH -A p_ml_il
#SBATCH --output /lustre/ssd/ws/s5968580-IL_TD2/slurm_hybrid-750#0.3_0.0_0.1_0.6_plots_out.txt
#SBATCH --error /lustre/ssd/ws/s5968580-IL_TD2/slurm_hybrid-750#0.3_0.0_0.1_0.6_plots_error.txt


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE



MPLCONFIGDIR=/lustre/ssd/ws/s5968580-IL_TD2/cache python3 -m pipenv run python /lustre/ssd/ws/s5968580-IL_TD2/imitating-weakal/plots.py --OUTPUT_DIRECTORY /lustre/ssd/ws/s5968580-IL_TD2/single_vs_batch --USER_QUERY_BUDGET_LIMIT 50 --TEST_NR_LEARNING_SAMPLES 10000 --TEST_COMPARISONS random uncertainty_max_margin uncertainty_lc uncertainty_entropy --BASE_PARAM_STRING batch_hybrid-750#0.3_0.0_0.1_0.6 --FINAL_PICTURE /lustre/ssd/ws/s5968580-IL_TD2/single_vs_batch/plots_batch_hybrid-750#0.3_0.0_0.1_0.6/ --PLOT_METRIC acc_auc
exit 0
    