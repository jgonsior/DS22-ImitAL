#!/bin/bash
#SBATCH --time=23:59:59   # walltime
#SBATCH --nodes=1  
#SBATCH --ntasks=1      
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8 
#SBATCH --mem-per-cpu=5250M   # memory per CPU core
#SBATCH --mail-user=julius.gonsior@tu-dresden.de   
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH -A p_ml_il
#SBATCH --output ../datasets/short_test/slurm_hybrid_train_ann_out.txt
#SBATCH --error ../datasets/short_test/slurm_hybrid_train_ann_error.txt


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE



MPLCONFIGDIR=../datasets/short_test/cache python3 -m pipenv run python ../datasets/short_test/imitating-weakal/train_ann.py --OUTPUT_DIRECTORY ../datasets/short_test/ --BASE_PARAM_STRING batch_hybrid
exit 0
    