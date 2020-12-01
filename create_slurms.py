import stat
from string import Template
import argparse
import datetime
import os
import random
import sys
import threading
import warnings

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--TITLE")
parser.add_argument("--TEST_NR_LEARNING_SAMPLES", default=1000, type=int)
parser.add_argument("--TRAIN_NR_LEARNING_SAMPLES", default=1000, type=int)
parser.add_argument("--ITERATIONS_PER_BATCH", default=10, type=int)
parser.add_argument("--OUT_DIR", default="slurms2")
parser.add_argument("--WS_DIR", default="/lustre/ssd/ws/s5968580-IL_TD2")


config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()

config.OUT_DIR = config.OUT_DIR + "/" + config.TITLE

create_ann_training_data = Template(
    """#!/bin/bash
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
#SBATCH --output ${WS_DIR}/slurm_${TITLE}_create_ann_training_data_out.txt
#SBATCH --error ${WS_DIR}/slurm_${TITLE}_create_ann_training_data_error.txt

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$$SLURM_CPUS_ON_NODE
export JOBLIB_TEMP_FOLDER=${WS_DIR}/tmp
MPLCONFIGDIR=${WS_DIR}/cache python3 -m pipenv run python ${WS_DIR}/imitating-weakal/full_experiment.py --TRAIN_STATE_DISTANCES --TRAIN_STATE_UNCERTAINTIES --TRAIN_STATE_PREDICTED_UNITY ${BATCH_MODE} --INITIAL_BATCH_SAMPLING_METHOD $INITIAL_BATCH_SAMPLING_METHOD --BASE_PARAM_STRING batch_$TITLE --INITIAL_BATCH_SAMPLING_ARG 200 --OUTPUT_DIRECTORY ${WS_DIR}/single_vs_batch/ --USER_QUERY_BUDGET_LIMIT 50 --TRAIN_NR_LEARNING_SAMPLES $TRAIN_NR_LEARNING_SAMPLES --ONLY_TRAINING_DATA 
exit 0
"""
)

create_ann_eval_data = Template(
    """#!/bin/bash
#SBATCH --time=23:59:59   # walltime
#SBATCH --ntasks=1      # limit to one node
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1  # number of processor cores (i.e. threads)
#SBATCH --mem-per-cpu=1972M   # memory per CPU core
#SBATCH --mail-user=julius.gonsior@tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH -A p_ml_il
#SBATCH --output ${WS_DIR}/slurm_${TITLE}_ann_eval_out.txt
#SBATCH --error ${WS_DIR}/slurm_${TITLE}_ann_eval_error.txt
#SBATCH --array $START-$END

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$$SLURM_CPUS_ON_NODE
i=$$(( 100000 + $$SLURM_ARRAY_TASK_ID * $ITERATIONS_PER_BATCH ))

MPLCONFIGDIR=${WS_DIR}/cache python3 -m pipenv run python ${WS_DIR}/imitating-weakal/full_experiment.py --TRAIN_STATE_DISTANCES --TRAIN_STATE_UNCERTAINTIES --TRAIN_STATE_PREDICTED_UNITY ${BATCH_MODE} --INITIAL_BATCH_SAMPLING_METHOD $INITIAL_BATCH_SAMPLING_METHOD --BASE_PARAM_STRING batch_$TITLE --INITIAL_BATCH_SAMPLING_ARG 200 --OUTPUT_DIRECTORY ${WS_DIR}/single_vs_batch/ --USER_QUERY_BUDGET_LIMIT 50 --TEST_NR_LEARNING_SAMPLES $ITERATIONS_PER_BATCH --SKIP_TRAINING_DATA_GENERATION --STOP_AFTER_ANN_EVAL --TEST_PARALLEL_OFFSET $$i

exit 0
    """
)


classics = Template(
    """#!/bin/bash
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
#SBATCH --output ${WS_DIR}/slurm_classic_out.txt
#SBATCH --error ${WS_DIR}/slurm_classic_error.txt

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$$SLURM_CPUS_ON_NODE
export JOBLIB_TEMP_FOLDER=${WS_DIR}/tmp
MPLCONFIGDIR=${WS_DIR}/cache python3 -m pipenv run python ${WS_DIR}/imitating-weakal/full_experiment.py --OUTPUT_DIRECTORY ${WS_DIR}/single_vs_batch/ --USER_QUERY_BUDGET_LIMIT 50 --TEST_NR_LEARNING_SAMPLES $TEST_NR_LEARNING_SAMPLES --TEST_COMPARISONS random uncertainty_max_margin uncertainty_lc uncertainty_entropy --SKIP_TRAINING_DATA_GENERATION --SKIP_ANN_EVAL --SKIP_PLOTS
exit 0
"""
)

plots = Template(
    """#!/bin/bash
#SBATCH --time=23:59:59   # walltime
#SBATCH --nodes=1  # number of processor cores (i.e. threads)
#SBATCH --ntasks=1      # limit to one node
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1 # equals 256 threads
#SBATCH --mem-per-cpu=5250M   # memory per CPU core
#SBATCH --mail-user=julius.gonsior@tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH -A p_ml_il
#SBATCH --output ${WS_DIR}/slurm_${TITLE}_plots_out.txt
#SBATCH --error ${WS_DIR}/slurm_${TITLE}_plots_error.txt

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$$SLURM_CPUS_ON_NODE
export JOBLIB_TEMP_FOLDER=${WS_DIR}/tmp
MPLCONFIGDIR=${WS_DIR}/cache python3 -m pipenv run python ${WS_DIR}/imitating-weakal/full_experiment.py --OUTPUT_DIRECTORY ${WS_DIR}/single_vs_batch/ --USER_QUERY_BUDGET_LIMIT 50 --TEST_NR_LEARNING_SAMPLES $TEST_NR_LEARNING_SAMPLES --TEST_COMPARISONS random uncertainty_max_margin uncertainty_lc uncertainty_entropy --BASE_PARAM_STRING batch_$TITLE --SKIP_TRAINING_DATA_GENERATION --SKIP_ANN_EVAL --FINAL_PICTURE ${WS_DIR}/single_vs_batch/plots_batch_${TITLE}/ --PLOT_METRIC acc_auc
exit 0
"""
)


submit_jobs = Template(
    """#!/bin/bash
create_ann_training_data_id=$$(sbatch --parsable ${WS_DIR}/imitating-weakal/${OUT_DIR}/create_ann_training_data.slurm)
create_ann_eval_id=$$(sbatch --parsable --dependency=afterok:$$create_ann_training_data_id ${WS_DIR}/imitating-weakal//${OUT_DIR}/create_ann_eval_data.slurm)
classics_id=$$(sbatch --parsable ${WS_DIR}/imitating-weakal//${OUT_DIR}/classics.slurm)
plots_id=$$(sbatch --parsable --dependency=afterok:$$create_ann_training_data_id:$$create_ann_eval_id:$$classics_id ${WS_DIR}/imitating-weakal//${OUT_DIR}/plots.slurm)
exit 0
"""
)

if not os.path.exists(config.OUT_DIR):
    os.makedirs(config.OUT_DIR)

if config.TITLE == "single":
    BATCH_MODE = ""
    INITIAL_BATCH_SAMPLING_METHOD = "furthest"
else:
    INITIAL_BATCH_SAMPLING_METHOD = config.TITLE
    BATCH_MODE = "--BATCH_MODE"

with open(config.OUT_DIR + "/create_ann_training_data.slurm", "w") as f:
    f.write(
        create_ann_training_data.substitute(
            WS_DIR=config.WS_DIR,
            TITLE=config.TITLE,
            INITIAL_BATCH_SAMPLING_METHOD=INITIAL_BATCH_SAMPLING_METHOD,
            TRAIN_NR_LEARNING_SAMPLES=config.TRAIN_NR_LEARNING_SAMPLES,
            BATCH_MODE=BATCH_MODE,
        )
    )

with open(config.OUT_DIR + "/create_ann_eval_data.slurm", "w") as f:
    START = 0
    END = int(config.TEST_NR_LEARNING_SAMPLES / config.ITERATIONS_PER_BATCH) - 1
    f.write(
        create_ann_eval_data.substitute(
            WS_DIR=config.WS_DIR,
            TITLE=config.TITLE,
            INITIAL_BATCH_SAMPLING_METHOD=INITIAL_BATCH_SAMPLING_METHOD,
            START=START,
            END=END,
            ITERATIONS_PER_BATCH=config.ITERATIONS_PER_BATCH,
            BATCH_MODE=BATCH_MODE,
        )
    )
with open(config.OUT_DIR + "/classics.slurm", "w") as f:
    f.write(
        classics.substitute(
            WS_DIR=config.WS_DIR,
            TITLE=config.TITLE,
            TEST_NR_LEARNING_SAMPLES=config.TEST_NR_LEARNING_SAMPLES,
        )
    )

with open(config.OUT_DIR + "/plots.slurm", "w") as f:
    f.write(
        plots.substitute(
            WS_DIR=config.WS_DIR,
            TITLE=config.TITLE,
            TEST_NR_LEARNING_SAMPLES=config.TEST_NR_LEARNING_SAMPLES,
        )
    )

with open(config.OUT_DIR + "/submit_jobs.sh", "w") as f:
    f.write(
        submit_jobs.substitute(
            WS_DIR=config.WS_DIR, OUT_DIR=config.OUT_DIR, TITLE=config.TITLE
        )
    )
st = os.stat(config.OUT_DIR + "/submit_jobs.sh")
os.chmod(config.OUT_DIR + "/submit_jobs.sh", st.st_mode | stat.S_IEXEC)
