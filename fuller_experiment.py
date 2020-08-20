import pandas as pd
import os
import subprocess
import multiprocessing
from pathlib import Path
from joblib import Parallel, delayed
from active_learning.experiment_setup_lib import standard_config

# example:
# python fuller_experiment.py --TRAIN_AMOUNT_OF_FEATURES 2 means that we call it once with this, and once with the default values, and compare both
# python fuller_experiment.py --TRAIN_AMOUNT_OF_FEATURES 2 3 means that we call it once with 2, and once with 3, and once with default values
# python fuller_experiment.py --TRAIN_HYPERCUBE means that we call it once with, and once without
# python fuller_experiment --TRAIN_HYPERCUBE --TEST_OLD_SYNTHETIC_PARAMS means that we have a 2x2 comparisn

config = standard_config(
    [
        (["--RANDOM_SEED"], {"default": 1}),
        (["--LOG_FILE"], {"default": "log.txt"}),
        (["--OUTPUT_DIRECTORY"], {"default": "/tmp"}),
        (["--TRAIN_VARIABLE_DATASET"], {"default": "experiment"}),
        (["--TRAIN_NR_LEARNING_SAMPLES"], {"default": "experiment"}),
        (["--TRAIN_REPRESENTATIVE_FEATURES"], {"default": "experiment"}),
        (["--TRAIN_AMOUNT_OF_FEATURES"], {"default": "experiment"}),
        (["--TRAIN_HYPERCUBE"], {"default": "experiment"}),
        (["--TRAIN_OLD_SYNTHETIC_PARAMS"], {"default": "experiment"}),
        (["--TEST_VARIABLE_DATASET"], {"default": "experiment"}),
        (["--TEST_NR_LEARNING_SAMPLES"], {"default": "experiment"}),
        (["--TEST_REPRESENTATIVE_FEATURES"], {"default": "experiment"}),
        (["--TEST_AMOUNT_OF_FEATURES"], {"default": "experiment"}),
        (["--TEST_HYPERCUBE"], {"default": "experiment"}),
        (["--TEST_OLD_SYNTHETIC_PARAMS"], {"default": "experiment"}),
        (["--TEST_COMPARISONS"], {"default": "experiment"}),
    ],
    standard_args=False,
)

cli_command = ""
