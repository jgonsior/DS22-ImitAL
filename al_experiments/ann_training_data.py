import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from .experiments_lib import (
    run_code_experiment,
    run_python_experiment,
    run_parallel_experiment,
    get_config,
)

(
    config,
    shared_arguments,
    PARENT_OUTPUT_DIRECTORY,
    train_base_param_string,
    test_base_param_string,
    evaluation_arguments,
    suffix,
) = get_config()

OUTPUT_FILE = PARENT_OUTPUT_DIRECTORY + train_base_param_string
run_parallel_experiment(
    "Creating dataset",
    OUTPUT_FILE=PARENT_OUTPUT_DIRECTORY
    + train_base_param_string
    + "/dataset_creation.csv"
    + SUFFIX,
    CLI_COMMAND="python imit_training.py",
    CLI_ARGUMENTS={
        "DATASETS_PATH": "../datasets",
        "CLASSIFIER": config.TRAIN_CLASSIFIER,
        "OUTPUT_DIRECTORY": PARENT_OUTPUT_DIRECTORY + train_base_param_string,
        "DATASET_NAME": "synthetic",
        "SAMPLING": "trained_nn",
        "AMOUNT_OF_PEAKED_OBJECTS": config.TRAIN_AMOUNT_OF_PEAKED_SAMPLES,
        "MAX_AMOUNT_OF_WS_PEAKS": 0,
        "AMOUNT_OF_LEARN_ITERATIONS": 1,
        "AMOUNT_OF_FEATURES": config.TRAIN_AMOUNT_OF_FEATURES,
        "VARIABLE_DATASET": config.TRAIN_VARIABLE_DATASET,
        "NEW_SYNTHETIC_PARAMS": config.TRAIN_NEW_SYNTHETIC_PARAMS,
        "HYPERCUBE": config.TRAIN_HYPERCUBE,
        "STOP_AFTER_MAXIMUM_ACCURACY_REACHED": config.TRAIN_STOP_AFTER_MAXIMUM_ACCURACY_REACHED,
        "GENERATE_NOISE": config.TRAIN_GENERATE_NOISE,
        "STATE_DISTANCES_LAB": config.TRAIN_STATE_DISTANCES_LAB,
        "STATE_DISTANCES_UNLAB": config.TRAIN_STATE_DISTANCES_UNLAB,
        "STATE_PREDICTED_CLASS": config.TRAIN_STATE_PREDICTED_CLASS,
        "STATE_PREDICTED_UNITY": config.TRAIN_STATE_PREDICTED_UNITY,
        "STATE_ARGSECOND_PROBAS": config.TRAIN_STATE_ARGSECOND_PROBAS,
        "STATE_ARGTHIRD_PROBAS": config.TRAIN_STATE_ARGTHIRD_PROBAS,
        "STATE_DIFF_PROBAS": config.TRAIN_STATE_DIFF_PROBAS,
        "STATE_DISTANCES": config.TRAIN_STATE_DISTANCES,
        "STATE_UNCERTAINTIES": config.TRAIN_STATE_UNCERTAINTIES,
        "INITIAL_BATCH_SAMPLING_METHOD": config.INITIAL_BATCH_SAMPLING_METHOD,
        "INITIAL_BATCH_SAMPLING_ARG": config.INITIAL_BATCH_SAMPLING_ARG,
        "INITIAL_BATCH_SAMPLING_HYBRID_UNCERT": config.INITIAL_BATCH_SAMPLING_HYBRID_UNCERT,
        "INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST": config.INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST,
        "INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB": config.INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB,
        "INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY": config.INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY,
        **shared_arguments,
    },
    PARALLEL_OFFSET=config.TRAIN_PARALLEL_OFFSET,
    PARALLEL_AMOUNT=config.TRAIN_NR_LEARNING_SAMPLES,
    OUTPUT_FILE_LENGTH=config.TRAIN_NR_LEARNING_SAMPLES,
    RESTART_IF_NOT_ENOUGH_SAMPLES=False,
    OUTPUT_FILES_SUFFIX=suffix,
)
