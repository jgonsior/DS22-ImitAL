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

for DATASET_NAME in [
    #  "emnist-byclass-test",
    "synthetic",
    #  "dwtc",
    #  "BREAST",
    #  "DIABETES",
    #  "FERTILITY",
    #  "GERMAN",
    #  "HABERMAN",
    #  "HEART",
    #  "ILPD",
    #  "IONOSPHERE",
    #  "PIMA",
    #  "PLANNING",
    #  "australian",
]:
    if DATASET_NAME != "synthetic":
        evaluation_arguments["USER_QUERY_BUDGET_LIMIT"] = 20
    evaluation_arguments["DATASET_NAME"] = DATASET_NAME

    EVALUATION_FILE_TRAINED_NN_PATH = (
        config.OUTPUT_DIRECTORY + config.BASE_PARAM_STRING + "_" + DATASET_NAME + ".csv"
    )

    original_test_base_param_string = test_base_param_string
    test_base_param_string += "_" + DATASET_NAME

    run_parallel_experiment(
        "Creating ann-evaluation data",
        OUTPUT_FILE=EVALUATION_FILE_TRAINED_NN_PATH,
        CLI_COMMAND="python single_al_cycle.py",
        CLI_ARGUMENTS={
            "NN_BINARY": config.OUTPUT_DIRECTORY
            + train_base_param_string
            + "/trained_ann.pickle",
            "OUTPUT_DIRECTORY": EVALUATION_FILE_TRAINED_NN_PATH,
            "SAMPLING": "trained_nn",
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
            **evaluation_arguments,
        },
        PARALLEL_OFFSET=config.TEST_PARALLEL_OFFSET,
        PARALLEL_AMOUNT=config.TEST_NR_LEARNING_SAMPLES,
        OUTPUT_FILE_LENGTH=config.TEST_NR_LEARNING_SAMPLES,
    )

    ERROR FOUND -> DAS HIER MUSS DANN DANACH WENN ALLES FERTIG UND SO
    # rename sampling column
    # Read in the file
    with open(EVALUATION_FILE_TRAINED_NN_PATH, "r") as file:
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace("trained_nn", config.OUTPUT_DIRECTORY)

    # Write the file out again
    with open(EVALUATION_FILE_TRAINED_NN_PATH, "w") as file:
        file.write(filedata)
