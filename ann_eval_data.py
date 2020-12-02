import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from experiments_lib import (
    run_code_experiment,
    run_python_experiment,
    run_parallel_experiment,
    get_config,
)

(
    config,
    shared_arguments,
    evaluation_arguments,
    ann_arguments,
    PARENT_OUTPUT_DIRECTORY,
    train_base_param_string,
    test_base_param_string,
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
    original_test_base_param_string = test_base_param_string
    test_base_param_string += "_" + DATASET_NAME

    if DATASET_NAME != "synthetic":
        evaluation_arguments["USER_QUERY_BUDGET_LIMIT"] = 20
    evaluation_arguments["DATASET_NAME"] = DATASET_NAME

    EVALUATION_FILE_TRAINED_NN_PATH = (
        config.OUTPUT_DIRECTORY + config.BASE_PARAM_STRING + "_" + DATASET_NAME + ".csv"
    )

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
            **ann_arguments,
            **evaluation_arguments,
        },
        PARALLEL_OFFSET=config.TEST_PARALLEL_OFFSET,
        PARALLEL_AMOUNT=config.TEST_NR_LEARNING_SAMPLES,
        OUTPUT_FILE_LENGTH=config.TEST_NR_LEARNING_SAMPLES,
    )

    #  ERROR FOUND -> DAS HIER MUSS DANN DANACH WENN ALLES FERTIG UND SO
    #  # rename sampling column
    #  # Read in the file
    #  with open(EVALUATION_FILE_TRAINED_NN_PATH, "r") as file:
    #      filedata = file.read()
    #
    #  # Replace the target string
    #  filedata = filedata.replace("trained_nn", config.OUTPUT_DIRECTORY)
    #
    #  # Write the file out again
    #  with open(EVALUATION_FILE_TRAINED_NN_PATH, "w") as file:
    #      file.write(filedata)
    #
    test_base_param_string = original_test_base_param_string
