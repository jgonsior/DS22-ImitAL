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
    evaluation_arguments,
    _,
    PARENT_OUTPUT_DIRECTORY,
    _,
    test_base_param_string,
    evaluation_arguments,
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
        #  config.TEST_NR_LEARNING_SAMPLES = 100
        evaluation_arguments["USER_QUERY_BUDGET_LIMIT"] = 20
    evaluation_arguments["DATASET_NAME"] = DATASET_NAME

    for comparison in config.TEST_COMPARISONS:
        COMPARISON_PATH = (
            PARENT_OUTPUT_DIRECTORY
            + "classics/"
            + comparison
            #  + test_base_param_string
            + ".csv"
        )
        run_parallel_experiment(
            "Creating " + comparison + "-evaluation data",
            OUTPUT_FILE=COMPARISON_PATH,
            CLI_COMMAND="python single_al_cycle.py",
            CLI_ARGUMENTS={
                "OUTPUT_DIRECTORY": COMPARISON_PATH,
                "SAMPLING": comparison,
                **evaluation_arguments,
            },
            PARALLEL_OFFSET=config.TEST_PARALLEL_OFFSET,
            PARALLEL_AMOUNT=config.TEST_NR_LEARNING_SAMPLES,
            OUTPUT_FILE_LENGTH=config.TEST_NR_LEARNING_SAMPLES,
        )
    test_base_param_string = original_test_base_param_string
