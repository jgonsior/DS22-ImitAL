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
    _,
    _,
    _,
    PARENT_OUTPUT_DIRECTORY,
    train_base_param_string,
    _,
) = get_config()

HYPER_SEARCH_OUTPUT_FILE = (
    config.OUTPUT_DIRECTORY + train_base_param_string + "/hyper_results.txt"
)
run_python_experiment(
    "ANN hyper_search",
    HYPER_SEARCH_OUTPUT_FILE,
    CLI_COMMAND="python train_lstm.py",
    CLI_ARGUMENTS={
        "DATA_PATH": config.OUTPUT_DIRECTORY + train_base_param_string,
        "STATE_ENCODING": "listwise",
        "TARGET_ENCODING": "binary",
        "SAVE_DESTINATION": config.OUTPUT_DIRECTORY
        + train_base_param_string
        + "/trained_ann.pickle",
        "RANDOM_SEED": 1,
        "HYPER_SEARCH": True,
        "N_ITER": config.NR_ANN_HYPER_SEARCH_ITERATIONS,
    },
)
assert os.path.exists(HYPER_SEARCH_OUTPUT_FILE)

with open(HYPER_SEARCH_OUTPUT_FILE, "r") as f:
    lines = f.read().splitlines()
    last_line = lines[-1]
    lower_params = json.loads(last_line)
    ANN_HYPER_PARAMS = {}
    for k, v in lower_params.items():
        ANN_HYPER_PARAMS[k.upper()] = v
