import json

import numpy as np
import pandas as pd
from tabulate import tabulate

from experiments_lib import (get_config, run_code_experiment,
                             run_parallel_experiment, run_python_experiment)

(
    config,
    _,
    _,
    _,
    PARENT_OUTPUT_DIRECTORY,
    train_base_param_string,
    _,
) = get_config()

if config.HYPER_SEARCHED:
    HYPER_SEARCH_OUTPUT_FILE = (
        config.OUTPUT_DIRECTORY + train_base_param_string + "/hyper_results.txt"
    )

    with open(HYPER_SEARCH_OUTPUT_FILE, "r") as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
        lower_params = json.loads(last_line)
        ANN_HYPER_PARAMS = {}
        for k, v in lower_params.items():
            ANN_HYPER_PARAMS[k.upper()] = v
else:
    if config.BATCH_MODE:
        ANN_HYPER_PARAMS = {
            "REGULAR_DROPOUT_RATE": 0.2,
            "OPTIMIZER": "RMSprop",
            "NR_HIDDEN_NEURONS": 900,
            "NR_HIDDEN_LAYERS": 3,
            "LOSS": "MeanSquaredError",
            "EPOCHS": 10000,
            "BATCH_SIZE": 128,
            "ACTIVATION": "elu",
        }
    else:
        ANN_HYPER_PARAMS = {
            "REGULAR_DROPOUT_RATE": 0.2,
            "OPTIMIZER": "Nadam",
            "NR_HIDDEN_NEURONS": 1100,
            "NR_HIDDEN_LAYERS": 2,
            "LOSS": "MeanSquaredError",
            "EPOCHS": 10000,
            "BATCH_SIZE": 128,
            "ACTIVATION": "elu",
        }
run_python_experiment(
    "Train ANN",
    PARENT_OUTPUT_DIRECTORY + train_base_param_string + "/trained_ann.pickle",
    CLI_COMMAND="python train_lstm.py",
    CLI_ARGUMENTS={
        "DATA_PATH": config.OUTPUT_DIRECTORY + train_base_param_string,
        "STATE_ENCODING": "listwise",
        "TARGET_ENCODING": "binary",
        "SAVE_DESTINATION": config.OUTPUT_DIRECTORY
        + train_base_param_string
        + "/trained_ann.pickle",
        "REGULAR_DROPOUT_RATE": ANN_HYPER_PARAMS["REGULAR_DROPOUT_RATE"],
        "OPTIMIZER": ANN_HYPER_PARAMS["OPTIMIZER"],
        "NR_HIDDEN_NEURONS": ANN_HYPER_PARAMS["NR_HIDDEN_NEURONS"],
        "NR_HIDDEN_LAYERS": ANN_HYPER_PARAMS["NR_HIDDEN_LAYERS"],
        "LOSS": ANN_HYPER_PARAMS["LOSS"],
        "EPOCHS": ANN_HYPER_PARAMS["EPOCHS"],
        "BATCH_SIZE": ANN_HYPER_PARAMS["BATCH_SIZE"],
        "ACTIVATION": ANN_HYPER_PARAMS["ACTIVATION"],
        "RANDOM_SEED": 1,
    },
)
