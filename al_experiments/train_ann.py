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

if not config.HYPER_SEARCHED:
    ANN_HYPER_PARAMS = {
        "REGULAR_DROPOUT_RATE": 0.3,
        "OPTIMIZER": "Adam",
        "NR_HIDDEN_NEURONS": 900,
        "NR_HIDDEN_LAYERS": 4,
        "LOSS": "MeanSquaredError",
        "EPOCHS": 10000,
        "BATCH_SIZE": 32,
        "ACTIVATION": "tanh",
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
