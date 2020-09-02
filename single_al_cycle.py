import os
import random

import numpy as np

from active_learning.al_cycle_wrapper import train_and_eval_dataset
from active_learning.experiment_setup_lib import (
    get_active_config,
    init_logger,
)
from fake_experiment_oracle import FakeExperimentOracle

config = get_active_config(
    [
        (["--NN_BINARY"], {"type": str}),
    ]
)

init_logger("console")

if config.RANDOM_SEED == -2:
    config.RANDOM_SEED = random.randint(0, 2147483647)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

if config.OUTPUT_DIRECTORY == "NN_BINARY":
    config.OUTPUT_DIRECTORY = (
        os.path.dirname(config.NN_BINARY) + "/evaluation_hyper_parameters.csv"
    )


score = train_and_eval_dataset(
    hyper_parameters=vars(config),
    oracle=FakeExperimentOracle(),
    DATASET_NAME=config.DATASET_NAME,
    DATASETS_PATH=config.DATASETS_PATH,
)
print("Done with ", score)
