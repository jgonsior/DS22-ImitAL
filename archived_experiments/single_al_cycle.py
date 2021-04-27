import argparse
import numpy as np
import os
import random

from active_learning.al_cycle_wrapper import train_and_eval_dataset
from active_learning.config import get_active_config
from active_learning.logger import init_logger
from active_learning.oracles import FakeExperimentOracle

config: argparse.Namespace = get_active_config(  # type: ignore
    [
        (["--NN_BINARY_PATH"], {"type": str}),
    ]
)

init_logger(config.LOG_FILE)

if config.RANDOM_SEED == -2:
    config.RANDOM_SEED = random.randint(0, 2147483647)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

if config.OUTPUT_PATH == "NN_BINARY":
    config.OUTPUT_PATH = (
        os.path.dirname(config.NN_BINARY) + "/evaluation_hyper_parameters.csv"
    )

score = train_and_eval_dataset(
    hyper_parameters=vars(config),
    oracle=FakeExperimentOracle(),
)
