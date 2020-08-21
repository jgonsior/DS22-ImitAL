import os
import random
import numpy as np
from active_learning.al_cycle_wrapper import train_and_eval_dataset
from active_learning.experiment_setup_lib import (
    standard_config,
    init_logger,
)
from fake_experiment_oracle import FakeExperimentOracle

config = standard_config(
    [
        (
            ["--SAMPLING"],
            {
                "required": True,
                "help": "Possible values: uncertainty, random, committe, boundary",
            },
        ),
        (["--DATASET_NAME"], {"required": True,}),
        (
            ["--CLUSTER"],
            {
                "default": "dummy",
                "help": "Possible values: dummy, random, mostUncertain, roundRobin",
            },
        ),
        (["--NR_LEARNING_ITERATIONS"], {"type": int, "default": 150000}),
        (["--NR_QUERIES_PER_ITERATION"], {"type": int, "default": 150}),
        (["--START_SET_SIZE"], {"type": int, "default": 1}),
        (
            ["--MINIMUM_TEST_ACCURACY_BEFORE_RECOMMENDATIONS"],
            {"type": float, "default": 0.5},
        ),
        (
            ["--UNCERTAINTY_RECOMMENDATION_CERTAINTY_THRESHOLD"],
            {"type": float, "default": 0.9},
        ),
        (["--UNCERTAINTY_RECOMMENDATION_RATIO"], {"type": float, "default": 1 / 100}),
        (["--SNUBA_LITE_MINIMUM_HEURISTIC_ACCURACY"], {"type": float, "default": 0.9}),
        (
            ["--CLUSTER_RECOMMENDATION_MINIMUM_CLUSTER_UNITY_SIZE"],
            {"type": float, "default": 0.7},
        ),
        (
            ["--CLUSTER_RECOMMENDATION_RATIO_LABELED_UNLABELED"],
            {"type": float, "default": 0.9},
        ),
        (["--WITH_UNCERTAINTY_RECOMMENDATION"], {"action": "store_true"}),
        (["--WITH_CLUSTER_RECOMMENDATION"], {"action": "store_true"}),
        (["--WITH_SNUBA_LITE"], {"action": "store_true"}),
        (["--PLOT"], {"action": "store_true"}),
        (["--STOPPING_CRITERIA_UNCERTAINTY"], {"type": float, "default": 0.7}),
        (["--STOPPING_CRITERIA_ACC"], {"type": float, "default": 0.7}),
        (["--STOPPING_CRITERIA_STD"], {"type": float, "default": 0.7}),
        (
            ["--ALLOW_RECOMMENDATIONS_AFTER_STOP"],
            {"action": "store_true", "default": False},
        ),
        (["--OUTPUT_DIRECTORY"], {"default": "tmp/"}),
        (["--HYPER_SEARCH_TYPE"], {"default": "random"}),
        (["--USER_QUERY_BUDGET_LIMIT"], {"type": float, "default": 200}),
        (["--AMOUNT_OF_PEAKED_OBJECTS"], {"type": int, "default": 12}),
        (["--MAX_AMOUNT_OF_WS_PEAKS"], {"type": int, "default": 1}),
        (["--NN_BINARY"], {"type": str}),
        (["--AMOUNT_OF_RANDOM_QUERY_SETS"], {"type": int, "default": 1}),
        (["--VARIABLE_INPUT_SIZE"], {"action": "store_true"}),
        (["--REPRESENTATIVE_FEATURES"], {"action": "store_true"}),
        (["--NEW_SYNTHETIC_PARAMS"], {"action": "store_true"}),
        (["--HYPERCUBE"], {"action": "store_true"}),
        (["--AMOUNT_OF_FEATURES"], {"type": int, "default": -1}),
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
