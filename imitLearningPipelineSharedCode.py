import json
import numpy as np
import os
import pandas as pd
import time
from joblib import Parallel, delayed
from pathlib import Path
from typing import Any, Callable, Dict, List

from active_learning.config.config import get_active_config
from active_learning.query_sampling_strategies.ImitationLearningBaseQuerySampler import (
    InputState,
    OutputState,
    PreSampledIndices,
)
from active_learning.query_sampling_strategies.TrainedImitALQuerySampler import (
    TrainedImitALBatchSampler,
    TrainedImitALSampler,
    TrainedImitALSingleSampler,
)

dataset_id_mapping = {
    0: ("synthetic", 50),
    1: ("BREAST", 50),
    2: ("DIABETES", 50),
    3: ("FERTILITY", 50),
    4: ("GERMAN", 50),
    5: ("HABERMAN", 50),
    6: ("HEART", 50),
    7: ("ILPD", 50),
    8: ("IONOSPHERE", 50),
    9: ("PIMA", 50),
    10: ("PLANNING", 50),
    11: ("australian", 50),
    12: ("dwtc", 50),
    13: ("emnist-byclass-test", 1000),
    14: ("glass", 50),
    15: ("olivetti", 50),
    16: ("cifar10", 1000),
    17: (
        "synthetic_euc_cos_test",
        50,
    ),
    18: ("wine", 50),
    19: ("adult", 50),
    20: ("abalone", 50),
    21: ("adult", 1000),
    22: ("emnist-byclass-test", 50),
    23: ("cifar10", 50),
    24: ("adult", 100),
    25: ("emnist-byclass-test", 100),
    26: ("cifar10", 100),
    27: ("zoo", 50),
    28: ("parkinsons", 50),
    29: ("flag", 50),
}

strategy_id_mapping = {
    0: ("QueryInstanceRandom", {}),
    1: ("QueryInstanceUncertainty", {"measure": "least_confident"}),
    2: ("QueryInstanceUncertainty", {"measure": "margin"}),
    3: ("QueryInstanceUncertainty", {"measure": "entropy"}),
    4: ("QueryInstanceQBC", {}),
    5: ("QureyExpectedErrorReduction", {}),
    6: ("QueryInstanceGraphDensity", {}),
    7: ("QueryInstanceQUIRE", {}),
    # the following are only for db4701
    8: ("QueryInstanceLAL", {}),  # memory
    9: ("QueryInstanceBMDR", {}),  # cvxpy
    10: ("QueryInstanceSPAL", {}),  # cvxpy
    # 11: ("QueryInstanceUncertainty", {"measure": "distance_to_boundary"}), only works with SVM
    # 12:
}
# non_slurm_strategy_ids = [8,0,10]
non_slurm_strategy_ids = [0, 1, 2]


def get_config():
    config, parser = get_active_config(
        [
            (["--BASE_PARAM_STRING"], {"default": "default"}),
            (["--AMOUNT_OF_PEAKED_SAMPLES"], {"type": int, "default": 20}),
            (["--NR_LEARNING_SAMPLES"], {"type": int, "default": 1000}),
            (
                ["--TEST_COMPARISONS"],
                {
                    "nargs": "+",
                    "default": [
                        "random",
                        "uncertainty_max_margin",
                        "uncertainty_entropy",
                        "uncertainty_lc",
                    ],
                },
            ),
            (["--FINAL_PICTURE"], {"default": ""}),
            (["--HYPER_SEARCHED"], {"action": "store_true"}),
            (["--PLOT_METRIC"], {"default": "acc_auc"}),
            (["--INCLUDE_OPTIMAL_IN_PLOT"], {"action": "store_true"}),
            (["--INCLUDE_ONLY_OPTIMAL_IN_PLOT"], {"action": "store_true"}),
            (["--COMPARE_ALL_FOLDERS"], {"action": "store_true"}),
            (["--NR_ANN_HYPER_SEARCH_ITERATIONS"], {"default": 50}),
            (["--RANDOM_ID_OFFSET"], {"default": 0}),
        ],
        return_parser=True,
    )  # type: ignore

    PARENT_OUTPUT_DIRECTORY = config.OUTPUT_DIRECTORY + "/"

    shared_arguments = {
        "CLUSTER": "dummy",
        "BATCH_SIZE": config.BATCH_SIZE,
        "TOTAL_BUDGET": config.TOTAL_BUDGET,
        "N_JOBS": 1,
        "BATCH_MODE": config.BATCH_MODE,
    }

    evaluation_arguments = {
        #  "DATASET_NAME": "synthetic",
        "AMOUNT_OF_FEATURES": config.AMOUNT_OF_FEATURES,
        "CLASSIFIER": config.CLASSIFIER,
        "VARIABLE_DATASET": config.VARIABLE_DATASET,
        "NEW_SYNTHETIC_PARAMS": config.NEW_SYNTHETIC_PARAMS,
        "HYPERCUBE": config.HYPERCUBE,
        "GENERATE_NOISE": config.GENERATE_NOISE,
        **shared_arguments,
    }
    base_param_string = config.BASE_PARAM_STRING

    # only include config key if it is different from default
    # for k, v in vars(config).items():
    #    if v != parser.get_default(k):
    #        base_param_string += str(v) + "_"

    # base_param_string = base_param_string[-1]

    ann_arguments = {
        "STATE_DISTANCES_LAB": config.STATE_DISTANCES_LAB,
        "STATE_DISTANCES_UNLAB": config.STATE_DISTANCES_UNLAB,
        "STATE_PREDICTED_CLASS": config.STATE_PREDICTED_CLASS,
        "STATE_PREDICTED_UNITY": config.STATE_PREDICTED_UNITY,
        "STATE_ARGSECOND_PROBAS": config.STATE_ARGSECOND_PROBAS,
        "STATE_ARGTHIRD_PROBAS": config.STATE_ARGTHIRD_PROBAS,
        "STATE_DIFF_PROBAS": config.STATE_DIFF_PROBAS,
        "STATE_DISTANCES": config.STATE_DISTANCES,
        "STATE_UNCERTAINTIES": config.STATE_UNCERTAINTIES,
        "STATE_INCLUDE_NR_FEATURES": config.STATE_INCLUDE_NR_FEATURES,
        "DISTANCE_METRIC": config.DISTANCE_METRIC,
        "INITIAL_BATCH_SAMPLING_METHOD": config.INITIAL_BATCH_SAMPLING_METHOD,
        "INITIAL_BATCH_SAMPLING_ARG": config.INITIAL_BATCH_SAMPLING_ARG,
        "INITIAL_BATCH_SAMPLING_HYBRID_UNCERT": config.INITIAL_BATCH_SAMPLING_HYBRID_UNCERT,
        "INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST": config.INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST,
        "INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB": config.INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB,
        "INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY": config.INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY,
    }

    return (
        config,
        shared_arguments,
        evaluation_arguments,
        ann_arguments,
        PARENT_OUTPUT_DIRECTORY,
        base_param_string,
    )


def run_code_experiment(
    EXPERIMENT_TITLE: str,
    OUTPUT_FILE: str,
    code: Callable,
    code_kwargs: Dict = {},
    OUTPUT_FILE_LENGTH=None,
) -> None:
    # check if folder for OUTPUT_FILE exists
    Path(os.path.dirname(OUTPUT_FILE)).mkdir(parents=True, exist_ok=True)
    # if not run it
    print("#" * 80)
    print(EXPERIMENT_TITLE + "\n")
    print("Saving to " + OUTPUT_FILE)

    # check if OUTPUT_FILE exists
    #  if os.path.isfile(OUTPUT_FILE):
    #      if OUTPUT_FILE_LENGTH is not None:
    #          if sum(1 for l in open(OUTPUT_FILE)) >= OUTPUT_FILE_LENGTH:
    #              print("zu ende")
    #              return
    #      else:
    #          return

    # if not run it
    print("#" * 80)
    print(EXPERIMENT_TITLE + "\n")
    print("Saving to " + OUTPUT_FILE)

    start = time.time()
    code(**code_kwargs)
    end = time.time()

    assert os.path.exists(OUTPUT_FILE)
    print("Done in ", end - start, " s\n")
    print("#" * 80)
    print("\n" * 5)


def run_python_experiment(
    EXPERIMENT_TITLE: str,
    OUTPUT_FILE: str,
    CLI_COMMAND: str,
    CLI_ARGUMENTS: Dict[str, Any],
    OUTPUT_FILE_LENGTH: float = None,
    SAVE_ARGUMENT_JSON: bool = True,
) -> None:
    def code(CLI_COMMAND):
        for k, v in CLI_ARGUMENTS.items():
            if str(v) == "True":
                CLI_COMMAND += " --" + k
            else:
                CLI_COMMAND += " --" + k + " " + str(v)

        if SAVE_ARGUMENT_JSON:
            with open(OUTPUT_FILE + "_params.json", "w") as f:
                json.dump({"CLI_COMMAND": CLI_COMMAND, **CLI_ARGUMENTS}, f)

        print(CLI_COMMAND)
        os.system(CLI_COMMAND)

    run_code_experiment(
        EXPERIMENT_TITLE,
        OUTPUT_FILE,
        code=code,
        code_kwargs={"CLI_COMMAND": CLI_COMMAND},
        OUTPUT_FILE_LENGTH=OUTPUT_FILE_LENGTH,
    )


def run_parallel_experiment(
    EXPERIMENT_TITLE: str,
    OUTPUT_FILE: str,
    CLI_COMMAND: str,
    CLI_ARGUMENTS: Dict[str, Any],
    RANDOM_IDS: List[int] = None,
    RANDOM_ID_OFFSET: int = 0,
    PARALLEL_AMOUNT: int = 0,
    OUTPUT_FILE_LENGTH: int = None,
    SAVE_ARGUMENT_JSON: bool = True,
    RESTART_IF_NOT_ENOUGH_SAMPLES: bool = False,
):
    # check if folder for OUTPUT_FILE exists
    Path(os.path.dirname(OUTPUT_FILE)).mkdir(parents=True, exist_ok=True)

    # save config.json
    if SAVE_ARGUMENT_JSON:
        with open(OUTPUT_FILE + "_params.json", "w") as f:
            json.dump({"CLI_COMMAND": CLI_COMMAND, **CLI_ARGUMENTS}, f)

    for k, v in CLI_ARGUMENTS.items():
        if isinstance(v, bool) and v == True:
            CLI_COMMAND += " --" + k
        elif isinstance(v, bool) and v == False:
            pass
        else:
            CLI_COMMAND += " --" + k + " " + str(v)
    print("\n" * 5)

    def run_parallel(CLI_COMMAND, RANDOM_SEED):
        CLI_COMMAND += " --RANDOM_SEED " + str(RANDOM_SEED)
        print(CLI_COMMAND)
        os.system(CLI_COMMAND)

    if RANDOM_IDS:
        ids = RANDOM_IDS
    else:
        possible_ids = range(
            int(RANDOM_ID_OFFSET), int(RANDOM_ID_OFFSET) + int(PARALLEL_AMOUNT)
        )
        if Path(OUTPUT_FILE).is_file():
            #  print(OUTPUT_FILE)
            df = pd.read_csv(OUTPUT_FILE, index_col=None, usecols=["random_seed"])
            rs = df["random_seed"].to_numpy()
            #  print(sorted(df["random_seed"]))
            ids = [i for i in possible_ids if i not in rs]
            #  print(list(possible_ids))
            #  print(ids)
        else:
            ids = list(possible_ids)

    if len(ids) == 0:
        return

    def code(CLI_COMMAND, PARALLEL_AMOUNT, RANDOM_ID_OFFSET):
        with Parallel(
            #  n_jobs=1,
            len(os.sched_getaffinity(0)),
            #  multiprocessing.cpu_count(),
            backend="loky",
        ) as parallel:
            output = parallel(delayed(run_parallel)(CLI_COMMAND, k) for k in ids)

    if Path(OUTPUT_FILE).is_file():
        OUTPUT_FILE_LENGTH = len(df) + len(ids)  # type: ignore
    run_code_experiment(
        EXPERIMENT_TITLE,
        OUTPUT_FILE,
        code=code,
        code_kwargs={
            "CLI_COMMAND": CLI_COMMAND,
            "PARALLEL_AMOUNT": PARALLEL_AMOUNT,
            "RANDOM_ID_OFFSET": RANDOM_ID_OFFSET,
        },
        OUTPUT_FILE_LENGTH=OUTPUT_FILE_LENGTH,
    )
    return


class ALiPYImitALSingle:
    trained_imitAL_sampler: TrainedImitALSampler
    X: np.ndarray
    Y: np.ndarray

    def __init__(self, X: np.ndarray, Y: np.ndarray, **kwargs):
        self.X = X
        self.Y = Y

        # load NN params from json file
        with open(
            os.path.dirname(kwargs["NN_BINARY_PATH"])
            + "/01_dataset_creation_stats.csv_params.json",
            "r",
        ) as f:
            content = str(f.read())
            dataset_stats = json.loads(content)

        print(dataset_stats)

        self.trained_imitAL_sampler = TrainedImitALSingleSampler(
            PRE_SAMPLING_METHOD=dataset_stats["INITIAL_BATCH_SAMPLING_METHOD"],
            PRE_SAMPLING_ARG=dataset_stats["INITIAL_BATCH_SAMPLING_ARG"],
            AMOUNT_OF_PEAKED_OBJECTS=dataset_stats["AMOUNT_OF_PEAKED_OBJECTS"],
            DISTANCE_METRIC=dataset_stats["DISTANCE_METRIC"],
            STATE_ARGSECOND_PROBAS=dataset_stats["STATE_ARGSECOND_PROBAS"],
            STATE_ARGTHIRD_PROBAS=dataset_stats["STATE_ARGTHIRD_PROBAS"],
            STATE_DIFF_PROBAS=dataset_stats["STATE_DIFF_PROBAS"],
            STATE_PREDICTED_CLASS=dataset_stats["STATE_PREDICTED_CLASS"],
            STATE_DISTANCES_LAB=dataset_stats["STATE_DISTANCES_LAB"],
            STATE_DISTANCES_UNLAB=dataset_stats["STATE_DISTANCES_UNLAB"],
            STATE_INCLUDE_NR_FEATURES=dataset_stats["STATE_INCLUDE_NR_FEATURES"],
            NN_BINARY_PATH=kwargs["NN_BINARY_PATH"],
        )

        self.trained_imitAL_sampler.data_storage = kwargs["data_storage"]

    def select(self, labeled_index, unlabeled_index, model, batch_size=1, **kwargs):
        self.trained_imitAL_sampler.data_storage.labeled_mask = labeled_index
        self.trained_imitAL_sampler.data_storage.unlabeled_mask = unlabeled_index

        # @TODO: check if data_storage index and Y etc. is updated accordingly!!

        # update data_storage with labeled_index and unlabeled_index
        pre_sampled_X_querie_indices: PreSampledIndices = (
            self.trained_imitAL_sampler.pre_sample_potential_X_queries()
        )

        # when using a pre-trained model this does nothing
        self.trained_imitAL_sampler.calculateImitationLearningData(
            pre_sampled_X_querie_indices
        )

        X_input_state: InputState = self.trained_imitAL_sampler.encode_input_state(
            pre_sampled_X_querie_indices
        )
        Y_output_state: OutputState = self.trained_imitAL_sampler.applyNN(X_input_state)
        return [
            v
            for v in self.trained_imitAL_sampler.decode_output_state(
                Y_output_state, pre_sampled_X_querie_indices, batch_size
            )
        ]
        # return [v for k, v in ordered_list_of_possible_sample_indices[:batch_size]]
