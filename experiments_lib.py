import argparse
from typing import Any, Callable, Dict, List
from active_learning.config.config import get_active_config
import json
import math
import multiprocessing
import os
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from active_learning.config import standard_config


def get_config():
    config: argparse.Namespace = get_active_config()  # type: ignore

    # calculate resulting pathes
    splitted_base_param_string = config.BASE_PARAM_STRING.split("#")
    train_base_param_string = "#".join(
        [x for x in splitted_base_param_string if not x.startswith("TEST_")]
    )
    test_base_param_string = "#".join(
        [x for x in splitted_base_param_string if not x.startswith("TRAIN_")]
    )

    if train_base_param_string == "":
        train_base_param_string = "DEFAULT"
    if test_base_param_string == "":
        test_base_param_string = "DEFAULT"

    PARENT_OUTPUT_DIRECTORY = config.OUTPUT_DIRECTORY + "/"

    shared_arguments = {
        "CLUSTER": "dummy",
        "NR_QUERIES_PER_ITERATION": config.NR_QUERIES_PER_ITERATION,
        "START_SET_SIZE": 1,
        "USER_QUERY_BUDGET_LIMIT": config.USER_QUERY_BUDGET_LIMIT,
        "N_JOBS": 1,
        "BATCH_MODE": config.BATCH_MODE,
    }

    evaluation_arguments = {
        #  "DATASET_NAME": "synthetic",
        "AMOUNT_OF_FEATURES": config.TEST_AMOUNT_OF_FEATURES,
        "CLASSIFIER": config.TEST_CLASSIFIER,
        "VARIABLE_DATASET": config.TEST_VARIABLE_DATASET,
        "NEW_SYNTHETIC_PARAMS": config.TEST_NEW_SYNTHETIC_PARAMS,
        "HYPERCUBE": config.TEST_HYPERCUBE,
        "GENERATE_NOISE": config.TEST_GENERATE_NOISE,
        **shared_arguments,
    }

    ann_arguments = {
        "STATE_DISTANCES_LAB": config.TRAIN_STATE_DISTANCES_LAB,
        "STATE_DISTANCES_UNLAB": config.TRAIN_STATE_DISTANCES_UNLAB,
        "STATE_PREDICTED_CLASS": config.TRAIN_STATE_PREDICTED_CLASS,
        "STATE_PREDICTED_UNITY": config.TRAIN_STATE_PREDICTED_UNITY,
        "STATE_ARGSECOND_PROBAS": config.TRAIN_STATE_ARGSECOND_PROBAS,
        "STATE_ARGTHIRD_PROBAS": config.TRAIN_STATE_ARGTHIRD_PROBAS,
        "STATE_DIFF_PROBAS": config.TRAIN_STATE_DIFF_PROBAS,
        "STATE_DISTANCES": config.TRAIN_STATE_DISTANCES,
        "STATE_UNCERTAINTIES": config.TRAIN_STATE_UNCERTAINTIES,
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
        train_base_param_string,
        test_base_param_string,
    )


def run_code_experiment(
    EXPERIMENT_TITLE: str,
    OUTPUT_FILE: str,
    code: Callable,
    code_kwargs:Dict={},
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
    EXPERIMENT_TITLE:str,
    OUTPUT_FILE:str,
    CLI_COMMAND:str,
    CLI_ARGUMENTS:Dict[str, Any],
    OUTPUT_FILE_LENGTH:float=None,
    SAVE_ARGUMENT_JSON:bool=True,
)-> None:
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
    EXPERIMENT_TITLE:str,
    OUTPUT_FILE:str,
    CLI_COMMAND:str,
    CLI_ARGUMENTS:Dict[str, Any],
    RANDOM_IDS:List[int]=None,
    PARALLEL_OFFSET:int=0,
    PARALLEL_AMOUNT:int=0,
    OUTPUT_FILE_LENGTH:int=None,
    SAVE_ARGUMENT_JSON:bool=True,
    RESTART_IF_NOT_ENOUGH_SAMPLES:bool=False,
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
            int(PARALLEL_OFFSET), int(PARALLEL_OFFSET) + int(PARALLEL_AMOUNT)
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

    def code(CLI_COMMAND, PARALLEL_AMOUNT, PARALLEL_OFFSET):
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
            "PARALLEL_OFFSET": PARALLEL_OFFSET,
        },
        OUTPUT_FILE_LENGTH=OUTPUT_FILE_LENGTH,
    )
    return
