import json
import math
import multiprocessing
import os
import time
from pathlib import Path

from joblib import Parallel, delayed

from active_learning.experiment_setup_lib import standard_config


def get_config():
    config, parser = standard_config(
        [
            (["--RANDOM_SEED"], {"default": 1, "type": int}),
            (["--LOG_FILE"], {"default": "log.txt"}),
            (["--OUTPUT_DIRECTORY"], {"default": "/tmp"}),
            (["--BASE_PARAM_STRING"], {"default": "default"}),
            (["--NR_QUERIES_PER_ITERATION"], {"type": int, "default": 5}),
            (["--USER_QUERY_BUDGET_LIMIT"], {"type": int, "default": 50}),
            (["--TRAIN_CLASSIFIER"], {"default": "MLP"}),
            (["--TRAIN_VARIABLE_DATASET"], {"action": "store_false"}),
            (["--TRAIN_AMOUNT_OF_PEAKED_SAMPLES"], {"type": int, "default": 20}),
            (["--TRAIN_NR_LEARNING_SAMPLES"], {"type": int, "default": 1000}),
            (["--TRAIN_AMOUNT_OF_FEATURES"], {"type": int, "default": -1}),
            (["--TRAIN_HYPERCUBE"], {"action": "store_true"}),
            (["--TRAIN_NEW_SYNTHETIC_PARAMS"], {"action": "store_false"}),
            (
                ["--TRAIN_STOP_AFTER_MAXIMUM_ACCURACY_REACHED"],
                {"action": "store_true"},
            ),
            (["--TRAIN_GENERATE_NOISE"], {"action": "store_true"}),
            (["--TRAIN_STATE_DIFF_PROBAS"], {"action": "store_true"}),
            (["--TRAIN_STATE_ARGSECOND_PROBAS"], {"action": "store_true"}),
            (["--TRAIN_STATE_ARGTHIRD_PROBAS"], {"action": "store_true"}),
            (["--TRAIN_STATE_DISTANCES_LAB"], {"action": "store_true"}),
            (["--TRAIN_STATE_DISTANCES_UNLAB"], {"action": "store_true"}),
            (["--TRAIN_STATE_PREDICTED_CLASS"], {"action": "store_true"}),
            (["--TRAIN_STATE_DISTANCES"], {"action": "store_true"}),
            (["--TRAIN_STATE_UNCERTAINTIES"], {"action": "store_true"}),
            (["--TRAIN_STATE_PREDICTED_UNITY"], {"action": "store_true"}),
            (["--TRAIN_INITIAL_BATCH_SAMPLING_METHOD"], {"default": "random"}),
            (["--TRAIN_INITIAL_BATCH_SAMPLING_ARG"], {"type": int, "default": 100}),
            (["--TEST_VARIABLE_DATASET"], {"action": "store_false"}),
            (["--TEST_NR_LEARNING_SAMPLES"], {"type": int, "default": 500}),
            (["--TEST_AMOUNT_OF_FEATURES"], {"type": int, "default": -1}),
            (["--TEST_HYPERCUBE"], {"action": "store_true"}),
            (["--TEST_NEW_SYNTHETIC_PARAMS"], {"action": "store_true"}),
            (["--TEST_CLASSIFIER"], {"default": "MLP"}),
            (["--TEST_GENERATE_NOISE"], {"action": "store_false"}),
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
            (["--SKIP_TRAINING_DATA_GENERATION"], {"action": "store_true"}),
            (["--ONLY_TRAINING_DATA"], {"action": "store_true"}),
            (["--PLOT_METRIC"], {"default": "acc_auc"}),
            (["--INCLUDE_OPTIMAL_IN_PLOT"], {"action": "store_true"}),
            (["--INCLUDE_ONLY_OPTIMAL_IN_PLOT"], {"action": "store_true"}),
            (["--COMPARE_ALL_FOLDERS"], {"action": "store_true"}),
            (["--INITIAL_BATCH_SAMPLING_METHOD"], {"default": "furthest"}),
            (["--INITIAL_BATCH_SAMPLING_ARG"], {"default": 100}),
            (["--BATCH_MODE"], {"action": "store_true"}),
            (["--NR_ANN_HYPER_SEARCH_ITERATIONS"], {"default": 50}),
        ],
        standard_args=False,
        return_argparse=True,
    )

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

    return (
        config,
        shared_arguments,
        PARENT_OUTPUT_DIRECTORY,
        train_base_param_string,
        test_base_param_string,
        evaluation_arguments,
    )


def run_code_experiment(
    EXPERIMENT_TITLE, OUTPUT_FILE, code, code_kwargs={}, OUTPUT_FILE_LENGTH=None
):
    # check if folder for OUTPUT_FILE exists
    Path(os.path.dirname(OUTPUT_FILE)).mkdir(parents=True, exist_ok=True)
    # if not run it
    print("#" * 80)
    print(EXPERIMENT_TITLE + "\n")
    print("Saving to " + OUTPUT_FILE)

    # check if OUTPUT_FILE exists
    if os.path.isfile(OUTPUT_FILE):
        if OUTPUT_FILE_LENGTH is not None:
            if sum(1 for l in open(OUTPUT_FILE)) >= OUTPUT_FILE_LENGTH:
                return
        else:
            return

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
    EXPERIMENT_TITLE,
    OUTPUT_FILE,
    CLI_COMMAND,
    CLI_ARGUMENTS,
    OUTPUT_FILE_LENGTH=None,
    SAVE_ARGUMENT_JSON=True,
):
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
    EXPERIMENT_TITLE,
    OUTPUT_FILE,
    CLI_COMMAND,
    CLI_ARGUMENTS,
    RANDOM_IDS=None,
    PARALLEL_OFFSET=0,
    PARALLEL_AMOUNT=0,
    OUTPUT_FILE_LENGTH=None,
    SAVE_ARGUMENT_JSON=True,
    RESTART_IF_NOT_ENOUGH_SAMPLES=False,
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

    # if file exists already and isn't empty we don't need to recreate all samples
    if Path(OUTPUT_FILE).is_file():
        existing_length = sum(1 for l in open(OUTPUT_FILE)) - 1
        PARALLEL_AMOUNT -= existing_length
        PARALLEL_OFFSET = existing_length

    if RANDOM_IDS:
        ids = RANDOM_IDS
    else:
        ids = range(1, PARALLEL_AMOUNT + 1)

    def code(CLI_COMMAND, PARALLEL_AMOUNT, PARALLEL_OFFSET):
        with Parallel(
            #  n_jobs=1,
            multiprocessing.cpu_count(),
            backend="threading",
        ) as parallel:
            output = parallel(
                delayed(run_parallel)(CLI_COMMAND, k + PARALLEL_OFFSET) for k in ids
            )

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
    if RESTART_IF_NOT_ENOUGH_SAMPLES:
        error_stop_counter = 3

        while (
            error_stop_counter > 0
            and (sum(1 for l in open(OUTPUT_FILE)) or not Path(OUTPUT_FILE).is_file())
            <= OUTPUT_FILE_LENGTH
        ):
            if Path(OUTPUT_FILE).is_file():
                amount_of_existing_states = sum(1 for l in open(OUTPUT_FILE))
            else:
                amount_of_existing_states = 0

            amount_of_missing_training_samples = (
                TRAIN_NR_LEARNING_SAMPLES - amount_of_existing_states
            )

            amount_of_processes = amount_of_missing_training_samples / (
                USER_QUERY_BUDGET_LIMIT / NR_QUERIES_PER_ITERATION
            )

            amount_of_processes = math.ceil(amount_of_processes)

            print("running ", amount_of_processes, "processes")
            run_code_experiment(
                EXPERIMENT_TITLE,
                OUTPUT_FILE,
                code=code,
                code_kwargs={
                    "CLI_COMMAND": CLI_COMMAND,
                    "PARALLEL_AMOUNT": amount_of_processes,
                    "PARALLEL_OFFSET": PARALLEL_OFFSET,
                },
                OUTPUT_FILE_LENGTH=OUTPUT_FILE_LENGTH,
            )

            new_amount_of_existing_states = sum(1 for l in open(OUTPUT_FILE))
            if new_amount_of_existing_states == amount_of_existing_states:
                error_stop_counter -= 1

        if sum(1 for l in open(OUTPUT_FILE)) > OUTPUT_FILE_LENGTH + 1:
            print(OUTPUT_FILE)
            print(os.path.dirname(OUTPUT_FILE))
            # black magic to trim file using python
            with open(OUTPUT_FILE, "r+") as f:
                with open(os.path.dirname(OUTPUT_FILE) + "/opt_pol.csv", "r+") as f2:
                    lines = f.readlines()
                    lines2 = f2.readlines()
                    f.seek(0)
                    f2.seek(0)

                    counter = 0
                    for l in lines:
                        counter += 1
                        if counter <= OUTPUT_FILE_LENGTH + 1:
                            f.write(l)
                    f.truncate()

                    counter = 0
                    for l in lines2:
                        counter += 1
                        if counter <= OUTPUT_FILE_LENGTH + 1:
                            f2.write(l)

                    f2.truncate()
