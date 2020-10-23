import numpy as np
from tabulate import tabulate
import json
import math
import multiprocessing
import os
from pathlib import Path
import time
import pandas as pd
from joblib import Parallel, delayed


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
                delayed(run_parallel)(CLI_COMMAND, k + PARALLEL_OFFSET)
                for k in RANDOM_IDS
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
                config.TRAIN_NR_LEARNING_SAMPLES - amount_of_existing_states
            )

            amount_of_processes = amount_of_missing_training_samples / (
                config.USER_QUERY_BUDGET_LIMIT / config.NR_QUERIES_PER_ITERATION
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
