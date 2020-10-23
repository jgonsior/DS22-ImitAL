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

from active_learning.experiment_setup_lib import standard_config

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
        (["--TRAIN_VARIANCE_BOUND"], {"type": int, "default": 1}),
        (["--TRAIN_HYPERCUBE"], {"action": "store_true"}),
        (["--TRAIN_NEW_SYNTHETIC_PARAMS"], {"action": "store_false"}),
        (["--TRAIN_CONVEX_HULL_SAMPLING"], {"action": "store_false"}),
        (["--TRAIN_STOP_AFTER_MAXIMUM_ACCURACY_REACHED"], {"action": "store_false"}),
        (["--TRAIN_GENERATE_NOISE"], {"action": "store_true"}),
        (["--TRAIN_STATE_DIFF_PROBAS"], {"action": "store_true"}),
        (["--TRAIN_STATE_ARGSECOND_PROBAS"], {"action": "store_true"}),
        (["--TRAIN_STATE_ARGTHIRD_PROBAS"], {"action": "store_true"}),
        (["--TRAIN_STATE_DISTANCES_LAB"], {"action": "store_true"}),
        (["--TRAIN_STATE_DISTANCES_UNLAB"], {"action": "store_true"}),
        (["--TRAIN_STATE_PREDICTED_CLASS"], {"action": "store_true"}),
        (["--TRAIN_STATE_NO_LRU_WEIGHTS"], {"action": "store_true"}),
        (["--TRAIN_STATE_LRU_AREAS_LIMIT"], {"type": int, "default": 0}),
        (["--TEST_VARIABLE_DATASET"], {"action": "store_false"}),
        (["--TEST_NR_LEARNING_SAMPLES"], {"type": int, "default": 500}),
        (["--TEST_AMOUNT_OF_FEATURES"], {"type": int, "default": -1}),
        (["--TEST_HYPERCUBE"], {"action": "store_true"}),
        (["--TEST_NEW_SYNTHETIC_PARAMS"], {"action": "store_true"}),
        (["--TEST_CONVEX_HULL_SAMPLING"], {"action": "store_false"}),
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
        (["--NR_HIDDEN_NEURONS"], {"type": int, "default": 300}),
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

PARENT_OUTPUT_DIRECTORY = config.OUTPUT_DIRECTORY


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
    PARALLEL_OFFSET,
    PARALLEL_AMOUNT,
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

    def code(CLI_COMMAND, PARALLEL_AMOUNT, PARALLEL_OFFSET):
        with Parallel(
            #  n_jobs=1,
            multiprocessing.cpu_count(),
            backend="threading",
        ) as parallel:
            output = parallel(
                delayed(run_parallel)(CLI_COMMAND, k + PARALLEL_OFFSET)
                for k in range(1, PARALLEL_AMOUNT + 1)
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


shared_arguments = {
    "CLUSTER": "dummy",
    "NR_QUERIES_PER_ITERATION": config.NR_QUERIES_PER_ITERATION,
    "START_SET_SIZE": 1,
    "USER_QUERY_BUDGET_LIMIT": config.USER_QUERY_BUDGET_LIMIT,
    "N_JOBS": 1,
}

if not config.SKIP_TRAINING_DATA_GENERATION:
    OUTPUT_FILE = PARENT_OUTPUT_DIRECTORY + train_base_param_string
    run_parallel_experiment(
        "Creating dataset",
        OUTPUT_FILE=PARENT_OUTPUT_DIRECTORY + train_base_param_string + "/states.csv",
        CLI_COMMAND="python imit_training.py",
        CLI_ARGUMENTS={
            "DATASETS_PATH": "../datasets",
            "OUTPUT_DIRECTORY": PARENT_OUTPUT_DIRECTORY + train_base_param_string,
            "DATASET_NAME": "synthetic",
            "SAMPLING": "trained_nn",
            "AMOUNT_OF_PEAKED_OBJECTS": config.TRAIN_AMOUNT_OF_PEAKED_SAMPLES,
            "MAX_AMOUNT_OF_WS_PEAKS": 0,
            "AMOUNT_OF_LEARN_ITERATIONS": 1,
            "AMOUNT_OF_FEATURES": config.TRAIN_AMOUNT_OF_FEATURES,
            "VARIANCE_BOUND": config.TRAIN_VARIANCE_BOUND,
            "VARIABLE_DATASET": config.TRAIN_VARIABLE_DATASET,
            "NEW_SYNTHETIC_PARAMS": config.TRAIN_NEW_SYNTHETIC_PARAMS,
            "HYPERCUBE": config.TRAIN_HYPERCUBE,
            "CONVEX_HULL_SAMPLING": config.TRAIN_CONVEX_HULL_SAMPLING,
            "STOP_AFTER_MAXIMUM_ACCURACY_REACHED": config.TRAIN_STOP_AFTER_MAXIMUM_ACCURACY_REACHED,
            "GENERATE_NOISE": config.TRAIN_GENERATE_NOISE,
            "STATE_LRU_AREAS_LIMIT": config.TRAIN_STATE_LRU_AREAS_LIMIT,
            "STATE_DISTANCES_LAB": config.TRAIN_STATE_DISTANCES_LAB,
            "STATE_DISTANCES_UNLAB": config.TRAIN_STATE_DISTANCES_UNLAB,
            "STATE_PREDICTED_CLASS": config.TRAIN_STATE_PREDICTED_CLASS,
            "STATE_ARGSECOND_PROBAS": config.TRAIN_STATE_ARGSECOND_PROBAS,
            "STATE_ARGTHIRD_PROBAS": config.TRAIN_STATE_ARGTHIRD_PROBAS,
            "STATE_DIFF_PROBAS": config.TRAIN_STATE_DIFF_PROBAS,
            "STATE_NO_LRU_WEIGHTS": config.TRAIN_STATE_NO_LRU_WEIGHTS,
            **shared_arguments,
        },
        PARALLEL_OFFSET=0,
        PARALLEL_AMOUNT=config.TRAIN_NR_LEARNING_SAMPLES,
        OUTPUT_FILE_LENGTH=config.TRAIN_NR_LEARNING_SAMPLES,
        RESTART_IF_NOT_ENOUGH_SAMPLES=True,
    )

if config.ONLY_TRAINING_DATA:
    exit(1)


run_python_experiment(
    "Train ANN",
    config.OUTPUT_DIRECTORY + train_base_param_string + "/trained_ann.pickle",
    CLI_COMMAND="python train_lstm.py",
    CLI_ARGUMENTS={
        "DATA_PATH": config.OUTPUT_DIRECTORY + train_base_param_string,
        "STATE_ENCODING": "listwise",
        "TARGET_ENCODING": "binary",
        "SAVE_DESTINATION": config.OUTPUT_DIRECTORY
        + train_base_param_string
        + "/trained_ann.pickle",
        "REGULAR_DROPOUT_RATE": 0.1,
        "OPTIMIZER": "Nadam",
        "NR_HIDDEN_NEURONS": config.NR_HIDDEN_NEURONS,
        "NR_HIDDEN_LAYERS": 2,
        "LOSS": "MeanSquaredError",
        "EPOCHS": 10000,
        "BATCH_SIZE": 32,
        "ACTIVATION": "tanh",
        "RANDOM_SEED": 1,
    },
)

evaluation_arguments = {
    #  "DATASET_NAME": "synthetic",
    "AMOUNT_OF_FEATURES": config.TEST_AMOUNT_OF_FEATURES,
    "CLASSIFIER": config.TEST_CLASSIFIER,
    "VARIABLE_DATASET": config.TEST_VARIABLE_DATASET,
    "NEW_SYNTHETIC_PARAMS": config.TEST_NEW_SYNTHETIC_PARAMS,
    "HYPERCUBE": config.TEST_HYPERCUBE,
    "CONVEX_HULL_SAMPLING": config.TEST_CONVEX_HULL_SAMPLING,
    "GENERATE_NOISE": config.TEST_GENERATE_NOISE,
    **shared_arguments,
}


for DATASET_NAME in [
    #  "emnist-byclass-test",
    "synthetic",
    "dwtc",
    "BREAST",
    "DIABETES",
    "FERTILITY",
    "GERMAN",
    "HABERMAN",
    "HEART",
    "ILPD",
    "IONOSPHERE",
    "PIMA",
    "PLANNING",
    "australian",
]:
    if DATASET_NAME != "synthetic":
        config.TEST_NR_LEARNING_SAMPLES = 100
        evaluation_arguments["USER_QUERY_BUDGET_LIMIT"] = 20
    evaluation_arguments["DATASET_NAME"] = DATASET_NAME

    EVALUATION_FILE_TRAINED_NN_PATH = (
        config.OUTPUT_DIRECTORY + config.BASE_PARAM_STRING + "_" + DATASET_NAME + ".csv"
    )

    original_test_base_param_string = test_base_param_string
    test_base_param_string += "_" + DATASET_NAME

    run_parallel_experiment(
        "Creating ann-evaluation data",
        OUTPUT_FILE=EVALUATION_FILE_TRAINED_NN_PATH,
        CLI_COMMAND="python single_al_cycle.py",
        CLI_ARGUMENTS={
            "NN_BINARY": config.OUTPUT_DIRECTORY
            + train_base_param_string
            + "/trained_ann.pickle",
            "OUTPUT_DIRECTORY": EVALUATION_FILE_TRAINED_NN_PATH,
            "SAMPLING": "trained_nn",
            "STATE_LRU_AREAS_LIMIT": config.TRAIN_STATE_LRU_AREAS_LIMIT,
            "STATE_DISTANCES_LAB": config.TRAIN_STATE_DISTANCES_LAB,
            "STATE_DISTANCES_UNLAB": config.TRAIN_STATE_DISTANCES_UNLAB,
            "STATE_PREDICTED_CLASS": config.TRAIN_STATE_PREDICTED_CLASS,
            "STATE_ARGSECOND_PROBAS": config.TRAIN_STATE_ARGSECOND_PROBAS,
            "STATE_ARGTHIRD_PROBAS": config.TRAIN_STATE_ARGTHIRD_PROBAS,
            "STATE_DIFF_PROBAS": config.TRAIN_STATE_DIFF_PROBAS,
            "STATE_NO_LRU_WEIGHTS": config.TRAIN_STATE_NO_LRU_WEIGHTS,
            **evaluation_arguments,
        },
        PARALLEL_OFFSET=100000,
        PARALLEL_AMOUNT=config.TEST_NR_LEARNING_SAMPLES,
        OUTPUT_FILE_LENGTH=config.TEST_NR_LEARNING_SAMPLES,
    )

    # rename sampling column
    p = Path(EVALUATION_FILE_TRAINED_NN_PATH)
    text = p.read_text()
    text = text.replace("trained_nn", config.OUTPUT_DIRECTORY)
    p.write_text(text)

    for comparison in config.TEST_COMPARISONS:
        COMPARISON_PATH = (
            PARENT_OUTPUT_DIRECTORY
            + "classics/"
            + comparison
            + test_base_param_string
            + ".csv"
        )
        run_parallel_experiment(
            "Creating " + comparison + "-evaluation data",
            OUTPUT_FILE=COMPARISON_PATH,
            CLI_COMMAND="python single_al_cycle.py",
            CLI_ARGUMENTS={
                "OUTPUT_DIRECTORY": COMPARISON_PATH,
                "SAMPLING": comparison,
                **evaluation_arguments,
            },
            PARALLEL_OFFSET=100000,
            PARALLEL_AMOUNT=config.TEST_NR_LEARNING_SAMPLES,
            OUTPUT_FILE_LENGTH=config.TEST_NR_LEARNING_SAMPLES,
        )

    if config.FINAL_PICTURE == "":
        comparison_path = (
            PARENT_OUTPUT_DIRECTORY
            + test_base_param_string
            + "_".join(config.TEST_COMPARISONS)
            + ".csv"
        )
    else:
        comparison_path = config.FINAL_PICTURE + "_" + DATASET_NAME

    def concatenate_evaluation_csvs():
        df = pd.read_csv(
            EVALUATION_FILE_TRAINED_NN_PATH,
            index_col=None,
            nrows=1 + config.TEST_NR_LEARNING_SAMPLES,
        )

        for comparison in config.TEST_COMPARISONS:
            df2 = pd.read_csv(
                PARENT_OUTPUT_DIRECTORY
                + "classics/"
                + comparison
                + test_base_param_string
                + ".csv",
                index_col=None,
                nrows=1 + config.TEST_NR_LEARNING_SAMPLES,
            )
            df = pd.concat([df, df2])

        #  print(df)
        df.to_csv(comparison_path, index=False)

    run_code_experiment(
        "Generating evaluation CSVs", comparison_path, code=concatenate_evaluation_csvs
    )

    run_python_experiment(
        "Evaluation plots",
        comparison_path + ".png",
        CLI_COMMAND="python compare_distributions.py",
        CLI_ARGUMENTS={
            "CSV_FILE": comparison_path,
            "GROUP_COLUMNS": "sampling",
            "SAVE_FILE": comparison_path,
            "TITLE": comparison_path,
            "METRIC": config.PLOT_METRIC,
        },
    )

    # print evaluation table using all metrics at once

    METRIC_TABLE_SUMMARY = config.FINAL_PICTURE + "_" + DATASET_NAME + "_" "table.txt"

    def plot_all_metrics_as_a_table():
        sources = []
        df = pd.read_csv(
            EVALUATION_FILE_TRAINED_NN_PATH,
            index_col=None,
            nrows=1 + config.TEST_NR_LEARNING_SAMPLES,
        )
        sources.append(df["sampling"][0])

        for comparison in config.TEST_COMPARISONS:
            df2 = pd.read_csv(
                PARENT_OUTPUT_DIRECTORY
                + "classics/"
                + comparison
                + test_base_param_string
                + ".csv",
                index_col=None,
                nrows=1 + config.TEST_NR_LEARNING_SAMPLES,
            )
            sources.append(df2["sampling"][0])
            df = pd.concat([df, df2])

        metrics = [
            "acc_auc",
            "acc_test",
            "acc_train",
            "roc_auc_macro_oracle",
            "roc_auc_weighted_oracle",
        ]

        table = pd.DataFrame(columns=["Source", *metrics])
        for source in sources:
            metric_values = {"Source": source}
            for metric in metrics:
                metric_values[metric] = df.loc[df["sampling"] == source][metric].mean()
                #  metric_values[metric] = df.loc[df["sampling"] == source][
                #      metric
                #  ].median()
                metric_values[metric + "_diff_to_mm"] = np.NaN
            table = table.append(metric_values, ignore_index=True)

        # add to table +- columns for each metric
        for source in sources:
            for metric in metrics:
                max_value = table.loc[table["Source"] == "uncertainty_max_margin"][
                    metric
                ].max()
                table.loc[table["Source"] == source, metric + "_diff_to_mm"] = (
                    table.loc[table["Source"] == source][metric] - max_value
                )

        print(
            tabulate(
                table,
                headers="keys",
                showindex=False,
                floatfmt=".2%",
                tablefmt="fancy_grid",
            )
        )
        with open(METRIC_TABLE_SUMMARY, "w") as f:
            f.write(
                "\n"
                + DATASET_NAME
                + "\n\n\n"
                + tabulate(
                    table,
                    headers="keys",
                    showindex=False,
                    floatfmt=".2%",
                    tablefmt="fancy_grid",
                )
            )
        with open(config.FINAL_PICTURE + "_table.txt", "a") as f:
            f.write(
                "\n"
                + DATASET_NAME
                + "\n\n\n"
                + tabulate(
                    table,
                    headers="keys",
                    showindex=False,
                    floatfmt=".2%",
                    tablefmt="fancy_grid",
                )
            )

    run_code_experiment(
        "Printing dataset_metrics",
        METRIC_TABLE_SUMMARY,
        code=plot_all_metrics_as_a_table,
    )
    test_base_param_string = original_test_base_param_string
    #  exit(-1)
