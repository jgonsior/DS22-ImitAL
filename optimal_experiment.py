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
from experiments_lib import (
    run_code_experiment,
    run_python_experiment,
    run_parallel_experiment,
    get_config,
)

(
    config,
    shared_arguments,
    PARENT_OUTPUT_DIRECTORY,
    train_base_param_string,
    test_base_param_string,
    evaluation_arguments,
) = get_config()

if not config.SKIP_TRAINING_DATA_GENERATION:

    for initial_batch_sampling_method, initial_batch_sampling_arg in [
        ("random", -1),
        ("furthest", 10),
        ("furthest", 100),
        ("graph_density", -1),
    ]:
        OUTPUT_FILE = (
            PARENT_OUTPUT_DIRECTORY
            + train_base_param_string
            + "_"
            + initial_batch_sampling_method
        )
        run_parallel_experiment(
            "Creating dataset",
            OUTPUT_FILE=OUTPUT_FILE + "/states.csv",
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
                "VARIABLE_DATASET": config.TRAIN_VARIABLE_DATASET,
                "NEW_SYNTHETIC_PARAMS": config.TRAIN_NEW_SYNTHETIC_PARAMS,
                "HYPERCUBE": config.TRAIN_HYPERCUBE,
                "STOP_AFTER_MAXIMUM_ACCURACY_REACHED": config.TRAIN_STOP_AFTER_MAXIMUM_ACCURACY_REACHED,
                "GENERATE_NOISE": config.TRAIN_GENERATE_NOISE,
                "STATE_DISTANCES_LAB": config.TRAIN_STATE_DISTANCES_LAB,
                "STATE_DISTANCES_UNLAB": config.TRAIN_STATE_DISTANCES_UNLAB,
                "STATE_PREDICTED_CLASS": config.TRAIN_STATE_PREDICTED_CLASS,
                "STATE_ARGSECOND_PROBAS": config.TRAIN_STATE_ARGSECOND_PROBAS,
                "STATE_ARGTHIRD_PROBAS": config.TRAIN_STATE_ARGTHIRD_PROBAS,
                "STATE_DIFF_PROBAS": config.TRAIN_STATE_DIFF_PROBAS,
                "INITIAL_BATCH_SAMPLING_METHOD": initial_batch_sampling_method,
                "INITIAL_BATCH_SAMPLING_ARG": initial_batch_sampling_arg,
                **shared_arguments,
            },
            PARALLEL_OFFSET=0,
            PARALLEL_AMOUNT=config.TRAIN_NR_LEARNING_SAMPLES,
            OUTPUT_FILE_LENGTH=config.TRAIN_NR_LEARNING_SAMPLES,
            RESTART_IF_NOT_ENOUGH_SAMPLES=True,
        )

if config.ONLY_TRAINING_DATA:
    exit(1)


for DATASET_NAME in [
    "synthetic",
]:
    evaluation_arguments["DATASET_NAME"] = DATASET_NAME

    original_test_base_param_string = test_base_param_string
    test_base_param_string += "_" + DATASET_NAME

    optimal_results = pd.read_csv(
        PARENT_OUTPUT_DIRECTORY + config.BASE_PARAM_STRING + "/dataset_creation.csv",
        nrows=config.TEST_NR_LEARNING_SAMPLES + 10,
    )
    RANDOM_IDS = optimal_results["random_seed"].unique()[
        : config.TEST_NR_LEARNING_SAMPLES
    ]

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
        df = optimal_results
        df["sampling"] = "optimal"

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
        df = optimal_results
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
