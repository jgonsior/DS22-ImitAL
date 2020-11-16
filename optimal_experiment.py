import glob

import numpy as np
import pandas as pd
from tabulate import tabulate

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

LIST_OF_BATCH_SAMPLING_METHODS = [
    #  ("random", -1),
    ("random", -1),
    ("furthest", 50),
    ("furthest_lab", 50),
    ("graph_density2", 50),
    #  ("uncertainty", 50),
    #  ("furthest", 100),
    #  ("furthest_lab", 100),
    #  ("graph_density2", 100),
    #  ("uncertainty", 100),
    #  ("furthest", 500),
    #  ("furthest_lab", 500),
    #  ("graph_density2", 500),
    #  ("uncertainty", 500),
    #  ("furthest", 1000),
    #  ("furthest_lab", 1000),
    #  ("graph_density2", 1000),
    ("uncertainty", 1000),
]

if not config.SKIP_TRAINING_DATA_GENERATION:
    for BATCH_MODE in ["batch", "single"]:
        for (
            initial_batch_sampling_method,
            initial_batch_sampling_arg,
        ) in LIST_OF_BATCH_SAMPLING_METHODS:
            if BATCH_MODE == "single" and initial_batch_sampling_method in [
                "graph_density2",
                "furthest_lab",
                "uncertainty",
            ]:
                continue
            OUTPUT_FILE = (
                PARENT_OUTPUT_DIRECTORY
                + train_base_param_string
                + "_"
                + BATCH_MODE
                + "_"
                + initial_batch_sampling_method
                + str(initial_batch_sampling_arg)
                + "/dataset_creation.csv"
            )
            print(OUTPUT_FILE)
            run_parallel_experiment(
                "Creating dataset",
                OUTPUT_FILE=OUTPUT_FILE,
                CLI_COMMAND="python imit_training.py",
                CLI_ARGUMENTS={
                    "DATASETS_PATH": "../datasets",
                    "OUTPUT_DIRECTORY": PARENT_OUTPUT_DIRECTORY
                    + train_base_param_string
                    + "_"
                    + BATCH_MODE
                    + "_"
                    + initial_batch_sampling_method
                    + str(initial_batch_sampling_arg),
                    "DATASET_NAME": "synthetic",
                    "SAMPLING": BATCH_MODE,
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
        df = pd.DataFrame()  # optimal_results

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

        for csv_file in list(
            glob.glob(PARENT_OUTPUT_DIRECTORY + "/*/dataset_creation.csv")
        ):
            print(csv_file)
            optimal_results = pd.read_csv(
                csv_file,
                nrows=config.TEST_NR_LEARNING_SAMPLES + 10,
            )
            optimal_results["sampling"] = csv_file.split("/")[-2]
            df = pd.concat([df, optimal_results])

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
        df = df  # optimal_results
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
