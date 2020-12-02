import os
import json
from pathlib import Path

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
    suffix,
) = get_config()

for DATASET_NAME in [
    #  "emnist-byclass-test",
    "synthetic",
    #  "dwtc",
    #  "BREAST",
    #  "DIABETES",
    #  "FERTILITY",
    #  "GERMAN",
    #  "HABERMAN",
    #  "HEART",
    #  "ILPD",
    #  "IONOSPHERE",
    #  "PIMA",
    #  "PLANNING",
    #  "australian",
]:
    if config.FINAL_PICTURE == "":
        comparison_path = (
            PARENT_OUTPUT_DIRECTORY
            #  + test_base_param_string
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
        df["sampling"] = "Imitation Learned Neural Network " + str(
            config.BASE_PARAM_STRING
        )

        if config.INCLUDE_OPTIMAL_IN_PLOT or config.INCLUDE_ONLY_OPTIMAL_IN_PLOT:
            optimal_df = pd.read_csv(OPTIMAL_OUTPUT_FILE)
            optimal_df["sampling"] = "Optimal Strategy"
            df = pd.concat([df, optimal_df])

        if config.INCLUDE_ONLY_OPTIMAL_IN_PLOT:
            df = optimal_df

        for comparison in config.TEST_COMPARISONS:
            df2 = pd.read_csv(
                PARENT_OUTPUT_DIRECTORY + "classics/" + comparison
                #  + test_base_param_string
                + ".csv",
                index_col=None,
                nrows=1 + config.TEST_NR_LEARNING_SAMPLES,
            )
            df = pd.concat([df, df2])

        #  print(df)
        df.to_csv(comparison_path, index=False)

    run_code_experiment(
        "Generating evaluation CSVs",
        comparison_path,
        code=concatenate_evaluation_csvs,
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

    def plot_all_metrics_as_a_table(df):
        sources = []
        sources = df["sampling"].unique()

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

    df = pd.read_csv(comparison_path, index_col=None)
    run_code_experiment(
        "Printing dataset_metrics",
        METRIC_TABLE_SUMMARY,
        code=plot_all_metrics_as_a_table,
        code_kwargs={"df": df},
    )
    test_base_param_string = original_test_base_param_string
