import glob
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
    _,
    _,
    _,
    PARENT_OUTPUT_DIRECTORY,
    _,
    test_base_param_string,
) = get_config()

# this file expects /*/dataset_creation.csv and *_synthetic.csv files, and just concatenates them, and displays that afterwards
DATASET_NAME = "synthetic"

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
    df = pd.DataFrame()

    for comparison in config.TEST_COMPARISONS:
        df2 = pd.read_csv(
            PARENT_OUTPUT_DIRECTORY + "classics/" + comparison
            #  + test_base_param_string
            + ".csv",
            index_col=None,
            nrows=1 + config.TEST_NR_LEARNING_SAMPLES,
        )
        df = pd.concat([df, df2])

    for csv_file in list(glob.glob(PARENT_OUTPUT_DIRECTORY + "*synthetic.csv")):
        more_results = pd.read_csv(
            csv_file, index_col=None, nrows=config.TEST_NR_LEARNING_SAMPLES
        )
        more_results["sampling"] = csv_file.split("/")[-1]
        df = pd.concat([df, more_results])

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
    with open(config.FINAL_PICTURE + "_table.txt", "a") as f:
        f.write(
            tabulate(
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
    config.FINAL_PICTURE + "_tabble.txt",
    code=plot_all_metrics_as_a_table,
    code_kwargs={"df": df},
)
