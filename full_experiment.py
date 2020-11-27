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
) = get_config()

if not config.SKIP_TRAINING_DATA_GENERATION:
    OUTPUT_FILE = PARENT_OUTPUT_DIRECTORY + train_base_param_string
    run_parallel_experiment(
        "Creating dataset",
        OUTPUT_FILE=PARENT_OUTPUT_DIRECTORY
        + train_base_param_string
        + "/dataset_creation.csv",
        CLI_COMMAND="python imit_training.py",
        CLI_ARGUMENTS={
            "DATASETS_PATH": "../datasets",
            "CLASSIFIER": config.TRAIN_CLASSIFIER,
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
            "STATE_PREDICTED_UNITY": config.TRAIN_STATE_PREDICTED_UNITY,
            "STATE_ARGSECOND_PROBAS": config.TRAIN_STATE_ARGSECOND_PROBAS,
            "STATE_ARGTHIRD_PROBAS": config.TRAIN_STATE_ARGTHIRD_PROBAS,
            "STATE_DIFF_PROBAS": config.TRAIN_STATE_DIFF_PROBAS,
            "STATE_DISTANCES": config.TRAIN_STATE_DISTANCES,
            "STATE_UNCERTAINTIES": config.TRAIN_STATE_UNCERTAINTIES,
            "INITIAL_BATCH_SAMPLING_METHOD": config.INITIAL_BATCH_SAMPLING_METHOD,
            "INITIAL_BATCH_SAMPLING_ARG": config.INITIAL_BATCH_SAMPLING_ARG,
            "INITIAL_BATCH_SAMPLING_HYBRID_UNCERT": config.INITIAL_BATCH_SAMPLING_HYBRID_UNCERT,
            "INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST": config.INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST,
            "INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB": config.INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB,
            "INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY": config.INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY,
            **shared_arguments,
        },
        PARALLEL_OFFSET=0,
        PARALLEL_AMOUNT=config.TRAIN_NR_LEARNING_SAMPLES,
        OUTPUT_FILE_LENGTH=config.TRAIN_NR_LEARNING_SAMPLES,
        RESTART_IF_NOT_ENOUGH_SAMPLES=False,
    )

if config.ONLY_TRAINING_DATA:
    exit(1)

if config.HYPER_SEARCH:
    HYPER_SEARCH_OUTPUT_FILE = (
        config.OUTPUT_DIRECTORY + train_base_param_string + "/hyper_results.txt"
    )
    run_python_experiment(
        "ANN hyper_search",
        HYPER_SEARCH_OUTPUT_FILE,
        CLI_COMMAND="python train_lstm.py",
        CLI_ARGUMENTS={
            "DATA_PATH": config.OUTPUT_DIRECTORY + train_base_param_string,
            "STATE_ENCODING": "listwise",
            "TARGET_ENCODING": "binary",
            "SAVE_DESTINATION": config.OUTPUT_DIRECTORY
            + train_base_param_string
            + "/trained_ann.pickle",
            "RANDOM_SEED": 1,
            "HYPER_SEARCH": True,
            "N_ITER": config.NR_ANN_HYPER_SEARCH_ITERATIONS,
        },
    )

if not config.SKIP_ANN_EVAL:
    HYPER_SEARCH_OUTPUT_FILE = (
        config.OUTPUT_DIRECTORY + train_base_param_string + "/hyper_results.txt"
    )
    print(HYPER_SEARCH_OUTPUT_FILE)
    assert os.path.exists(HYPER_SEARCH_OUTPUT_FILE)

    with open(HYPER_SEARCH_OUTPUT_FILE, "r") as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
        lower_params = json.loads(last_line)
        ANN_HYPER_PARAMS = {}
        for k, v in lower_params.items():
            ANN_HYPER_PARAMS[k.upper()] = v

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
            "REGULAR_DROPOUT_RATE": ANN_HYPER_PARAMS["REGULAR_DROPOUT_RATE"],
            "OPTIMIZER": ANN_HYPER_PARAMS["OPTIMIZER"],
            "NR_HIDDEN_NEURONS": ANN_HYPER_PARAMS["NR_HIDDEN_NEURONS"],
            "NR_HIDDEN_LAYERS": ANN_HYPER_PARAMS["NR_HIDDEN_LAYERS"],
            "LOSS": ANN_HYPER_PARAMS["LOSS"],
            "EPOCHS": ANN_HYPER_PARAMS["EPOCHS"],
            "BATCH_SIZE": ANN_HYPER_PARAMS["BATCH_SIZE"],
            "ACTIVATION": ANN_HYPER_PARAMS["ACTIVATION"],
            "RANDOM_SEED": 1,
        },
    )


if config.INCLUDE_OPTIMAL_IN_PLOT or config.INCLUDE_ONLY_OPTIMAL_IN_PLOT:
    OPTIMAL_OUTPUT_FILE = PARENT_OUTPUT_DIRECTORY + train_base_param_string + "_optimal"
    run_parallel_experiment(
        "Optimal evaluation",
        OUTPUT_FILE=OPTIMAL_OUTPUT_FILE + "/dataset_creation.csv",
        CLI_COMMAND="python imit_training.py",
        CLI_ARGUMENTS={
            "DATASETS_PATH": "../datasets",
            "CLASSIFIER": config.TRAIN_CLASSIFIER,
            "OUTPUT_DIRECTORY": OPTIMAL_OUTPUT_FILE,
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
            "STATE_PREDICTED_UNITY": config.TRAIN_STATE_PREDICTED_UNITY,
            "STATE_ARGSECOND_PROBAS": config.TRAIN_STATE_ARGSECOND_PROBAS,
            "STATE_ARGTHIRD_PROBAS": config.TRAIN_STATE_ARGTHIRD_PROBAS,
            "STATE_DIFF_PROBAS": config.TRAIN_STATE_DIFF_PROBAS,
            "STATE_DISTANCES": config.TRAIN_STATE_DISTANCES,
            "STATE_UNCERTAINTIES": config.TRAIN_STATE_UNCERTAINTIES,
            "INITIAL_BATCH_SAMPLING_METHOD": config.INITIAL_BATCH_SAMPLING_METHOD,
            "INITIAL_BATCH_SAMPLING_ARG": config.INITIAL_BATCH_SAMPLING_ARG,
            "INITIAL_BATCH_SAMPLING_HYBRID_UNCERT": config.INITIAL_BATCH_SAMPLING_HYBRID_UNCERT,
            "INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST": config.INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST,
            "INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB": config.INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB,
            "INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY": config.INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY,
            **shared_arguments,
        },
        PARALLEL_OFFSET=config.TEST_PARALLEL_OFFSET,
        PARALLEL_AMOUNT=config.TRAIN_NR_LEARNING_SAMPLES,
        OUTPUT_FILE_LENGTH=config.TRAIN_NR_LEARNING_SAMPLES,
        RESTART_IF_NOT_ENOUGH_SAMPLES=False,
    )
    OPTIMAL_OUTPUT_FILE += "/dataset_creation.csv"

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
    if DATASET_NAME != "synthetic":
        #  config.TEST_NR_LEARNING_SAMPLES = 100
        evaluation_arguments["USER_QUERY_BUDGET_LIMIT"] = 20
    evaluation_arguments["DATASET_NAME"] = DATASET_NAME

    EVALUATION_FILE_TRAINED_NN_PATH = (
        config.OUTPUT_DIRECTORY + config.BASE_PARAM_STRING + "_" + DATASET_NAME + ".csv"
    )

    original_test_base_param_string = test_base_param_string
    test_base_param_string += "_" + DATASET_NAME
    if not config.SKIP_ANN_EVAL:
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
                "STATE_DISTANCES_LAB": config.TRAIN_STATE_DISTANCES_LAB,
                "STATE_DISTANCES_UNLAB": config.TRAIN_STATE_DISTANCES_UNLAB,
                "STATE_PREDICTED_CLASS": config.TRAIN_STATE_PREDICTED_CLASS,
                "STATE_PREDICTED_UNITY": config.TRAIN_STATE_PREDICTED_UNITY,
                "STATE_ARGSECOND_PROBAS": config.TRAIN_STATE_ARGSECOND_PROBAS,
                "STATE_ARGTHIRD_PROBAS": config.TRAIN_STATE_ARGTHIRD_PROBAS,
                "STATE_DIFF_PROBAS": config.TRAIN_STATE_DIFF_PROBAS,
                "STATE_DISTANCES": config.TRAIN_STATE_DISTANCES,
                "STATE_UNCERTAINTIES": config.TRAIN_STATE_UNCERTAINTIES,
                "INITIAL_BATCH_SAMPLING_METHOD": config.INITIAL_BATCH_SAMPLING_METHOD,
                "INITIAL_BATCH_SAMPLING_ARG": config.INITIAL_BATCH_SAMPLING_ARG,
                "INITIAL_BATCH_SAMPLING_HYBRID_UNCERT": config.INITIAL_BATCH_SAMPLING_HYBRID_UNCERT,
                "INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST": config.INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST,
                "INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB": config.INITIAL_BATCH_SAMPLING_HYBRID_FURTHEST_LAB,
                "INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY": config.INITIAL_BATCH_SAMPLING_HYBRID_PRED_UNITY,
                **evaluation_arguments,
            },
            PARALLEL_OFFSET=config.TEST_PARALLEL_OFFSET,
            PARALLEL_AMOUNT=config.TEST_NR_LEARNING_SAMPLES,
            OUTPUT_FILE_LENGTH=config.TEST_NR_LEARNING_SAMPLES,
        )

        # rename sampling column
        p = Path(EVALUATION_FILE_TRAINED_NN_PATH)
        text = p.read_text()
        text = text.replace("trained_nn", config.OUTPUT_DIRECTORY)
        p.write_text(text)

    if config.STOP_AFTER_ANN_EVAL:
        exit(0)

        COMPARISON_PATH = (
            PARENT_OUTPUT_DIRECTORY
            + "classics/"
            + comparison
            #  + test_base_param_string
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
            PARALLEL_OFFSET=config.TEST_PARALLEL_OFFSET,
            PARALLEL_AMOUNT=config.TEST_NR_LEARNING_SAMPLES,
            OUTPUT_FILE_LENGTH=config.TEST_NR_LEARNING_SAMPLES,
        )

    if not config.SKIP_PLOTS:
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

        METRIC_TABLE_SUMMARY = (
            config.FINAL_PICTURE + "_" + DATASET_NAME + "_" "table.txt"
        )

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
                    metric_values[metric] = df.loc[df["sampling"] == source][
                        metric
                    ].mean()
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
        #  exit(-1)
