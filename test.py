import pandas as pd
import glob
import os

pathes = "../datasets/tmp_taurus/"
#  pathes = "../datasets/nn_size_hypersearch/"


for path in glob.glob(pathes + "/TRAIN_*"):
    if path.endswith(".csv") or path.endswith(".json"):
        continue
    columns_to_leave = os.path.basename(path)
    print(columns_to_leave)
    states_to_column_mapping = {
        "TRAIN_STATE_ARGSECOND_PROBAS": "TRAIN_STATE_ARGSECOND_PROBAS",
        "TRAIN_STATE_ARGTHIRD_PROBAS": "TRAIN_STATE_ARGTHIRD_PROBAS --TRAIN_STATE_ARGSECOND_PROBAS",
        "TRAIN_STATE_DIFF_PROBAS": "TRAIN_STATE_DIFF_PROBAS",
        "TRAIN_STATE_LRU_AREAS_LIMIT": "TRAIN_STATE_LRU_AREAS_LIMIT 5",
        "TRAIN_STATE_PREDICTED_CLASS": "TRAIN_STATE_PREDICTED_CLASS",
        "TRAIN_STATE_DIFF_PRED": "TRAIN_STATE_DIFF_PROBAS --TRAIN_STATE_PREDICTED_CLASS",
        "TRAIN_STATE_DIFF_DIST": "TRAIN_STATE_DIFF_PROBAS --TRAIN_STATE_DISTANCES_LAB --TRAIN_STATE_DISTANCES_UNLAB",
        "TRAIN_STATE_DIFF_ARGTHIRD": "TRAIN_STATE_DIFF_PROBAS --TRAIN_STATE_ARGTHIRD_PROBAS",
        "TRAIN_STATE_ARGSECOND_ARGTHIRD": "TRAIN_STATE_ARGSECOND_PROBAS --TRAIN_STATE_ARGTHIRD_PROBAS",
        "TRAIN_STATE_ALL_STATES": "TRAIN_STATE_DIFF_PROBAS --TRAIN_STATE_ARGSECOND_PROBAS --TRAIN_STATE_ARGTHIRD_PROBAS --TRAIN_STATE_DISTANCES_LAB --TRAIN_STATE_DISTANCES_UNLAB --TRAIN_STATE_PREDICTED_CLASS --TRAIN_STATE_NO_LRU_WEIGHTS --TRAIN_STATE_LRU_AREAS_LIMIT 5",
        "TRAIN_STATE_ARGSECOND_ARGTHIRD_DISTANCES": "TRAIN_STATE_ARGSECOND_PROBAS --TRAIN_STATE_ARGTHIRD_PROBAS --TRAIN_STATE_DISTANCES_LAB --TRAIN_STATE_DISTANCES_UNLAB",
        "TRAIN_STATE_DISTANCES_LAB": "TRAIN_STATE_DISTANCES_LAB",
        "TRAIN_STATE_DISTANCES_UNLAB": "TRAIN_STATE_DISTANCES_UNLAB",
    }
    os.makedirs("plots/" + columns_to_leave, exist_ok=True)
    #  NR_HIDDEN_NEURONS = columns_to_leave[-3:]
    NR_HIDDEN_NEURONS = "300"

    cli_arguments = (
        "python full_experiment.py --RANDOM_SEED 1 --LOG_FILE log.txt --TEST_NR_LEARNING_SAMPLES 10 --OUTPUT_DIRECTORY "
        + pathes
        + " --SKIP_TRAINING_DATA_GENERATION --FINAL_PICTURE plots/"
        + columns_to_leave
        + "/plot --BASE_PARAM_STRING "
        + columns_to_leave
        + " --"
        + states_to_column_mapping[
            columns_to_leave
        ]  #  + states_to_column_mapping[columns_to_leave[:-4]]
        + " --NR_HIDDEN_NEURONS "
        + NR_HIDDEN_NEURONS
    )
    print(cli_arguments)
    os.system(cli_arguments)
    exit(-1)
exit(-2)


for path in glob.glob(pathes + "/TRAIN_*"):
    if path.endswith(".csv"):
        continue
    columns_to_leave = os.path.basename(path)

    states = pd.read_csv("../datasets/tmp_taurus/ALL_STATES/states.csv")
    opt_pol = pd.read_csv("../datasets/tmp_taurus/ALL_STATES/opt_pol.csv")

    states_to_column_mapping = {
        "TRAIN_STATE_ARGSECOND_PROBAS": ["_proba_argfirst", "_proba_argsecond"],
        "TRAIN_STATE_ARGTHIRD_PROBAS": [
            "_proba_argfirst",
            "_proba_argsecond",
            "_proba_argthird",
        ],
        "TRAIN_STATE_DIFF_PROBAS": ["_proba_argfirst", "_proba_diff"],
        "TRAIN_STATE_DISTANCES_LAB": ["_proba_argfirst", "_avg_dist_lab"],
        "TRAIN_STATE_DISTANCES_UNLAB": ["_proba_argfirst", "_avg_dist_unlab"],
        "TRAIN_STATE_LRU_AREAS_LIMIT": ["_proba_argfirst", "_lru_dist"],
        "TRAIN_STATE_PREDICTED_CLASS": ["_proba_argfirst", "_pred_class"],
        "TRAIN_STATE_DIFF_PRED": ["_proba_argfirst", "_proba_diff", "_pred_class"],
        "TRAIN_STATE_DIFF_DIST": [
            "_proba_argfirst",
            "_proba_diff",
            "_avg_dist_lab",
            "_avg_dist_unlab",
        ],
        "TRAIN_STATE_DIFF_ARGTHIRD": [
            "_proba_argfirst",
            "_proba_diff",
            "_proba_argthird",
        ],
        "TRAIN_STATE_ALL_STATES": [
            "_proba_argfirst",
            "_proba_argsecond",
            "_proba_argthird",
            "_proba_diff",
            "_lru_dist",
            "_pred_class",
            "_avg_dist_unlab",
            "_avg_dist_lab",
        ],
        "TRAIN_STATE_ARGSECOND_ARGTHIRD": [
            "_proba_argfirst",
            "_proba_argsecond",
            "_proba_argthird",
        ],
        "TRAIN_STATE_DIFF_LAB": [
            "_proba_argfirst",
            "_avg_dist_lab",
        ],
        "TRAIN_STATE_DIFF_UNLAB": ["_proba_argfirst", "_avg_dist_unlab"],
        "TRAIN_STATE_ARGSECOND_ARGTHIRD_DISTANCES": [
            "_proba_argfirst",
            "_proba_argsecond",
            "_proba_argthird",
            "_avg_dist_lab",
            "_avg_dist_unlab",
        ],
    }

    # filter out old columns

    columns_to_keep = []
    print(path)
    for column_to_remove in states_to_column_mapping[columns_to_leave]:
        for column in states.columns:
            if column.endswith(column_to_remove):
                columns_to_keep.append(column)
    for column in states.columns:
        if not column in columns_to_keep:
            del states[column]
    print(states)
    states.to_csv(path + "/states.csv", index=False)
    opt_pol.to_csv(path + "/opt_pol.csv", index=False)
