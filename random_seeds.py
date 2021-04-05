import argparse
import csv
import math
import multiprocessing
import os
import random
import sys
from timeit import default_timer as timer
from operator import itemgetter
import dill
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from alipy.data_manipulate.al_split import split
from alipy.experiment.al_experiment import AlExperiment
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("--OUTPUT_PATH", default="../datasets/ali")
parser.add_argument("--DATASET_IDS")
parser.add_argument("--NON_SLURM", action="store_true")
parser.add_argument(
    "--AMOUNT_OF_RUNS", type=int, default=1, help="Specifies which dataset to use etc."
)
config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()

config.DATASET_IDS = [int(item) for item in config.DATASET_IDS.split(",")]

# this file creates a file called random_ids.csv which contains a list of random_ids for which the resulting data hasn't been created so far
# it contains in a second column the dataset we are dealing with about, and in a third column the id of the AL strategy to use
# the file baseline_comparison.py expects as the parameter "RANDOM_SEED_INDEX" the index for which the random seeds from random_ids.csv should be read

if os.path.isfile(config.OUTPUT_PATH + "/result.csv"):
    result_df = pd.read_csv(
        config.OUTPUT_PATH + "/result.csv",
        index_col=None,
        usecols=["dataset_id", "strategy_id", "dataset_random_seed"],
    )
else:
    result_df = pd.DataFrame(
        data=None, columns=["dataset_id", "strategy_id", "dataset_random_seed"]
    )

random_ids_which_should_be_in_evaluation = set(range(0, config.AMOUNT_OF_RUNS))

if config.NON_SLURM:
    #  strategy_ids_which_should_be_in_evaluation = set([10])
    #  strategy_ids_which_should_be_in_evaluation = set([12])
    strategy_ids_which_should_be_in_evaluation = set([10, 11, 12])
    strategy_ids_which_should_be_in_evaluation = set([15, 23])
    strategy_ids_which_should_be_in_evaluation = set([12, 11])
    strategy_ids_which_should_be_in_evaluation = set([12, 7])
else:
    strategy_ids_which_should_be_in_evaluation = set(range(1, 9 + 1))
    #  strategy_ids_which_should_be_in_evaluation.add(14)
    strategy_ids_which_should_be_in_evaluation.add(13)  # batch
    strategy_ids_which_should_be_in_evaluation.add(15)  # single_10
    #  strategy_ids_which_should_be_in_evaluation = set([16, 17, 18, 19, 20, 21])
    #  strategy_ids_which_should_be_in_evaluation = set([4, 15, 23])
    strategy_ids_which_should_be_in_evaluation = set([34,35])

missing_ids = []

for dataset_id, strategy_id, dataset_random_seed in itertools.product(
    config.DATASET_IDS,
    strategy_ids_which_should_be_in_evaluation,
    random_ids_which_should_be_in_evaluation,
    repeat=1,
):
    if (
        len(
            result_df.loc[
                (result_df["dataset_id"] == dataset_id)
                & (result_df["strategy_id"] == strategy_id)
                & (result_df["dataset_random_seed"] == dataset_random_seed)
            ]
        )
        == 0
    ):
        missing_ids.append([dataset_id, strategy_id, dataset_random_seed])


random_seed_df = pd.DataFrame(
    data=missing_ids, columns=["dataset_id", "strategy_id", "dataset_random_seed"]
)
print(len(random_seed_df))
random_seed_df.to_csv(config.OUTPUT_PATH + "/random_seeds.csv", header=True)
