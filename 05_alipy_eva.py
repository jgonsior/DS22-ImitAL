from typing import Tuple
from ALiPy.alipy.experiment.al_experiment import AlExperiment
from active_learning.ALiPy_imitAL_Query_Strategy import ALiPY_ImitAL_Query_Strategy
import argparse
import csv
import math
import multiprocessing
import numpy as np
import pandas as pd
import random
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from timeit import default_timer as timer
from sklearn.metrics import auc
from active_learning.dataStorage import DataStorage
from active_learning.datasets.dwtc import load_dwtc
from active_learning.datasets.synthetic import load_synthetic
from active_learning.datasets.uci import load_uci
from active_learning.logger.logger import init_logger
from imitLearningPipelineSharedCode import dataset_id_mapping, strategy_id_mapping


init_logger("log.txt")

parser = argparse.ArgumentParser()
parser.add_argument("--DATASETS_PATH", default="../datasets")
parser.add_argument("--N_JOBS", type=int, default=1)
parser.add_argument(
    "--INDEX", type=int, default=1, help="Specifies which dataset to use etc."
)
parser.add_argument("--OUTPUT_PATH", default="../datasets/ali")
parser.add_argument("--RANDOM_SEEDS_INPUT_FILE")
parser.add_argument("--BATCH_SIZE", type=int, default=5)

parser.add_argument("--EXCLUDING", action="store_true")
parser.add_argument("--EXCLUDING_STATE_DIFF_PROBAS", action="store_true")
parser.add_argument("--EXCLUDING_STATE_ARGFIRST_PROBAS", action="store_true")
parser.add_argument("--EXCLUDING_STATE_ARGSECOND_PROBAS", action="store_true")
parser.add_argument("--EXCLUDING_STATE_ARGTHIRD_PROBAS", action="store_true")
parser.add_argument("--EXCLUDING_STATE_DISTANCES_LAB", action="store_true")
parser.add_argument("--EXCLUDING_STATE_DISTANCES_UNLAB", action="store_true")
parser.add_argument("--EXCLUDING_STATE_PREDICTED_CLASS", action="store_true")
parser.add_argument("--EXCLUDING_STATE_PREDICTED_UNITY", action="store_true")
parser.add_argument("--EXCLUDING_STATE_DISTANCES", action="store_true")
parser.add_argument("--EXCLUDING_STATE_UNCERTAINTIES", action="store_true")
parser.add_argument("--EXCLUDING_STATE_INCLUDE_NR_FEATURES", action="store_true")

config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()

print("Reading in ", config.RANDOM_SEEDS_INPUT_FILE)
# first open job file to get the real random_seed from it
random_seed_df = pd.read_csv(
    config.RANDOM_SEEDS_INPUT_FILE,
    header=0,
    index_col=0,
    nrows=config.INDEX + 1,
)
DATASET_ID, STRATEGY_ID, DATASET_RANDOM_SEED = random_seed_df.loc[config.INDEX]
"""
config.DATASETS_PATH = "/home/julius/datasets"

DATASET_ID = 0
STRATEGY_ID = 2
# use other random seeds than during training
# DATASET_RANDOM_SEED += 100000

DATASET_RANDOM_SEED = 0
"""


DATASET_RANDOM_SEED += 100000
np.random.seed(DATASET_RANDOM_SEED)
random.seed(DATASET_RANDOM_SEED)
DATASET_NAME = dataset_id_mapping[DATASET_ID][0]

print(DATASET_NAME)
print(DATASET_RANDOM_SEED)

if DATASET_NAME == "synthetic":
    df, synthetic_creation_args = load_synthetic(
        DATASET_RANDOM_SEED,
    )
elif DATASET_NAME == "dwtc":
    df, synthetic_creation_args = load_dwtc(
        DATASETS_PATH=config.DATASETS_PATH, RANDOM_SEED=DATASET_RANDOM_SEED
    )
else:
    df, synthetic_creation_args = load_uci(
        DATASETS_PATH=config.DATASETS_PATH,
        RANDOM_SEED=DATASET_RANDOM_SEED,
        DATASET_NAME=DATASET_NAME,
    )

if "NN_BINARY_PATH" in strategy_id_mapping[STRATEGY_ID][1].keys():
    data_storage = DataStorage(df, TEST_FRACTION=0)
    X = data_storage.X
    Y = data_storage.true_Y
else:
    X = df[df.columns[:-1]].to_numpy()
    Y = df["label"].to_numpy()

shuffling = np.random.permutation(len(Y))
X = X[shuffling]
Y = Y[shuffling]

scaler = RobustScaler()
X = scaler.fit_transform(X)

# scale back to [0,1]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)


# fancy ALiPy train/test split
test_ratio = 0.5
indices = [i for i in range(0, len(Y))]
train_idx = indices[: math.floor(len(Y) * (1 - test_ratio))]
test_idx = indices[math.floor(len(Y) * (1 - test_ratio)) :]
unlabel_idx = train_idx.copy()
label_idx = []

for label in np.unique(Y):
    if label not in Y[train_idx]:
        print(np.where(Y[test_idx] == label))
    init_labeled_index = np.where(Y[train_idx] == label)[0][0]
    label_idx.append(init_labeled_index)
    unlabel_idx.remove(init_labeled_index)

train_idx = [np.array(train_idx)]  # type: ignore
test_idx = [np.array(test_idx)]  # type: ignore
label_idx = [np.array(label_idx)]  # type: ignore
unlabel_idx = [np.array(unlabel_idx)]  # type: ignore


if "NN_BINARY_PATH" in strategy_id_mapping[STRATEGY_ID][1].keys():
    data_storage.X = X
    data_storage.true_Y = None  # type: ignore
    data_storage.human_expert_Y = None  # type: ignore
    data_storage.Y_merged_final = None  # type: ignore
    # @TODO: previous could be important later for some state encodings??!

    data_storage.labeled_mask = label_idx[0]  # type: ignore
    data_storage.unlabeled_mask = unlabel_idx[0]  # type: ignore
    data_storage.test_mask = test_idx[0]  # type: ignore

""" print("Unlabeled before the experiment are: " + str(unlabel_idx))
print("Labeled before are " + str(label_idx))
print("train_idx" + str(train_idx))
print("test_idx" + str(test_idx))


print("ds.unl ", data_storage.unlabeled_mask)
print("ds.lab ", data_storage.labeled_mask)
print("ds.tes ", data_storage.test_mask) """


# update data_storage!

QUERY_STRATEGY: Tuple = strategy_id_mapping[STRATEGY_ID]


print(QUERY_STRATEGY)
if "NN_BINARY_PATH" in strategy_id_mapping[STRATEGY_ID][1].keys():
    # update NN_BINARY_PATH
    if "NN_BINARY_PATH" in QUERY_STRATEGY[1].keys():
        QUERY_STRATEGY[1]["NN_BINARY_PATH"] = (
            config.OUTPUT_PATH + "/03_imital_trained_ann.model"
        )
        QUERY_STRATEGY[1]["data_storage"] = data_storage

    if config.EXCLUDING:
        excluding_args = {
            "EXCLUDING_STATE_DIFF_PROBAS": config.EXCLUDING_STATE_DIFF_PROBAS,
            "EXCLUDING_STATE_ARGFIRST_PROBAS": config.EXCLUDING_STATE_ARGFIRST_PROBAS,
            "EXCLUDING_STATE_ARGSECOND_PROBAS": config.EXCLUDING_STATE_ARGSECOND_PROBAS,
            "EXCLUDING_STATE_ARGTHIRD_PROBAS": config.EXCLUDING_STATE_ARGTHIRD_PROBAS,
            "EXCLUDING_STATE_DISTANCES_LAB": config.EXCLUDING_STATE_DISTANCES_LAB,
            "EXCLUDING_STATE_DISTANCES_UNLAB": config.EXCLUDING_STATE_DISTANCES_UNLAB,
            "EXCLUDING_STATE_PREDICTED_CLASS": config.EXCLUDING_STATE_PREDICTED_CLASS,
            "EXCLUDING_STATE_PREDICTED_UNITY": config.EXCLUDING_STATE_PREDICTED_UNITY,
            "EXCLUDING_STATE_DISTANCES": config.EXCLUDING_STATE_DISTANCES,
            "EXCLUDING_STATE_UNCERTAINTIES": config.EXCLUDING_STATE_UNCERTAINTIES,
            "EXCLUDING_STATE_INCLUDE_NR_FEATURES": config.EXCLUDING_STATE_INCLUDE_NR_FEATURES,
        }
    else:
        excluding_args = {}

    test = ALiPY_ImitAL_Query_Strategy(X=X, Y=Y, **excluding_args, **QUERY_STRATEGY[1])

if "NN_BINARY_PATH" in strategy_id_mapping[STRATEGY_ID][1].keys():
    properties = {**excluding_args, **QUERY_STRATEGY[1]}
else:
    properties = QUERY_STRATEGY[1]
# print("num_of_queries", dataset_id_mapping[DATASET_ID][1] / config.BATCH_SIZE)
# print("stopping ", dataset_id_mapping[DATASET_ID][1] / config.BATCH_SIZE)


al = AlExperiment(
    X,
    Y,
    #  model=MLPClassifier(),
    model=RandomForestClassifier(n_jobs=multiprocessing.cpu_count()),
    stopping_criteria="num_of_queries",
    num_of_queries=dataset_id_mapping[DATASET_ID][1] / config.BATCH_SIZE,
    stopping_value=dataset_id_mapping[DATASET_ID][1] / config.BATCH_SIZE,
    batch_size=config.BATCH_SIZE,
    train_idx=train_idx,
    test_idx=test_idx,
    label_idx=label_idx,
    unlabel_idx=unlabel_idx,
)
print("batch: ", str(config.BATCH_SIZE))

al.set_query_strategy(strategy=QUERY_STRATEGY[0], **properties)

#  al.set_performance_metric("accuracy_score")
al.set_performance_metric("f1_score")

start = timer()
al.start_query(multi_thread=False)
end = timer()
# print(end - start)
# trained_model = al._model

r = al.get_experiment_result()

stateio = r[0]
metric_values = []
if stateio.initial_point is not None:
    metric_values.append(stateio.initial_point)
for state in stateio:
    print(state.get_value("select_index"))
    metric_values.append(state.get_value("performance"))

print(metric_values)
f1_auc = auc([i for i in range(0, len(metric_values))], metric_values) / (
    len(metric_values) - 1
)
print(f1_auc)
print("fertig")

""" for i in range(0, int(dataset_id_mapping[DATASET_ID][1] / config.BATCH_SIZE)):
    print()
    print()
    print((str(i) + " ") * 10)

    train_idx, test_idx, label_idx, unlabel_idx = r[0].get_workspace(i)
    print("Unlabeled before the experiment are: " + str(unlabel_idx))
    print("Labeled before are " + str(label_idx))
    print("train_idx" + str(train_idx))
    print("test_idx" + str(test_idx))


print("ds.unl ", data_storage.unlabeled_mask)
print("ds.lab ", data_storage.labeled_mask)
print("ds.tes ", data_storage.test_mask)


exit(-1)
 """
stateio = r[0]
metric_values = []
if stateio.initial_point is not None:
    metric_values.append(stateio.initial_point)
for state in stateio:
    # print(state.get_value("select_index"))
    metric_values.append(state.get_value("performance"))
print(metric_values)
f1_auc = auc([i for i in range(0, len(metric_values))], metric_values) / (
    len(metric_values) - 1
)
print()
print(f1_auc)

res = {}
res["dataset_id"] = DATASET_ID
res["strategy_id"] = str(STRATEGY_ID)
res["dataset_random_seed"] = DATASET_RANDOM_SEED
res["strategy"] = str(QUERY_STRATEGY[0]) + str(QUERY_STRATEGY[1])
res["duration"] = end - start
res["f1_auc"] = f1_auc
res = {**res, **synthetic_creation_args}
print(res["f1_auc"])

with open(config.OUTPUT_PATH + "/05_alipy_results.csv", "a") as f:
    w = csv.DictWriter(f, fieldnames=res.keys())
    if len(open(config.OUTPUT_PATH + "/05_alipy_results.csv").readlines()) == 0:
        print("write header")
        w.writeheader()
    w.writerow(res)
