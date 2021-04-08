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

from active_learning.dataStorage import DataStorage
from active_learning.datasets.dwtc import load_dwtc
from active_learning.datasets.synthetic import load_synthetic
from active_learning.datasets.uci import load_uci
from active_learning.logger.logger import init_logger
from imitLearningPipelineSharedCode import  dataset_id_mapping, strategy_id_mapping


init_logger("tmp.log")

parser = argparse.ArgumentParser()
parser.add_argument("--DATASETS_PATH", default="../datasets")
parser.add_argument("--N_JOBS", type=int, default=1)
parser.add_argument(
    "--INDEX", type=int, default=1, help="Specifies which dataset to use etc."
)
parser.add_argument("--OUTPUT_PATH", default="../datasets/ali")
parser.add_argument("--RANDOM_SEEDS_INPUT_FILE")
parser.add_argument(("--BATCH_SIZE", type=int, default=5)

config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()


# first open job file to get the real random_seed from it
random_seed_df = pd.read_csv(
    config.RANDOM_SEEDS_INPUT_FILE,
    header=0,
    index_col=0,
    nrows=config.INDEX + 1,
)
DATASET_ID, STRATEGY_ID, DATASET_RANDOM_SEED = random_seed_df.loc[config.INDEX]


np.random.seed(DATASET_RANDOM_SEED)
random.seed(DATASET_RANDOM_SEED)
DATASET_NAME = dataset_id_mapping[DATASET_ID][0]

# specify which dataset to load
print("dataset: ", DATASET_NAME)
print("random_seed: ", DATASET_RANDOM_SEED)

if DATASET_NAME == "synthetic":
    df, synthetic_creation_args = load_synthetic(
        DATASET_RANDOM_SEED,
        NEW_SYNTHETIC_PARAMS=True,
        VARIABLE_DATASET=True,
        AMOUNT_OF_FEATURES=20,
        HYPERCUBE=True,
        GENERATE_NOISE=True,
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

data_storage = DataStorage(df, TEST_FRACTION=0)
X = data_storage.X
Y = data_storage.exp_Y

test = ALiPYImitALSingle(
    X=X,
    Y=Y,
    NN_BINARY_PATH=config.OUTPUT_PATH + "/03_imital_trained_ann.model",
    data_storage=data_storage,
)


shuffling = np.random.permutation(len(Y))
X = X[shuffling]
Y = Y[shuffling]

scaler = RobustScaler()
X = scaler.fit_transform(X)

# scale back to [0,1]
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

test_ratio = 0.5
indices = [i for i in range(0, len(Y))]
train_idx = indices[: math.floor(len(Y) * (1 - test_ratio))]
test_idx = indices[math.floor(len(Y) * (1 - test_ratio)) :]
unlabel_idx = train_idx.copy()
label_idx = []
#  print(Y)
#  print(y[train_idx])
for label in np.unique(Y):
    if label not in y[train_idx]:
        print(np.where(y[test_idx] == label))
    init_labeled_index = np.where(y[train_idx] == label)[0][0]
    label_idx.append(init_labeled_index)
    unlabel_idx.remove(init_labeled_index)

train_idx = [np.array(train_idx)]
test_idx = [np.array(test_idx)]
label_idx = [np.array(label_idx)]
unlabel_idx = [np.array(unlabel_idx)]


def run_parallel(query_strategy):
    print(query_strategy)

    al = AlExperiment(
        X,
        y,
        #  model=MLPClassifier(),
        model=RandomForestClassifier(n_jobs=multiprocessing.cpu_count()),
        stopping_criteria="num_of_queries",
        num_of_queries=dataset_load_function[2],
        stopping_value=dataset_load_function[2],
        batch_size=BATCH_SIZE,
        train_idx=train_idx,
        test_idx=test_idx,
        label_idx=label_idx,
        unlabel_idx=unlabel_idx,
    )

    al.set_query_strategy(
        strategy=query_strategy[0], **query_strategy[1]
    )  # , measure="least_confident")

    #  al.set_performance_metric("accuracy_score")
    al.set_performance_metric("f1_score")

    start = timer()
    al.start_query(multi_thread=False)
    end = timer()

    trained_model = al._model

    r = al.get_experiment_result()

    stateio = r[0]
    metric_values = []
    if stateio.initial_point is not None:
        metric_values.append(stateio.initial_point)
    for state in stateio:
        metric_values.append(state.get_value("performance"))
    f1_auc = auc([i for i in range(0, len(metric_values))], metric_values) / (
        len(metric_values) - 1
    )
    print(f1_auc)
    for r2 in r:
        res = r2.get_result()
        res["dataset_id"] = DATASET_ID
        res["strategy_id"] = str(STRATEGY_ID)
        res["dataset_random_seed"] = DATASET_RANDOM_SEED
        res["strategy"] = str(query_strategy[0]) + str(query_strategy[1])
        res["duration"] = end - start
        res["f1_auc"] = f1_auc
        res = {**res, **synthetic_creation_args}
        with open(config.OUTPUT_PATH + "/result.csv", "a") as f:
            w = csv.DictWriter(f, fieldnames=res.keys())
            if len(open(config.OUTPUT_PATH + "/result.csv").readlines()) == 0:
                print("write header")
                w.writeheader()
            w.writerow(res)


run_parallel(query_strategies[STRATEGY_ID])
#  for query_strategy in query_strategies:
#  run_parallel((query_strategy))

# with Parallel(n_jobs=config.N_JOBS, backend="threading") as parallel:
#    output = parallel(
#        delayed(run_parallel)(query_strategy) for query_strategy in query_strategies
#    )
