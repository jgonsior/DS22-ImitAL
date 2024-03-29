import argparse
import json
from xml.dom import XML_NAMESPACE
import numpy as np
import os
import pandas as pd
import random
import sys
import tensorflow as tf
import joblib

from sklearn.metrics import mean_squared_error

# Tfrom keras.layers import Dense, Dropout
# from keras.models import Sequential
from scikeras.wrappers import KerasRegressor, KerasClassifier
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Any, Dict, List
from sklearn.preprocessing import MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument("--OUTPUT_PATH")
parser.add_argument(
    "--CLASSIFIER",
    default="RF",
    help="Supported types: RF, DTree, NB, SVM, Linear",
)
parser.add_argument("--N_JOBS", type=int, default=-1)
parser.add_argument(
    "--RANDOM_SEED", type=int, default=42, help="-1 Enables true Randomness"
)
parser.add_argument("--TEST_FRACTION", type=float, default=0.5)
parser.add_argument("--LOG_FILE", type=str, default="log.txt")
parser.add_argument("--REGULAR_DROPOUT_RATE", type=float, default=0.2)
parser.add_argument("--NR_HIDDEN_NEURONS", type=int, default=1100)
parser.add_argument("--OPTIMIZER", type=str, default="Nadam")
parser.add_argument("--NR_HIDDEN_LAYERS", type=int, default=2)
parser.add_argument("--LOSS", type=str, default="MeanSquaredError")
parser.add_argument("--KERNEL_INITIALIZER", type=str, default="glorot_uniform")
parser.add_argument("--EPOCHS", type=int, default=10000)
parser.add_argument("--ANN_BATCH_SIZE", type=int, default=128)
parser.add_argument("--MAX_NUM_TRAINING_DATA", type=int, default=1000000000000000)
parser.add_argument("--ACTIVATION", type=str, default="elu")
parser.add_argument("--HYPER_SEARCH", action="store_true")
parser.add_argument("--BATCH_SIZE", type=int, default=5)
parser.add_argument("--MAX_NUMBER", type=int, default=5)
parser.add_argument("--N_ITER", type=int, default=5)
parser.add_argument(
    "--STATE_ENCODING",
    type=str,
    help="pointwise, pairwise, listwise",
    default="listwise",
)
parser.add_argument(
    "--TARGET_ENCODING", type=str, help="regression, binary", default="regression"
)
parser.add_argument("--SAVE_DESTINATION", type=str)
parser.add_argument("--PERMUTATE_NN_TRAINING_INPUT", type=int, default=0)
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
parser.add_argument("--None", action="store_true")

config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()

if config.RANDOM_SEED != -1 and config.RANDOM_SEED != -2:
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

DATA_PATH = config.OUTPUT_PATH
states = pd.read_csv(
    DATA_PATH + "/01_state_encodings_X.csv", nrows=config.MAX_NUM_TRAINING_DATA
)
print(states)


# remove columns which are not interesting according to state
# state in this case menas removing, for the other files it menas adding, don't ask why…

if config.EXCLUDING_STATE_DISTANCES_LAB:
    states = states.loc[:, ~states.columns.str.endswith("avg_dist_lab")]
if config.EXCLUDING_STATE_DISTANCES_UNLAB:
    states = states.loc[:, ~states.columns.str.endswith("avg_dist_unlab")]
if config.EXCLUDING_STATE_PREDICTED_CLASS:
    print("not implemented yet")
    exit(-1)
if config.EXCLUDING_STATE_PREDICTED_UNITY:
    print("not implemented yet")
    exit(-1)
if config.EXCLUDING_STATE_ARGFIRST_PROBAS:
    states = states.loc[:, ~states.columns.str.endswith("proba_argfirst")]
if config.EXCLUDING_STATE_ARGSECOND_PROBAS:
    states = states.loc[:, ~states.columns.str.endswith("proba_argsecond")]
if config.EXCLUDING_STATE_ARGTHIRD_PROBAS:
    states = states.loc[:, ~states.columns.str.endswith("proba_argthird")]
if config.EXCLUDING_STATE_DIFF_PROBAS:
    print("not implemented yet")
    exit(-1)
if config.EXCLUDING_STATE_DISTANCES:
    print("not implemented yet")
    exit(-1)
if config.EXCLUDING_STATE_UNCERTAINTIES:
    print("not implemented yet")
    exit(-1)
if config.EXCLUDING_STATE_INCLUDE_NR_FEATURES:
    print("not implemented yet")
    exit(-1)


optimal_policies = pd.read_csv(
    DATA_PATH + "/01_expert_actions_Y.csv", nrows=config.MAX_NUM_TRAINING_DATA
)

#  states = states[0:100]
#  optimal_policies = optimal_policies[0:100]

AMOUNT_OF_PEAKED_OBJECTS = len(optimal_policies.columns)


def _binarize_targets(df, TOP_N=5):
    df = df.assign(threshold=np.sort(df.values)[:, -TOP_N : -(TOP_N - 1)])
    for column_name in df.columns:
        if column_name == "threshold":
            continue
        df[column_name].loc[df[column_name] < df.threshold] = 0
        df[column_name].loc[df[column_name] >= df.threshold] = 1
    del df["threshold"]
    return df


def _evaluate_top_k(Y_true, Y_pred):
    # evaluate based on accuracy of correct top five
    if config.TARGET_ENCODING == "binary":
        Y_true_binarized = Y_true
    else:
        Y_true_binarized = _binarize_targets(Y_true)

    Y_pred = pd.DataFrame(data=Y_pred, columns=Y_true.columns)

    Y_pred_binarized = _binarize_targets(Y_pred)

    accs: List[float] = []
    for i in range(0, AMOUNT_OF_PEAKED_OBJECTS):
        accs.append(
            accuracy_score(
                Y_true_binarized[str(i) + "_true_peaked_normalised_acc"].to_numpy(),
                Y_pred_binarized[str(i) + "_true_peaked_normalised_acc"].to_numpy(),
            )
        )
    return np.mean(np.array(accs))


if config.TARGET_ENCODING == "regression":
    wrapper = KerasRegressor
    # congrats, default values
    pass
elif config.TARGET_ENCODING == "binary":
    wrapper = KerasClassifier
    optimal_policies = _binarize_targets(optimal_policies)
else:
    print("Not a valid TARGET_ENCODING")
    exit(-1)

if config.STATE_ENCODING == "listwise":
    # congrats, the states are already in the correct form
    pass
elif config.STATE_ENCODING == "pointwise":
    print("Not yet implemented")
    states.to_csv(DATA_PATH + "/states_pointwise.csv", index=False)
    exit(-1)
elif config.STATE_ENCODING == "pairwise":
    print("Not yet implemented")
    states.to_csv(DATA_PATH + "/states_pairwise.csv", index=False)
    exit(-1)
else:
    print("Not a valid STATE_ENCODING")
    exit(-1)


X = states
Y = optimal_policies

# normalize states
# scaler = MinMaxScaler()
# scaler.fit(X)
# X = scaler.transform(X)


def _permutate_ndarray(X: np.ndarray, permutation: List[int]) -> np.ndarray:
    # use fancy indexing: https://stackoverflow.com/questions/20265229/rearrange-columns-of-numpy-2d-array
    new_idx = np.empty_like(permutation)
    new_idx[permutation] = np.arange(len(permutation))
    return X[:, new_idx]


""" new_X = X
new_Y = Y

for permutate_index in range(1, config.PERMUTATE_NN_TRAINING_INPUT + 1):
    SAMPLING_SIZE = Y.shape[1]

    # random rearrangement
    random_permutation = [x for x in range(0, SAMPLING_SIZE)]
    random.shuffle(random_permutation)

    Y2 = _permutate_ndarray(Y.to_numpy(), random_permutation)

    amount_of_states = int(np.shape(X)[1] / len(random_permutation))  # type: ignore
    X_permutation = random_permutation.copy()

    for i in range(1, amount_of_states):
        X_permutation += [rp + i * len(random_permutation) for rp in random_permutation]
    X2 = _permutate_ndarray(X, X_permutation)  # type: ignore

    new_Y = np.concatenate((new_Y, Y2))  # type: ignore
    new_X = np.concatenate((new_X, X2))  # type: ignore

# print(np.shape(new_X))  # type: ignore
# print(np.shape(new_Y))  # type: ignore

X = new_X
Y = new_Y
 """
# from sklearn.utils.multiclass import type_of_target

# print(Y)
# print(type_of_target(Y))


def tau_loss(Y_true, Y_pred):
    return tf.py_function(
        kendalltau,
        [tf.cast(Y_pred, tf.float32), tf.cast(Y_true, tf.float32)],
        Tout=tf.float32,
    )


def spearman_loss(Y_true, Y_pred):
    return tf.py_function(
        spearmanr,
        [tf.cast(Y_pred, tf.float32), tf.cast(Y_true, tf.float32)],
        Tout=tf.float32,
    )


def get_reg(
    input_size,
    output_size,
    nr_hidden_layers,
    nr_hidden_neurons,
    optimizer,
    activation,
    regular_dropout_rate,
    kernel_initializer,
):
    model = keras.Sequential()
    inp = keras.layers.Input(shape=input_size)
    model.add(inp)
    for _ in range(0, nr_hidden_layers):
        layer = keras.layers.Dense(
            nr_hidden_neurons,
            activation=activation,
            kernel_initializer=kernel_initializer,
        )
        model.add(layer)
        model.add(keras.layers.Dropout(regular_dropout_rate))

    out = keras.layers.Dense(output_size, activation="sigmoid")
    model.add(out)
    model.compile(loss="mse", optimizer=optimizer)
    return model


if config.HYPER_SEARCH:
    param_grid = {
        "loss": [
            "MeanSquaredError",
            #  "CategoricalCrossentropy",
            #  "BinaryCrossentropy",
            #  "CosineSimilarity",
            # spearman_loss,
            # tau_loss,
        ],
        "model__regular_dropout_rate": [0, 0.1, 0.2, 0.3],
        #  "recurrentDropoutRate": [0, 0.1, 0.2],
        "model__nr_hidden_neurons": [
            #  1,
            #  2,
            #  3,
            #  4,
            #  6,
            #  8,
            #  16,
            #  24,
            #  32,
            #  60,
            #  90,
            #  120,
            #  150,
            #  180,
            #  210,
            #  240,
            #  270,
            #  300
            100,
            #  200,
            300,
            #  400,
            500,
            #  600,
            700,
            #  800,
            900,
            #  1000,
            1100,
            1300,
            1500,
        ],  # [160, 240, 480, 720, 960],
        "epochs": [10000],  # <- early stopping :)
        "model__nr_hidden_layers": [2, 3, 4, 8],  # 16, 32, 64, 96, 128],  # , 2],
        "batch_size": [16, 32, 64, 128],
        #  "nTs": [15000],
        #  "k2": [1000],
        #  "diff": [False],
        "model__optimizer": ["RMSprop", "Adam", "Nadam"],
        "model__kernel_initializer": [
            "glorot_uniform",
            # "VarianceScaling",
            "lecun_uniform",
            "he_normal",
            # "he_uniform",
        ],
        "verbose": [1],
        #  "activation": ["softmax", "elu", "relu", "tanh", "sigmoid"],
        "model__activation": ["elu", "relu", "tanh"],
    }

    model = KerasRegressor(
        model=get_reg,
        input_size=X.shape[1:],
        output_size=np.shape(Y)[1],
        verbose=2,
        validation_split=0.3,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", patience=3, verbose=1),
        ],
    )

    gridsearch = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        cv=config.N_ITER,
        n_jobs=2,
        scoring=make_scorer(_evaluate_top_k, greater_is_better=True),
        verbose=1,
        n_iter=config.N_ITER,
    )
    #  X = X.to_numpy()
    #  Y = Y.to_numpy()
    #  print(X)
    #  print(Y)
    #  print(np.shape(X))
    #  print(np.shape(Y))

    fitted_model = gridsearch.fit(X, Y)

    fitted_model.best_estimator_.model_.save(
        config.OUTPUT_PATH + "/02_best_model.model"
    )

    # joblib.dump(scaler, config.OUTPUT_PATH + "/scaler.gz")

    with open(config.OUTPUT_PATH + "/params.json", "w") as f:
        json.dump({"TARGET_ENCODING": config.TARGET_ENCODING}, f)

    with open(config.OUTPUT_PATH + "/02_hyper_results.txt", "w") as handle:
        handle.write(str(fitted_model.best_score_))
        handle.write(str(fitted_model.cv_results_))
        handle.write("\n" * 5)
        handle.write(json.dumps(fitted_model.best_params_))


else:
    reg = KerasRegressor(
        model=get_reg,
        input_size=X.shape[1:],
        output_size=np.shape(Y)[1],
        verbose=1,
        model__activation=config.ACTIVATION,
        validation_split=0.3,
        loss=config.LOSS,
        model__regular_dropout_rate=config.REGULAR_DROPOUT_RATE,
        model__nr_hidden_layers=config.NR_HIDDEN_LAYERS,
        model__nr_hidden_neurons=config.NR_HIDDEN_NEURONS,
        kernel_initializer=config.KERNEL_INITIALIZER,
        model__optimizer=config.OPTIMIZER,
        epochs=config.EPOCHS,
        batch_size=config.ANN_BATCH_SIZE,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", patience=3, verbose=0),
        ],
        # random_state=1,
    )

    print("shape X")
    print(np.shape(X))
    print(X)
    print(Y)
    # exit(-1)
    # exit(-1)

    reg.fit(X, Y)

    print(reg.get_params())

    Y_pred = reg.predict(X)

    print("YYYYYYY")
    print(Y)
    print(Y_pred)
    print(mean_squared_error(Y, Y_pred))
    print(reg.score(X, Y))
    print(reg.model_.summary())

    if config.SAVE_DESTINATION:
        reg.model_.save(config.SAVE_DESTINATION)
        # joblib.dump(scaler, config.SAVE_DESTINATION + "/scaler.gz")

        with open(config.SAVE_DESTINATION + "/params.json", "w") as f:
            json.dump({"TARGET_ENCODING": config.TARGET_ENCODING}, f)
