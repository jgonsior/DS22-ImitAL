import sys
import argparse
from sklearn.utils.multiclass import type_of_target
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed, parallel_backend
import random
from itertools import chain
import copy
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scikeras.wrappers import KerasClassifier, KerasRegressor
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from keras.optimizers import Adam, Nadam, RMSprop, Adadelta
from keras.initializers import VarianceScaling, lecun_uniform, he_normal, he_uniform
from keras.activations import softmax, elu, relu, tanh, sigmoid
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    LearningRateScheduler,
)

parser = argparse.ArgumentParser()
parser.add_argument("--DATA_PATH", default="../datasets/")
parser.add_argument(
    "--CLASSIFIER", default="RF", help="Supported types: RF, DTree, NB, SVM, Linear",
)
parser.add_argument("--N_JOBS", type=int, default=-1)
parser.add_argument(
    "--RANDOM_SEED", type=int, default=42, help="-1 Enables true Randomness"
)
parser.add_argument("--TEST_FRACTION", type=float, default=0.5)
parser.add_argument("--LOG_FILE", type=str, default="log.txt")
parser.add_argument("--REGULAR_DROPOUT_RATE", type=float, default=0.2)
parser.add_argument("--NR_HIDDEN_NEURONS", type=int, default=40)
parser.add_argument("--OPTIMIZER", type=str, default="Nadam")
parser.add_argument("--NR_HIDDEN_LAYERS", type=int, default=1)
parser.add_argument("--LOSS", type=str, default="MeanSquaredError")
parser.add_argument("--EPOCHS", type=int, default=1000)
parser.add_argument("--BATCH_SIZE", type=int, default=16)
parser.add_argument("--ACTIVATION", type=str, default="elu")
parser.add_argument("--HYPER_SEARCH", action="store_true")
parser.add_argument(
    "--STATE_ENCODING",
    type=str,
    help="pointwise, pairwise, listwise",
    default="listwise",
)
parser.add_argument(
    "--TARGET_ENCODING", type=str, help="regression, binary", default="regression"
)
config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()

if config.RANDOM_SEED != -1 and config.RANDOM_SEED != -2:
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

AMOUNT_OF_PEAKED_OBJECTS = 20
DATA_PATH = config.DATA_PATH


def _binarize_targets(df, TOP_N=5):
    df = df.assign(threshold=np.sort(df.values)[:, -TOP_N : -(TOP_N - 1)])

    print(df)

    for column_name in df.columns:
        if column_name == "threshold":
            continue
        df[column_name].loc[df[column_name] < df.threshold] = 0
        df[column_name].loc[df[column_name] >= df.threshold] = 1
    del df["threshold"]
    return df


states = pd.read_csv(DATA_PATH + "/states.csv")
optimal_policies = pd.read_csv(DATA_PATH + "/opt_pol.csv")

print(config.TARGET_ENCODING)

if config.TARGET_ENCODING == "regression":
    # congrats, default values
    pass
elif config.TARGET_ENCODING == "binary":
    if os.path.isfile(DATA_PATH + "/opt_pol_binary.csv"):
        optimal_policies = pd.read_csv(DATA_PATH + "/opt_pol_binary.csv")
    else:
        regression_encoding = _binarize_targets(optimal_policies)
        #  print(optimal_policies)
        #  # convert top five to binary 1 and the rest to binary 0
        #  optimal_policies = pd.DataFrame(columns=regression_encoding.columns)
        #
        #  for index, row in regression_encoding.iterrows():
        #      threshold = min(row.nlargest(5))
        #
        #      zeros = row < threshold
        #      ones = row >= threshold
        #
        #      row.loc[zeros] = 0
        #      row.loc[ones] = 1
        #
        #      optimal_policies.loc[index] = row
        #  print(optimal_policies)
        #  optimal_policies.to_csv(DATA_PATH + "/opt_pol_binary.csv", index=False)
        exit(-1)
else:
    print("Not a valid TARGET_ENCODING")
    exit(-1)

print(config.STATE_ENCODING)
if config.STATE_ENCODING == "listwise":
    # congrats, the states are already in the correct form
    pass
elif config.STATE_ENCODING == "pointwise":
    print(states)
    #  states_new =
    states.to_csv(DATA_PATH + "/states_pointwise.csv", index=False)
    exit(-1)
elif config.STATE_ENCODING == "pairwise":
    print("Not yet implemented")
    states.to_csv(DATA_PATH + "/states_pairwise.csv", index=False)
    exit(-1)
else:
    print("Not a valid STATE_ENCODING")
    exit(-1)

print(states)
print(optimal_policies)

# test out pointwise LTR method
# test out pairwise LTR method, RankNet, RankSVM, RankBoost
# test out super complex listwise LTR method (LambdaMART, ApproxNDCG, List{NET, MLE}


# train/test/val split, verschiedene encodings und hyperparameterkombis für das ANN ausprobieren


X_train, X_test, Y_train, Y_test = train_test_split(
    states, optimal_policies, test_size=0.33
)


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


def build_nn(
    X,
    activation="relu",
    regular_dropout_rate=0.2,
    optimizer="Adam",
    nr_hidden_layers=3,
    nr_hidden_neurons=20,
    kernel_initializer="glorot_uniform",
    loss="MeanSquaredError",
    epochs=1,
    batch_size=1,
):
    model = Sequential()

    model.add(Dense(units=X.shape[1], input_shape=X.shape[1:], activation=activation,))

    for _ in range(0, nr_hidden_layers):
        model.add(
            Dense(
                units=nr_hidden_neurons,
                activation=activation,
                kernel_initializer=kernel_initializer,
            )
        )
        model.add(Dropout(regular_dropout_rate))
    model.add(Dense(len(Y_train.columns), activation="sigmoid"))

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=["MeanSquaredError", "accuracy"],  # , tau_loss, spearman_loss],
    )
    print(model.summary())

    return model


if config.HYPER_SEARCH:
    param_grid = {
        "loss": ["MeanSquaredError", spearman_loss, tau_loss],
        "regular_dropout_rate": [0, 0.1, 0.2, 0.3],
        #  "recurrentDropoutRate": [0, 0.1, 0.2],
        "nr_hidden_neurons": [10, 20, 40, 80],
        "epochs": [1000],  # <- early stopping :)
        "nr_hidden_layers": [1, 2, 4, 8, 16, 32, 64],  # , 2],
        "batch_size": [16, 32, 64, 128],
        #  "nTs": [15000],
        #  "k2": [1000],
        #  "diff": [False],
        "optimizer": ["sgd", "RMSprop", "Adadelta", "Adam", "Nadam"],
        #  "kernel_initializer": [
        #      "glorot_uniform",
        #      "VarianceScaling",
        #      "lecun_uniform",
        #      "he_normal",
        #      "he_uniform",
        #  ],
        "activation": ["softmax", "elu", "relu", "tanh", "sigmoid"],
    }

    if config.TARGET_ENCODING == "binary":
        model = KerasClassifier(
            build_fn=build_nn,
            verbose=0,
            activation=None,
            validation_split=0.3,
            loss=None,
            regular_dropout_rate=None,
            nr_hidden_layers=None,
            nr_epochs=None,
            nr_hidden_neurons=None,
            optimizer=None,
            epochs=None,
            batch_size=None,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=5, verbose=1),
                ReduceLROnPlateau(monitor="val_loss", patience=3, verbose=1),
            ],
        )
    else:
        model = KerasRegressor(
            build_fn=build_nn,
            verbose=0,
            activation=None,
            validation_split=0.3,
            loss=None,
            regular_dropout_rate=None,
            nr_hidden_layers=None,
            nr_epochs=None,
            nr_hidden_neurons=None,
            optimizer=None,
            epochs=None,
            batch_size=None,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=5, verbose=1),
                ReduceLROnPlateau(monitor="val_loss", patience=3, verbose=1),
            ],
        )

    gridsearch = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        cv=3,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
        verbose=2,
        n_iter=100,
    )

    fitted_model = gridsearch.fit(X_train, Y_train)
    Y_pred = fitted_model.predict(X_test)

    print(fitted_model.best_score_)
    print(fitted_model.best_params_)
    print(fitted_model.cv_results_)
else:
    if config.TARGET_ENCODING == "binary":
        model = KerasClassifier(
            build_fn=build_nn,
            verbose=1,
            activation=config.ACTIVATION,
            validation_split=0.3,
            loss=config.LOSS,
            regular_dropout_rate=config.REGULAR_DROPOUT_RATE,
            nr_hidden_layers=config.NR_HIDDEN_LAYERS,
            nr_hidden_neurons=config.NR_HIDDEN_NEURONS,
            optimizer=config.OPTIMIZER,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=5, verbose=1),
                ReduceLROnPlateau(monitor="val_loss", patience=3, verbose=1),
            ],
        )
    else:
        model = KerasRegressor(
            build_fn=build_nn,
            verbose=1,
            activation=config.ACTIVATION,
            validation_split=0.3,
            loss=config.LOSS,
            regular_dropout_rate=config.REGULAR_DROPOUT_RATE,
            nr_hidden_layers=config.NR_HIDDEN_LAYERS,
            nr_hidden_neurons=config.NR_HIDDEN_NEURONS,
            optimizer=config.OPTIMIZER,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=5, verbose=1),
                ReduceLROnPlateau(monitor="val_loss", patience=3, verbose=1),
            ],
        )

    fitted_model = model.fit(
        X=X_train, y=Y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
    )

    Y_pred = fitted_model.predict(X_test)

    print(Y_test)
    print(Y_pred)
    print(mean_squared_error(Y_test, Y_pred))

    # evaluate based on accuracy of correct top five
    Y_test_binarized = _binarize_targets(Y_test)

    Y_pred = pd.DataFrame(data=Y_pred, index=Y_test.index, columns=Y_test.columns)

    Y_pred_binarized = _binarize_targets(Y_pred)

    accs = []
    for i in range(0, 20):
        accs.append(
            accuracy_score(
                Y_test_binarized[str(i) + "_true_peaked_normalised_acc"].to_numpy(),
                Y_pred_binarized[str(i) + "_true_peaked_normalised_acc"].to_numpy(),
            )
        )
    print(accs)
    print(np.mean(accs))
    history = fitted_model.history_
    print(history.history)

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"], loc="upper left")
    #  plt.show()
