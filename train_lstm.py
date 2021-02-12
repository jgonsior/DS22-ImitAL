import argparse
import json
import os
import random
import sys

import dill
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import parallel_backend
from keras.layers import Dense, Dropout
from keras.models import Sequential
from scikeras.wrappers import KerasRegressor
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

parser = argparse.ArgumentParser()
parser.add_argument("--DATA_PATH", default="../datasets/")
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
parser.add_argument("--REGULAR_DROPOUT_RATE", type=float, default=0.3)
parser.add_argument("--NR_HIDDEN_NEURONS", type=int, default=40)
parser.add_argument("--OPTIMIZER", type=str, default="RMSprop")
parser.add_argument("--NR_HIDDEN_LAYERS", type=int, default=4)
parser.add_argument("--LOSS", type=str, default="MeanSquaredError")
parser.add_argument("--EPOCHS", type=int, default=1000)
parser.add_argument("--BATCH_SIZE", type=int, default=32)
parser.add_argument("--N_ITER", type=int, default=100)
parser.add_argument("--ACTIVATION", type=str, default="elu")
parser.add_argument("--HYPER_SEARCH", action="store_true")
parser.add_argument("--NR_QUERIES_PER_ITERATION", type=int, default=5)
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
config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()

if config.RANDOM_SEED != -1 and config.RANDOM_SEED != -2:
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

DATA_PATH = config.DATA_PATH
states = pd.read_csv(DATA_PATH + "/states.csv")
optimal_policies = pd.read_csv(DATA_PATH + "/opt_pol.csv")

#  states = states[0:100]
#  optimal_policies = optimal_policies[0:100]

#  states = states[states.columns.drop(list(states.filter(regex="avg_dist")))]
#  states = states[states.columns.drop(list(states.filter(regex="diff")))]
#  print(states)
#  exit(-1)

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

    accs = []
    for i in range(0, AMOUNT_OF_PEAKED_OBJECTS):
        accs.append(
            accuracy_score(
                Y_true_binarized[str(i) + "_true_peaked_normalised_acc"].to_numpy(),
                Y_pred_binarized[str(i) + "_true_peaked_normalised_acc"].to_numpy(),
            )
        )
    #  print(accs)
    return np.mean(accs)


#  print(config.TARGET_ENCODING)

if config.TARGET_ENCODING == "regression":
    # congrats, default values
    pass
elif config.TARGET_ENCODING == "binary":
    optimal_policies = _binarize_targets(optimal_policies)
else:
    print("Not a valid TARGET_ENCODING")
    exit(-1)

#  print(config.STATE_ENCODING)
if config.STATE_ENCODING == "listwise":
    # congrats, the states are already in the correct form
    pass
elif config.STATE_ENCODING == "pointwise":
    #  print(states)
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

#  print(states)
#  print(optimal_policies)
#  exit(-1)
# test out pointwise LTR method
# test out pairwise LTR method, RankNet, RankSVM, RankBoost
# test out super complex listwise LTR method (LambdaMART, ApproxNDCG, List{NET, MLE}


# train/test/val split, verschiedene encodings und hyperparameterkombis für das ANN ausprobieren


# normalise batch stuff
# uncertainty:  -5 bis 0
# distance:     1 bis 24?!

#  X_train, X_test, Y_train, Y_test = train_test_split(
#      states, optimal_policies, test_size=0.33
#  )
X = states
Y = optimal_policies


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


def get_clf(
    input_size,
    output_size,
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
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_size))
    for _ in range(0, nr_hidden_layers):
        model.add(
            keras.layers.Dense(
                nr_hidden_neurons,
                activation=activation,
                kernel_initializer=kernel_initializer,
            )
        )
        model.add(keras.layers.Dropout(regular_dropout_rate))
    model.add(keras.layers.Dense(output_size, activation="sigmoid"))
    return model


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

    model.add(
        Dense(
            units=X.shape[1],
            input_shape=X.shape[1:],
            activation=activation,
        )
    )

    for _ in range(0, nr_hidden_layers):
        model.add(
            Dense(
                units=nr_hidden_neurons,
                activation=activation,
                kernel_initializer=kernel_initializer,
            )
        )
        model.add(Dropout(regular_dropout_rate))
    model.add(Dense(len(Y.columns), activation="sigmoid"))

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=["MeanSquaredError", "accuracy"],  # , tau_loss, spearman_loss],
    )

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
        "regular_dropout_rate": [0, 0.1, 0.2, 0.3],
        #  "recurrentDropoutRate": [0, 0.1, 0.2],
        "nr_hidden_neurons": [
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
        "nr_hidden_layers": [2, 3, 4, 8],  # 16, 32, 64, 96, 128],  # , 2],
        "batch_size": [16, 32, 64, 128],
        #  "nTs": [15000],
        #  "k2": [1000],
        #  "diff": [False],
        "optimizer": ["RMSprop", "Adam", "Nadam"],
        #  "kernel_initializer": [
        #      "glorot_uniform",
        #      "VarianceScaling",
        #      "lecun_uniform",
        #      "he_normal",
        #      "he_uniform",
        #  ],
        #  "activation": ["softmax", "elu", "relu", "tanh", "sigmoid"],
        "activation": ["elu", "relu", "tanh"],
    }

    model = KerasRegressor(
        model=get_clf,
        input_size=X.shape[1:],
        output_size=len(Y.columns),
        verbose=2,
        activation=None,
        validation_split=0.3,
        loss=None,
        regular_dropout_rate=None,
        nr_hidden_layers=None,
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
        cv=5,
        n_jobs=-1,
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

    #  with parallel_backend("threading", n_jobs=len(os.sched_getaffinity(0))):
    fitted_model = gridsearch.fit(X, Y)
    #  Y_pred = fitted_model.predict(X_test)
    #
    #  print(fitted_model.best_score_)
    #  print(fitted_model.best_params_)
    #  print(fitted_model.cv_results_)
    #
    with open(config.DATA_PATH + "/best_model.pickle", "wb") as handle:
        dill.dump(fitted_model, handle)

    with open(config.DATA_PATH + "/hyper_results.txt", "w") as handle:
        handle.write(str(fitted_model.best_score_))
        handle.write(str(fitted_model.cv_results_))
        handle.write("\n" * 5)
        handle.write(json.dumps(fitted_model.best_params_))


else:
    model = KerasRegressor(
        model=get_clf,
        input_size=X.shape[1:],
        output_size=len(Y.columns),
        verbose=0,
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
            EarlyStopping(monitor="val_loss", patience=5, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", patience=3, verbose=0),
        ],
        #  random_state=config.RANDOM_SEED,
    )

    fitted_model = model.fit(
        X=X,
        y=Y,
    )

    #  Y_pred = fitted_model.predict(X_test)

    #  print(Y_test)
    #  print(Y_pred)
    #  print(mean_squared_error(Y_test, Y_pred))

    def random_sampling_probas(X_test):
        result = np.random.uniform(
            low=0, high=1, size=(len(X_test), AMOUNT_OF_PEAKED_OBJECTS)
        )
        return result

    def random_sampling(X_test):
        result = np.array(
            list(
                ([1] * 5 + [0] * (AMOUNT_OF_PEAKED_OBJECTS - 5))
                for _ in range(0, len(X_test))
            )
        )
        list(map(np.random.shuffle, result))
        return result

    def uncertainty_sampling(X_test, strategy="least_confident"):
        df = X_test.copy()
        if strategy == "least_confident":
            df = df.loc[:, ~df.columns.str.endswith("_proba_diff")]
        elif strategy == "max_margin":
            for i in df.columns:
                if i.endswith("diff"):
                    continue
                df[i[0 : i.find("_")]] = df[i] - df[i[:-3] + "diff"]
            df = df.loc[:, ~df.columns.str.contains("_")]
        # for entropy we are lacking the other classes
        #  elif strategy == "entropy":
        #      result = np.apply_along_axis(entropy, 1, Y_temp_proba)

        return _binarize_targets(df).to_numpy()

    #  Y_pred_random_probas = random_sampling_probas(X_test)
    #  Y_pred_random = random_sampling(X_test)
    #  Y_pred_uncertainty_lc = uncertainty_sampling(X_test, strategy="least_confident")
    #  Y_pred_uncertainty_mm = uncertainty_sampling(X_test, strategy="max_margin")
    #  Y_pred_uncertainty_ent = uncertainty_sampling(X_test, strategy="entropy")

    #  print(
    #      "Pleas note: uncertainty measures in real AL have access to ALL data from all samples, not just only from a small subset of only 20 samples!"
    #  )
    #  print("Y_pred:\t\t", _evaluate_top_k(Y_test, Y_pred))
    #  print("Y_pred_random:\t", _evaluate_top_k(Y_test, Y_pred_random))
    #  print("Y_pred_probas:\t", _evaluate_top_k(Y_test, Y_pred_random_probas))
    #  print("Y_pred_unc_lc:\t", _evaluate_top_k(Y_test, Y_pred_uncertainty_lc))
    #  print("Y_pred_unc_mm:\t", _evaluate_top_k(Y_test, Y_pred_uncertainty_mm))
    #  print("Y_pred_unc_ent:\t", _evaluate_top_k(Y_test, Y_pred_uncertainty_ent))

    #  history = fitted_model.history_
    #  print(history.history)
    #
    #  plt.plot(history.history["loss"])
    #  plt.plot(history.history["val_loss"])
    #  plt.title("Model loss")
    #  plt.ylabel("Loss")
    #  plt.xlabel("Epoch")
    #  plt.legend(["Train", "Test"], loc="upper left")
    #  plt.show()

    if config.SAVE_DESTINATION:
        with open(config.SAVE_DESTINATION, "wb") as handle:
            dill.dump(fitted_model, handle)

        with open(config.SAVE_DESTINATION, "rb") as handle:
            new_model = dill.load(handle)

            #  Y_pred2 = new_model.predict(X_test)
            #  print("Y_pred_random:\t", _evaluate_top_k(Y_test, Y_pred2))
