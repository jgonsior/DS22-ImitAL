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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

amount_of_peaked_objects = 20
DATA_PATH = "training_data"

model = keras.Sequential(
    [
        layers.Dense(
            2 * amount_of_peaked_objects,
            input_shape=(2 * amount_of_peaked_objects,),
            activation="relu",
        ),
        *[
            layers.Dense(4 * amount_of_peaked_objects, activation="relu")
            for _ in range(1, 5)
        ],
        layers.Dense(4 * amount_of_peaked_objects, activation="relu"),
        layers.Dense(
            amount_of_peaked_objects, activation="softmax"
        ),  # muss das softmax sein?!
    ]
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


model.compile(
    loss=spearman_loss,
    optimizer="adam",
    metrics=["MeanSquaredError", "accuracy", tau_loss],
)

#  print(model.summary())

# check if states and optimal policies file got provided or if we need to create a new one
states = pd.read_csv(DATA_PATH + "/states.csv")
optimal_policies = pd.read_csv(DATA_PATH + "/opt_pol.csv")
print(states)
print(optimal_policies)

# test out pointwise LTR method
# test out pairwise LTR method, RankNet, RankSVM, RankBoost
# test out super complex listwise LTR method (LambdaMART, ApproxNDCG, List{NET, MLE}


model.fit(
    x=states,
    y=optimal_policies,
    use_multiprocessing=True,
    #  epochs=1,
    epochs=10,
    batch_size=128,
)


def save_nn_training_data(DATA_PATH):
    states.to_csv(DATA_PATH + "/states.csv", index=False)
    optimal_policies.to_csv(DATA_PATH + "/opt_pol.csv", index=False)
