import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed, parallel_backend
import random
from itertools import chain
import copy
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import accuracy_score
from ..activeLearner import ActiveLearner
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


def _future_peak(
    unlabeled_sample_indice,
    weak_supervision_label_sources,
    data_storage,
    clf,
    MAX_AMOUNT_OF_WS_PEAKS,
):
    copy_of_data_storage = copy.deepcopy(data_storage)
    copy_of_classifier = copy.deepcopy(clf)

    copy_of_data_storage.label_samples(
        pd.Index([unlabeled_sample_indice]),
        [copy_of_data_storage.train_unlabeled_Y.loc[unlabeled_sample_indice]["label"]],
        "P",
    )
    copy_of_classifier.fit(
        copy_of_data_storage.train_labeled_X,
        copy_of_data_storage.train_labeled_Y["label"].to_list(),
    )
    for labelSource in weak_supervision_label_sources:
        labelSource.data_storage = copy_of_data_storage

    # what would happen if we apply WS after this one?
    for i in range(0, MAX_AMOUNT_OF_WS_PEAKS):
        for labelSource in weak_supervision_label_sources:
            (Y_query, query_indices, source,) = labelSource.get_labeled_samples()

            if Y_query is not None:
                break
        if Y_query is None:
            ws_still_applicable = False
            continue

        copy_of_data_storage.label_samples(query_indices, Y_query, source)

        copy_of_classifier.fit(
            copy_of_data_storage.train_labeled_X,
            copy_of_data_storage.train_labeled_Y["label"].to_list(),
        )

    Y_pred = copy_of_classifier.predict(copy_of_data_storage.train_unlabeled_X)

    accuracy_with_that_label = accuracy_score(
        Y_pred, copy_of_data_storage.train_unlabeled_Y["label"].to_list()
    )

    print(
        "Testing out : {}, acc: {}".format(
            unlabeled_sample_indice, accuracy_with_that_label
        )
    )
    return unlabeled_sample_indice, accuracy_with_that_label


class ImitationLearner(ActiveLearner):
    def set_amount_of_peaked_objects(self, amount_of_peaked_objects):
        self.amount_of_peaked_objects = amount_of_peaked_objects

    def init_sampling_classifier(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        model = keras.Sequential(
            [
                layers.Dense(
                    2 * self.amount_of_peaked_objects,
                    input_dim=2 * self.amount_of_peaked_objects,
                    activation="relu",
                ),
                layers.Dense(2 * self.amount_of_peaked_objects, activation="relu"),
                layers.Dense(2 * self.amount_of_peaked_objects, activation="relu"),
                layers.Dense(
                    self.nr_queries_per_iteration, activation="softmax"
                ),  # muss das softmax sein?!
            ]
        )
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        print(model.summary())
        #  inputs = keras.Input(input_dim=2*self.amount_of_peaked_objects)
        #
        #  x = layers.Dense(units=self.amount_of_peaked_objects)
        #  outputs=layers.Dense(self.nr_queries_per_iteration, activation="softmax")(x)
        #  self.sampling_classifier = keras.Model(inputs=inputs, outputs=outputs
        self.sampling_classifier = model
        #  self.sampling_classifier = RandomForestClassifier(
        #      n_jobs=self.N_JOBS, random_state=self.RANDOM_SEED
        #  )
        #  init_x = np.zeros(
        #      (self.nr_queries_per_iteration, self.amount_of_peaked_objects)
        #  )
        #  init_y = [i for i in range(0, self.nr_queries_per_iteration)]
        #  print(init_x)
        #  print(init_y)
        #  self.sampling_classifier.fit(init_x, init_y)

        self.states = pd.DataFrame(
            data=None,
            columns=[
                str(i) + "_proba_1" for i in range(0, self.amount_of_peaked_objects)
            ]
            + [str(i) + "_proba_2" for i in range(0, self.amount_of_peaked_objects)],
        )
        self.optimal_policies = pd.DataFrame(
            data=None,
            columns=[
                str(i) + "_true_best_sample_indice"
                for i in range(0, self.amount_of_peaked_objects)
            ]
            + [
                str(i) + "_true_best_sample_acc"
                for i in range(0, self.amount_of_peaked_objects)
            ],
        )

    def fit_sampling_classifier(self):
        self.sampling_classifier.fit(
            self.data_storage.states, self.data_storage.optimal_policies
        )

    def move_labeled_queries(self, X_query, Y_query, query_indices):
        # move new queries from unlabeled to labeled dataset
        self.train_labeled_X = self.train_labeled_X.append(X_query)
        self.train_unlabeled_X = self.train_unlabeled_X.drop(query_indices)

        try:
            self.Y_train_strong_labels = self.Y_train_strong_labels.append(
                self.Y_train_unlabeled.loc[query_indices]
            )
        except KeyError:
            # in a non experiment setting an error will be thrown because self.Y_train_unlabeled of course doesn't contains the labels
            for query_index in query_indices:
                self.Y_train_strong_labels.loc[query_index] = [-1]

        self.Y_train_labeled = self.Y_train_labeled.append(Y_query)
        self.Y_train_unlabeled = self.Y_train_unlabeled.drop(
            query_indices, errors="ignore"
        )

        # remove indices from all clusters in unlabeled and add to labeled
        for cluster_id in self.train_unlabeled_X_cluster_indices.keys():
            list_to_be_removed_and_appended = []
            for indice in query_indices:
                if indice in self.train_unlabeled_X_cluster_indices[cluster_id]:
                    list_to_be_removed_and_appended.append(indice)

            # don't change a list you're iterating over!
            for indice in list_to_be_removed_and_appended:
                self.train_unlabeled_X_cluster_indices[cluster_id].remove(indice)
                self.train_labeled_X_cluster_indices[cluster_id].append(indice)

        # remove possible empty clusters
        self.train_unlabeled_X_cluster_indices = {
            k: v
            for k, v in self.train_unlabeled_X_cluster_indices.items()
            if len(v) != 0
        }

    """
    We take a "peak" into the future and annotate exactly those samples where we KNOW that they will benefit us the most
    """

    def calculate_next_query_indices(self, train_unlabeled_X_cluster_indices, *args):
        # merge indices from all clusters together and take the n most uncertain ones from them
        train_unlabeled_X_indices = list(
            chain(*list(train_unlabeled_X_cluster_indices.values()))
        )

        scores = []

        random.shuffle(train_unlabeled_X_indices)
        possible_samples_indices = train_unlabeled_X_indices[
            : self.amount_of_peaked_objects
        ]

        # parallelisieren
        with parallel_backend("loky", n_jobs=self.N_JOBS):
            scores = Parallel()(
                delayed(_future_peak)(
                    unlabeled_sample_indice,
                    self.weak_supervision_label_sources,
                    self.data_storage,
                    self.clf,
                    self.MAX_AMOUNT_OF_WS_PEAKS,
                )
                for unlabeled_sample_indice in possible_samples_indices
            )
        for labelSource in self.weak_supervision_label_sources:
            labelSource.data_storage = self.data_storage

        scores = sorted(scores, key=lambda tup: tup[1], reverse=True)

        possible_samples_probas = self.clf.predict_proba(
            self.data_storage.train_unlabeled_X.loc[possible_samples_indices]
        )
        arg_sorted_probas = np.argsort(
            -possible_samples_probas, axis=1, kind="quicksort", order=None
        )
        argmax_probas = arg_sorted_probas[:, 0]
        argsecond_probas = arg_sorted_probas[:, 1]

        x_policy = np.array([*argmax_probas, *argsecond_probas])
        # take first and second most examples from possible_samples_probas and append them then to states
        self.states = self.states.append(
            pd.Series(dict(zip(self.states.columns, x_policy))), ignore_index=True,
        )
        print(self.states)

        # save the indices of the n_best possible states, order doesn't matter
        self.optimal_policies = self.optimal_policies.append(
            pd.Series(
                dict(
                    zip(
                        self.optimal_policies.columns,
                        [a for a, _ in scores] + [b for _, b in scores],
                    )
                )
            ),
            ignore_index=True,
        )
        print(self.optimal_policies)

        print(x_policy)
        Y_pred = self.sampling_classifier.predict(x_policy)
        print(Y_pred)
        return Y_pred

        # hier dann stattdessen die Antwort vom hier trainiertem classifier zurückgeben
        return [k for k, v in scores[: self.nr_queries_per_iteration]]
