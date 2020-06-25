from joblib import Parallel, delayed, parallel_backend
import random
from itertools import chain
import copy
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import accuracy_score
from ..activeLearner import ActiveLearner


class OptimalForecastSampler(ActiveLearner):
    def set_amount_of_peaked_objects(self, amount_of_peaked_objects):
        self.amount_of_peaked_objects = amount_of_peaked_objects

    def move_labeled_queries(self, X_query, Y_query, query_indices):
        # move new queries from unlabeled to labeled dataset
        self.X_train_labeled = self.X_train_labeled.append(X_query)
        self.X_train_unlabeled = self.X_train_unlabeled.drop(query_indices)

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
        for cluster_id in self.X_train_unlabeled_cluster_indices.keys():
            list_to_be_removed_and_appended = []
            for indice in query_indices:
                if indice in self.X_train_unlabeled_cluster_indices[cluster_id]:
                    list_to_be_removed_and_appended.append(indice)

            # don't change a list you're iterating over!
            for indice in list_to_be_removed_and_appended:
                self.X_train_unlabeled_cluster_indices[cluster_id].remove(indice)
                self.X_train_labeled_cluster_indices[cluster_id].append(indice)

        # remove possible empty clusters
        self.X_train_unlabeled_cluster_indices = {
            k: v
            for k, v in self.X_train_unlabeled_cluster_indices.items()
            if len(v) != 0
        }

    def _future_peak(
        self,
        unlabeled_sample_indice,
        copy_of_classifier,
        copy_of_X_train,
        copy_of_Y_train,
    ):

        # in loop meine WS-Strategien mit einbeziehen -> sinnvolle Cluster berechnen, DAgger vorschlagen pro Cluster nur einmal zu samplen, und den Rest dann weakly zu labeln
        copy_of_X_train = copy_of_X_train.append(
            self.data_storage.X_train_unlabeled.loc[unlabeled_sample_indice]
        )
        copy_of_Y_train = copy_of_Y_train.append(
            self.data_storage.Y_train_unlabeled.loc[unlabeled_sample_indice]
        )
        copy_of_classifier.fit(copy_of_X_train, copy_of_Y_train[0])

        Y_pred = copy_of_classifier.predict(self.data_storage.X_train_unlabeled)

        accuracy_with_that_label = accuracy_score(
            Y_pred, self.data_storage.Y_train_unlabeled[0]
        )

        # remove the indics again
        copy_of_X_train.drop(unlabeled_sample_indice, inplace=True)
        copy_of_Y_train.drop(unlabeled_sample_indice, inplace=True)

        print(
            "Testing out : {}, acc: {}".format(
                unlabeled_sample_indice, accuracy_with_that_label
            )
        )
        return unlabeled_sample_indice, accuracy_with_that_label

    """
    We take a "peak" into the future and annotate exactly those samples where we KNOW that they will benefit us the most
    """

    def calculate_next_query_indices(self, X_train_unlabeled_cluster_indices, *args):
        # merge indices from all clusters together and take the n most uncertain ones from them
        X_train_unlabeled_indices = list(
            chain(*list(X_train_unlabeled_cluster_indices.values()))
        )

        scores = []
        copy_of_classifier = copy.deepcopy(self.clf)
        copy_of_X_train = copy.deepcopy(self.data_storage.X_train_labeled)
        copy_of_Y_train = copy.deepcopy(self.data_storage.Y_train_labeled)

        random.shuffle(X_train_unlabeled_indices)

        # parallelisieren
        with parallel_backend("loky", n_jobs=self.N_JOBS):
            scores = Parallel()(
                delayed(self._future_peak)(
                    unlabeled_sample_indice,
                    copy_of_classifier,
                    copy_of_X_train,
                    copy_of_Y_train,
                )
                for unlabeled_sample_indice in X_train_unlabeled_indices[
                    : self.amount_of_peaked_objects
                ]
            )

        scores = sorted(scores, key=lambda tup: tup[1], reverse=True)
        return [k for k, v in scores[: self.nr_queries_per_iteration]]
