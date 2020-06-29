from itertools import chain

import numpy as np
from scipy.stats import entropy

from ..activeLearner import ActiveLearner


class UncertaintySampler(ActiveLearner):
    def set_uncertainty_strategy(self, strategy):
        self.strategy = strategy

    def setClassifierClasses(self, classes):
        self.classifier_classes = classes

    def calculate_next_query_indices(self, X_train_unlabeled_cluster_indices, *args):
        # merge indices from all clusters together and take the n most uncertain ones from them
        X_of_clusters = self.data_storage.get_X(
            "train", unlabeled=True, clusters=X_train_unlabeled_cluster_indices
        )
        indices_of_clusters = X_of_clusters.index

        # recieve predictions and probabilitys
        # for all possible classifications of CLASSIFIER
        Y_temp_proba = self.clf.predict_proba(X_of_clusters)

        if self.strategy == "least_confident":
            result = 1 - np.amax(Y_temp_proba, axis=1)
        elif self.strategy == "max_margin":
            margin = np.partition(-Y_temp_proba, 1, axis=1)
            result = -np.abs(margin[:, 0] - margin[:, 1])
        elif self.strategy == "entropy":
            result = np.apply_along_axis(entropy, 1, Y_temp_proba)

        # sort indices_of_cluster by argsort
        argsort = np.argsort(-result)
        query_indices = np.array(indices_of_clusters)[argsort]

        # return smallest probabilities
        return query_indices[: self.nr_queries_per_iteration]
