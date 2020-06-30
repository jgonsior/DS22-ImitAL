import collections
import random
import pandas as pd
from ..activeLearner import ActiveLearner
from .baseWeakSupervision import BaseWeakSupervision


class WeakClust(BaseWeakSupervision):
    # threshold params
    MINIMUM_CLUSTER_UNITY_SIZE = MINIMUM_RATIO_LABELED_UNLABELED = None

    def get_labeled_samples(self):
        certain_X = recommended_labels = certain_indices = None
        cluster_found = False

        # check if the most prominent label for one cluster can be propagated over to the rest of it's cluster
        for cluster_id in self.data_storage.get_df("train", unlabeled=True)[
            "cluster"
        ].unique():
            cluster_ys = self.data_storage.get_df("train", clusters=[cluster_id])[
                "label"
            ]
            cluster_indices = cluster_ys.index
            label_frequencies = collections.Counter(cluster_ys)

            if (
                1 - label_frequencies[-1] / len(cluster_indices)
            ) > self.MINIMUM_CLUSTER_UNITY_SIZE:
                if (
                    label_frequencies.most_common(1)[0][1]
                    > len(cluster_indices) * self.MINIMUM_RATIO_LABELED_UNLABELED
                ):
                    # contains only unlabeled samples from the cluster
                    certain_X = (
                        self.data_storage.get_df()
                        .loc[cluster_indices]
                        .loc[self.data_storage.get_df()["label"] == -1]
                    )
                    print(self.data_storage.get_df().loc[cluster_indices])
                    print(certain_X)
                    certain_indices = certain_X.index
                    recommended_labels = [
                        label_frequencies.most_common(1)[0][0] for _ in certain_indices
                    ]
                    #  recommended_labels = pd.DataFrame(
                    #      recommended_labels, index=certain_X.index
                    #  )
                    #  log_it("Cluster ", cluster_id, certain_indices)
                    cluster_found = True
                    break

        # delete this cluster from the list of possible cluster for the next round
        return certain_X, recommended_labels, certain_indices, "C"
