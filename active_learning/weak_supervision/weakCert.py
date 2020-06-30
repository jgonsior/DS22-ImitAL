import numpy as np
import collections
import random
import pandas as pd

from ..activeLearner import ActiveLearner
from .baseWeakSupervision import BaseWeakSupervision


class WeakCert(BaseWeakSupervision):
    # threshold param
    CERTAINTY_THRESHOLD = CERTAINTY_RATIO = None

    def get_labeled_samples(self):
        X_train_unlabeled = self.data_storage.get_X("train", unlabeled=True)
        # calculate certainties for all of X_train_unlabeled
        certainties = self.clf.predict_proba(X_train_unlabeled)

        amount_of_certain_labels = np.count_nonzero(
            np.where(np.max(certainties, 1) > self.CERTAINTY_THRESHOLD)
        )

        if amount_of_certain_labels > len(X_train_unlabeled) * self.CERTAINTY_RATIO:

            # for safety reasons I refrain from explaining the following
            certain_indices = [
                j
                for i, j in enumerate(X_train_unlabeled.index.tolist())
                if np.max(certainties, 1)[i] > self.CERTAINTY_THRESHOLD
            ]

            certain_X = self.data_storage.get_X().loc[certain_indices]

            recommended_labels = self.clf.predict(certain_X)

            return recommended_labels, certain_indices, "U"
        else:
            return None, None, None
