from active_learning.learner.standard import Learner, get_classifier
from active_learning.weak_supervision.SelfTraining import SelfTraining
import argparse
import numpy as np
import os
import pandas as pd
import random
from sklearn.metrics import accuracy_score, f1_score
from timeit import default_timer as timer
from typing import List
from active_learning.config import get_active_config
from active_learning.dataStorage import DataStorage
from active_learning.datasets import load_synthetic
from active_learning.logger import init_logger
from active_learning.merge_weak_supervision_label_strategies.MajorityVoteLabelMergeStrategy import (
    MajorityVoteLabelMergeStrategy,
)
from collections import Counter

from active_learning.weak_supervision import SyntheticLabelingFunctions
from active_learning.weak_supervision.BaseWeakSupervision import BaseWeakSupervision

config: argparse.Namespace = get_active_config()  # type: ignore

# python ws_accuracy_experiment.py --OUTPUT_DIRECTORY _experiment_resultss/tmp --DATASET_NAME synthetic --AMOUNT_OF_FEATURES -1 --RANDOM_SEED 0

# -2 means that a true random seed is used, all other numbers use the provided CLI argument random_seed
if config.RANDOM_SEED == -2:
    random_but_not_random = True
else:
    random_but_not_random = False


init_logger(config.LOG_FILE)

if random_but_not_random:
    config.RANDOM_SEED = random.randint(0, 2147483647)
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)

df, synthetic_creation_args = load_synthetic(
    config.RANDOM_SEED,
    config.NEW_SYNTHETIC_PARAMS,
    config.VARIABLE_DATASET,
    config.AMOUNT_OF_FEATURES,
    config.HYPERCUBE,
    config.GENERATE_NOISE,
)

data_storage: DataStorage = DataStorage(df=df, TEST_FRACTION=config.TEST_FRACTION)


learner = get_classifier("RF", random_state=config.RANDOM_SEED)

mask = data_storage.labeled_mask
learner.fit(data_storage.X[mask], data_storage.Y_merged_final[mask])


ws_list: List[BaseWeakSupervision] = [
    SyntheticLabelingFunctions(X=data_storage.X, Y=data_storage.exp_Y)
    for i in range(0, config.AMOUNT_OF_SYNTHETIC_LABELLING_FUNCTIONS)
]  # type: ignore


# tweak to do more than one iteration of self training!
ws_list.append(SelfTraining(0.99, 0.99))
ws_list.append(SelfTraining(0.9, 0.9))
ws_list.append(SelfTraining(0.8, 0.8))
ws_list.append(SelfTraining(0.7, 0.7))

# add label propagation

for ws in ws_list:
    # calculate f1 and acc for ws on test AND train dataset
    Y_pred = ws.get_labels(data_storage.test_mask, data_storage, learner)
    acc = accuracy_score(data_storage.exp_Y[data_storage.test_mask], Y_pred)
    f1 = f1_score(
        data_storage.exp_Y[data_storage.test_mask], Y_pred, average="weighted"
    )
    c = Counter(Y_pred)

    print(
        "{:<60} Acc: {:.2%} F1: {:.2%} MC: {}-{:.1%}".format(
            ws.identifier,
            acc,
            f1,
            c.most_common(1)[0][0],
            c.most_common(1)[0][1] / len(Y_pred),
        )
    )


data_storage.set_weak_supervisions(ws_list, MajorityVoteLabelMergeStrategy())
data_storage.generate_weak_labels(learner, mask=data_storage.test_mask)

# compute the acc + f1 with majority vote / snorkel
print()
Y_pred = data_storage.Y_merged_final[data_storage.test_mask]
acc = accuracy_score(data_storage.exp_Y[data_storage.test_mask], Y_pred)
f1 = f1_score(data_storage.exp_Y[data_storage.test_mask], Y_pred, average="weighted")
c = Counter(Y_pred)

print(
    "{:<60} Acc: {:.2%} F1: {:.2%} MC: {}-{:.1%}".format(
        "Majority Vote",
        acc,
        f1,
        c.most_common(1)[0][0],
        c.most_common(1)[0][1] / len(Y_pred),
    )
)


# compute the 50/100/200/500 worst wrongly classified samples -> classify them correctly (aka. fake active learning) -> is there really room for improvement after falsely applyed WS??

# apply it to the train_set
data_storage.generate_weak_labels(learner)

weights = []
for indice in data_storage.weakly_combined_mask:
    if indice in data_storage.labeled_mask:
        weights.append(100)
    else:
        weights.append(1)

learner.fit(
    data_storage.X[data_storage.weakly_combined_mask],
    data_storage.Y_merged_final[data_storage.weakly_combined_mask],
    sample_weight=weights,
)
Y_pred = learner.predict(data_storage.X[data_storage.test_mask])
Y_true = data_storage.exp_Y[data_storage.test_mask]

acc = accuracy_score(Y_true, Y_pred)
f1 = f1_score(Y_true, Y_pred, average="weighted")
c = Counter(Y_pred)

print(
    "{:<60} Acc: {:.2%} F1: {:.2%} MC: {}-{:.1%}".format(
        "Trained RF",
        acc,
        f1,
        c.most_common(1)[0][0],
        c.most_common(1)[0][1] / len(Y_pred),
    )
)


wrong_mask = np.logical_not(np.array_equal(Y_pred, Y_true))

print(data_storage.Y_merged_final[wrong_mask])
print(data_storage.exp_Y[wrong_mask])

# calculate acc/f1 a) mit weakly_labeled b) ohne weakly labeled c) mit gewichten d) mit höheren gewichten e) mit mehr echten labels (best-case AL, siehe unten!)
# calculate acc/f1 now and before ONLY on those without abstain!, but add "coverage" to the WS LF
# a) get those samples, who are least covered by the LF
# b) get those samples, where the classification is wrong by the merged LFs
# c) get those samples, with the greatest disagreement among the LFs