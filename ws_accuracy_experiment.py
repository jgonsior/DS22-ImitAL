from active_learning.learner.standard import Learner, get_classifier
from active_learning.weak_supervision.SelfTraining import SelfTraining
import argparse
import numpy as np
import copy
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

config: argparse.Namespace = get_active_config(  # type: ignore
    [
        (["--AMOUNT_OF_FEATURES"], {"type": int, "default": 1}),
    ],
    return_parser=False,
)

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


def evaluate_and_print_prediction(Y_pred, Y_true, title):
    acc = accuracy_score(Y_true, Y_pred)
    f1 = f1_score(Y_true, Y_pred, average="weighted")
    c = Counter(Y_pred)

    print(
        "{:<60} Acc: {:>6.2%} \t F1: {:>6.2%} \t MC: {:>3}-{:>4.1%}".format(
            title,
            acc,
            f1,
            c.most_common(1)[0][0],
            c.most_common(1)[0][1] / len(Y_pred),
        )
    )


df, synthetic_creation_args = load_synthetic(
    config.RANDOM_SEED,
)

data_storage: DataStorage = DataStorage(df=df, TEST_FRACTION=config.TEST_FRACTION)
learner = get_classifier("RF", random_state=config.RANDOM_SEED)

learner.fit(
    data_storage.X[data_storage.labeled_mask],
    data_storage.Y_merged_final[data_storage.labeled_mask],
)


ws_list: List[BaseWeakSupervision] = [
    SyntheticLabelingFunctions(X=data_storage.X, Y=data_storage.exp_Y)
    for _ in range(0, config.AMOUNT_OF_SYNTHETIC_LABELLING_FUNCTIONS)
]  # type: ignore


# tweak to do more than one iteration of self training!
""" ws_list.append(SelfTraining(0.99, 0.99))
ws_list.append(SelfTraining(0.9, 0.9))
ws_list.append(SelfTraining(0.8, 0.8))
ws_list.append(SelfTraining(0.7, 0.7)) """

# add label propagation

"""print(data_storage.test_mask)
print(data_storage.unlabeled_mask)
print(data_storage.labeled_mask)
print(len(data_storage.X))
"""
print("\n" * 5)
print("Metrics when only using single LFs")
for ws in ws_list:
    # calculate f1 and acc for ws on test AND train dataset
    # it actually only get's computed on the test mask, not the train mask itself
    Y_pred = ws.get_labels(data_storage.test_mask, data_storage, learner)
    evaluate_and_print_prediction(
        data_storage.exp_Y[data_storage.test_mask], Y_pred, ws.identifier
    )


data_storage.set_weak_supervisions(ws_list, MajorityVoteLabelMergeStrategy())
data_storage.generate_weak_labels(learner, mask=data_storage.test_mask)

# Only Majority Vote, no classifier
print()
evaluate_and_print_prediction(
    data_storage.Y_merged_final[data_storage.test_mask],
    data_storage.exp_Y[data_storage.test_mask],
    "Majority Vote",
)

# compute the 50/100/200/500 worst wrongly classified samples -> classify them correctly (aka. fake active learning) -> is there really room for improvement after falsely applyed WS??

# apply it to the train_set

# trained on WS + Minimal AL


def train_and_evaluate(title, original_data_storage, WEIGHTS=0, WS=True):
    data_storage = copy.deepcopy(original_data_storage)
    learner = get_classifier("RF", random_state=config.RANDOM_SEED)
    data_storage.generate_weak_labels(learner)

    if WEIGHTS != 0:
        weights = []
        for indice in data_storage.weakly_combined_mask:
            if indice in data_storage.labeled_mask:
                weights.append(WEIGHTS)
            else:
                weights.append(1)
    else:
        weights = None
    if WS:
        mask = data_storage.weakly_combined_mask
    else:
        mask = data_storage.labeled_mask
    learner.fit(
        data_storage.X[mask],
        data_storage.Y_merged_final[mask],
        sample_weight=weights,  # type: ignore
    )
    Y_pred = learner.predict(data_storage.X[data_storage.test_mask])

    Y_true = data_storage.exp_Y[data_storage.test_mask]

    evaluate_and_print_prediction(Y_pred, Y_true, title)


def test_one_labeled_set(original_data_storage, label_strategy="start_set", param=5):
    data_storage = copy.deepcopy(original_data_storage)

    if label_strategy == "random":
        random_sample_ids = np.random.choice(
            data_storage.unlabeled_mask,
            size=param,
            replace=False,
        )

        data_storage.label_samples(
            random_sample_ids, data_storage.exp_Y[random_sample_ids], "AL"
        )
    print()
    print(
        label_strategy
        + ": #lab: "
        + str(len(data_storage.labeled_mask))
        + "\t param: "
        + str(param),
    )
    train_and_evaluate("RF No WS", data_storage, WS=False)
    train_and_evaluate("RF No Weights", data_storage)
    train_and_evaluate("RF Weihgt 10", data_storage, WEIGHTS=10)
    train_and_evaluate("RF Weihgt 50", data_storage, WEIGHTS=50)
    train_and_evaluate("RF Weihgt 100", data_storage, WEIGHTS=100)
    train_and_evaluate("RF Weihgt 1000", data_storage, WEIGHTS=1000)


print("\n" * 3)
print("Combining WS functions using majority vote + random labeled correct samples")

test_one_labeled_set(data_storage, label_strategy="start_set")
test_one_labeled_set(data_storage, label_strategy="random", param=5)
exit(-1)
test_one_labeled_set(data_storage, label_strategy="random", param=10)
test_one_labeled_set(data_storage, label_strategy="random", param=25)
test_one_labeled_set(data_storage, label_strategy="random", param=50)
test_one_labeled_set(data_storage, label_strategy="random", param=100)
test_one_labeled_set(data_storage, label_strategy="random", param=200)

# -> how to combine AL and WS labels in a way, that the experiment actually benefits from the labels

# wrong_mask = np.logical_not(np.array_equal(Y_pred, Y_true))

# print(data_storage.Y_merged_final[wrong_mask])
# print(data_storage.exp_Y[wrong_mask])

# calculate acc/f1 now and before ONLY on those without abstain!, but add "coverage" to the WS LF
# a) get those samples, who are least covered by the LF
# b) get those samples, where the classification is wrong by the merged LFs
# c) get those samples, with the greatest disagreement among the LFs