import random
from collections import defaultdict
import math
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    jaccard_score,
    ndcg_score,
    cohen_kappa_score,
    label_ranking_loss,
    label_ranking_average_precision_score,
    dcg_score,
)

from itertools import combinations

#  df = pd.read_csv("metric_test_small.csv")
df = pd.read_csv("metric_test_50.csv")

# step 2: calculate correctnesses of rankings
TOP_N = 20
current_baseline = None
AMOUNT_OF_METRICS = len(df.source.unique())
NR_BATCHES = df.shape[1] - 1
AMOUNT_OF_HYBRID_TO_BE_ADDED = 4

df["source"] = df["source"].apply(
    lambda x: x.replace("<function _calculate_", "").split(" at ")[0]
)

print("Done renaming source")

new_index = []
print(AMOUNT_OF_METRICS)
print(AMOUNT_OF_HYBRID_TO_BE_ADDED)

counter = 1

for i in range(
    0,
    int(
        len(df) / AMOUNT_OF_METRICS * (AMOUNT_OF_METRICS + AMOUNT_OF_HYBRID_TO_BE_ADDED)
    ),
):
    if counter <= AMOUNT_OF_METRICS:
        new_index.append(i)
    elif counter == AMOUNT_OF_METRICS + AMOUNT_OF_HYBRID_TO_BE_ADDED:
        counter = 0
    counter += 1

df = df.set_index(pd.Index(new_index))
print(new_index[0:10])


def _jac_score(a, b):
    return len(np.intersect1d(a, b)) / len(np.union1d(a, b))


def random_padding(hybrid):
    padding_length = TOP_N - len(hybrid)
    return random.sample(
        [i for i in range(0, NR_BATCHES) if i not in hybrid],
        padding_length,
    )


#  make uncertainty positive, add 100 to it
#  for index, row in df.iterrows():
#      if index % AMOUNT_OF_METRICS == 2:
#          df.loc[index, df.columns != "source"] += 100

print("Done making uncertainty positive")
print(df[0:30])
for i in range(
    0, math.ceil((len(df) / (AMOUNT_OF_METRICS + AMOUNT_OF_HYBRID_TO_BE_ADDED)))
):
    #  print(i)
    real_index = i * (AMOUNT_OF_METRICS + AMOUNT_OF_HYBRID_TO_BE_ADDED)
    #  print(real_index)
    #  print(df.loc[real_index + 1]["source"])
    #  print(df.loc[real_index + 2]["source"])
    #  print(df.loc[real_index + 3]["source"])
    #  print(df.loc[real_index + 5]["source"])

    #  print(df.loc[real_index]["source"])
    top_future = (-df.loc[real_index].drop("source")).argsort(axis=0)[:TOP_N].tolist()

    top_furthest = (
        (-df.loc[real_index + 1].drop("source")).argsort(axis=0)[:TOP_N].tolist()
    )
    top_uncert = (
        (-df.loc[real_index + 2].drop("source")).argsort(axis=0)[:TOP_N].tolist()
    )
    top_furthest_lab = (
        (-df.loc[real_index + 3].drop("source")).argsort(axis=0)[:TOP_N].tolist()
    )

    hybrid1 = list(set(top_furthest_lab[:20]))
    df.loc[real_index + AMOUNT_OF_METRICS] = ["hybrid1"] + [
        1 if i in hybrid1 + random_padding(hybrid1) else 0 for i in range(0, NR_BATCHES)
    ]

    #  hybrid2 = list(set(top_furthest[-10:] + top_uncert[-5:]))
    hybrid2 = list(set(top_furthest[:20]))
    df.loc[real_index + AMOUNT_OF_METRICS + 1] = ["hybrid2"] + [
        1 if i in hybrid2 + random_padding(hybrid2) else 0 for i in range(0, NR_BATCHES)
    ]
    #  hybrid3 = list(set(top_furthest_lab[-10:] + top_uncert[-5:]))
    hybrid3 = list(set(top_future[:20]))
    df.loc[real_index + AMOUNT_OF_METRICS + 2] = ["hybrid3"] + [
        1 if i in hybrid3 + random_padding(hybrid3) else 0 for i in range(0, NR_BATCHES)
    ]
    #  hybrid4 = list(set(top_furthest_lab[-3:] + top_furthest[-3:] + top_uncert[-5:]))
    hybrid4 = list()
    df.loc[real_index + AMOUNT_OF_METRICS + 3] = ["hybrid4"] + [
        1 if i in hybrid4 + random_padding(hybrid4) else 0 for i in range(0, NR_BATCHES)
    ]

    #  print(df.loc[real_index])
    #  print(sorted(top_future))
    #  print(len(hybrid3))
    #  print(random_padding(hybrid3))
    #  print([1 if i in top_future else 0 for i in range(0, NR_BATCHES)])
    #  print(df.loc[real_index + AMOUNT_OF_METRICS + 2].drop("source").tolist())
    #  print([1 if i in top_furthest else 0 for i in range(0, NR_BATCHES)])
    #  print(df.loc[real_index + AMOUNT_OF_METRICS + 1].drop("source").tolist())
    #  print([1 if i in top_uncert else 0 for i in range(0, NR_BATCHES)])
    #  print(df.loc[real_index + AMOUNT_OF_METRICS + 2].drop("source").tolist())
    #  exit(-1)

    #  print("hybride:")
    #  print(sorted(hybrid1))
    #  print(sorted(hybrid2))
    #  print(sorted(hybrid3))
    #  print(sorted(hybrid4))
    #  if i > 10:
    #      break
df.sort_index(inplace=True)
print(df[0:50])
df.to_csv("metric_hybrid.csv", index=False)
