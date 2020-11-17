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
AMOUNT_OF_METRICS = 6
NR_BATCHES = 50
current_baseline = None


def _binarize_targets(df, TOP_N=5):
    df = df.to_frame()
    #  print(df.to_numpy())
    #  print("thres", np.sort(np.reshape(df.values, NR_BATCHES))[-TOP_N : -(TOP_N - 1)])
    df["threshold"] = [
        np.sort(np.reshape(df.values, NR_BATCHES))[-TOP_N : -(TOP_N - 1)]
        for _ in range(0, NR_BATCHES)
    ]

    for column_name in df.columns:
        if column_name == "threshold":
            continue
        df[column_name].loc[df[column_name] < df.threshold] = 0
        df[column_name].loc[df[column_name] >= df.threshold] = 1
    del df["threshold"]
    return np.reshape(df.to_numpy(), (1, NR_BATCHES))


def _jac_score(a, b):
    return len(np.intersect1d(a, b)) / len(np.union1d(a, b))


def _jac_score_binary(a, b):
    count_ones = 0
    for x, y in zip(a, b):
        if x == 1 and y == 1:
            count_ones += 1
    return count_ones / TOP_N


# make uncertainty positive, add 100 to it
for index, row in df.iterrows():
    if index % AMOUNT_OF_METRICS == 2:
        df.loc[index, df.columns != "source"] += 100


df["source"] = df["source"].apply(
    lambda x: x.replace("<function _calculate_", "").split(" at ")[0]
)


print(df.head())

cummulative_scores = defaultdict(int)

for i in range(0, math.ceil((len(df) / AMOUNT_OF_METRICS))):
    #  if i > 10:
    #      break
    top_ks = {}
    for j in range(i * AMOUNT_OF_METRICS, (i + 1) * AMOUNT_OF_METRICS):
        if j == len(df):
            continue
        top_ks[df.iloc[j]["source"]] = (
            (-df.iloc[j].drop("source")).argsort(axis=0)[:TOP_N].to_numpy()
        )
    #  pprint(top_ks)
    for a, b in combinations(top_ks.keys(), 2):
        cummulative_scores[str(a) + " --- " + str(b)] += _jac_score(
            top_ks[a], top_ks[b]
        )
    #  pprint(cummulative_scores)
    #  exit(-1)

pprint(cummulative_scores)


sns.set_theme(style="whitegrid")
df = pd.DataFrame(data=None, index=None, columns=["cummulative_scores", "value"])

for sampling_method, value in cummulative_scores.items():
    df = df.append(
        {
            "cummulative_scores": sampling_method,
            "value": value,
        },
        ignore_index=True,
    )
df["value"] = df["value"].astype(float)
print(df)

ax = sns.barplot(
    y="cummulative_scores",
    x="value",
    data=df,
)
plt.show()
