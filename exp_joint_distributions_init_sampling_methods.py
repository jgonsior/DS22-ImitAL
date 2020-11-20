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

from itertools import combinations, combinations_with_replacement, product

#  df = pd.read_csv("metric_test_small.csv")
#  df = pd.read_csv("metric_test_50.csv")
df = pd.read_csv("metric_hybrid.csv")
#  df = pd.read_csv("metric_hybrid_binary.csv")
print("Done reading csv")
# step 2: calculate correctnesses of rankings
TOP_N = 20
AMOUNT_OF_METRICS = len(df.source.unique())
NR_BATCHES = df.shape[1] - 1
current_baseline = None


def _jac_score(a, b):
    #  print(sorted(a))
    #  print(sorted(b))
    #  print(len(np.intersect1d(a, b)) / len(np.union1d(a, b)))
    return len(np.intersect1d(a, b)) / len(np.union1d(a, b))


print(df[0:20])

cummulative_scores = defaultdict(int)

for i in range(0, math.ceil((len(df) / AMOUNT_OF_METRICS))):
    top_ks = {}
    for j in range(i * AMOUNT_OF_METRICS, (i + 1) * AMOUNT_OF_METRICS):
        if j == len(df):
            continue
        top_ks[df.loc[j]["source"]] = (
            (-df.loc[j].drop("source")).argsort(axis=0)[:TOP_N].to_numpy()
        )
    #  for a, b in product(top_ks.keys(), repeat=2):
    for a, b in combinations(top_ks.keys(), 2):
        if (a == "future" and b == "hybrid3") or (a == "hybrid3" and b == "future"):
            if _jac_score(top_ks[a], top_ks[b]) != 1:
                print(_jac_score(top_ks[a], top_ks[b]))

        #  print(a)
        #  print(b)

        cummulative_scores[str(a) + " --- " + str(b)] += _jac_score(
            top_ks[a], top_ks[b]
        )
    #  exit(-1)
    cummulative_scores["future --- future"] += _jac_score(
        top_ks["future"], top_ks["future"]
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

ax = sns.barplot(y="cummulative_scores", x="value", data=df)
plt.show()
