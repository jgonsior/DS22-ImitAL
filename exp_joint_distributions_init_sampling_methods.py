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
import os
import glob

plot_data = pd.DataFrame(
    data=None, index=None, columns=["cummulative_scores", "value", "init_sample_size"]
)

for hybrid_metric_file in list(glob.glob("metric_test_*_hybrid.csv")):
    #  if hybrid_metric_file == "metric_test_50.csv":
    #      continue
    print(hybrid_metric_file)
    df = pd.read_csv(hybrid_metric_file)

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

    for sampling_method, value in cummulative_scores.items():
        if sampling_method.startswith("future"):
            # remove final ints from sampling_method
            if sampling_method.endswith(str(NR_BATCHES)):
                sampling_method = sampling_method[: -len(str(NR_BATCHES))]
            plot_data = plot_data.append(
                {
                    "cummulative_scores": sampling_method,
                    "value": value,
                    "init_sample_size": NR_BATCHES,
                },
                ignore_index=True,
            )

plot_data["value"] = plot_data["value"].astype(float)
print(plot_data)
sns.set_theme(style="whitegrid")

ax = sns.barplot(
    y="cummulative_scores", x="value", hue="init_sample_size", data=plot_data
)
plt.show()
