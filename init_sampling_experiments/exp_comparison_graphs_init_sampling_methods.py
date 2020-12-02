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


metrics = [
    jaccard_score,
    "jaccard_score_binary",
    dcg_score,
    ndcg_score,
    "ndcg_score_top_k",
    cohen_kappa_score,
    label_ranking_loss,
    #  label_ranking_average_precision_score,
]

evaluation = {}

for metric in metrics:
    evaluation[metric] = {key: 0 for key in df.iloc[0:AMOUNT_OF_METRICS]["source"]}
    evaluation[metric]["maximum"] = 0


# make uncertainty positive, add 100 to it
for index, row in df.iterrows():
    if index % AMOUNT_OF_METRICS == 2:
        df.loc[index, df.columns != "source"] += 100


print(df.head())

for i in range(0, math.ceil((len(df) / AMOUNT_OF_METRICS))):
    for j in range(i * AMOUNT_OF_METRICS, (i + 1) * AMOUNT_OF_METRICS):
        if j == len(df):
            continue

        if j % AMOUNT_OF_METRICS == 0:
            current_baseline = df.iloc[j].drop("source")
            df.loc[j, "source"] = "maximum"

        for metric in metrics:
            metric_function = metric
            kwargs = {}
            if metric == jaccard_score or metric == cohen_kappa_score:
                # top_n rankings
                baseline = (-current_baseline).argsort(axis=0)[:TOP_N].to_numpy()
                to_compare = (
                    (-df.iloc[j].drop("source")).argsort(axis=0)[:TOP_N].to_numpy()
                )

                if metric == jaccard_score:
                    metric_function = _jac_score
                else:
                    baseline = sorted(baseline)
                    to_compare = sorted(to_compare)
            elif metric == "jaccard_score_binary":
                baseline = _binarize_targets(current_baseline, TOP_N=TOP_N)[0].tolist()
                to_compare = _binarize_targets(df.iloc[j].drop("source"), TOP_N=TOP_N)[
                    0
                ].tolist()
                metric_function = _jac_score_binary
            elif (
                metric == ndcg_score
                or metric == "ndcg_score_top_k"
                or metric == dcg_score
            ):
                # true values
                baseline = np.reshape(current_baseline.to_numpy(), (1, NR_BATCHES))
                to_compare = np.reshape(
                    df.iloc[j].drop("source").to_numpy(), (1, NR_BATCHES)
                )
                if metric == "ndcg_score_top_k" or metric == dcg_score:
                    kwargs = {"k": TOP_N}
                    metric_function = ndcg_score
            elif metric == label_ranking_loss:
                # 000111000 arrays
                baseline = _binarize_targets(current_baseline, TOP_N=TOP_N)
                to_compare = _binarize_targets(df.iloc[j].drop("source"), TOP_N=TOP_N)
            elif metric == label_ranking_average_precision_score:
                baseline = _binarize_targets(current_baseline, TOP_N=TOP_N)
                to_compare = np.reshape(
                    df.iloc[j].drop("source").to_numpy(), (1, NR_BATCHES)
                )
            #  print(metric)
            #  print(df.iloc[j]["source"])
            #  if metric == jaccard_score:
            #      print(sorted(baseline))
            #      print(sorted(to_compare))
            #  else:
            #      print(baseline)
            #      print(to_compare)
            #  print(metric_function(to_compare, baseline, **kwargs))
            #  print()
            evaluation[metric][df.iloc[j]["source"]] += metric_function(
                baseline, to_compare, **kwargs
            )
    #      print("#" * 80)
    #      print()
    #      print()
    #  exit(-1)

# per metric a bar chart
pprint(evaluation)

sns.set_theme(style="whitegrid")
df = pd.DataFrame(
    data=None, index=None, columns=["eval_metric", "value", "sampling_method"]
)

minimum_values = []
maximum_values = []

for sampling_method, d1 in evaluation.items():
    for eval_metric, value in d1.items():
        if "future" in eval_metric:
            continue
        df = df.append(
            {
                "eval_metric": eval_metric.replace("<function _calculate_", "").split(
                    " at "
                )[0],
                "value": value,
                "sampling_method": str(sampling_method)
                .replace("<function ", "")
                .split(" at ")[0],
            },
            ignore_index=True,
        )
    minimum_values.append(min(list(d1.values())[1:]))
    maximum_values.append(max(list(d1.values())[1:]))
df["value"] = df["value"].astype(float)
print(df)

ax = sns.catplot(
    y="eval_metric",
    x="value",
    col="sampling_method",
    data=df,
    kind="bar",
    legend=True,
    sharex=False,
)
print(list(zip(minimum_values, maximum_values)))
for i, subplot in enumerate(ax.axes_dict.values()):
    subplot.set_xlim(minimum_values[i], maximum_values[i])
plt.show()
