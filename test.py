import timeit
from math import log, e

import numpy as np
from scipy.stats import entropy
from sklearn.metrics.cluster import entropy as sk_entropy


def entropy1(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


def entropy2(labels, base=None):
    """ Computes entropy of label distribution. """

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.0

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent


def entropy3(labels, base=None):
    return sk_entropy(labels)
    #  vc = pd.Series(labels).value_counts(normalize=True, sort=False)
    #  base = e if base is None else base
    #  return -(vc * np.log(vc) / np.log(base)).sum()


def entropy4(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    base = e if base is None else base
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()


repeat_number = 1000

labels = np.random.choice

label_list_str = str(np.random.sample(10000).tolist())

a = timeit.repeat(
    stmt="""entropy1(labels)""",
    setup="""labels=""" + label_list_str + """;from __main__ import entropy1""",
    repeat=3,
    number=repeat_number,
)

b = timeit.repeat(
    stmt="""entropy2(labels)""",
    setup="""labels=""" + label_list_str + """;from __main__ import entropy2""",
    repeat=3,
    number=repeat_number,
)

c = timeit.repeat(
    stmt="""entropy3(labels)""",
    setup="""labels=""" + label_list_str + """;from __main__ import entropy3""",
    repeat=3,
    number=repeat_number,
)

d = timeit.repeat(
    stmt="""entropy4(labels)""",
    setup="""labels=""" + label_list_str + """;from __main__ import entropy4""",
    repeat=3,
    number=repeat_number,
)

# for loop to print out results of timeit
for approach, timeit_results in zip(
    ["scipy/numpy", "numpy/math", "pandas/numpy", "numpy"], [a, b, c, d]
):
    print("Method: {}, Avg.: {:.6f}".format(approach, np.array(timeit_results).mean()))
