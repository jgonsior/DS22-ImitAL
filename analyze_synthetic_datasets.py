from itertools import product
import sys
import math
from queue import Queue
from itertools import chain, combinations, permutations
import pickle
from tabulate import tabulate
from IPython.core.display import display, HTML
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit
import locale
import argparse

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

parser = argparse.ArgumentParser()
parser.add_argument("--CSV_FILE", default="tmp/hyper_parameters.csv")
parser.add_argument("--GROUP_COLUMNS", action="append")
parser.add_argument("--VALUE_GROUPINGS")


config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()


df = pd.read_csv(config.CSV_FILE)
accs = df["acc_test"].multiply(100)
mu = accs.mean()
print(mu)
print(df.acc_test.std())

shuffled_accs = [accs.sample(frac=1) for _ in range(0, 10)]


for CUT in [10000]:  # [1000, 10000, 20000, 50000]:
    means = []
    for accs in shuffled_accs:
        for i in range(0, int(len(df) / CUT)):
            #  print("{}-{}", 0 + i * CUT, CUT + i * CUT)
            mean_i = accs[(0 + i * CUT) : (CUT + i * CUT)].mean()
            #  std_i = accs[(0 + i * CUT) : (CUT + i * CUT)].std()
            means.append(mean_i)
            print("{:>2}: {:>7.2f} {:>7.2f}".format(i, mu - mean_i, mean_i))  # , std_i

    print("{:>9} {:>7.2f}".format(CUT, np.std(means)))
    #  print(means)
