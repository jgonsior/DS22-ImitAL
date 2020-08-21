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
parser.add_argument("--MISSING_COLUMN_NAME")
parser.add_argument("--MISSING_COLUMN_VALUE")


config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()

print(config)


df = pd.read_csv(config.CSV_FILE)

print(len(df.columns))

if len(df.columns) == 57:
    df[config.MISSING_COLUMN_NAME] = config.MISSING_COLUMN_VALUE
    df.to_csv(config.CSV_FILE + "_2", index=False)
