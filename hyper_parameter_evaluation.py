import itertools
import os
import math
from pprint import pprint
import glob
import argparse
import locale
import math
import pickle
import sys
from itertools import chain, combinations
from itertools import product
from queue import Queue
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

parser = argparse.ArgumentParser()
parser.add_argument("--CSV_FILE", default="tmp/hyper_parameters.csv")
parser.add_argument("--GROUP_COLUMNS", action="append")
parser.add_argument("--VALUE_GROUPINGS")
parser.add_argument("--SAVE_FILE", default=None)
parser.add_argument("--TITLE", default="")


config = parser.parse_args()

if len(sys.argv[:-1]) == 0:
    parser.print_help()
    parser.exit()

# iterate over csv files
# get the used hyper_parameters out of the generated csv files


hyper_parameters_accs = defaultdict(list)
hyper_parameters_2d_accs = defaultdict(list)
hyper_parameters_3d_accs = defaultdict(list)

for file in list(
    glob.glob(
        "tmp/plots_NR_QUERIES_PER_ITERATION_TRAIN_CLASSIFIER_TRAIN_VARIABLE_DATASET_TRAIN_NEW_SYNTHETIC_PARAMS_TRAIN_CONVEX_HULL_SAMPLING_TRAIN_GENERATE_NOISE_TRAIN_STATE_DIFF_PROBAS_TRAIN_STATE_PREDICTED_CLASS_TEST_CLASSIFIER_TEST_GENERATE_NOISE/_*"
    )
)[:]:
    df = pd.read_csv(file)

    if "sampling" not in df.columns:
        print(file)
        continue

    samplings = df.sampling.unique()

    accs = {}
    for sampling_name in samplings:
        accs[sampling_name] = df.loc[df.sampling == sampling_name][
            "acc_test_oracle"
        ].mean()

    keys = list(accs.keys())
    keys.remove("uncertainty_max_margin")
    keys.remove("uncertainty_lc")
    keys.remove("uncertainty_entropy")
    keys.remove("random")
    hyper_parameters_string = keys[0]

    if type(hyper_parameters_string) == float:
        print(file)
        continue

    diff = accs["uncertainty_max_margin"] - accs[hyper_parameters_string]

    # dissect hyper_parameters_string
    hyper_parameters_string = hyper_parameters_string.split("/")[1]
    hyper_parameters = hyper_parameters_string.split("#")

    for hyper_parameter in hyper_parameters:
        hyper_parameters_accs[hyper_parameter].append(diff)

    for hp1, hp2 in itertools.combinations(hyper_parameters, 2):
        hyper_parameters_2d_accs[hp1 + "#" + hp2].append(diff)

    for hp1, hp2, hp3 in itertools.combinations(hyper_parameters, 3):
        hyper_parameters_3d_accs[hp1 + "#" + hp2 + "#" + hp3].append(diff)

    #  if diff < 0:
    #      #  print(file)
    #      print("{:.5f}: {}".format(diff, str(hyper_parameters)))
    #
    #      if diff <= -0.0026:
    #          os.system(
    #              "python compare_distributions.py --CSV_FILE "
    #              + file
    #              + "  --GROUP_COLUMNS sampling"
    #              + " --SAVE_FILE ui"
    #              #  + " --TITLE "
    #          )
    #  pprint(hyper_parameters_accs)
    #  exit(-2)
#  pprint(hyper_parameters_accs)

for hyper_parameter, diffs in hyper_parameters_accs.items():
    print(
        "{:<40} {:10.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(
            hyper_parameter,
            np.mean(diffs),
            np.std(diffs),
            np.var(diffs),
            np.max(diffs),
            np.min(diffs),
        )
    )
for hyper_parameter, diffs in hyper_parameters_2d_accs.items():
    print(
        "{:<80} {:10.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(
            hyper_parameter,
            np.mean(diffs),
            np.std(diffs),
            np.var(diffs),
            np.max(diffs),
            np.min(diffs),
        )
    )
for hyper_parameter, diffs in hyper_parameters_3d_accs.items():
    print(
        "{:<120} {:10.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:3}".format(
            hyper_parameter,
            np.mean(diffs),
            np.std(diffs),
            np.var(diffs),
            np.max(diffs),
            np.min(diffs),
            len(diffs),
        )
    )
