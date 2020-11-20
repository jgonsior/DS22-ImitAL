import argparse
import locale
import sys

import pandas as pd

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
