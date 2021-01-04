import glob
import csv
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import (
    olivetti_faces,
    fetch_20newsgroups_vectorized,
    fetch_lfw_people,
    fetch_rcv1,
    fetch_kddcup99,
)

data = fetch_kddcup99()
df = pd.DataFrame(data.data)
df["LABEL"] = data.target
print(df)


for column, dtype in df.dtypes.items():
    if column == "LABEL":
        continue
    if dtype not in ["int64", "float64"]:
        if len(df[column].unique()) > 2:
            #  print(pd.get_dummies(df[column], prefix=column))
            df = pd.concat(
                [
                    pd.get_dummies(df[column], prefix=column),
                    df.drop(column, axis=1),
                ],
                axis=1,
            )
        else:
            df.loc[:, column] = df.loc[:, column].astype("category").cat.codes
labelEncoder = LabelEncoder()
df["LABEL"] = labelEncoder.fit_transform(df.LABEL.values)

df = df.dropna()

print(df)

#  df.to_csv("kdd.csv")
#  df.to_csv(
#      "../datasets/large_cleaned/" + os.path.basename(f).split(".")[0] + ".csv",
#      index=False,
#  )
#

exit(-1)

#  for f in list(glob.glob("../datasets/uci/*")):
for f in list(glob.glob("../datasets/large_datasets/emnist/emnist-byclass*.csv")):
    print(f)
    parsing_args = parsing_dict["emnist"]

    df = pd.read_csv(
        f, sep=parsing_args[0], header=parsing_args[1], index_col=parsing_args[2]
    )
    if os.path.basename(f) == "PLANNING_plrx.txt":
        del df[13]
    if os.path.basename(f) == "BREAST.csv":
        del df["Unnamed: 32"]

    df = df.rename(columns={parsing_args[3]: "LABEL"})

    for column, dtype in df.dtypes.items():
        if column == "LABEL":
            continue
        if dtype not in ["int64", "float64"]:
            if len(df[column].unique()) > 2:
                #  print(pd.get_dummies(df[column], prefix=column))
                df = pd.concat(
                    [
                        pd.get_dummies(df[column], prefix=column),
                        df.drop(column, axis=1),
                    ],
                    axis=1,
                )
            else:
                df.loc[:, column] = df.loc[:, column].astype("category").cat.codes
    #  print(df)
    labelEncoder = LabelEncoder()
    #  df["LABEL"] = labelEncoder.fit_transform(df.LABEL.values)

    df = df.dropna()

    print(df)

    #  rf = RandomForestClassifier()
    #  rf.fit(df.drop("LABEL", axis=1), labelEncoder.fit_transform(df["LABEL"].values))
    #  print(os.path.basename(f))
    df.to_csv(
        "../datasets/large_cleaned/" + os.path.basename(f).split(".")[0] + ".csv",
        index=False,
    )
