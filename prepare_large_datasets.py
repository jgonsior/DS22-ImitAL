import glob
import csv
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from sklearn.datasets import (
    fetch_olivetti_faces,
    fetch_20newsgroups_vectorized,
    fetch_lfw_people,
    fetch_rcv1,
    fetch_kddcup99,
)
from keras.datasets import cifar10

# load the dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print shape of each variable
print("train image shape:", x_train.shape)  # output (50000, 32, 32, 3)
print("train label shape:", y_train.shape)  # output (50000, 1)
print("test image shape:", x_test.shape)  # output (10000, 32, 32, 3)
print("test label shape:", y_test.shape)  # output (10000, 1)

x_test = np.reshape(x_test, (10000, 3072))
df = pd.DataFrame(x_test)
df["LABEL"] = y_test.flatten()
print(df)

labelEncoder = LabelEncoder()
df["LABEL"] = labelEncoder.fit_transform(df.LABEL.values)

feature_columns = df.columns.to_list()
feature_columns.remove("LABEL")

# feature normalization
scaler = RobustScaler()
df[feature_columns] = scaler.fit_transform(df[feature_columns])

# scale back to [0,1]
scaler = MinMaxScaler()
df[feature_columns] = scaler.fit_transform(df[feature_columns])

print(df)

df.to_csv("cifar10.csv", index=False)

exit(-1)


for fetch_func in [
    #  fetch_olivetti_faces,
    #  fetch_20newsgroups_vectorized,
    #  fetch_lfw_people,
    fetch_rcv1,
    #  fetch_kddcup99,
]:
    print(fetch_func.__name__)
    data = fetch_func()
    df = pd.DataFrame(data.data)
    df["LABEL"] = data.target
    print(df)
    for column, dtype in df.dtypes.items():
        if column == "LABEL":
            continue
        if dtype not in ["int64", "float64", "float32"]:
            print(str(column) + ": \t \t " + str(dtype))
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

    feature_columns = df.columns.to_list()
    feature_columns.remove("LABEL")

    # feature normalization
    scaler = RobustScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    # scale back to [0,1]
    scaler = MinMaxScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])

    print(df)

    df.to_csv(fetch_func.__name__ + "_.csv", index=False)
    #  df.to_csv(
    #      "../datasets/large_cleaned/" + os.path.basename(f).split(".")[0] + ".csv",
    #      index=False,
    #  )
    #


exit(-1)

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
