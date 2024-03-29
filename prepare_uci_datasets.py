import glob
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

parsing_dict = {
    "dwtc.csv": [",", 0, 0, "CLASS"],
    "PLANNING_plrx.txt": ["\t", None, None, 12],
    "GERMAN_credit_data.data": ["\s+", None, None, 24],
    "FERTILITY.csv": [",", 0, None, "Diagnosis"],
    "PIMA-indians-diabetes.csv": [",", 0, None, "Outcome"],
    "australian.dat": [" ", None, None, 14],
    "ILPD_Indian Liver Patient Dataset (ILPD).csv": [",", 0, None, "label"],
    "IONOSPHERE_ionosphere.data": [",", None, None, 34],
    "HEART.csv": [",", 0, None, "target"],
    "HABERMAN.csv": [",", None, None, 3],
    "BREAST.csv": [",", 0, None, "diagnosis"],
    "DIABETES.csv": [",", 0, None, "Outcome"],
    "abalone.csv": [",", 0, None, "label"],
    "adult.csv": [",", 0, None, "Target"],
    #  "wine.csv": [",", 0, None, "Type"],
    "wine.csv": [",", 0, None, "Class"],
    "glass.csv": [",", 0, None, "Type"],
    "parkinsons.csv": [",", 0, None, "status"],
    "zoo.csv": [",", None, None, 17],
    "flag.csv": [",", None, None, 6],
    #  "hepatitis.csv": [",", None, None, 0], # missing values!
}

for f in list(glob.glob("../datasets/uci/*")):
    if f not in ["../datasets/uci/flag.csv"]:
        continue
    print(f)
    parsing_args = parsing_dict[os.path.basename(f)]

    df = pd.read_csv(
        f, sep=parsing_args[0], header=parsing_args[1], index_col=parsing_args[2]
    )
    if os.path.basename(f) == "PLANNING_plrx.txt":
        del df[13]
    if os.path.basename(f) == "BREAST.csv":
        del df["Unnamed: 32"]
    if os.path.basename(f) == "adult.csv":
        df["Age"].astype("int32")
    if os.path.basename(f) == "zoo.csv":
        df = df.drop(0, 1)
    if os.path.basename(f) == "parkinsons.csv":
        df = df.drop("name", 1)
    if os.path.basename(f) == "flag.csv":
        df = df.drop(0, 1)

    df = df.rename(columns={parsing_args[3]: "LABEL"})
    print(df)
    for column, dtype in df.dtypes.items():
        print(column, dtype)
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

    #  print(df)

    rf = RandomForestClassifier()
    rf.fit(df.drop("LABEL", axis=1), labelEncoder.fit_transform(df["LABEL"].values))
    #  print(os.path.basename(f))
    df.to_csv(
        "../datasets/uci_cleaned/" + os.path.basename(f).split(".")[0] + ".csv",
        index=False,
    )
