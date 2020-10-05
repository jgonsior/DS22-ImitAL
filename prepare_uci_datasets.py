import os
import glob
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

parsing_dict = {
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
}

for f in list(glob.glob("../datasets/uci/*")):

    print(f)
    parsing_args = parsing_dict[os.path.basename(f)]

    df = pd.read_csv(
        f, sep=parsing_args[0], header=parsing_args[1], index_col=parsing_args[2]
    )
    if os.path.basename(f) == "PLANNING_plrx.txt":
        del df[13]
    if os.path.basename(f) == "BREAST.csv":
        del df["Unnamed: 32"]

    df = df.rename(columns={parsing_args[3]: "LABEL"})
    print(df)
    print(df.dtypes)

    for column, dtype in df.dtypes:
        if dtype not in ["int64", "float64"]:
            print(pd.get_dummies(df[column], prefix=column))

    rf = RandomForestClassifier()
    rf.fit(df.drop("LABEL", axis=1), df["LABEL"])
