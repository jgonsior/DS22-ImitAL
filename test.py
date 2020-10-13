import pandas as pd
import glob
import os

df = pd.read_csv("../datasets/taurus_10_10/MORE_DATA/dataset_creation.csv")

print(df["fit_time"].mean())
print(df["fit_time"].median())
