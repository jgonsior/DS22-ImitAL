import os
import glob
import pandas as pd

for f in list(glob.glob("../datasets/uci/*")):
    print(f)
    df = pd.read_csv(f)
    print(df)
