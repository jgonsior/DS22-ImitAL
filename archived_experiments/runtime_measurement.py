import pandas as pd

OUTPUT_DIRECTORY = "../datasets/taurus_10_10/"

df = pd.read_csv(OUTPUT_DIRECTORY + "MORE_DATA/dataset_creation.csv")
df = pd.read_csv(OUTPUT_DIRECTORY + "MORE_DATA_synthetic.csv")

print(
    "{:<30} {:>6.2f}s {:>6.2f}s".format(
        "Neural Network", df["fit_time"].mean(), df["fit_time"].median()
    )
)

for metric in [
    "random",
    "uncertainty_lc",
    "uncertainty_entropy",
    "uncertainty_max_margin",
]:
    df = pd.read_csv(
        OUTPUT_DIRECTORY + "classics/" + metric + "MORE_DATA_synthetic.csv"
    )
    print(
        "{:<30} {:>6.2f}s {:>6.2f}s".format(
            metric, df["fit_time"].mean(), df["fit_time"].median()
        )
    )
