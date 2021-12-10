import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.ticker as ticker

font_size = 8

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    # "text.usetex": False,
    "font.family": "times",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": font_size,
    "font.size": font_size,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "xtick.bottom": True,
    "figure.autolayout": True,
}

sns.set_style("white")
sns.set_context("paper")
plt.rcParams.update(tex_fonts)  # type: ignore


# https://jwalton.info/Embed-Publication-Matplotlib-Latex/
def set_matplotlib_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5 ** 0.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


width = 505.89
fig = plt.figure(figsize=set_matplotlib_size(width, fraction=0.5))


exp_results_path = "~/exp_results/"

df = pd.DataFrame(columns=["\# Synthetic Datasets", "F1-AUC"])
for i in [
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    10000,
    100000,
]:
    csv_df = pd.read_csv(
        exp_results_path + "scale_test_100000_250_" + str(i) + "/05_alipy_results.csv"
    )
    print(
        csv_df.loc[(csv_df["dataset_id"] == 0) & (csv_df["strategy_id"] == 12)][
            "f1_auc"
        ]
    )
    values = csv_df.loc[(csv_df["dataset_id"] == 0) & (csv_df["strategy_id"] == 12)][
        "f1_auc"
    ].tolist()
    df = df.append(
        pd.DataFrame(
            {
                "\# Synthetic Datasets": [str(i)[:-3] + "k" for v in values],
                "F1-AUC": values,
            }
        ),
        ignore_index=True,
    )

print(df)
a = df.loc[df["\# Synthetic Datasets"] == "1k"]["F1-AUC"].tolist()
b = df.loc[df["\# Synthetic Datasets"] == "100k"]["F1-AUC"].tolist()
# sns.histplot(a)
# sns.histplot(b)
# ax = sns.histplot(
#    hue="\# Synthetic Datasets", x="F1-AUC", data=df, multiple="dodge", shrink=0.8
# )

# plt.show()
# exit(-1)

"""
df = pd.DataFrame(columns=["\# Synthetic Datasets", "F1-AUC"])

for i in range(1, 1000):
    df = df.append(
        pd.DataFrame(
            {
                "\# Synthetic Datasets": [
                    "100k",
                    "200k",
                    "300k",
                    "400k",
                    "500k",
                    "600k",
                    "700k",
                    "800k",
                    "900k",
                    "1,000k",
                ],
                "F1-AUC": [
                    i * random.uniform(0.9, 1.1)
                    for i in [0.3, 0.4, 0.5, 0.6, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
                ],
            }
        ),
        ignore_index=True,
    )
print(df)
"""
ax = sns.boxplot(x="\# Synthetic Datasets", y="F1-AUC", data=df)

plt.savefig("09_boxplot.pdf", dpi=300, format="pdf", bbox_inches="tight")
