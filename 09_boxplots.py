import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.ticker as ticker
import math

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


# width = 505.89
fig = plt.figure(figsize=(5, 5))  # set_matplotlib_size(width, fraction=0.5))


exp_results_path = "~/exp_results/"

df = pd.DataFrame(columns=["\# Synthetic Datasets", "F1-AUC"])

folder_names = [
    "~/exp_results/scale_test3_100",
    "~/exp_results/scale_test3_1000",
    "~/exp_results/scale_test3_10000",
    "~/exp_results/scale_test3_100000",
    "~/exp_results/scale_test3_1000000",
    # "~/exp_results/scale_test_100000_random_pre",
    # "~/exp_results/scale_test_100000_rf",
    # "~/exp_results/scale_test_distances",
    # "~/exp_results/scale_test_labeled",
    # "~/exp_results/scale_test_unlabeled",
    # "~/exp_results/scale_test_lc",
    # "~/exp_results/scale_test_mm",
]

for folder in folder_names:
    csv_df = pd.read_csv(folder + "/05_alipy_results.csv")

    values = csv_df.loc[(csv_df["dataset_id"] == 0) & (csv_df["strategy_id"] == 12)][
        "f1_auc"
    ].tolist()
    print(folder, ":\t", str(len(values)))
    df = df.append(
        pd.DataFrame(
            {
                "\# Synthetic Datasets": [
                    "{0:,>8}".format(folder[26:]).replace(",", " \ ") for v in values
                ],
                "F1-AUC": values,
            }
        ),
        ignore_index=True,
    )


# df.loc[df['\# Synthetic Datasets'] ==]


# ax = sns.boxplot(
#    x="\# Synthetic Datasets", y="F1-AUC", data=df, meanline=True, showmeans=True
# )
ax = sns.histplot(
    hue="\# Synthetic Datasets",
    x="F1-AUC",
    data=df,
    # multiple="dodge",
    # fill=False,
    # kde=True,
    # shrink=0.8,
    legend=True,
    element="step",
)

# ax = sns.displot(hue="\# Synthetic Datasets", x="F1-AUC", data=df, kind="kde")
# ax = sns.kdeplot(hue="\# Synthetic Datasets", x="F1-AUC", data=df)
"""for sd_title in df["\# Synthetic Datasets"].unique():
    selection = df.loc[df["\# Synthetic Datasets"] == sd_title]["F1-AUC"]
    mean = selection.mean()
    if selection.count() == 0:
        low = high = mean
    else:
        low = selection.mean() - 1.96 * selection.std() / math.sqrt(selection.count())
        high = selection.mean() + 1.96 * selection.std() / math.sqrt(selection.count())
    ax.axvline(mean, color=plt.gca().lines[-1].get_color())  # type: ignore
    ax.axvspan(low, high, alpha=0.2, color=plt.gca().lines[-1].get_color())  # type: ignore
    ax.set_xticklabels(["{:.0%}\\%".format(x) for x in ax.get_xticks()])
"""
legend = ax.get_legend()

handles = legend.legendHandles
legend.remove()
# print(handles)
ax.legend(
    labels=reversed(["1,000,000", "100,000", "10,000", "1,000", "100"]),
    handles=handles,
    loc="lower right",
    bbox_to_anchor=(1.0, -0.7),
    ncol=5,
    borderaxespad=0,
    frameon=False,
    columnspacing=0.7,
    handletextpad=0.1,
)

# plt.axvline(x=df["F1-AUC"].median(), color="blue", ls="--", lw=2.5)
# ax = sns.swarmplot(x="\# Synthetic Datasets", y="F1-AUC", data=df, color=".25", size=1)
# plt.show()
plt.savefig("09_boxplot.pdf", dpi=300, format="pdf", bbox_inches="tight")

exit(-1)
values = csv_df.loc[(csv_df["dataset_id"] == 0) & (csv_df["strategy_id"] == 12)][
    "f1_auc"
].tolist()
df = df.append(
    pd.DataFrame(
        {
            "\# Synthetic Datasets": [str(i)[:] + "k" for v in values],
            "F1-AUC": values,
        }
    ),
    ignore_index=True,
)


old_results = pd.read_csv("../supplementary_materia/source code/ALiPy/result.csv")

values = old_results.loc[(old_results["strategy_id"] == 12)]["f1_auc"].tolist()
df = df.append(
    pd.DataFrame(
        {
            "\# Synthetic Datasets": ["old_best" for v in values],
            "F1-AUC": values,
        }
    ),
    ignore_index=True,
)

values = old_results.loc[(old_results["strategy_id"] == 13)]["f1_auc"].tolist()
df = df.append(
    pd.DataFrame(
        {
            "\# Synthetic Datasets": ["old_new" for v in values],
            "F1-AUC": values,
        }
    ),
    ignore_index=True,
)


values = old_results.loc[(old_results["strategy_id"] == 77)]["f1_auc"].tolist()
df = df.append(
    pd.DataFrame(
        {
            "\# Synthetic Datasets": ["new_SD_old_eva" for v in values],
            "F1-AUC": values,
        }
    ),
    ignore_index=True,
)

values = old_results.loc[(old_results["strategy_id"] == 78)]["f1_auc"].tolist()
df = df.append(
    pd.DataFrame(
        {
            "\# Synthetic Datasets": ["new_SD_old_eva_5000" for v in values],
            "F1-AUC": values,
        }
    ),
    ignore_index=True,
)
""""old_back",
    "old_back2",
    "old_back3",
    "old_back4",
    "old_back5",
    "old_back6",
    "old_back7",
    "old_back8",
    "random",
    100,
    "100_2",
    1000,
    # 2000,
    # 3000,
    # 4000,
    # 5000,
    6000,
    7000,
    # 8000,
    # 9000,
    10000,
    100000,""",
for i in [
    "old_back_new",
    "old_back_new_200",
    "old_back_new_10000",
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
                "\# Synthetic Datasets": [str(i)[:] + "k" for v in values],
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
ax = sns.boxplot(
    x="\# Synthetic Datasets", y="F1-AUC", data=df, meanline=True, showmeans=True
)
# ax = sns.swarmplot(x="\# Synthetic Datasets", y="F1-AUC", data=df, color=".25")
plt.show()
plt.savefig("09_boxplot.pdf", dpi=300, format="pdf", bbox_inches="tight")
