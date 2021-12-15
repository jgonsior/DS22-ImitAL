import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import matplotlib.ticker as ticker
import math
from matplotlib.patches import Rectangle
from sklearn.metrics import auc

font_size = 10

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
size = 2.3
fig = plt.figure(figsize=(3.4, 2))

data = pd.DataFrame(
    {
        "A": [0.2, 0.4, 0.5, 0.7, 0.6, 0.7],
        "B": [0.1, 0.35, 0.45, 0.3, 0.55, 0.7],
        "C": [0.1, 0.3, 0.4, 0.4, 0.5, 0.5],
        "\% labelled data": [0, 5, 10, 15, 20, 25],
    },
)
data = data.set_index("\% labelled data")
print(data)

for t in ["A", "B", "C"]:
    print(t)
    area = auc(data.index, data[t])
    max_area = auc(data.index, [1, 1, 1, 1, 1, 1])
    print(area / max_area)


ax = sns.lineplot(data=data, legend=False)
ax.set_ylim(0, 1)
ax.set(ylabel="F1-Score")
plt.savefig("10_lc.png", dpi=300, format="png", bbox_inches="tight")

colors = sns.color_palette("tab10").as_hex()
print(colors)
linestyles = ["solid", "dashed", "dotted"]

for index, title in enumerate(["A", "B", "C"]):
    plt.clf()

    plt.stackplot(data.index, data[title], colors=colors[index], alpha=0.5)
    plt.plot(data.index, data[title], color=colors[index], linestyle=linestyles[index])

    ax = plt.gca()
    ax.tick_params(color="white", labelcolor="white")
    ax.set(ylabel="F1-Score", xlabel="\% labelled data")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.set_ylim(0, 1)
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    plt.savefig("10_lc_" + title + ".png", dpi=300, format="png", bbox_inches="tight")

plt.clf()

plt.stackplot(data.index, [1, 1, 1, 1, 1, 1], colors="black")

ax = plt.gca()
ax.tick_params(color="white", labelcolor="white")
ax.set(ylabel="F1-Score", xlabel="\% labelled data")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.set_ylim(0, 1)
for spine in ax.spines.values():
    spine.set_edgecolor("white")
plt.savefig("10_lc_dark.png", dpi=300, format="png", bbox_inches="tight")
