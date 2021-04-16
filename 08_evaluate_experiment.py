import glob
import numpy as np
import pandas as pd
from tabulate import tabulate
from active_learning.config.config import get_active_config
import matplotlib.pyplot as plt
import seaborn as sns

# plt.style.use('seaborn-paper')
# plt.style.use('tex')

font_size = 8

tex_fonts = {
    # Use LaTeX to write all text
    # "text.usetex": True,
    "text.usetex": False,
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


def duration_plot(df: pd.DataFrame, OUTPUT_PATH: str):
    def show_values_on_bars(axs, h_v="v", space=0.4):
        def _show_on_single_plot(ax):
            if h_v == "v":
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() / 2
                    _y = p.get_y() + p.get_height()
                    value = int(p.get_height())
                    ax.text(_x, _y, "{:,.0f}".format(value), ha="center")
            elif h_v == "h":
                for p in ax.patches:
                    _x = p.get_x() + p.get_width() + float(space)
                    _y = p.get_y() + p.get_height() - 0.15
                    value = int(p.get_width())
                    ax.text(_x, _y, "{:,.0f}".format(value), ha="left")

        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _show_on_single_plot(ax)
        else:
            _show_on_single_plot(axs)

    plt.figure(figsize=set_matplotlib_size(width, fraction=0.5))

    df_dur = df.groupby(["strategy", "dataset_id"])["duration"].mean().to_frame()
    df_dur = df_dur.reset_index()

    timout_duration = 604800
    # timout_duration = 259200

    # @TODO: use real values after new daatset mapping!
    timeout_baselines = {
        "13": ["LAL", "EER", "QUIRE", "BMDR", "SPAL"],
        "12": ["BMDR", "SPAL"],
        "16": ["EER", "QUIRE", "BMDR", "SPAL"],
    }
    for dataset_id, strategies in timeout_baselines.items():
        for strategy in strategies:
            df_dur = df_dur.append(
                {
                    "strategy": strategy,
                    "dataset_id": dataset_id,
                    "duration": timout_duration,
                },
                ignore_index=True,
            )

    df_hist_new = pd.DataFrame(columns=["strategy", "duration"])
    for strategy in df.strategy.unique():
        df_hist_new = df_hist_new.append(
            {
                "strategy": strategy,
                "duration": df_dur.loc[df_dur["strategy"] == strategy][
                    "duration"
                ].mean(),
            },
            ignore_index=True,
        )

    df_hist_new = df_hist_new.sort_values(by=["duration"])
    g = sns.barplot(
        data=df_hist_new, x="duration", y="strategy"
    )  # , ci=68, capsize=.3)#, fill=True, common_norm=False, alpha=.5, linewidth=0)
    show_values_on_bars(g, "h", 10.4)
    g.set_xscale("log")
    # g.xaxis.set_minor_locator(AutoMinorLocator(5))

    # print(g.get_xlim())
    g.set_xlim(g.get_xlim()[0], g.get_xlim()[1] + 170000)

    import matplotlib.ticker

    g.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    g.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    g.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ","))
    )

    g.xaxis.tick_bottom()
    g.set_xlabel("")
    # g.set_xlabel("duration in s (walltime)")
    g.set_ylabel("")
    g.get_figure().savefig(
        OUTPUT_PATH + "/08_performance.pdf", format="pdf", bbox_inches="tight"
    )


def plot_evaluation_ranking_table(df):
    from tabulate import tabulate

    # strategy_results = [['strategy', 'synthetic', 'breast', 'diabetes', 'fertility', 'german', 'haberman', 'heart', 'ilpd', 'ionosphere', 'pima', 'planning', 'australian', 'dwtc', 'emnist']]#,'mean', 'df_mean', 'df_mean2']]
    # strategy_results[0] = strategy_results[0][:len(df.dataset_id.unique())+1]
    strategy_results = []
    strategy_results.append(dataset_names)
    # results_df.loc['NN Single', 'fertility'] = 2

    def get_ranks(l):
        uniques = np.unique(l)
        argssorted = np.flip(np.argsort(uniques))
        mapping = dict(zip(uniques, argssorted))
        ranks = [mapping[x] for x in l]

        m2 = {}
        last_value = -1
        increase = 0
        for i, v in enumerate(sorted(ranks)):
            if last_value == v:
                increase += 1
                m2[v] = v
            else:
                m2[v] = v + increase
            last_value = v

        # ranks = [m2[x] for x in ranks]
        return ranks

    for strategy in df["strategy"].unique():
        results = [strategy]
        for dataset in df["dataset_id"].unique():
            results.append(
                "{:.40f}".format(
                    df.loc[
                        (df["dataset_id"] == dataset) & (df["strategy"] == strategy)
                    ]["f1_auc"].mean()
                )
            )

            # count how often per dataset
            # results.append("{}".format(len(df.loc[(df['dataset_id']==dataset) & (df['strategy']==strategy)])))

            results = [r if r != "nan" else -0 for r in results]

        # results += [np.mean(np.array(results[1:]).astype(np.float64))]

        # results += [df.loc[df['strategy']==strategy]['performance'].mean()]
        # results += [np.mean(df.loc[df['strategy']==strategy].groupby('dataset_id')['performance'].mean())]

        strategy_results.append(results)

    # print(strategy_results[10][2])
    # strategy_results[10][2] = 0.94
    # print(strategy_results[10])

    results_df = pd.DataFrame(strategy_results[1:], columns=strategy_results[0])
    results_df = results_df.set_index("strategy")
    results_df = results_df.replace(-1, np.NaN)
    results_df = results_df.apply(pd.to_numeric)

    results_df["mean"] = results_df.T.mean(skipna=True)

    # results_df = results_df.fillna(value=-1)
    # print(results_df)

    ranking_only_table = []
    transposed_results = list(map(list, zip(*strategy_results)))
    new_results = [transposed_results[0]]
    for row in transposed_results[1:]:
        row_sorting = get_ranks(np.array(row[1:]))
        row = row[:1] + [
            "{:.1%} ({})".format(float(a), b) for a, b in zip(row[1:], row_sorting)
        ]
        new_results.append(row)
        ranking_only_table.append(row[:1] + row_sorting)

    row = ["mean %"] + results_df["mean"].to_list()
    row_sorting = get_ranks(np.array(row[1:]))

    row = row[:1] + [
        "{:.1%} ({})".format(float(a), b) for a, b in zip(row[1:], row_sorting)
    ]
    new_results.append(row)

    ranking_only_table = [transposed_results[0]] + ranking_only_table
    new_results.append(
        ["mean (r)"]
        + [
            "{:1.2f}".format(sum(r) / len(r))
            for r in list(map(list, zip(*ranking_only_table[1:])))[1:]
        ]
    )

    # sort colmuns manually
    df2 = pd.DataFrame(new_results[1:], columns=new_results[0])
    # df2 = df2[['strategy', 'ANNSingle1010EuclidCos', 'ANNSingleCosNrf', 'ANNSingleCos', 'ANNSingleNrf','ANNBatchCosNrf', 'ANNBatchCos',  'NN Single', 'NN Batch', 'MM', 'QBC', 'LC', 'Ent',  'BMDR', 'GD', 'LAL', 'SPAL', 'EER',  'QUIRE', 'Rand']]
    # df2 = df2[['strategy', 'NN Single', 'NN Batch1', 'NN Batch', 'MM', 'LC', 'QBC', 'BMDR', 'GD',  'Ent', 'LAL', 'SPAL', 'EER', 'QUIRE', 'Rand']]
    df2 = df2[
        [
            "strategy",
            "NN Single",
            "NN Batch",
            "MM",
            "LC",
            "BMDR",
            "QBC",
            "GD",
            "Rand",
            "Ent",
            "LAL",
            "SPAL",
            "EER",
            "QUIRE",
        ]
    ]

    order = (
        ["strategy"]
        + sorted([v for v in dataset_to_id.values()], key=lambda v: v.upper())
        + ["mean %", "mean (r)"]
    )
    lol = [df2.columns.values.tolist()] + df2.values.tolist()
    lol = sorted(lol, key=lambda l: order.index(l[0]))

    with open("../paper/figures/ali_f1auc_table.tex", "w") as f:
        tex_code = tabulate(lol, headers="firstrow", tablefmt="latex_booktabs")
        splitted = [
            "\\fontseries{b}\\selectfont{" + t[1:-1] + "}" if "(0)" in t else t
            for t in tex_code.split("&")
        ]

        # for i in range(0,len(df.strategy.unique())):
        #    splitted = [t.replace("(" + str(i) + ")", "(\\textit{"+str(i)+"})") for t in splitted]

        tex_code = "&".join(splitted)
        tex_code = tex_code.replace("llllllllllllll", "L{1.4cm}cccccccccccccc")
        # tex_code = tex_code.replace("llllllllll", "lrrrrrrrrrr")
        tex_code = tex_code.replace("tabular", "tabularx")
        tex_code = tex_code.replace("begin{tabularx}", "begin{tabularx}{\linewidth}")

        tex_code = tex_code.replace("mean \%", "\\midrule \n mean \%")
        tex_code = tex_code.replace(" \\% ", "~~~")
        tex_code = tex_code.replace("\\% ", " ")
        tex_code = tex_code.replace("~~~", " \\%")
        for ind in range(0, 100):
            tex_code = tex_code.replace(" 0.0 (" + str(ind) + ")", "")
        #    tex_code = tex_code.replace(" ("+str(ind)+")", '')
        tex_code = tex_code.replace("1.36", "\\fontseries{b}\\selectfont{1.36}")
        tex_code = tex_code.replace(" \\\\", "\\\\")
        tex_code = tex_code.replace("cc", "rr")
        tex_code = tex_code.replace("strategy", "")
        f.write(tex_code)

    tabulate(lol, headers="firstrow", tablefmt="html")
    # tabulate(strategy_results, headers="firstrow", tablefmt="html")


config, parser = get_active_config(
    [
        (["--EXP1_PATH"], {}),
        (["--EXP2_PATH"], {}),
        (["--BASELINES_PATH"], {}),
    ],
    return_parser=True,
)  # type: ignore


baseline_names_mapping = {
    "QueryInstanceUncertainty{'measure': 'margin'}": "MM",
    "QueryInstanceUncertainty{'measure': 'least_confident'}": "LC",
    "QueryInstanceUncertainty{'measure': 'entropy'}": "Ent",
    "QueryInstanceGraphDensity{}": "GD",
    "QueryInstanceRandom{}": "Rand",
    "QueryInstanceQUIRE{}": "QUIRE",
    "QueryInstanceQBC{}": "QBC",
    "QureyExpectedErrorReduction{}": "EER",
    "QueryInstanceLAL{}": "LAL",
    "QueryInstanceSPAL{}": "SPAL",
    "QueryInstanceBMDR{}": "BMDR",
}


if config.EXP1_PATH:
    exp1_df = pd.read_csv(
        config.EXP1_PATH + "/05_alipy_results.csv",
        index_col=None,
    )

    # only extract the experiment results
    for baseline in baseline_names_mapping.keys():
        exp1_df = exp1_df[exp1_df["strategy"] != baseline]

    # merge strategies based on NN_BINARY_PATH
    full_strategy_names = exp1_df["strategy"].tolist()

    # extract unique NN_BINARY_PATHES
    unique_nn_binary_pathes = [
        f[f.find("NN_BINARY_PATH': '") : f.find("', 'data_storage")]
        for f in full_strategy_names
    ]

    unique_nn_binary_pathes = [
        u.replace("NN_BINARY_PATH': '", "").replace("/03_imital_trained_ann.model", "")
        for u in unique_nn_binary_pathes
    ]

    mapping = zip(full_strategy_names, unique_nn_binary_pathes)
    for full, unique in zip(full_strategy_names, unique_nn_binary_pathes):
        exp1_df["strategy"] = exp1_df["strategy"].replace(full, unique)

else:
    print("Please specify experiment!")
    exit(-1)

if config.EXP2_PATH:
    exp2_df = pd.read_csv(
        config.EXP1_PATH + "/05_alipy_results.csv",
        index_col=None,
    )

if config.BASELINES_PATH:
    baselines_df = pd.read_csv(
        config.EXP1_PATH + "/05_alipy_results.csv",
        index_col=None,
    )

    # only extract the baseline results
    for strategy_key in baselines_df["strategy"]:
        if strategy_key not in baseline_names_mapping.keys():
            baselines_df = baselines_df[baselines_df["strategy"] != strategy_key]

    baselines_df["strategy"] = baselines_df["strategy"].map(baseline_names_mapping)
else:
    print("Please specify baselines!")
    exit(-1)


print(exp1_df)
print(baselines_df)


duration_plot(pd.concat([exp1_df, baselines_df]), config.EXP1_PATH)