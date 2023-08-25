import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
rc("text", usetex=True)
sns.set_style("darkgrid")


def plot_alg_cv_comparison(results, error_type, skew, heteroscedasticity, savefig=False):
    df = pd.DataFrame(results)

    df["eps"] = df["sample_contaminated"]/df["sample_size"]
    df["ag_cv"] = df[['algorithm', 'fold_K', 'selection_strategy']].apply(lambda x: '-'.join(x), axis=1)
    df.sort_values(by=["ag_cv"], inplace=True)

    blues = sns.color_palette("Blues", n_colors=4)[1:-1]
    reds = sns.color_palette("Reds", n_colors=4)[1:-1]
    grays = sns.color_palette("Greys", n_colors=4)[1:-1]

    cp = {
        'gd-maxK/V-max_slope':blues[0],
        'gd-K/V-max_slope':blues[1],
        'gd-maxK/V-min_loss':blues[0],
        'gd-K/V-min_loss':blues[1],
        'plugin-maxK/V-max_slope':reds[0],
        'plugin-K/V-max_slope':reds[1],
        'plugin-maxK/V-min_loss':reds[0],
        'plugin-K/V-min_loss':reds[1],
    }

    plt.rcParams["figure.figsize"] = (10,4)
    gs = gridspec.GridSpec(1, 9)    
    mom_plot = plt.subplot(gs[0, :6])
    tm_plot = plt.subplot(gs[0, 6:])

    mom_bp = sns.boxplot(
        df[(df["method"]=="MOM") & (df["error_type"]==error_type) & (df["skew"]==skew) & (df["heteroscedasticity"]==heteroscedasticity)],
        y="L2_dist", x="eps", hue="ag_cv", palette=cp, ax=mom_plot
    )
    mom_bp.set(
        xlabel=r"$\varepsilon$", ylabel=r"$\left\| \hat{\beta}_n - \beta^\star \right\|_{L^2}$",
    )
    
    tm_bp = sns.boxplot(
        df[(df["method"]=="TM") & (df["error_type"]==error_type) & (df["skew"]==skew) & (df["heteroscedasticity"]==heteroscedasticity)],
        y="L2_dist", x="eps", hue="ag_cv", palette=cp, ax=tm_plot
    )
    tm_bp.set(
        xlabel=r"$\varepsilon$", ylabel=None,
    )
    
    for i,thisbar in enumerate([p for p in mom_bp.patches if type(p) == mpatches.PathPatch]):
        if i%2:
            thisbar.set_hatch("///")
            
    for i,thisbar in enumerate([p for p in tm_bp.patches if type(p) == mpatches.PathPatch]):
        if i%2:
            thisbar.set_hatch("///")
            
    mom_plot.legend([], frameon=False)
    mom_plot.set_yscale("log")
    mom_plot.set_ylim(0.01, 5000)
    mom_plot.set_title("MoM")
    
    tm_plot.legend([], frameon=False)
    tm_plot.set_yscale("log")
    tm_plot.set_ylim(0.01, 5000)
    tm_plot.set_yticks([.01,.1,1,10,100,1000],["","","","","",""])
    tm_plot.set_title("TM")
    
    hs = [
        mpatches.Patch(color=blues[1], lw=1, ec="#333", label='AASD'),
        mpatches.Patch(color=reds[1], lw=1, ec="#333", label='Plug-in'),
        mpatches.Patch(color=grays[1], lw=1, ec="#333", label=r"$K' = \frac{K}{V}$"),
        mpatches.Patch(color=grays[0], lw=1, ec="#333", label=r"$K' = \max\frac{K}{V}$"),
        mpatches.Patch(color="white", lw=1, ec="#333", label='Slope'),
        mpatches.Patch(color="white", lw=1, ec="#333", hatch="///", label='Min Loss'),
    ]
    plt.figlegend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    plt.tight_layout()
    if savefig:
        plt.savefig(f"plots/comparison_alg_cv_error={error_type}_skew={skew}_h={heteroscedasticity}.pdf", bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.clf()


def make_plot_setup(results, filter_v, savefig=False, vertical_at=None, vertical_label=None):
    plt.rcParams["figure.figsize"] = (5,4)
    gs = gridspec.GridSpec(10, 1)    
    dist_plot = plt.subplot(gs[:7, 0])
    k_plot = plt.subplot(gs[7:, 0])
    # v_plot = plt.subplot(gs[5, 0])
    
    df = pd.DataFrame(results)
    df = df.loc[(df[list(filter_v)] == pd.Series(filter_v)).all(axis=1)]
    
    df["eps"] = df["sample_contaminated"]/df["sample_size"]
    df["k"] = (df["method"] == "TM")*df["best_param"]*df["sample_size"] + (df["method"] == "MOM")*df["best_param"]
    df["en"] = df["eps"]*df["sample_size"]
    df["2en1"] = 2*df["eps"]*df["sample_size"]+1
    
    df_n = df[df["method"] != "OLS"][ ["eps", "method", "L2_dist"] ].groupby(["method", "eps"]).std()
    
    epss = np.unique(df["eps"].values)
    epsd = (np.max(epss) - np.min(epss))/(epss.size-1)
    
    sns.boxplot(df, y="L2_dist", x="eps", hue="method", ax=dist_plot).set(
        xlabel=None, ylabel=r"$\left\| \hat{\beta}_n - \beta^\star \right\|_{L^2}$"
    )
    if vertical_at is not None:
        dist_plot.axvline(x=vertical_at, ymin=0, ymax=1e10, label=vertical_label, c="black", ls=":")
        dist_plot.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False)
    else:
        dist_plot.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=False)
    dist_plot.set_xticks([i for i,e in enumerate(epss)], ["" for e in epss])
    dist_plot.set_yticks([.01,.1,1,10,100,1000],["","","","","",""])
    dist_plot.set_xlim(-.5, epss.size-.5)
    dist_plot.set_ylim(0.01, 500)
    dist_plot.set_yscale("log")
    dist_plot.grid(True, axis='both')
    
    sns.scatterplot(df[df["method"] != "OLS"], y="k", x="eps", hue="method", ax=k_plot, alpha=0.03, legend=False).set(
        xlabel=r"$\varepsilon$", ylabel=r"$k$")
    k_plot.set_xticks(epss, epss, rotation=45) #.set_xticks(epss, ["" for e in epss])
    k_plot.set_yticks(k_plot.get_yticks(), k_plot.get_yticklabels(), rotation=0)
    k_plot.set_xlim(np.min(epss) - epsd/2, np.max(epss) + epsd/2)
    sns.lineplot(df, x="eps",y="en", color="black", ls="--", label=r"$n\varepsilon$", ax=k_plot)#.set(xlabel=None)
    sns.lineplot(df, x="eps",y="2en1", color="black", ls=":", label=r"$2n\varepsilon+1$", ax=k_plot)#.set(xlabel=None)
    
    k_plot.legend(loc="upper left", ncol=2, frameon=False)
    
    # sns.lineplot(df_n.sort_values(by="method", ascending=False), x="eps", y="L2_dist", hue="method", ax=v_plot).set(
    #     xlabel=r"$\varepsilon$", ylabel=r"$\sigma$"
    # )
    # v_plot.get_legend().remove()
    # v_plot.set_xticks(epss, epss, rotation=45)
    # v_plot.set_yticks(v_plot.get_yticks(), v_plot.get_yticklabels(), rotation=0)
    # v_plot.set_xlim(np.min(epss) - epsd/2, np.max(epss) + epsd/2)
    # v_plot.set_yscale("log")

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    if savefig:
        filter_name = ""
        for k,v in filter_v.items():
            filter_name += f"_{k}={v}"
        plt.savefig(f"plots/setup{filter_name}.pdf", bbox_inches='tight', pad_inches=0)
    else:
        plt.show()
    plt.clf()
    