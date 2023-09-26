import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns


def plot_cv_comparison(results, error_type, skew, heteroscedasticity,
                       savefig=False, fix=("algorithm", "admm"), vary=("method",("MOM", "TM")), titles=("MoM", "TM")):
    df = pd.DataFrame(results)

    df = df[df[fix[0]] == fix[1]]

    df["eps"] = df["sample_contaminated"]/df["sample_size"]
    df["ag_cv"] = df[['cv_strategy', 'beta_strategy']].apply(lambda x: '-'.join(x), axis=1)
    df.sort_values(by=["ag_cv"], inplace=True)

    blues = sns.color_palette("Blues", n_colors=4)[1:-1]
    reds = sns.color_palette("Reds", n_colors=4)[1:-1]

    cp = {
        'max_slope-best':blues[1],
        'max_slope-last':blues[1],
        'min_loss-best':reds[1],
        'min_loss-last':reds[1],
    }

    plt.clf()
    rc("text", usetex=True)
    sns.set_style("darkgrid")
    plt.rcParams["figure.figsize"] = (9,3)
    gs = gridspec.GridSpec(1, 2)    
    mom_plot = plt.subplot(gs[0, 0])
    tm_plot = plt.subplot(gs[0, 1])

    mom_bp = sns.boxplot(
        df[(df[vary[0]]==vary[1][0]) & (df["error_type"]==error_type) & (df["skew"]==skew) & (df["heteroscedasticity"]==heteroscedasticity)],
        y="L2_dist", x="eps", hue="ag_cv", ax=mom_plot, palette=cp, linewidth=0.5
    )
    mom_bp.set(
        xlabel=r"$\varepsilon$", ylabel=r"$\left\| \hat{\beta}_n - \beta^\star \right\|_{L^2}$",
    )
    
    tm_bp = sns.boxplot(
        df[(df[vary[0]]==vary[1][1]) & (df["error_type"]==error_type) & (df["skew"]==skew) & (df["heteroscedasticity"]==heteroscedasticity)],
        y="L2_dist", x="eps", hue="ag_cv", ax=tm_plot, palette=cp, linewidth=0.5
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
    mom_plot.set_xticks(mom_plot.get_xticks(), mom_plot.get_xticklabels(), rotation=45)
    mom_plot.set_yscale("log")
    mom_plot.set_ylim(0.05, 1000)
    mom_plot.set_title(titles[0])
    
    tm_plot.legend([], frameon=False)
    tm_plot.set_xticks(tm_plot.get_xticks(), tm_plot.get_xticklabels(), rotation=45)
    tm_plot.set_yscale("log")
    tm_plot.set_ylim(0.05, 1000)
    tm_plot.set_yticks([.1,1,10,100,1000],["","","","",""])
    tm_plot.set_title(titles[1])
    
    hs = [
        mpatches.Patch(color=blues[1], lw=1, ec="#333", label='Max slope'),
        mpatches.Patch(color=reds[1], lw=1, ec="#333", label='Min loss'),
        mpatches.Patch(color="white", lw=1, ec="#333", label='Best beta'),
        mpatches.Patch(color="white", lw=1, ec="#333", hatch="///", label='Last beta'),
    ]
    plt.figlegend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=4, frameon=False)
    if savefig:
        plt.savefig(f"plots/comparison_alg_cv_{fix[0]}={fix[1]}_error={error_type}_skew={skew}_h={heteroscedasticity}.pdf", bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


def make_plot_setup_error(results, filter_v, savefig=False, vertical_at=None, vertical_label=None, ylims=(0.05, 500), figsize=(5,3)):
    plt.clf()
    rc("text", usetex=True)
    sns.set_style("darkgrid")
    plt.rcParams["figure.figsize"] = figsize
    gs = gridspec.GridSpec(1, 1)    
    dist_plot = plt.subplot(gs[:, :])
    
    df = pd.DataFrame(results)
    df['method'] = pd.Categorical(df['method'], ["TM", "MOM", "OLS"])
    df.sort_values("method", inplace=True)
    df = df.loc[(df[list(filter_v)] == pd.Series(filter_v)).all(axis=1)]
    
    df["eps"] = df["sample_contaminated"]/df["sample_size"]    
    epss = np.unique(df["eps"].values)
    
    sns.boxplot(df, y="L2_dist", x="eps", hue="method", ax=dist_plot, linewidth=0.5).set(
        xlabel=r"$\varepsilon$", ylabel=r"$\left\| \hat{\beta}_n^\varepsilon - \beta^\star \right\|_{L^2}$"
    )
    if vertical_at is not None:
        dist_plot.axvline(x=vertical_at, ymin=ylims[0], ymax=ylims[1], c="black", ls=":")
        dist_plot.text(vertical_at + .25, ylims[1]/3, vertical_label)
        
    dist_plot.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    dist_plot.set_yticks([.01,.1,1,10,100,1000],["","","","","",""])
    dist_plot.tick_params(axis='x', rotation=45)
    dist_plot.set_xlim(-.5, epss.size-.5)
    dist_plot.set_ylim(ylims[0], ylims[1])
    dist_plot.set_yscale("log")
    dist_plot.grid(True, axis='both')

    if savefig:
        filter_name = f"_{df.algorithm.values[0]}"
        for k,v in filter_v.items():
            filter_name += f"_{k}={v}"
        plt.savefig(f"plots/setup_error{filter_name}.pdf", bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


def make_plot_setup_parameter(results, filter_v, savefig=False, figsize=(5,2)):
    plt.clf()
    rc("text", usetex=True)
    sns.set_style("darkgrid")
    plt.rcParams["figure.figsize"] = figsize
    gs = gridspec.GridSpec(1, 1)    
    k_plot = plt.subplot(gs[:, :])
    
    df = pd.DataFrame(results)
    df = df[df["method"] != "OLS"]
    df['method'] = pd.Categorical(df['method'], ["TM", "MOM"])
    df.sort_values("method", inplace=True)
    df = df.loc[(df[list(filter_v)] == pd.Series(filter_v)).all(axis=1)]
    
    df["eps"] = df["sample_contaminated"]/df["sample_size"]
    df["k"] = (df["method"] == "TM")*df["best_param"]*df["sample_size"] + (df["method"] == "MOM")*df["best_param"]
    df["en"] = df["eps"]*df["sample_size"]
    df["2en1"] = 2*df["eps"]*df["sample_size"]+1
        
    epss = np.unique(df["eps"].values)    
    sns.lineplot(df, y="k", x="eps", hue="method", ax=k_plot, err_style="band", errorbar="sd").set(
        xlabel=r"$\varepsilon$", ylabel=r"$k$")
    k_plot.tick_params(axis='x', rotation=90)
    k_plot.set_ylim(-50, 350)
    k_plot.set_xticks([eps for eps in epss],[fr"${eps}$" for eps in epss])
    k_plot.set_yticks([0, 150, 300],[r"$0$", r"$150$", r"$300$"])
    sns.lineplot(df, x="eps",y="en", color="black", ls="--", label=r"$n\varepsilon$", ax=k_plot)#.set(xlabel=None)
    sns.lineplot(df, x="eps",y="2en1", color="black", ls=":", label=r"$2n\varepsilon+1$", ax=k_plot)#.set(xlabel=None)
    k_plot.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False)

    if savefig:
        filter_name = f"_{df.algorithm.values[0]}"
        for k,v in filter_v.items():
            filter_name += f"_{k}={v}"
        plt.savefig(f"plots/setup_parameter{filter_name}.pdf", bbox_inches='tight', pad_inches=0)
    else:
        plt.show()