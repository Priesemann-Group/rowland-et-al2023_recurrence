import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats
import xarray as xr
import seaborn as sns
import pandas as pd


vars_of_interest = ["ρ","μ_v", "σ_cc", "σ_cc/μ_v","R"]
vars_of_interest_latex = {
    "ρ": r"$\rho$",
    "μ_v": "$\mu_V$",
    "σ_cc": "$\sigma_\mathrm{CC}$",
    "σ_cc/μ_v": "$\sigma_\mathrm{CC}$/$\mu_V$",
    "R":"$R$"
}
vars_of_interest_latex_change = {
    "ρ": r"$\Delta \rho$" + "\n (% change)",
    "μ_v": r"$\Delta \mu_V$" + "\n (% change)",
    "σ_cc": r"$\Delta \sigma_\mathrm{CC}$" + "\n (% change)",
    "σ_cc/μ_v": r"$\Delta (\sigma_\mathrm{CC}$/$\mu_V)$" + "\n (% change)",
    "R":r"$\Delta R$" + "\n (% change)"
}


def map_reduce(data, func, dims_map, dims_reduce):
    """
    func has to take an array x as input and a tuple axis. Axis corresponds to the dims_reduce that have to be reduced by func.
    """

    def wrapper(x, **kwargs):
        # Remove tmp_dim before calling func
        return np.array(func(x[..., 0], **kwargs))[..., np.newaxis]

    return (
        data.stack({"tmp_dim": dims_map})
        .groupby("tmp_dim")
        .reduce(wrapper, dim=dims_reduce)
        .unstack("tmp_dim")
    )


def matrix_plots(
    func_ax,
    data,
    fig_or_grid,
    x_dim=None,
    y_dim=None,
    x_labels_dict=None,
    y_labels_dict=None,
    remove_xticklabels=True,
    remove_yticklabels=True,
    **kwargs,
):
    if x_dim is not None:
        x_coords = data.coords[x_dim].values
        n_x = len(x_coords)
    else:
        n_x = 1
        x_coords = [None]
    if y_dim is not None:
        y_coords = data.coords[y_dim].values
        n_y = len(y_coords)
    else:
        n_y = 1
        y_coords = [None]

    if isinstance(fig_or_grid, plt.Figure):
        grid = fig_or_grid.add_gridspec(
            n_y, n_x, **kwargs  # (like wspace=0.15, hspace=0.25)
        )
        fig = fig_or_grid
    elif isinstance(fig_or_grid, mpl.gridspec.SubplotSpec):
        grid = fig_or_grid.subgridspec(n_y, n_x, **kwargs)
        fig = fig_or_grid.get_gridspec().figure

    for y, y_coord in enumerate(y_coords):
        for x, x_coord in enumerate(x_coords):
            ax = fig.add_subplot(grid[y, x])
            choice_data = {
                d: c for d, c in [(x_dim, x_coord), (y_dim, y_coord)] if d is not None
            }
            func_ax(data.sel(choice_data), ax)
            if not x == 0:
                ax.set_ylabel("")
                if remove_yticklabels:
                    ax.set(yticklabels=[])
            elif y_coord is not None:
                label = y_coord if y_labels_dict is None else y_labels_dict[y_coord]
                ax.set_ylabel(label + "\n" + ax.get_ylabel())
            if not y == n_y - 1:
                ax.set_xlabel("")
                if remove_xticklabels:
                    ax.set(xticklabels=[])
            elif x_coord is not None:
                label = x_coord if x_labels_dict is None else x_labels_dict[x_coord]
                ax.set_xlabel(ax.get_xlabel() + "\n" + label)


def get_variable(xr_arr, variable):
    if variable == "R":
        s = xr_arr.sel(variable="var_cc").reset_coords(
            names=("variable",), drop=True
        ) / xr_arr.sel(variable="mean_var").reset_coords(
            names=("variable",), drop=True
        )
        N = 50000
        return np.sqrt(1-np.sqrt(1/(1+N * s**2)))
    if variable == "σ_cc/μ_v":
        return xr_arr.sel(variable="var_cc").reset_coords(
            names=("variable",), drop=True
        ) / xr_arr.sel(variable="mean_var").reset_coords(
            names=("variable",), drop=True
        )
    elif variable == "σ_cc":
        return xr_arr.sel(variable="var_cc").reset_coords(
            names=("variable",), drop=True
        )
    elif variable == "μ_v":
        return xr_arr.sel(variable="mean_var").reset_coords(
            names=("variable",), drop=True
        )
    elif variable == "ρ":
        return xr_arr.sel(variable="mean_rho").reset_coords(
            names=("variable",), drop=True
        )

def print_effect_size_table(res_dataset, n_fact):
    activity_type_list_tmp = ["all"] + list(res_dataset.coords["activity_type"].values)
    print()
    print(f"{'Effect size (p-value), for n_fact='+str(n_fact):^57}")
    print("        {:^16}{:^16}{:^16}".format(*activity_type_list_tmp))
    for variable in vars_of_interest:
        print(f"{variable+':':9} ", end="")
        for activity_type in activity_type_list_tmp:
            n_fact_tmp = n_fact
            if activity_type == "all":
                n_fact_tmp = 0
                activity_type = "shared"
            hit = get_variable(
                res_dataset.sel(
                    n_fact=n_fact_tmp, activity_type=activity_type, trial_type="hit"
                ),
                variable=variable,
            )
            miss = get_variable(
                res_dataset.sel(
                    n_fact=n_fact_tmp, activity_type=activity_type, trial_type="miss"
                ),
                variable=variable,
            )
            stat, pval = scipy.stats.wilcoxon(hit, miss)
            mean_hit = np.mean(hit)
            mean_miss = np.mean(miss)
            eff_size = (mean_hit - mean_miss) / (mean_miss + mean_hit) * 2
            print(f"{eff_size.values:>6.1%} ({pval:>5.1%}), ", end="")
        print()


def plot_pval(data, ax, colors=["navy", "purple"]):
    n_factors = data.coords["n_fact"]
    assert n_factors[0] == 0
    types = data.coords["activity_type"]
    for act_type, color in zip(types, colors):
        data_selected = data.sel({"activity_type": act_type})
        ax.plot(n_factors[1:], data_selected[1:], marker=".", markersize=7, color=color, clip_on=False)
        ax.scatter(1 / 2, data_selected[0], marker="D", color="black", clip_on=False)

    ax.axhline(0.05, color="black", linestyle="--")
    ax.axhline(0.01, color="black", linestyle="--")
    ax.axhline(0.001, color="black", linestyle="--")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_ylim((0.001, 1))
    ax.set_xlim((-1, max(n_factors)+1))

    ax.set_ylabel("p-val (hit vs miss)")
    ax.set_xlabel("number of factors")
    
def plot_pval_lin(data, ax, colors=["navy", "purple"]):
    n_factors = data.coords["n_fact"]
    assert n_factors[0] == 0
    types = data.coords["activity_type"]
    for act_type, color in zip(types, colors):
        data_selected = data.sel({"activity_type": act_type})
        ax.plot(n_factors[1:], data_selected[1:], marker=".", markersize=7, color=color, clip_on=False)
        if act_type=='shared':
            ax.scatter(0, data_selected[0], marker="D", color="black", clip_on=False)

    ax.axhline(0.05, color="black", linestyle="--")
    ax.axhline(0.01, color="black", linestyle="--")
    ax.axhline(0.001, color="black", linestyle="--")
    
    ax.axvline(5, color="black", linestyle="--")

    #ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_ylim((0.001, 1))
    ax.set_xlim((-1, max(n_factors)+1))

    ax.set_ylabel("p-val (hit vs miss)")
    ax.set_xlabel("number of factors")


def plot_effect(data, ax, colors=["navy", "purple"]):
    n_factors = data.coords["n_fact"]
    assert n_factors[0] == 0
    types = data.coords["activity_type"]
    for act_type, color in zip(types, colors):
        data_selected = data.sel({"activity_type": act_type}) * 100
        ax.plot(n_factors[1:], data_selected[1:], marker=".", markersize=7, color=color, clip_on=False)
        ax.scatter(1 / 2, data_selected[0], marker="D", color="black", clip_on=False)

    ax.axhline(0, color="black", linestyle="--")
    ax.set_xscale("log")
    ax.set_xlim((-1, max(n_factors)+1))
    ax.set_ylabel("effect size\n(% change)")
    ax.set_xlabel("number of factors")

def plot_effect_lin(data, ax, colors=["navy", "purple"]):
    n_factors = data.coords["n_fact"]
    assert n_factors[0] == 0
    types = data.coords["activity_type"]
    for act_type, color in zip(types, colors):
        data_selected = data.sel({"activity_type": act_type}) * 100
        ax.plot(n_factors[1:], data_selected[1:], marker=".", markersize=7, color=color, clip_on=False)
        if act_type=='shared':
            ax.scatter(0, data_selected[0], marker="D", color="black", clip_on=False)

    ax.axhline(0, color="black", linestyle="--")
    ax.axvline(5, color="black", linestyle="--")
    #ax.set_xscale("log")
    ax.set_xlim((-1, max(n_factors)+1))
    ax.set_ylabel("effect size\n(% change)")
    ax.set_xlabel("number of factors")


def figure_effect_pval(data, fig):

    data_variables = xr.concat(
        [get_variable(data, var) for var in vars_of_interest], dim="variable"
    ).assign_coords({"variable": vars_of_interest})

    i_miss = np.argwhere(data.coords["trial_type"].values == "miss").squeeze()
    i_hit = np.argwhere(data.coords["trial_type"].values == "hit").squeeze()

    def calc_pval(x, axis):
        try:
            pval = scipy.stats.wilcoxon(
                   x.take(i_hit, axis[0]), x.take(i_miss, axis[0])
            )[1]
        except:
            pval = np.nan
        return pval        

    def calc_effect(x, axis):
        hit = x.take(i_hit, axis[0])
        miss = x.take(i_miss, axis[0])
        mean_hit = np.mean(hit)
        mean_miss = np.mean(miss)
        eff_size = (mean_miss - mean_hit) / (mean_miss + mean_hit)
        return eff_size

    data_p_val = map_reduce(
        data_variables,
        calc_pval,
        dims_map=("variable", "n_fact", "activity_type", "sample"),
        dims_reduce=("trial_type", "session"),
    )
    data_effect = map_reduce(
        data_variables,
        calc_effect,
        dims_map=("variable", "n_fact", "activity_type", "sample"),
        dims_reduce=("trial_type", "session"),
    )

    grid = fig.add_gridspec(1, 2, wspace=0.5)

    matrix_plots(
        plot_effect,
        data_effect.sel(sample=0).reset_coords(drop=True),
        y_dim="variable",
        y_labels_dict=vars_of_interest_latex,
        fig_or_grid=grid[0, 0],
    )

    matrix_plots(
        plot_pval,
        data_p_val.sel(sample=0).reset_coords(drop=True),
        y_dim="variable",
        y_labels_dict=vars_of_interest_latex,
        fig_or_grid=grid[0, 1],
    )

    
def figure_effect_pval_lin(data, fig):

    data_variables = xr.concat(
        [get_variable(data, var) for var in vars_of_interest], dim="variable"
    ).assign_coords({"variable": vars_of_interest})

    i_miss = np.argwhere(data.coords["trial_type"].values == "miss").squeeze()
    i_hit = np.argwhere(data.coords["trial_type"].values == "hit").squeeze()

    def calc_pval(x, axis):
        try:
            pval = scipy.stats.wilcoxon(
                   x.take(i_hit, axis[0]), x.take(i_miss, axis[0])
            )[1]
        except:
            pval = np.nan
        return pval        

    def calc_effect(x, axis):
        hit = x.take(i_hit, axis[0])
        miss = x.take(i_miss, axis[0])
        mean_hit = np.mean(hit)
        mean_miss = np.mean(miss)
        eff_size = 2*(mean_miss - mean_hit) / (mean_miss + mean_hit)
        return eff_size

    data_p_val = map_reduce(
        data_variables,
        calc_pval,
        dims_map=("variable", "n_fact", "activity_type", "sample"),
        dims_reduce=("trial_type", "session"),
    )
    data_effect = map_reduce(
        data_variables,
        calc_effect,
        dims_map=("variable", "n_fact", "activity_type", "sample"),
        dims_reduce=("trial_type", "session"),
    )

    grid = fig.add_gridspec(1, 2, wspace=0.5)

    matrix_plots(
        plot_effect_lin,
        data_effect.sel(sample=0).reset_coords(drop=True),
        y_dim="variable",
        y_labels_dict=vars_of_interest_latex,
        fig_or_grid=grid[0, 0],
    )

    matrix_plots(
        plot_pval_lin,
        data_p_val.sel(sample=0).reset_coords(drop=True),
        y_dim="variable",
        y_labels_dict=vars_of_interest_latex,
        fig_or_grid=grid[0, 1],
    )


def plot_difference(data, ax):
    data = data.sel(sample=0)
    eff_size = 100*2*(data.sel(trial_type="miss") - data.sel(trial_type="hit")) / (
        data.sel(trial_type="miss") + data.sel(trial_type="hit")
    )

    p_val = scipy.stats.wilcoxon(
        data.sel(trial_type="miss"), data.sel(trial_type="hit")
    )[1]

    bonf_n_tests = 1

    bool_sign = p_val < (5e-2 / bonf_n_tests)

    tmp_df = pd.DataFrame(
        {
            "zscore": np.concatenate((eff_size, -1 * eff_size)),
            "trial_type": ["miss"] * len(eff_size) + ["hit"] * len(eff_size),
        }
    )
    
    ax = sns.stripplot(
        data=tmp_df,
        x="trial_type",
        y="zscore",
        ax=ax,
        s=8,
        color=("grey" if bool_sign else "white"),
        edgecolor=("k" if bool_sign else "grey"),
        linewidth=3,
    )
    ax = sns.pointplot(
        data=tmp_df,
        x="trial_type",
        y="zscore",
        ax=ax,
        s=80,
        color=("k" if bool_sign else "grey"),
        linewidth=3,
        ci=95,
    )

    ax.set_title("p = {:.3f}".format(p_val), fontsize=7)
    ax.set_xlabel("")
    ax.set_ylabel("")
    sns.despine()

    
def plot_difference_connecting_lines(data, ax):
    n_sessions = 11
    data = data.sel(sample=0)
    eff_size = 100*2*(data.sel(trial_type="miss") - data.sel(trial_type="hit")) / (
        data.sel(trial_type="miss") + data.sel(trial_type="hit")
    )

    p_val = scipy.stats.wilcoxon(
        data.sel(trial_type="miss"), data.sel(trial_type="hit")
    )[1]

    bonf_n_tests = 1

    bool_sign = p_val < (5e-2 / bonf_n_tests)

    tmp_df = pd.DataFrame(
        {
            "zscore": np.concatenate((eff_size, -1 * eff_size)),
            "trial_type": ["miss"] * len(eff_size) + ["hit"] * len(eff_size),
        }
    )
    
    tmp_df['x'] = np.random.randn(2*n_sessions) * 0.1
    
    ax.plot(tmp_df[tmp_df['trial_type']=='miss']['x'], tmp_df[tmp_df['trial_type']=='miss']['zscore'],
           '.', color='k',#('k' if bool_sign else 'grey'), 
                            markersize=10)
    ax.plot(1+tmp_df[tmp_df['trial_type']=='hit']['x'], tmp_df[tmp_df['trial_type']=='hit']['zscore'],
           '.', color='k',#('k' if bool_sign else 'grey'), 
                            markersize=10)

    ax.plot([tmp_df[tmp_df['trial_type']=='miss']['x'], 1+tmp_df[tmp_df['trial_type']=='hit']['x']],
            [tmp_df[tmp_df['trial_type']=='miss']['zscore'], tmp_df[tmp_df['trial_type']=='hit']['zscore']],
                c='k', alpha=0.7)
    
    ax.set_title("p = {:.3f}".format(p_val), fontsize=10)
    
    ax.set_xlabel('')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['miss', 'hit'])
    ax.tick_params(bottom=False)
    sns.despine()

def figure_single_nfact(data, fig, n_fact):

    data_variables = xr.concat(
        [get_variable(data, var) for var in vars_of_interest], dim="variable"
    ).assign_coords({"variable": vars_of_interest})

    data_var_acttype = xr.concat(
        [
            data_variables.sel(n_fact=0, activity_type="shared").reset_coords(
                drop=True
            ),
            data_variables.sel(n_fact=n_fact, activity_type="shared").reset_coords(
                drop=True
            ),
            data_variables.sel(n_fact=n_fact, activity_type="residual").reset_coords(
                drop=True
            ),
        ],
        dim="activity_type",
    ).assign_coords({"activity_type": ["all", "shared", "residual"]})

    i_miss = np.argwhere(data.coords["trial_type"].values == "miss").squeeze()
    i_hit = np.argwhere(data.coords["trial_type"].values == "hit").squeeze()

    matrix_plots(
        plot_difference,
        data_var_acttype,
        y_dim="variable",
        x_dim="activity_type",
        y_labels_dict=vars_of_interest_latex_change,
        fig_or_grid=fig,
        remove_xticklabels=False,
        remove_yticklabels=False,
        wspace=0.7,
        hspace=0.8,
    )
    fig.suptitle("n_fact={}".format(n_fact), x=0.02, y=0.98)
    
def figure_single_nfact_connecting_lines(data, fig, n_fact):

    data_variables = xr.concat(
        [get_variable(data, var) for var in vars_of_interest], dim="variable"
    ).assign_coords({"variable": vars_of_interest})

    data_var_acttype = xr.concat(
        [
            data_variables.sel(n_fact=0, activity_type="shared").reset_coords(
                drop=True
            ),
            data_variables.sel(n_fact=n_fact, activity_type="shared").reset_coords(
                drop=True
            ),
            data_variables.sel(n_fact=n_fact, activity_type="residual").reset_coords(
                drop=True
            ),
        ],
        dim="activity_type",
    ).assign_coords({"activity_type": ["all", "shared", "residual"]})

    i_miss = np.argwhere(data.coords["trial_type"].values == "miss").squeeze()
    i_hit = np.argwhere(data.coords["trial_type"].values == "hit").squeeze()

    matrix_plots(
        plot_difference_connecting_lines,
        data_var_acttype,
        y_dim="variable",
        x_dim="activity_type",
        y_labels_dict=vars_of_interest_latex_change,
        fig_or_grid=fig,
        remove_xticklabels=False,
        remove_yticklabels=False,
        wspace=0.7,
        hspace=0.8,
    )
    fig.suptitle("n_fact={}".format(n_fact), x=0.02, y=0.98)
