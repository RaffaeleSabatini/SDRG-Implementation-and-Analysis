import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from scipy.optimize import curve_fit
from matplotlib import colormaps
from utilities import *

#------------------------------------------------------------------------


params = {
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'figure.titlesize': 16
}
plt.rcParams.update(params)

#-----------------------------------------------------------------------------------------

def plot_results(type, results_vec, N, gamma=None, title="", h_val=None, y_size=7, x_lim=None, y_label=None):
    '''
        Plots the fraction of decimated sites (#site decimated/#total decimations) as a function of 
        remaining sites number.

        ARGUMENTS:
        ------------------------------
            type (str)              : decimation fractions or excitations (energies)
            results_vec (np.array)  : array of results with shape (K, N) or (N)
            N (int)                 : chain length
            gamma (np.array)        : longitudinal fields
            h (np.array)            : transversal fields
    '''

    error_message(results_vec.shape[0] != N, msg=f"Size of {type} vector(s) ({results_vec.shape[0]}) is different from number of sites ({N})")

    fig, ax = plt.subplots(figsize=(12, y_size), dpi=200)
    ax.grid()

    ylabel_dic = {
        "decimations": "site decimation fraction",
        "excitations": r"Average Log-Energy Scale, $\overline{\ln (\varepsilon)}$"
        }
    
    error_message(type not in ylabel_dic.keys(), f"Possible types are {ylabel_dic.keys()}")
    n_plots = results_vec.shape[1] if len(results_vec.shape) == 2 else 1

    gamma_provided = np.all(gamma != None)
    if gamma_provided:
        # Sort plots in order of gamma values
        sort_idx = np.argsort(gamma)[::-1]
        gamma = gamma[sort_idx]
        results_vec = results_vec[:, sort_idx]
    
    cmap = colormaps.get_cmap("viridis")
    for n in range(n_plots):
        results = results_vec[:, n] if len(results_vec.shape) == 2 else results_vec

        if type=="decimations":
            # plot site decimation fractions
            ax.plot(
                np.arange(N),
                results[::-1],
                "-o",
                label=fr"$\Gamma = {gamma[n]}$" if gamma_provided else n,
                ms=2.5,
                c = cmap((n+1)/n_plots)
                )

        elif type=="excitations":
            # plot log-excitations
            label = fr"$\ln (h_0) = {np.log(gamma[n]):.2f}$" if gamma[n] > 0 else rf"$h_0 = {gamma[n]:.2f}$"
            ax.plot(
                np.arange(N),
                np.log(results[::-1]),
                "-o",
                label= label if gamma_provided else n,
                ms=2.5,
                c = cmap((n+1)/n_plots)
                )
            
            if h_val is not None:
                ax.scatter(
                    np.arange(N),
                    np.log(np.abs(h_val[::-1])),
                    s=2,
                    c = "red"
                )
    

    ax.set_xlabel(r"Remaining sites, $n$")
    ax.set_ylabel(fr"{ylabel_dic[type]}") if not y_label else  ax.set_ylabel(fr"{y_label}")
    if title: ax.set_title(title)

    if type == "decimations":
        right_lim = x_lim if x_lim else N
        ax.set_xlim(0, right_lim)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
    elif type == "excitations":
        ax.set_xscale("log")
    
    ax.legend()

    plt.show()

#-----------------------------------------------------------------------------------------

def plot_critial_position(omega, gamma, h):
    '''
        Plots the |log(energy scale)| of a system in function of the gamma distribution
        parameter, for fixed h-values.
        Each h value defines an axe for itself.

        ARGUMENTS:
        -------------------------------------
        ...
    '''
    unique_h = np.unique(h)

    n_plots = unique_h.shape[0]
    fig, axes = plt.subplots(nrows=n_plots//2, ncols=2, dpi=200, figsize=(12, 7))

    for i, h0 in enumerate(unique_h):
        idx = (i//2, i%2)
        gamma_new = gamma[h==h0]
        omega_new = omega[h==h0]
        order = np.argsort(gamma_new)

        axes[idx].grid()
        axes[idx].scatter(np.round(gamma_new[order], 2), np.abs(np.log(omega_new[order])))
        axes[idx].set_title(rf"Transverse field $\ln(h_0) = {np.log(h0)}$")
        axes[idx].set_xlabel(r"$\Gamma_0$")
        axes[idx].set_ylabel(r"$| \log \epsilon |$")
    
    fig.tight_layout()
    plt.show()

#-----------------------------------------------------------------------------------------

def plot_critical_positions(final_values_dict, L, fit_parabolas = False):
    '''
        Plots excitations as a function of gamma_0 for different values of h_0, after providing L.

        If fit_parabolas is True, it returns the critical value and the curvature radius of the fit.
    '''

    filtered_dict = {keys[1]: np.array(final_values_dict[keys]) for keys in final_values_dict if keys[0] == L}
    h_list = list(filtered_dict.keys())
    n_plots = len(h_list)

    fig, axes = plt.subplots(nrows=n_plots//3, ncols=3, dpi=200, figsize=(12,7))
    axes = axes.flatten()
    fig.suptitle(fr"Cross-over position $\Gamma_0^c$ for $L={L}$", weight="bold")

    if fit_parabolas:
        out_dict = {h0: 0 for h0 in h_list}
        radius_dict = {h0: 0 for h0 in h_list}

    for i, h0 in enumerate(h_list): 
        gamma = filtered_dict[h0][:, 0]
        value = np.abs(np.log(filtered_dict[h0][:, 1]))

        if fit_parabolas:
            model_f = lambda x, a, b, c: a*x**2 + b*x + c
            p_fit, _ = curve_fit(model_f, gamma, value, bounds=([-np.inf, 0, -np.inf], [0, np.inf, np.inf]))
            
            gamma_r = np.linspace(np.min(gamma), np.max(gamma), 1000)
            fitted_y = model_f(gamma_r, *p_fit)

            crit_gamma = -p_fit[1]/(2*p_fit[0])
            curvature_radius = (1 + (2*p_fit[0]*crit_gamma + p_fit[1])**2)**1.5 / np.abs(p_fit[0])

            axes[i].vlines(crit_gamma, np.min(value), np.max(value) , label=fr"$\Gamma_0^c$={crit_gamma:.3f}", color="red")
            axes[i].plot(gamma_r, fitted_y)

            out_dict[h0] = crit_gamma
            radius_dict[h0] = curvature_radius

        axes[i].scatter(gamma, value)
        axes[i].legend()
        axes[i].set_title(fr"$\log_2 (h_0) = {np.log2(h0):.2f}$")
        axes[i].set_xlabel(r"$\Gamma_0$")
        axes[i].set_ylabel(r"$|\overline{\ln \epsilon}|$")
        axes[i].grid()
    
    fig.tight_layout()

    if fit_parabolas: 
        return out_dict, radius_dict

#-----------------------------------------------------------------------------------------

def plot_critical_lines(critical_positions, ylabel="", square_y=True, linear_fit = False):
    '''
        Plots the crossover positions as functions of 1/|ln(h_0)| for different values of L
    '''
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(critical_positions)))

    for i, (L, crit_position) in enumerate(sorted(critical_positions.items())):
        h_vals = np.array(list(crit_position.keys()))
        y_vals = np.array(list(crit_position.values()))

        if square_y: y_vals = np.sqrt(y_vals)
        
        x_vals = 1 / np.abs(np.log(h_vals))
        sort_idx = np.argsort(x_vals)

        if linear_fit:
            model_f = lambda x, a, b: a*x + b
            p_fit, _ = curve_fit(model_f, x_vals, y_vals)

            x_ran = np.linspace(np.min(x_vals), np.max(x_vals), 1000)
            fitted_y = model_f(x_ran, *p_fit)

            ax.plot(
                x_ran, fitted_y, "--", color=colors[i]
            )

            ax.scatter(x_vals[sort_idx], y_vals[sort_idx], 
                    label=f"$L = {L}$", 
                    color=colors[i],
                    s=50)
            
            pred_y = model_f(x_vals, *p_fit)
            chi_sq = np.sum((pred_y - y_vals)**2)
            ndof = len(x_vals) - len(p_fit)
            print(f"Chi squared for L={L}: {chi_sq/ndof:.3e}")
        else: 
            ax.plot(x_vals[sort_idx], y_vals[sort_idx], 
                    "-o", 
                    label=f"$L = {L}$", 
                    color=colors[i],
                    markersize=6, 
                    linewidth=1.5,
                    markerfacecolor='white', 
                    markeredgewidth=1.5)

    # label e titoli 
    ax.set_xlabel(r"Inv. log-field strength, $\left| \ln(h_0) \right|^{-1}$", fontsize=16)
    ax.set_ylabel(
        r"cross-over region's center, $\Gamma_0^c$" if not ylabel else ylabel,
        fontsize=16)
    
    # bordi
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)    # Rimuove bordi superflui
    ax.spines['right'].set_visible(False)
    
    # Legenda
    ax.legend(title="Chain Length", frameon=True, shadow=False, loc='best')

    plt.tight_layout()


#-----------------------------------------------------------------------------------------

def plot_scaling_behaviour(mm_array, h0_array, L_array):
    '''
        Plots the magnetic moments as a function of h_0 for different values of L (chain length)
    '''

    n_plots = L_array.shape[0]
    n_points = h0_array.shape[0]
    error_message(mm_array.shape[0] != n_points, msg = f"Expected number of points ({mm_array.shape[0]}) is different from {n_points}")
    error_message(mm_array.shape[1] != n_plots, msg = f"Expected number of plots ({mm_array.shape[1]}) is different from {n_plots}")
    
    fig, axes = plt.subplots()
    axes.grid()

    for plot in range(n_plots):
        axes.scatter(h0_array, mm_array[:, plot])



#-----------------------------------------------------------------------------------------

def plot_analysics_at_critical_point(results, ylabel="", log_abs = False):
    N_list      = results.keys()
    fig, axes = plt.subplots(dpi=200, figsize=(12, 7))
    cmaps = colormaps.get_cmap("viridis")

    for i, N in enumerate(N_list):
        result_matrix = np.array(results[N])

        x_val = np.log(result_matrix[:, 0])
        y_val = np.abs(np.log(result_matrix[:, 1])) if log_abs else result_matrix[:, 1] 
        sorter = np.argsort(x_val)

        axes.plot(x_val[sorter], y_val[sorter], 'o-', label = N, color = cmaps((i+1)/len(N_list)))
    
    axes.legend(title="N:")
    axes.grid()
    axes.set_xlabel(fr"$\ln (h_0)$")
    #axes.set_xscale("log")
    axes.set_ylabel(ylabel)

    fig.tight_layout()