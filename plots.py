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

def plot_results(type, results_vec, N, gamma=None, h=None):
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

    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    ax.grid()

    ylabel = {
        "decimations": "site decimation fraction",
        "excitations": "log-excitation"
        }
    
    error_message(type not in ylabel.keys(), f"Possible types are {ylabel.keys()}")
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
                ms=1.5,
                c = cmap((n+1)/n_plots)
                )

        elif type=="excitations":
            # plot log-excitations
            ax.plot(
                np.arange(N),
                np.log(results[::-1]),
                "-o",
                label=fr"$\Gamma = {gamma[n]}$" if gamma_provided else n,
                ms=1.5,
                c = cmap((n+1)/n_plots)
                )
    

    ax.set_xlabel(r"Remaining sites, $n$")
    ax.set_ylabel(fr"{ylabel[type]}")
    ax.set_xlim(0, N)

    if type == "decimations":
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
    '''

    filtered_dict = {keys[1]: np.array(final_values_dict[keys]) for keys in final_values_dict if keys[0] == L}
    h_list = list(filtered_dict.keys())
    n_plots = len(h_list)

    fig, axes = plt.subplots(nrows=n_plots//3, ncols=3, dpi=200, figsize=(12,7))
    axes = axes.flatten()
    fig.suptitle(fr"Cross-over position $\Gamma_c$ for $L={L}$", weight="bold")

    if fit_parabolas:
        out_dict = {h0: 0 for h0 in h_list}

    for i, h0 in enumerate(h_list): 
        gamma = filtered_dict[h0][:, 0]
        value = np.abs(np.log(filtered_dict[h0][:, 1]))

        if fit_parabolas:
            model_f = lambda x, a, b, c: a*x**2 + b*x + c
            p_fit, _ = curve_fit(model_f, gamma, value, bounds=([-np.inf, 0, -np.inf], [0, np.inf, np.inf]))
            
            gamma_r = np.linspace(np.min(gamma), np.max(gamma), 1000)
            fitted_y = model_f(gamma_r, *p_fit)
            crit_gamma = -p_fit[1]/(2*p_fit[0])
            axes[i].vlines(crit_gamma, np.min(value), np.max(value) , label=fr"$\Gamma_c$={crit_gamma:.3f}", color="red")
            axes[i].plot(gamma_r, fitted_y)

            out_dict[h0] = crit_gamma

        axes[i].scatter(gamma, value)
        axes[i].legend()
        axes[i].set_title(fr"$\log_2 (h_0) = {np.log2(h0):.2f}$")
        axes[i].set_xlabel(r"$\Gamma_c$")
        axes[i].set_ylabel(r"$\overline{\ln \epsilon}$")
        axes[i].grid()
    
    fig.tight_layout()

    if fit_parabolas: return out_dict

#-----------------------------------------------------------------------------------------

def plot_critical_lines(critical_positions):
    '''
        Plots the crossover positions as functions of (1/ln)h_0 for different values of L
    '''
        
    fig, ax = plt.subplots(dpi=200, figsize=(12, 9))
    fig.tight_layout()

    for (L, crit_position) in critical_positions.items():
        h_list     = np.array(list(crit_position.keys()))
        gamma_list = np.array(list(crit_position.values()))

        ax.plot(1/np.abs(np.log(h_list)), gamma_list, "-o", label=f"L = {L}")

        ax.set_xlabel(r"$|1/\ln{(h_0)}|$")
        ax.set_ylabel(r"$\Gamma_c$")

    ax.legend()
    ax.grid()

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