import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utilities import *

#------------------------------------------------------------------------


params = {
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 12,
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
        sort_idx = np.argsort(gamma)
        gamma = gamma[sort_idx]
        results_vec = results_vec[:, sort_idx]
    
    for n in range(n_plots):
        results = results_vec[:, n] if len(results_vec.shape) == 2 else results_vec

        if type=="decimations":
            # plot site decimation fractions
            ax.scatter(np.arange(N), results[::-1], s=2, label=fr"$\Gamma = {gamma[n]}$" if gamma_provided else n)

        elif type=="excitations":
            # plot log-excitations
            ax.scatter(np.arange(N), np.log(results[::-1]), s=2, label=fr"$\Gamma = {gamma[n]}$" if gamma_provided else n)
    

    ax.set_xlabel(r"Remaining sites, $n$")
    ax.set_ylabel(fr"{ylabel[type]}")

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