import numpy as np
import matplotlib.pyplot as plt

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

#------------------------------------------------------------------------

def plot_results(type, results_vec, N, gamma=None, h=None):
    '''
        Plots the fraction of decimated sites (#site decimated/#total decimations) as a function of 
        remaining sites number.
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