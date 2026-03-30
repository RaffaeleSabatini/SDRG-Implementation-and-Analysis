import numpy as np
import matplotlib.pyplot as plt

from utilities import *

#------------------------------------------------------------------------

def plot_site_decimation(site_decim_frac, N):
    '''
        Plots the fraction of decimated sites (#site decimated/#total decimations) as a function of 
        remaining sites number.
    '''

    error_message(site_decim_frac.shape[0] != N, msg=f"Size of site_decim_frac {site_decim_frac.shape[0]} is different from number of sites {N}")

    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)

    ax.scatter(np.arange(N), site_decim_frac[::-1], s=2) # site_decim_frac is ordered by iteration number!
    ax.set_xlabel(r"Remaining sites, $n$")
    ax.set_xlim(0, N)
    ax.set_ylabel(r"Site decimation fraction, $\rho$")
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    ax.grid(True)

    plt.show()