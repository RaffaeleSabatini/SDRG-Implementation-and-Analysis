import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy.random as rnd
from colorama import Fore

#------------------------------------------------------------------------------------------------

def checkpoint(DEBUG, msg="", col=""):
    if DEBUG:
        if msg != "":
            final_msg = Fore.GREEN + "Checkpoint: "
            final_msg += msg
        else:
            final_msg = Fore.Green + "Checkpoint!"
        print(final_msg)

def error_message(condition, msg=""):
    if condition:
        if msg != "":
            error = Fore.RED + f"Error: {msg}"
        else:
            error = Fore.RED + "Error!"
        print(error)

#------------------------------------------------------------------------------------------------

def RandomIsing_SDRG(N, zeta, gamma_0, h_0, thresh, DEBUG):
    '''
        Performs strong disorder RG algorithm to compute the ground state of the
        Random Transverse and Longitudinal Field Ising Chain Hamiltonian.
        We assume box-distributions for the longitudinal and transverse field parameters.

        ARGUMENTS:
        ------------------------------------------------------
            N (int)        : number of sites
            zeta (float)   : fraction of sites on which the transverse field acts
            gamma_0 (float): maximum transverse field
            h_0 (float)    : maximum longitudinal field
            thresh (float) : stopping threshold for energy scale
    '''
    # Checking that input parameters are meaningful
    error_message(not isinstance(N, int), "N must be an integer")
    error_message(zeta < 0 or zeta > 1, "zeta parameter must be in [0, 1]")
    error_message(gamma_0 < 0, "gamma_0 must be non-negative")
    error_message(h_0 < 0, "h_0 must be non-negative")
    error_message(thresh <= 0, "threshold must be positive")
    
    checkpoint(DEBUG, msg="INIIALIZING STRONG DISORDER RG FOR ISING CHAIN")

    # Initializing spin chain with field and coupling parameters
    gamma_chain = rnd.uniform(0, gamma_0, N)
    J_chain = rnd.uniform(size=N)

    h_chain = rnd.uniform(-h_0/2, h_0/2, N)
    mask = rnd.choice([True, False], size=N, replace=True, p=[zeta, 1-zeta])
    h_chain[mask == False] = 0

    # Iterating decimation as long as 
    it = 1; error = 1e9; N_s = N
    while error > thresh:
        print(f"Iteration {it}: \t error: {error} \t Number of sites: {N_s}")
        it += 1


