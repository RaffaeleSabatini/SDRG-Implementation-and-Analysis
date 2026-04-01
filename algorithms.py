import numpy as np
import numpy.random as rnd

from time import time
from joblib import delayed, Parallel
from utilities import *

#----------------------------------------------------------------------------------------------------------

def RandomIsing_SDRG_single_core(N_iter, N, gamma_0, h_0, J_0=1, zeta=1, DEBUG=False):
    '''
        Executes one iteration of the SDRG algorithm to collect the statistics about the decimation
        of local terms in the RTLFI Hamiltonian.

        ARGUMENTS:
        ------------------------------------------------------
            N (int)        : number of sites
            gamma_0 (float): maximum transverse field
            h_0 (float)    : maximum longitudinal field
            J_0 (float)    : maximum coupling term
            zeta (float)   : fraction of sites on which the transverse field acts
            thresh (float) : stopping threshold for energy scale
    '''

    print(f"{"="*90}\n---Executing SD renormalization algorithm on chain #{N_iter}---\n{"="*90}")

    # Output values
    omega_cache = np.zeros(shape=N)
    site_decimation_cache = np.zeros(shape=N)

    # Initializing spin chain with field and coupling parameters
    gamma = rnd.uniform(0, gamma_0, N)
    J     = rnd.uniform(0, J_0, N)
    h     = rnd.uniform(-h_0/2, h_0/2, N)

    mask  = rnd.choice([True, False], size=N, replace=True, p=[zeta, 1-zeta])
    h[mask == False] = 0

    # Iterating decimation as long as there are more than 2 sites
    it  = 0
    N_s = h.shape[0]
    while N_s > 2:
        N_s = h.shape[0] # Number of sites

        kappa = np.sqrt(gamma**2 + h**2)
        parameters = np.concatenate([J, kappa])
        max_idx = np.argmax(parameters)
        OMEGA = parameters[max_idx]

        error_message(h.shape[0] != gamma.shape[0], "Size mismatch between h-chain and gamma-chain")
        error_message(h.shape[0] != J.shape[0], "Size mismatch between h-chain and J-chain")
        checkpoint(DEBUG, f"{"="*90}\n\nIteration: {it} \t Ω: {OMEGA} \t Number of sites: {N_s}\n")

        checkpoint(DEBUG, msg=f"OMEGA: {OMEGA} \t maximum index: {(max_idx, "[COUPLING]") if max_idx < N_s else (max_idx-N_s, "[FIELD]")}")
        checkpoint(DEBUG, msg=f"Coupling chain: {J}")
        checkpoint(DEBUG, msg=f"kappa chain: {kappa}")
        
        if max_idx < N_s:     # ---- maximum parameter is a coupling
            next_idx = (max_idx+1)%N_s

            gamma_tilde = gamma[max_idx]*gamma[next_idx]/J[max_idx]
            h_tilde = h[max_idx] + h[next_idx]

            checkpoint(DEBUG, msg=f"next_idx: {next_idx}")

            # Decimation step
            J = np.delete(J, max_idx)

            gamma[max_idx] = gamma_tilde
            gamma = np.delete(gamma, next_idx)

            h[max_idx] = h_tilde
            h = np.delete(h, next_idx)

            if max_idx == N_s-1:
                # when removing first element, ring topology must be preserved
                h     = np.roll(h, 1)
                gamma = np.roll(gamma, 1)
            
            # Save sites decimation counter
            site_decimation_cache[it] = 0

        else:                # ---- maximum parameter is a field
            max_idx  = max_idx-N_s
            prev_idx = max_idx-1 if max_idx > 0 else N_s-1
            next_idx = (max_idx+1)%N_s

            checkpoint(DEBUG, msg=f"prev_idx: {prev_idx}, next_idx: {next_idx}")

            # Computing E_pp, E_pm, E_mp, E_mm
            E_pp = -np.sqrt(gamma[max_idx]**2 + (J[prev_idx] + J[max_idx] + h[max_idx])**2)
            E_pm = -np.sqrt(gamma[max_idx]**2 + (J[prev_idx] - J[max_idx] + h[max_idx])**2)
            E_mp = -np.sqrt(gamma[max_idx]**2 + (-J[prev_idx] + J[max_idx] + h[max_idx])**2)
            E_mm = -np.sqrt(gamma[max_idx]**2 + (-J[prev_idx] - J[max_idx] + h[max_idx])**2)

            # Decimation step
            J[prev_idx] = -(E_pp + E_mm - E_pm - E_mp) / 4
            J = np.delete(J, max_idx)

            gamma = np.delete(gamma, max_idx)

            h[prev_idx] += -(E_pp - E_mm + E_pm - E_mp) / 4
            h[next_idx] += -(E_pp - E_mm - E_pm + E_mp) / 4
            h = np.delete(h, max_idx)

            # Save sites decimation counter
            site_decimation_cache[it] = 1
        
        omega_cache[it] = OMEGA
        it += 1

    print(f"{"="*90}\n---SDRG algorithm executed on chain #{N_iter}---\n{"="*90}")
    return omega_cache, site_decimation_cache


#---------------------------------------------------------------------------------------------------


def RandomIsing_SDRG(M, N, gamma_0, h_0, J_0=1, zeta=1, n_cores = -2, DEBUG=False):
    '''
        Performs strong disorder RG algorithm to compute the ground state of the
        Random Transverse and Longitudinal Field Ising Chain Hamiltonian.

        This function iterates RandomIsing_SDRG_single_core and aggregates the statistics to compute
        relevant quantities.

        ARGUMENTS:
        ------------------------------------------------------
            M (int)        : number of samples to average on
            N (int)        : number of sites
            gamma_0 (float): maximum transverse field
            h_0 (float)    : maximum longitudinal field
            J_0 (float)    : maximum coupling term
            zeta (float)   : fraction of sites on which the transverse field acts
            n_cores (int)  : number of cores to parallelize execution
    '''
    # Checking that input parameters are meaningful
    error_message(not isinstance(M, int), "N must be an integer")
    error_message(not isinstance(N, int), "N must be an integer")
    error_message(zeta < 0 or zeta > 1, "zeta parameter must be in [0, 1]")
    error_message(gamma_0 < 0, "gamma_0 must be non-negative")
    error_message(J_0 < 0, "J_0 must be non-negative")
    error_message(h_0 < 0, "h_0 must be non-negative")
    
    print(f"\nSTARTED STRONG-DISORDER RG ALGORITHM FOR ISING CHAIN (GAMMA0={gamma_0} - H0={h_0} - N={N} - M={M})")
    start = time()

    results = Parallel(n_jobs=n_cores)(delayed(RandomIsing_SDRG_single_core)(N_iter, N, gamma_0, h_0, J_0, zeta, DEBUG) for N_iter in range(M))

    OMEGA_list, sites_decimation_list = zip(*results)
    
    OMEGA_matrix = np.vstack(OMEGA_list)
    excitation = np.mean(OMEGA_matrix, axis=0)

    sites_decimation_matrix = np.vstack(sites_decimation_list)
    sites_decimation_fraction = np.mean(sites_decimation_matrix, axis=0)

    end = time()
    print(f"{"="*90}\n")
    print(f"SDRG ALGORITHM EXECUTED WITH TIME {end-start} (s).")

    return excitation, sites_decimation_fraction

#----------------------------------------------------------------------------------------------------------

