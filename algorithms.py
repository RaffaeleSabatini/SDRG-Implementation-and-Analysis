import numpy as np
import numpy.random as rnd

from joblib import delayed, Parallel
from debugging import *

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

    # Output values
    decimated_sites = 0
    sites_decimation_fraction = np.zeros(shape=N)

    # Initializing spin chain with field and coupling parameters
    gamma_chain = rnd.uniform(0, gamma_0, N)
    J_chain     = rnd.uniform(size=N)

    h_chain     = rnd.uniform(-h_0/2, h_0/2, N)
    mask = rnd.choice([True, False], size=N, replace=True, p=[zeta, 1-zeta])
    h_chain[mask == False] = 0

    # Iterating decimation as long as OMEGA > thresh
    it = 0; OMEGA = 1e9; N_s = N

    while (OMEGA > thresh) and (N_s > 2):
        it += 1
        N_s = h_chain.shape[0] # Number of sites

        error_message(h_chain.shape[0] != gamma_chain.shape[0], "Size mismatch between h-chain and gamma-chain")
        error_message(h_chain.shape[0] != J_chain.shape[0], "Size mismatch between h-chain and J-chain")
        print(f"{"="*90}\n\nIteration: {it} \t Ω: {OMEGA} \t Number of sites: {N_s}\n")
        
        kappa_chain = np.sqrt(gamma_chain**2 + h_chain**2)
        parameters = np.concatenate([J_chain, kappa_chain])
        max_idx = np.argmax(parameters)
        OMEGA = parameters[max_idx]

        checkpoint(DEBUG, msg=f"OMEGA: {OMEGA} \t maximum index: {(max_idx, "[COUPLING]") if max_idx < N_s else (max_idx-N_s, "[FIELD]")}")
        checkpoint(DEBUG, msg=f"Coupling chain: {J_chain}")
        checkpoint(DEBUG, msg=f"kappa chain: {kappa_chain}")
        
        if max_idx < N_s:     # ---- maximum parameter is a coupling
            next_idx = (max_idx+1)%N_s
            gamma_tilde = 0
            if h_0 != 0:
                # 2-order approximation
                gamma_tilde = gamma_chain[max_idx]*gamma_chain[next_idx]/J_chain[max_idx]
            else:
                # Analitic expression
                gamma_tilde = 0.5*( 
                    np.sqrt(J_chain[max_idx]**2 + (gamma_chain[max_idx]+gamma_chain[next_idx])**2) -
                    np.sqrt(J_chain[max_idx]**2 + (gamma_chain[max_idx]-gamma_chain[next_idx])**2)
                    )

            h_tilde = h_chain[max_idx] + h_chain[next_idx]

            checkpoint(DEBUG, msg=f"next_idx: {next_idx}")

            # Decimation step
            J_chain = np.delete(J_chain, max_idx)

            if max_idx == N_s-1:
                # when removing the last coupling, ring topology must be preserved
                gamma_chain[0] = gamma_tilde
                gamma_chain = np.delete(gamma_chain, N_s-1)

                h_chain[0] = h_tilde
                h_chain = np.delete(h_chain, N_s-1)
            else:
                gamma_chain[max_idx] = gamma_tilde
                gamma_chain = np.delete(gamma_chain, next_idx)

                h_chain[max_idx] = h_tilde
                h_chain = np.delete(h_chain, next_idx)

        else:                # ---- maximum parameter is a field
            decimated_sites += 1

            max_idx  = max_idx-N_s
            prev_idx = max_idx-1 if max_idx > 0 else N_s-1
            next_idx = (max_idx+1)%N_s

            checkpoint(DEBUG, msg=f"prev_idx: {prev_idx}, next_idx: {next_idx}")

            J_tilde       = J_chain[max_idx]*J_chain[prev_idx]/kappa_chain[max_idx] * (gamma_chain[max_idx]/kappa_chain[max_idx])**2
            h_tilde_plus  = h_chain[next_idx] + J_chain[max_idx]*h_chain[max_idx]/kappa_chain[max_idx]
            h_tilde_minus = h_chain[prev_idx] + J_chain[prev_idx]*h_chain[max_idx]/kappa_chain[max_idx]

            # Decimation step
            J_chain[prev_idx] = J_tilde
            J_chain = np.delete(J_chain, max_idx)

            gamma_chain = np.delete(gamma_chain, max_idx)

            h_chain[prev_idx] = h_tilde_minus; h_chain[next_idx] = h_tilde_plus
            h_chain = np.delete(h_chain, max_idx)

        
        # Save sites decimation counter
        sites_decimation_fraction[it] = decimated_sites/it

    print(f"{"="*90}\n")
    print(f"SDRG algorithm converged with Ω = {OMEGA}.")

    return OMEGA, sites_decimation_fraction


def Iterated_RandomIsing_SDRG(N_iter, N, zeta, gamma_0, h_0, thresh, DEBUG):
    '''
        Iterates the RandomIsing_SDRG algorithm and provides averaged values for the decimation
        fractions with relative error.
    '''

    results = Parallel(n_jobs=-2)(delayed(RandomIsing_SDRG)(N, zeta, gamma_0, h_0, thresh, DEBUG) for _ in range(N_iter))
    
    OMEGA_list, decimation_fractions_list = zip(*results)
    decimation_fraction_matrix = np.vstack(decimation_fractions_list)

    return OMEGA_list, np.mean(decimation_fraction_matrix, axis=0), np.std(decimation_fraction_matrix, axis=0)