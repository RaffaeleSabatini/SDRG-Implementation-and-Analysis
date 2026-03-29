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
    gamma = rnd.uniform(0, gamma_0, N)
    J     = rnd.uniform(size=N)

    h     = rnd.uniform(-h_0/2, h_0/2, N)
    mask = rnd.choice([True, False], size=N, replace=True, p=[zeta, 1-zeta])
    h[mask == False] = 0

    # Iterating decimation as long as OMEGA > thresh
    it = 0; OMEGA = 1e9; N_s = N

    while (OMEGA > thresh) and (N_s > 2):
        it += 1
        N_s = h.shape[0] # Number of sites

        error_message(h.shape[0] != gamma.shape[0], "Size mismatch between h-chain and gamma-chain")
        error_message(h.shape[0] != J.shape[0], "Size mismatch between h-chain and J-chain")
        print(f"{"="*90}\n\nIteration: {it} \t Ω: {OMEGA} \t Number of sites: {N_s}\n")
        
        kappa = np.sqrt(gamma**2 + h**2)
        parameters = np.concatenate([2*J, kappa])
        max_idx = np.argmax(parameters)
        OMEGA = parameters[max_idx]

        checkpoint(DEBUG, msg=f"OMEGA: {OMEGA} \t maximum index: {(max_idx, "[COUPLING]") if max_idx < N_s else (max_idx-N_s, "[FIELD]")}")
        checkpoint(DEBUG, msg=f"Coupling chain: {J}")
        checkpoint(DEBUG, msg=f"kappa chain: {kappa}")
        
        if max_idx < N_s:     # ---- maximum parameter is a coupling
            next_idx = (max_idx+1)%N_s
            gamma_tilde = 0
            if h_0 != 0:
                # 2-order approximation
                gamma_tilde = gamma[max_idx]*gamma[next_idx]/J[max_idx]
            else:
                # Analitic expression
                gamma_tilde = 0.5*( 
                    np.sqrt(J[max_idx]**2 + (gamma[max_idx]+gamma[next_idx])**2) -
                    np.sqrt(J[max_idx]**2 + (gamma[max_idx]-gamma[next_idx])**2)
                    )

            h_tilde = h[max_idx] + h[next_idx]

            checkpoint(DEBUG, msg=f"next_idx: {next_idx}")

            # Decimation step
            J = np.delete(J, max_idx)

            if max_idx == N_s-1:
                # when removing the last coupling, ring topology must be preserved
                gamma[0] = gamma_tilde
                gamma = np.delete(gamma, N_s-1)

                h[0] = h_tilde
                h = np.delete(h, N_s-1)
            else:
                gamma[max_idx] = gamma_tilde
                gamma = np.delete(gamma, next_idx)

                h[max_idx] = h_tilde
                h = np.delete(h, next_idx)

        else:                # ---- maximum parameter is a field
            decimated_sites += 1

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
        sites_decimation_fraction[it] = decimated_sites/it

    print(f"{"="*90}\n")
    print(f"SDRG algorithm converged with Ω = {OMEGA}.")

    return OMEGA, sites_decimation_fraction


#------------------------------------------------------------------------------------------------


def LogRandomIsing_SDRG(N, zeta, gamma_0, thresh, DEBUG):
    '''
        Performs strong disorder RG algorithm to compute the ground state of the
        Random Transverse and Longitudinal Field Ising Chain Hamiltonian.
        We assume box-distributions for the longitudinal field parameters and we assume no transverse 
        field.

        ARGUMENTS:
        ------------------------------------------------------
            N (int)        : number of sites
            zeta (float)   : fraction of sites on which the transverse field acts
            gamma_0 (float): maximum transverse field
            thresh (float) : stopping threshold for energy scale
    '''
    # Checking that input parameters are meaningful
    error_message(not isinstance(N, int), "N must be an integer")
    error_message(zeta < 0 or zeta > 1, "zeta parameter must be in [0, 1]")
    error_message(gamma_0 < 0, "gamma_0 must be non-negative")
    error_message(thresh <= 0, "threshold must be positive")
    
    checkpoint(DEBUG, msg="INIIALIZING LOG-STRONG DISORDER RG FOR ISING CHAIN")

    # Output values
    decimated_sites = 0
    sites_decimation_fraction = np.zeros(shape=N)

    # Initializing spin chain with field and coupling parameters
    zeta = -np.log(rnd.uniform(size=N))
    beta = -np.log(rnd.uniform(0, gamma_0, N))

    # Iterating decimation as long as OMEGA > thresh
    it = 0; log_OMEGA = 10; N_s = N

    while N_s > 2:
        it += 1
        N_s = zeta.shape[0] # Number of sites

        error_message(zeta.shape[0] != beta.shape[0], "Size mismatch between zeta and beta")
        print(f"{"="*90}\n\nIteration: {it} \t Ω: {log_OMEGA} \t Number of sites: {N_s}\n")

        parameters = np.concatenate([zeta, beta])
        min_idx = np.argmin(parameters)
        log_OMEGA = parameters[min_idx]

        checkpoint(DEBUG, msg=f"log(OMEGA): {log_OMEGA} \t minimum index: {(min_idx, "[COUPLING]") if min_idx < N_s else (min_idx-N_s, "[FIELD]")}")
        checkpoint(DEBUG, msg=f"zeta (-logJ) chain: {zeta}")
        checkpoint(DEBUG, msg=f"beta (-logK) chain: {beta}")


        if min_idx < N_s:     
            # ---- minimum parameter is a log-coupling (zeta)

            next_idx = (min_idx+1)%N_s
            checkpoint(DEBUG, msg=f"next_idx: {next_idx}")

            u = np.exp(zeta[min_idx] - beta[min_idx])
            v = np.exp(zeta[min_idx] - beta[next_idx])
            beta_new = beta[min_idx] + beta[next_idx] - zeta[min_idx] - np.log(2) + np.log(np.sqrt(1 + (u+v)**2) + np.sqrt(1 + (u-v)**2))

            # Decimation step
            beta[min_idx] = beta_new
            beta          = np.delete(beta, next_idx)

            if min_idx == N_s-1:
                # Topology of the chain must be preserved
                beta = np.roll(beta, 1)
            
            zeta = np.delete(zeta, min_idx)

        else:               
            # ---- minimum parameter is a field

            decimated_sites += 1

            min_idx  = min_idx-N_s
            prev_idx = min_idx-1 if min_idx > 0 else N_s-1
            next_idx = (min_idx+1)%N_s

            checkpoint(DEBUG, msg=f"prev_idx: {prev_idx}, next_idx: {next_idx}")

            # Decimation step
            zeta[prev_idx] = zeta[prev_idx] + zeta[min_idx] - beta[min_idx]
            zeta = np.delete(zeta, min_idx)

            beta = np.delete(beta, min_idx)
        
        # Save sites decimation counter
        sites_decimation_fraction[it] = decimated_sites/it

    print(f"{"="*90}\n")
    print(f"SDRG algorithm converged with Ω = {log_OMEGA}.")

    return log_OMEGA, sites_decimation_fraction

#----------------------------------------------------------------------------


def Iterated_RandomIsing_SDRG(N_iter, N, zeta, gamma_0, h_0, thresh, LOG_flag, DEBUG):
    '''
        Iterates the RandomIsing_SDRG algorithm and provides averaged values for the decimation
        fractions with relative error.
    '''
    if LOG_flag:
        results = Parallel(n_jobs=-2)(delayed(LogRandomIsing_SDRG)(N, zeta, gamma_0, thresh, DEBUG) for _ in range(N_iter))
    else:
        results = Parallel(n_jobs=-2)(delayed(RandomIsing_SDRG)(N, zeta, gamma_0, h_0, thresh, DEBUG) for _ in range(N_iter))
    
    OMEGA_list, decimation_fractions_list = zip(*results)
    decimation_fraction_matrix = np.vstack(decimation_fractions_list)

    return OMEGA_list, np.mean(decimation_fraction_matrix, axis=0), np.std(decimation_fraction_matrix, axis=0)