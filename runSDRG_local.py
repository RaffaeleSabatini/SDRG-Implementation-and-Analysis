from algorithms import *
from plots import *
from utilities import *

from itertools import product


dir_name = input("Insert directory name:\n")

M      = int(input("Input number of samples (M):\n"))
N      = 2048
ZETA   = 1
H0     = np.exp(-6)
GAMMA0 = [0.85]
SAVE_H = True

for i, gamma0 in enumerate(GAMMA0):
        print(f"\nIteration {i+1}/{len(GAMMA0)}")
        n = N
        omega_list, decimations, magnetic_moment, h_val = RandomIsing_SDRG(M, n, gamma0, H0, J_0=1, zeta=ZETA, n_cores=6, save_h=SAVE_H, DEBUG=False)

        save_results(magnetic_moment, f"{dir_name}/mag_moments", gamma0, H0, n, M, 0)
        save_results(omega_list, f"{dir_name}/excitations", gamma0, H0, n, M, 0)
        save_results(decimations, f"{dir_name}/decimations", gamma0, H0, n, M, 0)
        save_results(h_val, f"{dir_name}/h_val", gamma0, H0, n, M, 0)

