from algorithms import *
from plots import *
from utilities import *

from itertools import product


dir_name = input("Insert directory name:\n")

M      = int(input("Input number of samples (M):\n"))
N      = 2048
ZETA   = 1
H0     = 0
GAMMA0 = [0.5, 0.8, 0.84, 0.85, 0.86, 0.89, 0.95, 1]

for i, gamma0 in enumerate(GAMMA0):
        print(f"\nIteration {i}/{len(GAMMA0)}")
        omega_list, decimations, magnetic_moment = RandomIsing_SDRG(M, N, gamma0, H0, J_0=1, zeta=ZETA, n_cores=4, DEBUG=False)

        save_results(magnetic_moment, f"{dir_name}/mag_moments", gamma0, H0, N, M, 0)
        save_results(omega_list, f"{dir_name}/excitations", gamma0, H0, N, M, 0)
        save_results(decimations, f"{dir_name}/decimations", gamma0, H0, N, M, 0)

