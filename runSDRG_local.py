from algorithms import *
from plots import *
from utilities import *

from itertools import product


dir_name = input("Insert directory name:\n")

M      = int(input("Input number of samples (M):\n"))
N      = [2**n for n in range(10, 12)]
ZETA   = 1
H0     = 2**(-6)
GAMMA0 = [0.8 - (e/16)*(1.05-0.8) for e in range(1, 6)]

combinations = list(product(GAMMA0, N))
total_c = len(combinations)

for i, (gamma0, n) in enumerate(combinations):
        print(f"\nIteration {i}/{total_c}")
        omega_list, decimations, magnetic_moment = RandomIsing_SDRG(M, n, gamma0, H0, J_0=1, zeta=ZETA, n_cores=6, DEBUG=False)

        save_results(magnetic_moment, f"{dir_name}/mag_moments", gamma0, H0, n, M, 0)
        save_results(omega_list, f"{dir_name}/excitations", gamma0, H0, n, M, 0)
        save_results(decimations, f"{dir_name}/decimations", gamma0, H0, n, M, 0)

