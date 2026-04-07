from algorithms import *
from plots import *
from utilities import *

from itertools import product

M      = int(input("Input number of samples (M):\n"))
N_list = [2**n for n in range(8, 13)]
ZETA   = 1
H0     = [2**(-3*exp) for exp in range(0,14)]
GAMMA0 = 0.93

for h0, N in product(H0, N_list):
        omega_list, decimations, magnetic_moment = RandomIsing_SDRG(M, N, GAMMA0, h0, J_0=1, zeta=ZETA, n_cores=4, DEBUG=False)
        save_results(omega_list, "log_gaps_at_gamma0C", GAMMA0, h0, N, M, 0)
        save_results(magnetic_moment, "mm_at_gamma0C", GAMMA0, h0, N, M, 0)
