from algorithms import *
from plots import *
from utilities import *

from itertools import product

M      = int(input("Input number of samples (M):\n"))
N      = 2048  #[2**n for n in range(8, 13)]
ZETA   = 1
H0     = 0 #[2**(-3*exp) for exp in range(0,14)]
GAMMA0 = [0.5, 0.75, 0.93, 1, 1.05, 1.25, 1.5] # 0.93

for gamma0 in GAMMA0:
        omega_list, decimations, magnetic_moment = RandomIsing_SDRG(M, N, gamma0, H0, J_0=1, zeta=ZETA, n_cores=4, DEBUG=False)
        save_results(decimations, "decimations2", gamma0, H0, N, M, 0)

