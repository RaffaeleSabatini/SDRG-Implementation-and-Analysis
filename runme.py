from algorithms import *
from plots import *
from utilities import *

from itertools import product

M     = 4
N     = 2048
ZETA  = 1
H0    = np.exp([-3*k for k in range(2, 6)])

Np = 10
interpol = np.array([np.sqrt(k) / np.sqrt(Np) for k in range(0, Np+1)])
interpol = np.concatenate([0.5*interpol, 1 - interpol[::-1]*0.5])
gamma_list = 0.85 + interpol*(1.05-0.85)

for h0, gamma0 in product(H0, gamma_list):
        omega_list, decimations = RandomIsing_SDRG(M, N, gamma0, h0, J_0=1, zeta=ZETA, n_cores=4, DEBUG=False)
        save_results(omega_list, "crossover-region2", gamma0, h0, N, M, 0)
