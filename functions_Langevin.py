import numpy as np
from math import *
import copy
from numba import njit

import os

# ============================================================
# DRIFT (selection only)
# ============================================================
@njit(cache=True)
def drift(x_ext, s, out, N):
    """
    out[:] = x * (s - <x,s>)
    """
    M = x_ext.shape[0]
    m = M - 1
    
    s_avg = 0.0
    for i in range(m):
        s_avg += x_ext[i] * s[i]

    for i in range(m):
        # THIS:
        #xi = x_ext[i]
        #out[i] = xi * (s[i] - s_avg) + (M * xi - 1.0) / (2.0 * N)
        # OR THAT:
        out[i] = x_ext[i] * (s[i] - s_avg)


# ============================================================
# ONE TIME STEP
# ============================================================
@njit(cache=True)
def step(x_ext, s, dt, sqrt_dt_over_N,
         z, sqrt_x, drift_buf, noise_buf, tol, N):

    M = x_ext.shape[0]
    m = M - 1

    # ---- Gaussian noise ----
    for i in range(M):
        z[i] = np.random.normal()

    # ---- sqrt(x) ----
    for i in range(M):
        sqrt_x[i] = np.sqrt(x_ext[i])

    # ---- projection: proj = (sqrt_x · z) ----
    proj = 0.0
    for i in range(M):
        proj += sqrt_x[i] * z[i]

    # ---- z = z - proj * sqrt_x ----
    for i in range(M):
        z[i] -= proj * sqrt_x[i]

    # ---- noise = sqrt_x * z ----
    for i in range(M):
        noise_buf[i] = sqrt_x[i] * z[i] * sqrt_dt_over_N # this is \eta^i

    # ---- drift on first M-1 components ----
    drift(x_ext, s, drift_buf, N)

    # ---- update internal coords ----
    for i in range(m):
        x_ext[i] += drift_buf[i] * dt + noise_buf[i]

    # ---- enforce simplex last component ----
    total = 0.0
    for i in range(m):
        total += x_ext[i]

    x_ext[m] = 1.0 - total

    # ---- numerical stabilization ----
    #for i in range(M):
    #    if x_ext[i] < 1e-12:
    #        x_ext[i] = 1e-12
    #    elif x_ext[i] > 1.0:
    #        x_ext[i] = 1.0
    
    # ---- fixation/extinction ----
    #flag_renorm = False
    for i in range(M):
        if x_ext[i] < tol:
            x_ext[i] = 0.0
            #flag_renorm = True
        elif x_ext[i] > 1.0 - tol:
            x_ext[i] = 1.0
            for j in range(M):
                if j != i:
                    x_ext[j] = 0.0
            #flag_renorm = True
    
    # ---- renormalize occasionally handled outside ----
    
    # ---- renormalize if needed ----
    ssum = 0.0
    for i in range(M):
        ssum += x_ext[i]
    
    #if flag_renorm == True or abs(ssum - 1.0) > 1e-10:
    if abs(ssum - 1.0) > 1e-10:
        for i in range(M):
            x_ext[i] /= ssum

    return 0


# ============================================================
# SIMULATION OF ONE TRAJECTORY
# ============================================================
@njit(cache=True)
def simulate_one(init_x, s, dt, N, Ltot, tol):

    M = init_x.shape[0]

    x_ext = init_x.copy()

    # Buffers (NO allocation inside loop):
    z = np.zeros(M)
    sqrt_x = np.zeros(M)
    
    drift_buf = np.zeros(M - 1)
    noise_buf = np.zeros(M)

    sqrt_dt_over_N = np.sqrt(dt / N)

    for l in range(Ltot):

        step(x_ext, s, dt, sqrt_dt_over_N,
             z, sqrt_x, drift_buf, noise_buf, tol, N)

        # Occasional renormalization (cheap stabilizer):
        #if l % 10 == 0:
        #    ssum = 0.0
        #    for i in range(M):
        #        ssum += x_ext[i]
        #    for i in range(M):
        #        x_ext[i] /= ssum

        # Fixation check:
        max_i = 0
        max_val = x_ext[0]

        for i in range(1, M):
            if x_ext[i] > max_val:
                max_val = x_ext[i]
                max_i = i

        if max_val > 1.0 - tol:
            return max_i, l

    return -1, Ltot


# ============================================================
# RUN MULTIPLE REALIZATIONS
# ============================================================
@njit(cache=True)
def run_fixation(init_x, s, Nruns, dt, N, Ltot, tol):

    M = init_x.shape[0]

    counts = np.zeros(M, dtype=np.int64)

    for r in range(Nruns):

        winner, _ = simulate_one(init_x, s, dt, N, Ltot, tol)

        if winner >= 0:
            counts[winner] += 1

    return counts

"""
####### TEST
M = 2
N = 100
Nruns = 5000

Ltot = 1000000

dt = 1.0 / (5 * N)
tol = 1 / (2 * N)

init_x = np.ones(M) / M

#eps = 1e-03 # s = 0.0 does not work in Kimura's formula
#s = np.array([-0.01, 0.0]) + eps # for [A1,A2]
s = np.array([0.00001])

counts = run_fixation(init_x, s, Nruns, dt, N, Ltot, tol)

print(counts / Nruns)
"""
