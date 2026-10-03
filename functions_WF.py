from math import *
import numpy as np
import copy

import os

# This function is used to compute "absolute" counts from genotype counts:
def get_counts(n_arr, Ms):
    
    #cnt = np.zeros(Ms, dtype=int)
    
    #for j in range(Ms):
    #    cnt[j] = np.sum(n_arr == j)
    
    #return cnt
    return np.bincount(n_arr, minlength=Ms)


# This function is used to compute frequencies from genotype counts:
def get_freqs(n_arr, Ms):
    
    freqs = np.zeros(Ms)
    
    for j in range(Ms):
        freqs[j] = np.sum(n_arr == j)
    
    freqs = freqs/n_arr.size
    
    return freqs

# This function creates a "balanced partition" of N entries into M<N bins, as evenly as possible:
def balanced_partition(N, M):
    q, r = divmod(N, M)
    return [q] * (M - r) + [q + 1] * r
