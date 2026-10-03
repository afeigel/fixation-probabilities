import numpy as np
from collections import Counter, defaultdict
from fractions import Fraction
from math import *
import copy

import os
import pickle

# \Sop
def selection_partitions(partition):
    """
    Apply the selection operator S to a partition.

    Returns
    -------
    dict
        Mapping partition -> coefficient.
    """
    counts = Counter(partition)
    ell = len(partition)
    results = {}

    # Positive term: (ell + 1) * (lambda union 1)
    new_counts = counts.copy()
    new_counts[1] += 1
    part = tuple(
        x
        for x in sorted(new_counts, reverse=True)
        for _ in range(new_counts[x])
    )
    results[part] = ell + 1

    # Negative terms grouped by distinct part size
    for a, m in counts.items():
        new_counts = counts.copy()
        new_counts[a] -= 1
        if new_counts[a] == 0:
            del new_counts[a]
        new_counts[a + 1] += 1

        part = tuple(
            x
            for x in sorted(new_counts, reverse=True)
            for _ in range(new_counts[x])
        )

        results[part] = results.get(part, 0) - m

        # Remove zero coefficients if cancellation occurs
        if results[part] == 0:
            del results[part]

    return results


# \Mop
# Input: sorted partition
def merge_partitions(partition):
    """
    Apply one pair-merging step to an integer partition.

    Parameters
    ----------
    partition : iterable of int
        Partition represented as a sequence of positive integers.
        Order does not matter. Example: (2, 1, 1).

    Returns
    -------
    dict
        Maps each resulting partition (stored as a tuple in descending order)
        to its multiplicity.

    Example
    -------
    >>> merge_partitions((2, 1, 1))
    {(3, 1): 2, (2, 2): 1}
    """
    # Count multiplicities of each part size
    counts = Counter(partition)

    # Work with sorted distinct part sizes for deterministic output
    values = sorted(counts)

    # Canonical sorted input partition (descending)
    #base = tuple(sorted(partition, reverse=True))
    
    #debug
    #print(values)

    results = {}

    # Precompute list version once
    for i, a in enumerate(values):
        ma = counts[a]

        # Case 1: merge two equal parts a+a
        if ma >= 2:
            mult = ma * (ma - 1) // 2

            new_counts = counts.copy()
            new_counts[a] -= 2
            if new_counts[a] == 0:
                del new_counts[a]
            new_counts[2 * a] += 1

            # Reconstruct partition in descending order
            new_partition = tuple(
                x
                for x in sorted(new_counts, reverse=True)
                for _ in range(new_counts[x])
            )

            results[new_partition] = mult

        # Case 2: merge distinct parts a+b
        for b in values[i + 1:]:
            mb = counts[b]
            mult = ma * mb

            new_counts = counts.copy()
            new_counts[a] -= 1
            if new_counts[a] == 0:
                del new_counts[a]
            new_counts[b] -= 1
            if new_counts[b] == 0:
                del new_counts[b]
            new_counts[a + b] += 1

            # Reconstruct partition in descending order
            new_partition = tuple(
                x
                for x in sorted(new_counts, reverse=True)
                for _ in range(new_counts[x])
            )

            results[new_partition] = mult

    return results

# ============================================================
# Example usage
# ============================================================

#print(merge_partitions((6,5,5,4,3,2,2,1,1,1,1,1,1,1)))
#print(merge_partitions((6,6)))

##############################################################

# ============================================================
# Global caches
# ============================================================

# Cache for single-partition merge actions:
#   partition -> {new_partition: multiplicity}
MERGE_CACHE = {}

# Cache for computed Y_lambda:
#   partition -> {partition: rational coefficient}
Y_CACHE = {}


# ============================================================
# Utility routines
# ============================================================

def canonical_partition(partition):
    """
    Return partition as a canonical tuple in descending order.
    """
    return tuple(sorted((x for x in partition if x > 0), reverse=True))


def counts_to_partition(counts):
    """
    Convert Counter -> canonical partition tuple.
    """
    return tuple(
        x
        for x in sorted(counts, reverse=True)
        for _ in range(counts[x])
    )


def add_scaled_dict(dest, src, scale):
    """
    dest += scale * src
    """
    if scale == 0:
        return

    for part, coeff in src.items():
        new_coeff = dest.get(part, Fraction(0)) + scale * coeff
        if new_coeff:
            dest[part] = new_coeff
        elif part in dest:
            del dest[part]


# ============================================================
# Efficient single-partition merge operator
# ============================================================

def merge_partition(partition):
    """
    Compute M(partition) as a dictionary:
        resulting_partition -> multiplicity

    Uses multiplicity formulas:
      a != b : m_a m_b
      a == b : binom(m_a, 2)

    Cached globally.
    """
    partition = canonical_partition(partition)

    if partition in MERGE_CACHE:
        return MERGE_CACHE[partition]

    counts = Counter(partition)
    values = sorted(counts)   # ascending distinct part sizes

    results = {}

    for i, a in enumerate(values):
        ma = counts[a]

        # Merge a + a
        if ma >= 2:
            mult = ma * (ma - 1) // 2

            new_counts = counts.copy()
            new_counts[a] -= 2
            if new_counts[a] == 0:
                del new_counts[a]
            new_counts[2 * a] += 1

            new_part = counts_to_partition(new_counts)
            results[new_part] = results.get(new_part, 0) + mult

        # Merge a + b, b > a
        for b in values[i + 1:]:
            mb = counts[b]
            mult = ma * mb

            new_counts = counts.copy()

            new_counts[a] -= 1
            if new_counts[a] == 0:
                del new_counts[a]

            new_counts[b] -= 1
            if new_counts[b] == 0:
                del new_counts[b]

            new_counts[a + b] += 1

            new_part = counts_to_partition(new_counts)
            results[new_part] = results.get(new_part, 0) + mult

    MERGE_CACHE[partition] = results
    return results


# ============================================================
# Merge operator on weighted partition dictionaries
# ============================================================

def apply_merge(weighted_partitions):
    """
    Apply M to a sparse linear combination of partitions.

    Input:
        {partition: coefficient}

    Output:
        {partition: coefficient}
    """
    out = {}

    for part, coeff in weighted_partitions.items():
        if coeff == 0:
            continue

        for new_part, mult in merge_partition(part).items():
            new_coeff = out.get(new_part, Fraction(0)) + coeff * mult
            if new_coeff:
                out[new_part] = new_coeff
            elif new_part in out:
                del out[new_part]

    return out


# ============================================================
# Compute Y_lambda using the F_j recursion
# ============================================================

def Y_lambda(partition):
    """
    Compute Y_lambda as a sparse dictionary:
        partition -> rational coefficient

    Uses:
        F_ell = p_lambda
        F_j = -2 / ((ell-j)(ell+j+1)) * M F_{j+1}

    Global memoization avoids recomputing previously requested Y_lambda.
    """
    partition = canonical_partition(partition)

    if partition in Y_CACHE:
        return Y_CACHE[partition]

    ell = len(partition)

    # F_current = F_{j+1}, initialized with F_ell
    F_current = {partition: Fraction(1)}

    # Y = sum_q F_{ell-q}
    Y = dict(F_current)

    # Descend: j = ell-1, ..., 1
    for j in range(ell - 1, 0, -1):
        factor = Fraction(-2, (ell - j) * (ell + j + 1))

        # Compute M F_{j+1}
        merged = apply_merge(F_current)

        # F_j
        F_next = {
            part: factor * coeff
            for part, coeff in merged.items()
            if coeff != 0
        }

        # Accumulate into Y
        add_scaled_dict(Y, F_next, Fraction(1))

        # Advance
        F_current = F_next

        if not F_current:
            break

    Y_CACHE[partition] = Y
    return Y


# ============================================================
# Optional closed-form implementation using M^q
# (mostly for verification)
# ============================================================

def Y_lambda_closed(partition):
    """
    Compute Y_lambda from the closed formula:
        Y = sum_q (-2)^q (2 ell - q)! / (q! (2 ell)!) M^q p_lambda
    """
    partition = canonical_partition(partition)
    ell = len(partition)

    current = {partition: Fraction(1)}
    Y = {}

    for q in range(ell):
        prefactor = Fraction(
            ((-2) ** q) * factorial(2 * ell - q),
            factorial(q) * factorial(2 * ell)
        )

        add_scaled_dict(Y, current, prefactor)

        if q < ell - 1:
            current = apply_merge(current)
            if not current:
                break

    return Y


# ============================================================
# Pretty printer
# ============================================================

def format_partition(part):
    return "(" + ",".join(map(str, part)) + ")"


def print_weighted_dict(d):
    for part in sorted(d, reverse=True):
        print(f"{format_partition(part)} : {d[part]}")


# ============================================================
# Example usage
# ============================================================


#lam = (10, 7, 7, 5, 4, 4, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1, 1)
#lam = (1,1,1)

#Y = Y_lambda(lam)
#print("Y_(...) =")
#print_weighted_dict(Y)

# Verification
#assert Y == Y_lambda_closed(lam)

##############################################################

# ============================================================
# Q_k recursion in the Y-basis
#
# Requires the previously defined routines:
#   - canonical_partition
#   - add_scaled_dict
#   - selection_partitions(partition)
#   - Y_lambda(partition)
#
# Global caches:
#   - Y_CACHE
#
# This code computes
#     Q_k = sum_{lambda ⊢ k} q_lambda^(k) Y_lambda
#
# using:
#     q_eta^(k+1)
#       = 2/[l(η)(l(η)+1)] * [Y_eta] S Q_k
#
# ============================================================

from collections import defaultdict
from fractions import Fraction

# Cache for Q_k in the Y basis:
#   k -> {partition: coefficient}
Q_CACHE = {
    0: {(): Fraction(1)}   # Q_0 = 1 = Y_()
}


# ============================================================
# Selection operator on a single partition
# ============================================================

SELECTION_CACHE = {}

def selection_partition(partition):
    """
    Compute S(partition) in the raw p-basis.

    Returns:
        {new_partition: integer multiplicity}

    Implements:
        S λ = (ℓ+1)(λ ∪ 1) - Σ_a m_a λ^[a]
    """
    partition = canonical_partition(partition)

    if partition in SELECTION_CACHE:
        return SELECTION_CACHE[partition]

    from collections import Counter

    counts = Counter(partition)
    ell = len(partition)
    results = {}

    # Positive term: (ell + 1) * (lambda union 1)
    new_counts = counts.copy()
    new_counts[1] += 1
    part = counts_to_partition(new_counts)
    results[part] = ell + 1

    # Negative terms grouped by multiplicity
    for a, m in counts.items():
        new_counts = counts.copy()

        new_counts[a] -= 1
        if new_counts[a] == 0:
            del new_counts[a]

        new_counts[a + 1] += 1

        part = counts_to_partition(new_counts)

        coeff = results.get(part, 0) - m
        if coeff:
            results[part] = coeff
        elif part in results:
            del results[part]

    SELECTION_CACHE[partition] = results
    return results


# ============================================================
# Apply selection to a Y-basis expansion
#
# Input:
#   QY = {partition: coefficient}
# representing Σ coeff * Y_partition
#
# Output:
#   GP = {partition: coefficient}
# representing raw p-basis coefficients of S Q
# ============================================================

def apply_selection_to_Y_basis(QY):
    """
    Apply S to a Y-basis expansion.

    Input:
        QY = {lambda: coeff}
    representing
        sum coeff * Y_lambda

    Output:
        GP = {partition: coeff}
    in the raw p-basis.
    """
    out = {}

    for lam, coeff in QY.items():
        if coeff == 0:
            continue

        # Expand Y_lambda in p-basis
        Yexp = Y_lambda(lam)

        # Apply S to every p-monomial in the expansion
        for part, ycoeff in Yexp.items():
            for new_part, mult in selection_partition(part).items():
                contrib = coeff * ycoeff * mult
                if contrib == 0:
                    continue

                new_coeff = out.get(new_part, Fraction(0)) + contrib
                if new_coeff:
                    out[new_part] = new_coeff
                elif new_part in out:
                    del out[new_part]

    return out


# ============================================================
# Triangular back-substitution:
# Convert raw p-basis expansion to Y-basis expansion.
# ============================================================

def p_to_Y(G):
    """
    Convert a sparse p-basis expansion G into the Y basis.

    Parameters
    ----------
    G : dict
        partition -> coefficient in p basis

    Returns
    -------
    dict
        partition -> coefficient in Y basis
    """
    # Remaining p-expansion to eliminate
    rem = {
        canonical_partition(part): Fraction(coeff)
        for part, coeff in G.items()
        if coeff != 0
    }

    coeffs_Y = {}

    # Process lengths from largest to smallest
    lengths = sorted({len(part) for part in rem}, reverse=True)

    for L in lengths:
        # All current partitions of length L
        current_parts = [
            part for part in list(rem.keys())
            if len(part) == L
        ]

        for part in current_parts:
            c = rem.get(part, Fraction(0))
            if c == 0:
                continue

            # Because Y_part = p_part + lower-length terms,
            # c is exactly the Y coefficient.
            coeffs_Y[part] = coeffs_Y.get(part, Fraction(0)) + c

            # Subtract c * Y_part from remainder
            Yexp = Y_lambda(part)
            add_scaled_dict(rem, Yexp, -c)

    # Remove zeros
    coeffs_Y = {p: c for p, c in coeffs_Y.items() if c != 0}

    return coeffs_Y


# ============================================================
# Compute Q_k recursively
# ============================================================

def Q_k(k):
    """
    Return Q_k in the Y basis:
        {partition: coefficient}
    """
    if k in Q_CACHE:
        return Q_CACHE[k]

    # Build recursively from Q_{k-1}
    Q_prev = Q_k(k - 1)

    # Step 1: Apply selection operator in raw p basis
    G_p = apply_selection_to_Y_basis(Q_prev)

    # Step 2: Re-expand in Y basis
    G_Y = p_to_Y(G_p)

    # Step 3: Multiply by 2 / [ℓ(ℓ+1)]
    Q_curr = {}

    for part, coeff in G_Y.items():
        ell = len(part)
        factor = Fraction(2, ell * (ell + 1))
        new_coeff = coeff * factor
        if new_coeff:
            Q_curr[part] = new_coeff

    Q_CACHE[k] = Q_curr
    return Q_curr


# ============================================================
# Optional: compute all Q_0,...,Q_n
# ============================================================

def compute_Q_up_to(n):
    """
    Compute and cache Q_0 through Q_n.
    """
    for k in range(n + 1):
        Q_k(k)
    return Q_CACHE


# ============================================================
# Pretty-print Q_k
# ============================================================

def print_Q(k):
    Q = Q_k(k)
    print(f"Q_{k} =")
    for part in sorted(Q, key=lambda p: (-len(p), p), reverse=False):
        print(f"  {format_partition(part)} : {Q[part]}")


# ============================================================
# Example usage
# ============================================================

# Compute Q_0 ... Q_L (Y-basis)
#compute_Q_up_to(16)

# Print Q_k:
#for i in range(10):
#    print_Q(i)

##############################################################


# ============================================================
# Numerical evaluation of Q_k
#
# Assumes the following symbolic caches already exist:
#   Q_CACHE[k]        : {lambda: q_lambda}       (Q_k in Y basis)
#   Y_CACHE[lambda]   : {mu: y_{lambda,mu}}      (Y_lambda in p basis)
#
# All symbolic work is assumed complete. This code performs
# only numerical evaluation.
# ============================================================


# ------------------------------------------------------------
# Compute power sums:
#   p_n = Σ_i x_i (s_gamma - s_i)^n
#
# Returns array p where p[n] = p_n.
# p[0] = 1 by convention.
# ------------------------------------------------------------
def compute_power_sums(x, s, gamma, nmax):
    """
    Parameters
    ----------
    x : 1D numpy array
        Frequencies, sum(x) = 1.
    s : 1D numpy array
        Selection coefficients.
    gamma : int
        Index specifying the focal type.
    nmax : int
        Maximum power sum required.

    Returns
    -------
    p : numpy.ndarray
        p[n] = Σ_i x_i (s_gamma - s_i)^n
    """
    x = np.asarray(x, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)

    delta = s[gamma] - s

    # p[0] = sum_i x_i = 1
    p = np.empty(nmax + 1, dtype=np.float64)
    p[0] = 1.0

    # Iteratively build delta^n to avoid repeated exponentiation
    power = np.ones_like(delta)
    for n in range(1, nmax + 1):
        power *= delta
        p[n] = np.dot(x, power)

    return p


# ------------------------------------------------------------
# Evaluate p_lambda = ∏_j p[lambda_j]
# ------------------------------------------------------------
def evaluate_partition(partition, p):
    """
    Numerically evaluate p_partition.

    Parameters
    ----------
    partition : tuple of ints
    p : array of precomputed power sums

    Returns
    -------
    float
    """
    value = 1.0
    for n in partition:
        value *= p[n]
    return value


# ------------------------------------------------------------
# Precompute numerical values of all Y_lambda in Q_k.
#
# Uses:
#   Y_lambda = Σ_mu y_{lambda,mu} p_mu
# ------------------------------------------------------------
def evaluate_all_Y_for_Q(k, p):
    """
    Evaluate all Y_lambda that appear in Q_k.

    Parameters
    ----------
    k : int
    p : array of power sums

    Returns
    -------
    dict
        lambda -> numerical value Y_lambda(x)
    """
    Q = Q_CACHE[k]
    Y_values = {}

    for lam in Q:
        expansion = Y_CACHE[lam]

        total = 0.0
        for mu, coeff in expansion.items():
            total += float(coeff) * evaluate_partition(mu, p)

        Y_values[lam] = total

    return Y_values


# ------------------------------------------------------------
# Numerically evaluate Q_k
# ------------------------------------------------------------
def evaluate_Q(k, x, s, gamma):
    """
    Compute Q_k numerically.

    Parameters
    ----------
    k : int
    x : array of frequencies
    s : array of selection coefficients
    gamma : int

    Returns
    -------
    float
        Numerical value of Q_k.
    """
    # Ensure symbolic data are available
    if k not in Q_CACHE:
        Q_k(k)   # symbolic recursion from earlier code

    # Compute all required p_n up to n = k
    p = compute_power_sums(x, s, gamma, k)

    # Evaluate all Y_lambda appearing in Q_k
    Y_values = evaluate_all_Y_for_Q(k, p)

    # Assemble Q_k = Σ q_lambda Y_lambda
    total = 0.0
    for lam, q_coeff in Q_CACHE[k].items():
        total += float(q_coeff) * Y_values[lam]

    return total


# ------------------------------------------------------------
# Optional: compile a fast numerical evaluator for repeated use
# ------------------------------------------------------------
NUMERIC_Q_CACHE = {}

def build_numeric_Q(k):
    """
    Precompute a flattened representation of Q_k directly in
    the p-basis, so numerical evaluation becomes a single sparse
    polynomial evaluation.

    Returns
    -------
    dict
        partition -> coefficient in p basis
    """
    if k in NUMERIC_Q_CACHE:
        return NUMERIC_Q_CACHE[k]

    if k not in Q_CACHE:
        Q_k(k)

    coeffs = {}

    # Q_k = Σ_lambda q_lambda Y_lambda
    #     = Σ_mu c_mu p_mu
    for lam, q_coeff in Q_CACHE[k].items():
        for mu, y_coeff in Y_CACHE[lam].items():
            coeff = float(q_coeff * y_coeff)
            coeffs[mu] = coeffs.get(mu, 0.0) + coeff

    # Remove zeros
    coeffs = {mu: c for mu, c in coeffs.items() if c != 0.0}

    NUMERIC_Q_CACHE[k] = coeffs
    return coeffs


def evaluate_Q_fast(k, x, s, gamma):
    """
    Fast numerical evaluation using a pre-flattened p-basis form.
    Recommended when evaluating Q_k many times.
    """
    coeffs = build_numeric_Q(k)
    p = compute_power_sums(x, s, gamma, k)

    total = 0.0
    for partition, coeff in coeffs.items():
        total += coeff * evaluate_partition(partition, p)

    return total


# ============================================================
# Example usage
# ============================================================

#M = 100
#rng = np.random.default_rng(12345)

# Random frequencies
#x = rng.random(M)
#x /= x.sum()

# Selection coefficients in [0, 0.01]
#s = rng.uniform(0.0, 0.01, size=M)

#gamma = 0
#k = 10

# First call includes symbolic preprocessing if needed
#value = evaluate_Q_fast(k, x, s, gamma)

#print(f"Q_{k} =", value)

##############################################################


# ============================================================
# Fixation probability evaluator
#
# Computes
#
#   P_fix^gamma
#     = x[gamma] * (1 + sum_{k>=1} N^k Q_k^gamma)
#
# where Q_k^gamma is obtained numerically via evaluate_Q_fast().
#
# The series is truncated adaptively when the newest term is
# smaller than a user-specified tolerance.
#
# Requires:
#   - evaluate_Q_fast(k, x, s, gamma)
#   - Q_k(k) symbolic recursion and caches from previous code
#
# ============================================================


def fixation_probability(
    x,
    s,
    gamma,
    N,
    tol=1e-12,
    max_k=100,
    precompute_symbolic=True,
    return_details=False
):
    """
    Compute fixation probability:

        P_fix^gamma = x_gamma [1 + sum_{k>=1} N^k Q_k^gamma]

    Parameters
    ----------
    x : array_like
        Initial frequencies, sum(x) = 1.
    s : array_like
        Selection coefficients.
    gamma : int
        Index of focal genotype.
    N : float or int
        Population size.
    tol : float, optional
        Absolute tolerance for truncating the series.
        The summation stops when

            |x_gamma * N^k * Q_k| < tol

        after at least one nontrivial term.
    max_k : int, optional
        Maximum order to include.
    precompute_symbolic : bool, optional
        If True, compute symbolic Q_k recursively as needed.
    return_details : bool, optional
        If True, return a dictionary with diagnostic information.

    Returns
    -------
    float
        Fixation probability.

    Or if return_details=True:
        {
            "P_fix": ...,
            "orders_used": ...,
            "series_sum": ...,
            "terms": [...],
        }
    """
    x = np.asarray(x, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)

    if not np.isclose(x.sum(), 1.0):
        raise ValueError("x must sum to 1.")

    if gamma < 0 or gamma >= len(x):
        raise IndexError("gamma out of bounds.")

    if len(x) != len(s):
        raise ValueError("x and s must have the same length.")

    if N <= 0:
        raise ValueError("N must be positive.")

    #if tol <= 0:
    #    raise ValueError("tol must be positive.")

    x_gamma = float(x[gamma])

    # Start with the leading neutral term
    series_sum = 1.0
    terms = []

    for k in range(1, max_k + 1):
        # Ensure symbolic coefficient is available if needed
        if precompute_symbolic and k not in Q_CACHE:
            Q_k(k)

        # Numerical evaluation of Q_k^gamma
        qk = evaluate_Q_fast(k, x, s, gamma)

        # Contribution to the bracket
        bracket_term = (N ** k) * qk

        # Contribution to the fixation probability
        contrib = x_gamma * bracket_term

        series_sum += bracket_term
        terms.append(
            {
                "k": k,
                "Q_k": qk,
                "bracket_term": bracket_term,
                "P_contribution": contrib,
            }
        )

        # Adaptive truncation (optional)
        if tol > 0.0 and abs(contrib) < tol:
            break
    else:
        # Reached max_k without satisfying tolerance
        if tol > 0.0:
            print("WARNING: Reached max_k without satisfying tolerance!")
        pass

    P_fix = x_gamma * series_sum

    if return_details:
        return {
            "P_fix": P_fix,
            "orders_used": len(terms),
            "series_sum": series_sum,
            "terms": terms,
        }

    return P_fix


# ------------------------------------------------------------
# Convenience function: compute fixation probabilities for all
# gamma = 0, ..., M-1
# ------------------------------------------------------------
def fixation_probabilities_all(
    x,
    s,
    N,
    tol=1e-12,
    max_k=100,
    precompute_symbolic=True
):
    """
    Compute fixation probabilities for every focal genotype.

    Returns
    -------
    numpy.ndarray
        P_fix for each gamma.
    """
    x = np.asarray(x, dtype=np.float64)
    M = len(x)

    P = np.empty(M, dtype=np.float64)

    # Symbolic caches are independent of gamma, so build once.
    if precompute_symbolic:
        for k in range(1, max_k + 1):
            if k not in Q_CACHE:
                Q_k(k)

    for gamma in range(M):
        P[gamma] = fixation_probability(
            x,
            s,
            gamma,
            N,
            tol=tol,
            max_k=max_k,
            precompute_symbolic=False,
            return_details=False,
        )

    return P


# ============================================================
# Example usage
# ============================================================

#rng = np.random.default_rng(12345)

#M = 2
#N = 100
#tol = 1e-3

# Random initial frequencies
#x = rng.random(M)
#x /= x.sum()

# Random selection coefficients in [0, 0.01]
#s = rng.uniform(0.0, 0.01, size=M)
#s[-1] = 0.0

#gamma = 0

# Single fixation probability
#P = fixation_probability(
#    x=x,
#    s=s,
#    gamma=gamma,
#    N=N,
#    tol=tol,
#    max_k=14
#)

#print(f"P_fix^{gamma} = {P:.16e}")

# All fixation probabilities
#P_all = fixation_probabilities_all(
#    x=x,
#    s=s,
#    N=N,
#    tol=tol,
#    max_k=20,
#)

#print("Sum of fixation probabilities:", P_all.sum())

##############################################################


def print_fixation_details(details, digits=16):
    """
    Pretty-print the output of fixation_probability(..., return_details=True).

    Parameters
    ----------
    details : dict
        Dictionary returned by fixation_probability(..., return_details=True).
    digits : int, optional
        Number of digits after the decimal point in scientific notation.
    """
    fmt = f"{{:.{digits}e}}"

    print()
    print(f"P_fix       = {fmt.format(details['P_fix'])}")
    print(f"Series sum  = {fmt.format(details['series_sum'])}")
    print(f"Orders used = {details['orders_used']}")
    print()

    # Column widths
    w_k = 4
    w_q = max(12, digits + 7)
    w_b = max(16, digits + 7)
    w_p = max(18, digits + 7)

    # Header
    header = (
        f"{'k':>{w_k}}  "
        f"{'Q_k':>{w_q}}  "
        f"{'N^k Q_k':>{w_b}}  "
        f"{'Contribution to P_fix':>{w_p}}"
    )
    print(header)
    print("-" * len(header))

    # Rows
    for term in details["terms"]:
        print(
            f"{term['k']:>{w_k}d}  "
            f"{fmt.format(term['Q_k']):>{w_q}}  "
            f"{fmt.format(term['bracket_term']):>{w_b}}  "
            f"{fmt.format(term['P_contribution']):>{w_p}}"
        )

    print()

"""
# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    details = fixation_probability(
        x=x,
        s=s,
        gamma=0,
        N=1000,
        tol=1e-12,
        max_k=20,
        return_details=True,
    )

    print_fixation_details(details, digits=12)
"""


# ============================================================
# Persistent cache storage with change detection
# ============================================================

CACHE_FILENAME = "symbolic_partition_caches.pkl"

# Fingerprint of the cache state when last loaded or saved.
_CACHE_STATE_FINGERPRINT = None


def cache_state_fingerprint():
    """
    Return a lightweight summary of the symbolic cache state.

    This avoids hashing all symbolic data, which could be expensive.
    The fingerprint changes whenever new objects are added to any cache.
    """
    return (
        len(MERGE_CACHE),
        len(SELECTION_CACHE),
        len(Y_CACHE),
        len(Q_CACHE),
        len(NUMERIC_Q_CACHE),
        max(Q_CACHE.keys(), default=0),
        max(NUMERIC_Q_CACHE.keys(), default=0),
        max((sum(part) for part in Y_CACHE.keys()), default=0),
    )


def load_symbolic_caches(filename=CACHE_FILENAME):
    """
    Load cached symbolic data from disk, if present.
    """
    global MERGE_CACHE
    global SELECTION_CACHE
    global Y_CACHE
    global Q_CACHE
    global NUMERIC_Q_CACHE
    global _CACHE_STATE_FINGERPRINT

    if not os.path.exists(filename):
        print(f"No symbolic cache file found: {filename}")
        _CACHE_STATE_FINGERPRINT = cache_state_fingerprint()
        return

    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)

        MERGE_CACHE.update(data.get("MERGE_CACHE", {}))
        SELECTION_CACHE.update(data.get("SELECTION_CACHE", {}))
        Y_CACHE.update(data.get("Y_CACHE", {}))

        loaded_Q = data.get("Q_CACHE", {})
        if 0 not in loaded_Q:
            loaded_Q[0] = {(): Fraction(1)}
        Q_CACHE.update(loaded_Q)

        NUMERIC_Q_CACHE.update(data.get("NUMERIC_Q_CACHE", {}))

        print(
            "Loaded symbolic caches: "
            f"{len(MERGE_CACHE)} merges, "
            f"{len(SELECTION_CACHE)} selections, "
            f"{len(Y_CACHE)} Y expansions, "
            f"{len(Q_CACHE)} Q_k objects, "
            f"{len(NUMERIC_Q_CACHE)} flattened Q_k."
        )

    except Exception as e:
        print(f"Warning: could not load symbolic caches: {e}")

    # Record the state after loading
    _CACHE_STATE_FINGERPRINT = cache_state_fingerprint()


def save_symbolic_caches(filename=CACHE_FILENAME):
    """
    Save symbolic caches only if they have changed since the
    last load or save.
    """
    global _CACHE_STATE_FINGERPRINT

    current_fingerprint = cache_state_fingerprint()

    # Nothing changed: skip writing the file.
    if current_fingerprint == _CACHE_STATE_FINGERPRINT:
        print("Symbolic caches unchanged; skipping save.")
        return

    data = {
        "MERGE_CACHE": MERGE_CACHE,
        "SELECTION_CACHE": SELECTION_CACHE,
        "Y_CACHE": Y_CACHE,
        "Q_CACHE": Q_CACHE,
        "NUMERIC_Q_CACHE": NUMERIC_Q_CACHE,
    }

    tmpfile = filename + ".tmp"

    with open(tmpfile, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Atomic replace to avoid corrupt files if interrupted.
    os.replace(tmpfile, filename)

    print(
        "Saved symbolic caches: "
        f"{len(MERGE_CACHE)} merges, "
        f"{len(SELECTION_CACHE)} selections, "
        f"{len(Y_CACHE)} Y expansions, "
        f"{len(Q_CACHE)} Q_k objects, "
        f"{len(NUMERIC_Q_CACHE)} flattened Q_k."
    )

    # Update the saved-state fingerprint.
    _CACHE_STATE_FINGERPRINT = current_fingerprint


##############################################################

def evaluate_Q_reference(k, x, s, gamma):
    """
    Independent reference implementation of Q_k.

    This routine does NOT use build_numeric_Q() or evaluate_Q_fast().
    Instead it follows the symbolic definition directly:

        Q_k = sum_lambda q_lambda Y_lambda
        Y_lambda = sum_mu y_{lambda,mu} p_mu

    This is slower, but serves as a reliable correctness check.
    """
    # Ensure symbolic objects are available
    if k not in Q_CACHE:
        Q_k(k)

    # Power sums p_n
    p = compute_power_sums(x, s, gamma, k)

    total = 0.0

    # Q_k = sum_lambda q_lambda Y_lambda
    for lam, q_coeff in Q_CACHE[k].items():
        Y_total = 0.0

        # Y_lambda = sum_mu y_coeff * p_mu
        for mu, y_coeff in Y_CACHE[lam].items():
            Y_total += float(y_coeff) * evaluate_partition(mu, p)

        total += float(q_coeff) * Y_total

    return total


def check_evaluate_Q_fast(
    max_k=10,
    M=50,
    n_trials=5,
    seed=12345,
    rtol=1e-12,
    atol=1e-14,
    verbose=True,
):
    """
    Compare evaluate_Q_fast() against an independent reference
    implementation for random test cases.

    Parameters
    ----------
    max_k : int
        Check k = 0, ..., max_k.
    M : int
        Number of genotypes.
    n_trials : int
        Number of random (x, s, gamma) test cases.
    seed : int
        Random seed.
    rtol, atol : float
        Tolerances passed to np.isclose().
    verbose : bool
        If True, print a detailed table.

    Returns
    -------
    True if all tests pass, otherwise raises AssertionError.
    """
    rng = np.random.default_rng(seed)

    # Build all symbolic objects once
    for k in range(max_k + 1):
        Q_k(k)
        build_numeric_Q(k)

    if verbose:
        print(
            f"{'trial':>5} {'k':>3} {'gamma':>5} "
            f"{'reference':>20} {'fast':>20} "
            f"{'abs error':>12} {'rel error':>12}"
        )
        print("-" * 95)

    for trial in range(1, n_trials + 1):
        # Random frequencies
        x = rng.random(M)
        x /= x.sum()

        # Random selection coefficients
        s = rng.uniform(0.0, 0.01, size=M)

        # Random focal genotype
        gamma = int(rng.integers(M))

        for k in range(max_k + 1):
            ref = evaluate_Q_reference(k, x, s, gamma)
            fast = evaluate_Q_fast(k, x, s, gamma)

            abs_err = abs(ref - fast)
            rel_err = abs_err / max(abs(ref), abs(fast), 1e-300)

            if verbose:
                print(
                    f"{trial:5d} {k:3d} {gamma:5d} "
                    f"{ref:20.12e} {fast:20.12e} "
                    f"{abs_err:12.3e} {rel_err:12.3e}"
                )

            if not np.isclose(ref, fast, rtol=rtol, atol=atol):
                raise AssertionError(
                    f"Mismatch detected:\n"
                    f"  trial = {trial}\n"
                    f"  k = {k}\n"
                    f"  gamma = {gamma}\n"
                    f"  reference = {ref:.16e}\n"
                    f"  fast      = {fast:.16e}\n"
                    f"  abs error = {abs_err:.3e}\n"
                    f"  rel error = {rel_err:.3e}"
                )

    if verbose:
        print("\nAll tests passed.")

    return True


"""
# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    check_evaluate_Q_fast(
        max_k=10,
        M=2,
        n_trials=3,
        seed=12345,
        rtol=1e-12,
        atol=1e-14,
        verbose=True,
    )
"""

