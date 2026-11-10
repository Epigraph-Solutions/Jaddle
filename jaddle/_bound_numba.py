"""
Lagrangian bounds compiled with numba (optional: ``pip install jaddle[fast]``).
``jaddle_mip.box_bound``, ``lagrangian_terms``, ``_exact_side_bound`` and
``_BranchAndBound.bound_over`` use these when numba is installed.

Each is the numpy version's arithmetic in one pass over the columns, with
no temporaries: the search calls them tens of thousands of times on
vectors of a few thousand entries, where numpy's temporaries and calls
cost more than the arithmetic. They release the GIL. The reduced costs
``c + Aᵀy`` are summed in the same (CSR) order as scipy's product, so they
match it bit for bit; the bound's sum runs in column order where numpy's is
pairwise, so the bounds can differ in the last bits.
"""

import numba as nb
import numpy as np


@nb.njit(cache=True, inline="always")
def _term(rj, lo, up):
    """``rⱼlⱼ`` or ``rⱼuⱼ``, whichever is smaller (0 when ``rⱼ`` is 0)."""
    if rj > 0:
        return rj * lo
    if rj < 0:
        return rj * up
    return 0.0


@nb.njit(cache=True, inline="always")
def _widen(value, down, continuous):
    """A continuous variable's bound moved out by 1e-7·(1 + |bound|)."""
    if not continuous:
        return value
    step = 1e-7 * (1.0 + abs(value))
    return value - step if down else value + step


@nb.njit(cache=True, nogil=True)
def box_bound(constant, r, lower, upper, tol):
    """``jaddle_mip.box_bound``: ``(bound, rigorous)``."""
    total = 0.0
    rigorous = True
    for j in range(r.size):
        rj = r[j]
        t = _term(rj, lower[j], upper[j])
        if not np.isfinite(t):
            if abs(rj) > tol:
                return -np.inf, True
            rigorous = False
            continue
        total += t
    return constant + total, rigorous


@nb.njit(cache=True, nogil=True)
def widened_box_bound(constant, r, lower, upper, is_int, tol):
    """``box_bound`` with the continuous variables' bounds widened
    (``_BranchAndBound.bound_over``)."""
    total = 0.0
    rigorous = True
    for j in range(r.size):
        rj = r[j]
        cont = not is_int[j]
        t = _term(rj, _widen(lower[j], True, cont), _widen(upper[j], False, cont))
        if not np.isfinite(t):
            if abs(rj) > tol:
                return -np.inf, True
            rigorous = False
            continue
        total += t
    return constant + total, rigorous


@nb.njit(cache=True, inline="always")
def _reduced_cost(c, ptr, idx, val, y, j):
    """``cⱼ + (Aᵀy)ⱼ`` from the CSR rows of ``Aᵀ``, in scipy's order."""
    s = 0.0
    for k in range(ptr[j], ptr[j + 1]):
        s += val[k] * y[idx[k]]
    return c[j] + s


@nb.njit(cache=True, inline="always")
def _clip_duals(y, n_eq):
    """The inequality duals clipped to ``y >= 0``, in place."""
    for i in range(n_eq, y.size):
        if y[i] < 0.0:
            y[i] = 0.0


@nb.njit(cache=True, inline="always")
def _constant(b, y):
    """``−bᵀy``."""
    s = 0.0
    for i in range(y.size):
        s += b[i] * y[i]
    return -s


@nb.njit(cache=True, nogil=True)
def lagrangian_terms(c, ptr, idx, val, b, n_eq, y):
    """``jaddle_mip.lagrangian_terms`` with ``Aᵀ`` as CSR arrays:
    ``(−bᵀy, r)``. ``y`` is copied, not changed."""
    y = y.copy()
    _clip_duals(y, n_eq)
    n = c.size
    r = np.empty(n)
    for j in range(n):
        r[j] = _reduced_cost(c, ptr, idx, val, y, j)
    return _constant(b, y), r


@nb.njit(cache=True, nogil=True)
def exact_side_bound(
    c, ptr, idx, val, b, n_eq, y_eq, y_ineq, lower, upper, is_int, roundoff, tol
):
    """``jaddle_mip._exact_side_bound`` in one pass: the duals assembled and
    clipped, reduced costs at infinite bounds no larger than
    ``roundoff·(1 + max|c|)`` zeroed, continuous bounds widened, and the
    box bound. ``y_ineq`` must hold at least ``m − n_eq`` entries."""
    m = b.size
    y = np.zeros(m)
    for i in range(min(n_eq, y_eq.size)):
        y[i] = y_eq[i]
    for i in range(n_eq, m):
        y[i] = y_ineq[i - n_eq]
    _clip_duals(y, n_eq)
    largest = 0.0
    for j in range(c.size):
        largest = max(largest, abs(c[j]))
    limit = roundoff * (1.0 + largest)
    total = 0.0
    rigorous = True
    for j in range(c.size):
        rj = _reduced_cost(c, ptr, idx, val, y, j)
        lo, up = lower[j], upper[j]
        if (
            (rj > 0 and not np.isfinite(lo)) or (rj < 0 and not np.isfinite(up))
        ) and abs(rj) <= limit:
            rj = 0.0
        cont = not is_int[j]
        t = _term(rj, _widen(lo, True, cont), _widen(up, False, cont))
        if not np.isfinite(t):
            if abs(rj) > tol:
                return -np.inf, True
            rigorous = False
            continue
        total += t
    return _constant(b, y) + total, rigorous
