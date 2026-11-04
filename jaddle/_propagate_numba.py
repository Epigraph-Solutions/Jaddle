"""
Activity-based bound propagation compiled with numba (optional:
``pip install jaddle[fast]``). ``jaddle_mip.BoundPropagator`` uses it when
numba is installed.

The rules are ``BoundPropagator.__call__``'s, step for step: each pass
computes every row's least and largest activity over the box (infinite
contributions counted apart), then every entry's implied bound on its
variable from the rest of its row, all from the same box (no bound found in
a pass is used before the next). Integer bounds are rounded, only clear
improvements of a sane size are kept, and the same infeasibility tests
apply. The row sums add the entries in the same (column-major) order as
the numpy version's ``bincount``, so both give the same bounds bit for bit.
The numpy version spends most of its time allocating temporaries for every
entry, several times a pass; this one keeps a few row and column arrays.
"""

import math

import numba as nb
import numpy as np


@nb.njit(cache=True)
def _activities(rows, vals, positive, col_ptr, lower, upper, m):
    """Per-row least and largest finite activity and the counts of infinite
    contributions to each, summed in entry order."""
    min_act = np.zeros(m)
    max_act = np.zeros(m)
    n_min_inf = np.zeros(m, np.int64)
    n_max_inf = np.zeros(m, np.int64)
    n = col_ptr.shape[0] - 1
    for j in range(n):
        lo, up = lower[j], upper[j]
        for k in range(col_ptr[j], col_ptr[j + 1]):
            a = vals[k]
            i = rows[k]
            if positive[k]:
                min_c, max_c = a * lo, a * up
            else:
                min_c, max_c = a * up, a * lo
            if math.isfinite(min_c):
                min_act[i] += min_c
            else:
                n_min_inf[i] += 1
            if math.isfinite(max_c):
                max_act[i] += max_c
            else:
                n_max_inf[i] += 1
    return min_act, max_act, n_min_inf, n_max_inf


@nb.njit(cache=True)
def _rows_missed(min_act, max_act, n_min_inf, n_max_inf, row_lower, row_upper, tol):
    for i in range(min_act.shape[0]):
        ru, rl = row_upper[i], row_lower[i]
        if n_min_inf[i] == 0 and min_act[i] > ru + tol * (1.0 + abs(ru)):
            return True
        if n_max_inf[i] == 0 and max_act[i] < rl - tol * (1.0 + abs(rl)):
            return True
    return False


@nb.njit(cache=True)
def propagate(
    rows, vals, positive, col_ptr, row_lower, row_upper, is_int, lower, upper,
    max_passes, tol, round_slack,
):
    """``BoundPropagator.__call__`` on CSC arrays (``rows``/``vals`` of the
    entries, ``col_ptr`` the column starts). Returns new ``(lower, upper,
    infeasible)``; the inputs are not changed."""
    lower = lower.copy()
    upper = upper.copy()
    m = row_lower.shape[0]
    n = col_ptr.shape[0] - 1
    new_lower = np.empty(n)
    new_upper = np.empty(n)
    for _ in range(max_passes):
        min_act, max_act, n_min_inf, n_max_inf = _activities(
            rows, vals, positive, col_ptr, lower, upper, m
        )
        if _rows_missed(min_act, max_act, n_min_inf, n_max_inf, row_lower, row_upper, tol):
            return lower, upper, True
        for j in range(n):
            lo, up = lower[j], upper[j]
            best_up, best_lo = np.inf, -np.inf
            for k in range(col_ptr[j], col_ptr[j + 1]):
                a = vals[k]
                i = rows[k]
                if positive[k]:
                    min_c, max_c = a * lo, a * up
                else:
                    min_c, max_c = a * up, a * lo
                # The rest of the row's activity, this entry left out.
                nmin, nmax = n_min_inf[i], n_max_inf[i]
                if not math.isfinite(min_c):
                    rest_min = min_act[i] if nmin == 1 else -np.inf
                else:
                    rest_min = min_act[i] - min_c if nmin == 0 else -np.inf
                if not math.isfinite(max_c):
                    rest_max = max_act[i] if nmax == 1 else np.inf
                else:
                    rest_max = max_act[i] - max_c if nmax == 0 else np.inf
                from_upper = (row_upper[i] - rest_min) / a  # a x <= ru - rest_min
                from_lower = (row_lower[i] - rest_max) / a  # a x >= rl - rest_max
                sign = 1.0 if a > 0 else (-1.0 if a < 0 else 0.0)
                if math.isnan(from_upper):
                    from_upper = np.inf * sign
                if math.isnan(from_lower):
                    from_lower = -np.inf * sign
                if positive[k]:
                    ub, lb = from_upper, from_lower
                else:
                    ub, lb = from_lower, from_upper
                if ub < best_up:
                    best_up = ub
                if lb > best_lo:
                    best_lo = lb
            if col_ptr[j] == col_ptr[j + 1]:
                best_up, best_lo = np.inf, -np.inf
            if is_int[j]:
                best_up = np.floor(best_up + round_slack)
                best_lo = np.ceil(best_lo - round_slack)
            new_upper[j] = best_up
            new_lower[j] = best_lo

        # Keep only clear improvements of a sane size.
        changed = False
        for j in range(n):
            up, lo = upper[j], lower[j]
            nu, nl = new_upper[j], new_lower[j]
            if abs(nu) < 1e12 and (not math.isfinite(up) or nu < up - tol * (1.0 + abs(up))):
                upper[j] = nu
                changed = True
            if abs(nl) < 1e12 and (not math.isfinite(lo) or nl > lo + tol * (1.0 + abs(lo))):
                lower[j] = nl
                changed = True
        if not changed:
            return lower, upper, False

        for j in range(n):
            if lower[j] > upper[j] and is_int[j]:
                return lower, upper, True
            if lower[j] > upper[j] + tol * (1.0 + abs(upper[j])):
                return lower, upper, True
        # Continuous bounds crossed by rounding error: meet in the middle.
        for j in range(n):
            if lower[j] > upper[j]:
                mid = 0.5 * (lower[j] + upper[j])
                lower[j] = mid
                upper[j] = mid
    # The passes ran out right after a tightening: check the rows against
    # the final box too.
    min_act, max_act, n_min_inf, n_max_inf = _activities(
        rows, vals, positive, col_ptr, lower, upper, m
    )
    if _rows_missed(min_act, max_act, n_min_inf, n_max_inf, row_lower, row_upper, tol):
        return lower, upper, True
    return lower, upper, False
