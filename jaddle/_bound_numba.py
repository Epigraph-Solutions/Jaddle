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


@nb.njit(cache=True, nogil=True)
def box_value(constant, r, lo, up):
    """``constant + Σⱼ min(rⱼloⱼ, rⱼupⱼ)``, or ``-inf`` when a reduced cost
    points at an infinite bound (no tolerance: ``dual_ascent``'s own test)."""
    total = 0.0
    for j in range(r.size):
        t = _term(r[j], lo[j], up[j])
        if not np.isfinite(t):
            return -np.inf
        total += t
    return constant + total


# ``dual_ascent``: a step that ends at a wall (a reduced cost that would
# point at an infinite bound past it) stops this far inside, relative to
# 1 + |t|, so the reduced cost recomputed from the duals keeps its sign.
_WALL_MARGIN = 1e-10
# A row moves only when its slope is more than this share of its terms.
_SLOPE_TOL = 1e-12


@nb.njit(cache=True, nogil=True)
def _row_step(idx, val, start, end, bi, floor, r, lo, up, bp, wt):
    """The step ``t`` of one row's dual that maximises the Lagrangian bound
    along it (``dual_ascent``), with ``t >= floor``. ``bp``/``wt`` are
    scratch of at least the row's length."""
    tlo, thi = floor, np.inf
    for k in range(start, end):
        a = val[k]
        if a == 0.0:
            continue
        j = idx[k]
        if lo[j] == -np.inf:  # rⱼ + a·t <= 0
            w = -r[j] / a
            if a > 0:
                thi = min(thi, w)
            else:
                tlo = max(tlo, w)
        if up[j] == np.inf:  # rⱼ + a·t >= 0
            w = -r[j] / a
            if a > 0:
                tlo = max(tlo, w)
            else:
                thi = min(thi, w)
    if not tlo <= thi:
        return 0.0
    t0 = 0.0
    if tlo > 0.0:
        t0 = tlo
        margin = _WALL_MARGIN * (1.0 + abs(t0))
        if t0 + margin <= thi:
            t0 += margin
    elif thi < 0.0:
        t0 = thi
        margin = _WALL_MARGIN * (1.0 + abs(t0))
        if t0 - margin >= tlo:
            t0 -= margin
    # The slopes just right and left of t0, and the kinks either side.
    right = -bi
    left = -bi
    scale = abs(bi)
    count = 0
    for k in range(start, end):
        a = val[k]
        if a == 0.0:
            continue
        j = idx[k]
        l, u = lo[j], up[j]
        if l == -np.inf:
            if u == np.inf:
                continue  # a free column: the interval is its one point
            right += a * u
            left += a * u
            scale += abs(a * u)
        elif u == np.inf or l == u:
            right += a * l
            left += a * l
            scale += abs(a * l)
        else:
            w = -r[j] / a
            at_l, at_u = a * l, a * u
            scale += max(abs(at_l), abs(at_u))
            # Before its kink a column sits where −a points, after it at a's.
            before = at_u if a > 0 else at_l
            after = at_l if a > 0 else at_u
            if w > t0:
                right += before
                left += before
            elif w < t0:
                right += after
                left += after
            else:
                right += after
                left += before
                continue
            bp[count] = w
            wt[count] = abs(a) * (u - l)
            count += 1
    tol = _SLOPE_TOL * scale + 1e-300
    if right > tol and t0 < thi:
        order = np.argsort(bp[:count])
        t = t0
        for q in range(count):
            w = bp[order[q]]
            if w <= t0:
                continue
            if w > thi:
                break
            t = w
            right -= wt[order[q]]
            if right <= 0.0:
                return t
        if thi < np.inf:
            margin = _WALL_MARGIN * (1.0 + abs(thi))
            return thi - margin if thi - margin >= t else t
        return t  # unbounded above (the row is infeasible): its last kink
    if left < -tol and t0 > tlo:
        order = np.argsort(bp[:count])
        t = t0
        for q in range(count - 1, -1, -1):
            w = bp[order[q]]
            if w >= t0:
                continue
            if w < tlo:
                break
            t = w
            left += wt[order[q]]
            if left >= 0.0:
                return t
        if tlo == floor:
            return tlo
        if tlo > -np.inf:
            margin = _WALL_MARGIN * (1.0 + abs(tlo))
            return tlo + margin if tlo + margin <= t else t
        return t
    return t0


@nb.njit(cache=True, nogil=True)
def dual_ascent(c, ptr, idx, val, b, n_eq, y, lower, upper, is_int, sweeps, target):
    """
    Coordinate ascent on the Lagrangian bound
    ``g(y) = −bᵀy + Σⱼ min(rⱼlⱼ, rⱼuⱼ)``, ``r = c + Aᵀy``, from the duals
    ``y``, with ``A`` as CSR arrays (by row) and the continuous bounds
    widened as ``widened_box_bound`` does.

    ``g`` is concave and piecewise linear. Along row ``i`` its slope is
    ``aᵢ·x − bᵢ`` at the box vertex ``x`` the reduced costs pick, which
    falls by ``|aᵢⱼ|(uⱼ − lⱼ)`` as ``rⱼ`` crosses zero at
    ``t = −rⱼ/aᵢⱼ``: the row's dual moves to where the slope changes sign
    (``_row_step``), within ``yᵢ >= 0`` on an inequality row and the
    interval where no reduced cost of the row points at an infinite bound.
    Rows are taken in order, each from the reduced costs the last left,
    for ``sweeps`` passes, or until a pass moves nothing or the bound
    reaches ``target``.

    Returns ``(y, moves)``; ``y`` is a new array.
    """
    m = b.size
    n = c.size
    y = y.copy()
    _clip_duals(y, n_eq)
    lo = np.empty(n)
    up = np.empty(n)
    for j in range(n):
        cont = not is_int[j]
        lo[j] = _widen(lower[j], True, cont)
        up[j] = _widen(upper[j], False, cont)
    r = c.copy()
    width = 0
    for i in range(m):
        width = max(width, ptr[i + 1] - ptr[i])
        yi = y[i]
        if yi != 0.0:
            for k in range(ptr[i], ptr[i + 1]):
                r[idx[k]] += val[k] * yi
    bp = np.empty(width)
    wt = np.empty(width)
    moves = 0
    for _ in range(sweeps):
        moved = 0
        for i in range(m):
            start, end = ptr[i], ptr[i + 1]
            if start == end:
                continue
            floor = -y[i] if i >= n_eq else -np.inf
            t = _row_step(idx, val, start, end, b[i], floor, r, lo, up, bp, wt)
            if t == 0.0:
                continue
            if t <= floor:
                t = floor
                y[i] = 0.0
            else:
                y[i] += t
            for k in range(start, end):
                r[idx[k]] += val[k] * t
            moved += 1
        moves += moved
        if moved == 0:
            break
        if target < np.inf and box_value(_constant(b, y), r, lo, up) >= target:
            break
    return y, moves
