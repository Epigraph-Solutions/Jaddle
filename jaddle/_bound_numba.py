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


# %%
# Sign repair: reduced costs moved off the infinite bounds they point at


@nb.njit(cache=True, nogil=True)
def _sign_fix_abs(c, ptr, idx, val, n_eq, y, lo, up, margin, sweeps):
    """Passes of ``dual_sign_fix`` on the total shortfall. Returns ``(y, left)``,
    ``left`` the shortfall after the last pass."""
    m = ptr.size - 1
    y = y.copy()
    r = c.copy()
    width = 0
    for i in range(m):
        width = max(width, ptr[i + 1] - ptr[i])
        if y[i] != 0.0:
            for k in range(ptr[i], ptr[i + 1]):
                r[idx[k]] += val[k] * y[i]
    bp = np.empty(2 * width); wt = np.empty(2 * width)
    history = np.zeros(sweeps + 1)
    for sweep in range(sweeps + 1):
        total = 0.0
        for j in range(c.size):
            if up[j] == np.inf and r[j] < margin[j]: total += margin[j] - r[j]
            if lo[j] == -np.inf and r[j] > -margin[j]: total += r[j] + margin[j]
        history[sweep] = total
        if total == 0.0 or sweep == sweeps:
            break
        for i in range(m):
            start, end = ptr[i], ptr[i + 1]
            floor = -y[i] if i >= n_eq else -np.inf
            right = 0.0; left = 0.0; count = 0
            for k in range(start, end):
                a = val[k]
                if a == 0.0: continue
                j = idx[k]
                for side in range(2):
                    if side == 0:
                        if up[j] != np.inf: continue
                        w = (margin[j] - r[j]) / a      # r + a t >= margin
                        falling = a > 0                # violated before w
                    else:
                        if lo[j] != -np.inf: continue
                        w = (-margin[j] - r[j]) / a    # r + a t <= -margin
                        falling = a < 0
                    wa = abs(a)
                    if falling:
                        if w > 0: right -= wa
                        if w >= 0: left -= wa
                    else:
                        if w <= 0: right += wa
                        if w < 0: left += wa
                    bp[count] = w; wt[count] = wa; count += 1
            if count == 0: continue
            t = 0.0
            if right < -1e-12:
                order = np.argsort(bp[:count])
                for q in range(count):
                    w = bp[order[q]]
                    if w <= 0: continue
                    right += wt[order[q]]; t = w
                    if right >= 0: break
            elif left > 1e-12 and floor < 0:
                order = np.argsort(bp[:count])
                for q in range(count - 1, -1, -1):
                    w = bp[order[q]]
                    if w >= 0: continue
                    if w < floor: t = floor; break
                    left -= wt[order[q]]; t = w
                    if left <= 0: break
            if t == 0.0: continue
            if t <= floor: t = floor; y[i] = 0.0
            else: y[i] += t
            for k in range(start, end):
                r[idx[k]] += val[k] * t
    return y, history[min(sweep, sweeps)]


@nb.njit(cache=True, nogil=True)
def _sign_fix_square(
    c, ptr, idx, val, cptr, cidx, n_eq, y, lo, up, margin, slack, full_sweeps, sweeps
):
    """Passes of ``dual_sign_fix`` on the squared shortfall. Returns
    ``(y, wrong)``, ``wrong`` the reduced costs still pointing at an
    infinite bound."""
    m = ptr.size - 1
    y = y.copy(); r = c.copy()
    width = 0
    for i in range(m):
        width = max(width, ptr[i + 1] - ptr[i])
        if y[i] != 0.0:
            for k in range(ptr[i], ptr[i + 1]):
                r[idx[k]] += val[k] * y[i]
    bw = np.empty(2 * width); ba = np.empty(2 * width); bd = np.empty(2 * width)
    hist = np.zeros((sweeps + 1, 2))
    visit = np.ones(m, np.bool_)
    done = 0
    for sweep in range(sweeps + 1):
        cnt = 0; worst = 0.0
        # After the first passes over every row, only the rows of the
        # columns still on the wrong side are visited.
        narrow = sweep >= full_sweeps
        if narrow:
            visit[:] = False
        for j in range(c.size):
            bad = False
            if up[j] == np.inf and r[j] < -slack[j]: cnt += 1; worst = max(worst, -r[j]); bad = True
            if lo[j] == -np.inf and r[j] > slack[j]: cnt += 1; worst = max(worst, r[j]); bad = True
            if bad and narrow:
                for k in range(cptr[j], cptr[j + 1]):
                    visit[cidx[k]] = True
        hist[sweep, 0] = cnt; hist[sweep, 1] = worst; done = sweep
        if cnt == 0 or sweep == sweeps: break
        for i in range(m):
            if narrow and not visit[i]:
                continue
            start, end = ptr[i], ptr[i + 1]
            floor = -y[i] if i >= n_eq else -np.inf
            # each term: v(t) = max(0, d - a t) with its own (a, d); active before w=d/a if a>0, after if a<0
            count = 0; P = 0.0; Q = 0.0   # over terms active just right of 0 ... recomputed per side
            for k in range(start, end):
                a = val[k]
                if a == 0.0: continue
                j = idx[k]
                if up[j] == np.inf:      # need r >= margin: d = margin - r, shrinks by a t
                    bw[count] = (margin[j] - r[j]) / a; ba[count] = a; bd[count] = margin[j] - r[j]; count += 1
                if lo[j] == -np.inf:     # need r <= -margin: d = r + margin, grows by a t -> coefficient -a
                    bw[count] = -(r[j] + margin[j]) / a; ba[count] = -a; bd[count] = r[j] + margin[j]; count += 1
            if count == 0: continue
            # derivative at 0: sum over active of -2 a d ; active iff d > 0 (at d == 0, by side)
            g0 = 0.0
            for q in range(count):
                if bd[q] > 0: g0 -= ba[q] * bd[q]
            if g0 == 0.0: continue
            order = np.argsort(bw[:count])
            t = 0.0
            if g0 < 0:   # move right
                P = 0.0; Q = 0.0
                for q in range(count):
                    a, d = ba[q], bd[q]
                    if (a > 0 and d > 0) or (a < 0 and d >= 0): P += a * d; Q += a * a   # active for small t > 0
                pos = 0
                while pos < count and bw[order[pos]] <= 0: pos += 1
                while True:
                    nxt = bw[order[pos]] if pos < count else np.inf
                    if Q > 0:
                        ts = P / Q
                        if ts <= nxt: t = max(ts, 0.0); break
                    if pos >= count: t = 0.0 if Q == 0 else t; break
                    q = order[pos]; a, d = ba[q], bd[q]
                    if a > 0: P -= a * d; Q -= a * a
                    else: P += a * d; Q += a * a
                    t = nxt; pos += 1
                    if Q <= 1e-300 and P >= 0: break
            else:        # move left
                P = 0.0; Q = 0.0
                for q in range(count):
                    a, d = ba[q], bd[q]
                    if (a > 0 and d >= 0) or (a < 0 and d > 0): P += a * d; Q += a * a   # active for small t < 0
                pos = count - 1
                while pos >= 0 and bw[order[pos]] >= 0: pos -= 1
                while True:
                    nxt = bw[order[pos]] if pos >= 0 else -np.inf
                    if Q > 0:
                        ts = P / Q
                        if ts >= nxt: t = min(ts, 0.0); break
                    if pos < 0: break
                    q = order[pos]; a, d = ba[q], bd[q]
                    if a > 0: P += a * d; Q += a * a
                    else: P -= a * d; Q -= a * a
                    t = nxt; pos -= 1
                    if Q <= 1e-300 and P <= 0: break
                if t < floor: t = floor
            if t == 0.0 or not np.isfinite(t): continue
            if i >= n_eq and y[i] + t <= 0.0: t = -y[i]; y[i] = 0.0
            else: y[i] += t
            for k in range(start, end):
                r[idx[k]] += val[k] * t
    return y, int(hist[done, 0])


@nb.njit(cache=True, nogil=True)
def dual_sign_fix(
    c, ptr, idx, val, cptr, cidx, n_eq, y, lower, upper, margin, sweeps,
    full_sweeps, square_sweeps,
):
    """
    Duals near ``y`` whose reduced costs ``r = c + Aᵀy`` point at no
    infinite bound (``rⱼ >= 0`` where the upper bound is infinite, ``<= 0``
    where the lower one is), so their Lagrangian bound is finite and
    rigorous. ``A`` is CSR by row; inequality duals stay ``>= 0``.

    A column's shortfall is how far ``rⱼ`` is from ``marginⱼ`` on the right
    side of zero. First ``sweeps`` passes take each row's dual to where the
    row's total shortfall (weighted by ``|aᵢⱼ|``) is least: a weighted
    median, as ``dual_ascent``'s steps are. Those clear almost every column
    and then stall where a row would only trade one column's shortfall for
    another's of the same size (buildingenergy: a column in 26,278 rows
    ``xₖ − z <= 0`` whose duals sum past its cost). Then up to
    ``square_sweeps`` passes on the squared shortfall, which such a trade
    lowers, so they spread what is left until none is: the first
    ``full_sweeps`` of them over every row, the rest over the rows of the
    columns still on the wrong side only (``cptr``/``cidx``: the rows of
    each column). The last of the shortfall goes slowly (on
    buildingenergy 2,387 columns after the first passes, the largest by
    2.2; under 1e-4 after 5 squared passes, and 250 to clear the last
    few hundred, each by less than 1e-6), so those passes have to be
    cheap.

    Returns ``(y, wrong)``: ``wrong`` reduced costs still point at an
    infinite bound (0: the bound of ``y`` is rigorous, but for a free
    column's reduced cost, which is zero only within round-off).
    """
    if not (np.any(lower == -np.inf) or np.any(upper == np.inf)):
        return y.copy(), 0
    y = y.copy()
    _clip_duals(y, n_eq)
    # A free column's reduced cost has no side to be on: it is taken to
    # zero, and counted as there within round-off.
    margin = margin.copy()
    slack = np.zeros(c.size)
    for j in range(c.size):
        if lower[j] == -np.inf and upper[j] == np.inf:
            slack[j] = 1e-3 * margin[j]
            margin[j] = 0.0
    if sweeps > 0:
        y, _ = _sign_fix_abs(c, ptr, idx, val, n_eq, y, lower, upper, margin, sweeps)
    return _sign_fix_square(
        c, ptr, idx, val, cptr, cidx, n_eq, y, lower, upper, margin, slack,
        full_sweeps, square_sweeps,
    )
