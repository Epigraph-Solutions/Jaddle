"""
Cut separation from rows compiled with numba (optional:
``pip install jaddle[fast]``). ``jaddle_mip.CutSeparator(backend="numba")``
uses it, and ``backend="auto"`` picks it when numba is installed.

One call (``separate``) does a whole round: for each tight row and sense,
the lifted cover cuts of the row, its aggregation path, the c-MIR cut of
each level (with variable-bound substitution), the flow cover cut of each
level with a continuous variable, and the cover cuts of the levels beyond
the row. The cuts are cleaned (``_clean_cut``) and come back in the order
the numpy separators give them. Every function here mirrors the numpy one
of the same name in ``jaddle_mip`` step for step, so the cuts are the same
up to rounding in the last bits: sums here run in entry order, numpy's
pairwise or in BLAS.

The numpy separators make thousands of calls on rows of a few dozen
entries, where numpy's per-call cost is most of the time; the round also
runs without the GIL (``nogil``), so the search's other threads keep
running while it does.
"""

import math

import numba as nb
import numpy as np

COVER, MIR, FLOW = 0, 1, 2  # cut families, as ``separate`` reports them
vb_tie = 1e-6  # ``jaddle_mip._VB_TIE``


# %%
# Small helpers


@nb.njit(cache=True, inline="always")
def _norm(v):
    total = 0.0
    for k in range(v.size):
        total += v[k] * v[k]
    return math.sqrt(total)


@nb.njit(cache=True)
def _efficacy(idx, coef, rhs, x):
    """``(coef · x[idx] − rhs) / ‖coef‖``."""
    total = 0.0
    for k in range(idx.size):
        total += coef[k] * x[idx[k]]
    return (total - rhs) / _norm(coef)


@nb.njit(cache=True)
def _stable_by_two(primary, secondary):
    """The order ``np.lexsort((secondary, primary))`` gives: ascending
    ``primary``, ties by ascending ``secondary``, then by position."""
    order = np.argsort(secondary, kind="mergesort")
    return order[np.argsort(primary[order], kind="mergesort")]


@nb.njit(cache=True)
def _clean_cut(idx, coef, rhs, lower, upper, cut_tiny):
    """``jaddle_mip._clean_cut``: ``(ok, idx, coef, rhs)``."""
    keep = coef != 0.0
    idx, coef = idx[keep], coef[keep]
    if idx.size == 0 or not np.isfinite(rhs):
        return False, idx, coef, rhs
    largest = 0.0
    for k in range(coef.size):
        largest = max(largest, abs(coef[k]))
    tiny = np.abs(coef) < cut_tiny * largest
    if tiny.any():
        worst = 0.0
        for k in range(idx.size):
            if not tiny[k]:
                continue
            c = coef[k]
            w = c * lower[idx[k]] if c > 0 else c * upper[idx[k]]
            if not np.isfinite(w):
                return False, idx, coef, rhs
            worst += w
        rhs -= worst
        idx, coef = idx[~tiny], coef[~tiny]
    return True, idx, coef, rhs + 1e-9 * (1.0 + abs(rhs))


# %%
# Lifted knapsack covers


@nb.njit(cache=True)
def _knapsack(idx, a, b, x, lower, upper, is_binary, tol):
    """``jaddle_mip._knapsack``: ``(ok, cols, w, complemented, z,
    capacity)``."""
    no_i, no_f = np.empty(0, np.int64), np.empty(0)
    no_b = np.empty(0, np.bool_)
    k = idx.size
    n_bin = 0
    for p in range(k):
        if is_binary[idx[p]]:
            n_bin += 1
    if n_bin < 2:
        return False, no_i, no_f, no_b, no_f, 0.0
    low = 0.0
    for p in range(k):
        j = idx[p]
        if is_binary[j]:
            continue
        v = a[p] * lower[j] if a[p] > 0 else a[p] * upper[j]
        if not np.isfinite(v):
            return False, no_i, no_f, no_b, no_f, 0.0
        low += v
    cols = np.empty(n_bin, np.int64)
    w = np.empty(n_bin)
    comp = np.empty(n_bin, np.bool_)
    z = np.empty(n_bin)
    q = 0
    comp_sum = 0.0
    for p in range(k):
        j = idx[p]
        if not is_binary[j]:
            continue
        c = a[p] < 0
        weight = abs(a[p])
        if c:
            comp_sum += weight
        zp = 1.0 - x[j] if c else x[j]
        cols[q], w[q], comp[q], z[q] = j, weight, c, min(max(zp, 0.0), 1.0)
        q += 1
    capacity = b - low + comp_sum
    capacity += tol * (1.0 + max(abs(b), abs(capacity)))
    useful = w > 1e-12
    cols, w, comp, z = cols[useful], w[useful], comp[useful], z[useful]
    if capacity < 0 or w.sum() <= capacity + 1e-9 * (1.0 + abs(capacity)):
        return False, no_i, no_f, no_b, no_f, 0.0
    return True, cols, w, comp, z, capacity


@nb.njit(cache=True)
def _lift_cover(cover, w, z, capacity):
    """``jaddle_mip._lift_cover``: the lifted coefficients ``π``."""
    rhs = cover.size - 1
    pi = np.zeros(w.size)
    least = np.full(rhs + 1, np.inf)
    least[0] = 0.0
    outside = np.ones(w.size, np.bool_)
    for item in cover:
        pi[item] = 1.0
        outside[item] = False
        for v in range(rhs, 0, -1):  # downwards: each item once
            least[v] = min(least[v], least[v - 1] + w[item])
    slack = 1e-9 * (1.0 + abs(capacity))
    rest = np.flatnonzero(outside)
    for item in rest[_stable_by_two(-z[rest], -w[rest])]:
        room = capacity - w[item] + slack
        if room < 0:
            alpha = rhs
        else:
            top = rhs
            while least[top] > room:
                top -= 1
            alpha = rhs - top
        if alpha <= 0:
            continue
        pi[item] = alpha
        for v in range(rhs, alpha - 1, -1):
            least[v] = min(least[v], least[v - alpha] + w[item])
    return pi


@nb.njit(cache=True)
def cover_cuts(idx, a, b, x, lower, upper, is_binary, max_covers, tol):
    """``jaddle_mip.cover_cuts``: lists of the cuts' columns, coefficients
    and right-hand sides, most violated first."""
    out_idx, out_coef, out_rhs = [np.empty(0, np.int64)], [np.empty(0)], [0.0]
    out_idx.pop()
    out_coef.pop()
    out_rhs.pop()
    ok, cols, w, comp, z, capacity = _knapsack(
        idx, a, b, x, lower, upper, is_binary, tol
    )
    if not ok:
        return out_idx, out_coef, out_rhs
    threshold = capacity + 1e-9 * (1.0 + abs(capacity))
    integral = True
    load = 0.0
    for q in range(z.size):
        z_int = math.floor(z[q] + 0.5)
        if abs(z[q] - z_int) > 1e-9:
            integral = False
            break
        load += w[q] * z_int
    if integral and load <= threshold:
        return out_idx, out_coef, out_rhs
    heaviest = np.argsort(-w, kind="mergesort")
    orders = [
        np.argsort((1.0 - z) / w, kind="mergesort"),
        _stable_by_two(1.0 - z, -w),
        heaviest,
    ]
    z_total = z.sum()  # z is in [0, 1]: its positive part is itself
    seen = [np.empty(0, np.int64)]
    seen.pop()
    efficacies = [0.0]
    efficacies.pop()
    for o in range(min(max_covers, 3)):
        order = orders[o]
        total = 0.0
        end = -1
        for p in range(order.size):
            total += w[order[p]]
            if total > threshold:
                end = p
                break
        if end < 0:
            continue
        cover = order[: end + 1]
        total = 0.0
        for item in cover:
            total += w[item]
        keep = np.ones(cover.size, np.bool_)
        for position in np.argsort(z[cover], kind="mergesort"):
            if total - w[cover[position]] > threshold:
                total -= w[cover[position]]
                keep[position] = False
        cover = np.sort(cover[keep])
        repeated = False
        for other in seen:
            if other.size == cover.size and np.all(other == cover):
                repeated = True
                break
        if repeated:
            continue
        seen.append(cover)
        rhs_z = cover.size - 1.0
        in_cover = 0.0
        for item in cover:
            in_cover += z[item]
        if in_cover - rhs_z + rhs_z * (z_total - in_cover) <= 1e-6:
            continue
        pi = _lift_cover(cover, w, z, capacity)
        violation = 0.0
        for q in range(pi.size):
            violation += pi[q] * z[q]
        violation -= rhs_z
        if violation <= 1e-6:
            continue
        members = np.flatnonzero(pi)
        coef = np.empty(members.size)
        rhs = rhs_z
        for k in range(members.size):
            q = members[k]
            coef[k] = -pi[q] if comp[q] else pi[q]
            if comp[q]:
                rhs -= pi[q]
        efficacies.append(violation / _norm(pi[members]))
        out_idx.append(cols[members])
        out_coef.append(coef)
        out_rhs.append(rhs)
    if len(efficacies) > 1:
        order = np.argsort(-np.array(efficacies), kind="mergesort")
        out_idx = [out_idx[k] for k in order]
        out_coef = [out_coef[k] for k in order]
        out_rhs = [out_rhs[k] for k in order]
    return out_idx, out_coef, out_rhs


# %%
# c-MIR


@nb.njit(cache=True)
def _cmir_for(a, b, xv, lo, up, integer, use_upper, delta):
    """``jaddle_mip._cmir_for``: ``(ok, efficacy, coef, rhs)``."""
    k = a.size
    beta = b
    for p in range(k):
        beta -= a[p] * (up[p] if use_upper[p] else lo[p])
    beta /= delta
    f = beta - math.floor(beta)
    if f < 0.01 or f > 0.99:
        return False, 0.0, np.empty(0), 0.0
    coef = np.empty(k)
    for p in range(k):
        sign = -1.0 if use_upper[p] else 1.0
        at = a[p] * sign
        if integer[p]:
            g = at / delta
            fj = g - math.floor(g)
            coef[p] = (math.floor(g) + max(fj - f, 0.0) / (1.0 - f)) * sign
        elif at < 0:
            coef[p] = at / (delta * (1.0 - f)) * sign
        else:
            coef[p] = 0.0
    norm = _norm(coef)
    if norm == 0:
        return False, 0.0, coef, 0.0
    rhs = math.floor(beta)
    lhs = 0.0
    for p in range(k):
        rhs += coef[p] * (up[p] if use_upper[p] else lo[p])
        lhs += coef[p] * xv[p]
    return True, (lhs - rhs) / norm, coef, rhs


@nb.njit(cache=True)
def _cmir_core(a, b, xv, lo, up, integer, max_divisors):
    """``jaddle_mip._cmir_search`` over gathered entries: ``(ok, coef,
    rhs)`` with ``coef`` aligned with the entries."""
    k = a.size
    if not integer.any():
        return False, np.empty(0), 0.0
    use_upper = np.empty(k, np.bool_)
    count = 0
    candidates = np.empty(k)
    for p in range(k):
        use_upper[p] = np.isfinite(up[p]) and (
            not np.isfinite(lo[p]) or up[p] - xv[p] < xv[p] - lo[p]
        )
        bound = up[p] if use_upper[p] else lo[p]
        if not np.isfinite(bound):
            return False, np.empty(0), 0.0
        t = max(up[p] - xv[p] if use_upper[p] else xv[p] - lo[p], 0.0)
        if integer[p] and t > 1e-6 and abs(a[p]) > 1e-9:
            candidates[count] = abs(a[p])
            count += 1
    if count == 0:
        return False, np.empty(0), 0.0
    distinct = np.unique(candidates[:count])[::-1][:max_divisors]
    divisors = np.empty(distinct.size + 1)
    divisors[:-1] = distinct
    divisors[-1] = 1.0

    found = False
    best_eff, best_coef, best_rhs, best_delta = 0.0, np.empty(0), 0.0, 1.0
    for delta in divisors:
        ok, eff, coef, rhs = _cmir_for(a, b, xv, lo, up, integer, use_upper, delta)
        if ok and (not found or eff > best_eff):
            found = True
            best_eff, best_coef, best_rhs, best_delta = eff, coef, rhs, delta
    if not found:
        return False, np.empty(0), 0.0
    for scale in (2.0, 4.0, 8.0):
        delta = best_delta / scale
        ok, eff, coef, rhs = _cmir_for(a, b, xv, lo, up, integer, use_upper, delta)
        if ok and eff > best_eff:
            best_eff, best_coef, best_rhs, best_delta = eff, coef, rhs, delta
    # Complementation flips, for integers strictly inside finite bounds.
    inside = np.empty(k, np.bool_)
    distance = np.empty(k)
    for p in range(k):
        inside[p] = (
            integer[p]
            and np.isfinite(lo[p])
            and np.isfinite(up[p])
            and xv[p] > lo[p] + 1e-6
            and xv[p] < up[p] - 1e-6
        )
        distance[p] = min(xv[p] - lo[p], up[p] - xv[p])
    which = np.flatnonzero(inside)
    for j in which[np.argsort(-distance[which], kind="mergesort")]:
        flipped = use_upper.copy()
        flipped[j] = not flipped[j]
        ok, eff, coef, rhs = _cmir_for(a, b, xv, lo, up, integer, flipped, best_delta)
        if ok and eff > best_eff + 1e-12:
            best_eff, best_coef, best_rhs = eff, coef, rhs
            use_upper = flipped
    if best_eff <= 1e-9:
        return False, np.empty(0), 0.0
    return True, best_coef, best_rhs


@nb.njit(cache=True)
def _cmir_search(idx, a, b, x, lower, upper, is_int, max_divisors):
    """``jaddle_mip._cmir_search``: ``(ok, coef, rhs)`` over ``idx``."""
    return _cmir_core(a, b, x[idx], lower[idx], upper[idx], is_int[idx], max_divisors)


@nb.njit(cache=True)
def _substituted_cmir(
    idx, a, b, x, lower, upper, is_int, max_divisors,
    vb_ptr, vb_kind, vb_z, vb_c, vb_d, pos, acc,
):
    """``jaddle_mip._substituted_cmir``: ``(ok, idx, coef, rhs)``. The
    variable bounds of column ``y`` are entries ``vb_ptr[y]:vb_ptr[y + 1]``
    (kind 0 for an upper bound ``y <= d + c·z``, 1 for a lower one; upper
    ones first, as the dict lists them). ``pos`` (all −1) and ``acc`` (all
    0) are scratch over the columns, left as they were found."""
    n = x.size
    size = idx.size
    no_i, no_f = np.empty(0, np.int64), np.empty(0)
    # ``coef`` as an insertion-ordered dict: keys, values, and alive.
    keys = np.empty(2 * size, np.int64)
    vals = np.empty(2 * size)
    alive = np.ones(2 * size, np.bool_)
    n_keys = 0
    for p in range(size):
        j = idx[p]
        if pos[j] < 0:
            pos[j] = n_keys
            keys[n_keys] = j
            n_keys += 1
        vals[pos[j]] = a[p]
    sub_kind = np.empty(size, np.int64)
    sub_y = np.empty(size, np.int64)
    sub_z = np.empty(size, np.int64)
    sub_c = np.empty(size)
    sub_d = np.empty(size)
    t_coef = np.empty(size)
    t_value = np.empty(size)
    n_subs = 0
    for p in range(size):
        y = idx[p]
        if is_int[y] or vb_ptr[y] == vb_ptr[y + 1]:
            continue
        a_y, x_y = vals[pos[y]], x[y]
        # Ties go to the side whose shifted term the MIR drops (kind 1, a
        # lower bound, when a_y > 0; kind 0 when a_y < 0), as in jaddle_mip.
        prefer = 1 if a_y > 0 else 0
        lb_distance, ub_distance = x_y - lower[y], upper[y] - x_y
        best_distance = min(lb_distance, ub_distance)
        held = 1 if lb_distance <= ub_distance else 0
        best = -1
        for e in range(vb_ptr[y], vb_ptr[y + 1]):
            implied = vb_d[e] + vb_c[e] * x[vb_z[e]]
            distance = implied - x_y if vb_kind[e] == 0 else x_y - implied
            if distance < best_distance - vb_tie:
                better = True
            elif distance <= best_distance + vb_tie:
                better = vb_kind[e] == prefer or held != prefer
            else:
                better = False
            if better:
                best_distance, best, held = distance, e, vb_kind[e]
        if best < 0:
            continue
        kind, z, c, d = vb_kind[best], vb_z[best], vb_c[best], vb_d[best]
        alive[pos[y]] = False
        if pos[z] < 0:
            pos[z] = n_keys
            keys[n_keys] = z
            vals[n_keys] = 0.0
            n_keys += 1
        vals[pos[z]] += a_y * c
        b = b - a_y * d
        implied = d + c * x[z]
        sub_kind[n_subs], sub_y[n_subs], sub_z[n_subs] = kind, y, z
        sub_c[n_subs], sub_d[n_subs] = c, d
        t_coef[n_subs] = -a_y if kind == 0 else a_y
        t_value[n_subs] = max(implied - x_y if kind == 0 else x_y - implied, 0.0)
        n_subs += 1
    for q in range(n_keys):
        pos[keys[q]] = -1
    if n_subs == 0:
        return False, no_i, no_f, 0.0
    # The search over the remaining columns and the ``t``s.
    total = n_keys + n_subs
    e_idx = np.empty(total, np.int64)
    e_a = np.empty(total)
    m = 0
    for q in range(n_keys):
        if alive[q]:
            e_idx[m], e_a[m] = keys[q], vals[q]
            m += 1
    for s in range(n_subs):
        e_idx[m], e_a[m] = n + s, t_coef[s]
        m += 1
    e_idx, e_a = e_idx[:m], e_a[:m]
    largest = 1.0
    for q in range(m):
        largest = max(largest, abs(e_a[q]))
    keep = np.abs(e_a) > 1e-12 * largest
    e_idx, e_a = e_idx[keep], e_a[keep]
    m = e_idx.size
    e_x, e_lo, e_up = np.empty(m), np.empty(m), np.empty(m)
    e_int = np.empty(m, np.bool_)
    for q in range(m):
        j = e_idx[q]
        if j < n:
            e_x[q], e_lo[q], e_up[q], e_int[q] = x[j], lower[j], upper[j], is_int[j]
        else:
            e_x[q], e_lo[q], e_up[q], e_int[q] = t_value[j - n], 0.0, np.inf, False
    ok, c_coef, rhs = _cmir_core(e_a, b, e_x, e_lo, e_up, e_int, max_divisors)
    if not ok:
        return False, no_i, no_f, 0.0
    # Back to the columns through each ``t``'s bound.
    touched = np.empty(2 * m, np.int64)
    n_touched = 0
    for q in range(m):
        j, g = e_idx[q], c_coef[q]
        if j < n:
            if pos[j] < 0:
                pos[j] = 0
                touched[n_touched] = j
                n_touched += 1
            acc[j] += g
            continue
        s = j - n
        sign = 1.0 if sub_kind[s] == 0 else -1.0  # t = sign·(d + c·z − y)
        rhs -= sign * g * sub_d[s]
        for col, v in ((sub_z[s], sign * g * sub_c[s]), (sub_y[s], -sign * g)):
            if pos[col] < 0:
                pos[col] = 0
                touched[n_touched] = col
                n_touched += 1
            acc[col] += v
    o_idx = np.sort(touched[:n_touched])
    o_coef = np.empty(n_touched)
    largest = 1.0
    for q in range(n_touched):
        o_coef[q] = acc[o_idx[q]]
        acc[o_idx[q]] = 0.0
        pos[o_idx[q]] = -1
        largest = max(largest, abs(o_coef[q]))
    nonzero = np.abs(o_coef) > 1e-12 * largest
    o_idx, o_coef = o_idx[nonzero], o_coef[nonzero]
    if o_idx.size == 0:
        return False, no_i, no_f, 0.0
    if _efficacy(o_idx, o_coef, rhs, x) <= 1e-9:
        return False, no_i, no_f, 0.0
    return True, o_idx, o_coef, rhs


@nb.njit(cache=True)
def cmir_cut(
    idx, a, b, x, lower, upper, is_int, max_divisors,
    has_vb, vb_ptr, vb_kind, vb_z, vb_c, vb_d, pos, acc,
):
    """``jaddle_mip.cmir_cut``: ``(ok, idx, coef, rhs)``."""
    plain, p_coef, p_rhs = _cmir_search(idx, a, b, x, lower, upper, is_int, max_divisors)
    if not has_vb:
        return plain, idx, p_coef, p_rhs
    sub, s_idx, s_coef, s_rhs = _substituted_cmir(
        idx, a, b, x, lower, upper, is_int, max_divisors,
        vb_ptr, vb_kind, vb_z, vb_c, vb_d, pos, acc,
    )
    if not sub:
        return plain, idx, p_coef, p_rhs
    if not plain or _efficacy(s_idx, s_coef, s_rhs, x) > _efficacy(
        idx, p_coef, p_rhs, x
    ):
        return True, s_idx, s_coef, s_rhs
    return True, idx, p_coef, p_rhs


# %%
# Flow covers


@nb.njit(cache=True)
def _flow_cut_columns(
    arc_fcol, arc_fcoef, arc_f0, arc_w, term_arc, term_cf, term_cw, rhs, pos, acc
):
    """``jaddle_mip._flow_cut_columns``: ``(ok, idx, coef, rhs)``."""
    touched = np.empty(2 * term_arc.size, np.int64)
    n_touched = 0
    for t in range(term_arc.size):
        i, cf, cw = term_arc[t], term_cf[t], term_cw[t]
        if cf != 0.0:
            j = arc_fcol[i]
            if pos[j] < 0:
                pos[j] = 0
                touched[n_touched] = j
                n_touched += 1
            acc[j] += cf * arc_fcoef[i]
            rhs -= cf * arc_f0[i]
        if cw != 0.0:
            if arc_w[i] < 0:
                rhs -= cw  # w ≡ 1
            else:
                j = arc_w[i]
                if pos[j] < 0:
                    pos[j] = 0
                    touched[n_touched] = j
                    n_touched += 1
                acc[j] += cw
    cols = np.sort(touched[:n_touched])
    coef = np.empty(n_touched)
    for q in range(n_touched):
        coef[q] = acc[cols[q]]
        acc[cols[q]] = 0.0
        pos[cols[q]] = -1
    keep = np.abs(coef) > 1e-12
    cols, coef = cols[keep], coef[keep]
    return cols.size > 0, cols, coef, rhs


@nb.njit(cache=True)
def _gfci(arc_sign, arc_u, arc_fval, arc_wval, b, plus, minus):
    """``jaddle_mip._gfci``: ``(ok, term_arc, term_cf, term_cw, rhs)``."""
    sum_plus = 0.0
    for i in plus:
        sum_plus += arc_u[i]
    sum_minus = 0.0
    for i in minus:
        sum_minus += arc_u[i]
    lam = sum_plus - sum_minus - b
    n_arcs = arc_sign.size
    term_arc = np.empty(plus.size + n_arcs, np.int64)
    term_cf = np.empty(plus.size + n_arcs)
    term_cw = np.empty(plus.size + n_arcs)
    if lam <= 1e-9:
        return False, term_arc[:0], term_cf[:0], term_cw[:0], 0.0
    rhs = b + sum_minus
    t = 0
    for i in plus:
        excess = max(arc_u[i] - lam, 0.0)
        term_arc[t], term_cf[t], term_cw[t] = i, 1.0, -excess
        rhs -= excess
        t += 1
    in_minus = np.zeros(n_arcs, np.bool_)
    for i in minus:
        in_minus[i] = True
    for i in range(n_arcs):
        if arc_sign[i] > 0 or in_minus[i]:
            continue
        if lam * arc_wval[i] < arc_fval[i]:
            term_arc[t], term_cf[t], term_cw[t] = i, 0.0, -lam
        else:
            term_arc[t], term_cf[t], term_cw[t] = i, -1.0, 0.0
        t += 1
    return True, term_arc[:t], term_cf[:t], term_cw[:t], rhs


@nb.njit(cache=True)
def flow_cover_cut(
    idx, a, b, x, lower, upper, is_int, has_vb, vb_ptr, vb_kind, vb_z, vb_c, vb_d,
    pos, acc,
):
    """``jaddle_mip.flow_cover_cut``: ``(ok, idx, coef, rhs)``."""
    no_i, no_f = np.empty(0, np.int64), np.empty(0)
    size = idx.size
    # The flow set (``_flow_set``), one arc per nonzero term.
    arc_sign = np.empty(size, np.int64)
    arc_u = np.empty(size)
    arc_fcol = np.empty(size, np.int64)
    arc_fcoef = np.empty(size)
    arc_f0 = np.empty(size)
    arc_w = np.empty(size, np.int64)  # −1: w ≡ 1
    arc_fval = np.empty(size)
    arc_wval = np.empty(size)
    n_arcs = 0
    for p in range(size):
        j, aj = idx[p], a[p]
        if aj == 0.0:
            continue
        sign, mag = (1, aj) if aj > 0 else (-1, -aj)
        binary = is_int[j] and lower[j] == 0.0 and upper[j] == 1.0
        vub_z, vub_c = -1, 0.0
        if not is_int[j] and lower[j] == 0.0 and has_vb:
            for e in range(vb_ptr[j], vb_ptr[j + 1]):
                if vb_kind[e] != 0:
                    continue
                z, c, d = vb_z[e], vb_c[e], vb_d[e]
                if d == 0.0 and c > 0 and is_int[z] and lower[z] == 0.0 and upper[z] == 1.0:
                    if vub_z < 0 or c * x[z] < vub_c * x[vub_z]:
                        vub_z, vub_c = z, c  # the tightest at x
        k = n_arcs
        if vub_z >= 0:
            arc_sign[k], arc_u[k], arc_fcol[k], arc_fcoef[k] = sign, mag * vub_c, j, mag
            arc_f0[k], arc_w[k], arc_fval[k], arc_wval[k] = 0.0, vub_z, mag * x[j], x[vub_z]
        elif binary:
            arc_sign[k], arc_u[k], arc_fcol[k], arc_fcoef[k] = sign, mag, j, mag
            arc_f0[k], arc_w[k], arc_fval[k], arc_wval[k] = 0.0, j, mag * x[j], x[j]
        else:
            lo, up = lower[j], upper[j]
            if np.isfinite(up) and (not np.isfinite(lo) or up - x[j] < x[j] - lo):
                b -= aj * up  # y = up − t: the term flips to the other side
                arc_sign[k], arc_u[k], arc_fcol[k], arc_fcoef[k] = -sign, mag * (up - lo), j, -mag
                arc_f0[k], arc_w[k] = mag * up, -1
                arc_fval[k], arc_wval[k] = mag * (up - x[j]), 1.0
            elif np.isfinite(lo):
                b -= aj * lo
                arc_sign[k], arc_u[k], arc_fcol[k], arc_fcoef[k] = sign, mag * (up - lo), j, mag
                arc_f0[k], arc_w[k] = -mag * lo, -1
                arc_fval[k], arc_wval[k] = mag * (x[j] - lo), 1.0
            else:
                return False, no_i, no_f, 0.0  # a free variable
        n_arcs += 1
    if n_arcs == 0:
        return False, no_i, no_f, 0.0
    arc_sign, arc_u = arc_sign[:n_arcs], arc_u[:n_arcs]
    arc_fval, arc_wval = arc_fval[:n_arcs], arc_wval[:n_arcs]
    finite = np.isfinite(arc_u)
    plus = np.flatnonzero((arc_sign > 0) & finite)
    minus = np.flatnonzero((arc_sign < 0) & finite)
    saturated = minus[
        (arc_wval[minus] > 1 - 1e-6) & (arc_fval[minus] >= arc_u[minus] - 1e-6)
    ]
    openness = (1 - arc_wval[plus]) / np.maximum(arc_u[plus], 1e-12)
    order = plus[_stable_by_two(openness, -arc_u[plus])]
    found = False
    best_idx, best_coef, best_rhs, best_eff = no_i, no_f, 0.0, 1e-6
    for side in range(2):
        cover_minus = minus[:0] if side == 0 else saturated
        need_minus = 0.0
        for i in cover_minus:
            need_minus += arc_u[i]
        need = b + need_minus
        end, total = 0, 0.0
        for i in order:
            end += 1
            total += arc_u[i]
            if total > need + 1e-9:
                break
        if total <= need + 1e-9:
            continue
        cover = order[:end]
        keep = np.ones(end, np.bool_)
        for position in np.argsort(arc_wval[cover], kind="mergesort"):
            if total - arc_u[cover[position]] > need + 1e-9:
                keep[position] = False
                total -= arc_u[cover[position]]
        for trimmed in range(2):
            cover_plus = cover[keep] if trimmed else cover
            ok, t_arc, t_cf, t_cw, rhs = _gfci(
                arc_sign, arc_u, arc_fval, arc_wval, b, cover_plus, cover_minus
            )
            if not ok:
                continue
            ok, c_idx, c_coef, c_rhs = _flow_cut_columns(
                arc_fcol, arc_fcoef, arc_f0, arc_w, t_arc, t_cf, t_cw, rhs, pos, acc
            )
            if not ok:
                continue
            eff = _efficacy(c_idx, c_coef, c_rhs, x)
            if eff > best_eff:
                found = True
                best_idx, best_coef, best_rhs, best_eff = c_idx, c_coef, c_rhs, eff
    return found, best_idx, best_coef, best_rhs


# %%
# Aggregation paths and the round


@nb.njit(cache=True)
def _cancel_order(idx, a, x, lower, upper, is_int, aggregate_integers):
    """``CutSeparator._cancel_order``: the positions worth cancelling, best
    first."""
    k = idx.size
    score = np.empty(k)
    best = -np.inf
    for p in range(k):
        j = idx[p]
        if is_int[j]:
            d = -1.0
        else:
            d = min(x[j] - lower[j], upper[j] - x[j])
            if np.isnan(d):
                d = 0.0
            elif d == np.inf:
                d = 1e30
            elif d == -np.inf:
                d = -np.finfo(np.float64).max
        score[p] = d
        best = max(best, d)
    if best <= 1e-6:
        if not aggregate_integers:
            return np.empty(0, np.int64)
        for p in range(k):
            j = idx[p]
            if is_int[j]:
                frac = x[j] - math.floor(x[j])
                score[p] = abs(a[p]) * min(frac, 1.0 - frac)
            else:
                score[p] = -1.0
    order = np.argsort(-score, kind="mergesort")
    count = 0
    while count < k and score[order[count]] > 1e-6:
        count += 1
    return order[:count]


@nb.njit(cache=True)
def _partner(col, coef, used, C_ptr, C_row, C_val, relative, is_eq):
    """``CutSeparator._partner``: ``(row, weight)``, row ``-1`` for none."""
    c_start, c_end = C_ptr[col], C_ptr[col + 1]
    rows = C_row[c_start:c_end]
    for q in np.argsort(relative[rows], kind="mergesort"):
        r2, a2 = rows[q], C_val[c_start + q]
        if a2 == 0 or r2 in used:
            continue
        weight = -coef / a2
        if weight < 0 and not is_eq[r2]:
            continue  # a <= row can only be added with a weight >= 0
        return r2, weight
    return -1, 0.0


@nb.njit(cache=True)
def _add_row(idx, a, b, col, partner, lam, A_ptr, A_idx, A_val, b_rows):
    """``CutSeparator._add_row``."""
    s2, e2 = A_ptr[partner], A_ptr[partner + 1]
    both = np.concatenate((idx, A_idx[s2:e2]))
    values = np.concatenate((a, lam * A_val[s2:e2]))
    order = np.argsort(both, kind="mergesort")
    merged_idx = np.empty(both.size, np.int64)
    merged = np.empty(both.size)
    m = -1
    for q in order:
        if m < 0 or both[q] != merged_idx[m]:
            m += 1
            merged_idx[m] = both[q]
            merged[m] = 0.0
        merged[m] += values[q]
    m += 1
    merged_idx, merged = merged_idx[:m], merged[:m]
    largest = 1.0
    for q in range(m):
        if merged_idx[q] == col:
            merged[q] = 0.0
        largest = max(largest, abs(merged[q]))
    nonzero = np.abs(merged) > 1e-12 * largest
    return merged_idx[nonzero], merged[nonzero], b + lam * b_rows[partner]


@nb.njit(cache=True)
def aggregation_path(
    r, s, x, lower, upper, is_int, A_ptr, A_idx, A_val, b_rows, is_eq,
    C_ptr, C_row, C_val, relative, max_aggregation, aggregate_integers,
    aggregation_width,
):
    """``CutSeparator._aggregation_path``: lists of each level's columns,
    coefficients and right-hand side."""
    start, end = A_ptr[r], A_ptr[r + 1]
    idx = A_idx[start:end].copy()
    a = s * A_val[start:end]
    b = s * b_rows[r]
    path_idx, path_a, path_b = [idx], [a], [b]
    if max_aggregation <= 0 or idx.size == 0:
        return path_idx, path_a, path_b
    start_col = [0]
    start_col.pop()
    start_row = [0]
    start_row.pop()
    start_lam = [0.0]
    start_lam.pop()
    first = [r]
    order = _cancel_order(idx, a, x, lower, upper, is_int, aggregate_integers)
    for pos in order[:aggregation_width]:
        r2, lam = _partner(idx[pos], a[pos], first, C_ptr, C_row, C_val, relative, is_eq)
        if r2 < 0:
            continue
        seen = False
        for q in range(len(start_row)):
            if start_row[q] == r2 and start_lam[q] == lam:
                seen = True
        if not seen:
            start_col.append(idx[pos])
            start_row.append(r2)
            start_lam.append(lam)
    for branch in range(len(start_row)):
        col, r2, lam = start_col[branch], start_row[branch], start_lam[branch]
        used = [r]
        level_idx, level_a, level_b = idx, a, b
        for level in range(max_aggregation):
            if level > 0:
                order = _cancel_order(
                    level_idx, level_a, x, lower, upper, is_int, aggregate_integers
                )
                if order.size == 0:
                    break
                pos = order[0]
                col = level_idx[pos]
                r2, lam = _partner(
                    col, level_a[pos], used, C_ptr, C_row, C_val, relative, is_eq
                )
                if r2 < 0:
                    break
            level_idx, level_a, level_b = _add_row(
                level_idx, level_a, level_b, col, r2, lam, A_ptr, A_idx, A_val, b_rows
            )
            used.append(r2)
            if level_idx.size == 0:
                break
            path_idx.append(level_idx)
            path_a.append(level_a)
            path_b.append(level_b)
    return path_idx, path_a, path_b


@nb.njit(cache=True)
def _add(fams, cut_idx, cut_coef, cut_rhs, family, idx, coef, rhs, lower, upper, cut_tiny):
    """Append the cut, cleaned (``_clean_cut``), unless cleaning drops it."""
    ok, idx, coef, rhs = _clean_cut(idx, coef, rhs, lower, upper, cut_tiny)
    if ok:
        fams.append(family)
        cut_idx.append(idx)
        cut_coef.append(coef)
        cut_rhs.append(rhs)


@nb.njit(cache=True, nogil=True)
def separate(
    rows, x, lower, upper, is_int, is_binary,
    A_ptr, A_idx, A_val, b_rows, is_eq, C_ptr, C_row, C_val, relative,
    has_vb, vb_ptr, vb_kind, vb_z, vb_c, vb_d,
    max_aggregation, aggregate_integers, flow_covers, max_divisors, max_covers,
    knapsack_tol, cut_tiny, aggregation_width=1, cmir_per_row=1,
):
    """``CutSeparator._separate`` without the clique and implied-bound cuts:
    the cleaned cuts of the rows ``rows`` (already the tight ones, in
    order) as ``(family, ptr, idx, coef, rhs)``, cut ``k`` being family
    ``family[k]`` (``COVER``, ``MIR``, ``FLOW``) over
    ``idx[ptr[k]:ptr[k + 1]]``."""
    n = x.size
    pos = np.full(n, -1, np.int64)
    acc = np.zeros(n)
    fams = [0]
    fams.pop()
    cut_idx = [np.empty(0, np.int64)]
    cut_idx.pop()
    cut_coef = [np.empty(0)]
    cut_coef.pop()
    cut_rhs = [0.0]
    cut_rhs.pop()

    for r in rows:
        start, end = A_ptr[r], A_ptr[r + 1]
        if end == start:
            continue
        for s in (1.0, -1.0):
            if s < 0 and not is_eq[r]:
                break
            c_idx, c_coef, c_rhs = cover_cuts(
                A_idx[start:end], s * A_val[start:end], s * b_rows[r], x,
                lower, upper, is_binary, max_covers, knapsack_tol,
            )
            for k in range(len(c_idx)):
                _add(fams, cut_idx, cut_coef, cut_rhs, COVER, c_idx[k], c_coef[k],
                     c_rhs[k], lower, upper, cut_tiny)
            p_idx, p_a, p_b = aggregation_path(
                r, s, x, lower, upper, is_int, A_ptr, A_idx, A_val, b_rows, is_eq,
                C_ptr, C_row, C_val, relative, max_aggregation, aggregate_integers,
                aggregation_width,
            )
            # ``_path_cuts``: the ``cmir_per_row`` best c-MIR cuts over the
            # levels, then each level's flow cover cut and, beyond the row,
            # its cover cuts.
            mir_eff = [0.0]
            mir_eff.pop()
            mir_idx = [np.empty(0, np.int64)]
            mir_idx.pop()
            mir_coef = [np.empty(0)]
            mir_coef.pop()
            mir_rhs = [0.0]
            mir_rhs.pop()
            extra_fam = [0]
            extra_fam.pop()
            extra_idx = [np.empty(0, np.int64)]
            extra_idx.pop()
            extra_coef = [np.empty(0)]
            extra_coef.pop()
            extra_rhs = [0.0]
            extra_rhs.pop()
            for level in range(len(p_idx)):
                idx, a, b = p_idx[level], p_a[level], p_b[level]
                ok, m_idx, m_coef, m_rhs = cmir_cut(
                    idx, a, b, x, lower, upper, is_int, max_divisors,
                    has_vb, vb_ptr, vb_kind, vb_z, vb_c, vb_d, pos, acc,
                )
                if ok:
                    eff = _efficacy(m_idx, m_coef, m_rhs, x)
                    if eff > 0.0:
                        mir_eff.append(eff)
                        mir_idx.append(m_idx)
                        mir_coef.append(m_coef)
                        mir_rhs.append(m_rhs)
                if flow_covers and not is_int[idx].all():
                    ok, f_idx, f_coef, f_rhs = flow_cover_cut(
                        idx, a, b, x, lower, upper, is_int,
                        has_vb, vb_ptr, vb_kind, vb_z, vb_c, vb_d, pos, acc,
                    )
                    if ok:
                        extra_fam.append(FLOW)
                        extra_idx.append(f_idx)
                        extra_coef.append(f_coef)
                        extra_rhs.append(f_rhs)
                if level > 0:
                    c_idx, c_coef, c_rhs = cover_cuts(
                        idx, a, b, x, lower, upper, is_binary, max_covers, knapsack_tol
                    )
                    for k in range(len(c_idx)):
                        extra_fam.append(COVER)
                        extra_idx.append(c_idx[k])
                        extra_coef.append(c_coef[k])
                        extra_rhs.append(c_rhs[k])
            if len(mir_eff):
                effs = np.empty(len(mir_eff))
                for k in range(len(mir_eff)):
                    effs[k] = -mir_eff[k]
                for k in np.argsort(effs, kind="mergesort")[:cmir_per_row]:
                    _add(fams, cut_idx, cut_coef, cut_rhs, MIR, mir_idx[k],
                         mir_coef[k], mir_rhs[k], lower, upper, cut_tiny)
            for k in range(len(extra_fam)):
                _add(fams, cut_idx, cut_coef, cut_rhs, extra_fam[k], extra_idx[k],
                     extra_coef[k], extra_rhs[k], lower, upper, cut_tiny)

    count = len(fams)
    ptr = np.zeros(count + 1, np.int64)
    for k in range(count):
        ptr[k + 1] = ptr[k] + cut_idx[k].size
    family = np.empty(count, np.int64)
    flat_idx = np.empty(ptr[count], np.int64)
    flat_coef = np.empty(ptr[count])
    rhs = np.empty(count)
    for k in range(count):
        family[k] = fams[k]
        flat_idx[ptr[k] : ptr[k + 1]] = cut_idx[k]
        flat_coef[ptr[k] : ptr[k + 1]] = cut_coef[k]
        rhs[k] = cut_rhs[k]
    return family, ptr, flat_idx, flat_coef, rhs
