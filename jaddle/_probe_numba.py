"""
Probing compiled with numba (optional: ``pip install jaddle[fast]``).
``jaddle_mip.probe(backend="numba")`` uses it, and ``backend="auto"`` picks it
when numba is installed.

Each side of a probe is propagated incrementally: only the rows of the
variables whose bounds changed are visited, in passes (the rows queued by
one pass are the next pass's), each row's activity recomputed from the box
when it is visited. A side's changes are logged and undone after it. The
rules are ``BoundPropagator``'s (``jaddle._propagate_numba``): the same
bound from each row, integer bounds rounded, only clear improvements of a
sane size kept, and the same infeasibility tests. The full-sweep
propagators visit every row in every pass, so a probe that changes a few
hundred bounds cost a pass over the whole matrix for each of its passes;
here it costs the rows those bounds are in.

A propagation that runs to its fixed point reaches the same box either
way; within ``max_passes`` the boxes can differ, since a pass here uses the
bounds an earlier row of the same pass found.
"""

import math

import numba as nb
import numpy as np


@nb.njit(cache=True, inline="always")
def _split(lo, up):
    """``jaddle_mip._probe_split`` of one variable."""
    lo_ok, up_ok = math.isfinite(lo), math.isfinite(up)
    if lo_ok and up_ok:
        return min(math.floor(0.5 * (lo + up)), up - 1.0)
    if lo_ok:
        return lo
    if up_ok:
        return up - 1.0
    return 0.0


@nb.njit(cache=True, inline="always")
def _log(j, lower, upper, mark, stamp, log_var, log_lo, log_up, n_log):
    """Log ``j``'s bounds the first time it changes under ``stamp``."""
    if mark[j] != stamp:
        mark[j] = stamp
        log_var[n_log] = j
        log_lo[n_log] = lower[j]
        log_up[n_log] = upper[j]
        n_log += 1
    return n_log


@nb.njit(cache=True)
def _row_activity(i, row_ptr, row_col, row_val, lower, upper):
    """Row ``i``'s least and largest finite activity and the counts of
    infinite contributions to each."""
    min_act = 0.0
    max_act = 0.0
    n_min = 0
    n_max = 0
    for k in range(row_ptr[i], row_ptr[i + 1]):
        a = row_val[k]
        j = row_col[k]
        if a > 0:
            min_c, max_c = a * lower[j], a * upper[j]
        else:
            min_c, max_c = a * upper[j], a * lower[j]
        if math.isfinite(min_c):
            min_act += min_c
        else:
            n_min += 1
        if math.isfinite(max_c):
            max_act += max_c
        else:
            n_max += 1
    return min_act, max_act, n_min, n_max


@nb.njit(cache=True, inline="always")
def _missed(min_act, max_act, n_min, n_max, rl, ru, tol):
    """Whether the row's whole activity range misses its bounds."""
    if n_min == 0 and min_act > ru + tol * (1.0 + abs(ru)):
        return True
    if n_max == 0 and max_act < rl - tol * (1.0 + abs(rl)):
        return True
    return False


@nb.njit(cache=True)
def _propagate(
    row_ptr, row_col, row_val, col_ptr, col_row, row_lower, row_upper, is_int,
    lower, upper, seeds, n_seeds, max_passes, tol, round_slack,
    queued, cur, nxt, mark, stamp, log_var, log_lo, log_up, n_log,
):
    """Propagate from the variables ``seeds[:n_seeds]`` (whose bounds just
    changed) in place, logging each changed variable once under ``stamp``.
    Returns ``(infeasible, n_log)``. ``queued`` is all False on entry and
    on return."""
    n_next = 0
    for s in range(n_seeds):
        j = seeds[s]
        for k in range(col_ptr[j], col_ptr[j + 1]):
            i = col_row[k]
            if not queued[i]:
                queued[i] = True
                nxt[n_next] = i
                n_next += 1
    passes = 0
    while n_next > 0:
        if passes >= max_passes:
            # The passes ran out with rows still queued: check them against
            # the final box, as the full sweep checks every row.
            infeasible = False
            for q in range(n_next):
                i = nxt[q]
                queued[i] = False
                if not infeasible:
                    min_act, max_act, n_min, n_max = _row_activity(
                        i, row_ptr, row_col, row_val, lower, upper
                    )
                    infeasible = _missed(
                        min_act, max_act, n_min, n_max, row_lower[i], row_upper[i], tol
                    )
            return infeasible, n_log
        passes += 1
        n_cur = n_next
        for q in range(n_cur):
            cur[q] = nxt[q]
            queued[cur[q]] = False
        n_next = 0
        for q in range(n_cur):
            i = cur[q]
            rl, ru = row_lower[i], row_upper[i]
            min_act, max_act, n_min, n_max = _row_activity(
                i, row_ptr, row_col, row_val, lower, upper
            )
            if _missed(min_act, max_act, n_min, n_max, rl, ru, tol):
                for r in range(n_next):
                    queued[nxt[r]] = False
                return True, n_log
            for k in range(row_ptr[i], row_ptr[i + 1]):
                a = row_val[k]
                j = row_col[k]
                lo, up = lower[j], upper[j]
                if a > 0:
                    min_c, max_c = a * lo, a * up
                else:
                    min_c, max_c = a * up, a * lo
                # The rest of the row's activity, this entry left out.
                if not math.isfinite(min_c):
                    rest_min = min_act if n_min == 1 else -np.inf
                else:
                    rest_min = min_act - min_c if n_min == 0 else -np.inf
                if not math.isfinite(max_c):
                    rest_max = max_act if n_max == 1 else np.inf
                else:
                    rest_max = max_act - max_c if n_max == 0 else np.inf
                from_upper = (ru - rest_min) / a  # a x <= ru - rest_min
                from_lower = (rl - rest_max) / a  # a x >= rl - rest_max
                sign = 1.0 if a > 0 else (-1.0 if a < 0 else 0.0)
                if math.isnan(from_upper):
                    from_upper = np.inf * sign
                if math.isnan(from_lower):
                    from_lower = -np.inf * sign
                if a > 0:
                    nu, nl = from_upper, from_lower
                else:
                    nu, nl = from_lower, from_upper
                if is_int[j]:
                    nu = math.floor(nu + round_slack)
                    nl = math.ceil(nl - round_slack)
                changed = False
                # Only clear improvements of a sane size.
                if abs(nu) < 1e12 and (not math.isfinite(up) or nu < up - tol * (1.0 + abs(up))):
                    n_log = _log(j, lower, upper, mark, stamp, log_var, log_lo, log_up, n_log)
                    upper[j] = nu
                    changed = True
                if abs(nl) < 1e12 and (not math.isfinite(lo) or nl > lo + tol * (1.0 + abs(lo))):
                    n_log = _log(j, lower, upper, mark, stamp, log_var, log_lo, log_up, n_log)
                    lower[j] = nl
                    changed = True
                if not changed:
                    continue
                if lower[j] > upper[j]:
                    if is_int[j] or lower[j] > upper[j] + tol * (1.0 + abs(upper[j])):
                        for r in range(n_next):
                            queued[nxt[r]] = False
                        return True, n_log
                    # Continuous bounds crossed by rounding error: meet in
                    # the middle.
                    mid = 0.5 * (lower[j] + upper[j])
                    lower[j] = mid
                    upper[j] = mid
                for kk in range(col_ptr[j], col_ptr[j + 1]):
                    r = col_row[kk]
                    if not queued[r]:
                        queued[r] = True
                        nxt[n_next] = r
                        n_next += 1
    return False, n_log


@nb.njit(cache=True)
def _reserve_int(a, need):
    if need <= a.size:
        return a
    b = np.empty(max(need, 2 * a.size), np.int64)
    b[: a.size] = a
    return b


@nb.njit(cache=True)
def _reserve_float(a, need):
    if need <= a.size:
        return a
    b = np.empty(max(need, 2 * a.size))
    b[: a.size] = a
    return b


@nb.njit(cache=True)
def propagate(
    row_ptr, row_col, row_val, col_ptr, col_row, row_lower, row_upper, is_int,
    lower, upper, seeds, max_passes, tol, round_slack,
):
    """Propagate in place from the variables ``seeds`` (whose bounds
    changed). Returns whether the box is infeasible."""
    n, m = lower.size, row_lower.size
    queued = np.zeros(m, np.bool_)
    cur = np.empty(m, np.int64)
    nxt = np.empty(m, np.int64)
    mark = np.full(n, -1, np.int64)
    log_var = np.empty(n, np.int64)
    log_lo = np.empty(n)
    log_up = np.empty(n)
    infeasible, _ = _propagate(
        row_ptr, row_col, row_val, col_ptr, col_row, row_lower, row_upper, is_int,
        lower, upper, seeds, seeds.size, max_passes, tol, round_slack,
        queued, cur, nxt, mark, 0, log_var, log_lo, log_up, 0,
    )
    return infeasible


@nb.njit(cache=True, nogil=True)
def probe_chunk(
    order, row_ptr, row_col, row_val, col_ptr, col_row, row_lower, row_upper,
    is_int, lower, upper, max_passes, tol, round_slack, useless, max_useless,
):
    """
    ``jaddle_mip.probe`` over the variables ``order``, in order, with the
    box ``lower``/``upper`` tightened in place. Each variable is split
    (``_split``) and both sides propagated: one side empty fixes the
    variable to the other (its box holds everywhere); both feasible, every
    bound tightens to the hull of the two; both empty proves the box
    infeasible. A probe is useless when it changes no bound of the box and
    neither side implies a bound on another variable; probing stops once
    ``useless`` reaches ``max_useless`` (−1: never).

    Returns ``(infeasible, consumed, probed, useless, var, split, empty,
    ptr, idx, lo, up)``: the variables of ``order`` gone through, those
    probed, and for probe ``p`` and side ``s`` (0: ``x <= split``) whether
    it is empty and, when it is not, its implications ``idx``/``lo``/``up``
    over ``ptr[2p + s]:ptr[2p + s + 1]`` (the variables it changed, the
    probed one included, and their bounds there).
    """
    n, m = lower.size, row_lower.size
    queued = np.zeros(m, np.bool_)
    cur = np.empty(m, np.int64)
    nxt = np.empty(m, np.int64)
    mark = np.full(n, -1, np.int64)
    log_var = np.empty(n, np.int64)
    log_lo = np.empty(n)
    log_up = np.empty(n)
    down_mark = np.full(n, -1, np.int64)
    down_lo = np.empty(n)
    down_up = np.empty(n)
    seeds = np.empty(n, np.int64)
    k_max = order.size
    out_var = np.empty(k_max, np.int64)
    out_split = np.empty(k_max)
    out_empty = np.zeros((k_max, 2), np.bool_)
    out_ptr = np.zeros(2 * k_max + 1, np.int64)
    flat_idx = np.empty(1024, np.int64)
    flat_lo = np.empty(1024)
    flat_up = np.empty(1024)
    n_flat = 0
    stamp = 0
    probed = 0
    consumed = 0
    infeasible = False
    for t in range(k_max):
        if max_useless >= 0 and useless >= max_useless:
            break
        consumed = t + 1
        v = order[t]
        if lower[v] >= upper[v]:
            continue
        split = _split(lower[v], upper[v])
        p = probed
        for side in range(2):
            stamp += 1
            n_log = _log(v, lower, upper, mark, stamp, log_var, log_lo, log_up, 0)
            if side == 0:
                upper[v] = min(upper[v], split)
            else:
                lower[v] = max(lower[v], split + 1.0)
            seeds[0] = v
            empty, n_log = _propagate(
                row_ptr, row_col, row_val, col_ptr, col_row, row_lower, row_upper,
                is_int, lower, upper, seeds, 1, max_passes, tol, round_slack,
                queued, cur, nxt, mark, stamp, log_var, log_lo, log_up, n_log,
            )
            out_empty[p, side] = empty
            if not empty:
                flat_idx = _reserve_int(flat_idx, n_flat + n_log)
                flat_lo = _reserve_float(flat_lo, n_flat + n_log)
                flat_up = _reserve_float(flat_up, n_flat + n_log)
                for q in range(n_log):
                    j = log_var[q]
                    flat_idx[n_flat] = j
                    flat_lo[n_flat] = lower[j]
                    flat_up[n_flat] = upper[j]
                    n_flat += 1
                    if side == 0:
                        down_mark[j] = p
                        down_lo[j] = lower[j]
                        down_up[j] = upper[j]
            out_ptr[2 * p + side + 1] = n_flat
            for q in range(n_log):  # undo the side
                j = log_var[q]
                lower[j] = log_lo[q]
                upper[j] = log_up[q]
        out_var[p] = v
        out_split[p] = split
        probed += 1
        down_empty, up_empty = out_empty[p, 0], out_empty[p, 1]
        if down_empty and up_empty:
            infeasible = True
            break
        # What holds everywhere: the feasible side's box when the other is
        # empty, else the hull of the two.
        n_seeds = 0
        if down_empty or up_empty:
            side = 1 if down_empty else 0
            for q in range(out_ptr[2 * p + side], out_ptr[2 * p + side + 1]):
                j = flat_idx[q]
                changed = False
                if flat_lo[q] > lower[j]:
                    lower[j] = flat_lo[q]
                    changed = True
                if flat_up[q] < upper[j]:
                    upper[j] = flat_up[q]
                    changed = True
                if changed:
                    seeds[n_seeds] = j
                    n_seeds += 1
        else:
            for q in range(out_ptr[2 * p + 1], out_ptr[2 * p + 2]):
                j = flat_idx[q]
                if down_mark[j] != p:
                    continue
                nl = min(down_lo[j], flat_lo[q])
                nu = max(down_up[j], flat_up[q])
                changed = False
                if nl > lower[j]:
                    lower[j] = nl
                    changed = True
                if nu < upper[j]:
                    upper[j] = nu
                    changed = True
                if changed:
                    seeds[n_seeds] = j
                    n_seeds += 1
        if n_seeds > 0:
            stamp += 1
            empty, _ = _propagate(
                row_ptr, row_col, row_val, col_ptr, col_row, row_lower, row_upper,
                is_int, lower, upper, seeds, n_seeds, max_passes, tol, round_slack,
                queued, cur, nxt, mark, stamp, log_var, log_lo, log_up, 0,
            )
            if empty:
                infeasible = True
                break
        learned = (
            out_ptr[2 * p + 1] - out_ptr[2 * p] > 1
            or out_ptr[2 * p + 2] - out_ptr[2 * p + 1] > 1
        )
        useless = 0 if (n_seeds > 0 or learned) else useless + 1
    return (
        infeasible, consumed, probed, useless,
        out_var[:probed], out_split[:probed], out_empty[:probed],
        out_ptr[: 2 * probed + 1], flat_idx[:n_flat], flat_lo[:n_flat], flat_up[:n_flat],
    )
