"""
Incremental Feasibility Jump on the host, compiled with numba (optional:
``pip install jaddle[fast]``). ``jaddle_mip.feasibility_jump`` uses it when
numba is installed.

The rules are those of the device version (``jaddle_mip._jump_chunk``): each
variable's jump value is the value in its bounds that minimises the weighted
violation of its column's rows (the weighted median of the rows' interval
ends; integers try both roundings, and the best neighbour when that is the
current value), and its score is the exact drop in weighted violation. The
difference is in the bookkeeping, as in Luteberget & Sartor's and HiGHS's
implementations: one variable moves per step, and only what it touches is
updated. Its rows' activities change, and the jump values of the variables
in those rows are recomputed. A step costs about the nonzeros of those rows,
instead of a pass (and a sort) over every nonzero.

Each step moves the best of up to ``_SAMPLE`` variables sampled from those
with a positive score. When there is none, the violated rows get heavier by
1 and the best variable of a random violated row moves anyway (1% of the
time a random one).
"""

import numba as nb
import numpy as np

_SAMPLE = 25  # improving variables sampled per step, as in HiGHS
_EPS = 1e-12


@nb.njit(cache=True, inline="always")
def _violation(act, lo, hi):
    return max(act - hi, 0.0) + max(lo - act, 0.0)


@nb.njit(cache=True)
def _change(j, value, x, act, w, col_ptr, col_row, col_val, row_lower, row_upper):
    """The change in weighted violation from moving variable ``j`` alone to
    ``value``."""
    d = value - x[j]
    total = 0.0
    for p in range(col_ptr[j], col_ptr[j + 1]):
        i = col_row[p]
        new = act[i] + col_val[p] * d
        total += w[i] * (
            _violation(new, row_lower[i], row_upper[i])
            - _violation(act[i], row_lower[i], row_upper[i])
        )
    return total


@nb.njit(cache=True)
def _jump(
    j, x, act, w, col_ptr, col_row, col_val, row_lower, row_upper,
    lower, upper, is_int, buf_v, buf_w,
):
    """``(target, change)`` of variable ``j`` (see the module docstring)."""
    start = col_ptr[j]
    k = col_ptr[j + 1] - start
    xj = x[j]
    half = 0.0
    for q in range(k):
        p = start + q
        i = col_row[p]
        a = col_val[p]
        rest = act[i] - a * xj
        t_lower = (row_lower[i] - rest) / a
        t_upper = (row_upper[i] - rest) / a
        if a > 0:
            buf_v[2 * q], buf_v[2 * q + 1] = t_lower, t_upper
        else:
            buf_v[2 * q], buf_v[2 * q + 1] = t_upper, t_lower
        wt = w[i] * abs(a)
        buf_w[2 * q] = wt
        buf_w[2 * q + 1] = wt
        half += wt
    best = xj
    if k > 0:
        # The weighted median of the interval ends.
        order = np.argsort(buf_v[: 2 * k])
        running = 0.0
        threshold = half * (1.0 - 1e-12)
        for q in range(2 * k):
            o = order[q]
            running += buf_w[o]
            if running >= threshold:
                best = buf_v[o]
                break
    lo, hi = lower[j], upper[j]
    best = min(max(best, lo), hi)
    if not np.isfinite(best):
        best = xj
    if is_int[j]:
        down = min(max(np.floor(best), lo), hi)
        up = min(max(np.ceil(best), lo), hi)
    else:
        down = best
        up = best
    args = (x, act, w, col_ptr, col_row, col_val, row_lower, row_upper)
    change_down = _change(j, down, *args)
    change_up = change_down if up == down else _change(j, up, *args)
    if change_up < change_down:
        target, change = up, change_up
    else:
        target, change = down, change_down
    if is_int[j] and abs(target - xj) <= _EPS:
        # The best value other than the current one: a neighbour (the
        # function is convex).
        below = min(max(xj - 1.0, lo), hi)
        above = min(max(xj + 1.0, lo), hi)
        change_below = _change(j, below, *args) if below < xj else np.inf
        change_above = _change(j, above, *args) if above > xj else np.inf
        if change_above < change_below:
            target, change = above, change_above
        else:
            target, change = below, change_below
    return target, change


@nb.njit(cache=True, inline="always")
def _set_member(k, member, items, pos, count):
    """Put ``k`` in (``member``) or out of the set ``items[:count]``, whose
    positions ``pos`` holds (-1: out). Returns the new count."""
    if member and pos[k] < 0:
        items[count] = k
        pos[k] = count
        return count + 1
    if not member and pos[k] >= 0:
        last = items[count - 1]
        items[pos[k]] = last
        pos[last] = pos[k]
        pos[k] = -1
        return count - 1
    return count


@nb.njit(cache=True)
def run(
    x, w,
    row_ptr, row_col, row_val, col_ptr, col_row, col_val,
    row_lower, row_upper, tol, lower, upper, is_int,
    iterations, seed,
    improve_row, improve_abs, improve_rel, best_x,
):
    """Up to ``iterations`` moves from ``x`` with row weights ``w`` (both
    updated in place). Returns ``(moves, found, target, score)``: whether
    a point meeting every row was reached, and each variable's jump value
    and score at the end.

    With ``improve_row`` (the objective row ``c·x <= cutoff``, scaled), the
    search goes on from each point it reaches: the point is copied to
    ``best_x``, the row's bound drops to the point's objective less
    ``max(improve_abs, improve_rel·|objective|)`` (scaled), and the moves
    continue with the weights as they are. ``best_x`` then holds the best
    point found, and ``row_upper`` (changed in place) its tightened bound."""
    np.random.seed(seed)
    m = row_lower.shape[0]
    n = x.shape[0]

    act = np.zeros(m)
    for i in range(m):
        s = 0.0
        for p in range(row_ptr[i], row_ptr[i + 1]):
            s += row_val[p] * x[row_col[p]]
        act[i] = s
    violated = np.empty(m, np.int64)
    violated_pos = np.full(m, -1, np.int64)
    n_violated = 0
    for i in range(m):
        if _violation(act[i], row_lower[i], row_upper[i]) > tol[i]:
            n_violated = _set_member(i, True, violated, violated_pos, n_violated)

    longest = 1
    for j in range(n):
        longest = max(longest, col_ptr[j + 1] - col_ptr[j])
    buf_v = np.empty(2 * longest)
    buf_w = np.empty(2 * longest)
    target = np.empty(n)
    score = np.empty(n)
    good = np.empty(n, np.int64)
    good_pos = np.full(n, -1, np.int64)
    n_good = 0

    def refresh(k, n_good):
        t, change = _jump(
            k, x, act, w, col_ptr, col_row, col_val, row_lower, row_upper,
            lower, upper, is_int, buf_v, buf_w,
        )
        target[k] = t
        movable = abs(t - x[k]) > _EPS
        score[k] = -change if movable else -np.inf
        return _set_member(k, score[k] > _EPS, good, good_pos, n_good)

    for j in range(n):
        n_good = refresh(j, n_good)

    # Variables with two values (binaries): the jump is always the other
    # value, so a row's change only changes that row's term in the score,
    # updated in O(1). Others get their jump value recomputed.
    flip = np.zeros(n, np.bool_)
    for j in range(n):
        flip[j] = is_int[j] and upper[j] - lower[j] == 1.0

    def row_term(k, a, act_i, i):
        """Row ``i``'s term in ``score[k]`` at activity ``act_i``, per unit weight."""
        moved = act_i + a * (target[k] - x[k])
        return _violation(act_i, row_lower[i], row_upper[i]) - _violation(
            moved, row_lower[i], row_upper[i]
        )

    stamp = np.zeros(n, np.int64)
    dirty = np.empty(n, np.int64)
    epoch = 0
    moves = 0
    found = False
    while True:
        if n_violated == 0:
            found = True
            if improve_row < 0:
                break
            best_x[:] = x
            # Ask for a better point: the objective row's bound drops below
            # this one's objective. Only that row changes: its two-valued
            # variables' terms are updated, the rest recomputed.
            i = improve_row
            gap = max(improve_abs, improve_rel * abs(act[i]))
            epoch += 1
            n_dirty = 0
            for p in range(row_ptr[i], row_ptr[i + 1]):
                k = row_col[p]
                if flip[k]:
                    score[k] -= w[i] * row_term(k, row_val[p], act[i], i)
            row_upper[i] = act[i] - gap
            for p in range(row_ptr[i], row_ptr[i + 1]):
                k = row_col[p]
                if flip[k]:
                    score[k] += w[i] * row_term(k, row_val[p], act[i], i)
                    n_good = _set_member(k, score[k] > _EPS, good, good_pos, n_good)
                elif stamp[k] != epoch:
                    stamp[k] = epoch
                    dirty[n_dirty] = k
                    n_dirty += 1
            for q in range(n_dirty):
                n_good = refresh(dirty[q], n_good)
            bad = _violation(act[i], row_lower[i], row_upper[i]) > tol[i]
            n_violated = _set_member(i, bad, violated, violated_pos, n_violated)
            if n_violated == 0:
                break  # the gap is within the row's tolerance
            continue
        if moves >= iterations:
            break
        moves += 1
        j = -1
        if n_good > 0:
            best = -np.inf
            if n_good <= _SAMPLE:
                for q in range(n_good):
                    k = good[q]
                    if score[k] > best:
                        best, j = score[k], k
            else:
                for _ in range(_SAMPLE):
                    k = good[np.random.randint(n_good)]
                    if score[k] > best:
                        best, j = score[k], k
        else:
            # A local minimum: the violated rows get heavier (which changes
            # the scores of their variables), then a variable of a random
            # violated row moves anyway.
            epoch += 1
            n_dirty = 0
            for q in range(n_violated):
                i = violated[q]
                w[i] += 1.0
                for p in range(row_ptr[i], row_ptr[i + 1]):
                    k = row_col[p]
                    if flip[k]:
                        score[k] += row_term(k, row_val[p], act[i], i)
                        n_good = _set_member(k, score[k] > _EPS, good, good_pos, n_good)
                    elif stamp[k] != epoch:
                        stamp[k] = epoch
                        dirty[n_dirty] = k
                        n_dirty += 1
            for q in range(n_dirty):
                n_good = refresh(dirty[q], n_good)
            r = violated[np.random.randint(n_violated)]
            pick_random = np.random.random() < 0.01
            best = -np.inf
            seen = 0
            for p in range(row_ptr[r], row_ptr[r + 1]):
                k = row_col[p]
                if abs(target[k] - x[k]) <= _EPS:
                    continue
                seen += 1
                if pick_random:
                    # Reservoir sampling: each movable variable equally likely.
                    if np.random.randint(seen) == 0:
                        j = k
                elif score[k] > best or j < 0:
                    best, j = score[k], k
        if j < 0:
            continue

        # Move ``j``: its rows' activities change; in each, the other
        # variables' scores (two-valued ones) or jump values (the rest).
        d = target[j] - x[j]
        x[j] = target[j]
        epoch += 1
        stamp[j] = epoch
        n_dirty = 0
        for p in range(col_ptr[j], col_ptr[j + 1]):
            i = col_row[p]
            old = act[i]
            new = old + col_val[p] * d
            act[i] = new
            bad = _violation(new, row_lower[i], row_upper[i]) > tol[i]
            n_violated = _set_member(i, bad, violated, violated_pos, n_violated)
            for q in range(row_ptr[i], row_ptr[i + 1]):
                k = row_col[q]
                if k == j:
                    continue
                if flip[k]:
                    a = row_val[q]
                    score[k] += w[i] * (row_term(k, a, new, i) - row_term(k, a, old, i))
                    n_good = _set_member(k, score[k] > _EPS, good, good_pos, n_good)
                elif stamp[k] != epoch:
                    stamp[k] = epoch
                    dirty[n_dirty] = k
                    n_dirty += 1
        n_good = refresh(j, n_good)
        for q in range(n_dirty):
            n_good = refresh(dirty[q], n_good)
    return moves, found, target, score
