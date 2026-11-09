"""Jaddle-specific presolve reductions, applied on top of HiGHS presolve.

Standard presolvers are tuned for simplex/IPM, where fill-in hurts the
factorisation and a dense row costs little. PDHG has the opposite trade-off:
extra nonzeros are cheap (matvec-bound) but some row structures wreck the
conditioning. The reductions here target those structures.
"""

import numpy as np
import scipy.sparse as sp

from jaddle.jaddle_basic_types import LP


class DefinedVariablePostsolve:
    """Recovers eliminated variables, ``x_full[J] = E @ x + const``, and the
    duals of the rows that defined them."""

    def __init__(
        self, n_full, keep, elim, E, const, rows, keep_rows, piv, c_elim, W,
        lb_rows, ub_rows,
    ):
        self.n_full = n_full
        self.keep = keep
        self.elim = elim
        self.E = E
        self.const = const
        self.rows = rows  # the eliminated equality rows (one per variable)
        self.keep_rows = keep_rows
        self.piv = piv  # each variable's coefficient in its row
        self.c_elim = c_elim
        self.W = W  # the variables' columns in the inequality rows
        self.lb_rows = lb_rows  # positions in `elim` given a lower-bound row
        self.ub_rows = ub_rows  # ... and an upper-bound row

    def primal(self, x):
        x = np.asarray(x, dtype=np.float64)
        x_full = np.zeros(self.n_full)
        x_full[self.keep] = x
        # E is indexed by original columns and is zero on the eliminated ones.
        x_full[self.elim] = self.E @ x_full + self.const
        return x_full

    def dual(self, dual_eq, dual_ineq):
        """Map ``(dual_eq, dual_ineq)`` of the reduced LP to the LP before
        elimination (Jaddle's convention: reduced cost ``c + Aᵀy``).

        The inequality rows keep their duals; the bound rows added for an
        eliminated ``z`` become its reduced cost ``r_z = y_lb − y_ub`` (0
        without them). The row that defined ``z`` gets the dual that makes
        ``z``'s own reduced cost ``r_z``:
        ``y_row = (r_z − c_z − W_zᵀ y_ineq) / a_z``. Substituting this back
        reproduces the reduced LP's reduced cost on every kept column, so the
        mapped pair is exactly as feasible and optimal as the reduced one.
        """
        dual_eq = np.asarray(dual_eq, dtype=np.float64)
        dual_ineq = np.asarray(dual_ineq, dtype=np.float64)
        n_ineq = self.W.shape[0]
        y_ineq = dual_ineq[:n_ineq]
        n_lb = self.lb_rows.size
        r_elim = np.zeros(self.elim.size)
        r_elim[self.lb_rows] += dual_ineq[n_ineq : n_ineq + n_lb]
        r_elim[self.ub_rows] -= dual_ineq[n_ineq + n_lb : n_ineq + n_lb + self.ub_rows.size]
        y_rows = (r_elim - self.c_elim - self.W.T @ y_ineq) / self.piv
        y_eq_full = np.zeros(self.keep_rows.size + self.rows.size)
        y_eq_full[self.keep_rows] = dual_eq
        y_eq_full[self.rows] = y_rows
        return y_eq_full, y_ineq


class ChainedPostsolve:
    """Composes the postsolves of successive elimination passes."""

    def __init__(self, steps):
        self.steps = steps

    def primal(self, x):
        # Each pass maps its reduced space back to the previous pass's space.
        for step in reversed(self.steps):
            x = step.primal(x)
        return x

    def dual(self, dual_eq, dual_ineq):
        for step in reversed(self.steps):
            dual_eq, dual_ineq = step.dual(dual_eq, dual_ineq)
        return dual_eq, dual_ineq


def eliminate_defined_variables(
    lp: LP,
    min_row_nnz=100,
    min_costed_row_nnz=2,
    max_col_uses=4,
    max_fill=1.0,
    max_passes=10,
    verbose=False,
):
    """Substitute out variables defined by an equality row.

    Targets rows ``a*z + r.x = b`` where ``z`` has zero cost and appears in no
    other equality row, e.g. ``z = sum_j a_j x_j`` aggregates whose coefficients mirror the
    objective (gmut-*, leo1). Also targets the objective-row variant where ``z``
    is the row's only costed entry (``min z, z = sum_j a_j x_j``, proteindesign*):
    the substitution then turns the dense row into a dense cost vector. HiGHS/SCIP leave these in place: ``z`` is not a
    column singleton (it is used by inequality rows too), not free (so the
    substitution needs an explicit bound row), and substituting a dense row
    exceeds their fill-in limits. For PDHG the dense row is the problem -- it
    hides the objective in a constraint and the primal drifts off feasibility.

    With ``z = E_z.x + const_z`` (``E_z = -r/a``, ``const_z = b/a``):
      * cost:        c += c_z * E_z, objective offset += c_z * const_z
      * inequality:  rows using z get w * E_z added, rhs -= w * const_z
      * bounds of z: become rows ``lb_z <= E_z.x + const_z <= ub_z``, skipped
                     when already implied by the bounds on x.

    Zero-cost aggregate rows are eligible only with at least ``min_row_nnz``
    nonzeros (the dense aggregates that cause trouble). Objective rows are
    eligible from ``min_costed_row_nnz`` nonzeros: a SHORT objective row hides
    the objective just as well (radiationm40-10-02: ``min z, z = 1601 N + T``,
    3 nnz, with N and T each defined by an 11-nnz row). A row shorter than
    ``min_row_nnz`` additionally needs ``z`` to appear in no inequality row, so
    the substitution only relocates the cost. Further passes, for objective
    rows only, repeat until nothing more is eliminated (at most
    ``max_passes``), so the cost is pushed down such chains onto the decision
    variables. ``z`` may appear in at most
    ``max_col_uses`` inequality rows, and total added nonzeros over all passes
    are capped at ``max_fill`` times the original nnz.

    Returns ``(reduced_lp, offset, postsolve)``; add ``offset`` to the reduced
    objective. ``postsolve.primal(x)`` maps a reduced solution back, and
    ``postsolve.dual(dual_eq, dual_ineq)`` its duals. Returns
    ``(lp, 0.0, None)`` when nothing is eliminated.
    """
    budget = max_fill * (lp.A_eq.nnz + lp.A_ineq.nnz)
    offset, steps = 0.0, []
    for p in range(max_passes):
        # Later passes follow objective chains only: repeating the zero-cost
        # rule picks up second-level aggregates that slow ns1760995 down.
        lp, pass_offset, step, fill = _eliminate_pass(
            lp, min_row_nnz, min_costed_row_nnz, max_col_uses, budget, verbose,
            costed_only=p > 0,
        )
        if step is None:
            break
        offset += pass_offset
        budget -= fill
        steps.append(step)
    if not steps:
        return lp, 0.0, None
    return lp, offset, steps[0] if len(steps) == 1 else ChainedPostsolve(steps)


def _eliminate_pass(
    lp, min_row_nnz, min_costed_row_nnz, max_col_uses, budget, verbose, costed_only
):
    """One elimination pass of ``eliminate_defined_variables``.

    Returns ``(reduced_lp, offset, postsolve, fill)``, with ``postsolve`` None
    when nothing is eliminated; ``fill`` is the nonzero budget consumed.
    """
    c = np.asarray(lp.c, dtype=np.float64)
    lb = np.asarray(lp.lower_bounds, dtype=np.float64)
    ub = np.asarray(lp.upper_bounds, dtype=np.float64)
    b_eq = np.asarray(lp.b_eq, dtype=np.float64)
    b_ineq = np.asarray(lp.b_ineq, dtype=np.float64)
    A_eq = sp.csr_matrix(lp.A_eq, dtype=np.float64)
    A_ineq = sp.csc_matrix(lp.A_ineq, dtype=np.float64)
    n = len(c)
    if A_eq.shape[0] == 0:
        return lp, 0.0, None, 0

    col_eq_uses = np.diff(sp.csc_matrix(A_eq).indptr)
    col_ineq_uses = np.diff(A_ineq.indptr)

    rows, cols, fill = [], [], 0
    for i in range(A_eq.shape[0]):
        lo, hi = A_eq.indptr[i], A_eq.indptr[i + 1]
        nnz = hi - lo
        if nnz < 2 or nnz < min(min_row_nnz, min_costed_row_nnz):
            continue
        idx, val = A_eq.indices[lo:hi], A_eq.data[lo:hi]
        ok = (col_eq_uses[idx] == 1) & (col_ineq_uses[idx] <= max_col_uses)
        ok &= np.abs(val) > 1e-9
        costed = c[idx] != 0.0
        # A short objective row only qualifies when its aggregate is used by no
        # inequality row, so the substitution just relocates the cost (no
        # matrix fill). Without this guard, short rows like ``x = y`` with a
        # costed ``x`` are substituted all over the set (mzzv11, neos-827175).
        # It also needs b = 0: otherwise the objective moves into the offset,
        # which solve() never sees, so the relative gap is normalised by a
        # spuriously large objective (seqsolve2short4288: offset ~ -1e4 on an
        # optimum near 0, falsely certified).
        if (
            nnz >= min_costed_row_nnz
            and np.count_nonzero(costed) == 1
            and ok[costed][0]
            and (
                nnz >= min_row_nnz
                or (col_ineq_uses[idx[costed][0]] == 0 and b_eq[i] == 0.0)
            )
        ):
            # Objective row ``min z, z = sum_j a_j x_j`` (proteindesign*): the
            # row's only costed entry IS the aggregate, so substitute it and its
            # cost moves onto the zero-cost summands.
            ok = costed
        elif costed_only or nnz < min_row_nnz:
            continue
        else:
            # The aggregate carries no cost of its own (the cost is spread over
            # the summands). Without this, a costed summand x_j would be
            # substituted instead, leaving the aggregate -- and the pathology --
            # in place.
            ok &= ~costed
        if not ok.any():
            continue
        # Fewest inequality uses (least fill -- the aggregate variable itself,
        # not an ordinary summand), then largest pivot for stability.
        cand = np.flatnonzero(ok)
        k = cand[np.lexsort((-np.abs(val[cand]), col_ineq_uses[idx[cand]]))[0]]
        j = idx[k]
        # Substituted rows plus up to two bound rows.
        cost = (hi - lo - 1) * (col_ineq_uses[j] + 2)
        if fill + cost > budget:
            continue
        fill += cost
        rows.append(i)
        cols.append(j)

    if not rows:
        return lp, 0.0, None, 0

    rows, cols = np.array(rows), np.array(cols)
    piv = np.asarray(A_eq[rows, cols]).ravel()
    R = A_eq[rows].tolil()
    R[np.arange(len(rows)), cols] = 0.0
    # z_J = E @ x + const (E has zero columns at J: each z_j sits in one eq row).
    E = sp.csr_matrix(-sp.diags(1.0 / piv) @ R.tocsr())
    E.eliminate_zeros()
    const = b_eq[rows] / piv

    c_J = c[cols]
    c_new = c + E.T @ c_J
    offset = float(c_J @ const)

    W = A_ineq[:, cols]
    A_ineq_new = (A_ineq + W @ E).tocsr()
    b_ineq_new = b_ineq - W @ const

    # Bound rows for z, skipped when implied by the bounds on x.
    Ep, En = E.maximum(0), E.minimum(0)
    lb_safe, ub_safe = np.where(np.isfinite(lb), lb, 0.0), np.where(np.isfinite(ub), ub, 0.0)
    has_lb, has_ub = np.isfinite(lb), np.isfinite(ub)
    no_lb, no_ub = (~has_lb).astype(np.float64), (~has_ub).astype(np.float64)
    # Activity bounds of E.x: +inf/-inf wherever an unbounded var contributes.
    act_min = Ep @ lb_safe + En @ ub_safe
    act_min[(Ep @ no_lb - En @ no_ub) > 0] = -np.inf
    act_max = Ep @ ub_safe + En @ lb_safe
    act_max[(Ep @ no_ub - En @ no_lb) > 0] = np.inf
    lbz, ubz = lb[cols] - const, ub[cols] - const
    need_lb = np.isfinite(lbz) & (act_min < lbz)
    need_ub = np.isfinite(ubz) & (act_max > ubz)
    extra_A = [-E[need_lb], E[need_ub]]
    extra_b = [-lbz[need_lb], ubz[need_ub]]

    keep = np.setdiff1d(np.arange(n), cols)
    keep_rows = np.setdiff1d(np.arange(A_eq.shape[0]), rows)
    A_ineq_out = sp.vstack([A_ineq_new] + extra_A, format="csc")[:, keep]
    A_ineq_out.eliminate_zeros()
    reduced = LP(
        c_new[keep],
        A_eq[keep_rows][:, keep].tocsc(),
        b_eq[keep_rows],
        A_ineq_out,
        np.concatenate([b_ineq_new] + extra_b),
        lb[keep],
        ub[keep],
    )
    if verbose:
        print(
            f"Defined-variable presolve: eliminated {len(rows)} eq rows/cols, "
            f"added {int(need_lb.sum() + need_ub.sum())} bound rows, "
            f"nnz {A_eq.nnz + A_ineq.nnz} -> "
            f"{reduced.A_eq.nnz + reduced.A_ineq.nnz}"
        )
    postsolve = DefinedVariablePostsolve(
        n,
        keep,
        cols,
        E,
        const,
        rows=rows,
        keep_rows=keep_rows,
        piv=piv,
        c_elim=c_J,
        W=sp.csr_matrix(W),
        lb_rows=np.flatnonzero(need_lb),
        ub_rows=np.flatnonzero(need_ub),
    )
    return reduced, offset, postsolve, fill


class DefinedIntegers:
    """A MIP's rows with the integers its equality rows define substituted
    out (``defined_integers``), for a search that moves one variable at a
    time, such as feasibility jump.

    Each eliminated ``z`` is ``E_z·x + const_z`` over the kept columns. The
    rows are, in order: the equality rows that define nothing, the
    inequality rows (``<= b``) with each ``z`` replaced, and one row
    ``E_z·x`` per ``z`` whose bounds come from ``z``'s (``rows``), so the
    rows hold for whatever box the caller searches. ``A`` is over the kept
    columns (``keep``); ``primal`` maps a point back.
    """

    def __init__(self, A, b, n_eq, rows, cols):
        A = sp.csr_matrix(A, dtype=np.float64)
        b = np.asarray(b, np.float64)
        n = A.shape[1]
        rows, cols = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
        piv = np.asarray(A[rows, cols], np.float64).ravel()
        R = A[rows].tolil()
        R[np.arange(rows.size), cols] = 0.0
        # z_J = E @ x + const; E is zero on the eliminated columns, since each
        # z sits in one equality row only.
        E = sp.csr_matrix(-sp.diags(1.0 / piv) @ R.tocsr())
        E.eliminate_zeros()
        const = b[rows] / piv
        keep_rows = np.setdiff1d(np.arange(n_eq), rows)
        A_ineq = A[n_eq:]
        W = sp.csr_matrix(A_ineq[:, cols])
        A_ineq = sp.csr_matrix(A_ineq + W @ E)
        b_ineq = b[n_eq:] - W @ const
        self.n_full = n
        self.elim = cols
        self.keep = np.setdiff1d(np.arange(n), cols)
        self.E, self.const = E, const
        self.n_eq = keep_rows.size
        A_out = sp.vstack([A[keep_rows], A_ineq, E], format="csc")[:, self.keep]
        self.A = sp.csr_matrix(A_out)
        self.A.eliminate_zeros()
        k = rows.size
        self.row_lower = np.concatenate(
            [b[keep_rows], np.full(b_ineq.size, -np.inf), np.full(k, -np.inf)]
        )
        self.row_upper = np.concatenate([b[keep_rows], b_ineq, np.full(k, np.inf)])
        # Each row's size, for its tolerance (``1 + |b|``); the bound rows,
        # like bounds, get the absolute tolerance.
        self.b = np.concatenate([b[keep_rows], b_ineq, np.zeros(k)])

    def __len__(self):
        return self.elim.size

    def rows(self, lower, upper):
        """``(row_lower, row_upper)`` with the eliminated integers' bound
        rows set from the box ``lower``/``upper`` (over all columns)."""
        k = self.elim.size
        row_lower, row_upper = self.row_lower.copy(), self.row_upper.copy()
        row_lower[-k:] = np.asarray(lower, np.float64)[self.elim] - self.const
        row_upper[-k:] = np.asarray(upper, np.float64)[self.elim] - self.const
        return row_lower, row_upper

    def primal(self, x):
        """The point over all columns from one over the kept columns."""
        full = np.zeros(self.n_full)
        full[self.keep] = np.asarray(x, np.float64)
        full[self.elim] = self.E @ full + self.const
        return full


def defined_integers(A, b, n_eq, c, integrality, lower, upper, max_fill=1.0):
    """The integers that equality rows define, substituted out
    (``DefinedIntegers``), or ``None`` when there are none.

    A row ``a_z z + r·x = b`` (one of the first ``n_eq`` rows of ``A``; the
    rest are ``<= b``) defines ``z`` when ``z`` is an integer with zero
    cost, ``|a_z| = 1``, ``z`` is in no other equality row, and every other
    entry is an integer coefficient on an integer variable, with ``b`` an
    integer. Then ``z = (b − r·x) / a_z`` is an integer at every integer
    ``x``, so it can be dropped and recomputed. ``z`` must be a general
    integer (``upper − lower > 1`` over the box ``lower``/``upper``): a
    binary in a row ``Σ x = 1`` passes the other tests, but that row
    chooses among its variables rather than defining one of them, and
    substituting it turns the choice into packing rows. Time-indexed scheduling
    models are full of these: on 30n20b8, 57 start times
    ``S_j = Σ_t t·x_jt``. A search that moves one variable at a time cannot
    change such a row's ``x`` without breaking it, and moving ``S_j`` too
    breaks the precedence rows ``S_i <= S_j``; feasibility jump stalled
    there with a few rows violated, and found points in seconds once the
    start times were gone.

    Of a row's candidates, the one in fewest inequality rows goes. The
    nonzeros added (each ``z``'s row times its inequality uses plus its
    bound row) are capped at ``max_fill`` times ``A``'s.
    """
    A = sp.csr_matrix(A, dtype=np.float64)
    A.eliminate_zeros()
    b = np.asarray(b, np.float64)
    c = np.asarray(c, np.float64)
    ints = np.asarray(integrality, bool)
    general = ints & (np.asarray(upper, np.float64) - np.asarray(lower, np.float64) > 1.0)
    n = A.shape[1]
    split = A.indptr[n_eq]
    eq_uses = np.bincount(A.indices[:split], minlength=n)
    ineq_uses = np.bincount(A.indices[split:], minlength=n)
    budget = max_fill * A.nnz
    rows, cols, fill = [], [], 0
    for i in range(n_eq):
        start, end = A.indptr[i], A.indptr[i + 1]
        if end - start < 2 or b[i] != np.round(b[i]):
            continue
        idx, val = A.indices[start:end], A.data[start:end]
        if not (ints[idx].all() and np.all(val == np.round(val))):
            continue
        ok = (np.abs(val) == 1.0) & general[idx] & (eq_uses[idx] == 1) & (c[idx] == 0.0)
        if not ok.any():
            continue
        cand = np.flatnonzero(ok)
        j = idx[cand[np.argmin(ineq_uses[idx[cand]])]]  # first of the fewest
        cost = (end - start - 1) * (ineq_uses[j] + 1)
        if fill + cost > budget:
            continue
        fill += cost
        rows.append(i)
        cols.append(j)
    if not rows:
        return None
    return DefinedIntegers(A, b, n_eq, rows, cols)
