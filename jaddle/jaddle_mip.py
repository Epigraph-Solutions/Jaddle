"""
Branch and bound for mixed-integer linear programs, with Jaddle's LP solver
(``jaddle_linear.solve``) at every node.

A MIP here is a Jaddle LP plus a mask of integer columns:

    minimise    cᵀx
    subject to  A_eq   x  = b_eq
                A_ineq x <= b_ineq
                l <= x <= u,     x_j integer for j in the mask

``load`` and ``from_highs`` read one from any model HiGHS reads (MPS or LP
files, a ``highspy.Highs``, a ``highspy.HighsLp``), keeping its integrality;
``solve`` runs the branch and bound.

The node LPs are solved by a first-order method, so their solutions are
approximate. Branch and bound stays sound because:

* nodes are pruned on the Lagrangian dual bound of the node's duals (see
  ``lagrangian_bound``), which is a valid lower bound for any duals, whether
  the LP converged or not;
* an incumbent is accepted only after its integers are rounded and the point
  is checked against the rows and bounds;
* branching on an inaccurate LP point costs extra nodes but loses no
  solutions.

The one exception is a reduced cost of the wrong sign on a variable with an
infinite bound, which makes the Lagrangian bound ``-inf``. A small one (within
the LP tolerance) is dropped, and the result reports ``bound_rigorous=False``.
"""

import heapq
import itertools
import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from jaddle import highs_helpers as hh
from jaddle import jaddle_linear as jl
from jaddle.jaddle_basic_types import LP, JaddleLP

# Slack when rounding bounds of integer variables, so 2.9999999 becomes 3.
_ROUND_SLACK = 1e-6


# %%
# The MIP model and its readers
class MIP:
    """A mixed-integer LP: a Jaddle LP (``lp``, as a minimisation) and a
    boolean mask of its integer columns (``integrality``).

    ``offset`` is the objective constant, so ``c @ x + offset`` is the
    minimisation's value. ``maximise=True`` marks a model that was a
    maximisation: its own value is ``-(c @ x + offset)``. ``objective(x)``
    returns the model's own value either way.

    The bounds of integer variables are rounded inward (``ceil`` of the lower,
    ``floor`` of the upper bound).
    """

    def __init__(self, lp, integrality, offset=0.0, maximise=False):
        if isinstance(lp, LP):
            lp = jl.to_jaddle_sparse(lp)
        elif not isinstance(lp, JaddleLP):
            raise TypeError(f"lp must be an LP or JaddleLP, not {type(lp).__name__}")
        integrality = np.asarray(integrality, dtype=bool).reshape(-1)
        n = lp.c.shape[0]
        if integrality.shape[0] != n:
            raise ValueError(
                f"integrality has {integrality.shape[0]} entries for {n} variables"
            )
        self.integrality = integrality
        self.offset = float(offset)
        self.maximise = bool(maximise)

        lower = np.array(lp.lower_bounds, dtype=np.float64)
        upper = np.array(lp.upper_bounds, dtype=np.float64)
        lower[integrality] = np.ceil(lower[integrality] - _ROUND_SLACK)
        upper[integrality] = np.floor(upper[integrality] + _ROUND_SLACK)
        if not (
            np.array_equal(lower, np.asarray(lp.lower_bounds))
            and np.array_equal(upper, np.asarray(lp.upper_bounds))
        ):
            dtype = lp.c.dtype
            lp = lp.with_values(
                lp.values()._replace(
                    lower_bounds=jnp.asarray(lower, dtype),
                    upper_bounds=jnp.asarray(upper, dtype),
                )
            )
        self.lp = lp

        # Float64 host copies, for propagation, bounds and feasibility checks.
        host = lp.to_scipy()
        self.c = np.asarray(host.c, dtype=np.float64)
        self.n_eq = host.A_eq.shape[0]
        self.A = sp.vstack([host.A_eq, host.A_ineq], format="csr")
        self.A.eliminate_zeros()
        self.b = np.concatenate([host.b_eq, host.b_ineq]).astype(np.float64)
        self.row_lower = np.concatenate(
            [host.b_eq, np.full(host.b_ineq.shape, -np.inf)]
        ).astype(np.float64)
        self.row_upper = self.b.copy()
        self.lower, self.upper = lower, upper

    @property
    def num_variables(self):
        return self.c.shape[0]

    @property
    def num_integers(self):
        return int(self.integrality.sum())

    @property
    def sign(self):
        """``-1`` for a maximisation, ``+1`` otherwise: the model's value is
        ``sign * (c @ x + offset)``."""
        return -1.0 if self.maximise else 1.0

    def objective(self, x):
        """The model's own objective value at ``x`` (offset included)."""
        return self.sign * (float(self.c @ np.asarray(x, np.float64)) + self.offset)

    def violation(self, x):
        """The largest violation at ``x``, as a dict: ``"rows"`` (relative to
        ``1 + ‖b‖∞``), ``"bounds"`` (absolute) and ``"integrality"`` (distance
        of an integer variable from the nearest integer)."""
        x = np.asarray(x, dtype=np.float64)
        Ax = self.A @ x
        rows = np.maximum(
            np.maximum(self.row_lower - Ax, Ax - self.row_upper), 0.0
        ).max(initial=0.0)
        bounds = np.maximum(
            np.maximum(self.lower - x, x - self.upper), 0.0
        ).max(initial=0.0)
        xi = x[self.integrality]
        return {
            "rows": float(rows / (1.0 + np.abs(self.b).max(initial=0.0))),
            "bounds": float(bounds),
            "integrality": float(np.abs(xi - np.round(xi)).max(initial=0.0)),
        }

    def with_rows(self, A_rows, b_rows):
        """A ``MIP`` with the rows ``A_rows x <= b_rows`` appended to the
        inequality rows. Explicit zero entries of ``A_rows`` are kept in the
        device LP (they pad its shape)."""
        lp = LP(
            self.c,
            self.A[: self.n_eq].tocsc(),
            self.b[: self.n_eq],
            sp.vstack([self.A[self.n_eq :], sp.csr_matrix(A_rows)], format="csr"),
            np.concatenate([self.b[self.n_eq :], np.asarray(b_rows, np.float64)]),
            self.lower,
            self.upper,
        )
        return MIP(lp, self.integrality, self.offset, self.maximise)

    def is_feasible(self, x, tol=1e-6):
        """Whether every entry of ``violation(x)`` is at most ``tol``."""
        return all(v <= tol for v in self.violation(x).values())


def _mip_from_highs_lp(highs_lp, maximise=None):
    """A ``MIP`` for a ``highspy.HighsLp``. A maximisation is stored as the
    minimisation of its negated objective. ``maximise`` overrides the
    reporting sense (HiGHS's presolve turns a maximisation into a
    minimisation, whose value is the negated model value)."""
    import highspy

    lp = hh.highs_to_standard_form_sparse(highs_lp)
    offset = float(highs_lp.offset_)
    is_max = highs_lp.sense_ == highspy.ObjSense.kMaximize
    if is_max:
        lp.c, offset = -lp.c, -offset
    return MIP(
        lp,
        hh.highs_integrality(highs_lp),
        offset=offset,
        maximise=is_max if maximise is None else maximise,
    )


def _presolve(highs):
    """Run HiGHS's presolve on ``highs`` and return its status name."""
    import highspy

    highs.presolve()
    status = highs.getModelPresolveStatus()
    PS = highspy.HighsPresolveStatus
    if status in (PS.kNullError, PS.kOptionsError):
        raise RuntimeError(f"HiGHS presolve failed: {status.name}")
    return status


def _highs_mip(model, highs_options=None):
    """A ``highspy.Highs`` holding ``model`` with its integrality kept."""
    if isinstance(model, MIP):
        model = to_highs(model)
    return jl._highs_model(model, highs_options, relax_integrality=False)


def from_highs(model, presolve=False, highs_options=None):
    """
    A ``MIP`` from a HiGHS model: a ``highspy.Highs`` with a model loaded, a
    ``highspy.HighsLp``, or a path to a file HiGHS reads (MPS, LP, gzipped or
    not). The rows are Jaddle's standard form
    (``highs_helpers.rows_to_standard_form`` of the model's rows), the
    integer columns come from the model's integrality, and the objective
    constant and sense are kept in the ``MIP``.

    ``presolve=True`` runs HiGHS's presolve and returns the reduced MIP. Its
    variables and rows are not the model's, and nothing here maps a solution
    back: ``solve(model, presolve=True)`` does that. Raises ``ValueError``
    when presolve proves the model infeasible or unbounded.
    ``highs_options`` (a dict) is applied to the HiGHS instance.

    Semi-continuous and semi-integer columns raise ``NotImplementedError``.
    """
    import highspy

    highs = _highs_mip(model, highs_options)
    highs_lp = highs.getLp()
    if not presolve:
        return _mip_from_highs_lp(highs_lp)
    maximise = highs_lp.sense_ == highspy.ObjSense.kMaximize
    status = _presolve(highs)
    PS = highspy.HighsPresolveStatus
    if status in (PS.kInfeasible, PS.kUnboundedOrInfeasible):
        raise ValueError(
            "HiGHS presolve found the model "
            + ("infeasible" if status == PS.kInfeasible else "unbounded or infeasible")
        )
    return _mip_from_highs_lp(highs.getPresolvedLp(), maximise=maximise)


def load(path, presolve=False, highs_options=None):
    """
    Load a MIP from a file into a ``MIP``, ready for ``solve()``.

    ``path`` is any file HiGHS reads (MPS, LP, gzipped or not). This is
    ``from_highs`` for files; see it for ``presolve`` and ``highs_options``.
    Raises ``ValueError`` when HiGHS cannot read the file.
    """
    return from_highs(os.fspath(path), presolve=presolve, highs_options=highs_options)


def to_highs(mip):
    """A ``highspy.HighsLp`` for a ``MIP``, with its integrality, objective
    constant and sense, so HiGHS reports the model's own objective value."""
    import highspy

    out = hh.jaddle_lp_to_highs(mip.lp, mip.integrality)
    if mip.maximise:
        out.col_cost_ = -np.asarray(out.col_cost_)
        out.offset_ = -mip.offset
        out.sense_ = highspy.ObjSense.kMaximize
    else:
        out.offset_ = mip.offset
    return out


# %%
# Bound propagation and the Lagrangian bound
class BoundPropagator:
    """Activity-based bound tightening for the rows
    ``row_lower <= A x <= row_upper``.

    For each row the smallest and largest activity over the current box are
    computed (counting infinite contributions separately). Each entry then
    gives the bound on its variable that the row implies once the other
    variables are at their extremes. Integer bounds are rounded. A pass is
    O(nnz) and fully vectorised.
    """

    def __init__(self, A, row_lower, row_upper, integrality, tol=1e-6):
        A = sp.csc_matrix(A)
        A.eliminate_zeros()
        A.sort_indices()
        self.m, self.n = A.shape
        self.cols = np.repeat(np.arange(self.n), np.diff(A.indptr))
        self.rows = A.indices
        self.vals = A.data.astype(np.float64)
        self.positive = self.vals > 0
        nonempty = np.diff(A.indptr) > 0
        self.col_ids = np.flatnonzero(nonempty)
        self.col_starts = A.indptr[:-1][nonempty]
        self.row_lower = np.asarray(row_lower, dtype=np.float64)
        self.row_upper = np.asarray(row_upper, dtype=np.float64)
        self.is_int = np.asarray(integrality, dtype=bool)
        self.tol = tol

    def _activities(self, lower, upper):
        """Per-entry extreme contributions and per-row extreme activities."""
        lo, up, a = lower[self.cols], upper[self.cols], self.vals
        with np.errstate(invalid="ignore"):
            min_c = np.where(self.positive, a * lo, a * up)
            max_c = np.where(self.positive, a * up, a * lo)
        min_inf, max_inf = ~np.isfinite(min_c), ~np.isfinite(max_c)
        min_act = np.bincount(self.rows, np.where(min_inf, 0.0, min_c), self.m)
        max_act = np.bincount(self.rows, np.where(max_inf, 0.0, max_c), self.m)
        n_min_inf = np.bincount(self.rows, min_inf, self.m)
        n_max_inf = np.bincount(self.rows, max_inf, self.m)
        return min_c, max_c, min_inf, max_inf, min_act, max_act, n_min_inf, n_max_inf

    def __call__(self, lower, upper, max_passes=5):
        """Tightened ``(lower, upper, infeasible)``. ``infeasible`` is True when
        a row cannot be met inside the box or a variable's bounds cross."""
        lower = np.array(lower, dtype=np.float64)
        upper = np.array(upper, dtype=np.float64)
        if self.vals.size == 0:
            return lower, upper, bool(np.any(lower > upper + self.tol))
        tol = self.tol
        rl, ru = self.row_lower[self.rows], self.row_upper[self.rows]
        for _ in range(max_passes):
            (min_c, max_c, min_inf, max_inf, min_act, max_act, n_min_inf, n_max_inf) = (
                self._activities(lower, upper)
            )
            if self._rows_missed(min_act, max_act, n_min_inf, n_max_inf):
                return lower, upper, True

            # Activity of the rest of the row, with this entry left out.
            nmin, nmax = n_min_inf[self.rows], n_max_inf[self.rows]
            rest_min = np.where(
                min_inf,
                np.where(nmin == 1, min_act[self.rows], -np.inf),
                np.where(nmin == 0, min_act[self.rows] - min_c, -np.inf),
            )
            rest_max = np.where(
                max_inf,
                np.where(nmax == 1, max_act[self.rows], np.inf),
                np.where(nmax == 0, max_act[self.rows] - max_c, np.inf),
            )
            with np.errstate(invalid="ignore", over="ignore"):
                from_upper = (ru - rest_min) / self.vals  # a x <= ru - rest_min
                from_lower = (rl - rest_max) / self.vals  # a x >= rl - rest_max
            from_upper = np.nan_to_num(from_upper, nan=np.inf * np.sign(self.vals))
            from_lower = np.nan_to_num(from_lower, nan=-np.inf * np.sign(self.vals))
            ub_cand = np.where(self.positive, from_upper, from_lower)
            lb_cand = np.where(self.positive, from_lower, from_upper)

            new_upper = np.full(self.n, np.inf)
            new_lower = np.full(self.n, -np.inf)
            new_upper[self.col_ids] = np.minimum.reduceat(ub_cand, self.col_starts)
            new_lower[self.col_ids] = np.maximum.reduceat(lb_cand, self.col_starts)
            new_upper[self.is_int] = np.floor(new_upper[self.is_int] + _ROUND_SLACK)
            new_lower[self.is_int] = np.ceil(new_lower[self.is_int] - _ROUND_SLACK)

            # Keep only clear improvements of a sane size: tiny ones would
            # repeat forever, huge ones come from cancellation.
            sane_u = np.abs(new_upper) < 1e12
            sane_l = np.abs(new_lower) < 1e12
            margin_u = tol * (1.0 + np.abs(upper))
            margin_l = tol * (1.0 + np.abs(lower))
            with np.errstate(invalid="ignore"):
                better_u = sane_u & (
                    ~np.isfinite(upper) | (new_upper < upper - margin_u)
                )
                better_l = sane_l & (
                    ~np.isfinite(lower) | (new_lower > lower + margin_l)
                )
            if not (better_u.any() or better_l.any()):
                break
            upper = np.where(better_u, new_upper, upper)
            lower = np.where(better_l, new_lower, lower)

            crossed = lower > upper
            if np.any(crossed & self.is_int) or np.any(
                lower > upper + tol * (1.0 + np.abs(upper))
            ):
                return lower, upper, True
            # Continuous bounds crossed by rounding error: meet in the middle.
            mid = 0.5 * (lower + upper)
            lower = np.where(crossed, mid, lower)
            upper = np.where(crossed, mid, upper)
        else:
            # The passes ran out right after a tightening: check the rows
            # against the final box too.
            activities = self._activities(lower, upper)
            if self._rows_missed(*activities[4:]):
                return lower, upper, True
        return lower, upper, False

    def _rows_missed(self, min_act, max_act, n_min_inf, n_max_inf):
        """Whether some row's whole activity range misses its bounds."""
        tol = self.tol
        return bool(
            np.any(
                (n_min_inf == 0)
                & (min_act > self.row_upper + tol * (1.0 + np.abs(self.row_upper)))
            )
            or np.any(
                (n_max_inf == 0)
                & (max_act < self.row_lower - tol * (1.0 + np.abs(self.row_lower)))
            )
        )


def propagate_bounds(
    A, row_lower, row_upper, lower, upper, integrality, max_passes=5, tol=1e-6
):
    """Activity-based bound tightening (see ``BoundPropagator``) for
    ``row_lower <= A x <= row_upper, lower <= x <= upper``, with integer
    bounds rounded. Returns ``(lower, upper, infeasible)``."""
    propagator = BoundPropagator(A, row_lower, row_upper, integrality, tol=tol)
    return propagator(lower, upper, max_passes=max_passes)


def tighten_coefficients(mip, lower, upper):
    """
    Big-M coefficient tightening of the inequality rows over the box
    ``[lower, upper]``, which must hold every feasible point (the root's,
    after propagation).

    In a row ``a·x <= b`` let ``e = maxact − b``, the most the row's maximum
    activity over the box exceeds ``b``. A binary ``z`` with ``|a_z| > e``
    can have its coefficient cut to magnitude ``e`` without changing the
    row at ``z = 0`` or ``z = 1``:

    * ``a_z < 0`` (a big-M row such as ``y − M z <= 0``): the row is
      redundant at ``z = 1`` already, so ``a_z`` becomes ``−e``;
    * ``a_z > 0``: the row is redundant at ``z = 0``, so ``a_z`` becomes
      ``e`` and ``b`` drops by ``a_z − e``.

    Neither change moves ``e``, so every binary of a row is tightened in one
    pass. The integer-feasible points are unchanged; the LP relaxation is
    tighter and the rows' coefficient spread smaller, which helps a
    first-order LP solver. Continuous bounds are widened by 1e-7 (relative)
    against rounding error, and a cut is made only when it shrinks a
    coefficient by more than 1e-3 (relative), on rows with ``e`` above
    1e-6·(1 + |b|).

    Returns ``(mip, count)``: a new ``MIP`` (the same object when nothing
    changed) and the number of coefficients tightened.
    """
    A = sp.csr_matrix(mip.A, copy=True)
    m = A.shape[0]
    integrality = mip.integrality
    cont = ~integrality
    lower = np.where(cont, lower - 1e-7 * (1.0 + np.abs(lower)), lower)
    upper = np.where(cont, upper + 1e-7 * (1.0 + np.abs(upper)), upper)
    binary = integrality & (lower == 0.0) & (upper == 1.0)

    rows = np.repeat(np.arange(m), np.diff(A.indptr))
    cols, a = A.indices, A.data
    with np.errstate(invalid="ignore"):
        contribution = np.where(a > 0, a * upper[cols], a * lower[cols])
    infinite = ~np.isfinite(contribution)
    n_infinite = np.bincount(rows, weights=infinite.astype(np.float64), minlength=m)
    max_activity = np.bincount(
        rows, weights=np.where(infinite, 0.0, contribution), minlength=m
    )
    b = mip.b.copy()
    excess = max_activity - b
    usable = (
        (np.arange(m) >= mip.n_eq)
        & (n_infinite == 0)
        & (excess > 1e-6 * (1.0 + np.abs(b)))
    )
    # A little looser than e, against rounding error (looser stays valid).
    target = excess + 1e-9 * (1.0 + np.abs(b) + np.abs(max_activity))
    e = target[rows]
    hit = usable[rows] & binary[cols] & (np.abs(a) > (1.0 + 1e-3) * e)
    if not hit.any():
        return mip, 0
    positive = hit & (a > 0)
    b -= np.bincount(rows[positive], weights=a[positive] - e[positive], minlength=m)
    A.data = np.where(hit, np.sign(a) * e, a)

    lp = LP(
        mip.c,
        A[: mip.n_eq].tocsc(),
        b[: mip.n_eq],
        A[mip.n_eq :].tocsr(),
        b[mip.n_eq :],
        mip.lower,
        mip.upper,
    )
    return MIP(lp, integrality, mip.offset, mip.maximise), int(hit.sum())


def lagrangian_bound(
    c, A, b, n_eq, lower, upper, y, A_T=None, reduced_cost_tolerance=0.0
):
    """
    The Lagrangian dual bound of the LP
    ``min cᵀx s.t. A[:n_eq] x = b[:n_eq], A[n_eq:] x <= b[n_eq:], lower <= x <= upper``
    at the duals ``y`` (Jaddle's convention: ``L = cᵀx + yᵀ(Ax − b)``):

        g(y) = −bᵀy + Σⱼ min(rⱼlⱼ, rⱼuⱼ),    r = c + Aᵀy,

    with the inequality duals clipped to ``y >= 0``. It is a lower bound on
    the LP's optimum (and so on any MIP whose feasible set lies in the box)
    for every ``y``. A reduced cost pointing at an infinite bound makes it
    ``-inf``; those with ``|rⱼ| <= reduced_cost_tolerance`` are dropped
    instead, and the bound is then not rigorous.

    Returns ``(bound, rigorous)``.
    """
    constant, r = lagrangian_terms(c, A, b, n_eq, y, A_T=A_T)
    return box_bound(constant, r, lower, upper, reduced_cost_tolerance)


def lagrangian_terms(c, A, b, n_eq, y, A_T=None):
    """The parts of ``lagrangian_bound`` that do not depend on the box:
    ``(−bᵀy, r)`` with ``r = c + Aᵀy`` (inequality duals clipped to
    ``y >= 0``). For every feasible ``x``, ``cᵀx >= −bᵀy + rᵀx``."""
    y = np.array(y, dtype=np.float64)
    y[n_eq:] = np.maximum(y[n_eq:], 0.0)
    r = c + (A_T @ y if A_T is not None else A.T @ y)
    return float(-(b @ y)), r


def box_bound(constant, r, lower, upper, reduced_cost_tolerance=0.0):
    """``constant + Σⱼ min(rⱼlⱼ, rⱼuⱼ)``: the Lagrangian bound over the box
    ``[lower, upper]`` from ``lagrangian_terms``. One set of duals bounds
    any box this way, so a parent's duals bound its children. Returns
    ``(bound, rigorous)`` as ``lagrangian_bound`` does."""
    with np.errstate(invalid="ignore"):
        term = np.where(r > 0, r * lower, np.where(r < 0, r * upper, 0.0))
    bad = ~np.isfinite(term)
    if bad.any():
        if np.any(np.abs(r[bad]) > reduced_cost_tolerance):
            return -np.inf, True
        rigorous = False
    else:
        rigorous = True
    return float(constant + term[~bad].sum()), rigorous


def reduced_cost_tightening(r, bound, cutoff, lower, upper, integrality):
    """
    Integer bounds tightened by reduced costs: integer values that cannot
    give an objective below ``cutoff``.

    ``bound`` is the Lagrangian bound over ``[lower, upper]`` with reduced
    costs ``r`` (``box_bound``). Every feasible ``x`` in the box has
    ``cᵀx >= bound + rⱼ(xⱼ − lⱼ)`` for ``rⱼ > 0`` (and
    ``bound + |rⱼ|(uⱼ − xⱼ)`` for ``rⱼ < 0``), so ``cᵀx < cutoff`` needs

        xⱼ <= lⱼ + (cutoff − bound)/rⱼ     or     xⱼ >= uⱼ − (cutoff − bound)/|rⱼ|.

    Valid for any duals; the bound must be rigorous. Returns
    ``(lower, upper, changed)`` (new arrays when ``changed``).
    """
    room = cutoff - bound
    if not (np.isfinite(room) and room >= 0):
        return lower, upper, False
    ints = integrality
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        reach = room / np.abs(r)
        new_upper = np.floor(lower + reach + 1e-6)
        new_lower = np.ceil(upper - reach - 1e-6)
    tighten_upper = ints & (r > 0) & np.isfinite(new_upper) & (new_upper < upper)
    tighten_lower = ints & (r < 0) & np.isfinite(new_lower) & (new_lower > lower)
    if not (tighten_upper.any() or tighten_lower.any()):
        return lower, upper, False
    lower, upper = lower.copy(), upper.copy()
    upper[tighten_upper] = new_upper[tighten_upper]
    lower[tighten_lower] = new_lower[tighten_lower]
    return lower, upper, True


def objective_step(c, integrality):
    """
    The largest ``g`` with ``cᵀx`` a multiple of ``g`` at every ``x`` with
    integral integers, or ``None``: continuous variables must have zero
    cost, and the integer costs, scaled by a power of ten up to 10⁴, must
    be integers. Their greatest common divisor (unscaled) is ``g``.
    """
    c = np.asarray(c, np.float64)
    if np.any(c[~integrality] != 0.0):
        return None
    costs = np.abs(c[integrality])
    costs = costs[costs != 0.0]
    if costs.size == 0:
        return None
    for scale in (1.0, 10.0, 100.0, 1e3, 1e4):
        scaled = costs * scale
        if scaled.max() >= 2.0**52:
            return None
        rounded = np.round(scaled)
        if np.all(np.abs(scaled - rounded) <= 1e-9 * np.maximum(1.0, scaled)):
            return float(np.gcd.reduce(rounded.astype(np.int64))) / scale
    return None


# %%
# Cutting planes
#
# Cuts are added at the root only (cut-and-branch): each round changes the
# LP's row count, and every new shape recompiles the LP solver, so cuts at
# the nodes would recompile at nearly every node. Both families here are
# built from a row and the variable bounds alone, so they are valid however
# inaccurate the LP point is; the point only decides which cuts to keep.

# A coefficient below this fraction of the largest is removed from a cut.
_CUT_TINY = 1e-6
# Rows whose slack at the LP point is within this fraction of 1 + |b| start
# a separation.
_CUT_TIGHT_ROWS = 0.05
# At most this many rows start a separation per round.
_CUT_MAX_ROWS = 2000
# Rows aggregated into one c-MIR base inequality, beyond the first.
_CUT_MAX_AGGREGATION = 2
# Two cuts this parallel (cosine) are duplicates.
_CUT_MAX_PARALLELISM = 0.99
# Tree cuts (``tree_cuts``): most added per node round, and the least room
# reserved for them (rows) beyond the root's cuts.
_TREE_CUTS_PER_ROUND = 10
_TREE_CUT_MIN_ROWS = 32


def _clean_cut(idx, coef, rhs, lower, upper):
    """Remove tiny coefficients from the cut ``coef · x[idx] <= rhs``, moving
    each one's worst case over the bounds to the right-hand side so the cut
    stays valid, and add a small safety margin. ``None`` if a removed
    variable has an infinite bound in the way."""
    keep = coef != 0.0
    idx, coef = idx[keep], coef[keep]
    if idx.size == 0 or not np.isfinite(rhs):
        return None
    tiny = np.abs(coef) < _CUT_TINY * np.abs(coef).max()
    if tiny.any():
        c = coef[tiny]
        with np.errstate(invalid="ignore"):
            worst = np.where(c > 0, c * lower[idx[tiny]], c * upper[idx[tiny]])
        if not np.all(np.isfinite(worst)):
            return None
        rhs -= worst.sum()
        idx, coef = idx[~tiny], coef[~tiny]
    return idx, coef, rhs + 1e-9 * (1.0 + abs(rhs))


# Covers tried per row: greedy by (1 − z*)/w, by 1 − z*, and heaviest first.
_COVER_ORDERS = 3


def _knapsack(idx, a, b, x, lower, upper, is_binary):
    """The binary knapsack of the row ``a · x[idx] <= b``: non-binary
    variables moved to the bound that minimises the row, binaries with
    negative coefficients complemented (``z = 1 − x``). Returns
    ``(cols, w, complemented, z, capacity)`` with weights ``w > 0`` and the
    LP point ``z`` in complemented form, or ``None`` when no cover exists."""
    binary = is_binary[idx]
    if binary.sum() < 2:
        return None
    other = ~binary
    a_other = a[other]
    with np.errstate(invalid="ignore"):
        low = np.where(
            a_other > 0, a_other * lower[idx[other]], a_other * upper[idx[other]]
        )
    if not np.all(np.isfinite(low)):
        return None
    cols, w = idx[binary], a[binary]
    complemented = w < 0
    w = np.abs(w)
    capacity = b - low.sum() + w[complemented].sum()
    z = np.clip(np.where(complemented, 1.0 - x[cols], x[cols]), 0.0, 1.0)
    useful = w > 1e-12
    cols, w, complemented, z = cols[useful], w[useful], complemented[useful], z[useful]
    if capacity < 0 or w.sum() <= capacity + 1e-9 * (1.0 + abs(capacity)):
        return None
    return cols, w, complemented, z, capacity


def _lift_cover(cover, w, z, capacity):
    """Sequential up-lifting of the cover inequality ``Σ_C z <= |C| − 1``
    over the knapsack ``w · z <= capacity``. Each item outside ``C`` (most
    used by the LP point first) gets ``α_j = |C| − 1 − max{π · z : w · z <=
    capacity − w_j}`` over the items placed so far, solved exactly by a
    dynamic program over the (integer) values ``π``: ``least[v]`` is the
    least weight reaching value ``v``. Returns the coefficients ``π``."""
    rhs = cover.size - 1
    pi = np.zeros(w.size)
    pi[cover] = 1.0
    least = np.full(rhs + 1, np.inf)
    least[0] = 0.0
    for item in cover:
        update = least[:-1] + w[item]
        least[1:] = np.minimum(least[1:], update)
    slack = 1e-9 * (1.0 + abs(capacity))
    rest = np.setdiff1d(np.arange(w.size), cover)
    for item in rest[np.lexsort((-w[rest], -z[rest]))]:
        room = capacity - w[item] + slack
        if room < 0:
            alpha = rhs  # z_item = 1 alone breaks the row
        else:
            alpha = rhs - int(np.flatnonzero(least <= room).max())
        if alpha <= 0:
            continue
        pi[item] = alpha
        if alpha <= rhs:
            update = least[: rhs + 1 - alpha] + w[item]
            least[alpha:] = np.minimum(least[alpha:], update)
    return pi


def cover_cuts(idx, a, b, x, lower, upper, is_binary, max_covers=_COVER_ORDERS):
    """
    Lifted knapsack cover cuts from the row ``a · x[idx] <= b`` that cut off
    ``x``, most violated first.

    Variables that are not binary are moved to their bound that minimises
    the row, leaving a knapsack over the binaries; binaries with negative
    coefficients are complemented (``z = 1 − x``). A cover ``C`` is a set
    whose weights exceed the capacity, so not all of it can be 1:
    ``Σ_C z <= |C| − 1``. Covers are built greedily in up to ``max_covers``
    orders at the LP point (by ``(1 − z*)/w``, by ``1 − z*``, heaviest
    first), made minimal, and lifted exactly (``_lift_cover``). Returns a
    list of ``(idx, coef, rhs)`` for ``coef · x[idx] <= rhs``.
    """
    knapsack = _knapsack(idx, a, b, x, lower, upper, is_binary)
    if knapsack is None:
        return []
    cols, w, complemented, z, capacity = knapsack
    threshold = capacity + 1e-9 * (1.0 + abs(capacity))
    orders = (
        np.argsort((1.0 - z) / w, kind="stable"),
        np.lexsort((-w, 1.0 - z)),
        np.argsort(-w, kind="stable"),
    )[:max_covers]
    seen, cuts = set(), []
    for order in orders:
        exceeds = np.cumsum(w[order]) > threshold
        if not exceeds.any():
            continue
        cover = order[: np.argmax(exceeds) + 1]
        # Minimal cover: drop the items the LP point uses least while it
        # still exceeds the capacity.
        total = w[cover].sum()
        keep = np.ones(cover.size, dtype=bool)
        for position in np.argsort(z[cover], kind="stable"):
            if total - w[cover[position]] > threshold:
                total -= w[cover[position]]
                keep[position] = False
        cover = np.sort(cover[keep])
        key = cover.tobytes()
        if key in seen:
            continue
        seen.add(key)
        pi = _lift_cover(cover, w, z, capacity)
        rhs_z = cover.size - 1.0
        violation = pi @ z - rhs_z
        if violation <= 1e-6:
            continue
        members = np.flatnonzero(pi)
        coef = np.where(complemented[members], -pi[members], pi[members])
        rhs = rhs_z - pi[members][complemented[members]].sum()
        cuts.append((violation / np.linalg.norm(pi[members]), cols[members], coef, rhs))
    cuts.sort(key=lambda cut: -cut[0])
    return [cut[1:] for cut in cuts]


def cover_cut(idx, a, b, x, lower, upper, is_binary):
    """The most violated of ``cover_cuts`` (by efficacy), or ``None``."""
    cuts = cover_cuts(idx, a, b, x, lower, upper, is_binary)
    return cuts[0] if cuts else None


def _cmir_for(a, b, xv, lo, up, integer, use_upper, delta):
    """The c-MIR cut of ``a · x <= b`` for one complementation (``use_upper``:
    which variables are shifted to their upper bound) and scaling ``delta``,
    as ``(efficacy, coef, rhs)`` in the original variables, or ``None``."""
    bound = np.where(use_upper, up, lo)
    sign = np.where(use_upper, -1.0, 1.0)
    at = a * sign
    beta = (b - a @ bound) / delta
    f = beta - np.floor(beta)
    if f < 0.01 or f > 0.99:
        return None
    g = at / delta
    fj = g - np.floor(g)
    coef_t = np.where(
        integer,
        np.floor(g) + np.maximum(fj - f, 0.0) / (1.0 - f),
        np.where(~integer & (at < 0), at / (delta * (1.0 - f)), 0.0),
    )
    coef = coef_t * sign
    norm = np.linalg.norm(coef)
    if norm == 0:
        return None
    rhs = np.floor(beta) + coef @ bound
    return (coef @ xv - rhs) / norm, coef, rhs


def cmir_cut(idx, a, b, x, lower, upper, is_int, max_divisors=64):
    """
    The best complemented mixed-integer rounding (c-MIR) cut from the
    inequality ``a · x[idx] <= b``, or ``None``.

    Every variable is shifted to a finite bound, ``t = x − l`` or
    ``t = u − x``, so ``t >= 0`` (and integral for an integer variable, whose
    bounds are integers). With ``s = Σ −aⱼtⱼ`` over continuous terms with
    negative coefficient (the positive ones are dropped), the row gives
    ``Σ_I aⱼtⱼ − s <= b'``. For a scaling ``δ``, with ``f = frac(b'/δ)`` and
    ``fⱼ = frac(aⱼ/δ)``, the MIR inequality

        Σ_I (⌊aⱼ/δ⌋ + max(fⱼ − f, 0)/(1 − f)) tⱼ − s / (δ(1 − f)) <= ⌊b'/δ⌋

    holds for every integer point. The search (Marchand and Wolsey):
    complement each variable to its bound nearer the LP point ``x``; try as
    ``δ`` every distinct |coefficient| of an integer variable away from its
    bound (up to ``max_divisors``, largest first) and 1, then halvings of the
    best; then flip the complementation of each integer variable strictly
    between its bounds (farthest from a bound first), keeping a flip when it
    raises the efficacy. Cuts are compared by efficacy (distance cut off) in
    the original variables. Returns ``(idx, coef, rhs)``, only if it cuts
    off ``x``.
    """
    lo, up, xv, integer = lower[idx], upper[idx], x[idx], is_int[idx]
    if not integer.any():
        return None
    use_upper = np.isfinite(up) & (~np.isfinite(lo) | (up - xv < xv - lo))
    if not np.all(np.isfinite(np.where(use_upper, up, lo))):
        return None
    t = np.maximum(np.where(use_upper, up - xv, xv - lo), 0.0)
    divisors = np.abs(a[integer & (t > 1e-6)])
    divisors = np.unique(divisors[divisors > 1e-9])[::-1][:max_divisors]
    if divisors.size == 0:
        return None
    divisors = np.append(divisors, 1.0)

    best, best_delta = None, None
    for delta in divisors:
        cut = _cmir_for(a, b, xv, lo, up, integer, use_upper, delta)
        if cut is not None and (best is None or cut[0] > best[0]):
            best, best_delta = cut, delta
    if best is None:
        return None
    for k in (2.0, 4.0, 8.0):
        cut = _cmir_for(a, b, xv, lo, up, integer, use_upper, best_delta / k)
        if cut is not None and cut[0] > best[0]:
            best, best_delta = cut, best_delta / k
    # Complementation flips, for integer variables strictly inside their
    # (finite) bounds.
    with np.errstate(invalid="ignore"):
        inside = integer & np.isfinite(lo) & np.isfinite(up) & (xv > lo + 1e-6) & (xv < up - 1e-6)
    distance = np.minimum(xv - lo, up - xv)
    for j in np.flatnonzero(inside)[np.argsort(-distance[inside], kind="stable")]:
        flipped = use_upper.copy()
        flipped[j] = not flipped[j]
        cut = _cmir_for(a, b, xv, lo, up, integer, flipped, best_delta)
        if cut is not None and cut[0] > best[0] + 1e-12:
            best, use_upper = cut, flipped
    efficacy, coef, rhs = best
    if efficacy <= 1e-9:
        return None
    return idx, coef, rhs


def mir_cut(idx, a, b, x, lower, upper, is_int, max_divisors=64):
    """``cmir_cut`` (kept under its earlier name)."""
    return cmir_cut(idx, a, b, x, lower, upper, is_int, max_divisors)


def gomory_cuts(
    A, b, n_eq, lower, upper, integrality, basis, x, max_cuts=200, tableau_mir=False
):
    """
    Gomory mixed-integer (GMI) cuts from an optimal basis of the LP
    ``A[:n_eq] x = b[:n_eq], A[n_eq:] x <= b[n_eq:], lower <= x <= upper``.

    ``basis`` is a ``highspy.HighsBasis`` for that LP's rows in this order
    (as ``jaddle_linear.crossover`` returns it) and ``x`` the vertex. Each
    row is ``aᵢx + sᵢ = bᵢ`` with a slack ``sᵢ >= 0`` (fixed at 0 on an
    equality row). For each integer basic variable ``x_k`` with a value
    whose fractional part ``f0`` is in [0.01, 0.99], the tableau row
    ``x_k + Σ āⱼ tⱼ = x_k*`` over the nonbasic variables shifted to their
    bounds (``tⱼ >= 0``) gives the GMI cut

        Σ_{int, fⱼ<=f0} fⱼ/f0 tⱼ + Σ_{int, fⱼ>f0} (1−fⱼ)/(1−f0) tⱼ
            + Σ_{cont, āⱼ>0} āⱼ/f0 tⱼ − Σ_{cont, āⱼ<0} āⱼ/(1−f0) tⱼ  >=  1,

    with ``fⱼ = frac(āⱼ)``; slacks count as continuous. It is mapped back to
    ``x`` (slacks through their rows) as ``coef · x[idx] <= rhs``. Tableau
    rows come from a sparse LU of the basis matrix; a row whose basic
    columns do not come out as a unit vector is skipped as numerically
    unreliable. Returns a list of ``("gomory", idx, coef, rhs)``, at most
    ``max_cuts``, for the basic variables nearest ½ first.

    With ``tableau_mir``, each tableau row is also handed to ``cmir_cut`` as
    an equation over the variables and the inequality slacks (continuous,
    ``>= 0``), in both senses; the slacks are mapped back through
    ``s = b − A x``. Per row, the deeper of the GMI cut and the c-MIR cuts
    is kept, the latter as ``"tableau_mir"``.
    """
    import highspy
    from scipy.sparse.linalg import splu

    BS = highspy.HighsBasisStatus
    A = sp.csr_matrix(A)
    m, n = A.shape
    x = np.asarray(x, np.float64)
    col_status = list(basis.col_status)
    row_status = list(basis.row_status)
    if len(col_status) != n or len(row_status) != m:
        return []
    col_basic = np.array([s == BS.kBasic for s in col_status])
    row_basic = np.array([s == BS.kBasic for s in row_status])
    if col_basic.sum() + row_basic.sum() != m:
        return []

    # Which bound each nonbasic column sits at (None: free).
    at_upper = np.array([s == BS.kUpper for s in col_status])
    at_lower = np.array([s == BS.kLower for s in col_status])
    undecided = ~col_basic & ~at_upper & ~at_lower
    with np.errstate(invalid="ignore"):
        nearer_upper = np.isfinite(upper) & (
            ~np.isfinite(lower) | (np.abs(x - upper) < np.abs(x - lower))
        )
    at_upper |= undecided & nearer_upper
    at_lower |= undecided & ~nearer_upper & np.isfinite(lower)
    free = ~col_basic & ~at_upper & ~at_lower
    bound = np.where(at_upper, upper, np.where(at_lower, lower, 0.0))

    basic_cols = np.flatnonzero(col_basic)
    basic_rows = np.flatnonzero(row_basic)
    B = sp.hstack(
        [
            A.tocsc()[:, basic_cols],
            sp.csc_matrix(
                (np.ones(basic_rows.size), (basic_rows, np.arange(basic_rows.size))),
                shape=(m, basic_rows.size),
            ),
        ],
        format="csc",
    )
    try:
        lu = splu(B)
    except RuntimeError:  # singular
        return []

    frac = x - np.floor(x)
    candidates = [
        (abs(frac[j] - 0.5), p, j)
        for p, j in enumerate(basic_cols)
        if integrality[j] and 0.01 <= frac[j] <= 0.99
    ]
    candidates.sort()
    # Slacks of inequality rows are continuous variables >= 0; equality-row
    # slacks are fixed at 0, so they drop out.
    ineq_row = np.arange(m) >= n_eq
    nonbasic_slack = ~row_basic & ineq_row
    AT = A.T.tocsr()
    cuts = []
    if tableau_mir:
        # The extended space: the variables, then the inequality slacks.
        ineq = np.flatnonzero(ineq_row)
        A_ineq_T = A[ineq].T.tocsr()
        ext_lower = np.concatenate([lower, np.zeros(ineq.size)])
        ext_upper = np.concatenate([upper, np.full(ineq.size, np.inf)])
        ext_int = np.concatenate([integrality, np.zeros(ineq.size, bool)])
        ext_x = np.concatenate([x, np.maximum(b[ineq] - A[ineq] @ x, 0.0)])

    def efficacy(cut):
        _, idx, coef, rhs = cut
        return (coef @ x[idx] - rhs) / np.linalg.norm(coef)

    for _, p, k in candidates[:max_cuts]:
        e = np.zeros(m)
        e[p] = 1.0
        u = lu.solve(e, trans="T")
        abar = AT @ u  # tableau entries of the columns
        expected = np.zeros(basic_cols.size)
        expected[p] = 1.0
        if np.abs(abar[basic_cols] - expected).max() > 1e-7:
            continue
        nonbasic = ~col_basic & (np.abs(abar) > 0)
        if np.any(free & nonbasic):
            continue
        f0 = frac[k]
        # Shifted coefficients a'ⱼ of tⱼ: t = x − l (a' = ā) or u − x (a' = −ā).
        a_col = np.where(at_upper, -abar, abar)
        a_col[col_basic] = 0.0
        integer_col = integrality & ~col_basic
        fj = a_col - np.floor(a_col)
        pi_col = np.where(
            integer_col,
            np.where(fj <= f0, fj / f0, (1.0 - fj) / (1.0 - f0)),
            np.where(a_col >= 0, a_col / f0, -a_col / (1.0 - f0)),
        )
        pi_col[col_basic] = 0.0
        a_slack = np.where(nonbasic_slack, u, 0.0)
        pi_slack = np.where(a_slack >= 0, a_slack / f0, -a_slack / (1.0 - f0))
        pi_slack[~nonbasic_slack] = 0.0

        # Σ π t >= 1 in x: t = ±(x − bound), s = b − A x.
        sign = np.where(at_upper, -1.0, 1.0)
        coef_ge = pi_col * sign - AT @ pi_slack
        constant = -(pi_col * sign) @ bound + pi_slack @ b
        rhs_ge = 1.0 - constant
        idx = np.flatnonzero(coef_ge)
        if idx.size == 0 or idx.size > max(n // 2, min(n, 100)):
            continue
        cleaned = _clean_cut(idx, -coef_ge[idx], -rhs_ge, lower, upper)
        best = ("gomory", *cleaned) if cleaned is not None else None
        if tableau_mir:
            # The row Σ āⱼxⱼ + Σ uᵢsᵢ = u·b (equality slacks are 0).
            row = np.concatenate([abar, u[ineq]])
            row_idx = np.flatnonzero(np.abs(row) > 1e-12)
            for sense in (1.0, -1.0):
                found = cmir_cut(
                    row_idx, sense * row[row_idx], sense * (u @ b),
                    ext_x, ext_lower, ext_upper, ext_int,
                )
                if found is None:
                    continue
                c_idx, c_coef, c_rhs = found
                c_x = np.zeros(n)
                c_s = np.zeros(ineq.size)
                in_x = c_idx < n
                c_x[c_idx[in_x]] = c_coef[in_x]
                c_s[c_idx[~in_x] - n] = c_coef[~in_x]
                coef_x = c_x - A_ineq_T @ c_s
                rhs_x = c_rhs - c_s @ b[ineq]
                x_idx = np.flatnonzero(coef_x)
                if x_idx.size == 0 or x_idx.size > max(n // 2, min(n, 100)):
                    continue
                cleaned = _clean_cut(x_idx, coef_x[x_idx], rhs_x, lower, upper)
                if cleaned is None:
                    continue
                candidate = ("tableau_mir", *cleaned)
                if best is None or efficacy(candidate) > efficacy(best):
                    best = candidate
        if best is not None:
            cuts.append(best)
    return cuts


class CutSeparator:
    """Cover and c-MIR separation from the rows of a ``MIP`` over bounds that
    hold for every feasible point (the root's, after propagation)."""

    def __init__(
        self, mip, lower, upper, aggregate_integers=False,
        max_aggregation=_CUT_MAX_AGGREGATION,
    ):
        # ``aggregate_integers``: when an aggregate has no continuous variable
        # left to cancel, cancel its most fractional integer term instead (so
        # pure-integer rows aggregate too). ``max_aggregation``: rows added.
        self.aggregate_integers = aggregate_integers
        self.max_aggregation = max_aggregation
        self.A = sp.csr_matrix(mip.A)
        self.A_csc = self.A.tocsc()
        self.b = mip.b
        self.is_eq = np.arange(self.A.shape[0]) < mip.n_eq
        self.lower, self.upper = np.asarray(lower), np.asarray(upper)
        self.is_int = mip.integrality
        self.is_binary = self.is_int & (self.lower == 0.0) & (self.upper == 1.0)

    def row(self, r):
        start, end = self.A.indptr[r], self.A.indptr[r + 1]
        return self.A.indices[start:end], self.A.data[start:end]

    def __call__(self, x):
        """Cuts violated by ``x``, as ``(family, idx, coef, rhs)``."""
        x = np.asarray(x, np.float64)
        slack = self.b - self.A @ x
        relative = np.where(self.is_eq, np.abs(slack), slack) / (1.0 + np.abs(self.b))
        rows = np.flatnonzero(relative <= _CUT_TIGHT_ROWS)
        rows = rows[np.argsort(relative[rows], kind="stable")][:_CUT_MAX_ROWS]
        self.relative = relative
        cuts = []
        for r in rows:
            idx, a = self.row(r)
            if idx.size == 0:
                continue
            for s in (1.0, -1.0) if self.is_eq[r] else (1.0,):
                for found in cover_cuts(
                    idx, s * a, s * self.b[r], x, self.lower, self.upper, self.is_binary
                ):
                    cuts.append(("cover", *found))
                cuts += self.aggregated_mir(r, s, x)
        out = []
        for family, idx, coef, rhs in cuts:
            cleaned = _clean_cut(idx, coef, rhs, self.lower, self.upper)
            if cleaned is not None:
                out.append((family, *cleaned))
        return out

    def aggregated_mir(self, r, s, x):
        """Cuts from row ``r`` (times ``s``) and aggregates of it with up to
        ``max_aggregation`` more rows, each added to cancel the continuous
        variable furthest from its bounds (or, with ``aggregate_integers``
        and none left, the integer term ``|aⱼ|·min(fⱼ, 1 − fⱼ)`` largest at
        ``x``). Returns the best c-MIR cut over the aggregates, and the cover
        cuts of the aggregates beyond the row itself."""
        idx, a = self.row(r)
        a, b = s * a, s * self.b[r]
        used = {r}
        best, best_efficacy = None, 0.0
        covers = []
        for level in range(self.max_aggregation + 1):
            cut = cmir_cut(idx, a, b, x, self.lower, self.upper, self.is_int)
            if cut is not None:
                c_idx, c_coef, c_rhs = cut
                efficacy = (c_coef @ x[c_idx] - c_rhs) / np.linalg.norm(c_coef)
                if efficacy > best_efficacy:
                    best, best_efficacy = cut, efficacy
            if level > 0:
                covers += [
                    ("cover", *found)
                    for found in cover_cuts(
                        idx, a, b, x, self.lower, self.upper, self.is_binary
                    )
                ]
            if level == self.max_aggregation:
                break
            continuous = ~self.is_int[idx]
            with np.errstate(invalid="ignore"):
                distance = np.minimum(
                    x[idx] - self.lower[idx], self.upper[idx] - x[idx]
                )
            distance = np.where(continuous, np.nan_to_num(distance, posinf=1e30), -1.0)
            pos = int(np.argmax(distance))
            if distance[pos] <= 1e-6:
                if not self.aggregate_integers:
                    break
                frac = x[idx] - np.floor(x[idx])
                score = np.where(
                    self.is_int[idx], np.abs(a) * np.minimum(frac, 1.0 - frac), -1.0
                )
                pos = int(np.argmax(score))
                if score[pos] <= 1e-6:
                    break
            col, coef = idx[pos], a[pos]
            start, end = self.A_csc.indptr[col], self.A_csc.indptr[col + 1]
            partner = None
            for r2, a2 in sorted(
                zip(self.A_csc.indices[start:end], self.A_csc.data[start:end]),
                key=lambda item: self.relative[item[0]],
            ):
                if r2 in used or a2 == 0:
                    continue
                lam = -coef / a2
                if lam < 0 and not self.is_eq[r2]:
                    continue  # a <= row can only be added with a weight >= 0
                partner = (r2, lam)
                break
            if partner is None:
                break
            r2, lam = partner
            idx2, a2 = self.row(r2)
            merged_idx, inverse = np.unique(
                np.concatenate([idx, idx2]), return_inverse=True
            )
            merged = np.bincount(
                inverse, np.concatenate([a, lam * a2]), minlength=merged_idx.size
            )
            merged[merged_idx == col] = 0.0
            nonzero = np.abs(merged) > 1e-12 * np.abs(merged).max(initial=1.0)
            idx, a = merged_idx[nonzero], merged[nonzero]
            b = b + lam * self.b[r2]
            used.add(r2)
            if idx.size == 0:
                break
        return ([("mir", *best)] if best is not None else []) + covers


def select_cuts(cuts, x, n, min_efficacy=1e-4, max_cuts=100):
    """The cuts that ``x`` violates by an efficacy (distance) of at least
    ``min_efficacy``, most efficacious first, skipping near-duplicates, at
    most ``max_cuts`` of them."""
    x = np.asarray(x, np.float64)
    scored = []
    for cut in cuts:
        _, idx, coef, rhs = cut
        norm = np.linalg.norm(coef)
        efficacy = (coef @ x[idx] - rhs) / norm
        if efficacy >= min_efficacy:
            scored.append((efficacy, cut, norm))
    scored.sort(key=lambda item: -item[0])
    chosen, directions = [], []
    for _, cut, norm in scored:
        _, idx, coef, _ = cut
        direction = sp.csr_matrix(
            (coef / norm, idx, [0, idx.size]), shape=(1, n)
        )
        if directions and (
            np.abs(sp.vstack(directions) @ direction.T).max() > _CUT_MAX_PARALLELISM
        ):
            continue
        chosen.append(cut)
        directions.append(direction)
        if len(chosen) >= max_cuts:
            break
    return chosen


def _next_power_of_two(k):
    return 1 << max(int(k) - 1, 0).bit_length()


def _cut_block(cuts, n, min_rows=0, min_nnz=0):
    """The cuts as a CSR block ``A x <= b``, padded with empty rows and
    explicit zero entries to at least ``min_rows`` rows and ``min_nnz``
    nonzeros, rounded up to powers of two, so cut rounds of similar size
    share one compiled LP solver."""
    nnz = sum(cut["idx"].size for cut in cuts)
    total_nnz = _next_power_of_two(max(nnz, min_nnz, 1))
    padding = total_nnz - nnz
    rows = _next_power_of_two(max(len(cuts) + -(-padding // n), min_rows))
    indices = [cut["idx"] for cut in cuts]
    data = [cut["coef"] for cut in cuts]
    lengths = [cut["idx"].size for cut in cuts]
    # Zero entries fill the padding rows, at most one per column per row.
    for start in range(0, padding, n):
        width = min(n, padding - start)
        indices.append(np.arange(width))
        data.append(np.zeros(width))
        lengths.append(width)
    lengths += [0] * (rows - len(lengths))
    indptr = np.concatenate([[0], np.cumsum(lengths)])
    A = sp.csr_matrix(
        (
            np.concatenate(data) if data else np.zeros(0),
            np.concatenate(indices).astype(np.int64) if indices else np.zeros(0, np.int64),
            indptr,
        ),
        shape=(rows, n),
    )
    b = np.zeros(rows)
    b[: len(cuts)] = [cut["rhs"] for cut in cuts]
    return A, b


# %%
# Primal heuristics
def fix_and_propagate(
    x, lower, upper, integrality, propagator, chunks=20, max_propagations=200
):
    """
    Round the LP point ``x`` to an integer assignment that bound propagation
    accepts, or return ``None``.

    The integer variables are fixed to ``round(x)`` (within the box
    ``lower``/``upper``) in chunks, and ``propagator`` (a
    ``BoundPropagator``) runs after each chunk. Variables that round above
    their lower bound go first, the most nearly integral first; the rest
    follow by decreasing value. On assignment and partitioning rows this
    fixes the ones first and lets propagation decide the zeros, rather than
    fixing zeros early and leaving a row nowhere to go. Propagation fixes or tightens
    the rest, so later chunks round within what earlier ones left feasible.
    A chunk that propagation rejects is split in half and retried; a single
    variable that is rejected tries its other rounding, and if that fails
    too the heuristic gives up. At most ``max_propagations`` propagations
    run, so the cost is bounded on large models.

    Returns the point with every integer fixed and the continuous variables
    at ``x`` clipped to their propagated bounds (the caller re-solves them).
    """
    x = np.asarray(x, np.float64)
    lower, upper, infeasible = propagator(lower, upper)
    if infeasible:
        return None
    ints = np.flatnonzero(integrality)
    frac = np.abs(x[ints] - np.round(x[ints]))
    raised = np.clip(np.round(x[ints]), lower[ints], upper[ints]) > lower[ints]
    first = ints[raised][np.argsort(frac[raised], kind="stable")]
    rest = ints[~raised][np.argsort(-x[ints][~raised], kind="stable")]
    pieces = []
    for group in (first, rest):
        near = int(np.count_nonzero(np.abs(x[group] - np.round(x[group])) <= 1e-6))
        if group is first and near:
            pieces.append(group[:near])  # already integral: one chunk
            group = group[near:]
        pieces += [p for p in np.array_split(group, chunks // 2 or 1) if p.size]
    budget = [max_propagations]

    def attempt(cols, values, lo, up):
        if budget[0] <= 0:
            return None
        budget[0] -= 1
        lo, up = lo.copy(), up.copy()
        lo[cols] = up[cols] = values
        lo, up, infeasible = propagator(lo, up)
        return None if infeasible else (lo, up)

    def fix(cols, lo, up):
        cols = cols[lo[cols] < up[cols]]  # propagation may have fixed some
        if cols.size == 0:
            return lo, up
        target = np.clip(np.round(x[cols]), lo[cols], up[cols])
        fixed = attempt(cols, target, lo, up)
        if fixed is not None:
            return fixed
        if cols.size == 1:
            # The other rounding: towards x from the rejected value.
            other = target + (1.0 if x[cols[0]] > target[0] else -1.0)
            if other[0] < lo[cols[0]] or other[0] > up[cols[0]]:
                other = target + (-1.0 if x[cols[0]] > target[0] else 1.0)
                if other[0] < lo[cols[0]] or other[0] > up[cols[0]]:
                    return None
            return attempt(cols, other, lo, up)
        half = cols.size // 2
        first = fix(cols[:half], lo, up)
        if first is None:
            return None
        return fix(cols[half:], *first)

    box = (lower, upper)
    for piece in pieces:
        box = fix(piece, *box)
        if box is None:
            return None
    lower, upper = box
    candidate = np.clip(x, lower, upper)
    candidate[integrality] = lower[integrality]
    return candidate


def _jump_values(data, x, act, weights):
    """The jump value of every variable: the value in its bounds (integral for
    an integer) that minimises the weighted violation of all the rows in its
    column with the other variables held, and that minimum's change from the
    current weighted violation. Each row adds ``w·|a|·dist(v, [lo, hi])``,
    where ``[lo, hi]`` are the values of the variable that satisfy it; the sum
    is convex and piecewise linear, so its minimiser is the weighted median
    of the interval ends. Integers try both roundings of it."""
    rows, cols, a = data["rows"], data["cols"], data["a"]
    row_lower, row_upper = data["row_lower"], data["row_upper"]
    lower, upper, is_int = data["lower"], data["upper"], data["is_int"]
    n = lower.shape[0]
    safe_a = jnp.where(a != 0, a, 1.0)
    rest = act[rows] - a * x[cols]
    t_lower = (row_lower[rows] - rest) / safe_a
    t_upper = (row_upper[rows] - rest) / safe_a
    nonzero = a != 0
    lo = jnp.where(nonzero, jnp.where(a > 0, t_lower, t_upper), -jnp.inf)
    hi = jnp.where(nonzero, jnp.where(a > 0, t_upper, t_lower), jnp.inf)
    wt = jnp.where(nonzero, weights[rows] * jnp.abs(a), 0.0)
    values = jnp.concatenate([lo, hi])
    point_weight = jnp.concatenate([wt, wt])
    point_col = jnp.concatenate([cols, cols])
    order = jnp.lexsort((values, point_col))
    v_sorted, c_sorted = values[order], point_col[order]
    running = jnp.cumsum(point_weight[order])
    half = jax.ops.segment_sum(wt, cols, num_segments=n)
    before = jnp.cumsum(2.0 * half) - 2.0 * half
    within = running - before[c_sorted]
    reached = within >= half[c_sorted] * (1.0 - 1e-12)
    positions = jnp.arange(values.shape[0])
    first = jax.ops.segment_min(
        jnp.where(reached, positions, values.shape[0]), c_sorted, num_segments=n
    )
    found = first < values.shape[0]
    best = jnp.where(found, v_sorted[jnp.clip(first, 0, values.shape[0] - 1)], x)
    best = jnp.clip(jnp.nan_to_num(best, nan=0.0), lower, upper)
    down = jnp.clip(jnp.where(is_int, jnp.floor(best), best), lower, upper)
    up = jnp.clip(jnp.where(is_int, jnp.ceil(best), best), lower, upper)

    def violation_entries(act_entries):
        return jnp.maximum(act_entries - row_upper[rows], 0.0) + jnp.maximum(
            row_lower[rows] - act_entries, 0.0
        )

    viol = violation_entries(act[rows])

    def change_for(target):
        act_new = act[rows] + a * (target - x)[cols]
        return jax.ops.segment_sum(
            weights[rows] * (violation_entries(act_new) - viol), cols, num_segments=n
        )

    change_down, change_up = change_for(down), change_for(up)
    target = jnp.where(change_up < change_down, up, down)
    change = jnp.minimum(change_down, change_up)
    # The jump value is the best value other than the current one (as in
    # HiGHS): when the minimiser is the current integer value, take the
    # better neighbour (the function is convex, so it is one of them).
    stay = is_int & (jnp.abs(target - x) <= 1e-12)
    below = jnp.clip(x - 1.0, lower, upper)
    above = jnp.clip(x + 1.0, lower, upper)
    change_below = jnp.where(below < x, change_for(below), jnp.inf)
    change_above = jnp.where(above > x, change_for(above), jnp.inf)
    neighbour = jnp.where(change_above < change_below, above, below)
    target = jnp.where(stay, neighbour, target)
    change = jnp.where(stay, jnp.minimum(change_below, change_above), change)
    return target, change


@jax.jit
def _jump_chunk(data, x, weights, key, iterations):
    """Up to ``iterations`` Feasibility Jump steps (see ``feasibility_jump``).
    ``data`` holds the normalised rows as COO entries. Returns
    ``(x, weights, key, steps, feasible)``."""
    rows, cols, a = data["rows"], data["cols"], data["a"]
    row_lower, row_upper = data["row_lower"], data["row_upper"]
    lower, upper, is_int, tol = data["lower"], data["upper"], data["is_int"], data["tol"]
    m, n = row_lower.shape[0], lower.shape[0]

    def violation(act):
        return jnp.maximum(act - row_upper, 0.0) + jnp.maximum(row_lower - act, 0.0)

    def step(carry):
        x, weights, key, i, _ = carry
        key, k_noise, k_row, k_pick, k_coin = jax.random.split(key, 5)
        act = jax.ops.segment_sum(a * x[cols], rows, num_segments=m)
        viol = violation(act)
        violated = viol > tol
        feasible = ~jnp.any(violated)
        in_violated = (
            jax.ops.segment_max(violated[rows].astype(jnp.int32), cols, num_segments=n)
            > 0
        )

        # Every variable's jump value (the best value other than its current
        # one, over all its rows) and the exact change of moving it alone.
        target, change = _jump_values(data, x, act, weights)
        movable = in_violated & (jnp.abs(target - x) > 1e-12)
        score = jnp.where(movable, -change, -jnp.inf)
        good = score > 1e-12

        # Batched moves: an improving variable moves when it has the highest
        # (noisy) score in every row it touches, so no two moving variables
        # share a row. Each row then changes by one variable's move, so the
        # scores add up exactly and every chosen move can be made at once.
        priority = jnp.where(
            good, score * (1.0 + 1e-6 * jax.random.uniform(k_noise, (n,))), -jnp.inf
        )
        row_best = jax.ops.segment_max(priority[cols], rows, num_segments=m)
        winner = good & (
            jax.ops.segment_min(
                (priority[cols] >= row_best[rows]).astype(jnp.int32),
                cols,
                num_segments=n,
            )
            > 0
        )

        # A local minimum (no improving move): as HiGHS's Feasibility Jump,
        # make the violated rows heavier and still move the best variable of
        # a random violated row, even if that makes things worse (1% of the
        # time a random variable of that row).
        r = jnp.argmax(jnp.where(violated, jax.random.uniform(k_row, (m,)), -1.0))
        in_row = (
            jax.ops.segment_max((rows == r).astype(jnp.int32), cols, num_segments=n) > 0
        ) & movable
        best_in_row = jnp.argmax(jnp.where(in_row, score, -jnp.inf))
        random_in_row = jnp.argmax(jnp.where(in_row, jax.random.uniform(k_pick, (n,)), -1.0))
        forced_var = jnp.where(jax.random.uniform(k_coin) < 0.01, random_in_row, best_in_row)
        stuck = ~jnp.any(good) & ~feasible
        forced = stuck & jnp.any(in_row) & (jnp.arange(n) == forced_var)

        moving = (winner & ~feasible) | forced
        x = jnp.where(moving, target, x)
        weights = jnp.where(stuck, weights + violated, weights)
        return x, weights, key, i + jnp.where(feasible, 0, 1), feasible

    def keep_going(carry):
        _, _, _, i, feasible = carry
        return (i < iterations) & ~feasible

    x, weights, key, steps, feasible = jax.lax.while_loop(
        keep_going, step, (x, weights, key, jnp.zeros((), jnp.int32), False)
    )
    return x, weights, key, steps, feasible


def _jump_data(mip, lower, upper, tol):
    """The device arrays ``_jump_chunk`` reads: the MIP's rows (as given, no
    cuts) scaled to unit ∞-norm, with a per-row tolerance matching
    ``MIP.violation``'s, and the box."""
    A = sp.coo_matrix(mip.A)
    dtype = mip.lp.c.dtype
    scale = np.abs(sp.csr_matrix(mip.A)).max(axis=1).toarray().ravel()
    scale = np.where(scale > 0, scale, 1.0)
    row_tol = 0.5 * tol * (1.0 + np.abs(mip.b).max(initial=0.0)) / scale

    def put(v):
        return jax.device_put(np.asarray(v).astype(dtype))

    return dict(
        rows=jax.device_put(A.row.astype(np.int32)),
        cols=jax.device_put(A.col.astype(np.int32)),
        a=put(A.data / scale[A.row]),
        row_lower=put(mip.row_lower / scale),
        row_upper=put(mip.row_upper / scale),
        lower=put(lower),
        upper=put(upper),
        is_int=jax.device_put(mip.integrality),
        tol=put(row_tol),
    )


def feasibility_jump(
    mip,
    x0,
    lower=None,
    upper=None,
    tol=1e-6,
    max_iterations=None,
    deadline=None,
    seed=0,
    chunk=1000,
):
    """
    Feasibility Jump (Luteberget & Sartor, 2023): an LP-free local search
    for a point that meets ``mip``'s rows, from ``x0``.

    Each step looks, for every variable in a violated row, at the value that
    satisfies its most heavily weighted violated row exactly (integers
    rounded far enough), scores each such "jump" by the exact change in the
    weighted violation over the variable's column, and makes the best one.
    When no jump helps, every violated row's weight grows by 1, which
    eventually makes some jump worthwhile. Rows are scaled to unit ∞-norm.
    Every step is a few sparse passes over the nonzeros, on the device, in
    compiled chunks of ``chunk`` steps; ``deadline`` (a ``time.time()``
    value) is checked between chunks.

    ``lower``/``upper`` default to ``mip``'s bounds. ``max_iterations``
    defaults to ``max(10000, 20·(n + m))``. Returns ``(x, feasible, steps)``:
    ``feasible`` means every row is met to ``tol`` (as ``MIP.violation``
    measures it); the caller should still check the point.
    """
    lower = mip.lower if lower is None else np.asarray(lower, np.float64)
    upper = mip.upper if upper is None else np.asarray(upper, np.float64)
    n, m = mip.num_variables, mip.A.shape[0]
    if max_iterations is None:
        max_iterations = max(10_000, 20 * (n + m))
    x = np.clip(np.asarray(x0, np.float64), lower, upper)
    ints = mip.integrality
    x[ints] = np.clip(np.round(x[ints]), lower[ints], upper[ints])
    data = _jump_data(mip, lower, upper, tol)
    dtype = mip.lp.c.dtype
    x = jax.device_put(x.astype(dtype))
    weights = jax.device_put(np.ones(m, dtype=dtype))
    key = jax.random.PRNGKey(seed)
    steps = 0
    feasible = False
    best, best_count = np.asarray(x, np.float64), np.inf
    while steps < max_iterations:
        if deadline is not None and time.time() >= deadline:
            break
        size = int(min(chunk, max_iterations - steps))
        x, weights, key, done, feasible = _jump_chunk(data, x, weights, key, size)
        steps += int(done)
        if bool(feasible):
            return np.asarray(x, np.float64), True, steps
        # Keep the near-miss with the fewest violated rows, for ``repair``.
        point = np.asarray(x, np.float64)
        count = _violated_rows(mip, point, tol).size
        if count < best_count:
            best, best_count = point, count
    return best, False, steps


def highs_incumbent(mip, max_seconds=30.0):
    """A feasible point of ``mip`` from HiGHS's MIP solver, which stops at
    its first solution (``mip_max_improving_sols=1``) or after
    ``max_seconds``. Returns it in ``mip``'s variables, or ``None``. This
    hands the search to HiGHS; ``solve`` uses it only when asked
    (``highs_incumbent_seconds``)."""
    import highspy

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("time_limit", float(max_seconds))
    highs.setOptionValue("mip_max_improving_sols", 1)
    highs.passModel(to_highs(mip))
    highs.run()
    if highs.getInfo().primal_solution_status != 2:  # kSolutionStatusFeasible
        return None
    return np.array(highs.getSolution().col_value, dtype=np.float64)


def highs_rins(mip, incumbent, x_lp, cutoff, max_seconds=5.0, min_fixed=0.3):
    """
    RINS (relaxation induced neighbourhood search) with HiGHS as the sub-MIP
    solver: the integer variables on which the ``incumbent`` and the LP point
    ``x_lp`` agree are fixed at the incumbent's value, a row
    ``c·x <= cutoff`` (``c`` of ``mip``'s minimisation, no offset) keeps only
    better solutions, and HiGHS searches the rest for up to ``max_seconds``.
    Skipped (``None``) when fewer than ``min_fixed`` of the integers agree,
    since the sub-MIP would then be nearly the whole MIP. Returns the best
    point HiGHS found, or ``None``.
    """
    import highspy

    ints = mip.integrality
    incumbent = np.asarray(incumbent, np.float64)
    agree = ints & (np.abs(np.asarray(x_lp, np.float64) - incumbent) <= 1e-6)
    if ints.sum() == 0 or agree.sum() < min_fixed * ints.sum():
        return None
    model = hh.jaddle_lp_to_highs(mip.lp, ints)
    lower = np.array(model.col_lower_, dtype=np.float64)
    upper = np.array(model.col_upper_, dtype=np.float64)
    lower[agree] = upper[agree] = incumbent[agree]
    model.col_lower_, model.col_upper_ = lower, upper
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("time_limit", float(max_seconds))
    highs.passModel(model)
    nonzero = np.flatnonzero(mip.c)
    highs.addRow(
        -highspy.kHighsInf,
        float(cutoff),
        nonzero.size,
        nonzero.astype(np.int32),
        mip.c[nonzero],
    )
    highs.run()
    if highs.getInfo().primal_solution_status != 2:  # kSolutionStatusFeasible
        return None
    return np.array(highs.getSolution().col_value, dtype=np.float64)


def _violated_rows(mip, x, tol):
    """The rows ``x`` violates by more than ``tol`` (relative to
    ``1 + ‖b‖∞``, as ``MIP.violation`` measures)."""
    Ax = mip.A @ x
    excess = np.maximum(np.maximum(mip.row_lower - Ax, Ax - mip.row_upper), 0.0)
    return np.flatnonzero(excess > tol * (1.0 + np.abs(mip.b).max(initial=0.0)))


def repair(
    mip,
    x,
    lower=None,
    upper=None,
    max_seconds=60.0,
    max_radius=2,
    tol=1e-6,
    max_propagations=20000,
    propagator=None,
):
    """
    Repair a near-feasible integer point ``x`` by a backtracking search with
    bound propagation.

    The variables in the rows ``x`` violates are freed (``radius`` 0); if no
    repair is found, so is every variable sharing a row with them, and so on
    up to ``max_radius``. The rest stay fixed at ``x``. Within that box a
    depth-first search fixes the free integers one at a time, trying ``x``'s
    value first, propagates after each (``BoundPropagator``), and backtracks
    when propagation finds the box infeasible. It solves no LP, so it costs
    no compilation, which on a small neighbourhood would dwarf the search.
    ``lower``/``upper`` (default: ``mip``'s) bound the freed variables;
    ``max_propagations`` and ``max_seconds`` cap the whole call.

    Returns the repaired point in ``mip``'s variables (continuous variables
    clipped to their final bounds, for the caller to re-solve), or ``None``.
    """
    started = time.time()
    lower = mip.lower if lower is None else np.asarray(lower, np.float64)
    upper = mip.upper if upper is None else np.asarray(upper, np.float64)
    x = np.clip(np.asarray(x, np.float64), lower, upper)
    ints = mip.integrality
    x[ints] = np.round(x[ints])
    bad = _violated_rows(mip, x, tol)
    if bad.size == 0:
        return x
    if propagator is None:
        propagator = BoundPropagator(mip.A, mip.row_lower, mip.row_upper, ints)
    A = sp.csr_matrix(mip.A)
    A_csc = A.tocsc()
    free = np.zeros(mip.num_variables, dtype=bool)
    free[np.unique(A[bad].indices)] = True
    budget = [max_propagations]
    deadline = started + max_seconds
    for radius in range(max_radius + 1):
        if radius:
            rows = np.unique(A_csc[:, free].indices)
            grown = free.copy()
            grown[np.unique(A[rows].indices)] = True
            if np.array_equal(grown, free):
                break
            free = grown
        box_lower = np.where(free | ~ints, lower, x)
        box_upper = np.where(free | ~ints, upper, x)
        found = _propagation_search(
            propagator, x, box_lower, box_upper, ints, free, budget, deadline
        )
        if found is not None:
            return found
        if budget[0] <= 0 or time.time() >= deadline:
            break
    return None


def _propagation_search(propagator, hint, lower, upper, ints, free, budget, deadline):
    """Depth-first search for integer values of the ``free`` integers in the
    box that propagation accepts, trying ``hint``'s value first (then the
    values below and above it). The point, or ``None``."""
    lower, upper, infeasible = propagator(lower, upper)
    budget[0] -= 1
    if infeasible:
        return None
    # Variables in the violated rows' neighbourhood, most "decided" first.
    order = np.flatnonzero(free & ints)
    stack = [(lower, upper)]
    while stack:
        if budget[0] <= 0 or time.time() >= deadline:
            return None
        lo, up = stack.pop()
        open_vars = order[lo[order] < up[order]]
        if open_vars.size == 0:
            point = np.clip(hint, lo, up)
            point[ints] = lo[ints]
            return point
        j = open_vars[0]
        value = float(np.clip(np.round(hint[j]), lo[j], up[j]))
        children = []
        # Pushed in reverse: the hint's value is tried first.
        if value + 1 <= up[j]:
            children.append((j, value + 1, up[j]))
        if value - 1 >= lo[j]:
            children.append((j, lo[j], value - 1))
        children.append((j, value, value))
        for var, new_lower, new_upper in children:
            child_lo, child_up = lo.copy(), up.copy()
            child_lo[var], child_up[var] = new_lower, new_upper
            child_lo, child_up, infeasible = propagator(child_lo, child_up)
            budget[0] -= 1
            if not infeasible:
                stack.append((child_lo, child_up))
    return None


# ``device="auto"`` runs MIPs with at most this many nonzeros on the CPU:
# their node LPs are latency-bound on an accelerator. Measured (ms per LP
# epoch, CPU vs GPU): 98 nonzeros 0.7 vs 4.2, 8,937 nonzeros 4.2 vs 5.1,
# 100,933 nonzeros 73 vs 7.5; the crossover is near 10k.
_SMALL_MIP_NNZ = 10_000


def _search_device(device, nnz):
    """The JAX device for the search (``None``: JAX's default)."""
    if device is None:
        return None
    if device == "auto":
        if nnz > _SMALL_MIP_NNZ:
            return None
        device = "cpu"
    if device not in ("cpu", "gpu", "tpu"):
        raise ValueError(f"unknown device {device!r}")
    try:
        return jax.devices(device)[0]
    except RuntimeError as error:
        raise ValueError(f"no {device} device available") from error


# %%
# Branch and bound
class _Node:
    """An open node: its bound, depth, bound changes against the root (as
    ``(index, lower, upper)``), the parent's warm start and the branching
    that made it (``(var, down, distance, parent_value)``)."""

    __slots__ = ("bound", "depth", "changes", "warm", "branch")

    def __init__(self, bound, depth, changes, warm, branch):
        self.bound, self.depth, self.changes = bound, depth, changes
        self.warm, self.branch = warm, branch


_LP_RESERVED = ("initial_solution", "initial_opt_state", "max_epochs", "max_seconds")


class _BranchAndBound:
    def __init__(
        self,
        mip,
        *,
        mip_rel_gap,
        mip_abs_gap,
        max_nodes,
        deadline,
        branching,
        node_selection,
        integrality_tolerance,
        feasibility_tolerance,
        lp_tolerance,
        node_max_epochs,
        root_max_epochs,
        infeasibility_tolerance,
        heuristic_every,
        propagate,
        coefficient_tightening,
        cuts,
        cut_rounds,
        max_cuts_per_round,
        cut_min_progress,
        cut_round_epochs,
        gomory_cuts,
        aggregate_integers,
        tree_cuts,
        tree_cut_every,
        fix_and_propagate,
        feasibility_jump,
        jump_iterations,
        repair,
        repair_max_epochs,
        highs_incumbent_seconds,
        rins_seconds,
        rins_every,
        plunge_depth,
        batch_size,
        feasibility_pump,
        pump_iterations,
        pump_tolerance,
        heuristic_seconds,
        lp_options,
        verbose,
        log_every_seconds,
        start,
        device=None,
        rins_solver="auto",
        rins_nodes=200,
        rens_seconds=None,
        rens_nodes=500,
        dive_every=0,
        dive_depth=20,
        initial_cutoff=None,
    ):
        # Every setting, for sub-searches (``jaddle_rins``).
        self.settings = {
            key: value for key, value in locals().items() if key not in ("self", "mip")
        }
        if rins_solver not in ("auto", "jaddle", "highs"):
            raise ValueError(
                f"rins_solver must be 'auto', 'jaddle' or 'highs', not {rins_solver!r}"
            )
        if rins_solver == "auto":
            # On large MIPs a Jaddle sub-search is slow and weak: on
            # ns1830653 it took a third of the run and left a 71% gap,
            # against 52% with HiGHS.
            rins_solver = "jaddle" if mip.A.nnz <= _SMALL_MIP_NNZ else "highs"
        self.rins_solver, self.rins_nodes = rins_solver, rins_nodes
        self.rins_seen = set()  # neighbourhoods searched by jaddle_rins
        # A warm start for the root LP (a sub-search's, from its parent).
        self.root_warm = None
        self.rins_skipped = 0
        # A cutoff known before the search (a sub-MIP's parent's).
        self.initial_cutoff = np.inf if initial_cutoff is None else float(initial_cutoff)
        if branching not in ("pseudocost", "most_fractional"):
            raise ValueError(f"unknown branching rule {branching!r}")
        if node_selection not in ("hybrid", "best_first", "depth_first"):
            raise ValueError(f"unknown node selection {node_selection!r}")
        lp_options = dict(lp_options or {})
        reserved = [key for key in _LP_RESERVED if key in lp_options]
        if reserved:
            raise ValueError(f"lp_options cannot set {', '.join(reserved)}")

        self.original = mip  # the MIP as given: incumbents are checked on it
        self.device = _search_device(device, mip.A.nnz)
        # The MIP searched: the original, or it with tightened coefficients;
        # cuts extend ``self.mip``.
        self.base = mip
        self.use_coefficient_tightening = coefficient_tightening
        self.coefficients_tightened = 0
        self.cuts, self.cut_rounds = cuts, cut_rounds
        self.max_cuts_per_round, self.cut_min_progress = (
            max_cuts_per_round,
            cut_min_progress,
        )
        # Small MIPs get Gomory cuts and more cut rounds by default: their
        # crossovers take milliseconds.
        small = mip.A.nnz <= _SMALL_MIP_NNZ
        if gomory_cuts not in ("auto", True, False):
            raise ValueError(f"gomory_cuts must be 'auto', True or False, not {gomory_cuts!r}")
        self.gomory = small if gomory_cuts == "auto" else bool(gomory_cuts)
        if aggregate_integers not in ("auto", True, False):
            raise ValueError(
                f"aggregate_integers must be 'auto', True or False, not {aggregate_integers!r}"
            )
        self.aggregate_integers = (
            small if aggregate_integers == "auto" else bool(aggregate_integers)
        )
        many_rounds = self.gomory and small
        if cut_rounds is None:
            self.cut_rounds = 40 if many_rounds else 5
        if cut_min_progress is None:
            # Late rounds slow the root LP for little gain, so stop once three
            # rounds together gain less than 0.1%.
            self.cut_min_progress = 1e-3 if many_rounds else 1e-4
        # Global cuts in the tree need one sparsity pattern per compiled
        # solver; the batched solver captures it, so they are serial only.
        self.use_tree_cuts = bool(tree_cuts) and int(batch_size) == 1
        if tree_cuts and not self.use_tree_cuts and verbose:
            print("Tree cuts are off with batch_size > 1")
        self.tree_cut_every = tree_cut_every
        self.tree_cuts_added = 0
        self.cut_list = []
        self.root_bound_before_cuts = None
        self.use_fix_and_propagate = fix_and_propagate
        self.use_jump, self.jump_iterations = feasibility_jump, jump_iterations
        self.use_repair = repair
        self.repair_max_epochs = repair_max_epochs
        self.highs_seconds = highs_incumbent_seconds
        # RENS defaults on for small MIPs (``None``: auto).
        if rens_seconds is None:
            rens_seconds = 5.0 if small else 0.0
        self.rins_seconds, self.rins_every = rins_seconds, rins_every
        self.rins_pending = False  # a new incumbent asks for a RINS call
        self.rens_seconds, self.rens_nodes = rens_seconds, rens_nodes
        self.dive_every, self.dive_depth = dive_every, dive_depth
        self.dives = 0
        self.use_objective_propagation = True
        self.rins_calls = 0
        self.plunge_depth = plunge_depth
        self.plunge_node = None  # the child the next step plunges into
        self.plunge_left = 0
        if int(batch_size) < 1:
            raise ValueError("batch_size must be at least 1")
        self.batch_size = int(batch_size)
        self.batches = 0
        self._batched = None  # (mip, jitted vmapped node solver)
        self.pump_last = None
        self.use_pump = feasibility_pump
        self.pump_iterations, self.pump_tolerance = pump_iterations, pump_tolerance
        self.heuristic_budget = heuristic_seconds
        self.heuristic_seconds = 0.0
        self.rel_gap, self.abs_gap = mip_rel_gap, mip_abs_gap
        self.max_nodes, self.deadline = max_nodes, deadline
        self.branching, self.node_selection = branching, node_selection
        self.int_tol = integrality_tolerance
        self.feas_tol = feasibility_tolerance
        self.node_max_epochs, self.root_max_epochs = node_max_epochs, root_max_epochs
        self.cut_round_epochs = cut_round_epochs
        self.infeas_tol = infeasibility_tolerance
        self.heuristic_every = heuristic_every
        self.verbose, self.log_every = verbose, log_every_seconds
        self.start = start
        self.lp_kwargs = dict(
            primal_feasibility_tolerance=lp_tolerance,
            dual_feasibility_tolerance=lp_tolerance,
            dual_gap_tolerance=lp_tolerance,
        )
        self.lp_kwargs.update(lp_options)
        self.reduced_cost_tol = lp_tolerance * (1.0 + np.linalg.norm(mip.c))

        self.is_int = mip.integrality
        self.has_continuous = not self.is_int.all()
        # Rows of the MIP as given with a continuous variable, which an LP
        # over the continuous variables could fix.
        self.row_has_continuous = (
            np.asarray(abs(mip.A) @ (~self.is_int).astype(np.float64)) > 0
        )
        self.failed_roundings = set()  # integer parts whose repair failed
        self.lp_repairs = 0
        self.lp_repairs_skipped = 0
        self.propagate = propagate
        self.set_mip(mip)

        self.incumbent = None  # (min-form value, x)
        self.closed_floor = np.inf  # least bound of the nodes closed so far
        # Least bound of the nodes closed without proof: an LP that did not
        # converge, or an integral LP point that failed the feasibility check.
        self.unresolved_floor = np.inf
        self.bound_rigorous = True
        self.nodes = 0
        self.lp_epochs = 0
        self.infeasible_nodes = 0
        self.unresolved_nodes = 0
        self.history = []
        self.open = []
        self.counter = itertools.count()
        self.pc_sum = np.zeros((2, mip.num_variables))  # [down, up]
        self.pc_count = np.zeros((2, mip.num_variables))
        self.last_log = -np.inf
        # The bound of the node being evaluated (it is in no list meanwhile).
        self.active_bound = np.inf
        # Every objective value is a multiple of this (None: no such step).
        self.objective_step = objective_step(mip.c, mip.integrality)
        # The root's Lagrangian terms, for reduced-cost fixing of the root box
        # when the incumbent improves, and the incumbent last used for it.
        self.root_terms = None
        self.root_fixed_at = None
        self.reduced_cost_tightenings = 0
        self.children_pruned = 0

    def set_mip(self, mip):
        """Search ``mip`` (the MIP as given, or it with cuts)."""
        self.mip = mip
        self.A_T = mip.A.T.tocsr()
        self.m_ineq = mip.A.shape[0] - mip.n_eq
        self.values = mip.lp.values()
        self.dtype = mip.lp.c.dtype
        # The heuristics propagate even when the search does not.
        self.row_propagator = BoundPropagator(
            mip.A, mip.row_lower, mip.row_upper, self.is_int
        )
        self.propagator = self.row_propagator if self.propagate else None
        # The rows plus the objective row c·x <= cutoff, for heuristics that
        # should only find improving points (``heuristic_propagator``).
        self.objective_propagator = BoundPropagator(
            sp.vstack([mip.A, sp.csr_matrix(mip.c[None, :])], format="csr"),
            np.append(mip.row_lower, -np.inf),
            np.append(mip.row_upper, np.inf),
            self.is_int,
        )

    def heuristic_propagator(self):
        """The propagator for heuristics: with the objective row at the
        current cutoff once there is one, so roundings that cannot improve
        on the incumbent are rejected; else the rows alone."""
        cutoff = self.cutoff()
        if not (self.use_objective_propagation and np.isfinite(cutoff)):
            return self.row_propagator
        self.objective_propagator.row_upper[-1] = cutoff
        return self.objective_propagator

    # -- bookkeeping ---------------------------------------------------------
    def seconds_left(self):
        """Seconds to the deadline (``None``: no deadline), at least 1ms:
        the deadline can pass between a time check and an LP solve, and
        ``jl.solve`` needs a positive limit (it then stops at once)."""
        if self.deadline is None:
            return None
        return max(self.deadline - time.time(), 1e-3)

    def out_of_time(self):
        return self.deadline is not None and time.time() >= self.deadline

    def cutoff(self):
        if self.incumbent is None:
            return self.initial_cutoff
        value = self.incumbent[0] + self.mip.offset
        cutoff = self.incumbent[0] - max(self.abs_gap, self.rel_gap * abs(value))
        if self.objective_step is not None:
            # A better solution is at least one step better.
            cutoff = min(cutoff, self.incumbent[0] - 0.5 * self.objective_step)
        return min(cutoff, self.initial_cutoff)

    def round_bound(self, bound):
        """``bound`` rounded up to a multiple of the objective step, less a
        little against rounding error in the bound itself."""
        step = self.objective_step
        if step is None or not np.isfinite(bound):
            return bound
        q = bound / step
        return step * math.ceil(q - 1e-6 - 1e-9 * abs(q))

    def cutoff_bound(self):
        """The bound of a region shown to hold no solution below the cutoff."""
        return self.round_bound(self.cutoff())

    def global_bound(self):
        bound = min((node.bound for node in self.open_nodes()), default=np.inf)
        bound = min(bound, self.closed_floor, self.active_bound)
        if self.incumbent is not None:
            bound = min(bound, self.incumbent[0])
        return bound

    def close(self, bound, rigorous=True):
        self.closed_floor = min(self.closed_floor, bound)
        self.bound_rigorous &= rigorous

    def gap(self, bound):
        if self.incumbent is None:
            return np.inf
        value = self.incumbent[0] + self.mip.offset
        return max(self.incumbent[0] - bound, 0.0) / max(abs(value), 1e-10)

    def model_value(self, value):
        """A min-form value (without offset) in the model's own sense."""
        return self.mip.sign * (value + self.mip.offset)

    # -- node keys -----------------------------------------------------------
    def diving(self):
        return self.node_selection == "depth_first" or (
            self.node_selection == "hybrid" and self.incumbent is None
        )

    def key(self, node, order):
        if self.diving():
            return (-order,)
        return (node.bound, -node.depth, order)

    def push(self, node):
        order = next(self.counter)
        heapq.heappush(self.open, (self.key(node, order), order, node))

    def rekey(self):
        self.open = [(self.key(n, o), o, n) for _, o, n in self.open]
        heapq.heapify(self.open)

    def open_nodes(self):
        nodes = [entry[2] for entry in self.open]
        if self.plunge_node is not None:
            nodes.append(self.plunge_node)
        return iter(nodes)

    def plunging(self):
        """Best-bound search with plunges: after each best-bound node, up to
        ``plunge_depth`` steps follow its preferred child, so the search
        keeps reaching leaves (and incumbents) while it raises the bound."""
        return (
            self.node_selection == "hybrid"
            and self.incumbent is not None
            and self.plunge_depth > 0
        )

    # -- LP solves -----------------------------------------------------------
    def node_lp(self, lower, upper):
        # Integer tightenings go into the LP; continuous ones only into the
        # Lagrangian bound, so rounding error in them cannot cut off the LP.
        lp_lower = np.where(self.is_int, lower, self.mip.lower)
        lp_upper = np.where(self.is_int, upper, self.mip.upper)
        return self.mip.lp.with_values(
            self.values._replace(
                lower_bounds=jnp.asarray(lp_lower, self.dtype),
                upper_bounds=jnp.asarray(lp_upper, self.dtype),
            )
        )

    def solve_lp(self, lp, warm, max_epochs, **overrides):
        kwargs = dict(self.lp_kwargs, **overrides)
        if warm is not None:
            kwargs.update(initial_solution=warm[0], initial_opt_state=warm[1])
        result = jl.solve(
            lp, max_epochs=max_epochs, max_seconds=self.seconds_left(), **kwargs
        )
        self.lp_epochs += result["epochs"]
        return result

    def detect(self, lp, result, max_epochs):
        """``detect_infeasibility`` warm-started from a solve's last point."""
        out = jl.detect_infeasibility(
            lp,
            tol=self.infeas_tol,
            max_epochs=max_epochs,
            max_seconds=self.seconds_left(),
            initial_solution=None if result is None else result["solution"],
            initial_opt_state=None if result is None else result["opt_state"],
            **self.lp_kwargs,
        )
        self.lp_epochs += out["epochs"]
        return out

    def duals(self, solution):
        return np.concatenate(
            [
                np.asarray(solution.dual_eq, np.float64)[: self.mip.n_eq],
                np.asarray(solution.dual_ineq, np.float64)[: self.m_ineq],
            ]
        )

    def dual_terms(self, solution):
        """``lagrangian_terms`` of a node's duals."""
        return lagrangian_terms(
            self.mip.c,
            self.mip.A,
            self.mip.b,
            self.mip.n_eq,
            self.duals(solution),
            A_T=self.A_T,
        )

    def bound_over(self, terms, lower, upper):
        """The Lagrangian bound of ``terms`` over a propagated box, widened a
        little on continuous variables against rounding error."""
        cont = ~self.is_int
        lower = np.where(cont, lower - 1e-7 * (1.0 + np.abs(lower)), lower)
        upper = np.where(cont, upper + 1e-7 * (1.0 + np.abs(upper)), upper)
        return box_bound(
            *terms, lower, upper, reduced_cost_tolerance=self.reduced_cost_tol
        )

    def lp_bound(self, solution, lower, upper):
        """The Lagrangian bound of a node's duals over its propagated box."""
        return self.bound_over(self.dual_terms(solution), lower, upper)

    def reduced_cost_fix(self, terms, lower, upper):
        """``reduced_cost_tightening`` of a box against the cutoff, then
        propagation. Returns ``(lower, upper, infeasible)``, where
        ``infeasible`` means the box holds no solution below the cutoff."""
        if not np.isfinite(self.cutoff()):
            return lower, upper, False
        bound, rigorous = self.bound_over(terms, lower, upper)
        if not rigorous:
            return lower, upper, False
        new_lower, new_upper, changed = reduced_cost_tightening(
            terms[1], bound, self.cutoff(), lower, upper, self.is_int
        )
        if not changed:
            return lower, upper, False
        self.reduced_cost_tightenings += int(
            np.count_nonzero((new_lower != lower) | (new_upper != upper))
        )
        if np.any(new_lower > new_upper):
            return new_lower, new_upper, True
        if self.propagator is not None:
            return self.propagator(new_lower, new_upper)
        return new_lower, new_upper, False

    def tighten_root(self):
        """Reduced-cost fixing of the root box with the root's duals, once
        per incumbent. Returns False when it shows that no solution below
        the cutoff is left."""
        if (
            self.root_terms is None
            or self.incumbent is None
            or self.incumbent[0] == self.root_fixed_at
        ):
            return True
        self.root_fixed_at = self.incumbent[0]
        lower, upper, infeasible = self.reduced_cost_fix(
            self.root_terms, self.root_lower, self.root_upper
        )
        if infeasible:
            return False
        self.root_lower, self.root_upper = lower, upper
        return True

    def close_all(self):
        """Close every open node: none holds a solution below the cutoff."""
        self.close(self.cutoff_bound())
        self.open = []
        self.plunge_node = None

    # -- incumbents ----------------------------------------------------------
    def worth_repairing(self, candidate, fixed_lower, fixed_upper):
        """Whether a rounded point that fails the check is worth an LP over
        its continuous variables. Not when the same integers already failed,
        when no violated row has a continuous variable to move, or when
        propagation shows the box with the integers fixed is infeasible."""
        key = candidate[self.is_int].tobytes()
        if key in self.failed_roundings:
            return False
        rows = _violated_rows(self.original, candidate, self.feas_tol)
        worth = bool(self.row_has_continuous[rows].any())
        if worth:
            _, _, infeasible = self.row_propagator(fixed_lower, fixed_upper)
            worth = not infeasible
        if not worth:
            self.failed_roundings.add(key)
        return worth

    def try_incumbent(self, x, source, warm):
        """Round ``x``'s integers and make it the incumbent if it is feasible
        and better. With continuous variables, a rounded point that is not
        feasible is repaired by fixing the integers and re-solving the LP for
        the rest. Returns whether a feasible point was found."""
        mip = self.mip
        candidate = np.clip(np.asarray(x, np.float64), mip.lower, mip.upper)
        candidate[self.is_int] = np.round(candidate[self.is_int])
        if not self.original.is_feasible(candidate, self.feas_tol):
            if not self.has_continuous or self.out_of_time():
                return False
            fixed_lower = np.where(self.is_int, candidate, mip.lower)
            fixed_upper = np.where(self.is_int, candidate, mip.upper)
            if np.any(fixed_lower > fixed_upper):
                return False
            if not self.worth_repairing(candidate, fixed_lower, fixed_upper):
                self.lp_repairs_skipped += 1
                return False
            self.lp_repairs += 1
            lp = self.mip.lp.with_values(
                self.values._replace(
                    lower_bounds=jnp.asarray(fixed_lower, self.dtype),
                    upper_bounds=jnp.asarray(fixed_upper, self.dtype),
                )
            )
            tight = 0.1 * self.feas_tol
            result = self.solve_lp(
                lp,
                warm,
                self.repair_max_epochs,
                primal_feasibility_tolerance=tight,
                termination_norm="inf",
            )
            candidate = np.clip(
                np.asarray(result["solution"].primal, np.float64),
                fixed_lower,
                fixed_upper,
            )
            candidate[self.is_int] = fixed_lower[self.is_int]
            if not self.original.is_feasible(candidate, self.feas_tol):
                self.failed_roundings.add(candidate[self.is_int].tobytes())
                return False
        value = float(mip.c @ candidate)
        if value >= self.initial_cutoff:
            return True  # feasible, but no better than the parent search's cutoff
        if self.incumbent is not None and value >= self.incumbent[0] - 1e-12 * (
            1.0 + abs(self.incumbent[0])
        ):
            return True
        switch = self.incumbent is None
        self.incumbent = (value, candidate)
        if source != "N":
            self.rins_pending = True
        bound = self.global_bound()
        self.history.append(
            dict(
                seconds=time.time() - self.start,
                nodes=self.nodes,
                objective=self.model_value(value),
                bound=self.model_value(bound),
                source=source,
            )
        )
        self.log(source, bound, depth=None, force=True)
        if switch and self.node_selection == "hybrid":
            self.rekey()
        return True

    # -- branching -----------------------------------------------------------
    def choose_variable(self, x, fractional):
        candidates = np.flatnonzero(fractional)
        f = x[candidates] - np.floor(x[candidates])
        if self.branching == "most_fractional" or not self.pc_count.any():
            return candidates[np.argmax(np.minimum(f, 1.0 - f))]
        known = self.pc_count > 0
        mean = np.array(
            [
                self.pc_sum[d][known[d]].sum() / self.pc_count[d][known[d]].sum()
                if known[d].any()
                else 1.0
                for d in range(2)
            ]
        )
        pc = np.where(
            known[:, candidates],
            self.pc_sum[:, candidates] / np.maximum(self.pc_count[:, candidates], 1),
            mean[:, None],
        )
        score = np.maximum(f * pc[0], 1e-6) * np.maximum((1.0 - f) * pc[1], 1e-6)
        return candidates[np.argmax(score)]

    def update_pseudocost(self, node, value):
        var, down, distance, parent_value = node.branch
        if parent_value is None or distance <= 0:
            return
        gain = max(value - parent_value, 0.0) / distance
        self.pc_sum[0 if down else 1, var] += gain
        self.pc_count[0 if down else 1, var] += 1

    def expand(self, changes):
        """A node's box: its changes, within the root box (which reduced-cost
        fixing may have tightened since the node was made). It can be
        empty."""
        lower, upper = self.root_lower.copy(), self.root_upper.copy()
        if changes is not None:
            idx, lo, up = changes
            lower[idx] = np.maximum(lo, lower[idx])
            upper[idx] = np.minimum(up, upper[idx])
        return lower, upper

    def compress(self, lower, upper):
        idx = np.flatnonzero((lower != self.root_lower) | (upper != self.root_upper))
        return idx, lower[idx], upper[idx]

    def branch(self, node, lower, upper, x, var, bound, value, warm, terms):
        """Split on ``var``. Each child is bounded by the parent's duals
        (``terms``) over its own propagated box, and closed at once when
        that reaches the cutoff."""
        xv = x[var]
        children = []
        for down in (True, False):
            lo, up = lower.copy(), upper.copy()
            if down:
                up[var] = min(up[var], math.floor(xv))
            else:
                lo[var] = max(lo[var], math.ceil(xv))
            if lo[var] > up[var]:
                continue
            if self.propagator is not None:
                lo, up, infeasible = self.propagator(lo, up)
                if infeasible:
                    self.infeasible_nodes += 1
                    continue
            child_bound, rigorous = self.bound_over(terms, lo, up)
            child_bound = max(self.round_bound(child_bound), bound)
            if child_bound >= self.cutoff():
                self.children_pruned += 1
                self.close(child_bound, rigorous)
                continue
            distance = xv - math.floor(xv) if down else math.ceil(xv) - xv
            children.append(
                _Node(
                    child_bound,
                    node.depth + 1,
                    self.compress(lo, up),
                    warm,
                    (var, down, distance, value),
                )
            )
        # In a dive the child in x's rounding direction goes first (pushed last).
        prefer_up = xv - math.floor(xv) >= 0.5
        children.sort(key=lambda child: child.branch[1] == prefer_up)
        if (
            children
            and self.plunging()
            and self.plunge_left > 0
            and self.plunge_node is None
        ):
            # Plunge: the next step continues into the preferred child.
            self.plunge_node = children.pop()
            self.plunge_left -= 1
        for child in children:
            self.push(child)

    # -- the search ------------------------------------------------------------
    def evaluate(self, node, lower, upper, lp, result):
        """Bound, prune, accept or branch on a solved node."""
        self.active_bound = node.bound
        try:
            self._evaluate(node, lower, upper, lp, result)
        finally:
            self.active_bound = np.inf

    def _evaluate(self, node, lower, upper, lp, result):
        terms = self.dual_terms(result["solution"])
        raw, rigorous = self.bound_over(terms, lower, upper)
        bound = max(self.round_bound(raw), node.bound)
        self.active_bound = bound
        if not result["converged"] and bound < self.cutoff() and not self.out_of_time():
            # An LP that will not converge is often infeasible.
            detection = self.detect(lp, result, self.node_max_epochs)
            if detection["status"] == "primal_infeasible":
                self.infeasible_nodes += 1
                return
            result = detection["result"]
            new_terms = self.dual_terms(result["solution"])
            new_raw, new_rigorous = self.bound_over(new_terms, lower, upper)
            if new_raw > raw:
                terms, raw, rigorous = new_terms, new_raw, new_rigorous
                bound = max(self.round_bound(raw), bound)
                self.active_bound = bound
        if node.depth == 0 and node.branch is None:
            self.root_terms = terms
        converged = result["converged"]
        x = np.asarray(result["solution"].primal, np.float64)
        value = float(self.mip.c @ x) if converged else None
        if node.branch is not None and converged:
            self.update_pseudocost(node, value)
        if bound >= self.cutoff():
            self.close(bound, rigorous)
            return
        warm = (result["solution"], result["opt_state"])

        xi = x[self.is_int]
        fractional = np.zeros(x.shape[0], dtype=bool)
        fractional[self.is_int] = np.abs(xi - np.round(xi)) > self.int_tol
        if not fractional.any():
            feasible = self.try_incumbent(x, "T", warm)
            if not (converged and feasible):
                self.unresolved_nodes += 1
                self.unresolved_floor = min(self.unresolved_floor, bound)
            self.close(bound, rigorous)
            return
        self.bound_rigorous &= rigorous
        if self.heuristic_every and (self.nodes - 1) % self.heuristic_every == 0:
            if not self.try_incumbent(x, "H", warm) and self.use_fix_and_propagate:
                started = time.time()
                candidate = fix_and_propagate(
                    x, lower, upper, self.is_int, self.heuristic_propagator()
                )
                if candidate is not None:
                    self.try_incumbent(candidate, "P", warm)
                self.heuristic_seconds += time.time() - started
            if bound >= self.cutoff():
                self.close(bound, rigorous)
                return
        if (
            self.rins_seconds
            and self.incumbent is not None
            and (self.nodes % self.rins_every == 0 or self.rins_pending)
            and not self.out_of_time()
        ):
            self.rins_pending = False
            self.rins(x, warm)
            if bound >= self.cutoff():
                self.close(bound, rigorous)
                return
        if self.dive_every and self.nodes % self.dive_every == 0 and not self.out_of_time():
            self.dive(x, lower, upper, warm)
            if bound >= self.cutoff():
                self.close(bound, rigorous)
                return
        node_lower = lower
        lower, upper, infeasible = self.reduced_cost_fix(terms, lower, upper)
        if infeasible:
            self.close(max(bound, self.cutoff_bound()), rigorous)
            return
        # Branch on a variable whose LP value is inside the (fixed) box, so
        # both children are new.
        branchable = fractional & (x > lower) & (x < upper)
        if branchable.any():
            var = self.choose_variable(x, branchable)
        elif lower is not node_lower:  # the fixing tightened the box
            # Reduced-cost fixing moved every fractional variable's bounds
            # past the LP point: solve the node again in its smaller box.
            self.push(_Node(bound, node.depth, self.compress(lower, upper), warm, None))
            return
        else:
            var = self.choose_variable(x, fractional)
        if (
            self.use_tree_cuts
            and self.tree_cut_every
            and self.nodes % self.tree_cut_every == 0
        ):
            warm = self.tree_cut_round(x, warm)
        self.branch(node, lower, upper, x, var, bound, value, warm, terms)

    def log(self, source, bound, depth, force=False):
        """A progress line (``bound=None``: the global bound)."""
        if not self.verbose:
            return
        now = time.time()
        if not force and now - self.last_log < self.log_every:
            return
        self.last_log = now
        if bound is None:
            bound = self.global_bound()
        incumbent = (
            f"{self.model_value(self.incumbent[0]):13.6e}"
            if self.incumbent is not None
            else f"{'-':>13}"
        )
        gap = self.gap(bound)
        gap_text = f"{100 * gap:8.2f}%" if np.isfinite(gap) else f"{'-':>9}"
        print(
            f"{source:>1} {self.nodes:8d} {len(self.open):8d} "
            f"{'' if depth is None else depth:>6} {self.model_value(bound):13.6e} "
            f"{incumbent} {gap_text} {now - self.start:8.1f}s"
        )

    def result(self, status):
        bound = self.global_bound()
        if status == "optimal":
            # The search ended: proven unless a node closed without proof
            # could still hold a better solution.
            if self.incumbent is None:
                if self.unresolved_nodes:
                    status = "unresolved"
                else:
                    status = "infeasible"
            elif self.unresolved_floor < self.cutoff():
                status = "unresolved"
        x = None if self.incumbent is None else self.incumbent[1]
        objective = None if x is None else self.model_value(self.incumbent[0])
        if self.verbose:
            print("----------------------------------------------")
            print(f"Status: {status}")
            print(f"Nodes: {self.nodes}, LP epochs: {self.lp_epochs}")
            if self.cut_list or self.root_bound_before_cuts is not None:
                families = [cut["family"] for cut in self.cut_list]
                line = (
                    f"Cuts: {len(families)} in the LP ({families.count('cover')} "
                    f"cover, {families.count('mir')} MIR, "
                    f"{families.count('gomory')} Gomory)"
                )
                if self.use_tree_cuts:
                    line += f", {self.tree_cuts_added} added in the tree"
                print(line)
            if objective is not None:
                print(f"Objective: {objective:.10g}")
            print(f"Bound: {self.model_value(bound):.10g}")
            print(f"Time: {time.time() - self.start:.2f} seconds")
        return {
            "x": x,
            "objective": objective,
            "bound": self.model_value(bound),
            "gap": self.gap(bound),
            "status": status,
            "converged": status == "optimal",
            "bound_rigorous": self.bound_rigorous,
            "nodes": self.nodes,
            "infeasible_nodes": self.infeasible_nodes,
            "unresolved_nodes": self.unresolved_nodes,
            "lp_epochs": self.lp_epochs,
            "seconds": time.time() - self.start,
            "history": self.history,
            "cuts": {
                family: sum(cut["family"] == family for cut in self.cut_list)
                for family in ("gomory", "cover", "mir")
            },
            "root_bound_before_cuts": None
            if self.root_bound_before_cuts is None
            else self.model_value(self.root_bound_before_cuts),
            "heuristic_seconds": self.heuristic_seconds,
            "batches": self.batches,
            "objective_step": self.objective_step,
            "reduced_cost_tightenings": self.reduced_cost_tightenings,
            "children_pruned": self.children_pruned,
            "lp_repairs": self.lp_repairs,
            "lp_repairs_skipped": self.lp_repairs_skipped,
            "coefficients_tightened": self.coefficients_tightened,
            "tree_cuts_added": self.tree_cuts_added,
            "rins_calls": self.rins_calls,
        }

    def root_vertex(self, lower, upper, solution):
        """Cross the root LP's point over to an optimal vertex with HiGHS
        (``jaddle_linear.crossover``). Returns ``(x, basis)``, or
        ``(None, None)`` when HiGHS does not prove the vertex optimal within
        its time limit (5% of ``max_seconds``, or 60 s without one)."""
        limit = 60.0 if self.deadline is None else max(
            1.0, 0.05 * (self.deadline - self.start)
        )
        crossed = jl.crossover(
            self.node_lp(lower, upper), solution, highs_options={"time_limit": limit}
        )
        if not crossed["converged"] or crossed["basis"] is None:
            return None, None
        return np.asarray(crossed["solution"].primal, np.float64), crossed["basis"]

    def cut_loop(self, lower, upper, result):
        """Rounds of root cuts: separate at the LP point, append the cuts
        (retiring ones slack with a zero dual for two rounds), re-solve warm.
        Stops when a round finds no cut or the bound gains less than
        ``cut_min_progress`` (relative). Returns the last root LP result, or
        a status string when a re-solve settles the search."""
        base = self.base
        n = base.num_variables
        base_m = base.A.shape[0] - base.n_eq
        # Separate from the rows as searched and, when coefficient tightening
        # changed them, from the rows as given too: tightening lifts the LP
        # bound, but the original knapsack rows can give stronger covers.
        separators = [self.separator(base, lower, upper)]
        if self.original is not base:
            separators.append(self.separator(self.original, lower, upper))
        bound, _ = self.lp_bound(result["solution"], lower, upper)
        self.root_bound_before_cuts = bound
        history = [bound]  # the best bound after each round
        capacity = None
        for round_number in range(1, self.cut_rounds + 1):
            if self.out_of_time():
                break
            solution = result["solution"]
            x = np.asarray(solution.primal, np.float64)
            candidates = []
            if self.gomory:
                # Cross over to a vertex: its basis gives the Gomory cuts, and
                # the other families separate better from a vertex too.
                vertex, basis = self.root_vertex(lower, upper, solution)
                if vertex is not None:
                    x = vertex
                    candidates += gomory_cuts(
                        self.mip.A,
                        self.mip.b,
                        self.mip.n_eq,
                        np.where(self.is_int, lower, base.lower),
                        np.where(self.is_int, upper, base.upper),
                        self.is_int,
                        basis,
                        x,
                        max_cuts=2 * self.max_cuts_per_round,
                    )
            for separator in separators:
                candidates += separator(x)
            found = select_cuts(candidates, x, n, max_cuts=self.max_cuts_per_round)
            if not found:
                break
            dual_ineq = np.asarray(solution.dual_ineq)
            kept = [
                (cut, dual_ineq[base_m + i])
                for i, cut in enumerate(self.cut_list)
                if cut["age"] < 2
            ]
            self.cut_list = [cut for cut, _ in kept] + [
                dict(family=family, idx=idx, coef=coef, rhs=rhs, age=0)
                for family, idx, coef, rhs in found
            ]
            # Room to grow: the first round reserves 4x its size, so later
            # rounds (and the tree) usually keep one shape; a round that
            # outgrows it doubles it.
            need_rows = len(self.cut_list)
            need_nnz = sum(cut["idx"].size for cut in self.cut_list)
            if capacity is None:
                capacity = (max(16, 4 * need_rows), max(256, 4 * need_nnz))
            elif need_rows > capacity[0] or need_nnz > capacity[1]:
                capacity = (2 * max(need_rows, capacity[0]), 2 * max(need_nnz, capacity[1]))
            A_cuts, b_cuts = _cut_block(self.cut_list, n, *capacity)
            self.set_mip(base.with_rows(A_cuts, b_cuts))

            # Warm start: the old duals, zero for the new rows and padding.
            duals = np.zeros(base_m + A_cuts.shape[0], dtype=dual_ineq.dtype)
            duals[:base_m] = dual_ineq[:base_m]
            duals[base_m : base_m + len(kept)] = [dual for _, dual in kept]
            warm = solution._replace(dual_ineq=jnp.asarray(duals))
            lp = self.node_lp(lower, upper)
            detection = self.detect(
                lp, {"solution": warm, "opt_state": result["opt_state"]},
                self.round_epochs(),
            )
            if detection["status"] == "primal_infeasible":
                return "infeasible"
            result = detection["result"]
            if result["stop_reason"] in ("interrupted", "time_limit"):
                return result

            # Age the cuts: slack, with a zero dual.
            y = np.asarray(result["solution"].dual_ineq, np.float64)[base_m:]
            x = np.asarray(result["solution"].primal, np.float64)
            y_scale = 1e-9 * (1.0 + np.abs(y).max(initial=0.0))
            for i, cut in enumerate(self.cut_list):
                slack = cut["rhs"] - cut["coef"] @ x[cut["idx"]]
                idle = slack > 1e-6 * (1.0 + abs(cut["rhs"])) and y[i] <= y_scale
                cut["age"] = cut["age"] + 1 if idle else 0

            new_bound, _ = self.lp_bound(result["solution"], lower, upper)
            if self.verbose:
                families = [cut["family"] for cut in self.cut_list]
                print(
                    f"Cut round {round_number}: {len(found)} new, "
                    f"{families.count('gomory')} Gomory + "
                    f"{families.count('cover')} cover + {families.count('mir')} "
                    f"MIR in the LP, bound {self.model_value(new_bound):.6e}, "
                    f"{time.time() - self.start:.1f}s"
                )
            bound = max(bound, new_bound)
            history.append(bound)
            # Progress comes in bursts, so judge it over the last three
            # rounds rather than the last one.
            gain = history[-1] - history[max(0, len(history) - 4)]
            scale = max(1.0, abs(bound + self.mip.offset))
            if not np.isfinite(gain) or gain <= self.cut_min_progress * scale:
                break
        result = self.compact_cuts(lower, upper, result)
        if isinstance(result, dict) and not result["converged"] and not self.out_of_time():
            # The rounds stopped early; the tree starts from a converged root.
            detection = self.detect(self.node_lp(lower, upper), result, self.root_max_epochs)
            if detection["status"] == "primal_infeasible":
                return "infeasible"
            result = detection["result"]
        return result

    def round_epochs(self):
        """The epoch budget of a cut round's re-solve: ``cut_round_epochs``
        (separation needs only a reasonable point, and the Lagrangian bound
        holds at any duals), within ``root_max_epochs``."""
        budgets = [e for e in (self.cut_round_epochs, self.root_max_epochs) if e]
        return min(budgets) if budgets else None

    def compact_cuts(self, lower, upper, result):
        """Rebuild the cut block for the tree: keep the cuts that were tight
        or had a dual at the last solve, and size the block to them without
        the room to grow the rounds reserved. Every node LP then pays only
        for the cuts it uses. When that shrinks the LP's nonzeros by a
        quarter or more, the root is re-solved warm on the compact LP (its
        point must match the rows the children start from)."""
        if not self.cut_list:
            return result
        base = self.base
        base_m = base.A.shape[0] - base.n_eq
        kept = [(i, cut) for i, cut in enumerate(self.cut_list) if cut["age"] == 0]
        A_cuts, b_cuts = _cut_block([cut for _, cut in kept], base.num_variables)
        if base.A.nnz + A_cuts.nnz > 0.75 * self.mip.lp.A.nse:
            return result
        solution = result["solution"]
        dual_ineq = np.asarray(solution.dual_ineq)
        duals = np.zeros(base_m + A_cuts.shape[0], dtype=dual_ineq.dtype)
        duals[:base_m] = dual_ineq[:base_m]
        duals[base_m : base_m + len(kept)] = [dual_ineq[base_m + i] for i, _ in kept]
        self.cut_list = [cut for _, cut in kept]
        self.set_mip(base.with_rows(A_cuts, b_cuts))
        warm = solution._replace(dual_ineq=jnp.asarray(duals))
        detection = self.detect(
            self.node_lp(lower, upper),
            {"solution": warm, "opt_state": result["opt_state"]},
            self.root_max_epochs,
        )
        if detection["status"] == "primal_infeasible":
            return "infeasible"
        return detection["result"]

    def separator(self, mip, lower, upper):
        """A ``CutSeparator`` with this search's aggregation settings."""
        return CutSeparator(
            mip,
            lower,
            upper,
            aggregate_integers=self.aggregate_integers,
            max_aggregation=4 if self.aggregate_integers else _CUT_MAX_AGGREGATION,
        )

    def reserve_tree_cuts(self, result):
        """Rebuild the cut block at a fixed capacity with room for tree cuts,
        so every later block (``tree_cut_round``) has the same shape and the
        node LPs keep one compiled solver. Returns the root result with its
        duals padded to the new rows (padding rows are ``0 <= 0``)."""
        base = self.base
        n = base.num_variables
        base_m = base.A.shape[0] - base.n_eq
        root_rows = len(self.cut_list)
        root_nnz = sum(cut["idx"].size for cut in self.cut_list)
        self.tree_nnz_cap = _next_power_of_two(
            max(256, 2 * root_nnz, _TREE_CUT_MIN_ROWS * min(n, 16))
        )
        # Enough rows for the cuts plus the zero rows that pad the nonzeros.
        self.tree_padding_rows = -(-self.tree_nnz_cap // n)
        self.tree_rows_cap = _next_power_of_two(
            max(_TREE_CUT_MIN_ROWS, 2 * root_rows) + self.tree_padding_rows
        )
        dual_ineq = np.asarray(result["solution"].dual_ineq)
        self.set_mip(base.with_rows(*self.tree_block()))
        if self.verbose:
            print(
                f"Tree cuts: room for {self.tree_rows_cap - self.tree_padding_rows} "
                f"cuts ({self.tree_nnz_cap} nonzeros), {root_rows} from the root, "
                f"a round every {self.tree_cut_every} nodes"
            )
        duals = np.zeros(base_m + self.tree_rows_cap, dtype=dual_ineq.dtype)
        keep = base_m + root_rows
        duals[:keep] = dual_ineq[:keep]
        solution = result["solution"]._replace(dual_ineq=jnp.asarray(duals))
        return dict(result, solution=solution)

    def tree_block(self):
        """The cut pool as a block of exactly the reserved capacity."""
        n = self.base.num_variables
        A_cuts, b_cuts = _cut_block(
            self.cut_list, n, self.tree_rows_cap, self.tree_nnz_cap
        )
        assert A_cuts.shape[0] == self.tree_rows_cap
        assert A_cuts.nnz == self.tree_nnz_cap
        return A_cuts, b_cuts

    def tree_cut_round(self, x, warm):
        """Separate globally valid cuts at a node's LP point ``x`` (over the
        root box) and put the violated ones into the pool: in free capacity,
        or in place of cuts that have been slack with a zero dual for two
        rounds. The current node is not re-solved; its children solve
        against the new cuts. Returns ``warm`` with the duals of replaced
        rows zeroed."""
        base = self.base
        n = base.num_variables
        base_m = base.A.shape[0] - base.n_eq
        solution = warm[0]
        y = np.asarray(solution.dual_ineq, np.float64)[base_m:]
        y_scale = 1e-9 * (1.0 + np.abs(y).max(initial=0.0))
        for i, cut in enumerate(self.cut_list):
            slack = cut["rhs"] - cut["coef"] @ x[cut["idx"]]
            idle = slack > 1e-6 * (1.0 + abs(cut["rhs"])) and y[i] <= y_scale
            cut["age"] = cut["age"] + 1 if idle else 0

        separators = [self.separator(base, self.root_lower, self.root_upper)]
        if self.original is not base:
            separators.append(
                self.separator(self.original, self.root_lower, self.root_upper)
            )
        candidates = [cut for separator in separators for cut in separator(x)]
        # Not near-duplicates of the pool's cuts.
        if self.cut_list and candidates:
            pool = sp.csr_matrix(
                (
                    np.concatenate([c["coef"] / np.linalg.norm(c["coef"]) for c in self.cut_list]),
                    np.concatenate([c["idx"] for c in self.cut_list]),
                    np.concatenate([[0], np.cumsum([c["idx"].size for c in self.cut_list])]),
                ),
                shape=(len(self.cut_list), n),
            )
            fresh = []
            for cut in candidates:
                _, idx, coef, _ = cut
                direction = np.zeros(n)
                direction[idx] = coef / np.linalg.norm(coef)
                if np.abs(pool @ direction).max() <= _CUT_MAX_PARALLELISM:
                    fresh.append(cut)
            candidates = fresh
        found = select_cuts(candidates, x, n, max_cuts=_TREE_CUTS_PER_ROUND)
        if not found:
            return warm

        max_rows = self.tree_rows_cap - self.tree_padding_rows
        nnz = sum(cut["idx"].size for cut in self.cut_list)
        replaced, placed, families = [], 0, []
        for family, idx, coef, rhs in found:
            new = dict(family=family, idx=idx, coef=coef, rhs=rhs, age=0)
            if len(self.cut_list) < max_rows and nnz + idx.size <= self.tree_nnz_cap:
                self.cut_list.append(new)
                nnz += idx.size
                placed += 1
                families.append(family)
                continue
            slots = [
                i
                for i, cut in enumerate(self.cut_list)
                if cut["age"] >= 2 and nnz - cut["idx"].size + idx.size <= self.tree_nnz_cap
            ]
            if not slots:
                continue
            slot = max(slots, key=lambda i: self.cut_list[i]["age"])
            nnz += idx.size - self.cut_list[slot]["idx"].size
            self.cut_list[slot] = new
            replaced.append(slot)
            placed += 1
            families.append(family)
        if not placed:
            return warm
        self.tree_cuts_added += placed
        self.set_mip(base.with_rows(*self.tree_block()))
        if self.verbose:
            print(
                f"C {self.nodes:8d} {len(self.open):8d}   cuts +{placed} "
                f"({families.count('cover')} cover, {families.count('mir')} MIR)"
                + (f", {len(replaced)} replacing idle cuts" if replaced else "")
                + f", pool {len(self.cut_list)}/{max_rows}"
            )
        if replaced:
            duals = np.asarray(solution.dual_ineq).copy()
            duals[[base_m + i for i in replaced]] = 0.0
            warm = (solution._replace(dual_ineq=jnp.asarray(duals)), warm[1])
        return warm

    def feasibility_pump(self, lower, upper, result, deadline):
        """The objective feasibility pump (Achterberg & Berthold) from the
        root LP point: alternately round, and solve the LP that minimises the
        L1 distance to the rounded point (over integers at a bound), blended
        with the objective at a weight that decays by 0.9 per iteration. Only
        the cost changes, so the pump reuses the root's compiled LP shape.
        Stops at the first incumbent, after ``pump_iterations`` or at
        ``deadline``. Returns whether it found one."""
        mip, ints = self.mip, self.is_int
        n_int = int(ints.sum())
        if n_int == 0:
            return False
        rng = np.random.default_rng(0)
        c_unit = mip.c / max(np.linalg.norm(mip.c), 1e-12)
        lp_lower = np.where(ints, lower, mip.lower)
        lp_upper = np.where(ints, upper, mip.upper)
        x = np.asarray(result["solution"].primal, np.float64)
        warm = (result["solution"], result["opt_state"])
        alpha, previous, seen = 1.0, None, set()
        best_distance = np.inf
        tolerances = dict(
            primal_feasibility_tolerance=self.pump_tolerance,
            dual_feasibility_tolerance=self.pump_tolerance,
            dual_gap_tolerance=self.pump_tolerance,
        )
        for iteration in range(self.pump_iterations):
            if time.time() >= deadline or self.out_of_time():
                break
            rounded = np.clip(np.round(x), lower, upper)
            if previous is not None and np.array_equal(rounded[ints], previous[ints]):
                # A 1-cycle: flip the integers furthest from their rounding.
                gap = np.where(ints, np.abs(x - rounded), -1.0)
                flips = int(rng.integers(10, 31))
                for j in np.argsort(-gap)[: min(flips, n_int)]:
                    if gap[j] <= 0:
                        break
                    rounded[j] += 1.0 if x[j] > rounded[j] else -1.0
            key = rounded[ints].tobytes()
            if key in seen:
                # A longer cycle: perturb at random.
                rho = rng.uniform(-0.3, 0.7, x.shape[0])
                flip = ints & (np.abs(x - rounded) + np.maximum(rho, 0.0) > 0.5)
                rounded[flip] = np.where(
                    x[flip] > rounded[flip], rounded[flip] + 1.0, rounded[flip] - 1.0
                )
                key = rounded[ints].tobytes()
            rounded = np.clip(rounded, lower, upper)
            seen.add(key)
            previous = rounded
            self.pump_last = rounded

            candidate = x.copy()
            candidate[ints] = rounded[ints]
            if self.try_incumbent(candidate, "F", warm):
                return True

            # Distance to the rounded point, over integers at a bound.
            delta = np.zeros_like(x)
            at_lower = ints & (rounded <= lower)
            at_upper = ints & (rounded >= upper) & ~at_lower
            delta[at_lower], delta[at_upper] = 1.0, -1.0
            if not delta.any():
                break
            cost = (1.0 - alpha) * delta / np.linalg.norm(delta) + alpha * c_unit
            alpha *= 0.9
            lp = mip.lp.with_values(
                self.values._replace(
                    c=jnp.asarray(cost, self.dtype),
                    lower_bounds=jnp.asarray(lp_lower, self.dtype),
                    upper_bounds=jnp.asarray(lp_upper, self.dtype),
                )
            )
            pumped = self.solve_lp(lp, warm, self.node_max_epochs, **tolerances)
            if pumped["stop_reason"] == "interrupted":
                break
            x = np.asarray(pumped["solution"].primal, np.float64)
            warm = (pumped["solution"], pumped["opt_state"])

            # Close to integral: let propagation finish the rounding.
            distance = np.abs(x - np.round(x))[ints].sum()
            if self.use_fix_and_propagate and distance < 0.9 * best_distance:
                best_distance = distance
                candidate = fix_and_propagate(
                    x, lower, upper, ints, self.row_propagator
                )
                if candidate is not None and self.try_incumbent(candidate, "F", warm):
                    return True
        return False

    def rins(self, x, warm):
        """RINS around the incumbent and the node LP point ``x``, with the
        sub-MIP solved by Jaddle (``jaddle_rins``) or HiGHS
        (``highs_rins``), per ``rins_solver``."""
        started = time.time()
        seconds = self.rins_seconds
        if self.deadline is not None:
            seconds = min(seconds, self.deadline - started)
        if self.rins_solver == "highs":
            found = highs_rins(
                self.base, self.incumbent[1], x, self.cutoff(), max_seconds=seconds
            )
        else:
            found = self.jaddle_rins(x, seconds, warm)
        self.rins_calls += 1
        if found is not None:
            self.try_incumbent(found, "N", warm)
        self.heuristic_seconds += time.time() - started

    def sub_search(self, lower, upper, seconds, nodes, warm=None):
        """
        Search the sub-MIP of ``self.base`` with the box ``[lower, upper]``
        for up to ``seconds`` and ``nodes`` nodes, pruned at this search's
        cutoff (``initial_cutoff``), so only better points come back. The
        sub-MIP has the rows of ``self.base`` and so reuses its compiled LP
        solver; root cuts, tree cuts, the pump, the feasibility jump and the
        sub-MIP heuristics are off in it. Its root LP is warm-started from
        ``warm`` (with the duals of cut rows dropped). Returns the best point
        found, or ``None``.
        """
        base = self.base
        dtype = base.lp.c.dtype
        lp = base.lp.with_values(
            base.lp.values()._replace(
                lower_bounds=jnp.asarray(lower, dtype),
                upper_bounds=jnp.asarray(upper, dtype),
            )
        )
        sub_mip = MIP(lp, self.is_int, base.offset, base.maximise)
        started = time.time()
        settings = dict(
            self.settings,
            deadline=started + seconds,
            start=started,
            max_nodes=nodes,
            initial_cutoff=self.cutoff(),
            cuts=False,
            gomory_cuts=False,
            tree_cuts=False,
            coefficient_tightening=False,
            feasibility_pump=False,
            feasibility_jump=False,
            highs_incumbent_seconds=0.0,
            rins_seconds=0.0,
            rens_seconds=0.0,
            dive_every=0,
            batch_size=1,
            verbose=False,
        )
        search = _BranchAndBound(sub_mip, **settings)
        if warm is not None:
            m_ineq = base.A.shape[0] - base.n_eq
            solution = warm[0]._replace(dual_ineq=warm[0].dual_ineq[:m_ineq])
            search.root_warm = {"solution": solution, "opt_state": None}
        return search.run()["x"]

    def rens(self, x, lower, upper, warm, min_fixed=0.5):
        """
        RENS (relaxation enforced neighbourhood search) from the root LP
        point ``x``: the integers integral at ``x`` are fixed there, the
        fractional ones restricted to ``[floor, ceil]``, and that sub-MIP
        searched (``sub_search``, ``rens_seconds`` and ``rens_nodes``).
        Needs no incumbent. Skipped when fewer than ``min_fixed`` of the
        integers are integral. A point found is marked ``"E"``.
        """
        ints = self.is_int
        x = np.asarray(x, np.float64)
        integral = ints & (np.abs(x - np.round(x)) <= self.int_tol)
        if integral.sum() < min_fixed * ints.sum():
            return
        started = time.time()
        sub_lower = np.where(ints, np.maximum(lower, np.floor(x + self.int_tol)), self.base.lower)
        sub_upper = np.where(ints, np.minimum(upper, np.ceil(x - self.int_tol)), self.base.upper)
        sub_lower = np.where(integral, np.round(x), sub_lower)
        sub_upper = np.where(integral, np.round(x), sub_upper)
        if np.any(sub_lower > sub_upper):
            return
        seconds = self.rens_seconds
        if self.deadline is not None:
            seconds = min(seconds, self.deadline - started)
        if seconds <= 0:
            return
        found = self.sub_search(sub_lower, sub_upper, seconds, self.rens_nodes, warm)
        if found is not None:
            self.try_incumbent(found, "E", warm)
        self.heuristic_seconds += time.time() - started

    def dive(self, x, lower, upper, warm, rule=None):
        """
        An LP-based dive from a node's LP point ``x`` in its box: round one
        fractional integer (chosen by ``rule``), propagate (objective-aware,
        ``heuristic_propagator``), re-solve the LP warm with a quarter of the
        node epoch budget, and repeat, up to ``dive_depth`` times. Stops at
        an integral LP point (tried as an incumbent, ``"D"``), an infeasible
        box (after flipping the last rounding once), or a bound at the
        cutoff. ``rule``: ``"fractional"`` (the most nearly integral
        variable, to its nearer integer), ``"guided"`` (towards the
        incumbent, the variable closest to it first) or ``"pseudocost"``
        (the cheaper direction by pseudocost); ``None`` rotates through
        them.
        """
        started = time.time()
        ints = self.is_int
        if rule is None:
            rules = ["fractional", "pseudocost"] + (
                ["guided"] if self.incumbent is not None else []
            )
            rule = rules[self.dives % len(rules)]
        self.dives += 1
        x = np.asarray(x, np.float64)
        lower, upper = lower.copy(), upper.copy()
        epochs = max(self.node_max_epochs // 4, 10)
        flipped = False
        try:
            for _ in range(self.dive_depth):
                frac = x - np.floor(x)
                fractional = (
                    ints & (np.abs(x - np.round(x)) > self.int_tol) & (x > lower) & (x < upper)
                )
                if not fractional.any():
                    self.try_incumbent(x, "D", warm)
                    return
                cand = np.flatnonzero(fractional)
                f = frac[cand]
                if rule == "guided" and self.incumbent is not None:
                    target = self.incumbent[1][cand]
                    pick = int(np.argmin(np.abs(x[cand] - target)))
                    up = target[pick] > x[cand[pick]]
                elif rule == "pseudocost" and self.pc_count.any():
                    known = self.pc_count > 0
                    mean = np.array([
                        self.pc_sum[d][known[d]].sum() / self.pc_count[d][known[d]].sum()
                        if known[d].any() else 1.0
                        for d in range(2)
                    ])
                    pc = np.where(
                        known[:, cand],
                        self.pc_sum[:, cand] / np.maximum(self.pc_count[:, cand], 1),
                        mean[:, None],
                    )
                    down_cost, up_cost = f * pc[0], (1.0 - f) * pc[1]
                    cheaper = np.minimum(down_cost, up_cost)
                    pick = int(np.argmin(cheaper))
                    up = up_cost[pick] < down_cost[pick]
                else:
                    distance = np.minimum(f, 1.0 - f)
                    pick = int(np.argmin(distance))
                    up = f[pick] >= 0.5
                j = cand[pick]
                for attempt in (up, not up):
                    lo, hi = lower.copy(), upper.copy()
                    if attempt:
                        lo[j] = math.ceil(x[j])
                    else:
                        hi[j] = math.floor(x[j])
                    lo, hi, infeasible = self.heuristic_propagator()(lo, hi)
                    if not infeasible:
                        break
                    if flipped:
                        return
                    flipped = True
                else:
                    return
                lower, upper = lo, hi
                result = self.solve_lp(self.node_lp(lower, upper), warm, epochs)
                if result["stop_reason"] in ("interrupted", "time_limit"):
                    return
                bound, _ = self.lp_bound(result["solution"], lower, upper)
                if self.round_bound(bound) >= self.cutoff():
                    return
                x = np.asarray(result["solution"].primal, np.float64)
                warm = (result["solution"], result["opt_state"])
            self.try_incumbent(x, "D", warm)
        finally:
            self.heuristic_seconds += time.time() - started

    def jaddle_rins(self, x, seconds, warm=None, min_fixed=0.3):
        """
        RINS with Jaddle's own search as the sub-MIP solver. The integers on
        which the incumbent and ``x`` agree (within the integrality
        tolerance) are fixed at the incumbent's values, and the rest is
        searched for up to ``seconds`` and ``rins_nodes`` nodes, pruned at
        this search's cutoff (``initial_cutoff``), so only better points come
        back. The sub-MIP has the rows of ``self.base`` and so reuses its
        compiled LP solver; root cuts, tree cuts, the pump and RINS itself
        are off in it, as is the feasibility jump (fix-and-propagate covers a
        small neighbourhood). Its root LP is warm-started from ``warm`` (the
        node's point, with the duals of the cut rows dropped). Skipped
        (``None``) when fewer than ``min_fixed`` of the integers agree, or
        when the same neighbourhood was searched at the same cutoff. Returns
        the best point found, or ``None``.
        """
        ints = self.is_int
        incumbent = self.incumbent[1]
        agree = ints & (np.abs(np.asarray(x, np.float64) - incumbent) <= self.int_tol)
        if seconds <= 0 or agree.sum() < min_fixed * ints.sum():
            return None
        # The same neighbourhood with the same cutoff has been searched.
        key = (agree.tobytes(), incumbent[agree].tobytes(), self.cutoff())
        if key in self.rins_seen:
            self.rins_skipped += 1
            return None
        self.rins_seen.add(key)
        lower = np.where(agree, incumbent, self.base.lower)
        upper = np.where(agree, incumbent, self.base.upper)
        return self.sub_search(lower, upper, seconds, self.rins_nodes, warm)

    def jump(self, start, lower, upper, warm, deadline, seed):
        """``feasibility_jump`` on the MIP as given from ``start``; a point it
        finds goes through ``try_incumbent`` (which re-solves continuous
        variables)."""
        x, feasible, _ = feasibility_jump(
            self.base,
            start,
            lower,
            upper,
            tol=self.feas_tol,
            max_iterations=self.jump_iterations,
            deadline=None if not np.isfinite(deadline) else deadline,
            seed=seed,
        )
        if feasible:
            self.try_incumbent(x, "J", warm)
        elif self.use_repair and time.time() < deadline:
            # A near miss: free the variables around its violated rows and
            # solve that small MIP.
            left = deadline - time.time() if np.isfinite(deadline) else 60.0
            repaired = repair(
                self.base,
                x,
                lower,
                upper,
                max_seconds=left,
                tol=self.feas_tol,
                propagator=BoundPropagator(
                    self.base.A, self.base.row_lower, self.base.row_upper, self.is_int
                ),
            )
            if repaired is not None:
                self.try_incumbent(repaired, "R", warm)

    def root_heuristics(self, lower, upper, result):
        """Fix-and-propagate from the root LP point, then the feasibility
        pump while there is no incumbent, within the heuristic budget."""
        started = time.time()
        if self.heuristic_budget is not None:
            deadline = started + self.heuristic_budget
        elif self.deadline is not None:
            deadline = started + 0.2 * (self.deadline - self.start)
        else:
            deadline = np.inf
        warm = (result["solution"], result["opt_state"])
        x = np.asarray(result["solution"].primal, np.float64)
        if self.use_fix_and_propagate:
            candidate = fix_and_propagate(
                x, lower, upper, self.is_int, self.heuristic_propagator()
            )
            if candidate is not None:
                self.try_incumbent(candidate, "P", warm)
        if self.use_jump and self.incumbent is None:
            # From the rounded LP point, then from the bounds nearest zero.
            nearest_zero = np.clip(np.zeros_like(x), lower, upper)
            for start, seed in ((x, 0), (nearest_zero, 1)):
                if self.incumbent is not None or time.time() >= deadline:
                    break
                self.jump(start, lower, upper, warm, deadline, seed)
        if self.use_pump and self.incumbent is None:
            self.feasibility_pump(lower, upper, result, deadline)
            if (
                self.use_jump
                and self.incumbent is None
                and self.pump_last is not None
                and time.time() < deadline
            ):
                self.jump(self.pump_last, lower, upper, warm, deadline, seed=2)
        if self.incumbent is None and self.highs_seconds and not self.out_of_time():
            seconds = self.highs_seconds
            if self.deadline is not None:
                seconds = min(seconds, self.deadline - time.time())
            found = highs_incumbent(self.base, max_seconds=seconds)
            if found is not None:
                self.try_incumbent(found, "X", warm)
        self.heuristic_seconds += time.time() - started
        if self.verbose:
            print(
                f"Root heuristics: {self.heuristic_seconds:.1f}s, "
                + (
                    f"incumbent {self.model_value(self.incumbent[0]):.6e}"
                    if self.incumbent is not None
                    else "no incumbent"
                )
            )

    def next_node(self):
        """The next node to solve: the pending plunge child, or else the top
        of the heap (which starts a new plunge)."""
        if self.plunge_node is not None:
            node, self.plunge_node = self.plunge_node, None
            return node
        _, _, node = heapq.heappop(self.open)
        self.plunge_left = self.plunge_depth
        return node

    def batched_solver(self):
        """``jax.vmap`` of ``jaddle_linear.make_solver`` over node bounds and
        warm starts, jitted once for the current LP (the shape is final once
        the root cuts are in)."""
        if self._batched is None or self._batched[0] is not self.mip:
            import inspect

            accepted = set(inspect.signature(jl.make_solver).parameters)
            settings = {
                key: value
                for key, value in self.lp_kwargs.items()
                if key in accepted and key != "build_lp"
            }
            solve_fn = jl.make_solver(
                self.mip.lp.with_values, max_epochs=self.node_max_epochs, **settings
            )
            axes = jl.LPValues(*([None] * len(jl.LPValues._fields)))._replace(
                lower_bounds=0, upper_bounds=0
            )
            self._batched = (
                self.mip,
                jax.jit(jax.vmap(solve_fn, in_axes=(axes, 0))),
            )
        return self._batched[1]

    def batch_step(self):
        """Solve up to ``batch_size`` open nodes in one batched device call,
        then evaluate them in order on the host exactly as the serial search
        does. A short batch is padded with copies of its last member, whose
        results are ignored."""
        members, boxes = [], []
        while len(members) < self.batch_size and (
            self.open or self.plunge_node is not None
        ):
            if self.max_nodes is not None and self.nodes + len(members) >= self.max_nodes:
                break
            node = self.next_node()
            if node.bound >= self.cutoff():
                self.close(node.bound)
                continue
            box = self.expand(node.changes)
            if np.any(box[0] > box[1]):
                # Emptied by reduced-cost fixing of the root box.
                self.close(max(node.bound, self.cutoff_bound()))
                continue
            members.append(node)
            boxes.append(box)
        if not members:
            return
        padded = members + [members[-1]] * (self.batch_size - len(members))
        padded_boxes = boxes + [boxes[-1]] * (self.batch_size - len(members))
        lp_lower = np.stack(
            [np.where(self.is_int, lo, self.mip.lower) for lo, _ in padded_boxes]
        )
        lp_upper = np.stack(
            [np.where(self.is_int, up, self.mip.upper) for _, up in padded_boxes]
        )
        values = self.values._replace(
            lower_bounds=jnp.asarray(lp_lower, self.dtype),
            upper_bounds=jnp.asarray(lp_upper, self.dtype),
        )
        warm_starts = jax.tree.map(
            lambda *leaves: jnp.stack(leaves), *[node.warm[0] for node in padded]
        )
        out = self.batched_solver()(values, warm_starts)
        self.batches += 1
        converged = np.asarray(out.converged)
        stop_code = np.asarray(out.stop_code)
        epochs = np.asarray(out.epochs)
        for i, node in enumerate(members):
            self.nodes += 1
            self.lp_epochs += int(epochs[i])
            # An incumbent found by an earlier member may prune this one.
            if node.bound >= self.cutoff():
                self.close(node.bound)
                continue
            result = {
                "solution": jax.tree.map(lambda leaf: leaf[i], out.solution),
                "converged": bool(converged[i]),
                "stop_reason": "certificate" if stop_code[i] == 1 else "max_epochs",
                "epochs": int(epochs[i]),
                "opt_state": node.warm[1],
            }
            lower, upper = boxes[i]
            self.evaluate(node, lower, upper, self.node_lp(lower, upper), result)
            self.log(" ", None, node.depth)

    def run(self):
        """The search, on ``self.device`` when one is set."""
        if self.device is None:
            return self._run()
        with jax.default_device(self.device):
            return self._run()

    def _run(self):
        mip = self.mip
        lower, upper = mip.lower.copy(), mip.upper.copy()
        if np.any(lower > upper):
            return self.result("infeasible")
        if self.propagator is not None:
            lower, upper, infeasible = self.propagator(lower, upper, max_passes=20)
            if infeasible:
                return self.result("infeasible")
        if self.use_coefficient_tightening:
            box = (lower, upper)
            if self.propagator is None:
                box = self.row_propagator(lower, upper, max_passes=20)[:2]
            tightened, count = tighten_coefficients(mip, *box)
            if count:
                # Same integer-feasible points; search the tightened rows
                # (incumbents are still checked against the MIP as given).
                self.coefficients_tightened = count
                self.base = tightened
                self.set_mip(tightened)
                mip = tightened
                if self.propagator is not None:
                    lower, upper, infeasible = self.propagator(
                        lower, upper, max_passes=20
                    )
                    if infeasible:
                        return self.result("infeasible")
        self.root_lower, self.root_upper = lower, upper
        if self.verbose:
            print(
                f"MIP: {mip.num_variables} variables ({mip.num_integers} integer), "
                f"{mip.A.shape[0]} rows, {mip.A.nnz} nonzeros, on "
                f"{(self.device or jax.devices()[0]).platform}"
            )
            if self.coefficients_tightened:
                print(f"Coefficient tightening: {self.coefficients_tightened} coefficients")
            if self.objective_step is not None:
                print(f"Objective values are multiples of {self.objective_step:g}")

        root = _Node(-np.inf, 0, None, None, None)
        lp = self.node_lp(lower, upper)
        detection = self.detect(lp, self.root_warm, self.root_max_epochs)
        if detection["status"] == "primal_infeasible":
            return self.result("infeasible")
        if detection["status"] == "dual_infeasible":
            return self.result("unbounded_or_infeasible")
        result = detection["result"]
        self.nodes = 1
        if self.verbose:
            root_bound, _ = self.lp_bound(result["solution"], lower, upper)
            print(
                f"Root LP: {result['stop_reason']}, objective "
                f"{self.model_value(float(mip.c @ np.asarray(result['solution'].primal))):.6e}, "
                f"bound {self.model_value(root_bound):.6e}, "
                f"{time.time() - self.start:.1f}s"
            )
        if result["stop_reason"] == "interrupted":
            return self.result("interrupted")
        if (
            self.cuts
            and self.cut_rounds
            and result["stop_reason"] != "time_limit"
            and not self.out_of_time()
        ):
            result = self.cut_loop(lower, upper, result)
            if result == "infeasible":
                return self.result("infeasible")
            lp = self.node_lp(lower, upper)
            if self.cut_list and self.propagator is not None:
                lower, upper, infeasible = self.propagator(lower, upper, max_passes=20)
                if infeasible:
                    return self.result("infeasible")
                self.root_lower, self.root_upper = lower, upper
                lp = self.node_lp(lower, upper)
            if result["stop_reason"] == "interrupted":
                return self.result("interrupted")
        if self.use_tree_cuts:
            result = self.reserve_tree_cuts(result)
            lp = self.node_lp(lower, upper)
        if result["stop_reason"] == "time_limit" or self.out_of_time():
            bound, rigorous = self.lp_bound(result["solution"], lower, upper)
            self.close(bound, rigorous)
            return self.result("time_limit")
        if (
            self.use_fix_and_propagate
            or self.use_jump
            or self.use_pump
            or self.highs_seconds
        ):
            self.root_heuristics(lower, upper, result)
        root_x = np.asarray(result["solution"].primal, np.float64)
        root_warm = (result["solution"], result["opt_state"])
        if self.rens_seconds and not self.out_of_time():
            self.rens(root_x, lower, upper, root_warm)
        if self.dive_every and not self.out_of_time():
            self.dive(root_x, lower, upper, root_warm, rule="fractional")
            if self.incumbent is not None:
                self.dive(root_x, lower, upper, root_warm, rule="guided")
        if self.rins_seconds and self.incumbent is not None and not self.out_of_time():
            self.rins_pending = False
            self.rins(root_x, root_warm)
        if self.verbose:
            print(
                f"  {'Nodes':>8} {'Open':>8} {'Depth':>6} {'Bound':>13} "
                f"{'Incumbent':>13} {'Gap':>9} {'Time':>9}"
            )
        self.evaluate(root, lower, upper, lp, result)

        status = "optimal"
        try:
            while self.open or self.plunge_node is not None:
                if self.out_of_time():
                    status = "time_limit"
                    break
                if self.max_nodes is not None and self.nodes >= self.max_nodes:
                    status = "node_limit"
                    break
                if self.incumbent is not None and self.nodes % 20 == 0:
                    if self.global_bound() >= self.cutoff():
                        break
                if not self.tighten_root():
                    self.close_all()
                    break
                if self.batch_size > 1:
                    self.batch_step()
                    continue
                node = self.next_node()
                if node.bound >= self.cutoff():
                    self.close(node.bound)
                    continue
                lower, upper = self.expand(node.changes)
                if np.any(lower > upper):
                    # Emptied by reduced-cost fixing of the root box.
                    self.close(max(node.bound, self.cutoff_bound()))
                    continue
                lp = self.node_lp(lower, upper)
                result = self.solve_lp(lp, node.warm, self.node_max_epochs)
                self.nodes += 1
                if result["stop_reason"] == "interrupted":
                    self.push(node)
                    status = "interrupted"
                    break
                if result["stop_reason"] == "time_limit":
                    self.push(node)
                    status = "time_limit"
                    break
                self.evaluate(node, lower, upper, lp, result)
                self.log(" ", None, node.depth)
        except KeyboardInterrupt:
            status = "interrupted"
        return self.result(status)


def solve(
    model,
    mip_rel_gap=1e-4,
    mip_abs_gap=1e-6,
    max_nodes=None,
    max_seconds=None,
    branching="pseudocost",
    node_selection="hybrid",
    integrality_tolerance=1e-5,
    feasibility_tolerance=1e-6,
    lp_tolerance=1e-6,
    node_max_epochs=200,
    root_max_epochs=None,
    infeasibility_tolerance=1e-8,
    heuristic_every=10,
    propagate=True,
    coefficient_tightening=True,
    cuts=True,
    cut_rounds=None,
    max_cuts_per_round=100,
    cut_min_progress=None,
    cut_round_epochs=20,
    gomory_cuts="auto",
    aggregate_integers="auto",
    tree_cuts=False,
    tree_cut_every=10,
    fix_and_propagate=True,
    feasibility_jump=True,
    jump_iterations=None,
    repair=True,
    repair_max_epochs=20,
    highs_incumbent_seconds=0.0,
    rins_seconds=0.0,
    rins_every=50,
    rins_solver="auto",
    rins_nodes=200,
    rens_seconds=None,
    rens_nodes=500,
    dive_every=0,
    dive_depth=20,
    plunge_depth=8,
    batch_size=1,
    feasibility_pump=True,
    pump_iterations=30,
    pump_tolerance=1e-4,
    heuristic_seconds=None,
    presolve=False,
    highs_options=None,
    lp_options=None,
    verbose=False,
    log_every_seconds=1.0,
    device="auto",
):
    """
    Solve a mixed-integer LP by branch and bound, with ``jaddle_linear.solve``
    for the LP at every node.

    ``model`` is a ``MIP``, or anything ``from_highs`` reads: a path to a file
    HiGHS reads, a ``highspy.Highs`` with a model loaded or a
    ``highspy.HighsLp``.

    The search:

    * **Root.** Bounds are tightened by activity-based propagation
      (``propagate``), and the root LP is solved by
      ``jaddle_linear.detect_infeasibility``, which certifies optimality,
      infeasibility or an unbounded relaxation.
    * **Coefficient tightening** (``coefficient_tightening``). Over the
      propagated root box, each binary's coefficient in an inequality row is
      cut to the row's excess ``maxact − b`` when it is larger
      (``tighten_coefficients``). This shrinks big-M rows such as
      ``y − M z <= 0`` to the bound on ``y`` that the other rows imply,
      without changing the integer-feasible points; the search then runs on
      the tightened rows, and incumbents are still checked against the MIP
      as given.
    * **Root cuts** (``cuts``). Up to ``cut_rounds`` rounds (default: 40
      for a MIP with at most 10,000 nonzeros and Gomory cuts, else 5) of
      Gomory mixed-integer cuts (``gomory_cuts``; ``"auto"``, the default,
      turns them on for MIPs with at most 10,000 nonzeros, whose crossovers
      take milliseconds), knapsack
      cover cuts (``cover_cut``) and c-MIR cuts on single rows and on
      aggregations of up to three (``mir_cut``). With Gomory cuts on, each
      round first crosses the root LP point over to an optimal vertex
      (``jaddle_linear.crossover``, limited to 5% of ``max_seconds``): the
      Gomory cuts come from its basis, and all three families are separated
      at the vertex. MIR cuts are complemented MIR (``cmir_cut``): every
      integer coefficient is tried as the scaling, then each integer
      variable's complementation is flipped while that deepens the cut.
      They come from single rows and from aggregates of up to three rows
      that cancel continuous variables; with ``aggregate_integers``
      (``"auto"``, the default: on for MIPs with at most 10,000 nonzeros),
      aggregates of up to five rows also cancel the most fractional integer
      term, so pure-integer rows combine. On lseu that lifts the root bound
      (exact LPs) from 1,061 to 1,100. Cover and MIR cuts are separated by
      ``CutSeparator``:
      up to three covers per row (``cover_cuts``), each lifted exactly by
      a dynamic program over the knapsack. With coefficient tightening,
      they are separated from the rows as given as well as the tightened
      rows, since the original knapsacks can give stronger covers. A
      round keeps at most
      ``max_cuts_per_round`` cuts with efficacy (distance cut off) at least
      1e-4 that are not near-parallel, retires cuts that have been slack
      with a zero dual for two rounds, and re-solves the LP warm. The rounds
      stop when the bound gains less than ``cut_min_progress`` (relative)
      over the last three rounds (default 1e-3 with the 40-round setting,
      else 1e-4). Each round's re-solve stops after ``cut_round_epochs``
      (default 20): separation needs only a reasonable point and the
      Lagrangian bound holds at any duals; the root is solved to convergence
      once the rounds end. On lseu that cuts the root from 17s to 7s (bound
      1,091 → 1,095).
      Cover and MIR cuts follow from a row and the bounds alone, so they
      are valid whatever the LP point's accuracy; Gomory cuts rest on the
      basis, which HiGHS proves optimal, and skip tableau rows that do not
      reproduce the basis to 1e-7. The cut block is padded to
      power-of-two sizes, so rounds rarely recompile the LP solver.
    * **Tree cuts** (``tree_cuts``, off by default). Every
      ``tree_cut_every`` nodes, cover and MIR cuts are separated at the
      node's LP point over the root box (so they are valid everywhere),
      and up to 10 violated ones join the cut pool, in free room or in
      place of cuts slack with a zero dual for two rounds. The node is not
      re-solved; its children solve against the new cuts. The pool has a
      fixed capacity, reserved after the root cuts, so its rows and
      nonzeros keep one shape and the LP solver is not recompiled (the
      compiled solver takes the sparsity pattern as data). Serial search
      only: the batched solver captures the pattern, so ``batch_size > 1``
      turns tree cuts off.
    * **Nodes.** Each child is warm-started from its parent's primal-dual
      point and step-size state; with the compiled solver cached, a small
      node takes milliseconds. A node is pruned when its Lagrangian bound
      (``lagrangian_bound``) reaches the cutoff
      ``incumbent − max(mip_abs_gap, mip_rel_gap·|incumbent|)``. A node LP
      that does not converge in ``node_max_epochs`` gets an infeasibility
      check of the same length, and is pruned if it is infeasible.
    * **Bounds from duals.** The Lagrangian terms of a node's duals bound
      any box, so each child gets the bound of its parent's duals over its
      own propagated box, and a child whose bound reaches the cutoff is
      closed without an LP solve. The same terms drive reduced-cost fixing
      (``reduced_cost_tightening``): once there is an incumbent, each node
      tightens the integer bounds its children inherit, and the root box is
      tightened with the root's duals whenever the incumbent improves.
    * **Integral objective.** When every objective value is a multiple of
      a step ``g`` (``objective_step``: integer variables with costs that
      are multiples of ``g``, continuous ones with no cost), node bounds
      are rounded up to multiples of ``g`` and the cutoff is at most
      ``incumbent − g/2``.
    * **Batched nodes** (``batch_size``). With ``batch_size > 1`` the tree
      takes up to that many open nodes at a time (the plunge child first,
      then the heap's top) and solves their LPs in one device call:
      ``jax.vmap`` of ``jaddle_linear.make_solver`` over the nodes' bounds
      and warm starts, jitted once. The nodes are then evaluated in order on
      the host exactly as in the serial search. Members run in lockstep, so
      a node that will not converge holds its batch to ``node_max_epochs``.
      Worth it on a GPU; on a CPU the members run one after another.
    * **Device** (``device``). ``"auto"`` (default) runs MIPs with at
      most 10,000 nonzeros on the CPU: their node LPs are latency-bound
      on an accelerator (flugpl: 3s on the CPU, 18s on a GPU).
      ``"cpu"``, ``"gpu"`` or ``"tpu"`` pick one; ``None`` keeps JAX's
      default device.
    * **Branching** (``branching``): ``"pseudocost"`` (default) scores each
      fractional variable by the product of its estimated down and up bound
      gains, learnt from the nodes solved so far (most fractional until
      there are any); ``"most_fractional"`` picks the variable nearest
      ½. Each child's bounds are propagated, so many infeasible children
      are dropped without an LP solve.
    * **Node selection** (``node_selection``): ``"hybrid"`` (default) dives
      depth first until the first incumbent, then takes the best bound with
      plunges: after each best-bound node, up to ``plunge_depth`` steps
      continue into its preferred child, so the search keeps reaching leaves
      (and new incumbents) while it raises the bound;
      ``"best_first"`` and ``"depth_first"`` do one throughout.
    * **Incumbents.** A node LP point with integral integers (within
      ``integrality_tolerance``), and the LP point rounded at every
      ``heuristic_every``-th node, are rounded and checked. A point is
      feasible when its row violation relative to ``1 + ‖b‖∞`` and its bound
      violation are at most ``feasibility_tolerance``. With continuous
      variables, a rounded point that is not is repaired by fixing the
      integers and solving the LP in the rest, for up to
      ``repair_max_epochs`` epochs. The repair is skipped when the same
      integers already failed, when no violated row has a continuous
      variable, or when bound propagation shows the fixed box infeasible.
    * **Primal heuristics.** After the root cuts, ``fix_and_propagate``
      (the function of that name) rounds the root LP point in chunks with
      bound propagation after each. While there is still no incumbent,
      ``feasibility_jump`` (the function; an LP-free weighted local search
      on the rows, on the device) runs for up to ``jump_iterations`` steps
      from the rounded LP point and from the bounds nearest zero, then the
      objective feasibility pump (``feasibility_pump``) runs for up to
      ``pump_iterations`` warm-started LP solves at ``pump_tolerance``,
      followed by one more jump from the pump's last rounded point. They
      share ``heuristic_seconds`` (default: 20% of ``max_seconds``, or no
      time limit without one). In the tree, fix-and-propagate also runs at
      every ``heuristic_every``-th node where plain rounding fails. Their
      incumbents are marked ``"P"``, ``"J"`` and ``"F"`` in the log and
      history. A jump that ends near-feasible is handed to ``repair`` (a
      backtracking search with propagation over the variables around its
      violated rows), marked ``"R"``. ``highs_incumbent_seconds > 0`` asks
      HiGHS (``highs_incumbent``) for a first incumbent when all of these
      fail, marked ``"X"``; it is off by default, so the search stays
      Jaddle's own. Likewise ``rins_seconds > 0`` runs RINS once at the
      root and every ``rins_every`` nodes once there is an incumbent: the
      integers on which the incumbent and the node's LP point agree are
      fixed, and the rest is searched for a better solution, marked
      ``"N"``. With ``rins_solver="jaddle"`` the sub-MIP is
      searched by this branch and bound (``_BranchAndBound.jaddle_rins``:
      up to ``rins_nodes`` nodes, pruned at the current cutoff, reusing
      the compiled LP solver); with ``"highs"``, by HiGHS
      (``highs_rins``). ``"auto"`` (default) picks Jaddle for MIPs with at
      most 10,000 nonzeros and HiGHS above that, where the Jaddle
      sub-searches are slow and find little. A new incumbent from any other source also triggers
      a RINS call.
    * **RENS** (``rens_seconds``; ``None``, the default, gives 5s for MIPs
      with at most 10,000 nonzeros, else off). After the root heuristics,
      the integers that are integral at the root LP point are fixed and the
      fractional ones restricted to their floor and ceiling, and that
      sub-MIP is searched as RINS's is (up to ``rens_nodes`` nodes), marked
      ``"E"``. It needs no incumbent.
    * **Dives** (``dive_every``, off by default). LP-based dives at the
      root and every ``dive_every`` nodes: round a fractional integer
      (fractional, guided or pseudocost rule), propagate, re-solve the LP
      warm, up to ``dive_depth`` times; an integral end point is marked
      ``"D"``.
    * **Objective-aware propagation.** Once there is a cutoff, the
      propagation inside fix-and-propagate and dives includes the row
      ``c·x <= cutoff``, so they reject roundings that cannot improve on the
      incumbent.

    ``lp_tolerance`` is the relative tolerance of every node LP (all three of
    ``solve()``'s tolerances); ``lp_options`` (a dict) passes any other
    ``solve()`` option. Run in float64 (``jaddle_optimisers.configure_jax``)
    for the default tolerances.

    ``presolve=True`` presolves with HiGHS, runs the branch and bound on the
    reduced MIP, and maps the incumbent back with HiGHS's postsolve. The
    result then also has ``"postsolve_feasible"``: whether the postsolved
    point passes the feasibility check on the model as given.
    ``highs_options`` (a dict) is applied to the HiGHS instance that reads
    and presolves.

    Stops when the gap closes, the tree is empty, or at ``max_nodes``,
    ``max_seconds`` (wall clock, from entry) or Ctrl-C.

    Returns a dict:

    * ``"x"``: the best solution found (NumPy, float64), or ``None``.
    * ``"objective"``: its value in the model's own sense, offset included.
    * ``"bound"``: the proven bound on the optimum in the model's sense (a
      lower bound for a minimisation, an upper bound for a maximisation).
    * ``"gap"``: ``|objective − bound| / |objective|``.
    * ``"status"``: ``"optimal"`` (gap within tolerance), ``"infeasible"``,
      ``"unbounded_or_infeasible"`` (the LP relaxation is unbounded),
      ``"unresolved"`` (the search ended, but nodes closed without proof
      leave the gap open; see ``"unresolved_nodes"``), ``"time_limit"``,
      ``"node_limit"`` or ``"interrupted"``; ``"converged"`` is
      ``status == "optimal"``.
    * ``"bound_rigorous"``: False when a bound used a dropped reduced cost
      (see ``lagrangian_bound``).
    * ``"nodes"``, ``"infeasible_nodes"`` (dropped by propagation or proven
      infeasible), ``"unresolved_nodes"`` (nodes with no fractional integer
      left to branch on, closed on an LP that did not converge or a point
      that failed the feasibility check; their bound still counts), ``"lp_epochs"``,
      ``"seconds"``.
    * ``"history"``: one dict per new incumbent (``seconds``, ``nodes``,
      ``objective``, ``bound``, ``source``: ``"H"`` rounding, ``"P"``
      fix-and-propagate, ``"J"`` feasibility jump, ``"R"`` repair, ``"F"``
      feasibility pump, ``"X"`` HiGHS (``highs_incumbent_seconds``), ``"N"``
      RINS (``rins_seconds``) or ``"T"`` tree);
      ``"heuristic_seconds"``: time spent in the two heuristics.
    * ``"batches"``: batched node solves (``batch_size > 1``).
    * ``"objective_step"``: the step every objective value is a multiple
      of (``objective_step``), or ``None``; ``"reduced_cost_tightenings"``:
      integer bounds tightened by reduced-cost fixing;
      ``"children_pruned"``: children closed on their parent's duals
      before their LP was solved.
    * ``"lp_repairs"``, ``"lp_repairs_skipped"``: repair LPs solved for
      rounded points, and rounded points rejected without one.
    * ``"coefficients_tightened"``: coefficients cut by coefficient
      tightening; ``"tree_cuts_added"``: cuts added in the tree.
    * ``"cuts"``: the number of cuts in the final LP, by family
      (``"gomory"``, ``"cover"``, ``"mir"``); ``"root_bound_before_cuts"``: the root bound
      before them (``None`` when no cut round ran).
    """
    start = time.time()
    deadline = None if max_seconds is None else start + max_seconds
    settings = dict(
        mip_rel_gap=mip_rel_gap,
        mip_abs_gap=mip_abs_gap,
        max_nodes=max_nodes,
        deadline=deadline,
        branching=branching,
        node_selection=node_selection,
        integrality_tolerance=integrality_tolerance,
        feasibility_tolerance=feasibility_tolerance,
        lp_tolerance=lp_tolerance,
        node_max_epochs=node_max_epochs,
        root_max_epochs=root_max_epochs,
        infeasibility_tolerance=infeasibility_tolerance,
        heuristic_every=heuristic_every,
        propagate=propagate,
        coefficient_tightening=coefficient_tightening,
        cuts=cuts,
        cut_rounds=cut_rounds,
        max_cuts_per_round=max_cuts_per_round,
        cut_min_progress=cut_min_progress,
        cut_round_epochs=cut_round_epochs,
        gomory_cuts=gomory_cuts,
        aggregate_integers=aggregate_integers,
        tree_cuts=tree_cuts,
        tree_cut_every=tree_cut_every,
        fix_and_propagate=fix_and_propagate,
        feasibility_jump=feasibility_jump,
        jump_iterations=jump_iterations,
        repair=repair,
        repair_max_epochs=repair_max_epochs,
        highs_incumbent_seconds=highs_incumbent_seconds,
        rins_seconds=rins_seconds,
        rins_every=rins_every,
        rins_solver=rins_solver,
        rins_nodes=rins_nodes,
        rens_seconds=rens_seconds,
        rens_nodes=rens_nodes,
        dive_every=dive_every,
        dive_depth=dive_depth,
        plunge_depth=plunge_depth,
        batch_size=batch_size,
        feasibility_pump=feasibility_pump,
        pump_iterations=pump_iterations,
        pump_tolerance=pump_tolerance,
        heuristic_seconds=heuristic_seconds,
        lp_options=lp_options,
        verbose=verbose,
        log_every_seconds=log_every_seconds,
        start=start,
        device=device,
    )
    if presolve:
        return _solve_with_presolve(model, highs_options, feasibility_tolerance, settings)
    mip = model if isinstance(model, MIP) else from_highs(model, highs_options=highs_options)
    return _BranchAndBound(mip, **settings).run()


def _solve_with_presolve(model, highs_options, feasibility_tolerance, settings):
    """``solve(..., presolve=True)``: HiGHS presolve, branch and bound on the
    reduced MIP, HiGHS postsolve of the incumbent."""
    import highspy

    start = settings["start"]
    highs = _highs_mip(model, highs_options)
    original = _mip_from_highs_lp(highs.getLp())
    status = _presolve(highs)
    PS = highspy.HighsPresolveStatus
    if settings["verbose"]:
        print(f"HiGHS presolve: {status.name}")

    def settled(status_name):
        return {
            "x": None,
            "objective": None,
            "bound": None,
            "gap": np.inf,
            "status": status_name,
            "converged": False,
            "bound_rigorous": True,
            "nodes": 0,
            "infeasible_nodes": 0,
            "unresolved_nodes": 0,
            "lp_epochs": 0,
            "seconds": time.time() - start,
            "history": [],
            "cuts": {"gomory": 0, "cover": 0, "mir": 0},
            "root_bound_before_cuts": None,
            "heuristic_seconds": 0.0,
            "batches": 0,
            "objective_step": None,
            "reduced_cost_tightenings": 0,
            "children_pruned": 0,
            "lp_repairs": 0,
            "lp_repairs_skipped": 0,
            "coefficients_tightened": 0,
            "tree_cuts_added": 0,
            "rins_calls": 0,
            "presolve": status.name,
        }

    if status == PS.kInfeasible:
        return settled("infeasible")
    if status == PS.kUnboundedOrInfeasible:
        return settled("unbounded_or_infeasible")
    if status == PS.kNotReduced:
        result = _BranchAndBound(original, **settings).run()
        result["presolve"] = status.name
        return result

    if status == PS.kReducedToEmpty:
        reduced_x = np.zeros(0)
        result = settled("optimal")
    else:
        reduced = _mip_from_highs_lp(highs.getPresolvedLp(), maximise=original.maximise)
        if settings["verbose"]:
            print(
                f"Reduced MIP: {reduced.num_variables} variables, "
                f"{reduced.A.shape[0]} rows (from {original.num_variables}, "
                f"{original.A.shape[0]})"
            )
        result = _BranchAndBound(reduced, **settings).run()
        result["presolve"] = status.name
        reduced_x = result["x"]
    if reduced_x is None:
        return result

    solution = highspy.HighsSolution()
    solution.col_value = list(np.asarray(reduced_x, np.float64))
    solution.value_valid = True
    solution.dual_valid = False
    highs.postsolve(solution)
    x = np.array(highs.getSolution().col_value, dtype=np.float64)
    x[original.integrality] = np.round(x[original.integrality])
    result["x"] = x
    result["objective"] = original.objective(x)
    result["postsolve_feasible"] = original.is_feasible(x, feasibility_tolerance)
    if status == PS.kReducedToEmpty:
        result["bound"] = result["objective"]
        result["gap"] = 0.0
    result["seconds"] = time.time() - start
    return result
