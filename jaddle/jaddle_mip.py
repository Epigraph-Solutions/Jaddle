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
            # A row whose whole activity range misses its bounds.
            if np.any(
                (n_min_inf == 0)
                & (min_act > self.row_upper + tol * (1.0 + np.abs(self.row_upper)))
            ) or np.any(
                (n_max_inf == 0)
                & (max_act < self.row_lower - tol * (1.0 + np.abs(self.row_lower)))
            ):
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
        return lower, upper, False


def propagate_bounds(
    A, row_lower, row_upper, lower, upper, integrality, max_passes=5, tol=1e-6
):
    """Activity-based bound tightening (see ``BoundPropagator``) for
    ``row_lower <= A x <= row_upper, lower <= x <= upper``, with integer
    bounds rounded. Returns ``(lower, upper, infeasible)``."""
    propagator = BoundPropagator(A, row_lower, row_upper, integrality, tol=tol)
    return propagator(lower, upper, max_passes=max_passes)


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
    y = np.array(y, dtype=np.float64)
    y[n_eq:] = np.maximum(y[n_eq:], 0.0)
    r = c + (A_T @ y if A_T is not None else A.T @ y)
    with np.errstate(invalid="ignore"):
        term = np.where(r > 0, r * lower, np.where(r < 0, r * upper, 0.0))
    bad = ~np.isfinite(term)
    if bad.any():
        if np.any(np.abs(r[bad]) > reduced_cost_tolerance):
            return -np.inf, True
        rigorous = False
    else:
        rigorous = True
    return float(-(b @ y) + term[~bad].sum()), rigorous


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
        lp_options,
        verbose,
        log_every_seconds,
        start,
    ):
        if branching not in ("pseudocost", "most_fractional"):
            raise ValueError(f"unknown branching rule {branching!r}")
        if node_selection not in ("hybrid", "best_first", "depth_first"):
            raise ValueError(f"unknown node selection {node_selection!r}")
        lp_options = dict(lp_options or {})
        reserved = [key for key in _LP_RESERVED if key in lp_options]
        if reserved:
            raise ValueError(f"lp_options cannot set {', '.join(reserved)}")

        self.mip = mip
        self.rel_gap, self.abs_gap = mip_rel_gap, mip_abs_gap
        self.max_nodes, self.deadline = max_nodes, deadline
        self.branching, self.node_selection = branching, node_selection
        self.int_tol = integrality_tolerance
        self.feas_tol = feasibility_tolerance
        self.node_max_epochs, self.root_max_epochs = node_max_epochs, root_max_epochs
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
        self.A_T = mip.A.T.tocsr()
        self.m_ineq = mip.A.shape[0] - mip.n_eq
        self.values = mip.lp.values()
        self.dtype = mip.lp.c.dtype
        self.propagator = (
            BoundPropagator(mip.A, mip.row_lower, mip.row_upper, self.is_int)
            if propagate
            else None
        )

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

    # -- bookkeeping ---------------------------------------------------------
    def seconds_left(self):
        return None if self.deadline is None else self.deadline - time.time()

    def out_of_time(self):
        return self.deadline is not None and time.time() >= self.deadline

    def cutoff(self):
        if self.incumbent is None:
            return np.inf
        value = self.incumbent[0] + self.mip.offset
        return self.incumbent[0] - max(self.abs_gap, self.rel_gap * abs(value))

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
        return (entry[2] for entry in self.open)

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

    def lp_bound(self, solution, lower, upper):
        """The Lagrangian bound of a node's duals over its propagated box,
        widened a little on continuous variables against rounding error."""
        cont = ~self.is_int
        lower = np.where(cont, lower - 1e-7 * (1.0 + np.abs(lower)), lower)
        upper = np.where(cont, upper + 1e-7 * (1.0 + np.abs(upper)), upper)
        return lagrangian_bound(
            self.mip.c,
            self.mip.A,
            self.mip.b,
            self.mip.n_eq,
            lower,
            upper,
            self.duals(solution),
            A_T=self.A_T,
            reduced_cost_tolerance=self.reduced_cost_tol,
        )

    # -- incumbents ----------------------------------------------------------
    def try_incumbent(self, x, source, warm):
        """Round ``x``'s integers and make it the incumbent if it is feasible
        and better. With continuous variables, a rounded point that is not
        feasible is repaired by fixing the integers and re-solving the LP for
        the rest. Returns whether a feasible point was found."""
        mip = self.mip
        candidate = np.clip(np.asarray(x, np.float64), mip.lower, mip.upper)
        candidate[self.is_int] = np.round(candidate[self.is_int])
        if not mip.is_feasible(candidate, self.feas_tol):
            if not self.has_continuous or self.out_of_time():
                return False
            fixed_lower = np.where(self.is_int, candidate, mip.lower)
            fixed_upper = np.where(self.is_int, candidate, mip.upper)
            if np.any(fixed_lower > fixed_upper):
                return False
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
                self.node_max_epochs,
                primal_feasibility_tolerance=tight,
                termination_norm="inf",
            )
            candidate = np.clip(
                np.asarray(result["solution"].primal, np.float64),
                fixed_lower,
                fixed_upper,
            )
            candidate[self.is_int] = fixed_lower[self.is_int]
            if not mip.is_feasible(candidate, self.feas_tol):
                return False
        value = float(mip.c @ candidate)
        if self.incumbent is not None and value >= self.incumbent[0] - 1e-12 * (
            1.0 + abs(self.incumbent[0])
        ):
            return True
        switch = self.incumbent is None
        self.incumbent = (value, candidate)
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
        lower, upper = self.root_lower.copy(), self.root_upper.copy()
        if changes is not None:
            idx, lo, up = changes
            lower[idx], upper[idx] = lo, up
        return lower, upper

    def compress(self, lower, upper):
        idx = np.flatnonzero((lower != self.root_lower) | (upper != self.root_upper))
        return idx, lower[idx], upper[idx]

    def branch(self, node, lower, upper, x, var, bound, value, warm):
        xv = x[var]
        children = []
        for down in (True, False):
            lo, up = lower.copy(), upper.copy()
            if down:
                up[var] = math.floor(xv)
            else:
                lo[var] = math.ceil(xv)
            if lo[var] > up[var]:
                continue
            if self.propagator is not None:
                lo, up, infeasible = self.propagator(lo, up)
                if infeasible:
                    self.infeasible_nodes += 1
                    continue
            distance = xv - math.floor(xv) if down else math.ceil(xv) - xv
            children.append(
                _Node(
                    bound,
                    node.depth + 1,
                    self.compress(lo, up),
                    warm,
                    (var, down, distance, value),
                )
            )
        # In a dive the child in x's rounding direction goes first (pushed last).
        prefer_up = xv - math.floor(xv) >= 0.5
        children.sort(key=lambda child: child.branch[1] == prefer_up)
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
        bound, rigorous = self.lp_bound(result["solution"], lower, upper)
        bound = max(bound, node.bound)
        self.active_bound = bound
        if not result["converged"] and bound < self.cutoff() and not self.out_of_time():
            # An LP that will not converge is often infeasible.
            detection = self.detect(lp, result, self.node_max_epochs)
            if detection["status"] == "primal_infeasible":
                self.infeasible_nodes += 1
                return
            result = detection["result"]
            new_bound, new_rigorous = self.lp_bound(result["solution"], lower, upper)
            if new_bound > bound:
                bound, rigorous = new_bound, new_rigorous
                self.active_bound = bound
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
            self.try_incumbent(x, "H", warm)
            if bound >= self.cutoff():
                self.close(bound, rigorous)
                return
        var = self.choose_variable(x, fractional)
        self.branch(node, lower, upper, x, var, bound, value, warm)

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
        }

    def run(self):
        mip = self.mip
        lower, upper = mip.lower.copy(), mip.upper.copy()
        if np.any(lower > upper):
            return self.result("infeasible")
        if self.propagator is not None:
            lower, upper, infeasible = self.propagator(lower, upper, max_passes=20)
            if infeasible:
                return self.result("infeasible")
        self.root_lower, self.root_upper = lower, upper
        if self.verbose:
            print(
                f"MIP: {mip.num_variables} variables ({mip.num_integers} integer), "
                f"{mip.A.shape[0]} rows, {mip.A.nnz} nonzeros"
            )

        root = _Node(-np.inf, 0, None, None, None)
        lp = self.node_lp(lower, upper)
        detection = self.detect(lp, None, self.root_max_epochs)
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
            print(
                f"  {'Nodes':>8} {'Open':>8} {'Depth':>6} {'Bound':>13} "
                f"{'Incumbent':>13} {'Gap':>9} {'Time':>9}"
            )
        if result["stop_reason"] == "interrupted":
            return self.result("interrupted")
        if result["stop_reason"] == "time_limit" or self.out_of_time():
            bound, rigorous = self.lp_bound(result["solution"], lower, upper)
            self.close(bound, rigorous)
            return self.result("time_limit")
        self.evaluate(root, lower, upper, lp, result)

        status = "optimal"
        try:
            while self.open:
                if self.out_of_time():
                    status = "time_limit"
                    break
                if self.max_nodes is not None and self.nodes >= self.max_nodes:
                    status = "node_limit"
                    break
                if self.incumbent is not None and self.nodes % 20 == 0:
                    if self.global_bound() >= self.cutoff():
                        break
                _, _, node = heapq.heappop(self.open)
                if node.bound >= self.cutoff():
                    self.close(node.bound)
                    continue
                lower, upper = self.expand(node.changes)
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
    presolve=False,
    highs_options=None,
    lp_options=None,
    verbose=False,
    log_every_seconds=1.0,
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
    * **Nodes.** Each child is warm-started from its parent's primal-dual
      point and step-size state; with the compiled solver cached, a small
      node takes milliseconds. A node is pruned when its Lagrangian bound
      (``lagrangian_bound``) reaches the cutoff
      ``incumbent − max(mip_abs_gap, mip_rel_gap·|incumbent|)``. A node LP
      that does not converge in ``node_max_epochs`` gets an infeasibility
      check of the same length, and is pruned if it is infeasible.
    * **Branching** (``branching``): ``"pseudocost"`` (default) scores each
      fractional variable by the product of its estimated down and up bound
      gains, learnt from the nodes solved so far (most fractional until
      there are any); ``"most_fractional"`` picks the variable nearest
      ½. Each child's bounds are propagated, so many infeasible children
      are dropped without an LP solve.
    * **Node selection** (``node_selection``): ``"hybrid"`` (default) dives
      depth first until the first incumbent, then takes the best bound;
      ``"best_first"`` and ``"depth_first"`` do one throughout.
    * **Incumbents.** A node LP point with integral integers (within
      ``integrality_tolerance``), and the LP point rounded at every
      ``heuristic_every``-th node, are rounded and checked. A point is
      feasible when its row violation relative to ``1 + ‖b‖∞`` and its bound
      violation are at most ``feasibility_tolerance``. With continuous
      variables, a rounded point that is not is repaired by fixing the
      integers and solving the LP in the rest.

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
      ``objective``, ``bound``, ``source``: ``"H"`` heuristic or ``"T"`` tree).
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
        lp_options=lp_options,
        verbose=verbose,
        log_every_seconds=log_every_seconds,
        start=start,
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
