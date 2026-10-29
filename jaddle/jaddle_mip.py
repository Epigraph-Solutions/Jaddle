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

import collections
import functools
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
from jaddle import jaddle_optimisers as jo
from jaddle.jaddle_basic_types import LP, JaddleLP, _wide

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


def _propagation_data(A, row_lower, row_upper, integrality):
    """The rows of ``BoundPropagator`` as float64 device arrays, for
    ``_propagate_jax`` and ``_lagrangian_terms_jax``. Call under x64."""
    A = sp.csc_matrix(A)
    A.eliminate_zeros()
    A.sort_indices()
    m, n = A.shape
    cols = np.repeat(np.arange(n), np.diff(A.indptr))
    return {
        "rows": jnp.asarray(A.indices, jnp.int32),
        "cols": jnp.asarray(cols, jnp.int32),
        "vals": jnp.asarray(A.data, jnp.float64),
        "row_lower": jnp.asarray(row_lower, jnp.float64),
        "row_upper": jnp.asarray(row_upper, jnp.float64),
        "is_int": jnp.asarray(integrality, bool),
        "has_entries": jnp.asarray(np.diff(A.indptr) > 0),
    }


def _propagate_jax(data, lower, upper, m, max_passes=5, tol=1e-6):
    """``BoundPropagator.__call__`` in JAX, for one box (``vmap`` it over
    many): the same passes, roundings, margins and infeasibility tests.
    Returns ``(lower, upper, infeasible)``. Runs in float64 (under x64)."""
    rows, cols, vals = data["rows"], data["cols"], data["vals"]
    is_int, has_entries = data["is_int"], data["has_entries"]
    row_lower, row_upper = data["row_lower"], data["row_upper"]
    n = lower.shape[0]
    if vals.shape[0] == 0:
        return lower, upper, jnp.any(lower > upper + tol)
    positive = vals > 0
    rl, ru = row_lower[rows], row_upper[rows]
    sign = jnp.sign(vals)

    def activities(lo, up):
        lc, uc = lo[cols], up[cols]
        min_c = jnp.where(positive, vals * lc, vals * uc)
        max_c = jnp.where(positive, vals * uc, vals * lc)
        min_inf, max_inf = ~jnp.isfinite(min_c), ~jnp.isfinite(max_c)
        seg = functools.partial(jax.ops.segment_sum, segment_ids=rows, num_segments=m)
        min_act = seg(jnp.where(min_inf, 0.0, min_c))
        max_act = seg(jnp.where(max_inf, 0.0, max_c))
        n_min_inf = seg(min_inf.astype(jnp.int32))
        n_max_inf = seg(max_inf.astype(jnp.int32))
        return min_c, max_c, min_inf, max_inf, min_act, max_act, n_min_inf, n_max_inf

    def missed(min_act, max_act, n_min_inf, n_max_inf):
        return jnp.any(
            (n_min_inf == 0) & (min_act > row_upper + tol * (1.0 + jnp.abs(row_upper)))
        ) | jnp.any(
            (n_max_inf == 0) & (max_act < row_lower - tol * (1.0 + jnp.abs(row_lower)))
        )

    def body(state):
        lo, up, passes, done, infeasible = state
        min_c, max_c, min_inf, max_inf, min_act, max_act, n_min, n_max = activities(lo, up)
        miss = missed(min_act, max_act, n_min, n_max)
        nmin, nmax = n_min[rows], n_max[rows]
        rest_min = jnp.where(
            min_inf,
            jnp.where(nmin == 1, min_act[rows], -jnp.inf),
            jnp.where(nmin == 0, min_act[rows] - min_c, -jnp.inf),
        )
        rest_max = jnp.where(
            max_inf,
            jnp.where(nmax == 1, max_act[rows], jnp.inf),
            jnp.where(nmax == 0, max_act[rows] - max_c, jnp.inf),
        )
        from_upper = (ru - rest_min) / vals
        from_lower = (rl - rest_max) / vals
        from_upper = jnp.where(jnp.isnan(from_upper), jnp.inf * sign, from_upper)
        from_lower = jnp.where(jnp.isnan(from_lower), -jnp.inf * sign, from_lower)
        ub_cand = jnp.where(positive, from_upper, from_lower)
        lb_cand = jnp.where(positive, from_lower, from_upper)
        new_upper = jax.ops.segment_min(ub_cand, cols, num_segments=n)
        new_lower = jax.ops.segment_max(lb_cand, cols, num_segments=n)
        new_upper = jnp.where(has_entries, new_upper, jnp.inf)
        new_lower = jnp.where(has_entries, new_lower, -jnp.inf)
        new_upper = jnp.where(is_int, jnp.floor(new_upper + _ROUND_SLACK), new_upper)
        new_lower = jnp.where(is_int, jnp.ceil(new_lower - _ROUND_SLACK), new_lower)

        sane_u = jnp.abs(new_upper) < 1e12
        sane_l = jnp.abs(new_lower) < 1e12
        better_u = sane_u & (
            ~jnp.isfinite(up) | (new_upper < up - tol * (1.0 + jnp.abs(up)))
        )
        better_l = sane_l & (
            ~jnp.isfinite(lo) | (new_lower > lo + tol * (1.0 + jnp.abs(lo)))
        )
        improved = jnp.any(better_u) | jnp.any(better_l)
        up2 = jnp.where(better_u, new_upper, up)
        lo2 = jnp.where(better_l, new_lower, lo)
        crossed = lo2 > up2
        bad = jnp.any(crossed & is_int) | jnp.any(lo2 > up2 + tol * (1.0 + jnp.abs(up2)))
        mid = 0.5 * (lo2 + up2)
        lo3 = jnp.where(crossed, mid, lo2)
        up3 = jnp.where(crossed, mid, up2)
        # As the numpy loop: a missed row returns the box as it was; no
        # improvement stops; a bad crossing returns the crossed box.
        keep = miss | ~improved
        new_lo = jnp.where(keep, lo, jnp.where(bad, lo2, lo3))
        new_up = jnp.where(keep, up, jnp.where(bad, up2, up3))
        return (
            new_lo,
            new_up,
            passes + 1,
            keep | bad,
            miss | (improved & bad),
        )

    def cond(state):
        return (state[2] < max_passes) & ~state[3]

    lo, up, _, done, infeasible = jax.lax.while_loop(
        cond, body, (lower, upper, jnp.int32(0), jnp.bool_(False), jnp.bool_(False))
    )
    # The passes ran out right after a tightening: check the final box too.
    final_missed = missed(*activities(lo, up)[4:])
    infeasible = infeasible | (~done & final_missed)
    return lo, up, infeasible


def _box_bound_jax(constant, r, lower, upper, reduced_cost_tolerance):
    """``box_bound`` in JAX: ``(bound, rigorous)``."""
    term = jnp.where(r > 0, r * lower, jnp.where(r < 0, r * upper, 0.0))
    bad = ~jnp.isfinite(term)
    hard = jnp.any(bad & (jnp.abs(r) > reduced_cost_tolerance))
    value = constant + jnp.sum(jnp.where(bad, 0.0, term))
    return jnp.where(hard, -jnp.inf, value), hard | ~jnp.any(bad)


def _widen_continuous(lower, upper, is_int):
    """The box ``bound_over`` bounds over: continuous bounds widened a
    little against rounding error."""
    lower = jnp.where(is_int, lower, lower - 1e-7 * (1.0 + jnp.abs(lower)))
    upper = jnp.where(is_int, upper, upper + 1e-7 * (1.0 + jnp.abs(upper)))
    return lower, upper


class ProbingCache:
    """
    What probing learnt about each probed integer variable ``j``: its split
    ``s`` (the sides are ``x_j <= s`` and ``x_j >= s + 1``) and, for each
    side, the bounds propagation implies there, as ``(idx, lower, upper)``
    over the variables that change (``None``: the side is infeasible). The
    implications hold for every feasible point on that side, so they can
    be intersected into any box (``implied``). On the device backend each
    side keeps at most ``max_implications`` of them.
    """

    def __init__(self):
        self.split = {}
        self.sides = {}

    def __len__(self):
        return len(self.split)

    def add(self, var, split, down, up):
        self.split[int(var)] = float(split)
        self.sides[int(var)] = (down, up)

    def implied(self, var, value, lower, upper):
        """``[lower, upper]`` intersected with what ``x_var = value``
        implies (in place). Returns False when that side is infeasible."""
        split = self.split.get(int(var))
        if split is None:
            return True
        side = self.sides[int(var)][0 if value <= split else 1]
        if side is None:
            return False
        idx, lo, up = side
        np.maximum.at(lower, idx, lo)
        np.minimum.at(upper, idx, up)
        return True


class ProbeResult(collections.namedtuple(
    "ProbeResult", "lower upper infeasible fixed tightened probed cache"
)):
    """``probe``'s result: the tightened box, whether probing proved the
    box empty, the integer variables it fixed, the bounds it tightened
    (fixings included), the variables probed and the ``ProbingCache``."""

    __slots__ = ()


def _probe_order(A, row_lower, row_upper, integrality, lower, upper, tol=1e-6):
    """The integer variables to probe, most conflict-prone first, as the
    cuOpt paper ranks them: the rows a variable violates when it alone
    moves to the other end of its bounds (the others at their least
    activity), then the largest such violation, then its column's length."""
    propagator = BoundPropagator(A, row_lower, row_upper, integrality, tol=tol)
    free = np.flatnonzero(integrality & (lower < upper))
    if propagator.vals.size == 0 or free.size == 0:
        return free
    min_c, max_c, min_inf, max_inf, min_act, max_act, n_min, n_max = (
        propagator._activities(np.asarray(lower, np.float64), np.asarray(upper, np.float64))
    )
    rows, cols = propagator.rows, propagator.cols
    ru, rl = propagator.row_upper[rows], propagator.row_lower[rows]
    with np.errstate(invalid="ignore"):
        flipped_min = min_act[rows] - np.where(min_inf, 0.0, min_c) + max_c
        flipped_max = max_act[rows] - np.where(max_inf, 0.0, max_c) + min_c
        over = np.where(
            (n_min[rows] - min_inf == 0) & np.isfinite(flipped_min), flipped_min - ru, -np.inf
        )
        under = np.where(
            (n_max[rows] - max_inf == 0) & np.isfinite(flipped_max), rl - flipped_max, -np.inf
        )
    violation = np.maximum(over, under)
    violated = violation > tol * (1.0 + np.abs(np.where(over >= under, ru, rl)))
    n = A.shape[1]
    count = np.bincount(cols, violated, n)
    worst = np.full(n, -np.inf)
    np.maximum.at(worst, cols, np.where(violated, violation, -np.inf))
    degree = np.bincount(cols, minlength=n)
    order = np.lexsort((-degree[free], -worst[free], -count[free]))
    return free[order]


def _probe_split(lower, upper):
    """Each variable's split ``s``: ``x <= s`` or ``x >= s + 1``. A binary
    splits at 0, a boxed integer at its midpoint, a one-sided one at its
    finite bound (``x = l`` or ``x >= l + 1``)."""
    lo_ok, up_ok = np.isfinite(lower), np.isfinite(upper)
    mid = np.floor(0.5 * (np.where(lo_ok, lower, 0.0) + np.where(up_ok, upper, 0.0)))
    return np.where(
        lo_ok & up_ok,
        np.minimum(mid, upper - 1.0),
        np.where(lo_ok, lower, np.where(up_ok, upper - 1.0, 0.0)),
    )


def probe(
    A,
    row_lower,
    row_upper,
    integrality,
    lower,
    upper,
    max_variables=None,
    deadline=None,
    backend="auto",
    max_implications=64,
    tol=1e-6,
    max_useless=None,
):
    """
    Probing on the integer variables (Savelsbergh, 1994), as in cuOpt's
    probing cache (Çördük et al., 2025): each variable is split in two
    (``_probe_split``), and bound propagation runs on both sides. A side
    that propagation empties fixes the variable to the other side, and
    every bound the other side implies holds everywhere; when both sides
    survive, every variable's bounds tighten to the hull of the two. Both
    sides empty proves the box infeasible.

    Variables go most conflict-prone first (``_probe_order``), at most
    ``max_variables`` of them, until ``deadline`` (a ``time.time()`` value),
    and probing stops once ``max_useless`` variables in a row (``None``: no
    limit) have fixed or tightened nothing, as SCIP's probing does: on
    fiball all 32,399 variables took 11 s for one tightened bound.
    ``backend`` is ``"numpy"`` (one ``BoundPropagator`` call per side, each
    from the box tightened so far), ``"jax"`` (chunks of variables at once,
    both sides of each propagated by ``_propagate_jax`` in one vmapped call
    of fixed shape) or ``"auto"`` (numpy up to 10,000 nonzeros).

    Returns a ``ProbeResult``.
    """
    lower = np.array(lower, np.float64)
    upper = np.array(upper, np.float64)
    integrality = np.asarray(integrality, bool)
    start_lower, start_upper = lower.copy(), upper.copy()
    cache = ProbingCache()
    propagator = BoundPropagator(A, row_lower, row_upper, integrality, tol=tol)
    lower, upper, infeasible = propagator(lower, upper)

    def result(infeasible, probed):
        tightened = int(np.count_nonzero((lower > start_lower) | (upper < start_upper)))
        fixed = int(np.count_nonzero(integrality & (lower == upper) & (start_lower < start_upper)))
        return ProbeResult(lower, upper, infeasible, fixed, tightened, probed, cache)

    if infeasible:
        return result(True, 0)
    order = _probe_order(A, row_lower, row_upper, integrality, lower, upper, tol)
    if max_variables is not None:
        order = order[: int(max_variables)]
    if backend == "auto":
        backend = "numpy" if A.nnz <= _SMALL_MIP_NNZ else "jax"
    if backend not in ("numpy", "jax"):
        raise ValueError(f"backend must be 'auto', 'numpy' or 'jax', not {backend!r}")
    probed = 0
    useless = 0  # variables probed in a row that changed no bound

    def progress(old_lower, old_upper):
        return bool(np.any(lower != old_lower) or np.any(upper != old_upper))

    if backend == "numpy":
        for var in order:
            if deadline is not None and time.time() >= deadline:
                break
            if max_useless is not None and useless >= max_useless:
                break
            before = (lower.copy(), upper.copy())
            if lower[var] >= upper[var]:
                continue
            split = float(_probe_split(lower[var : var + 1], upper[var : var + 1])[0])
            sides = []
            for down in (True, False):
                lo, up = lower.copy(), upper.copy()
                if down:
                    up[var] = min(up[var], split)
                else:
                    lo[var] = max(lo[var], split + 1.0)
                lo, up, empty = propagator(lo, up)
                sides.append(None if empty else (lo, up))
            probed += 1
            down, up_side = sides
            if down is None and up_side is None:
                return result(True, probed)
            if down is None or up_side is None:
                lower, upper = up_side if down is None else down
            else:
                lower = np.maximum(lower, np.minimum(down[0], up_side[0]))
                upper = np.minimum(upper, np.maximum(down[1], up_side[1]))
                lower, upper, empty = propagator(lower, upper)
                if empty:
                    return result(True, probed)
            useless = 0 if progress(*before) else useless + 1

            def changes(side):
                if side is None:
                    return None
                idx = np.flatnonzero((side[0] > start_lower) | (side[1] < start_upper))
                return idx, side[0][idx], side[1][idx]

            cache.add(var, split, changes(down), changes(up_side))
        return result(False, probed)

    # The device backend: chunks of ``size`` variables, both sides at once.
    m, n = A.shape
    width = max(n, A.nnz, 1)
    size = 1
    while size < 256 and 4 * size * width <= 2**24:
        size *= 2
    k = int(min(max_implications, n))
    with jax.enable_x64(True):
        data = _propagation_data(A, row_lower, row_upper, integrality)
        chunk = jax.jit(
            functools.partial(_probe_chunk_jax, m=m, k=k)
        )
        for top in range(0, order.size, size):
            if deadline is not None and time.time() >= deadline:
                break
            if max_useless is not None and useless >= max_useless:
                break
            before = (lower.copy(), upper.copy())
            cols = order[top : top + size]
            cols = cols[lower[cols] < upper[cols]]
            if cols.size == 0:
                continue
            count = cols.size
            padded = np.zeros(size, np.int32)
            padded[:count] = cols
            valid = np.zeros(size, bool)
            valid[:count] = True
            split = np.zeros(size)
            split[:count] = _probe_split(lower[cols], upper[cols])
            out = jax.device_get(
                chunk(data, jnp.asarray(lower), jnp.asarray(upper), padded, split, valid)
            )
            probed += count
            if bool(out["both_empty"]):
                return result(True, probed)
            new_lower, new_upper = np.asarray(out["lower"]), np.asarray(out["upper"])
            if np.any(new_lower > new_upper):
                return result(True, probed)
            if np.any(new_lower != lower) or np.any(new_upper != upper):
                lower, upper, empty = propagator(new_lower, new_upper)
                if empty:
                    return result(True, probed)
            useless = 0 if progress(*before) else useless + count
            for i in range(count):
                sides = []
                for side in range(2):
                    if out["empty"][i, side]:
                        sides.append(None)
                        continue
                    kept = min(int(out["count"][i, side]), k)
                    idx = np.asarray(out["idx"][i, side, :kept], np.int64)
                    sides.append(
                        (idx, out["implied_lower"][i, side, :kept], out["implied_upper"][i, side, :kept])
                    )
                cache.add(cols[i], split[i], sides[0], sides[1])
    return result(False, probed)


def _probe_chunk_jax(data, lower, upper, cols, split, valid, m, k):
    """Both sides of each variable in ``cols`` propagated at once; returns
    the box that all of them together imply, whether some variable has both
    sides empty, and each side's emptiness and (at most ``k``) implied
    bounds. Runs under x64."""
    n = lower.shape[0]

    def sides(j, s):
        up_down = upper.at[j].min(s)
        lo_up = lower.at[j].max(s + 1.0)
        return jnp.stack([lower, lo_up]), jnp.stack([up_down, upper])

    lo, up = jax.vmap(sides)(cols, split)
    lo, up = lo.reshape(-1, n), up.reshape(-1, n)
    empty0 = jnp.any(lo > up, axis=1)
    lo, up, empty = jax.vmap(lambda a, b: _propagate_jax(data, a, b, m))(lo, up)
    empty = (empty | empty0).reshape(-1, 2) & valid[:, None]
    lo, up = lo.reshape(-1, 2, n), up.reshape(-1, 2, n)
    e0, e1 = empty[:, 0][:, None], empty[:, 1][:, None]
    implied_lo = jnp.where(e0, lo[:, 1], jnp.where(e1, lo[:, 0], jnp.minimum(lo[:, 0], lo[:, 1])))
    implied_up = jnp.where(e0, up[:, 1], jnp.where(e1, up[:, 0], jnp.maximum(up[:, 0], up[:, 1])))
    both = empty[:, 0] & empty[:, 1]
    use = (valid & ~both)[:, None]
    new_lower = jnp.maximum(lower, jnp.max(jnp.where(use, implied_lo, -jnp.inf), axis=0))
    new_upper = jnp.minimum(upper, jnp.min(jnp.where(use, implied_up, jnp.inf), axis=0))
    changed = (lo > lower) | (up < upper)
    idx = jax.vmap(jax.vmap(lambda c: jnp.nonzero(c, size=k, fill_value=0)[0]))(changed)
    return {
        "lower": new_lower,
        "upper": new_upper,
        "both_empty": jnp.any(both & valid),
        "empty": empty,
        "count": changed.sum(axis=-1),
        "idx": idx,
        "implied_lower": jnp.take_along_axis(lo, idx, axis=-1),
        "implied_upper": jnp.take_along_axis(up, idx, axis=-1),
    }


class _ChildKernels:
    """
    The children of a whole batch of nodes on the device, for batched tree
    search: for each member's branching variable, both child boxes are
    built, propagated (``_propagate_jax``, the host ``BoundPropagator``'s
    rules) and bounded by the member's Lagrangian terms over them
    (``_box_bound_jax`` over ``_widen_continuous`` boxes, as
    ``bound_over``). One jitted call of fixed shape (``size`` members, two
    children each), so it compiles once per MIP.
    """

    def __init__(self, mip, propagate, reduced_cost_tolerance, size):
        with jax.enable_x64(True):
            self.data = _propagation_data(
                mip.A, mip.row_lower, mip.row_upper, mip.integrality
            )
        self.m, self.n = mip.A.shape
        self.propagate = bool(propagate)
        self.rc_tol = float(reduced_cost_tolerance)
        self.size = int(size)
        self._children = jax.jit(self._children_impl)

    def __call__(self, lower, upper, var, xv, r, constant):
        """Host arrays in (``size`` rows each), host arrays out: ``lower``,
        ``upper`` (``size, 2, n``), ``crossed`` (the branching emptied the
        box before propagation), ``empty``, ``bound``, ``rigorous``
        (``size, 2``); side 0 is the down child."""
        with jax.enable_x64(True):
            return jax.device_get(
                self._children(self.data, lower, upper, var, xv, r, constant)
            )

    def _children_impl(self, data, lower, upper, var, xv, r, constant):
        f64 = jnp.float64
        lower, upper = lower.astype(f64), upper.astype(f64)
        n = self.n

        def sides(lo, up, j, v):
            up_down = up.at[j].min(jnp.floor(v))
            lo_up = lo.at[j].max(jnp.ceil(v))
            return jnp.stack([lo, lo_up]), jnp.stack([up_down, up])

        lo, up = jax.vmap(sides)(lower, upper, var, xv.astype(f64))  # (B, 2, n)
        at = var[:, None, None]
        crossed = (
            jnp.take_along_axis(lo, jnp.broadcast_to(at, (lo.shape[0], 2, 1)), axis=2)
            > jnp.take_along_axis(up, jnp.broadcast_to(at, (up.shape[0], 2, 1)), axis=2)
        )[..., 0]
        lo, up = lo.reshape(-1, n), up.reshape(-1, n)
        if self.propagate:
            lo, up, empty = jax.vmap(lambda a, b: _propagate_jax(data, a, b, self.m))(lo, up)
        else:
            empty = jnp.any(lo > up, axis=1)
        wlo, wup = jax.vmap(_widen_continuous, in_axes=(0, 0, None))(lo, up, data["is_int"])
        bound, rigorous = jax.vmap(_box_bound_jax, in_axes=(0, 0, 0, 0, None))(
            jnp.repeat(constant.astype(f64), 2),
            jnp.repeat(r.astype(f64), 2, axis=0),
            wlo,
            wup,
            self.rc_tol,
        )
        return {
            "lower": lo.reshape(-1, 2, n),
            "upper": up.reshape(-1, 2, n),
            "crossed": crossed,
            "empty": empty.reshape(-1, 2),
            "bound": bound.reshape(-1, 2),
            "rigorous": rigorous.reshape(-1, 2),
        }


class _StrongKernels:
    """
    Device kernels for strong branching on one MIP (``_BranchAndBound``
    with ``strong_backend="jax"``), jitted once for its shapes:

    * ``screen``: both children of up to ``screen_size`` candidates, built,
      propagated (``_propagate_jax``) and bounded by the parent's duals over
      their boxes, then the ``k`` best eligible candidates picked for child
      LPs (by the product of their bound gains, each at least its
      pseudocost estimate);
    * ``child_bounds``: the Lagrangian bound of each child LP's duals over
      its box (``lagrangian_terms`` and ``bound_over`` on the device).

    Everything stays on the device, so the host can dispatch the whole
    pipeline and read it back later.
    """

    def __init__(self, mip, propagate, reduced_cost_tolerance, screen_size, k, lp_dtype):
        self.lp_dtype = lp_dtype
        with jax.enable_x64(True):
            self.data = _propagation_data(
                mip.A, mip.row_lower, mip.row_upper, mip.integrality
            )
            self.data["c"] = jnp.asarray(mip.c, jnp.float64)
            self.data["b"] = jnp.asarray(mip.b, jnp.float64)
            self.data["lower"] = jnp.asarray(mip.lower, jnp.float64)
            self.data["upper"] = jnp.asarray(mip.upper, jnp.float64)
        self.m, self.n = mip.A.shape
        self.n_eq = mip.n_eq
        self.propagate = bool(propagate)
        self.rc_tol = float(reduced_cost_tolerance)
        self.screen_size, self.k = int(screen_size), int(k)
        self._screen = jax.jit(self._screen_impl)
        self._child_bounds = jax.jit(self._child_bounds_impl)

    def screen(self, *args):
        with jax.enable_x64(True):
            return self._screen(self.data, *args)

    def child_bounds(self, solution, lower, upper):
        with jax.enable_x64(True):
            return self._child_bounds(
                self.data, solution.dual_eq, solution.dual_ineq, lower, upper
            )

    def _screen_impl(
        self, data, lower, upper, cand, valid, xc, r, constant,
        est_down, est_up, eligible, bound, cutoff,
    ):
        f64 = jnp.float64
        lower, upper = lower.astype(f64), upper.astype(f64)
        r, constant = r.astype(f64), constant.astype(f64)
        is_int = data["is_int"]

        def children(j, xv):
            up_down = upper.at[j].min(jnp.floor(xv))
            lo_up = lower.at[j].max(jnp.ceil(xv))
            return jnp.stack([lower, lo_up]), jnp.stack([up_down, upper])

        lo, up = jax.vmap(children)(cand, xc.astype(f64))  # (C, 2, n)
        lo, up = lo.reshape(-1, self.n), up.reshape(-1, self.n)
        empty = jnp.any(lo > up, axis=1)
        if self.propagate:
            lo, up, infeasible = jax.vmap(
                lambda a, b: _propagate_jax(data, a, b, self.m)
            )(lo, up)
            infeasible = infeasible | empty
        else:
            infeasible = empty
        wlo, wup = jax.vmap(_widen_continuous, in_axes=(0, 0, None))(lo, up, is_int)
        pre, pre_rig = jax.vmap(_box_bound_jax, in_axes=(None, None, 0, 0, None))(
            constant, r, wlo, wup, self.rc_tol
        )
        pre = jnp.maximum(pre, bound)
        infeasible = infeasible.reshape(-1, 2)
        pre, pre_rig = pre.reshape(-1, 2), pre_rig.reshape(-1, 2)
        closed = infeasible | (pre >= cutoff)
        gain = jnp.maximum(pre - bound, 0.0)
        gain_down = jnp.maximum(gain[:, 0], est_down)
        gain_up = jnp.maximum(gain[:, 1], est_up)
        score = jnp.maximum(gain_down, 1e-6) * jnp.maximum(gain_up, 1e-6)
        pick = valid & eligible & ~closed.any(axis=1)
        score = jnp.where(pick, score, -jnp.inf)
        top, sel = jax.lax.top_k(score, self.k)
        lo, up = lo.reshape(-1, 2, self.n), up.reshape(-1, 2, self.n)
        sel_lo, sel_up = lo[sel].reshape(-1, self.n), up[sel].reshape(-1, self.n)
        # The child LPs' bounds: integers from the propagated boxes, continuous
        # variables as in the MIP (as ``node_lp``), in the LP's own dtype.
        lp_lo = jnp.where(is_int, sel_lo, data["lower"]).astype(self.lp_dtype)
        lp_up = jnp.where(is_int, sel_up, data["upper"]).astype(self.lp_dtype)
        return {
            "infeasible": infeasible,
            "pre": pre,
            "pre_rigorous": pre_rig,
            "lower": lo,
            "upper": up,
            "sel": sel,
            "sel_valid": jnp.isfinite(top),
            "sel_lower": sel_lo,
            "sel_upper": sel_up,
            "lp_lower": lp_lo,
            "lp_upper": lp_up,
        }

    def _child_bounds_impl(self, data, dual_eq, dual_ineq, lower, upper):
        f64 = jnp.float64
        y = jnp.concatenate(
            [
                dual_eq[:, : self.n_eq].astype(f64),
                jnp.maximum(dual_ineq[:, : self.m - self.n_eq].astype(f64), 0.0),
            ],
            axis=1,
        )
        rows, cols, vals = data["rows"], data["cols"], data["vals"]
        r = data["c"] + jax.vmap(
            lambda yy: jax.ops.segment_sum(vals * yy[rows], cols, num_segments=self.n)
        )(y)
        constant = -(y @ data["b"])
        wlo, wup = jax.vmap(_widen_continuous, in_axes=(0, 0, None))(
            lower.astype(f64), upper.astype(f64), data["is_int"]
        )
        bound, rigorous = jax.vmap(_box_bound_jax, in_axes=(0, 0, 0, 0, None))(
            constant, r, wlo, wup, self.rc_tol
        )
        return {"bound": bound, "rigorous": rigorous, "r": r, "constant": constant}


def propagate_bounds(
    A, row_lower, row_upper, lower, upper, integrality, max_passes=5, tol=1e-6
):
    """Activity-based bound tightening (see ``BoundPropagator``) for
    ``row_lower <= A x <= row_upper, lower <= x <= upper``, with integer
    bounds rounded. Returns ``(lower, upper, infeasible)``."""
    propagator = BoundPropagator(A, row_lower, row_upper, integrality, tol=tol)
    return propagator(lower, upper, max_passes=max_passes)


class ConflictPool:
    """
    Conflicts learned in the search, as rows ``coef · x[idx] <= rhs`` that
    every solution better than the cutoff satisfies, for propagation.

    Most are dual proofs (as HiGHS keeps them). Lagrangian terms
    ``(constant, r)`` at any duals give ``c·x >= constant + r·x`` at every
    feasible ``x`` (``lagrangian_terms``), so a solution better than the
    cutoff has ``constant + r·x <= cutoff``: an *objective* proof, whose
    right-hand side follows the cutoff. The terms of a dual ray with ``c = 0``
    give ``constant + r·x <= 0`` outright: a *Farkas* proof. Either holds
    whatever the accuracy of the duals; it is worth keeping only when it
    excludes the box of the node it came from. A *no-good* is a set of
    binary fixings that propagation shows infeasible, as
    ``Σ_{fixed at 1} x_j − Σ_{fixed at 0} x_j <= |ones| − 1``.

    A proof is sparsified to the variables whose node bounds differ from the
    global box (on the side that raises ``r·x``); the rest are moved to the
    right-hand side at their worst case over the global box, so it still
    excludes the node. The global box is the root's, which reduced-cost
    fixing may tighten: valid for solutions better than the cutoff, which
    is all pruning needs. Proofs with more than ``max_nnz`` variables after
    sparsifying are dropped, and the pool keeps the newest ``max_rows``.
    """

    def __init__(self, n, integrality, max_rows=2000, max_nnz=None, tol=1e-6):
        self.n = int(n)
        self.is_int = np.asarray(integrality, dtype=bool)
        self.max_rows = int(max_rows)
        self.max_nnz = max(25, self.n // 10) if max_nnz is None else int(max_nnz)
        self.tol = tol
        # (idx, coef, offset, objective): rhs = offset (+ cutoff if objective).
        self.rows = collections.deque()
        self.added = {"objective": 0, "farkas": 0, "nogood": 0}
        self._propagator = None
        self._objective = None  # mask of objective rows in the propagator
        self._offsets = None
        self._cutoff = None

    def __len__(self):
        return len(self.rows)

    def add(self, constant, r, local_lower, local_upper, global_lower, global_upper, kind):
        """Add the proof ``constant + r·x <= cutoff`` (``kind="objective"``)
        or ``<= 0`` (``"farkas"``) learned at the box ``[local_lower,
        local_upper]``, sparsified against ``[global_lower, global_upper]``.
        Returns whether it was kept: it must exclude the local box (given the
        cutoff, for an objective proof) and be short enough."""
        r = np.asarray(r, np.float64)
        scale = np.abs(r).max(initial=0.0)
        if scale == 0.0 or not np.isfinite(constant):
            return False
        significant = np.abs(r) > 1e-9 * scale
        tighter = np.where(
            r > 0, local_lower > global_lower, local_upper < global_upper
        )
        keep = significant & tighter
        if not keep.any():
            return False
        with np.errstate(invalid="ignore"):
            global_least = np.where(r > 0, r * global_lower, r * global_upper)
            local_least = np.where(r > 0, r * local_lower, r * local_upper)
        if not np.all(np.isfinite(global_least[~keep & (r != 0)])):
            return False
        # Relax the kept variables that do least to exclude the node (their
        # local worst case over the global one), while it stays excluded.
        target = self.cutoff_value() if kind == "objective" else 0.0
        rest = ~keep & (r != 0)
        rhs = target - constant - global_least[rest].sum()
        kept = np.flatnonzero(keep)
        with np.errstate(invalid="ignore"):
            gain = local_least[kept] - global_least[kept]
        gain = np.where(np.isfinite(gain), gain, np.inf)  # never relaxed
        slack = local_least[kept].sum() - rhs
        if not (np.isfinite(rhs) and slack > 0):
            return False
        margin = self.tol * (1.0 + abs(rhs)) + 1e-9 * (1.0 + abs(constant))
        order = np.argsort(gain, kind="stable")
        relaxable = np.cumsum(gain[order]) < slack - 2.0 * margin
        relax = kept[order[relaxable & np.isfinite(gain[order])]]
        if relax.size and not np.all(np.isfinite(global_least[relax])):
            return False
        keep[relax] = False
        if np.count_nonzero(keep) > self.max_nnz:
            return False
        rest = ~keep & (r != 0)
        worst = global_least[rest]
        offset = -constant - worst.sum()
        offset -= 1e-9 * (1.0 + abs(constant) + np.abs(worst).sum())  # loosen: safe
        idx = np.flatnonzero(keep)
        coef = r[idx]
        return self._add_row(idx, coef, offset, kind == "objective", kind, local_lower, local_upper)

    def add_nogood(self, ones, zeros):
        """Add the no-good that the binaries ``ones`` at 1 and ``zeros`` at
        0 cannot all hold."""
        ones, zeros = np.asarray(ones, np.int64), np.asarray(zeros, np.int64)
        idx = np.concatenate([ones, zeros])
        if idx.size == 0:
            return False
        coef = np.concatenate([np.ones(ones.size), -np.ones(zeros.size)])
        order = np.argsort(idx)
        return self._add_row(idx[order], coef[order], ones.size - 1.0, False, "nogood")

    def _add_row(self, idx, coef, offset, objective, kind, local_lower=None, local_upper=None):
        if local_lower is not None:
            # Keep it only if it excludes the box it came from (else it is
            # useless here, or the ray had the wrong sign).
            rhs = offset + (self.cutoff_value() if objective else 0.0)
            with np.errstate(invalid="ignore"):
                least = np.where(
                    coef > 0, coef * local_lower[idx], coef * local_upper[idx]
                ).sum()
            if not (np.isfinite(least) and least > rhs + self.tol * (1.0 + abs(rhs))):
                return False
        if len(self.rows) >= self.max_rows:
            self.rows.popleft()
        self.rows.append((idx, coef, float(offset), bool(objective)))
        self.added[kind] += 1
        self._propagator = None
        return True

    def cutoff_value(self):
        return np.inf if self._cutoff is None else self._cutoff

    def set_cutoff(self, cutoff):
        """The cutoff that objective proofs use from now on."""
        self._cutoff = float(cutoff)
        if self._propagator is not None:
            self._propagator.row_upper[:] = self._offsets + np.where(
                self._objective, self._cutoff, 0.0
            )

    def propagate(self, lower, upper, max_passes=3):
        """``BoundPropagator`` over the pool's rows: ``(lower, upper,
        infeasible)``."""
        if not self.rows:
            return lower, upper, False
        if self._propagator is None:
            idx, coef, offsets, objective = zip(*self.rows)
            lengths = [i.size for i in idx]
            A = sp.csr_matrix(
                (np.concatenate(coef), np.concatenate(idx), np.concatenate([[0], np.cumsum(lengths)])),
                shape=(len(self.rows), self.n),
            )
            self._offsets = np.array(offsets)
            self._objective = np.array(objective)
            cutoff = self.cutoff_value()
            row_upper = self._offsets + np.where(self._objective, cutoff, 0.0)
            self._propagator = BoundPropagator(
                A, np.full(len(self.rows), -np.inf), row_upper, self.is_int, tol=self.tol
            )
        return self._propagator(lower, upper, max_passes=max_passes)


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
# The least time one call of the tree heuristics (``tree_heuristics``) gets.
_TREE_HEURISTIC_MIN_SECONDS = 1.0


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


def cmir_cut(
    idx, a, b, x, lower, upper, is_int, max_divisors=64, variable_bounds=None
):
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

    With ``variable_bounds`` (from ``variable_bounds``), a continuous ``y``
    may also be shifted to a variable bound ``y <= d + c·z`` or
    ``y >= d + c·z`` on an integer ``z``, when that bound is at least as
    close to ``x`` as ``y``'s simple bounds (``_substituted_cmir``); the
    better of that cut and the plain one is returned. On fixed-charge rows
    (a demand over flows ``y_j <= u_j·z_j``) the plain cut sees only
    ``0 <= y_j <= U``: with exact LPs, Gomory and these cuts close the root
    gap of vpm1 (it is solved at the root) against 77% without, vpm2 71%
    against 56%, pp08a 95% against 91%.
    """
    plain = _cmir_search(idx, a, b, x, lower, upper, is_int, max_divisors)
    if not variable_bounds:
        return plain
    substituted = _substituted_cmir(
        idx, a, b, x, lower, upper, is_int, max_divisors, variable_bounds
    )
    if substituted is None:
        return plain
    if plain is None or _efficacy(substituted, x) > _efficacy(plain, x):
        return substituted
    return plain


def _efficacy(cut, x):
    idx, coef, rhs = cut
    return (coef @ x[idx] - rhs) / np.linalg.norm(coef)


def variable_bounds(mip):
    """
    The variable bounds in ``mip``'s rows: each row with exactly two
    nonzeros, on a continuous ``y`` and an integer ``z``, reads
    ``a_y·y + a_z·z <= b`` (or ``= b``) and so bounds ``y`` by
    ``d + c·z`` with ``d = b/a_y`` and ``c = −a_z/a_y``: from above when
    ``a_y > 0`` (a VUB, such as ``y <= u·z``), from below when ``a_y < 0``,
    and both ways on an equality row. Returns
    ``{y: {"ub": [(z, c, d), ...], "lb": [...]}}``. These are rows of the
    MIP, so cuts that use them are valid everywhere.
    """
    A = sp.csr_matrix(mip.A)
    integer = mip.integrality
    out = {}
    two = np.flatnonzero(np.diff(A.indptr) == 2)
    for r in two:
        start = A.indptr[r]
        cols, vals = A.indices[start : start + 2], A.data[start : start + 2]
        if integer[cols].sum() != 1:
            continue
        k = 0 if not integer[cols[0]] else 1
        y, z, a_y, a_z = int(cols[k]), int(cols[1 - k]), vals[k], vals[1 - k]
        if abs(a_y) <= 1e-9 or not np.isfinite(mip.b[r]):
            continue
        d, c = mip.b[r] / a_y, -a_z / a_y
        if r < mip.n_eq:
            kinds = ("ub", "lb")
        else:
            kinds = ("ub",) if a_y > 0 else ("lb",)
        bounds = out.setdefault(y, {"ub": [], "lb": []})
        for kind in kinds:
            bounds[kind].append((z, float(c), float(d)))
    return out


def _substituted_cmir(idx, a, b, x, lower, upper, is_int, max_divisors, bounds):
    """
    ``cmir_cut`` of the row ``a · x[idx] <= b`` with each continuous ``y``
    shifted to its variable bound nearest ``x`` (``bounds``, from
    ``variable_bounds``), when that is at least as near as a simple bound:
    ``y = d + c·z − t`` (upper) or ``y = d + c·z + t`` (lower), with a new
    continuous ``t >= 0``. The cut is found over the variables and the
    ``t``s, then each ``t`` is mapped back through its bound. ``None`` when
    no term has a variable bound, or no cut is found.
    """
    n = x.size
    coef = dict(zip(idx.tolist(), a.tolist()))
    subs = []  # (kind, y, z, c, d)
    t_coef, t_value = [], []
    for y in idx.tolist():
        if is_int[y] or y not in bounds:
            continue
        a_y, x_y = coef[y], x[y]
        best_distance = min(x_y - lower[y], upper[y] - x_y)
        best = None
        for kind in ("ub", "lb"):
            for z, c, d in bounds[y][kind]:
                implied = d + c * x[z]
                distance = implied - x_y if kind == "ub" else x_y - implied
                if distance <= best_distance + 1e-9:
                    best_distance, best = distance, (kind, z, c, d)
        if best is None:
            continue
        kind, z, c, d = best
        del coef[y]
        coef[z] = coef.get(z, 0.0) + a_y * c
        b = b - a_y * d
        implied = d + c * x[z]
        subs.append((kind, y, z, c, d))
        t_coef.append(-a_y if kind == "ub" else a_y)
        t_value.append(max(implied - x_y if kind == "ub" else x_y - implied, 0.0))
    if not subs:
        return None
    k = len(subs)
    ext_idx = np.array(list(coef) + list(range(n, n + k)), dtype=np.int64)
    ext_a = np.array(list(coef.values()) + t_coef)
    keep = np.abs(ext_a) > 1e-12 * np.abs(ext_a).max(initial=1.0)
    found = _cmir_search(
        ext_idx[keep],
        ext_a[keep],
        b,
        np.concatenate([x, t_value]),
        np.concatenate([lower, np.zeros(k)]),
        np.concatenate([upper, np.full(k, np.inf)]),
        np.concatenate([is_int, np.zeros(k, dtype=bool)]),
        max_divisors,
    )
    if found is None:
        return None
    c_idx, c_coef, rhs = found
    out = {}
    for j, g in zip(c_idx.tolist(), c_coef.tolist()):
        if j < n:
            out[j] = out.get(j, 0.0) + g
            continue
        kind, y, z, c, d = subs[j - n]
        sign = 1.0 if kind == "ub" else -1.0  # t = sign·(d + c·z − y)
        rhs -= sign * g * d
        out[z] = out.get(z, 0.0) + sign * g * c
        out[y] = out.get(y, 0.0) - sign * g
    o_idx = np.array(sorted(out), dtype=np.int64)
    o_coef = np.array([out[j] for j in o_idx])
    nonzero = np.abs(o_coef) > 1e-12 * np.abs(o_coef).max(initial=1.0)
    o_idx, o_coef = o_idx[nonzero], o_coef[nonzero]
    if o_idx.size == 0:
        return None
    cut = (o_idx, o_coef, rhs)
    return cut if _efficacy(cut, x) > 1e-9 else None


def _cmir_search(idx, a, b, x, lower, upper, is_int, max_divisors):
    """The search of ``cmir_cut`` over simple bounds."""
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


def _flow_set(idx, a, b, x, lower, upper, is_int, variable_bounds):
    """
    The single-node flow set of the row ``a · x[idx] <= b``:
    ``Σ_{N+} f_j − Σ_{N−} f_j <= b'`` with ``0 <= f_j <= u_j·w_j`` and
    binary ``w_j``. Each term becomes one arc:

    * a continuous ``y >= 0`` with a variable upper bound ``y <= u·z`` on a
      binary ``z`` (``variable_bounds``, ``d = 0``): ``f = |a|·y``, ``w = z``;
    * a binary ``x``: ``f = |a|·x``, ``w = x``;
    * anything else, shifted to its simple bound nearer ``x``:
      ``f = |a|·t`` with ``w ≡ 1``.

    Arcs that share a binary only restrict the set, so cuts for it hold.
    Returns ``(arcs, b')`` with each arc a dict: ``sign`` (+1 for ``N+``),
    capacity ``u``, ``f`` as ``[(column, coefficient)]`` plus a constant
    ``f0``, the binary column ``w`` (``None`` for ``w ≡ 1``) and the LP
    values ``fval``, ``wval``; or ``(None, None)`` with a free variable.
    """
    arcs = []
    for j, aj in zip(idx.tolist(), a.tolist()):
        if aj == 0.0:
            continue
        sign, mag = (1, aj) if aj > 0 else (-1, -aj)
        binary = is_int[j] and lower[j] == 0.0 and upper[j] == 1.0
        vub = None
        if not is_int[j] and lower[j] == 0.0 and variable_bounds and j in variable_bounds:
            for z, c, d in variable_bounds[j]["ub"]:
                if d == 0.0 and c > 0 and is_int[z] and lower[z] == 0.0 and upper[z] == 1.0:
                    if vub is None or c * x[z] < vub[1] * x[vub[0]]:
                        vub = (z, c)  # the tightest at x
        if vub is not None:
            z, c = vub
            arcs.append(dict(sign=sign, u=mag * c, f=[(j, mag)], f0=0.0, w=z,
                             fval=mag * x[j], wval=x[z]))
        elif binary:
            arcs.append(dict(sign=sign, u=mag, f=[(j, mag)], f0=0.0, w=j,
                             fval=mag * x[j], wval=x[j]))
        else:
            lo, up = lower[j], upper[j]
            if np.isfinite(up) and (not np.isfinite(lo) or up - x[j] < x[j] - lo):
                # y = up − t: the term flips to the other side.
                b -= aj * up
                arcs.append(dict(sign=-sign, u=mag * (up - lo), f=[(j, -mag)],
                                 f0=mag * up, w=None, fval=mag * (up - x[j]), wval=1.0))
            elif np.isfinite(lo):
                b -= aj * lo
                arcs.append(dict(sign=sign, u=mag * (up - lo), f=[(j, mag)],
                                 f0=-mag * lo, w=None, fval=mag * (x[j] - lo), wval=1.0))
            else:
                return None, None
    return arcs, b


def _gfci(arcs, b, plus, minus):
    """
    The generalised flow cover inequality of the cover ``(plus, minus)``
    (arc indices) of a flow set: with ``λ = Σ_{C+} u − Σ_{C−} u − b > 0``,

        Σ_{C+} (f_j + (u_j − λ)⁺(1 − w_j)) − Σ_{L2−} f_j − λ Σ_{L−} w_j
            <= b + Σ_{C−} u_j,

    where the arcs of ``N− \\ C−`` go to ``L−`` or ``L2−``, whichever term
    is smaller at the LP point. Returns ``(terms, rhs)``, with terms
    ``(arc, coefficient of f, coefficient of w)``, or ``None``.
    """
    lam = sum(arcs[i]["u"] for i in plus) - sum(arcs[i]["u"] for i in minus) - b
    if lam <= 1e-9:
        return None
    rhs = b + sum(arcs[i]["u"] for i in minus)
    terms = []
    for i in plus:
        excess = max(arcs[i]["u"] - lam, 0.0)
        terms.append((i, 1.0, -excess))  # f + excess·(1 − w)
        rhs -= excess
    in_minus = set(minus)
    for i, arc in enumerate(arcs):
        if arc["sign"] > 0 or i in in_minus:
            continue
        if lam * arc["wval"] < arc["fval"]:
            terms.append((i, 0.0, -lam))
        else:
            terms.append((i, -1.0, 0.0))
    return terms, rhs


def _flow_cut_columns(arcs, terms, rhs):
    """A flow set inequality in the original columns: ``(idx, coef, rhs)``."""
    coef = {}
    for i, cf, cw in terms:
        arc = arcs[i]
        if cf:
            for j, m in arc["f"]:
                coef[j] = coef.get(j, 0.0) + cf * m
            rhs -= cf * arc["f0"]
        if cw:
            if arc["w"] is None:
                rhs -= cw  # w ≡ 1
            else:
                coef[arc["w"]] = coef.get(arc["w"], 0.0) + cw
    idx = np.array(sorted(j for j in coef if abs(coef[j]) > 1e-12), dtype=np.int64)
    if idx.size == 0:
        return None
    return idx, np.array([coef[j] for j in idx]), rhs


def flow_cover_cut(idx, a, b, x, lower, upper, is_int, variable_bounds=None):
    """
    The most efficacious generalised flow cover cut (``_gfci``) from the
    single-node flow set of the row ``a · x[idx] <= b`` (``_flow_set``), or
    ``None``. Covers: ``C−`` empty, or the ``N−`` arcs saturated at ``x``;
    ``C+`` the ``N+`` arcs by ``(1 − w*)/u`` ascending (open arcs, largest
    first) until they overflow the row, as is and trimmed of arcs it does
    not need (least open first). For fixed-charge rows such as
    ``Σ y_j − Σ y_k <= d`` with ``y_j <= u_j·z_j``. With exact LPs, adding
    these at every aggregation level closed 82% of fixnet6's root gap
    against 65% without them, vpm1 95% against 84%, vpm2 73% against 70%.
    Returns ``(idx, coef, rhs)``.
    """
    arcs, b = _flow_set(idx, a, b, x, lower, upper, is_int, variable_bounds)
    if not arcs:
        return None
    # C+ may be empty: on a demand row −Σ y_j <= −D the inequality reads
    # Σ_{L2−} y_j + λ Σ_{L−} z_j >= λ.
    plus = [i for i, arc in enumerate(arcs) if arc["sign"] > 0 and np.isfinite(arc["u"])]
    minus = [i for i, arc in enumerate(arcs) if arc["sign"] < 0 and np.isfinite(arc["u"])]
    saturated = [
        i for i in minus
        if arcs[i]["wval"] > 1 - 1e-6 and arcs[i]["fval"] >= arcs[i]["u"] - 1e-6
    ]
    order = sorted(
        plus,
        key=lambda i: ((1 - arcs[i]["wval"]) / max(arcs[i]["u"], 1e-12), -arcs[i]["u"]),
    )
    best, best_efficacy = None, 1e-6
    for cover_minus in ([], saturated):
        need = b + sum(arcs[i]["u"] for i in cover_minus)
        cover, total = [], 0.0
        for i in order:
            cover.append(i)
            total += arcs[i]["u"]
            if total > need + 1e-9:
                break
        if total <= need + 1e-9:
            continue
        trimmed = list(cover)
        for i in sorted(cover, key=lambda i: arcs[i]["wval"]):
            if total - arcs[i]["u"] > need + 1e-9:
                trimmed.remove(i)
                total -= arcs[i]["u"]
        for cover_plus in (cover, trimmed):
            made = _gfci(arcs, b, cover_plus, cover_minus)
            cut = None if made is None else _flow_cut_columns(arcs, *made)
            if cut is None:
                continue
            efficacy = _efficacy(cut, x)
            if efficacy > best_efficacy:
                best, best_efficacy = cut, efficacy
    return best


# %%
# Batched separation (JAX). Each base inequality ``a · x[idx] <= b`` of a
# round becomes a row of padded "slot" arrays, and one jitted call runs the
# c-MIR search (``_cmir_search``) or the flow cover search
# (``flow_cover_cut``) on a chunk of rows at once. The host side builds the
# slots and maps the cuts back with flat numpy over all rows, with no loop
# per row. Chunks have a fixed number of rows and a power-of-two width, so
# few shapes compile. The kernels run in float64 whatever the profile: a
# cut's floors and right-hand side must be exact enough to stay valid. The
# cuts are the numpy searches' (``CutSeparator(backend="numpy")``), up to
# rounding in the last bits.

# Below this many base inequalities a round uses the numpy separators.
_SEPARATION_MIN_BATCH = 32
# Rows per kernel call.
_SEPARATION_CHUNK = 256


def _flat_rows(lengths):
    """Rows of the given lengths laid end to end: each entry's row and slot,
    and each row's start."""
    lengths = np.asarray(lengths, np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
    row = np.repeat(np.arange(lengths.size), lengths)
    slot = np.arange(row.size) - starts[row]
    return row, slot, starts


def _width_of(lengths):
    """The padded width of rows of these lengths: 16, 64, 256, then powers
    of four. Coarse steps mean fewer kernel calls, which cost more than the
    padding at these sizes (a call is about 1.5 ms even when small)."""
    lengths = np.maximum(np.asarray(lengths, np.int64), 1)
    return np.maximum(16, 4 ** np.ceil(np.log(lengths) / np.log(4) - 1e-9).astype(np.int64))


def _run_batched(kernel, row, slot, lengths, fields, fills, b, n_slot_outputs, **static):
    """Run ``kernel(*padded fields, valid, b, **static) -> (ok, rhs,
    *per_slot)`` over rows given as flat entries (``row``, ``slot``), in
    chunks of ``_SEPARATION_CHUNK`` rows padded to a width from
    ``_width_of``. Returns ``ok`` and ``rhs`` per row and each per-slot
    output as a flat per-entry array."""
    lengths = np.asarray(lengths, np.int64)
    rows = lengths.size
    ok = np.zeros(rows, dtype=bool)
    rhs = np.zeros(rows)
    outs = [np.zeros(row.size) for _ in range(n_slot_outputs)]
    if rows == 0:
        return ok, rhs, outs
    width_of = _width_of(lengths)
    chunk = _SEPARATION_CHUNK
    for width in np.unique(width_of):
        width = int(width)
        members = np.flatnonzero(width_of == width)
        local = np.full(rows, -1, dtype=np.int64)
        local[members] = np.arange(members.size)
        entries = np.flatnonzero(width_of[row] == width)
        r, s = local[row[entries]], slot[entries]
        # The group's padded arrays at once; the chunks are slices of them.
        height = -(-members.size // chunk) * chunk
        padded = []
        for values, fill in zip(fields, fills):
            array = np.full((height, width), fill, dtype=values.dtype)
            array[r, s] = values[entries]
            padded.append(array)
        valid = np.zeros((height, width), dtype=bool)
        valid[r, s] = True
        b_group = np.zeros(height)
        b_group[: members.size] = b[members]
        results = []
        with jax.enable_x64(True):
            for top in range(0, height, chunk):
                part = slice(top, top + chunk)
                results.append(
                    kernel(*(array[part] for array in padded), valid[part], b_group[part], **static)
                )
            stacked = [np.concatenate([np.asarray(res[k]) for res in results]) for k in range(len(results[0]))]
        ok[members] = stacked[0][: members.size]
        rhs[members] = stacked[1][: members.size]
        for out, values in zip(outs, stacked[2:]):
            out[entries] = values[r, s]
    return ok, rhs, outs


def _cmir_eval(a, b, xv, lo, up, integer, valid, use_upper, delta):
    """``_cmir_for`` on a batch: ``(ok, efficacy, coef, rhs)``."""
    bound = jnp.where(valid, jnp.where(use_upper, up, lo), 0.0)
    sign = jnp.where(use_upper, -1.0, 1.0)
    at = a * sign
    beta = (b - jnp.sum(a * bound, axis=1)) / delta
    f = beta - jnp.floor(beta)
    ok = (f >= 0.01) & (f <= 0.99)
    one_f = jnp.where(ok, 1.0 - f, 1.0)[:, None]
    g = at / delta[:, None]
    fj = g - jnp.floor(g)
    coef_t = jnp.where(
        integer,
        jnp.floor(g) + jnp.maximum(fj - f[:, None], 0.0) / one_f,
        jnp.where(at < 0, at / (delta[:, None] * one_f), 0.0),
    )
    coef = jnp.where(valid, coef_t * sign, 0.0)
    norm = jnp.sqrt(jnp.sum(coef * coef, axis=1))
    ok = ok & (norm > 0)
    rhs = jnp.floor(beta) + jnp.sum(coef * bound, axis=1)
    efficacy = (jnp.sum(coef * xv, axis=1) - rhs) / jnp.where(ok, norm, 1.0)
    return ok, efficacy, coef, rhs


@functools.partial(jax.jit, static_argnames=("max_divisors",))
def _cmir_kernel(a, lo, up, xv, integer, valid, b, max_divisors):
    """``_cmir_search`` on a batch of rows: ``(ok, rhs, coef)``, ``coef``
    per slot in the original variables."""
    rows = a.shape[0]
    lo = jnp.where(valid, lo, 0.0)
    up = jnp.where(valid, up, 0.0)
    xv = jnp.where(valid, xv, 0.0)
    a = jnp.where(valid, a, 0.0)
    integer = integer & valid
    has_int = jnp.any(integer, axis=1)
    use_upper = valid & jnp.isfinite(up) & (~jnp.isfinite(lo) | (up - xv < xv - lo))
    finite = jnp.all(~valid | jnp.isfinite(jnp.where(use_upper, up, lo)), axis=1)
    t = jnp.where(valid, jnp.maximum(jnp.where(use_upper, up - xv, xv - lo), 0.0), 0.0)
    # Divisors: distinct |a| of integer terms away from their bound, largest
    # first, at most max_divisors; then 1.
    d = jnp.where(integer & (t > 1e-6) & (jnp.abs(a) > 1e-9), jnp.abs(a), -1.0)
    d = -jnp.sort(-d, axis=1)
    first = jnp.concatenate([jnp.ones((rows, 1), bool), d[:, 1:] != d[:, :-1]], axis=1)
    first = first & (d > 0)
    candidate = first & (jnp.cumsum(first, axis=1) <= max_divisors)
    any_divisor = jnp.any(candidate, axis=1)

    def evaluate(use, delta):
        return _cmir_eval(a, b, xv, lo, up, integer, valid, use, delta)

    def try_delta(state, allowed, delta):
        has, best, best_delta = state
        ok, eff, _, _ = evaluate(use_upper, delta)
        take = allowed & ok & (~has | (eff > best))
        return has | take, jnp.where(take, eff, best), jnp.where(take, delta, best_delta)

    def divisor_step(k, state):
        allowed = candidate[:, k]
        return try_delta(state, allowed, jnp.where(allowed, d[:, k], 1.0))

    state = (jnp.zeros(rows, bool), jnp.full(rows, -jnp.inf), jnp.ones(rows))
    state = jax.lax.fori_loop(0, jnp.max(jnp.sum(d > 0, axis=1)), divisor_step, state)
    has, best, best_delta = try_delta(state, any_divisor, jnp.ones(rows))
    for k in (2.0, 4.0, 8.0):
        ok, eff, _, _ = evaluate(use_upper, best_delta / k)
        take = has & ok & (eff > best)
        best = jnp.where(take, eff, best)
        best_delta = jnp.where(take, best_delta / k, best_delta)
    # Complementation flips, farthest from a bound first.
    inside = (
        integer & jnp.isfinite(lo) & jnp.isfinite(up) & (xv > lo + 1e-6) & (xv < up - 1e-6)
    )
    distance = jnp.minimum(xv - lo, up - xv)
    order = jnp.argsort(jnp.where(inside, -distance, jnp.inf), axis=1, stable=True)
    slots = jnp.arange(a.shape[1])

    def flip_step(k, carry):
        use, best = carry
        j = order[:, k]
        active = has & jnp.take_along_axis(inside, j[:, None], axis=1)[:, 0]
        flipped = use ^ (slots[None, :] == j[:, None])
        ok, eff, _, _ = evaluate(flipped, best_delta)
        take = active & ok & (eff > best + 1e-12)
        return jnp.where(take[:, None], flipped, use), jnp.where(take, eff, best)

    use_upper, best = jax.lax.fori_loop(
        0, jnp.max(jnp.sum(inside, axis=1)), flip_step, (use_upper, best)
    )
    ok, eff, coef, rhs = evaluate(use_upper, best_delta)
    ok = has_int & finite & has & ok & (eff > 1e-9)
    return ok, rhs, coef


def _substitution_tables(bounds, x, lower, upper):
    """Per column, the variable bound ``_substituted_cmir`` would shift a
    continuous column to at ``x``: ``kind`` (0: none, 1: upper, 2: lower)
    and ``z``, ``c``, ``d`` arrays."""
    n = x.size
    kind_of = np.zeros(n, np.int64)
    z_of = np.zeros(n, np.int64)
    c_of, d_of = np.zeros(n), np.zeros(n)
    for y, kinds in bounds.items():
        x_y = x[y]
        best_distance = min(x_y - lower[y], upper[y] - x_y)
        best = None
        for kind in ("ub", "lb"):
            for z, c, d in kinds[kind]:
                implied = d + c * x[z]
                distance = implied - x_y if kind == "ub" else x_y - implied
                if distance <= best_distance + 1e-9:
                    best_distance, best = distance, (kind, z, c, d)
        if best is not None:
            kind, z, c, d = best
            kind_of[y] = 1 if kind == "ub" else 2
            z_of[y], c_of[y], d_of[y] = z, c, d
    return kind_of, z_of, c_of, d_of


def _group_sum(keys, values):
    """Sum ``values`` by ``keys``: ``(unique keys, sums, inverse)``."""
    unique, inverse = np.unique(keys, return_inverse=True)
    return unique, np.bincount(inverse, values, minlength=unique.size), inverse


def _substituted_batch(row, slot, cols, a, b, x, lower, upper, is_int, tables):
    """The rows of ``_substituted_cmir`` for every base that has a term to
    substitute, as flat entries in its slot order: the row's other terms in
    order, then each new ``z`` at its first ``y``, then each ``t`` in ``y``
    order (the order of its dict, which decides ties in the flips). Returns
    ``None`` or a dict of the new rows' entries and what maps them back."""
    kind_of, z_of, c_of, d_of = tables
    n = x.size
    sub = (kind_of[cols] > 0) & ~is_int[cols]
    if not sub.any():
        return None
    rows = np.unique(row[sub])
    take = np.isin(row, rows)
    row, slot, cols, a, sub = row[take], slot[take], cols[take], a[take], sub[take]
    kept, s_idx = ~sub, np.flatnonzero(sub)
    y, z = cols[s_idx], z_of[cols[s_idx]]
    c, d = c_of[cols[s_idx]], d_of[cols[s_idx]]
    kind = kind_of[cols[s_idx]]
    a_y = a[s_idx]
    # Terms over the variables: the kept ones and each substituted y's z.
    v_row = np.concatenate([row[kept], row[s_idx]])
    v_col = np.concatenate([cols[kept], z])
    v_a = np.concatenate([a[kept], a_y * c])
    v_k1 = np.concatenate([np.zeros(kept.sum(), np.int64), np.ones(s_idx.size, np.int64)])
    v_k2 = np.concatenate([slot[kept], slot[s_idx]])
    keys, sums, inverse = _group_sum(v_row * n + v_col, v_a)
    # A group's place: its kept term's slot, else its first y's.
    rank = v_k1 * (slot.max(initial=0) + 1) + v_k2
    place = np.full(keys.size, np.iinfo(np.int64).max)
    np.minimum.at(place, inverse, rank)
    g_row, g_col = keys // n, keys % n
    # The t's: one per substituted y, after everything else, in y order.
    t_order = np.lexsort((slot[s_idx], row[s_idx]))
    s_idx, y, z, c, d, kind, a_y = (v[t_order] for v in (s_idx, y, z, c, d, kind, a_y))
    t_row = row[s_idx]
    implied = d + c * x[z]
    t_value = np.maximum(np.where(kind == 1, implied - x[y], x[y] - implied), 0.0)
    t_a = np.where(kind == 1, -a_y, a_y)
    big = np.iinfo(np.int64).max // 4
    e_row = np.concatenate([g_row, t_row])
    e_place = np.concatenate([place, big + np.arange(t_row.size)])
    e_a = np.concatenate([sums, t_a])
    e_col = np.concatenate([g_col, np.full(t_row.size, -1)])
    e_t = np.concatenate([np.full(g_row.size, -1), np.arange(t_row.size)])
    order = np.lexsort((e_place, e_row))
    e_row, e_a, e_col, e_t = e_row[order], e_a[order], e_col[order], e_t[order]
    # Drop near-zero terms, relative to the row's largest (at least 1).
    scale = np.maximum(np.maximum.reduceat(np.abs(e_a), np.flatnonzero(np.r_[True, e_row[1:] != e_row[:-1]])), 1.0)
    row_ids, row_start = np.unique(e_row, return_index=True)
    scale_of = np.zeros(row_ids.max(initial=0) + 1)
    scale_of[row_ids] = scale
    keep = np.abs(e_a) > 1e-12 * scale_of[e_row]
    e_row, e_a, e_col, e_t = e_row[keep], e_a[keep], e_col[keep], e_t[keep]
    real = e_col >= 0
    safe = np.where(real, e_col, 0)
    t_safe = np.where(real, 0, e_t)
    b_new = b[rows] - np.bincount(
        np.searchsorted(rows, t_row), a_y * d, minlength=rows.size
    )
    local = np.searchsorted(rows, e_row)
    lengths = np.bincount(local, minlength=rows.size)
    return dict(
        rows=rows, local=local, lengths=lengths, b=b_new,
        a=e_a, col=e_col, t=e_t,
        lo=np.where(real, lower[safe], 0.0),
        up=np.where(real, upper[safe], np.inf),
        x=np.where(real, x[safe], t_value[t_safe]),
        integer=real & is_int[safe],
        t_kind=kind, t_y=y, t_z=z, t_c=c, t_d=d,
    )


def _cuts_from_entries(rows, e_row, e_col, e_coef, rhs, n, relative):
    """Cuts ``(idx, coef, rhs)`` per row from flat ``(row, column,
    coefficient)`` contributions, summed per column and sorted; terms at
    most 1e-12 (times the row's largest, at least 1, if ``relative``) are
    dropped. Returns ``{row: cut}``."""
    if e_row.size == 0:
        return {}
    keys, sums, _ = _group_sum(e_row * n + e_col, e_coef)
    k_row, k_col = keys // n, keys % n
    starts = np.flatnonzero(np.r_[True, k_row[1:] != k_row[:-1]])
    if relative:
        scale = np.maximum(np.maximum.reduceat(np.abs(sums), starts), 1.0)
        limit = 1e-12 * np.repeat(scale, np.diff(np.r_[starts, sums.size]))
    else:
        limit = np.full(sums.size, 1e-12)
    out = {}
    for start, end in zip(starts, np.r_[starts[1:], sums.size]):
        keep = np.abs(sums[start:end]) > limit[start:end]
        if keep.any():
            r = int(k_row[start])
            out[r] = (k_col[start:end][keep], sums[start:end][keep], float(rhs[r]))
    return out


def batched_cmir_cuts(
    bases, x, lower, upper, is_int, variable_bounds=None, max_divisors=64
):
    """``cmir_cut`` of every base inequality ``(idx, a, b)`` in ``bases``,
    in batched JAX calls: a list of ``(idx, coef, rhs)`` or ``None``."""
    x = np.asarray(x, np.float64)
    count = len(bases)
    cuts = [None] * count
    if count == 0:
        return cuts
    lengths = np.array([idx.size for idx, _, _ in bases], np.int64)
    row, slot, starts = _flat_rows(lengths)
    cols = np.concatenate([np.asarray(idx, np.int64) for idx, _, _ in bases])
    a = np.concatenate([np.asarray(coef, np.float64) for _, coef, _ in bases])
    b = np.array([rhs for _, _, rhs in bases], np.float64)
    batch = None
    if variable_bounds:
        tables = _substitution_tables(variable_bounds, x, lower, upper)
        batch = _substituted_batch(row, slot, cols, a, b, x, lower, upper, is_int, tables)
    # One batched search over the plain rows and, after them, the
    # substituted ones.
    fields = [a, lower[cols], upper[cols], x[cols], is_int[cols]]
    all_lengths, all_b = lengths, b
    if batch is not None:
        fields = [
            np.concatenate([field, batch[key]])
            for field, key in zip(fields, ("a", "lo", "up", "x", "integer"))
        ]
        all_lengths = np.concatenate([lengths, batch["lengths"]])
        all_b = np.concatenate([b, batch["b"]])
    all_row, all_slot, _ = _flat_rows(all_lengths)
    ok_all, rhs_all, (coef_all,) = _run_batched(
        _cmir_kernel, all_row, all_slot, all_lengths, fields,
        [0.0, 0.0, 0.0, 0.0, False], all_b, 1, max_divisors=max_divisors,
    )
    ok, rhs, coef = ok_all[:count], rhs_all[:count], coef_all[: row.size]
    for i in np.flatnonzero(ok):
        cuts[i] = (bases[i][0], coef[starts[i] : starts[i] + lengths[i]].copy(), float(rhs[i]))
    if batch is None:
        return cuts
    ok, s_rhs, s_coef = ok_all[count:], rhs_all[count:], coef_all[row.size :]
    # Map the t's back through their bounds (t = sign·(d + c·z − y)).
    n = x.size
    local, col, t = batch["local"], batch["col"], batch["t"]
    live = ok[local]
    real = live & (col >= 0)
    is_t = live & (col < 0)
    ti = t[is_t]
    g = s_coef[is_t]
    sign = np.where(batch["t_kind"][ti] == 1, 1.0, -1.0)
    rhs_t = s_rhs - np.bincount(local[is_t], sign * g * batch["t_d"][ti], minlength=s_rhs.size)
    found = _cuts_from_entries(
        np.arange(s_rhs.size),
        np.concatenate([local[real], local[is_t], local[is_t]]),
        np.concatenate([col[real], batch["t_z"][ti], batch["t_y"][ti]]),
        np.concatenate([s_coef[real], sign * g * batch["t_c"][ti], -sign * g]),
        rhs_t, n, relative=True,
    )
    for r, cut in found.items():
        if _efficacy(cut, x) <= 1e-9:
            continue
        i = int(batch["rows"][r])
        if cuts[i] is None or _efficacy(cut, x) > _efficacy(cuts[i], x):
            cuts[i] = cut
    return cuts


def _flow_vubs(bounds, x, lower, upper, is_int):
    """Per column, the variable upper bound ``_flow_set`` would use:
    ``(z, c)`` arrays with ``z = −1`` where there is none."""
    n = x.size
    z_of, c_of = np.full(n, -1, np.int64), np.zeros(n)
    for j, kinds in (bounds or {}).items():
        if is_int[j] or lower[j] != 0.0:
            continue
        pick = None
        for z, c, d in kinds["ub"]:
            if d == 0.0 and c > 0 and is_int[z] and lower[z] == 0.0 and upper[z] == 1.0:
                if pick is None or c * x[z] < pick[1] * x[pick[0]]:
                    pick = (z, c)
        if pick is not None:
            z_of[j], c_of[j] = pick
    return z_of, c_of


@jax.jit
def _flow_kernel(sign, u, fval, wval, f_mult, f0, w_col, x_f, x_w, f_col, live, valid, b):
    """``flow_cover_cut``'s search on a batch of flow sets: ``(ok, rhs, cf,
    cw)``, ``cf``, ``cw`` the coefficients of each arc's flow and binary in
    the best cut, ``rhs`` with the flows' constants and the ``w ≡ 1`` terms
    folded in."""
    valid = valid & live
    rows, width = sign.shape
    slots = jnp.arange(width)
    u_fin = jnp.isfinite(u)
    plus = valid & (sign > 0) & u_fin
    minus = valid & (sign < 0) & u_fin
    saturated = minus & (wval > 1 - 1e-6) & (fval >= u - 1e-6)
    u0 = jnp.where(u_fin, u, 0.0)
    n_plus = jnp.sum(plus, axis=1)
    key = jnp.where(plus, (1 - wval) / jnp.maximum(u0, 1e-12), jnp.inf)
    order = jnp.lexsort((jnp.broadcast_to(slots, sign.shape), -u0, key), axis=1)
    rank = jnp.argsort(order, axis=1)
    prefix = jnp.cumsum(jnp.take_along_axis(jnp.where(plus, u0, 0.0), order, axis=1), axis=1)

    # The norm of a cut over merged columns: entries f (f_col) and w (w_col).
    cols = jnp.concatenate([jnp.where(valid, f_col, -1), jnp.where(valid, w_col, -1)], axis=1)
    col_order = jnp.argsort(cols, axis=1, stable=True)
    sorted_cols = jnp.take_along_axis(cols, col_order, axis=1)
    new = jnp.concatenate(
        [jnp.ones((rows, 1), bool), sorted_cols[:, 1:] != sorted_cols[:, :-1]], axis=1
    )
    segment = jnp.cumsum(new, axis=1) - 1 + (jnp.arange(rows) * 2 * width)[:, None]
    real_col = sorted_cols >= 0

    def cut_for(in_plus, in_minus, cover_ok):
        lam = (
            jnp.sum(jnp.where(in_plus, u0, 0.0), axis=1)
            - jnp.sum(jnp.where(in_minus, u0, 0.0), axis=1)
            - b
        )
        excess = jnp.where(in_plus, jnp.maximum(u0 - lam[:, None], 0.0), 0.0)
        rest = valid & (sign < 0) & ~in_minus
        as_w = rest & (lam[:, None] * wval < fval)
        cf = jnp.where(in_plus, 1.0, jnp.where(rest & ~as_w, -1.0, 0.0))
        cw = jnp.where(in_plus, -excess, jnp.where(as_w, -lam[:, None], 0.0))
        rhs = (
            b + jnp.sum(jnp.where(in_minus, u0, 0.0), axis=1)
            - jnp.sum(excess, axis=1)
            - jnp.sum(cf * f0, axis=1)
            - jnp.sum(jnp.where(w_col < 0, cw, 0.0), axis=1)
        )
        values = jnp.concatenate([cf * f_mult, jnp.where(w_col >= 0, cw, 0.0)], axis=1)
        merged = jax.ops.segment_sum(
            jnp.where(real_col, jnp.take_along_axis(values, col_order, axis=1), 0.0).ravel(),
            segment.ravel(),
            num_segments=rows * 2 * width,
        ).reshape(rows, 2 * width)
        keep = jnp.abs(merged) > 1e-12
        norm = jnp.sqrt(jnp.sum(jnp.where(keep, merged * merged, 0.0), axis=1))
        dot = jnp.sum(cf * f_mult * x_f, axis=1) + jnp.sum(
            jnp.where(w_col >= 0, cw * x_w, 0.0), axis=1
        )
        ok = cover_ok & (lam > 1e-9) & jnp.any(keep, axis=1)
        efficacy = jnp.where(ok, (dot - rhs) / jnp.where(ok, norm, 1.0), -jnp.inf)
        return ok, efficacy, cf, cw, rhs

    best = (
        jnp.zeros(rows, bool), jnp.full(rows, 1e-6),
        jnp.zeros_like(u0), jnp.zeros_like(u0), jnp.zeros(rows),
    )
    for in_minus in (jnp.zeros_like(valid), saturated):
        need = b + jnp.sum(jnp.where(in_minus, u0, 0.0), axis=1)
        crossed = (prefix > (need + 1e-9)[:, None]) & (slots[None, :] < n_plus[:, None])
        size = jnp.where(jnp.any(crossed, axis=1), jnp.argmax(crossed, axis=1) + 1, n_plus)
        total = jnp.where(
            size > 0,
            jnp.take_along_axis(prefix, jnp.maximum(size - 1, 0)[:, None], axis=1)[:, 0],
            0.0,
        )
        cover_ok = total > need + 1e-9
        in_cover = plus & (rank < size[:, None])
        # Trim: least open first (ties in cover order), while still a cover.
        trim_order = jnp.lexsort((rank, jnp.where(in_cover, wval, jnp.inf)), axis=1)

        def trim(k, carry):
            kept, total = carry
            j = trim_order[:, k]
            u_j = jnp.take_along_axis(u0, j[:, None], axis=1)[:, 0]
            active = jnp.take_along_axis(in_cover, j[:, None], axis=1)[:, 0]
            drop = active & (total - u_j > need + 1e-9)
            kept = kept & ~((slots[None, :] == j[:, None]) & drop[:, None])
            return kept, jnp.where(drop, total - u_j, total)

        trimmed, _ = jax.lax.fori_loop(
            0, jnp.max(jnp.sum(in_cover, axis=1)), trim, (in_cover, total)
        )
        for in_plus in (in_cover, trimmed):
            ok, eff, cf, cw, rhs = cut_for(in_plus, in_minus, cover_ok)
            take = ok & (eff > best[1])
            best = (
                best[0] | take,
                jnp.where(take, eff, best[1]),
                jnp.where(take[:, None], cf, best[2]),
                jnp.where(take[:, None], cw, best[3]),
                jnp.where(take, rhs, best[4]),
            )
    return best[0], best[4], best[2], best[3]


def batched_flow_cover_cuts(bases, x, lower, upper, is_int, variable_bounds=None):
    """``flow_cover_cut`` of every base inequality in ``bases`` in batched
    JAX calls: a list of ``(idx, coef, rhs)`` or ``None``."""
    x = np.asarray(x, np.float64)
    count = len(bases)
    out = [None] * count
    if count == 0:
        return out
    z_of, c_of = _flow_vubs(variable_bounds, x, lower, upper, is_int)
    lengths = np.array([idx.size for idx, _, _ in bases], np.int64)
    row, slot, _ = _flat_rows(lengths)
    col = np.concatenate([np.asarray(idx, np.int64) for idx, _, _ in bases])
    a = np.concatenate([np.asarray(coef, np.float64) for _, coef, _ in bases])
    b = np.array([rhs for _, _, rhs in bases], np.float64)
    live = a != 0.0
    mag, sgn = np.abs(a), np.where(a > 0, 1.0, -1.0)
    lo, up, xv = lower[col], upper[col], x[col]
    vub = z_of[col] >= 0
    binary = ~vub & is_int[col] & (lo == 0.0) & (up == 1.0)
    other = ~vub & ~binary
    with np.errstate(invalid="ignore"):
        use_up = other & np.isfinite(up) & (~np.isfinite(lo) | (up - xv < xv - lo))
    use_lo = other & ~use_up & np.isfinite(lo)
    free = np.bincount(row, live & other & ~use_up & ~use_lo, minlength=count) > 0
    w_col = np.where(vub, z_of[col], np.where(binary, col, -1))
    with np.errstate(invalid="ignore"):
        u = np.where(vub, mag * c_of[col], np.where(binary, mag, mag * (up - lo)))
        fval = np.where(use_up, mag * (up - xv), np.where(use_lo, mag * (xv - lo), mag * xv))
    bound = np.where(use_up, up, np.where(use_lo, lo, 0.0))
    shift = np.where(live & (use_up | use_lo), a * bound, 0.0)
    b_flow = b - np.bincount(row, shift, minlength=count)
    f_mult = np.where(use_up, -mag, mag)
    ok, rhs, (cf, cw) = _run_batched(
        _flow_kernel, row, slot, lengths,
        [
            np.where(use_up, -sgn, sgn), u, fval,
            np.where(vub, x[np.maximum(w_col, 0)], np.where(binary, xv, 1.0)),
            f_mult, np.where(use_up, mag * up, np.where(use_lo, -mag * lo, 0.0)),
            w_col, xv, np.where(w_col >= 0, x[np.maximum(w_col, 0)], 0.0), col, live,
        ],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1, 0.0, 0.0, -1, False],
        b_flow, 2,
    )
    ok &= ~free
    live_f = ok[row] & (cf != 0)
    live_w = ok[row] & (cw != 0) & (w_col >= 0)
    found = _cuts_from_entries(
        np.arange(count),
        np.concatenate([row[live_f], row[live_w]]),
        np.concatenate([col[live_f], w_col[live_w]]),
        np.concatenate([cf[live_f] * f_mult[live_f], cw[live_w]]),
        rhs, x.size, relative=False,
    )
    for r, cut in found.items():
        out[r] = cut
    return out


def gomory_cuts(
    A,
    b,
    n_eq,
    lower,
    upper,
    integrality,
    basis,
    x,
    max_cuts=200,
    tableau_mir=False,
    max_nnz="auto",
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

    ``max_nnz`` caps a cut's nonzeros: ``"auto"`` (the default) allows
    ``max(n // 2, min(n, 100))``, and ``None`` allows any. Slack terms
    mapped back through their rows make GMI cuts dense, so the cap can
    reject nearly all of them (197 of 198 at the root of ns1830653). The
    root cut loop lifts it and keeps only the cuts that end up binding.
    """
    import highspy
    from scipy.sparse.linalg import splu

    BS = highspy.HighsBasisStatus
    A = sp.csr_matrix(A)
    m, n = A.shape
    if max_nnz == "auto":
        max_nnz = max(n // 2, min(n, 100))
    elif max_nnz is None:
        max_nnz = n
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
        if idx.size == 0 or idx.size > max_nnz:
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
                if x_idx.size == 0 or x_idx.size > max_nnz:
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
        max_aggregation=_CUT_MAX_AGGREGATION, flow_covers=True, backend="jax",
        min_batch=_SEPARATION_MIN_BATCH,
    ):
        self.flow_covers = flow_covers  # also flow cover cuts (``flow_cover_cut``)
        if backend not in ("jax", "numpy"):
            raise ValueError(f"backend must be 'jax' or 'numpy', not {backend!r}")
        # ``backend``: batched JAX searches, or the per-row numpy ones (the
        # reference); rounds with fewer than ``min_batch`` base inequalities
        # use numpy, where compiling would cost more than it saves.
        self.backend, self.min_batch = backend, int(min_batch)
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
        # Variable bounds for c-MIR (``None`` without any, as on a pure
        # integer MIP).
        self.variable_bounds = variable_bounds(mip) or None

    def row(self, r):
        start, end = self.A.indptr[r], self.A.indptr[r + 1]
        return self.A.indices[start:end], self.A.data[start:end]

    def tight_rows(self, x):
        """The rows that start a separation at ``x``: slack within
        ``_CUT_TIGHT_ROWS`` of ``1 + |b|``, tightest first."""
        slack = self.b - self.A @ x
        relative = np.where(self.is_eq, np.abs(slack), slack) / (1.0 + np.abs(self.b))
        rows = np.flatnonzero(relative <= _CUT_TIGHT_ROWS)
        self.relative = relative
        return rows[np.argsort(relative[rows], kind="stable")][:_CUT_MAX_ROWS]

    def __call__(self, x):
        """Cuts violated by ``x``, as ``(family, idx, coef, rhs)``. With
        ``backend="jax"`` and at least ``_SEPARATION_MIN_BATCH`` base
        inequalities, the c-MIR and flow cover searches run batched
        (``batched_cmir_cuts``, ``batched_flow_cover_cuts``) and give the
        same cuts as the per-row numpy searches."""
        x = np.asarray(x, np.float64)
        jobs = []  # per (row, sense): its covers and its aggregation path
        for r in self.tight_rows(x):
            idx, a = self.row(r)
            if idx.size == 0:
                continue
            for s in (1.0, -1.0) if self.is_eq[r] else (1.0,):
                covers = [
                    ("cover", *found)
                    for found in cover_cuts(
                        idx, s * a, s * self.b[r], x, self.lower, self.upper, self.is_binary
                    )
                ]
                jobs.append((covers, self._aggregation_path(r, s, x)))
        bases = [base for _, path in jobs for base in path]
        if self.backend == "jax" and len(bases) >= self.min_batch:
            mir = batched_cmir_cuts(
                bases, x, self.lower, self.upper, self.is_int, self.variable_bounds
            )
            flows = [None] * len(bases)
            if self.flow_covers:
                which = [i for i, (idx, _, _) in enumerate(bases) if not self.is_int[idx].all()]
                found = batched_flow_cover_cuts(
                    [bases[i] for i in which], x, self.lower, self.upper, self.is_int,
                    self.variable_bounds,
                )
                for i, flow in zip(which, found):
                    flows[i] = flow
        else:
            mir = [
                cmir_cut(
                    idx, a, b, x, self.lower, self.upper, self.is_int,
                    variable_bounds=self.variable_bounds,
                )
                for idx, a, b in bases
            ]
            flows = [
                flow_cover_cut(
                    idx, a, b, x, self.lower, self.upper, self.is_int, self.variable_bounds
                )
                if self.flow_covers and not self.is_int[idx].all()
                else None
                for idx, a, b in bases
            ]
        cuts = []
        position = 0
        for covers, path in jobs:
            cuts += covers
            cuts += self._path_cuts(
                path, x, mir[position : position + len(path)],
                flows[position : position + len(path)],
            )
            position += len(path)
        out = []
        for family, idx, coef, rhs in cuts:
            cleaned = _clean_cut(idx, coef, rhs, self.lower, self.upper)
            if cleaned is not None:
                out.append((family, *cleaned))
        return out

    def _path_cuts(self, path, x, mir, flows):
        """The cuts of one aggregation path from its levels' c-MIR and flow
        cover cuts: the best c-MIR cut over the levels, each level's flow
        cover cut, and the cover cuts of the levels beyond the row."""
        best, best_efficacy = None, 0.0
        extra = []
        for level, ((idx, a, b), cut, flow) in enumerate(zip(path, mir, flows)):
            if cut is not None:
                c_idx, c_coef, c_rhs = cut
                efficacy = (c_coef @ x[c_idx] - c_rhs) / np.linalg.norm(c_coef)
                if efficacy > best_efficacy:
                    best, best_efficacy = cut, efficacy
            if flow is not None:
                extra.append(("flowcover", *flow))
            if level > 0:
                extra += [
                    ("cover", *found)
                    for found in cover_cuts(
                        idx, a, b, x, self.lower, self.upper, self.is_binary
                    )
                ]
        return ([("mir", *best)] if best is not None else []) + extra

    def aggregated_mir(self, r, s, x):
        """Cuts from row ``r`` (times ``s``) and its aggregates
        (``_aggregation_path``): the best c-MIR cut over the aggregates, the
        cover cuts of the aggregates beyond the row itself, and the flow
        cover cut (``flow_cover_cut``) of the row and each aggregate with a
        continuous variable. Needs ``tight_rows(x)`` first (it orders the
        partner rows)."""
        path = self._aggregation_path(r, s, x)
        mir = [
            cmir_cut(
                idx, a, b, x, self.lower, self.upper, self.is_int,
                variable_bounds=self.variable_bounds,
            )
            for idx, a, b in path
        ]
        flows = [
            flow_cover_cut(
                idx, a, b, x, self.lower, self.upper, self.is_int, self.variable_bounds
            )
            if self.flow_covers and not self.is_int[idx].all()
            else None
            for idx, a, b in path
        ]
        return self._path_cuts(path, x, mir, flows)

    def _aggregation_path(self, r, s, x):
        """Row ``r`` (times ``s``) and its aggregates with up to
        ``max_aggregation`` more rows, as ``[(idx, a, b), ...]``: each adds
        the row that cancels the continuous variable furthest from its
        bounds (or, with ``aggregate_integers`` and none left, the integer
        term ``|aⱼ|·min(fⱼ, 1 − fⱼ)`` largest at ``x``), preferring the
        tightest rows."""
        idx, a = self.row(r)
        a, b = s * a, s * self.b[r]
        used = {r}
        path = []
        for level in range(self.max_aggregation + 1):
            path.append((idx, a, b))
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
        return path


def select_cuts(cuts, x, n, min_efficacy=1e-4, max_cuts=100):
    """The cuts that ``x`` violates by an efficacy (distance) of at least
    ``min_efficacy``, most efficacious first, skipping near-duplicates, at
    most ``max_cuts`` of them.

    All candidates' efficacies come from one pass, and a chosen cut's
    cosines with every candidate from one sparse product, so the greedy pass
    costs ``max_cuts`` products instead of a stack of the chosen cuts per
    candidate (2–3 s a root on vpm2 and fixnet6 before)."""
    x = np.asarray(x, np.float64)
    k = len(cuts)
    if k == 0:
        return []
    lengths = np.array([cut[1].size for cut in cuts])
    rows = np.repeat(np.arange(k), lengths)
    cols = np.concatenate([np.asarray(cut[1], np.int64) for cut in cuts])
    vals = np.concatenate([np.asarray(cut[2], np.float64) for cut in cuts])
    rhs = np.array([cut[3] for cut in cuts], np.float64)
    norm = np.sqrt(np.bincount(rows, vals * vals, minlength=k))
    with np.errstate(divide="ignore", invalid="ignore"):
        efficacy = (np.bincount(rows, vals * x[cols], minlength=k) - rhs) / norm
    keep = np.flatnonzero(efficacy >= min_efficacy)
    if keep.size == 0:
        return []
    order = keep[np.argsort(-efficacy[keep], kind="stable")]
    with np.errstate(divide="ignore", invalid="ignore"):
        directions = sp.csr_matrix(
            (vals / norm[rows], (rows, cols)), shape=(k, n)
        )
    # The largest |cosine| of each candidate with a cut chosen so far.
    nearest = np.zeros(k)
    chosen = []
    for i in order:
        if nearest[i] > _CUT_MAX_PARALLELISM:
            continue
        chosen.append(i)
        if len(chosen) >= max_cuts:
            break
        cosine = (directions @ directions[i].T).toarray().ravel()
        np.maximum(nearest, np.abs(cosine), out=nearest)
    return [cuts[i] for i in chosen]


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


@functools.partial(jax.jit, static_argnames=("m",))
def _propagate_batch_jax(data, lower, upper, m):
    """``_propagate_jax`` over a batch of boxes (one compile per shape)."""
    return jax.vmap(lambda lo, up: _propagate_jax(data, lo, up, m))(lower, upper)


def _implied_slack_scores(propagator, lower, upper, eps=1e-9):
    """cuOpt's implied-slack score of each variable: ``Σ_rows (a / slack)²``,
    where ``slack`` is the row's room left at its least (or most) activity
    over the box. Variables that use up the most room per unit go first."""
    if propagator.vals.size == 0:
        return np.zeros(lower.shape[0])
    _, _, _, _, min_act, max_act, n_min, n_max = propagator._activities(lower, upper)
    rows, cols, a = propagator.rows, propagator.cols, propagator.vals
    ru, rl = propagator.row_upper, propagator.row_lower
    with np.errstate(invalid="ignore", divide="ignore"):
        slack_up = np.where(np.isfinite(ru) & (n_min == 0), ru - min_act, np.inf)
        slack_lo = np.where(np.isfinite(rl) & (n_max == 0), max_act - rl, np.inf)
        slack = np.minimum(slack_up, slack_lo)[rows]
        unit = np.where(np.isfinite(slack), np.abs(a) / np.maximum(slack, eps), 0.0)
    return np.bincount(cols, unit * unit, lower.shape[0])


def _draw_rounding(xv, lower, upper, rng, band=0.3):
    """Round ``xv``: at random (up with probability its fraction) when the
    fraction is within ``band`` of ½, else to the nearest integer; clipped
    to the box."""
    floor = np.floor(xv)
    f = xv - floor
    random = np.abs(f - 0.5) < band
    up = np.where(random, rng.random(xv.shape) < f, f >= 0.5)
    return np.clip(floor + up, lower, upper)


def bulk_fix_and_propagate(
    starts,
    lower,
    upper,
    integrality,
    A,
    row_lower,
    row_upper,
    cache=None,
    seed=0,
    backend="auto",
    deadline=None,
    single_tail=36,
    tol=1e-6,
    propagation_data=None,
    boxes=None,
    pad_to=None,
    propagator=None,
):
    """
    Round several LP points at once with bound propagation, in bulks, as in
    cuOpt's bound-propagation rounding (Çördük et al., 2025, Algorithm 1).

    For each start point, integer variables are fixed in bulks of about the
    square root of those still free (one at a time for the last
    ``single_tail``), narrow domains first and then most integral, re-sorted
    before every bulk by ``_implied_slack_scores``. Even-numbered starts
    first fix the values that round above their lower bound (as
    ``fix_and_propagate`` does: on assignment rows the ones go first and
    propagation settles the zeros); odd ones use the paper's order alone. Each bulk gets **two**
    candidate roundings from ``_draw_rounding`` (random near ½, nearest
    elsewhere); a ``ProbingCache`` first intersects what each fixed value
    implies, which rejects some candidates outright, and the survivors of
    every start are propagated together (``backend``: ``"numpy"``, ``"jax"``
    for one vmapped call per step, or ``"auto"``: numpy up to 10,000
    nonzeros). The first feasible candidate is kept. When both fail, the
    start backtracks and fixes one variable at a time until one succeeds;
    a single variable whose roundings both fail ends that start.

    ``propagation_data`` is ``_propagation_data`` of the rows, to reuse
    across calls (``None``: built here), and ``propagator`` a
    ``BoundPropagator`` of the same rows (``None``: built here). ``boxes``
    gives each start its own ``(lower, upper)`` (``None``: all share
    ``lower``/``upper``). On the device every step propagates
    ``2·max(len(starts), pad_to)`` boxes, so a fixed ``pad_to`` keeps one
    compiled shape across calls with different numbers of starts.

    Returns one point per start (every integer fixed, continuous variables
    clipped to their propagated bounds, for the caller to re-solve), or
    ``None`` for a start that failed or ran out of time.
    """
    starts = [np.asarray(x, np.float64) for x in starts]
    integrality = np.asarray(integrality, bool)
    if propagator is None:
        propagator = BoundPropagator(A, row_lower, row_upper, integrality, tol=tol)
    if not starts:
        return []
    if boxes is None:
        boxes = [(lower, upper)] * len(starts)
    start_boxes = []
    for lo, up in boxes:
        lo, up, empty = propagator(np.asarray(lo, np.float64), np.asarray(up, np.float64))
        start_boxes.append(None if empty else (lo, up))
    if backend == "auto":
        backend = "numpy" if A.nnz <= _SMALL_MIP_NNZ else "jax"
    if backend not in ("numpy", "jax"):
        raise ValueError(f"backend must be 'auto', 'numpy' or 'jax', not {backend!r}")
    batch = 2 * max(len(starts), int(pad_to or 0))
    if backend == "jax":
        data = propagation_data
        if data is None:
            with jax.enable_x64(True):
                data = _propagation_data(A, row_lower, row_upper, integrality)

    def propagate_all(boxes):
        if backend == "numpy" or not boxes:
            return [propagator(lo, up) for lo, up in boxes]
        padded = boxes + [boxes[-1]] * (batch - len(boxes))
        with jax.enable_x64(True):
            lo, up, empty = jax.device_get(
                _propagate_batch_jax(
                    data,
                    jnp.asarray(np.stack([b[0] for b in padded])),
                    jnp.asarray(np.stack([b[1] for b in padded])),
                    A.shape[0],
                )
            )
        return [(lo[i], up[i], bool(empty[i]) or bool(np.any(lo[i] > up[i]))) for i in range(len(boxes))]

    rng = np.random.default_rng(seed)
    ints = np.flatnonzero(integrality)
    states = []
    for x, box in zip(starts, start_boxes):
        if box is None:
            states.append({"status": "failed"})
            continue
        lower, upper = box
        frac = np.abs(x[ints] - np.round(x[ints]))
        wide = (upper[ints] - lower[ints]) > 2
        # Even starts fix the values that round up first (as
        # ``fix_and_propagate``), odd ones follow the paper's order alone:
        # each suits different models, and the starts run side by side.
        # (One start alone fixes the raised values first.)
        raised_first = len(states) % 2 == 0 or len(starts) == 1
        raised = np.clip(np.round(x[ints]), lower[ints], upper[ints]) > lower[ints]
        keys = (frac, wide, ~raised) if raised_first else (frac, wide)
        states.append(
            {
                "x": x,
                "raised_first": raised_first,
                "order": ints[np.lexsort(keys)],
                "lower": lower.copy(),
                "upper": upper.copy(),
                "recovery": False,
                "status": "running",
            }
        )
    while any(state["status"] == "running" for state in states):
        if deadline is not None and time.time() >= deadline:
            break
        boxes, owners = [], []
        for p, state in enumerate(states):
            if state["status"] != "running":
                continue
            lo, up = state["lower"], state["upper"]
            order = state["order"]
            free = order[lo[order] < up[order]]
            if free.size == 0:
                state["status"] = "done"
                continue
            if state["recovery"] or free.size <= single_tail:
                size = 1
            else:
                size = max(1, int(math.sqrt(free.size)))
                scores = _implied_slack_scores(propagator, lo, up)[free]
                if state["raised_first"]:
                    raised = np.clip(np.round(state["x"][free]), lo[free], up[free]) > lo[free]
                    free = free[np.lexsort((-scores, ~raised))]
                else:
                    free = free[np.argsort(-scores, kind="stable")]
            bulk = free[:size]
            xv = state["x"][bulk]
            first = _draw_rounding(xv, lo[bulk], up[bulk], rng)
            second = _draw_rounding(xv, lo[bulk], up[bulk], rng)
            if size == 1 and second[0] == first[0]:
                # The other rounding, when the box has room for it.
                other = first + (1.0 if xv[0] > first[0] else -1.0)
                second = other if lo[bulk][0] <= other[0] <= up[bulk][0] else None
            state["candidates"] = 0
            for values in (first, second):
                if values is None:
                    continue
                new_lo, new_up = lo.copy(), up.copy()
                new_lo[bulk] = values
                new_up[bulk] = values
                if cache is not None and len(cache):
                    ok = all(
                        cache.implied(j, v, new_lo, new_up) for j, v in zip(bulk, values)
                    )
                    if not ok or np.any(new_lo > new_up + tol * (1.0 + np.abs(new_up))):
                        continue
                    new_lo = np.minimum(new_lo, new_up)
                boxes.append((new_lo, new_up))
                owners.append(p)
                state["candidates"] += 1
            state["size"] = size
        results = propagate_all(boxes)
        accepted = set()
        for p, (new_lo, new_up, empty) in zip(owners, results):
            if p in accepted or empty:
                continue
            states[p]["lower"], states[p]["upper"] = np.asarray(new_lo), np.asarray(new_up)
            states[p]["recovery"] = False
            accepted.add(p)
        for p, state in enumerate(states):
            if state["status"] != "running" or "size" not in state or p in accepted:
                continue
            if state["size"] == 1:
                state["status"] = "failed"  # both roundings of one variable fail
            else:
                state["recovery"] = True  # back off to one variable at a time
            state.pop("size")
        for state in states:
            state.pop("size", None)

    points = []
    for state in states:
        if state["status"] != "done":
            points.append(None)
            continue
        point = np.clip(state["x"], state["lower"], state["upper"])
        point[integrality] = state["lower"][integrality]
        points.append(point)
    return points


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
        # Rows that are not ``exclusive`` (the objective cutoff row, which
        # touches nearly every variable) are left out of this test, so they
        # do not cut the step to one move; their scores are then inexact.
        priority = jnp.where(
            good, score * (1.0 + 1e-6 * jax.random.uniform(k_noise, (n,))), -jnp.inf
        )
        row_best = jax.ops.segment_max(priority[cols], rows, num_segments=m)
        top = (priority[cols] >= row_best[rows]) | ~data["exclusive"][rows]
        winner = good & (
            jax.ops.segment_min(top.astype(jnp.int32), cols, num_segments=n) > 0
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


def _jump_data(mip, lower, upper, tol, cutoff=None):
    """The device arrays ``_jump_chunk`` reads: the MIP's rows (as given, no
    cuts) scaled to unit ∞-norm, with a per-row tolerance matching
    ``MIP.violation``'s, and the box. With a ``cutoff``, the row
    ``c·x <= cutoff`` (its nonzero ``c`` entries) is appended last, so a
    point meets every row only when it also improves on the cutoff."""
    A = sp.csr_matrix(mip.A)
    row_lower, row_upper = mip.row_lower, mip.row_upper
    exclusive = np.ones(A.shape[0], dtype=bool)
    if cutoff is not None:
        A = sp.vstack([A, sp.csr_matrix(np.asarray(mip.c, np.float64)[None, :])], format="csr")
        A.eliminate_zeros()
        row_lower = np.append(row_lower, -np.inf)
        row_upper = np.append(row_upper, cutoff)
        exclusive = np.append(exclusive, False)
    dtype = mip.lp.c.dtype
    scale = np.abs(A).max(axis=1).toarray().ravel()
    scale = np.where(scale > 0, scale, 1.0)
    row_tol = 0.5 * tol * (1.0 + np.abs(mip.b).max(initial=0.0)) / scale
    if cutoff is not None:
        row_tol[-1] = 0.5 * tol * (1.0 + abs(cutoff)) / scale[-1]
    A = A.tocoo()

    def put(v):
        return jax.device_put(np.asarray(v).astype(dtype))

    return dict(
        rows=jax.device_put(A.row.astype(np.int32)),
        cols=jax.device_put(A.col.astype(np.int32)),
        a=put(A.data / scale[A.row]),
        row_lower=put(row_lower / scale),
        row_upper=put(row_upper / scale),
        lower=put(lower),
        upper=put(upper),
        is_int=jax.device_put(mip.integrality),
        tol=put(row_tol),
        exclusive=jax.device_put(exclusive),
        # The objective row's scale and the tolerance, to move the cutoff
        # (``_jump_with``).
        cutoff_scale=float(scale[-1]) if cutoff is not None else 1.0,
        feasibility_tolerance=float(tol),
    )


def _jump_with(data, lower, upper, cutoff=None):
    """``data`` from ``_jump_data`` with a new box and (when it has an
    objective row) a new cutoff, without copying the rows again."""
    dtype = data["lower"].dtype
    new = dict(
        data,
        lower=jax.device_put(np.asarray(lower).astype(dtype)),
        upper=jax.device_put(np.asarray(upper).astype(dtype)),
    )
    if cutoff is not None:
        scale, tol = data["cutoff_scale"], data["feasibility_tolerance"]
        new["row_upper"] = data["row_upper"].at[-1].set(cutoff / scale)
        new["tol"] = data["tol"].at[-1].set(0.5 * tol * (1.0 + abs(cutoff)) / scale)
    return new


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
    cutoff=None,
    data=None,
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
    defaults to ``max(10000, 20·(n + m))``. With a ``cutoff`` the search
    also has the row ``c·x <= cutoff``, so it looks for a point better than
    an incumbent. ``data`` (from ``_jump_data``, with the same ``cutoff``
    or none, and the same box via ``_jump_with``) saves copying the rows to
    the device again. Returns ``(x, feasible, steps)``: ``feasible`` means
    every row (and the cutoff) is met to ``tol`` (as ``MIP.violation``
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
    if data is None:
        data = _jump_data(mip, lower, upper, tol, cutoff)
    m = data["row_lower"].shape[0]  # with the objective row, if any
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
    """The best feasible point of ``mip`` that HiGHS's MIP solver finds
    within ``max_seconds`` (or its optimum, if it gets there first). Returns
    it in ``mip``'s variables, or ``None``. HiGHS's first solution is
    usually a trivial one from its own feasibility jump (133808 on
    ns1830653, against 36622 from its root sub-MIP a few seconds later), so
    it runs for the whole budget. This hands the search to HiGHS; ``solve``
    uses it only when asked (``highs_incumbent_seconds``)."""
    import highspy

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("time_limit", float(max_seconds))
    highs.passModel(to_highs(mip))
    highs.run()
    if highs.getInfo().primal_solution_status != 2:  # kSolutionStatusFeasible
        return None
    return np.array(highs.getSolution().col_value, dtype=np.float64)


def highs_submip(
    mip, lower, upper, cutoff=None, max_seconds=5.0, max_nodes=None, start=None
):
    """
    The sub-MIP of ``mip`` over the box ``[lower, upper]`` solved by HiGHS,
    as HiGHS's own RENS and RINS do: HiGHS presolves it (fixed columns and
    what they imply drop out, so a neighbourhood with most integers fixed
    becomes a small MIP) and searches it for up to ``max_seconds`` and
    ``max_nodes`` nodes. ``cutoff`` (``c·x`` of ``mip``'s minimisation, no
    offset) is passed as HiGHS's ``objective_bound``, so only better points
    come back; a dense cutoff row would get in presolve's way. ``start``,
    if given (an incumbent for RINS), is offered to HiGHS as a starting
    solution. Returns the best point found, or ``None``.
    """
    import highspy

    model = hh.jaddle_lp_to_highs(mip.lp, mip.integrality)
    model.col_lower_ = np.asarray(lower, np.float64)
    model.col_upper_ = np.asarray(upper, np.float64)
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("time_limit", float(max(max_seconds, 1e-3)))
    if max_nodes is not None:
        highs.setOptionValue("mip_max_nodes", int(max_nodes))
    finite_cutoff = cutoff is not None and np.isfinite(cutoff)
    if finite_cutoff:
        highs.setOptionValue("objective_bound", float(cutoff))
    highs.passModel(model)
    if start is not None:
        solution = highspy.HighsSolution()
        solution.col_value = list(np.asarray(start, np.float64))
        solution.value_valid = True
        highs.setSolution(solution)
    highs.run()
    if highs.getInfo().primal_solution_status != 2:  # kSolutionStatusFeasible
        return None
    x = np.array(highs.getSolution().col_value, dtype=np.float64)
    if finite_cutoff and mip.c @ x > cutoff + 1e-9 * (1.0 + abs(cutoff)):
        return None  # e.g. the start itself, no better than the cutoff
    return x


def highs_rins(mip, incumbent, x_lp, cutoff, max_seconds=5.0, min_fixed=0.3):
    """
    RINS (relaxation induced neighbourhood search) with HiGHS as the sub-MIP
    solver (``highs_submip``): the integer variables on which the
    ``incumbent`` and the LP point ``x_lp`` agree are fixed at the
    incumbent's value, and HiGHS searches the rest for points below
    ``cutoff`` (``c`` of ``mip``'s minimisation, no offset) for up to
    ``max_seconds``. Skipped (``None``) when fewer than ``min_fixed`` of the
    integers agree, since the sub-MIP would then be nearly the whole MIP.
    Returns the best point HiGHS found, or ``None``.
    """
    ints = mip.integrality
    incumbent = np.asarray(incumbent, np.float64)
    agree = ints & (np.abs(np.asarray(x_lp, np.float64) - incumbent) <= 1e-6)
    if ints.sum() == 0 or agree.sum() < min_fixed * ints.sum():
        return None
    lower = np.where(agree, incumbent, mip.lower)
    upper = np.where(agree, incumbent, mip.upper)
    return highs_submip(mip, lower, upper, cutoff, max_seconds=max_seconds, start=incumbent)


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
    if isinstance(device, str) and device == "auto":
        if nnz > _SMALL_MIP_NNZ:
            return None
        device = "cpu"
    return jo._resolve_device(device)


# %%
# Branch and bound
def _raw_of(child):
    """A strong child's unrounded bound."""
    return child.bound if child.raw is None else child.raw


def _child_box(box, var, xv, down):
    """The box of a child of ``box`` that branches on ``var`` at ``xv``."""
    lo, up = box[0].copy(), box[1].copy()
    if down:
        up[var] = min(up[var], math.floor(xv))
    else:
        lo[var] = max(lo[var], math.ceil(xv))
    return lo, up


def _batch_member(tree, i):
    """Member ``i`` of a batched pytree of device arrays."""
    return jax.tree.map(lambda leaf: leaf[i], tree)


class _StrongChild(
    collections.namedtuple("_StrongChild", "bound closed solution raw", defaults=(None,))
):
    """A strong-branching child: its bound (rounded as node bounds are),
    whether it is closed (empty or at the cutoff), its LP solution for a
    warm start (``None`` when closed or not solved, or a function giving it)
    and its unrounded bound, for scores and pseudocosts (``None``: the
    bound). Rounding to an objective step would make most gains 0 or one
    step."""

    __slots__ = ()


class _Node:
    """An open node: its bound, depth, bound changes against the root (as
    ``(index, lower, upper)``), the parent's warm start, the branching
    that made it (``(var, down, distance, parent_value)``) and the LP
    epochs it has had in earlier batches (``epochs``)."""

    __slots__ = ("bound", "depth", "changes", "warm", "branch", "epochs")

    def __init__(self, bound, depth, changes, warm, branch, epochs=0):
        self.bound, self.depth, self.changes = bound, depth, changes
        self.warm, self.branch = warm, branch
        self.epochs = epochs


_LP_RESERVED = (
    "initial_solution",
    "initial_opt_state",
    "max_epochs",
    "max_seconds",
    "device",
    "precision",
)


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
        rins_solver="highs",
        rins_nodes=200,
        rens_seconds=None,
        rens_nodes=500,
        dive_every=0,
        dive_depth=20,
        initial_cutoff=None,
        conflicts=True,
        conflict_pool_size=2000,
        conflict_max_nnz=None,
        flow_covers=True,
        separation_backend="jax",
        node_bound_stop=True,
        node_lp_tolerance=None,
        heuristic_effort=0.1,
        reliability_threshold=4,
        strong_candidates=8,
        strong_epochs=50,
        strong_max_depth=None,
        strong_screen=32,
        strong_backend="jax",
        pump_jump_iterations=3000,
        pump_jump_share=0.2,
        probing=True,
        probing_seconds=None,
        probing_max_variables=None,
        probing_backend="auto",
        probing_max_useless=1000,
        bulk_rounding=True,
        bulk_starts=4,
        repair_share=0.2,
        batch_epochs=10,
        children_backend="auto",
        batch_roundings=False,
        node_lp_options=None,
        tree_heuristics=False,
        tree_heuristic_every=20,
        tree_heuristic_share=0.05,
        tree_jump_iterations=3000,
        node_crossover=False,
        node_crossover_every=1,
        node_crossover_seconds=2.0,
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
        # Above the largest objective over the root box (set in ``_run``):
        # no feasible point reaches it.
        self.box_cutoff = np.inf
        if branching not in ("pseudocost", "most_fractional", "reliability"):
            raise ValueError(f"unknown branching rule {branching!r}")
        # Reliability branching: a variable is strong-branched until each
        # direction has this many pseudocost observations.
        self.reliability_threshold = int(reliability_threshold)
        self.strong_candidates = int(strong_candidates)
        self.strong_epochs = int(strong_epochs)
        self.strong_max_depth = strong_max_depth
        self.strong_screen = max(int(strong_screen), self.strong_candidates)
        if strong_backend not in ("jax", "numpy"):
            raise ValueError(
                f"strong_backend must be 'jax' or 'numpy', not {strong_backend!r}"
            )
        # Tree cuts change the LP's shape, which the device kernels fix.
        self.strong_backend = "numpy" if tree_cuts and int(batch_size) == 1 else strong_backend
        self._strong_kernels = None  # (mip, _StrongKernels)
        self.strong_branch_nodes = 0
        self.strong_branch_lps = 0
        self.strong_branch_seconds = 0.0
        self.strong_fixings = 0
        if node_selection not in ("hybrid", "best_first", "depth_first"):
            raise ValueError(f"unknown node selection {node_selection!r}")
        lp_options = dict(lp_options or {})
        reserved = [key for key in _LP_RESERVED if key in lp_options]
        if reserved:
            raise ValueError(f"lp_options cannot set {', '.join(reserved)}")
        node_lp_options = dict(node_lp_options or {})
        reserved = [key for key in _LP_RESERVED if key in node_lp_options]
        if reserved:
            raise ValueError(f"node_lp_options cannot set {', '.join(reserved)}")

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
        # Small MIPs get Gomory cuts by default: their crossovers take
        # milliseconds.
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
        # Rounds are cheap with Gomory cuts on (slack ones are dropped every
        # round), and the bound keeps climbing past five on larger MIPs.
        many_rounds = self.gomory
        if cut_rounds is None:
            self.cut_rounds = 40 if many_rounds else 5
        if cut_min_progress is None:
            # Late rounds slow the root LP for little gain, so stop once three
            # rounds together gain less than 0.1%.
            self.cut_min_progress = 1e-3 if many_rounds else 1e-4
        # Global cuts in the tree need one sparsity pattern per compiled
        # solver; the batched solver captures it, so they are serial only.
        self.use_flow_covers = bool(flow_covers)
        if separation_backend not in ("jax", "numpy"):
            raise ValueError(
                f"separation_backend must be 'jax' or 'numpy', not {separation_backend!r}"
            )
        self.separation_backend = separation_backend
        # Node LPs stop once their Lagrangian bound passes the cutoff.
        self.node_bound_stop = bool(node_bound_stop)
        self.bound_stops = 0
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
        if not 0.0 <= float(repair_share) <= 1.0:
            raise ValueError(f"repair_share must be in [0, 1], not {repair_share!r}")
        self.repair_share = float(repair_share)
        self.repair_max_epochs = repair_max_epochs
        self.highs_seconds = highs_incumbent_seconds
        # RENS at the root (``None``: auto): with HiGHS solving the sub-MIP,
        # 10% of the time limit up to 30 s (10 s without one), as HiGHS's own
        # root RENS is the heuristic that finds its early incumbents; the
        # slower Jaddle sub-search keeps to small MIPs (5 s).
        if rens_seconds is None:
            if rins_solver == "jaddle":
                rens_seconds = 5.0 if small else 0.0
            elif deadline is None:
                rens_seconds = 10.0
            else:
                rens_seconds = min(30.0, 0.1 * max(deadline - start, 0.0))
        if rins_seconds is None:
            rins_seconds = 5.0  # RINS on, 5 s a call (``None``: auto)
        # The share of the elapsed time RINS in the tree may take (the root
        # RENS is outside it), and the time and calls of sub-MIPs so far.
        self.heuristic_effort = float(heuristic_effort)
        self.root_vertex_x = None  # the cut loop's last crossover vertex
        self.submip_seconds = 0.0
        self.submip_calls = 0
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
        # Batched node LPs run at most this many epochs a call (``None``:
        # ``node_max_epochs``); unfinished ones go back to the queue.
        self.batch_epochs = int(node_max_epochs if batch_epochs is None else batch_epochs)
        if self.batch_epochs < 1:
            raise ValueError("batch_epochs must be at least 1")
        self.batch_requeues = 0
        # Batched search on the device: the members' children are built,
        # propagated and bounded in one call (``_ChildKernels``), and their
        # tree roundings run as one bulk rounding (``"auto"``: above 10,000
        # nonzeros; ``"jax"``, ``"numpy"``).
        if children_backend not in ("auto", "jax", "numpy"):
            raise ValueError(
                f"children_backend must be 'auto', 'jax' or 'numpy', not {children_backend!r}"
            )
        self.children_backend = children_backend
        # Off by default: one bulk rounding for a batch's tree roundings took
        # ~1 s a batch on ns1830653, against ~55 ms a node for the classic
        # fix-and-propagate, which mostly runs only every few nodes.
        self.batch_roundings = bool(batch_roundings)
        # GPU heuristics from node LP points once there is an incumbent
        # (``tree_heuristics``): bulk rounding, then a feasibility jump with
        # the objective cutoff row, every ``tree_heuristic_every`` nodes
        # within ``tree_heuristic_share`` of the elapsed time.
        if tree_heuristic_every < 1:
            raise ValueError("tree_heuristic_every must be at least 1")
        if tree_heuristic_share < 0:
            raise ValueError("tree_heuristic_share must be non-negative")
        self.use_tree_heuristics = bool(tree_heuristics)
        self.tree_heuristic_every = int(tree_heuristic_every)
        self.tree_heuristic_share = float(tree_heuristic_share)
        self.tree_jump_iterations = int(tree_jump_iterations)
        self.tree_heuristic_seen = set()  # (rounded integers, cutoff) tried
        self.tree_heuristic_calls = 0
        self.tree_heuristic_finds = 0
        self.tree_heuristic_seconds = 0.0
        self._tree_jump_data = None  # (cutoff-row jump data on the device)
        # Crossover of node LP points to exact vertices with HiGHS
        # (``node_crossover``), every ``node_crossover_every`` nodes, each
        # call limited to ``node_crossover_seconds``.
        if node_crossover_every < 1:
            raise ValueError("node_crossover_every must be at least 1")
        if node_crossover_seconds <= 0:
            raise ValueError("node_crossover_seconds must be positive")
        self.use_node_crossover = bool(node_crossover)
        self.node_crossover_every = int(node_crossover_every)
        self.node_crossover_limit = float(node_crossover_seconds)
        self.node_crossovers = 0  # calls
        self.node_crossover_vertices = 0  # calls that proved a vertex optimal
        self.node_crossover_integral = 0  # vertices integral, PDHG point not
        self.node_crossover_seconds = 0.0
        self._child_kernels = None  # (mip, _ChildKernels)
        self.deferring = False  # a batch is being evaluated
        self.deferred_branches = []
        self.deferred_roundings = []
        self.batched_children = 0
        self.batched_roundings = 0
        self._vmapped = {}  # max_epochs -> (mip, jitted vmapped node solver)
        self.pump_last = None
        self.use_pump = feasibility_pump
        self.pump_iterations, self.pump_tolerance = pump_iterations, pump_tolerance
        # The fused pump: a short feasibility jump from each rejected rounding
        # (``pump_jump_iterations`` steps, within ``pump_jump_share`` of the
        # pump's time), whose near miss also breaks the pump's cycles.
        self.pump_jump_iterations = int(pump_jump_iterations)
        self.pump_jump_share = float(pump_jump_share)
        self.pump_jumps = 0
        self.pump_jump_finds = 0
        # Root probing (``probe``), its budget (``None``: 5% of the time
        # limit up to 30 s, 10 s without one) and what it found.
        self.use_probing = bool(probing)
        if probing_seconds is None:
            probing_seconds = 10.0 if deadline is None else min(30.0, 0.05 * max(deadline - start, 0.0))
        self.probing_budget = float(probing_seconds)
        self.probing_max_variables = probing_max_variables
        self.probing_backend = probing_backend
        self.probing_max_useless = probing_max_useless
        self.probe_cache = None
        # Bulk fix-and-propagate (``bulk_fix_and_propagate``) from
        # ``bulk_starts`` starts at once, after the classic one at the root
        # and in the pump.
        self.use_bulk_rounding = bool(bulk_rounding)
        self.bulk_starts = max(1, int(bulk_starts))
        self._bulk_data = None  # (mip, device rows) for the jax backend
        self.probing_stats = {"probed": 0, "fixed": 0, "tightened": 0, "seconds": 0.0}
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
        # Node LPs only, over ``lp_kwargs``: ``node_lp_tolerance`` (``None``:
        # ``lp_tolerance``), then ``node_lp_options``.
        self.node_lp_kwargs = (
            {}
            if node_lp_tolerance is None
            else dict(
                primal_feasibility_tolerance=node_lp_tolerance,
                dual_feasibility_tolerance=node_lp_tolerance,
                dual_gap_tolerance=node_lp_tolerance,
            )
        )
        self.node_lp_options = node_lp_options
        self.node_lp_kwargs.update(node_lp_options)
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
        # Conflicts (dual proofs and no-goods) for propagation at each node.
        self.conflicts = (
            ConflictPool(
                mip.num_variables,
                mip.integrality,
                max_rows=conflict_pool_size,
                max_nnz=conflict_max_nnz,
            )
            if conflicts
            else None
        )
        self.conflict_pruned = 0
        self.conflict_tightenings = 0

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
        cutoff = self.cutoff(box=False)
        if not (self.use_objective_propagation and np.isfinite(cutoff)):
            return self.row_propagator
        self.objective_propagator.row_upper[-1] = cutoff
        return self.objective_propagator

    def bulk_round(
        self, starts, lower, upper, deadline=None, seed=0, boxes=None, pad_to=None
    ):
        """``bulk_fix_and_propagate`` of ``starts`` over the searched rows
        (plus the objective row at the cutoff once there is an incumbent, as
        ``heuristic_propagator``), with the probing cache. ``boxes`` gives
        each start its own box; ``pad_to`` fixes the device batch."""
        mip = self.mip
        propagator = self.heuristic_propagator()
        objective = propagator is self.objective_propagator
        if objective:
            A = sp.vstack([mip.A, sp.csr_matrix(mip.c[None, :])], format="csr")
            row_lower = propagator.row_lower
            row_upper = propagator.row_upper
        else:
            A, row_lower, row_upper = mip.A, mip.row_lower, mip.row_upper
        data = None
        if A.nnz > _SMALL_MIP_NNZ:
            key = (mip, objective)
            if self._bulk_data is None or self._bulk_data[0] != key:
                with jax.enable_x64(True):
                    rows = _propagation_data(A, row_lower, row_upper, self.is_int)
                self._bulk_data = (key, rows)
            data = self._bulk_data[1]
            if objective:
                # Only the cutoff moves between calls.
                with jax.enable_x64(True):
                    data = dict(data, row_upper=jnp.asarray(row_upper, jnp.float64))
        return bulk_fix_and_propagate(
            starts,
            lower,
            upper,
            self.is_int,
            A,
            row_lower,
            row_upper,
            cache=self.probe_cache,
            seed=seed,
            deadline=None if deadline is None or not np.isfinite(deadline) else deadline,
            propagation_data=data,
            boxes=boxes,
            pad_to=pad_to,
            propagator=propagator,
        )

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

    def cutoff(self, box=True):
        """Nodes whose bound reaches this hold no solution worth finding:
        below the incumbent by the gaps, and (``box``) below the largest
        objective any point of the root box has (``box_cutoff``), which
        lets infeasible node LPs stop on their bound before an incumbent."""
        if self.incumbent is None:
            return min(self.initial_cutoff, self.box_cutoff) if box else self.initial_cutoff
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
        # Nodes a batch has evaluated but not yet split are in no list.
        bound = min(bound, min((d["bound"] for d in self.deferred_branches), default=np.inf))
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

    # -- conflicts -----------------------------------------------------------
    def node_bound_target(self):
        """The ``bound_target`` for node LPs (``jl.solve``): a Lagrangian
        bound this high closes the node (``round_bound`` of it reaches the
        cutoff), plus a margin for the host's own recomputation of the bound
        over the node box. ``None`` without a cutoff or with
        ``node_bound_stop`` off."""
        cutoff = self.cutoff()
        if not self.node_bound_stop or not np.isfinite(cutoff):
            return None
        step = self.objective_step
        if step is None:
            target = cutoff
        else:
            # round_bound(b) >= k·step once b/step passes k − 1 by its slack.
            k = math.ceil(cutoff / step - 1e-12)
            target = step * (k - 1 + 2e-6 + 2e-9 * abs(k))
        return target + 1e-6 * (1.0 + abs(target))

    def proof_cutoff(self):
        """The largest objective a solution better than the incumbent can
        have: the cutoff, or with an objective step the multiple of it at or
        below the cutoff (rounded up a hair, which only weakens proofs)."""
        cutoff = self.cutoff()
        step = self.objective_step
        if step is None or not np.isfinite(cutoff):
            return cutoff
        return step * math.floor(cutoff / step + 1e-9)

    def learn(self, terms, lower, upper, kind="objective"):
        """A dual proof from Lagrangian terms that close the box
        ``[lower, upper]`` (see ``ConflictPool``)."""
        if self.conflicts is None:
            return
        self.conflicts.set_cutoff(self.proof_cutoff())
        self.conflicts.add(
            *terms, lower, upper, self.root_lower, self.root_upper, kind
        )

    def learn_farkas(self, detection, lower, upper):
        """A Farkas proof from ``detect``'s dual ray for an infeasible box.
        The proof holds for either sign of the ray; the one that excludes
        the box is kept."""
        if self.conflicts is None or detection.get("dual_ray_ineq") is None:
            return
        y = np.concatenate(
            [
                np.asarray(detection["dual_ray_eq"], np.float64)[: self.mip.n_eq],
                np.asarray(detection["dual_ray_ineq"], np.float64)[: self.m_ineq],
            ]
        )
        zero = np.zeros(self.mip.num_variables)
        for sign in (1.0, -1.0):
            terms = lagrangian_terms(
                zero, self.mip.A, self.mip.b, self.mip.n_eq, sign * y, A_T=self.A_T
            )
            if self.conflicts.add(
                *terms, lower, upper, self.root_lower, self.root_upper, "farkas"
            ):
                return

    def learn_nogood(self, lower, upper, max_size=64):
        """A no-good from a child box that propagation found infeasible: its
        binary fixings (against the root box), cut down by a deletion filter
        to a set that propagation still finds infeasible."""
        if self.conflicts is None or self.propagator is None:
            return
        root_lower, root_upper = self.root_lower, self.root_upper
        binary = self.is_int & (root_lower == 0.0) & (root_upper == 1.0)
        fixed = np.flatnonzero(binary & (lower == upper))
        if fixed.size == 0 or fixed.size > max_size:
            return

        def infeasible(keep):
            lo, up = root_lower.copy(), root_upper.copy()
            lo[keep] = lower[keep]
            up[keep] = upper[keep]
            return self.propagator(lo, up)[2]

        if not infeasible(fixed):
            return  # the box needed more than its binary fixings
        kept = list(fixed)
        for j in list(fixed):
            trial = [k for k in kept if k != j]
            if trial and infeasible(np.array(trial)):
                kept = trial
        kept = np.array(kept)
        self.conflicts.add_nogood(kept[lower[kept] == 1.0], kept[lower[kept] == 0.0])

    def conflict_propagate(self, lower, upper):
        """Propagate the conflict pool over a node box, alternating with the
        rows while either tightens it. Returns ``(lower, upper,
        infeasible)``."""
        if self.conflicts is None or not len(self.conflicts):
            return lower, upper, False
        self.conflicts.set_cutoff(self.proof_cutoff())
        for _ in range(3):
            new_lower, new_upper, infeasible = self.conflicts.propagate(lower, upper)
            if infeasible:
                return new_lower, new_upper, True
            changed = int(
                np.count_nonzero((new_lower != lower) | (new_upper != upper))
            )
            if not changed:
                break
            self.conflict_tightenings += changed
            lower, upper = new_lower, new_upper
            if self.propagator is not None:
                lower, upper, infeasible = self.propagator(lower, upper)
                if infeasible:
                    return lower, upper, True
        return lower, upper, False

    def prune_by_conflicts(self, node, lower, upper):
        """Conflict propagation at a node about to be solved. Returns the
        tightened box, or ``None`` when the node is closed without its LP."""
        lower, upper, infeasible = self.conflict_propagate(lower, upper)
        if infeasible:
            self.conflict_pruned += 1
            self.close(max(node.bound, self.cutoff_bound()))
            return None
        return lower, upper

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
    def pseudocost_scores(self, x, candidates):
        """The product score of each candidate's estimated gains."""
        down, up = self.pseudocost_estimates(x, candidates)
        return np.maximum(down, 1e-6) * np.maximum(up, 1e-6)

    def pseudocost_estimates(self, x, candidates):
        """Each candidate's estimated down and up bound gains: fraction times
        pseudocost, the mean pseudocost where a direction has no observation
        yet."""
        f = x[candidates] - np.floor(x[candidates])
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
        return f * pc[0], (1.0 - f) * pc[1]

    def choose_variable(self, x, fractional):
        candidates = np.flatnonzero(fractional)
        if self.branching == "most_fractional" or not self.pc_count.any():
            f = x[candidates] - np.floor(x[candidates])
            return candidates[np.argmax(np.minimum(f, 1.0 - f))]
        return candidates[np.argmax(self.pseudocost_scores(x, candidates))]

    def record_gain(self, var, down, gain):
        self.pc_sum[0 if down else 1, var] += gain
        self.pc_count[0 if down else 1, var] += 1

    def update_pseudocost(self, node, value):
        var, down, distance, parent_value = node.branch
        if parent_value is None or distance <= 0:
            return
        self.record_gain(var, down, max(value - parent_value, 0.0) / distance)

    def strong_branching_on(self, node):
        return (
            self.branching == "reliability"
            and self.strong_candidates > 0
            and (self.strong_max_depth is None or node.depth <= self.strong_max_depth)
            and not self.out_of_time()
        )

    def choose_branch(
        self, node, x, fractional, lower, upper, warm, terms, bound, pending=None,
        raw=None,
    ):
        """
        The branching variable for a node, by ``self.branching``. Returns
        ``(var, strong, lower, upper, bound)``: ``strong`` holds the chosen
        variable's strong-branching children (``None`` without them), and
        the box and bound may be tightened by them. Returns ``None`` when
        strong branching closed the node or put it back in the queue.

        Reliability branching strong-branches the unreliable candidates (a
        direction with fewer than ``reliability_threshold`` pseudocost
        observations), at most ``strong_candidates`` of them, and scores them
        by their children's bounds; the others keep their pseudocost scores
        (raised to their screened bound gains on the device backend). A
        candidate with one child closed is fixed to the other side, and the
        node's bound rises to the best candidate's lesser child bound.
        ``pending`` is the device pipeline ``launch_strong`` started for this
        node (``strong_backend="jax"``); ``raw`` is the node's unrounded
        bound, which gains are measured from (``None``: ``bound``).
        """
        if raw is None or not np.isfinite(raw):
            raw = bound
        candidates = np.flatnonzero(fractional)
        if not self.strong_branching_on(node):
            return self.choose_variable(x, fractional), None, lower, upper, bound
        est_down, est_up = self.pseudocost_estimates(x, candidates)
        score = np.maximum(est_down, 1e-6) * np.maximum(est_up, 1e-6)
        started = time.time()
        if self.strong_backend == "jax":
            if pending is None:  # every candidate was reliable
                return candidates[np.argmax(score)], None, lower, upper, bound
            self.strong_branch_nodes += 1
            try:
                children, solved = self.finish_strong(pending, x, terms, bound, raw)
            finally:
                self.strong_branch_seconds += time.time() - started
        else:
            unreliable = np.flatnonzero(
                self.pc_count[:, candidates].min(axis=0) < self.reliability_threshold
            )
            if unreliable.size == 0:
                return candidates[np.argmax(score)], None, lower, upper, bound
            chosen = unreliable[np.argsort(-score[unreliable], kind="stable")]
            chosen = candidates[chosen[: self.strong_candidates]]
            self.strong_branch_nodes += 1
            try:
                children = self.strong_branch(
                    chosen, x, lower, upper, warm, terms, bound, raw
                )
            finally:
                self.strong_branch_seconds += time.time() - started
            solved = set(chosen.tolist())

        lower, upper = lower.copy(), upper.copy()
        strong_bound, fixed = bound, False
        results = {}
        position = {var: k for k, var in enumerate(candidates.tolist())}
        for var, (down, up) in children.items():
            k = position.get(var)
            if k is None:  # screened, but reduced-cost fixing took it out
                continue
            if down.closed and up.closed:
                self.close(max(min(down.bound, up.bound), bound))
                return None
            # Either child of a candidate holds every solution of its side,
            # so the lesser child bound bounds the node.
            strong_bound = max(strong_bound, min(down.bound, up.bound))
            if down.closed or up.closed:
                # The closed side leaves the search: its bound counts as closed.
                self.close(max((down if down.closed else up).bound, bound))
                if down.closed:
                    lower[var] = max(lower[var], math.ceil(x[var]))
                else:
                    upper[var] = min(upper[var], math.floor(x[var]))
                self.strong_fixings += 1
                fixed = True
                score[k] = -np.inf
                continue
            gain_down, gain_up = _raw_of(down) - raw, _raw_of(up) - raw
            if var not in solved:  # screened only: no child LPs
                gain_down, gain_up = max(gain_down, est_down[k]), max(gain_up, est_up[k])
            score[k] = max(gain_down, 1e-6) * max(gain_up, 1e-6)
            results[var] = {True: down, False: up}
        if strong_bound >= self.cutoff():
            self.close(strong_bound)
            return None
        if fixed:
            infeasible = False
            if self.propagator is not None:
                lower, upper, infeasible = self.propagator(lower, upper)
            if not infeasible:
                lower, upper, infeasible = self.conflict_propagate(lower, upper)
            if infeasible:
                self.close(max(strong_bound, self.cutoff_bound()))
                return None
            # Only candidates still inside the tightened box can be split.
            inside = (x[candidates] > lower[candidates]) & (
                x[candidates] < upper[candidates]
            )
            score[~inside] = -np.inf
        best = int(np.argmax(score))
        if not np.isfinite(score[best]):
            # Every candidate was fixed: solve the node again in its box.
            self.push(
                _Node(strong_bound, node.depth, self.compress(lower, upper), warm, None)
            )
            return None
        var = candidates[best]
        return var, results.get(var), lower, upper, strong_bound

    def strong_kernels(self):
        """The device kernels for the current LP, built once per shape."""
        if self._strong_kernels is None or self._strong_kernels[0] is not self.mip:
            screen = self.strong_screen
            # Screened boxes are (screen, 2, n) float64 arrays: keep them small.
            width = max(self.mip.num_variables, self.mip.A.nnz)
            while screen > self.strong_candidates and 2 * screen * width > 2**24:
                screen //= 2
            kernels = _StrongKernels(
                self.mip,
                self.propagator is not None,
                self.reduced_cost_tol,
                screen,
                self.strong_candidates,
                self.dtype,
            )
            self._strong_kernels = (self.mip, kernels)
        return self._strong_kernels[1]

    def launch_strong(self, x, fractional, lower, upper, warm, terms, raw):
        """
        Dispatch strong branching for a node on the device without waiting
        for it (``finish_strong`` reads it back): the candidates are screened
        (``_StrongKernels.screen``), the picked ones' children solved for
        ``strong_epochs`` in one vmapped call warm-started from ``warm``, and
        their bounds computed (``_StrongKernels.child_bounds``). Every shape
        is fixed for the search, so nothing recompiles. Returns ``None`` when
        every candidate's pseudocosts are reliable.
        """
        candidates = np.flatnonzero(fractional)
        unreliable = self.pc_count[:, candidates].min(axis=0) < self.reliability_threshold
        if not unreliable.any():
            return None
        started = time.time()
        kernels = self.strong_kernels()
        size = kernels.screen_size
        est_down, est_up = self.pseudocost_estimates(x, candidates)
        if candidates.size > size:
            # The unreliable first, then by pseudocost score.
            score = np.maximum(est_down, 1e-6) * np.maximum(est_up, 1e-6)
            keep = np.lexsort((-score, ~unreliable))[:size]
            candidates, unreliable = candidates[keep], unreliable[keep]
            est_down, est_up = est_down[keep], est_up[keep]
        count = candidates.size

        def padded(values, fill=0):
            out = np.full(size, fill, dtype=np.asarray(values).dtype)
            out[:count] = values
            return out

        valid = padded(np.ones(count, bool), False)
        screened = kernels.screen(
            np.asarray(lower, np.float64),
            np.asarray(upper, np.float64),
            padded(candidates.astype(np.int32)),
            valid,
            padded(x[candidates].astype(np.float64), 0.5),
            np.asarray(terms[1], np.float64),
            np.float64(terms[0]),
            padded(est_down.astype(np.float64)),
            padded(est_up.astype(np.float64)),
            padded(unreliable, False),
            np.float64(raw if np.isfinite(raw) else -1e300),
            np.float64(self.cutoff()),
        )
        values = self.values._replace(
            lower_bounds=screened["lp_lower"], upper_bounds=screened["lp_upper"]
        )
        width = 2 * kernels.k
        warm_starts = jax.tree.map(
            lambda leaf: jnp.broadcast_to(leaf, (width,) + jnp.shape(leaf)), warm[0]
        )
        out = self.batched_solver(self.strong_epochs)(
            values, warm_starts, *self.bound_target_arg()
        )
        bounds = kernels.child_bounds(
            out.solution, screened["sel_lower"], screened["sel_upper"]
        )
        self.strong_branch_seconds += time.time() - started
        return {
            "box": (np.array(lower, np.float64), np.array(upper, np.float64)),
            "candidates": candidates,
            "screened": screened,
            "out": out,
            "bounds": bounds,
        }

    def finish_strong(self, pending, x, terms, bound, raw):
        """Read back a ``launch_strong`` pipeline (one transfer) and turn it
        into ``({var: (down, up)}, solved)`` of ``_StrongChild``, as
        ``strong_branch`` gives, for every screened candidate; ``solved``
        holds the candidates whose children got LPs. Child LP solutions stay
        on the device until a chosen child needs one as its warm start."""
        candidates = pending["candidates"]
        screened, out, bounds = pending["screened"], pending["out"], pending["bounds"]
        small = jax.device_get(
            {
                "infeasible": screened["infeasible"],
                "pre": screened["pre"],
                "pre_rigorous": screened["pre_rigorous"],
                "sel": screened["sel"],
                "sel_valid": screened["sel_valid"],
                "bound": bounds["bound"],
                "rigorous": bounds["rigorous"],
                "epochs": out.epochs,
            }
        )
        cutoff = self.cutoff()
        children = {}
        for i, var in enumerate(candidates.tolist()):
            sides = []
            for side, down in enumerate((True, False)):
                if small["infeasible"][i, side]:
                    self.learn_nogood(*_child_box(pending["box"], var, x[var], down))
                    sides.append(_StrongChild(np.inf, True, None))
                    continue
                pre_raw = max(float(small["pre"][i, side]), raw)
                pre = max(self.round_bound(pre_raw), bound)
                rigorous = bool(small["pre_rigorous"][i, side])
                if pre >= cutoff:
                    self.bound_rigorous &= rigorous
                    if rigorous:
                        lo, up = jax.device_get(
                            (screened["lower"][i, side], screened["upper"][i, side])
                        )
                        self.learn(terms, lo, up)
                    sides.append(_StrongChild(pre, True, None, pre_raw))
                else:
                    sides.append(_StrongChild(pre, False, None, pre_raw))
            children[var] = sides

        solved = set()
        for slot, (i, ok) in enumerate(zip(small["sel"], small["sel_valid"])):
            if not ok:
                continue
            var = int(candidates[i])
            solved.add(var)
            for side, down in enumerate((True, False)):
                member = 2 * slot + side
                self.strong_branch_lps += 1
                self.lp_epochs += int(small["epochs"][member])
                pre, pre_raw = children[var][side].bound, children[var][side].raw
                child_raw = max(float(small["bound"][member]), pre_raw)
                rigorous = bool(small["rigorous"][member])
                child_bound = max(self.round_bound(child_raw), pre)
                if child_bound > pre:
                    self.bound_rigorous &= rigorous
                xv = x[var]
                distance = xv - math.floor(xv) if down else math.ceil(xv) - xv
                if distance > 0 and np.isfinite(child_raw):
                    self.record_gain(var, down, (child_raw - raw) / distance)
                closed = child_bound >= cutoff
                if closed and rigorous:
                    r, constant, lo, up = jax.device_get(
                        (
                            bounds["r"][member],
                            bounds["constant"][member],
                            screened["sel_lower"][member],
                            screened["sel_upper"][member],
                        )
                    )
                    self.learn((float(constant), np.asarray(r)), lo, up)
                solution = None
                if not closed:
                    solution = functools.partial(_batch_member, out.solution, member)
                children[var][side] = _StrongChild(child_bound, closed, solution, child_raw)
        return {var: tuple(sides) for var, sides in children.items()}, solved

    def strong_branch(self, variables, x, lower, upper, warm, terms, bound, raw=None):
        """
        Both children of each variable in ``variables``, as
        ``{var: (down, up)}`` of ``_StrongChild``. A child is closed when
        propagation or the conflict pool empties its box, or when its bound
        reaches the cutoff; the bound is first the parent's duals over the
        child's box, then the duals of a child LP stopped after
        ``strong_epochs`` (a Lagrangian bound, valid however far the LP
        got). Each child LP also records a pseudocost observation: its
        bound gain over the parent per unit of change, which a truncated LP
        underestimates a little. Gains are measured between unrounded
        bounds, from ``raw`` (the node's; ``None``: ``bound``).
        """
        if raw is None:
            raw = bound
        cutoff = self.cutoff()
        children, pending = {}, []
        for var in variables:
            xv = x[var]
            for down in (True, False):
                lo, up = lower.copy(), upper.copy()
                if down:
                    up[var] = min(up[var], math.floor(xv))
                else:
                    lo[var] = max(lo[var], math.ceil(xv))
                child = None
                if lo[var] > up[var]:
                    child = _StrongChild(np.inf, True, None)
                elif self.propagator is not None:
                    branched = (lo, up)
                    lo, up, infeasible = self.propagator(lo, up)
                    if infeasible:
                        self.learn_nogood(*branched)
                        child = _StrongChild(np.inf, True, None)
                if child is None:
                    lo, up, infeasible = self.conflict_propagate(lo, up)
                    if infeasible:
                        child = _StrongChild(self.cutoff_bound(), True, None)
                if child is None:
                    pre_raw, rigorous = self.bound_over(terms, lo, up)
                    pre_raw = max(pre_raw, raw)
                    pre = max(self.round_bound(pre_raw), bound)
                    if pre >= cutoff:
                        self.bound_rigorous &= rigorous
                        if rigorous:
                            self.learn(terms, lo, up)
                        child = _StrongChild(pre, True, None, pre_raw)
                    else:
                        pending.append((var, down, lo, up, pre, pre_raw))
                children.setdefault(var, {})[down] = child

        results = self.solve_children([box[2:4] for box in pending], warm)
        self.strong_branch_lps += len(pending)
        for (var, down, lo, up, pre, pre_raw), result in zip(pending, results):
            solution = result["solution"]
            child_terms = self.dual_terms(solution)
            child_raw, rigorous = self.bound_over(child_terms, lo, up)
            child_raw = max(child_raw, pre_raw)
            child_bound = max(self.round_bound(child_raw), pre)
            if child_bound > pre:
                self.bound_rigorous &= rigorous
            distance = x[var] - math.floor(x[var]) if down else math.ceil(x[var]) - x[var]
            if distance > 0 and np.isfinite(child_raw):
                self.record_gain(var, down, (child_raw - raw) / distance)
            closed = child_bound >= cutoff
            if closed and rigorous:
                self.learn(child_terms, lo, up)
            children[var][down] = _StrongChild(
                child_bound, closed, None if closed else solution, child_raw
            )
        return {var: (sides[True], sides[False]) for var, sides in children.items()}

    def solve_children(self, boxes, warm):
        """The LPs of ``boxes`` for ``strong_epochs``, warm-started from
        ``warm``: in one vmapped call (padded to a power of two), or one by
        one while tree cuts change the LP's shape."""
        if not boxes:
            return []
        if self.use_tree_cuts:
            target = self.node_bound_target()
            results = []
            for lo, up in boxes:
                results.append(
                    self.solve_lp(
                        self.node_lp(lo, up),
                        warm,
                        self.strong_epochs,
                        **self.node_lp_kwargs,
                        **({} if target is None else {"bound_target": target}),
                    )
                )
            return results
        # One batch shape for the whole search, so one compile.
        size = max(_next_power_of_two(2 * self.strong_candidates), len(boxes))
        results = self.solve_batch(
            boxes, [warm[0]] * len(boxes), [warm[1]] * len(boxes),
            self.strong_epochs, size,
        )
        self.lp_epochs += sum(result["epochs"] for result in results)
        return results

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

    def branch(
        self, node, lower, upper, x, var, bound, value, warm, terms, strong=None,
        precomputed=None,
    ):
        """Split on ``var``. Each child is bounded by the parent's duals
        (``terms``) over its own propagated box, and closed at once when
        that reaches the cutoff. ``strong`` (``{down: _StrongChild}``, from
        strong branching) gives the children a bound and a warm start of
        their own. ``precomputed`` is this node's row of ``_ChildKernels``
        (the children already propagated and bounded on the device)."""
        xv = x[var]
        children = []
        for side, down in enumerate((True, False)):
            if precomputed is not None:
                if precomputed["crossed"][side]:
                    continue
                if precomputed["empty"][side]:
                    self.infeasible_nodes += 1
                    self.learn_nogood(*_child_box((lower, upper), var, xv, down))
                    continue
                lo = np.array(precomputed["lower"][side], np.float64)
                up = np.array(precomputed["upper"][side], np.float64)
            else:
                lo, up = lower.copy(), upper.copy()
                if down:
                    up[var] = min(up[var], math.floor(xv))
                else:
                    lo[var] = max(lo[var], math.ceil(xv))
                if lo[var] > up[var]:
                    continue
                if self.propagator is not None:
                    branched = (lo, up)
                    lo, up, infeasible = self.propagator(lo, up)
                    if infeasible:
                        self.infeasible_nodes += 1
                        self.learn_nogood(*branched)
                        continue
            lo, up, infeasible = self.conflict_propagate(lo, up)
            if infeasible:
                self.conflict_pruned += 1
                self.close(max(bound, self.cutoff_bound()))
                continue
            if precomputed is not None:
                # Over the box before the conflict pool, which holds it.
                child_bound = float(precomputed["bound"][side])
                rigorous = bool(precomputed["rigorous"][side])
            else:
                child_bound, rigorous = self.bound_over(terms, lo, up)
            child_bound = max(self.round_bound(child_bound), bound)
            child_warm = warm
            if strong is not None:
                # Its box is inside the strong child's, so that bound holds.
                child_bound = max(child_bound, strong[down].bound)
                solution = strong[down].solution
                if solution is not None:
                    child_warm = (solution() if callable(solution) else solution, warm[1])
            if child_bound >= self.cutoff():
                self.children_pruned += 1
                self.close(child_bound, rigorous)
                if rigorous:
                    self.learn(terms, lo, up)
                continue
            distance = xv - math.floor(xv) if down else math.ceil(xv) - xv
            children.append(
                _Node(
                    child_bound,
                    node.depth + 1,
                    self.compress(lo, up),
                    child_warm,
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
        spent = node.epochs + result["epochs"]
        if (
            not result["converged"]
            and bound < self.cutoff()
            and not self.out_of_time()
            and self.batch_size > 1
            and spent < self.node_max_epochs
        ):
            # Continuous batching: a batched node LP runs at most
            # ``batch_epochs`` a call, so one slow member cannot hold the
            # batch. An unfinished one goes back, warm-started where it
            # stopped, for its next epochs in a later batch (until
            # ``node_max_epochs`` in all). Its Lagrangian bound holds
            # whatever the duals, and an infeasible LP's bound passes the
            # cutoff along its Farkas ray (``node_bound_stop``).
            self.batch_requeues += 1
            self.push(
                _Node(
                    bound,
                    node.depth,
                    self.compress(lower, upper),
                    (result["solution"], result["opt_state"]),
                    node.branch,
                    spent,
                )
            )
            return
        if not result["converged"] and bound < self.cutoff() and not self.out_of_time():
            # An LP that will not converge is often infeasible.
            if lp is None:  # batched search builds the LP only when needed
                lp = self.node_lp(lower, upper)
            detection = self.detect(lp, result, self.node_max_epochs)
            if detection["status"] == "primal_infeasible":
                self.infeasible_nodes += 1
                self.learn_farkas(detection, lower, upper)
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
            if rigorous:
                self.learn(terms, lower, upper)
            return
        warm = (result["solution"], result["opt_state"])
        if self.node_crossover_due():
            # An exact vertex of the node LP: its point is integral far more
            # often than a PDHG point, and its duals may raise the bound.
            # Children still warm-start from the PDHG solution (``warm``).
            if lp is None:
                lp = self.node_lp(lower, upper)
            vertex = self.cross_over_node(lp, result["solution"], x)
            if vertex is not None:
                x, converged = np.asarray(vertex.primal, np.float64), True
                new_terms = self.dual_terms(vertex)
                new_raw, new_rigorous = self.bound_over(new_terms, lower, upper)
                if new_raw > raw:
                    terms, raw, rigorous = new_terms, new_raw, new_rigorous
                    bound = max(self.round_bound(raw), bound)
                    self.active_bound = bound
                if bound >= self.cutoff():
                    self.close(bound, rigorous)
                    if rigorous:
                        self.learn(terms, lower, upper)
                    return

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
        # Strong branching runs on the device while the host does the
        # heuristics and reduced-cost fixing below.
        pending = None
        if self.strong_backend == "jax" and self.strong_branching_on(node):
            inside = fractional & (x > lower) & (x < upper)
            pending = self.launch_strong(
                x, inside if inside.any() else fractional, lower, upper, warm, terms, raw
            )
        if self.heuristic_every and (self.nodes - 1) % self.heuristic_every == 0:
            if self.try_incumbent(x, "H", warm) or not self.use_fix_and_propagate:
                pass
            elif self.deferring and self.batch_roundings:
                # Batched on the device: rounded with the batch's others.
                self.deferred_roundings.append((x, lower.copy(), upper.copy(), warm))
            else:
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
            and self.submip_budget_left()
        ):
            self.rins_pending = False
            self.rins(x, warm)
            if bound >= self.cutoff():
                self.close(bound, rigorous)
                return
        if self.tree_heuristics_due():
            self.tree_heuristics(x, lower, upper, warm)
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
            candidates = branchable
        elif lower is not node_lower:  # the fixing tightened the box
            # Reduced-cost fixing moved every fractional variable's bounds
            # past the LP point: solve the node again in its smaller box.
            self.push(_Node(bound, node.depth, self.compress(lower, upper), warm, None))
            return
        else:
            candidates = fractional
        choice = self.choose_branch(
            node, x, candidates, lower, upper, warm, terms, bound, pending, raw
        )
        if choice is None:  # strong branching closed or requeued the node
            return
        var, strong, lower, upper, bound = choice
        self.active_bound = bound
        if (
            self.use_tree_cuts
            and self.tree_cut_every
            and self.nodes % self.tree_cut_every == 0
        ):
            warm = self.tree_cut_round(x, warm)
            # The strong children's duals miss the new cut rows.
            if strong is not None:
                strong = {d: child._replace(solution=None) for d, child in strong.items()}
        if self.deferring:
            # Batched on the device: split with the batch's other members.
            self.deferred_branches.append(
                dict(
                    node=node, lower=lower, upper=upper, x=x, var=var, bound=bound,
                    value=value, warm=warm, terms=terms, strong=strong,
                )
            )
            return
        self.branch(node, lower, upper, x, var, bound, value, warm, terms, strong)

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
                    f"{families.count('flowcover')} flow cover, "
                    f"{families.count('gomory')} Gomory)"
                )
                if self.use_tree_cuts:
                    line += f", {self.tree_cuts_added} added in the tree"
                print(line)
            if self.conflicts is not None and any(self.conflicts.added.values()):
                added = self.conflicts.added
                print(
                    f"Conflicts: {added['objective']} objective + "
                    f"{added['farkas']} Farkas proofs, {added['nogood']} no-goods; "
                    f"{self.conflict_pruned} nodes pruned before their LP, "
                    f"{self.conflict_tightenings} bounds tightened"
                )
            if self.strong_branch_nodes:
                print(
                    f"Strong branching: {self.strong_branch_nodes} nodes, "
                    f"{self.strong_branch_lps} LPs, {self.strong_fixings} fixings, "
                    f"{self.strong_branch_seconds:.2f} seconds"
                )
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
                for family in ("gomory", "cover", "mir", "flowcover")
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
            "conflicts": dict(self.conflicts.added)
            if self.conflicts is not None
            else {"objective": 0, "farkas": 0, "nogood": 0},
            "conflict_pruned": self.conflict_pruned,
            "conflict_tightenings": self.conflict_tightenings,
            "bound_stops": self.bound_stops,
            "submip_calls": self.submip_calls,
            "submip_seconds": self.submip_seconds,
            "strong_branch_nodes": self.strong_branch_nodes,
            "strong_branch_lps": self.strong_branch_lps,
            "strong_branch_seconds": self.strong_branch_seconds,
            "strong_fixings": self.strong_fixings,
            "pump_jumps": self.pump_jumps,
            "pump_jump_finds": self.pump_jump_finds,
            "probing": dict(self.probing_stats),
            "batch_requeues": self.batch_requeues,
            "batched_children": self.batched_children,
            "batched_roundings": self.batched_roundings,
            "tree_heuristic_calls": self.tree_heuristic_calls,
            "tree_heuristic_finds": self.tree_heuristic_finds,
            "tree_heuristic_seconds": self.tree_heuristic_seconds,
            "node_crossovers": self.node_crossovers,
            "node_crossover_vertices": self.node_crossover_vertices,
            "node_crossover_integral": self.node_crossover_integral,
            "node_crossover_seconds": self.node_crossover_seconds,
        }

    def node_crossover_due(self):
        """Whether this node's LP point is crossed over (``node_crossover``,
        every ``node_crossover_every`` nodes, time permitting)."""
        return (
            self.use_node_crossover
            and self.nodes % self.node_crossover_every == 0
            and not self.out_of_time()
        )

    def cross_over_node(self, lp, solution, x):
        """Cross the node LP's PDHG ``solution`` over to an optimal vertex
        with HiGHS (``jaddle_linear.crossover``), within
        ``node_crossover_seconds`` and the time left. Returns the vertex's
        ``SaddleState``, or ``None`` when HiGHS does not prove it optimal.
        ``x`` (the PDHG point) only feeds the statistics."""
        started = time.time()
        limit = self.node_crossover_limit
        if self.deadline is not None:
            limit = min(limit, self.seconds_left())
        crossed = jl.crossover(lp, solution, highs_options={"time_limit": limit})
        self.node_crossovers += 1
        self.node_crossover_seconds += time.time() - started
        if not crossed["converged"]:
            return None
        self.node_crossover_vertices += 1
        vertex = np.asarray(crossed["solution"].primal, np.float64)
        ints = self.is_int

        def integral(point):
            return np.all(np.abs(point[ints] - np.round(point[ints])) <= self.int_tol)

        self.node_crossover_integral += int(integral(vertex) and not integral(x))
        return crossed["solution"]

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
        """Rounds of root cuts: separate at the LP point, append the cuts,
        re-solve warm. With Gomory cuts, each round's crossover basis says
        which cuts are binding, and the slack ones (basic rows) are dropped
        before the new cuts go in. Dense GMI cuts are needed for a round or
        two, so this keeps the LP near its own size: on ns1830653 the cut
        rows held about 170 nonzeros after eight rounds instead of 220,000,
        for the same bound. Without a basis, a cut is dropped once it has
        been slack with a zero dual for two rounds.
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
            basis = None
            if self.gomory:
                # Cross over to a vertex: its basis gives the Gomory cuts, and
                # the other families separate better from a vertex too.
                vertex, basis = self.root_vertex(lower, upper, solution)
                if vertex is not None:
                    x = vertex
                    self.root_vertex_x = vertex
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
                        # Dense cuts are fine here: the ones left slack
                        # are dropped next round (see ``binding``).
                        max_nnz=None,
                    )
            for separator in separators:
                candidates += separator(x)
            found = select_cuts(candidates, x, n, max_cuts=self.max_cuts_per_round)
            if not found:
                break
            dual_ineq = np.asarray(solution.dual_ineq)
            if basis is not None:
                import highspy

                # Basic rows are slack at the optimal vertex: dropping them
                # keeps it optimal.
                first = self.mip.n_eq + base_m
                status = basis.row_status
                binding = [
                    status[first + i] != highspy.HighsBasisStatus.kBasic
                    for i in range(len(self.cut_list))
                ]
            else:
                binding = [cut["age"] < 2 for cut in self.cut_list]
            kept = [
                (cut, dual_ineq[base_m + i])
                for i, cut in enumerate(self.cut_list)
                if binding[i]
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
                    f"MIR + {families.count('flowcover')} flow cover in the LP, bound {self.model_value(new_bound):.6e}, "
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
            flow_covers=self.use_flow_covers,
            backend=self.separation_backend,
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
                f"({families.count('cover')} cover, {families.count('mir')} MIR, "
                f"{families.count('flowcover')} flow cover)"
                + (f", {len(replaced)} replacing idle cuts" if replaced else "")
                + f", pool {len(self.cut_list)}/{max_rows}"
            )
        if replaced:
            duals = np.asarray(solution.dual_ineq).copy()
            duals[[base_m + i for i in replaced]] = 0.0
            warm = (solution._replace(dual_ineq=jnp.asarray(duals)), warm[1])
        return warm

    def feasibility_pump(self, lower, upper, result, deadline):
        """
        The objective feasibility pump (Achterberg & Berthold) from the root
        LP point, fused with the feasibility jump as in cuOpt's extended pump
        (Çördük et al., 2025): alternately round, and solve the LP that
        minimises the L1 distance to the rounded point (over integers at a
        bound), blended with the objective at a weight that decays by 0.9 per
        iteration. Only the cost changes, so the pump reuses the root's
        compiled LP shape.

        With ``feasibility_jump`` on, every rounding that is not feasible
        gets a short feasibility jump (``pump_jump_iterations`` steps, while
        the jumps have taken at most ``pump_jump_share`` of the pump's time),
        which looks for a feasible point near it. When the pump cycles (the
        rounding repeats, or the projection distance has not improved by 10%
        on the mean of the last five), the jump's best near miss becomes the
        next rounding; without a jump, random flips break the cycle. Stops at
        the first incumbent, after ``pump_iterations`` or at ``deadline``.
        Returns whether it found one.
        """
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
        distances = []  # the L1 distance of each projection to its rounding
        near_miss = None  # the last jump's best point, to break a cycle with
        started, jump_seconds = time.time(), 0.0
        tolerances = dict(
            primal_feasibility_tolerance=self.pump_tolerance,
            dual_feasibility_tolerance=self.pump_tolerance,
            dual_gap_tolerance=self.pump_tolerance,
        )
        for iteration in range(self.pump_iterations):
            if time.time() >= deadline or self.out_of_time():
                break
            rounded = np.clip(np.round(x), lower, upper)
            stalled = len(distances) > 5 and distances[-1] > 0.9 * np.mean(distances[-6:-1])
            repeated = previous is not None and np.array_equal(rounded[ints], previous[ints])
            key = rounded[ints].tobytes()
            if (repeated or stalled or key in seen) and near_miss is not None:
                # Cycling: continue from the jump's near miss.
                rounded[ints] = np.clip(np.round(near_miss[ints]), lower[ints], upper[ints])
                near_miss = None
                distances.clear()
            elif repeated:
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
            if (
                self.use_jump
                and self.pump_jump_iterations > 0
                and jump_seconds <= self.pump_jump_share * (time.time() - started)
                and time.time() < deadline
            ):
                jumped = time.time()
                point, feasible, _ = feasibility_jump(
                    self.base,
                    candidate,
                    lower,
                    upper,
                    tol=self.feas_tol,
                    max_iterations=self.pump_jump_iterations,
                    deadline=None if not np.isfinite(deadline) else deadline,
                    seed=iteration,
                )
                jump_seconds += time.time() - jumped
                self.pump_jumps += 1
                if feasible and self.try_incumbent(point, "F", warm):
                    self.pump_jump_finds += 1
                    return True
                near_miss = point

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
            distances.append(float(np.abs(x - rounded)[ints].sum()))

            # Close to integral: let propagation finish the rounding.
            distance = np.abs(x - np.round(x))[ints].sum()
            if self.use_fix_and_propagate and distance < 0.9 * best_distance:
                best_distance = distance
                candidate = fix_and_propagate(
                    x, lower, upper, ints, self.row_propagator
                )
                if candidate is not None and self.try_incumbent(candidate, "F", warm):
                    return True
                if self.use_bulk_rounding:
                    for point in self.bulk_round(
                        [x] * self.bulk_starts, lower, upper, deadline, seed=iteration
                    ):
                        if point is not None and self.try_incumbent(point, "F", warm):
                            return True
        return False

    def rins(self, x, warm):
        """RINS around the incumbent and the node LP point ``x``, with the
        sub-MIP solved by HiGHS (``highs_submip``, the default) or Jaddle
        (``jaddle_rins``), per ``rins_solver``. The node limit
        (``rins_nodes``) doubles after a call that improves the incumbent
        and halves after one that does not, within [50, 2000], as HiGHS
        adapts its own sub-MIP effort."""
        started = time.time()
        seconds = self.rins_seconds
        if self.deadline is not None:
            seconds = min(seconds, self.deadline - started)
        before = self.incumbent[0]
        if self.rins_solver == "highs":
            found = self.highs_rins_search(x, seconds)
        else:
            found = self.jaddle_rins(x, seconds, warm)
        self.rins_calls += 1
        if found is not None:
            self.try_incumbent(found, "N", warm)
        improved = self.incumbent[0] < before
        self.rins_nodes = int(
            min(2000, max(50, self.rins_nodes * (2 if improved else 0.5)))
        )
        elapsed = time.time() - started
        self.heuristic_seconds += elapsed
        self.submip_seconds += elapsed
        self.submip_calls += 1

    def submip_budget_left(self):
        """Whether RINS is within ``heuristic_effort`` of the elapsed time
        (never with ``heuristic_effort=0``)."""
        return self.submip_seconds < self.heuristic_effort * (time.time() - self.start)

    def highs_rins_search(self, x, seconds, min_fixed=0.3):
        """RINS with HiGHS (``highs_submip``): the integers on which the
        incumbent and ``x`` agree are fixed, the incumbent is HiGHS's start
        and this search's cutoff its objective bound. Skipped (``None``)
        when fewer than ``min_fixed`` of the integers agree or the same
        neighbourhood was searched at the same cutoff."""
        ints = self.is_int
        incumbent = self.incumbent[1]
        agree = ints & (np.abs(np.asarray(x, np.float64) - incumbent) <= self.int_tol)
        if seconds <= 0 or agree.sum() < min_fixed * ints.sum():
            return None
        key = (agree.tobytes(), incumbent[agree].tobytes(), self.cutoff())
        if key in self.rins_seen:
            self.rins_skipped += 1
            return None
        self.rins_seen.add(key)
        lower = np.where(agree, incumbent, self.base.lower)
        upper = np.where(agree, incumbent, self.base.upper)
        return highs_submip(
            self.base, lower, upper, cutoff=self.cutoff(), max_seconds=seconds,
            max_nodes=self.rins_nodes, start=incumbent,
        )

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
            tree_heuristics=False,
            node_crossover=False,
            batch_size=1,
            verbose=False,
        )
        if settings["branching"] == "reliability":
            # Strong branching would spend the sub-MIP's short budget.
            settings["branching"] = "pseudocost"
        settings["probing"] = False
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
        searched for ``rens_seconds`` and ``rens_nodes`` nodes, by HiGHS
        (``highs_submip``, which presolves it: the HiGHS run of the same
        neighbourhood is what finds HiGHS's early incumbents) or, with
        ``rins_solver="jaddle"``, by ``sub_search``. Needs no incumbent; a
        cutoff, when there is one, keeps only better points. Skipped when
        fewer than ``min_fixed`` of the integers are integral. A point found
        is marked ``"E"``.
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
        if self.rins_solver == "highs":
            cutoff = self.cutoff()
            found = highs_submip(
                self.base, sub_lower, sub_upper,
                cutoff=cutoff if np.isfinite(cutoff) else None,
                max_seconds=seconds, max_nodes=self.rens_nodes,
            )
        else:
            found = self.sub_search(sub_lower, sub_upper, seconds, self.rens_nodes, warm)
        self.submip_calls += 1
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

    def tree_heuristics_due(self):
        """Whether the GPU tree heuristics run at this node: on, with an
        incumbent to improve, every ``tree_heuristic_every`` nodes, and
        within ``tree_heuristic_share`` of the elapsed time."""
        return (
            self.use_tree_heuristics
            and self.incumbent is not None
            and self.nodes % self.tree_heuristic_every == 0
            and self.tree_heuristic_seconds
            < self.tree_heuristic_share * (time.time() - self.start)
            and not self.out_of_time()
        )

    def tree_heuristics(self, x, lower, upper, warm):
        """The GPU heuristics from the node LP point ``x``, for a point
        better than the incumbent: bulk rounding in the node box (with the
        objective row at the cutoff, as ``bulk_round``), then, if that finds
        nothing better, a feasibility jump in the root box with the row
        ``c·x <= cutoff``. Skipped when the same rounding of ``x`` was tried
        at the same cutoff. Finds are marked ``"B"`` and ``"J"``."""
        cutoff = self.cutoff(box=False)
        key = (np.round(x[self.is_int]).tobytes(), cutoff)
        if key in self.tree_heuristic_seen:
            return
        self.tree_heuristic_seen.add(key)
        started = time.time()
        before = self.incumbent[0]
        # Each call stops once it uses up the share left (at least
        # _TREE_HEURISTIC_MIN_SECONDS); otherwise one bulk rounding of a large
        # model can run on to the solve's deadline.
        allowance = (
            self.tree_heuristic_share * (started - self.start) - self.tree_heuristic_seconds
        )
        deadline = started + max(allowance, _TREE_HEURISTIC_MIN_SECONDS)
        if self.deadline is not None:
            deadline = min(deadline, self.deadline)
        try:
            if self.use_bulk_rounding:
                points = self.bulk_round([x] * self.bulk_starts, lower, upper, deadline)
                for point in points:
                    if point is not None:
                        self.try_incumbent(point, "B", warm)
            if self.use_jump and self.incumbent[0] >= before and not self.out_of_time():
                lo, up = self.root_lower, self.root_upper
                if self._tree_jump_data is None:
                    self._tree_jump_data = _jump_data(
                        self.base, lo, up, self.feas_tol, cutoff
                    )
                data = _jump_with(self._tree_jump_data, lo, up, cutoff)
                point, feasible, _ = feasibility_jump(
                    self.base,
                    x,
                    lo,
                    up,
                    tol=self.feas_tol,
                    max_iterations=self.tree_jump_iterations,
                    deadline=deadline,
                    seed=self.tree_heuristic_calls,
                    cutoff=cutoff,
                    data=data,
                )
                if feasible:
                    self.try_incumbent(point, "J", warm)
        finally:
            self.tree_heuristic_calls += 1
            self.tree_heuristic_finds += int(self.incumbent[0] < before)
            elapsed = time.time() - started
            self.tree_heuristic_seconds += elapsed
            self.heuristic_seconds += elapsed

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
            # solve that small MIP, within ``repair_share`` of the heuristic
            # time left, so later jump starts and the pump still get theirs
            # (on ns1830653 an unlimited repair took the whole budget, and
            # the next start would have found a point in 10 s).
            left = deadline - time.time() if np.isfinite(deadline) else 60.0
            repaired = repair(
                self.base,
                x,
                lower,
                upper,
                max_seconds=max(self.repair_share * left, 0.0),
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
            if self.use_bulk_rounding and self.incumbent is None:
                # Several bulk roundings of the LP point at once: they
                # succeed on other models than the classic one.
                for point in self.bulk_round([x] * self.bulk_starts, lower, upper, deadline):
                    if point is not None:
                        self.try_incumbent(point, "B", warm)
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

    def batched_solver(self, max_epochs=None):
        """``jax.vmap`` of ``jaddle_linear.make_solver`` over node bounds and
        warm starts, jitted once for the current LP (the shape is final once
        the root cuts are in) and each epoch limit (``None``:
        ``node_max_epochs``). With ``node_bound_stop`` it takes a third
        argument, the bound target (one for the whole batch): a member stops
        once its Lagrangian bound reaches it, as serial node LPs do."""
        if max_epochs is None:
            max_epochs = self.node_max_epochs
        cached = self._vmapped.get(max_epochs)
        if cached is None or cached[0] is not self.mip:
            import inspect

            accepted = set(inspect.signature(jl.make_solver).parameters) - {"build_lp"}
            unknown = sorted(set(self.node_lp_options) - accepted)
            if unknown:
                raise ValueError(
                    f"node_lp_options {', '.join(unknown)} cannot be used by batched "
                    "node LPs (jaddle_linear.make_solver)"
                )
            # Node LPs: ``lp_options`` (what make_solver takes), then the
            # node layer (``node_lp_tolerance``, ``node_lp_options``).
            settings = {
                key: value
                for key, value in {**self.lp_kwargs, **self.node_lp_kwargs}.items()
                if key in accepted
            }
            solve_fn = jl.make_solver(
                self.mip.lp.with_values,
                max_epochs=max_epochs,
                bound_stop=self.node_bound_stop,
                **settings,
            )
            axes = jl.LPValues(*([None] * len(jl.LPValues._fields)))._replace(
                lower_bounds=0, upper_bounds=0
            )
            in_axes = (axes, 0, None) if self.node_bound_stop else (axes, 0)
            cached = (self.mip, jax.jit(jax.vmap(solve_fn, in_axes=in_axes)))
            self._vmapped[max_epochs] = cached
        return cached[1]

    def bound_target_arg(self):
        """The batched solver's bound-target argument: ``()`` without
        ``node_bound_stop``, else the node target (``inf`` before there is
        a cutoff, so nothing stops)."""
        if not self.node_bound_stop:
            return ()
        target = self.node_bound_target()
        return (np.inf if target is None else float(target),)

    def solve_batch(self, boxes, solutions, opt_states, max_epochs, size):
        """The node LPs of ``boxes`` in one vmapped call, each warm-started
        from its solution, padded to ``size`` members with copies of the
        last. With ``node_bound_stop``, a member also stops once its
        Lagrangian bound passes ``node_bound_target()`` (``stop_reason``
        ``"bound_target"``). Returns a ``solve_lp``-style result per box
        (its ``opt_state`` is the one passed in: the batched solver keeps
        none)."""
        pad = size - len(boxes)
        boxes = list(boxes) + [boxes[-1]] * pad
        solutions = list(solutions) + [solutions[-1]] * pad
        lp_lower = np.stack([np.where(self.is_int, lo, self.mip.lower) for lo, _ in boxes])
        lp_upper = np.stack([np.where(self.is_int, up, self.mip.upper) for _, up in boxes])
        values = self.values._replace(
            lower_bounds=jnp.asarray(lp_lower, self.dtype),
            upper_bounds=jnp.asarray(lp_upper, self.dtype),
        )
        warm_starts = jax.tree.map(lambda *leaves: jnp.stack(leaves), *solutions)
        # One transfer for the whole batch, then slices on the host.
        out = jax.device_get(
            self.batched_solver(max_epochs)(values, warm_starts, *self.bound_target_arg())
        )
        converged = np.asarray(out.converged)
        stop_code = np.asarray(out.stop_code)
        epochs = np.asarray(out.epochs)
        return [
            {
                "solution": jax.tree.map(lambda leaf: leaf[i], out.solution),
                "converged": bool(converged[i]),
                "stop_reason": {1: "certificate", 2: "primal_stall", 4: "bound_target"}.get(
                    int(stop_code[i]), "max_epochs"
                ),
                "epochs": int(epochs[i]),
                "opt_state": opt_states[i],
            }
            for i in range(len(opt_states))
        ]

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
            box = self.prune_by_conflicts(node, *box)
            if box is None:
                continue
            members.append(node)
            boxes.append(box)
        if not members:
            return
        results = self.solve_batch(
            boxes,
            [node.warm[0] for node in members],
            [node.warm[1] for node in members],
            min(self.batch_epochs, self.node_max_epochs),
            self.batch_size,
        )
        self.batches += 1
        self.deferring = self.batch_on_device()
        try:
            for i, (node, result) in enumerate(zip(members, results)):
                self.nodes += 1
                self.lp_epochs += result["epochs"]
                self.bound_stops += result["stop_reason"] == "bound_target"
                # An incumbent found by an earlier member may prune this one.
                if node.bound >= self.cutoff():
                    self.close(node.bound)
                    continue
                lower, upper = boxes[i]
                self.evaluate(node, lower, upper, None, result)
                self.log(" ", None, node.depth)
        finally:
            self.deferring = False
        self.flush_roundings()
        self.flush_branches()

    def batch_on_device(self):
        """Whether batched search splits and rounds its members on the
        device (``children_backend``)."""
        if self.children_backend == "numpy" or self.batch_size < 2:
            return False
        if self.children_backend == "jax":
            return True
        width = max(self.mip.num_variables, self.mip.A.nnz)
        return self.mip.A.nnz > _SMALL_MIP_NNZ and 2 * self.batch_size * width <= 2**25

    def child_kernels(self):
        if self._child_kernels is None or self._child_kernels[0] is not self.mip:
            kernels = _ChildKernels(
                self.mip, self.propagator is not None, self.reduced_cost_tol, self.batch_size
            )
            self._child_kernels = (self.mip, kernels)
        return self._child_kernels[1]

    def flush_roundings(self):
        """The batch's deferred tree roundings, as one ``bulk_round`` with a
        box per node (padded to ``batch_size`` starts: one compiled shape)."""
        pending, self.deferred_roundings = self.deferred_roundings, []
        if not pending:
            return
        started = time.time()
        points = self.bulk_round(
            [x for x, _, _, _ in pending],
            None,
            None,
            seed=self.batches,
            boxes=[(lo, up) for _, lo, up, _ in pending],
            pad_to=self.batch_size,
        )
        self.batched_roundings += 1
        for point, (_, _, _, warm) in zip(points, pending):
            if point is not None:
                self.try_incumbent(point, "P", warm)
        self.heuristic_seconds += time.time() - started

    def flush_branches(self):
        """Split the batch's deferred nodes: their children propagated and
        bounded in one ``_ChildKernels`` call, then ``branch`` on the host
        for the conflict pool, pruning and the queue."""
        pending, self.deferred_branches = self.deferred_branches, []
        if not pending:
            return
        kernels = self.child_kernels()
        size, n = kernels.size, self.mip.num_variables
        lower = np.zeros((size, n))
        upper = np.zeros((size, n))
        var = np.zeros(size, np.int32)
        xv = np.zeros(size)
        r = np.zeros((size, n))
        constant = np.zeros(size)
        for i, item in enumerate(pending):
            lower[i], upper[i] = item["lower"], item["upper"]
            var[i] = item["var"]
            xv[i] = item["x"][item["var"]]
            constant[i], r[i] = item["terms"]
        for i in range(len(pending), size):  # padding: copies of the first
            lower[i], upper[i], var[i], xv[i], r[i], constant[i] = (
                lower[0], upper[0], var[0], xv[0], r[0], constant[0]
            )
        out = kernels(lower, upper, var, xv, r, constant)
        self.batched_children += 1
        for i, item in enumerate(pending):
            row = {key: value[i] for key, value in out.items()}
            self.branch(**item, precomputed=row)

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
        if self.use_probing and self.propagator is not None and self.probing_budget > 0:
            started = time.time()
            probed = probe(
                mip.A,
                mip.row_lower,
                mip.row_upper,
                self.is_int,
                lower,
                upper,
                max_variables=self.probing_max_variables,
                deadline=started + self.probing_budget,
                backend=self.probing_backend,
                max_useless=self.probing_max_useless,
            )
            self.probing_stats = {
                "probed": probed.probed,
                "fixed": probed.fixed,
                "tightened": probed.tightened,
                "seconds": time.time() - started,
            }
            if probed.infeasible:
                return self.result("infeasible")
            lower, upper = probed.lower, probed.upper
            self.probe_cache = probed.cache
        self.root_lower, self.root_upper = lower, upper
        with np.errstate(invalid="ignore"):
            top = float(np.sum(np.where(mip.c > 0, mip.c * upper, np.where(mip.c < 0, mip.c * lower, 0.0))))
        if np.isfinite(top):
            # Every feasible point has c·x <= top, so a node whose bound
            # passes this holds none: an infeasible node LP, whose bound
            # diverges along its Farkas ray, stops on it before there is an
            # incumbent (one ran all 200 epochs on ns1830653, and held its
            # batch for 1.8 s).
            self.box_cutoff = top + 1e-6 * (1.0 + abs(top)) + 1e-9
        if self.verbose:
            if self.probing_stats["probed"]:
                stats = self.probing_stats
                print(
                    f"Probing: {stats['probed']} variables, {stats['fixed']} fixed, "
                    f"{stats['tightened']} bounds tightened, {stats['seconds']:.2f}s"
                )
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
            # From an exact vertex, as HiGHS's RENS works from its simplex
            # root: far more of it is integral than of a PDHG point. The cut
            # loop keeps its last one; otherwise cross over now (HiGHS).
            vertex = self.root_vertex_x
            if vertex is None and self.rins_solver == "highs":
                vertex, _ = self.root_vertex(lower, upper, result["solution"])
            if vertex is not None:
                self.rens(vertex, lower, upper, root_warm, min_fixed=0.3)
            else:
                self.rens(root_x, lower, upper, root_warm)
        if self.dive_every and not self.out_of_time():
            self.dive(root_x, lower, upper, root_warm, rule="fractional")
            if self.incumbent is not None:
                self.dive(root_x, lower, upper, root_warm, rule="guided")
        if (
            self.rins_seconds
            and self.incumbent is not None
            and not self.out_of_time()
            and self.submip_budget_left()
        ):
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
                box = self.prune_by_conflicts(node, lower, upper)
                if box is None:
                    continue
                lower, upper = box
                lp = self.node_lp(lower, upper)
                target = self.node_bound_target()
                result = self.solve_lp(
                    lp, node.warm, self.node_max_epochs, **self.node_lp_kwargs,
                    **({} if target is None else {"bound_target": target}),
                )
                self.bound_stops += result["stop_reason"] == "bound_target"
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
    flow_covers=True,
    separation_backend="jax",
    node_bound_stop=True,
    node_lp_tolerance=None,
    heuristic_effort=0.1,
    reliability_threshold=4,
    strong_candidates=8,
    strong_epochs=50,
    strong_max_depth=None,
    strong_screen=32,
    strong_backend="jax",
    tree_cuts=False,
    tree_cut_every=10,
    fix_and_propagate=True,
    feasibility_jump=True,
    jump_iterations=None,
    repair=True,
    repair_max_epochs=20,
    highs_incumbent_seconds=0.0,
    rins_seconds=None,
    rins_every=50,
    rins_solver="highs",
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
    pump_jump_iterations=3000,
    pump_jump_share=0.2,
    probing=True,
    probing_seconds=None,
    probing_max_variables=None,
    probing_backend="auto",
    probing_max_useless=1000,
    bulk_rounding=True,
    bulk_starts=4,
    repair_share=0.2,
    batch_epochs=10,
    children_backend="auto",
    batch_roundings=False,
    tree_heuristics=False,
    tree_heuristic_every=20,
    tree_heuristic_share=0.05,
    tree_jump_iterations=3000,
    node_crossover=False,
    node_crossover_every=1,
    node_crossover_seconds=2.0,
    heuristic_seconds=None,
    conflicts=True,
    conflict_pool_size=2000,
    conflict_max_nnz=None,
    presolve=False,
    highs_options=None,
    lp_options=None,
    node_lp_options=None,
    verbose=False,
    log_every_seconds=1.0,
    device="auto",
    precision=None,
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
      with Gomory cuts on, else 5) of
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
      A continuous variable with a variable bound in the MIP's rows
      (``variable_bounds``: a two-term row such as ``y <= u·z``) may be
      shifted to that bound instead of its simple ones, as HiGHS does, and
      the better cut is kept; on vpm1 that takes the root bound from 18.9 to
      20, the optimum. The row and each aggregate with a continuous
      variable also give a flow cover cut (``flow_cover_cut``, family
      ``"flowcover"``; ``flow_covers``, on by default), from its single-node
      flow set over those variable bounds. MIR cuts come from single rows
      and from aggregates of up to three rows that cancel continuous
      variables; with ``aggregate_integers``
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
      1e-4 that are not near-parallel, retires cuts that are slack, and
      re-solves the LP warm. With Gomory cuts on, a cut is slack when its
      row is basic in the round's crossover basis, and it goes at once:
      GMI cuts are dense (hundreds of nonzeros on ns1830653), so they are
      allowed at the root (``gomory_cuts(max_nnz=None)``) but leave as soon
      as they stop binding. Without them, a cut goes after two rounds slack
      with a zero dual. The rounds
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
      Separation is batched (``separation_backend="jax"``, the default):
      once a round has at least 32 base inequalities (rows and their
      aggregates), the c-MIR and flow cover searches run on all of them in
      jitted float64 JAX calls (``batched_cmir_cuts``,
      ``batched_flow_cover_cuts``), with the same cuts as the per-row numpy
      searches (``"numpy"``) up to rounding in near-ties. Per round that is
      2.4× faster on fixnet6 and 4× on vpm2; the first calls compile a few
      shapes (one or two seconds, then cached on disk).
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
      Since that bound holds at any duals, a node LP stops as soon as it
      closes the node (``node_bound_stop``, on by default:
      ``jl.solve(bound_target=...)`` checks it every epoch). An infeasible
      node's duals diverge along a Farkas ray and pass the cutoff within a
      few epochs, instead of running ``node_max_epochs`` and
      ``detect_infeasibility``. On ns1830653 (3 minutes) that took the
      search from 140 nodes to 385 (21.9 to 10.2 epochs a node) and the
      bound from 11,678 to 11,976 (``bound_stops`` in the result).
    * **Conflicts** (``conflicts``, on by default; ``ConflictPool``). Each
      node or child closed by its Lagrangian bound leaves a dual proof,
      ``constant + r·x <= cutoff``, which every better solution satisfies
      whatever the accuracy of the duals; an LP that ``detect`` proves
      infeasible leaves one from its dual ray, and a child that propagation
      proves infeasible leaves a no-good over its binary fixings (cut down
      by a deletion filter). Proofs are reduced to the variables whose node
      bounds differ from the root's (the others go to the right-hand side),
      kept only when that still excludes the node and has at most
      ``conflict_max_nnz`` variables (default ``max(25, n // 10)``), and the
      newest ``conflict_pool_size`` (2000) are propagated at every node
      before its LP, alternating with the rows. A node they show empty is
      closed without its LP (``conflict_pruned`` in the result).
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
      ``"cpu"``, ``"gpu"``, ``"tpu"`` or a ``jax.Device`` pick one;
      ``None`` keeps JAX's default device.
    * **Precision** (``precision``). ``"float64"``, ``"float32"`` or
      ``"float16"`` runs the whole search, node LPs included, at that
      profile (see ``jaddle_optimisers.configure_jax``); ``None`` (default)
      keeps the global one. A ``MIP`` built at another precision is
      converted. Propagation, cuts and feasibility checks stay in float64
      on the host either way.
    * **Branching** (``branching``): ``"pseudocost"`` (default) scores each
      fractional variable by the product of its estimated down and up bound
      gains, learnt from the nodes solved so far (most fractional until
      there are any); ``"most_fractional"`` picks the variable nearest
      ½. ``"reliability"`` is pseudocost branching that first
      strong-branches the variables whose pseudocosts are unreliable
      (fewer than ``reliability_threshold`` observations in a direction).
      With ``strong_backend="jax"`` (the default) it runs on the device,
      dispatched before the node's heuristics so the host works meanwhile:
      both children of up to ``strong_screen`` candidates are propagated
      (``_propagate_jax``, the same rules as ``BoundPropagator``) and
      bounded by the node's duals; the ``strong_candidates`` unreliable
      ones with the best bound gains (each at least its pseudocost
      estimate) get both child LPs solved for ``strong_epochs`` epochs in
      one vmapped call warm-started from the node; and the children's
      Lagrangian bounds (valid however far the LPs got) are computed on
      the device. All shapes are fixed for the search, so it compiles once.
      ``"numpy"`` does the same on the host for the ``strong_candidates``
      best by pseudocost score, with the conflict pool too; tree cuts
      change the LP's shape and use it. The children's bounds feed the
      pseudocosts (gains between unrounded bounds), fix a candidate with a
      closed child to its other side, raise the node's bound, and become
      the children's bounds and warm starts. ``strong_max_depth`` (``None``:
      no limit) stops it below a depth; sub-MIP searches use
      ``"pseudocost"``. Each child's bounds are propagated, so many
      infeasible children are dropped without an LP solve.
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
    * **Probing** (``probing``, on by default). Before the root LP, every
      integer variable is split in two (a binary into 0 and 1, a general
      integer at its midpoint) and both sides are propagated (``probe``):
      an empty side fixes the variable to the other one (with all it
      implies), and otherwise every bound tightens to the hull of the two
      sides. Conflict-prone variables go first, up to
      ``probing_max_variables`` within ``probing_seconds`` (``None``: 5% of
      ``max_seconds`` up to 30 s, 10 s without a limit), and it stops after
      ``probing_max_useless`` variables in a row that change nothing. Models with more
      than 10,000 nonzeros probe chunks of variables at once in one vmapped
      JAX call (``probing_backend``). What each side implies is kept as a
      ``ProbingCache`` for bulk rounding.
    * **Primal heuristics.** After the root cuts, ``fix_and_propagate``
      (the function of that name) rounds the root LP point in chunks with
      bound propagation after each; if that fails, ``bulk_fix_and_propagate``
      (``bulk_rounding``) rounds it from ``bulk_starts`` starts side by side
      (cuOpt's bulk rounding: bulks of √(free) variables, two random
      roundings per bulk propagated together, the probing cache screening
      them first), marked ``"B"``. While there is still no incumbent,
      ``feasibility_jump`` (the function; an LP-free weighted local search
      on the rows, on the device) runs for up to ``jump_iterations`` steps
      from the rounded LP point and from the bounds nearest zero, then the
      objective feasibility pump (``feasibility_pump``) runs for up to
      ``pump_iterations`` warm-started LP solves at ``pump_tolerance``,
      followed by one more jump from the pump's last rounded point. The
      pump is fused with the jump (cuOpt's extended pump): each rounding
      that is not feasible gets a jump of ``pump_jump_iterations`` steps,
      within ``pump_jump_share`` of the pump's time, and when the pump
      cycles the jump's near miss becomes its next rounding. They
      share ``heuristic_seconds`` (default: 20% of ``max_seconds``, or no
      time limit without one). In the tree, fix-and-propagate also runs at
      every ``heuristic_every``-th node where plain rounding fails. Their
      incumbents are marked ``"P"``, ``"J"`` and ``"F"`` in the log and
      history. A jump that ends near-feasible is handed to ``repair`` (a
      backtracking search with propagation over the variables around its
      violated rows), marked ``"R"``, for at most ``repair_share`` of the
      heuristic time left. ``highs_incumbent_seconds > 0`` asks
      HiGHS (``highs_incumbent``) for its best incumbent within that many
      seconds when all of these fail, marked ``"X"``; it is off by default, so the search stays
      Jaddle's own. RINS (``rins_seconds``; ``None``, the default, gives
      5s a call) runs once at the root and every ``rins_every`` nodes once
      there is an incumbent, and after each new incumbent from another
      source: the integers on which the incumbent and the node's LP point
      agree are fixed, and the rest is searched for a better solution,
      marked ``"N"``. With ``rins_solver="highs"`` (default) HiGHS searches
      it (``highs_submip``: it presolves the neighbourhood, takes the
      incumbent as its start and the cutoff as its objective bound, up to
      ``rins_nodes`` nodes); with ``"jaddle"``, this branch and bound
      (``_BranchAndBound.jaddle_rins``, reusing the compiled LP solver);
      ``"auto"`` picks Jaddle for MIPs with at most 10,000 nonzeros.
      ``rins_nodes`` doubles after a call that improves the incumbent and
      halves after one that does not (within 50–2000), and RINS calls stop
      while they have taken more than ``heuristic_effort`` (0.1) of the
      elapsed time (``heuristic_effort=0``: no RINS).
    * **RENS** (``rens_seconds``; ``None``, the default, gives 10% of
      ``max_seconds`` up to 30s, or 10s without a time limit; with
      ``rins_solver="jaddle"``, 5s for MIPs with at most 10,000 nonzeros,
      else off). After the root heuristics, the integers that are integral
      at an exact root vertex (the cut loop's last crossover, else one
      crossed over for RENS) are fixed and the fractional ones restricted to
      their floor and ceiling, and that sub-MIP is searched as RINS's is (up
      to ``rens_nodes`` nodes), marked ``"E"``. It needs no incumbent. This
      is HiGHS's most useful heuristic: switching HiGHS's RENS off slows
      misc07 from 20s to 45s and leaves ns1830653 at 30,808 instead of
      25,808 after 60s. With HiGHS solving it, Jaddle reaches vpm2's
      optimum in 12s (never within 60s before) and pp08a's within 1% in
      4.6s (30s before).
    * **Dives** (``dive_every``, off by default). LP-based dives at the
      root and every ``dive_every`` nodes: round a fractional integer
      (fractional, guided or pseudocost rule), propagate, re-solve the LP
      warm, up to ``dive_depth`` times; an integral end point is marked
      ``"D"``.
    * **GPU tree heuristics** (``tree_heuristics``, off by default). Once
      there is an incumbent, every ``tree_heuristic_every`` (20) nodes, the
      node's LP point is bulk rounded (``bulk_starts`` starts, in the node
      box, with the objective row at the cutoff), marked ``"B"``; if that
      finds nothing better, ``feasibility_jump`` runs from it for up to
      ``tree_jump_iterations`` (3000) steps in the root box with the row
      ``c·x <= cutoff`` added, so it looks only for better points, marked
      ``"J"``. They run while they have taken at most
      ``tree_heuristic_share`` (0.05) of the elapsed time, each call stops
      once it has used the share left (but gets at least 1 s), and they skip a
      rounding of the LP point already tried at the same cutoff. A find
      also triggers RINS at the next node.
    * **Node crossover** (``node_crossover``, off by default). Every
      ``node_crossover_every`` (1) nodes that the LP bound does not close,
      HiGHS crosses the node LP's PDHG point over to an optimal vertex
      (``jaddle_linear.crossover``, at most ``node_crossover_seconds`` (2)
      a call, on the CPU). The vertex replaces the PDHG point for the
      integrality test (an integral vertex becomes an incumbent, ``"T"``),
      branching and the heuristics, and its duals' Lagrangian bound
      replaces the node's when higher. A first-order point lies inside the
      optimal face, so it is integral far less often than a vertex.
      Children still warm-start from the PDHG solution. When HiGHS does not
      prove the vertex optimal in time, the PDHG point is used.
    * **Objective-aware propagation.** Once there is a cutoff, the
      propagation inside fix-and-propagate and dives includes the row
      ``c·x <= cutoff``, so they reject roundings that cannot improve on the
      incumbent.

    ``lp_tolerance`` is the relative tolerance of every node LP (all three of
    ``solve()``'s tolerances); ``lp_options`` (a dict) passes any other
    ``solve()`` option to every LP the search solves (root, cut rounds,
    heuristics, nodes). ``node_lp_options`` (a dict) applies to node LPs
    only, over ``lp_options`` and ``node_lp_tolerance``: serial node solves,
    batched ones (``jaddle_linear.make_solver`` options only; others raise)
    and strong branching's child LPs. Run in float64 (``jaddle_optimisers.configure_jax``)
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
    * ``"tree_heuristic_calls"``, ``"tree_heuristic_finds"``,
      ``"tree_heuristic_seconds"``: GPU tree heuristic calls
      (``tree_heuristics``), those that improved the incumbent, and their time.
    * ``"node_crossovers"``, ``"node_crossover_vertices"``,
      ``"node_crossover_integral"``, ``"node_crossover_seconds"``: node
      crossovers (``node_crossover``), those that proved a vertex optimal,
      the vertices that were integral where the PDHG point was not, and
      their time.
    * ``"coefficients_tightened"``: coefficients cut by coefficient
      tightening; ``"tree_cuts_added"``: cuts added in the tree.
    * ``"strong_branch_nodes"``, ``"strong_branch_lps"``,
      ``"strong_branch_seconds"``, ``"strong_fixings"``: nodes that
      strong-branched, the child LPs they solved, the time taken and the
      variables fixed because one child was closed.
    * ``"cuts"``: the number of cuts in the final LP, by family
      (``"gomory"``, ``"cover"``, ``"mir"``, ``"flowcover"``); ``"root_bound_before_cuts"``: the root bound
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
        pump_jump_iterations=pump_jump_iterations,
        pump_jump_share=pump_jump_share,
        probing=probing,
        probing_seconds=probing_seconds,
        probing_max_variables=probing_max_variables,
        probing_backend=probing_backend,
        probing_max_useless=probing_max_useless,
        bulk_rounding=bulk_rounding,
        bulk_starts=bulk_starts,
        repair_share=repair_share,
        batch_epochs=batch_epochs,
        children_backend=children_backend,
        batch_roundings=batch_roundings,
        tree_heuristics=tree_heuristics,
        tree_heuristic_every=tree_heuristic_every,
        tree_heuristic_share=tree_heuristic_share,
        tree_jump_iterations=tree_jump_iterations,
        node_crossover=node_crossover,
        node_crossover_every=node_crossover_every,
        node_crossover_seconds=node_crossover_seconds,
        heuristic_seconds=heuristic_seconds,
        lp_options=lp_options,
        node_lp_options=node_lp_options,
        verbose=verbose,
        log_every_seconds=log_every_seconds,
        start=start,
        device=device,
        conflicts=conflicts,
        conflict_pool_size=conflict_pool_size,
        conflict_max_nnz=conflict_max_nnz,
        flow_covers=flow_covers,
        separation_backend=separation_backend,
        node_bound_stop=node_bound_stop,
        node_lp_tolerance=node_lp_tolerance,
        heuristic_effort=heuristic_effort,
        reliability_threshold=reliability_threshold,
        strong_candidates=strong_candidates,
        strong_epochs=strong_epochs,
        strong_max_depth=strong_max_depth,
        strong_screen=strong_screen,
        strong_backend=strong_backend,
    )
    with jo.local_settings(precision=precision):
        if presolve:
            return _solve_with_presolve(
                model, highs_options, feasibility_tolerance, settings
            )
        if isinstance(model, MIP):
            mip = model
            if precision is not None:
                # The width from_highs would build the LP at under this profile.
                dtype = _wide(jo.jaddle_dtype())
                if mip.lp.c.dtype != dtype:
                    mip = MIP(
                        jl._cast_lp(mip.lp, dtype),
                        mip.integrality,
                        mip.offset,
                        mip.maximise,
                    )
        else:
            mip = from_highs(model, highs_options=highs_options)
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
            "cuts": {"gomory": 0, "cover": 0, "mir": 0, "flowcover": 0},
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
