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
import signal
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from jaddle import highs_helpers as hh
from jaddle import jaddle_linear as jl
from jaddle import jaddle_optimisers as jo
from jaddle import presolve as jaddle_presolve
from jaddle.jaddle_basic_types import LP, JaddleLP, SaddleState, _wide

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

    ``implied`` marks continuous columns that some optimal solution has
    integral (HiGHS's presolve calls them implicit integers). They stay
    continuous: a point may have them fractional, and branch and bound
    splits them only with ``solve(branch_implied=True)``. They count as integral only for the objective's step
    (``objective_step``), as HiGHS's "objective function is integral" does.
    Default: none.
    """

    def __init__(self, lp, integrality, offset=0.0, maximise=False, implied=None):
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
        if implied is None:
            implied = np.zeros(n, dtype=bool)
        implied = np.asarray(implied, dtype=bool).reshape(-1)
        if implied.shape[0] != n:
            raise ValueError(
                f"implied has {implied.shape[0]} entries for {n} variables"
            )
        self.implied = implied & ~integrality
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
        """The largest violation at ``x``, as a dict: ``"rows"`` (each row's
        relative to its own ``1 + |b_i|``), ``"bounds"`` (absolute) and
        ``"integrality"`` (distance of an integer variable from the nearest
        integer). Scaling by ``1 + ‖b‖∞`` instead let one large right-hand
        side excuse large violations of every other row."""
        x = np.asarray(x, dtype=np.float64)
        Ax = self.A @ x
        excess = np.maximum(np.maximum(self.row_lower - Ax, Ax - self.row_upper), 0.0)
        rows = (excess / (1.0 + np.abs(self.b))).max(initial=0.0)
        bounds = np.maximum(np.maximum(self.lower - x, x - self.upper), 0.0).max(
            initial=0.0
        )
        xi = x[self.integrality]
        return {
            "rows": float(rows),
            "bounds": float(bounds),
            "integrality": float(np.abs(xi - np.round(xi)).max(initial=0.0)),
        }

    def with_rows(self, A_rows, b_rows):
        """A ``MIP`` with the rows ``A_rows x <= b_rows`` appended to the
        inequality rows. Explicit zero entries of ``A_rows`` are kept in the
        device LP (they pad its shape). The new right-hand sides are rounded
        up to the LP's dtype (``_round_up``), so a cut stored at float32 is
        never tighter than the cut."""
        b_rows = _round_up(b_rows, self.lp.c.dtype)
        lp = LP(
            self.c,
            self.A[: self.n_eq].tocsc(),
            self.b[: self.n_eq],
            sp.vstack([self.A[self.n_eq :], sp.csr_matrix(A_rows)], format="csr"),
            np.concatenate([self.b[self.n_eq :], b_rows]),
            self.lower,
            self.upper,
        )
        return MIP(lp, self.integrality, self.offset, self.maximise, self.implied)

    def rows_with(self, A_rows, b_rows):
        """``(A, b)`` as ``with_rows(A_rows, b_rows)`` holds them on the host,
        without building its device LP: the new rows' numbers pass through
        the LP's dtype, as the device copy rounds them."""
        dtype = self.lp.c.dtype
        rows = sp.csr_matrix(A_rows, dtype=np.float64)
        rows.sum_duplicates()
        rows.data = np.asarray(rows.data, dtype).astype(np.float64)
        A = sp.vstack([self.A, rows], format="csr")
        A.eliminate_zeros()
        return A, np.concatenate([self.b, _round_up(b_rows, dtype)])

    def is_feasible(self, x, tol=1e-6):
        """Whether every entry of ``violation(x)`` is at most ``tol``."""
        return all(v <= tol for v in self.violation(x).values())


def _round_up(values, dtype):
    """``values`` (float64) rounded up to the nearest numbers ``dtype``
    holds, as float64: a right-hand side stored at float32 then only
    loosens its row. Rounding to nearest can tighten a cut by an ulp, and
    an integer-rounding cut separated from that row (a cover) can then cut
    off a point that fills the true row exactly."""
    values = np.asarray(values, np.float64)
    if np.dtype(dtype) == np.float64:
        return values.copy()
    stored = values.astype(dtype)
    low = stored.astype(np.float64) < values
    stored[low] = np.nextafter(stored[low], np.array(np.inf, dtype))
    return stored.astype(np.float64)


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
    # Implicit integers (from HiGHS's presolve) were continuous columns of the
    # model; they stay continuous (``MIP.implied``). Counted as integers, a
    # node LP point was integral only with them integral too: on ns1830653
    # (134 of 592 columns) no node LP gave an incumbent, and the search
    # branched on them.
    kinds = list(highs_lp.integrality_)
    implied = np.array(
        [k == highspy.HighsVarType.kImplicitInteger for k in kinds], bool
    )
    integrality = hh.highs_integrality(highs_lp)
    if kinds:
        integrality = integrality & ~implied
    return MIP(
        lp,
        integrality,
        offset=offset,
        maximise=is_max if maximise is None else maximise,
        implied=implied if kinds else None,
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
    O(nnz).

    ``backend``: ``"auto"`` (the default) runs the passes compiled with
    numba when it is installed (``jaddle._propagate_numba``, the same rules
    and the same bounds bit for bit, several times faster), else in numpy
    (``"numpy"``); ``"numba"`` requires numba.
    """

    def __init__(self, A, row_lower, row_upper, integrality, tol=1e-6, backend="auto"):
        A = sp.csc_matrix(A)
        A.eliminate_zeros()
        A.sort_indices()
        if backend not in ("auto", "numba", "numpy"):
            raise ValueError(
                f"backend must be 'auto', 'numba' or 'numpy', not {backend!r}"
            )
        kernel = None if backend == "numpy" else _numba_propagate()
        if backend == "numba" and kernel is None:
            raise ImportError(
                "the numba propagation backend needs numba: pip install jaddle[fast]"
            )
        self._kernel = kernel
        self.col_ptr = A.indptr.astype(np.int64)
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
        if self._kernel is not None:
            lower, upper, infeasible = self._kernel.propagate(
                self.rows,
                self.vals,
                self.positive,
                self.col_ptr,
                self.row_lower,
                self.row_upper,
                self.is_int,
                lower,
                upper,
                int(max_passes),
                self.tol,
                _ROUND_SLACK,
            )
            return lower, upper, bool(infeasible)
        tol = self.tol
        rl, ru = self.row_lower[self.rows], self.row_upper[self.rows]
        for _ in range(max_passes):
            min_c, max_c, min_inf, max_inf, min_act, max_act, n_min_inf, n_max_inf = (
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


class CutoffPropagator(BoundPropagator):
    """
    A ``BoundPropagator`` over the rows plus the objective row
    ``c·x <= cutoff()``, with ``cutoff`` (a callable) read at every call, so
    the row follows the incumbent without being rebuilt. A box it finds
    infeasible holds no solution better than the cutoff, which is all a
    branch and bound needs from it (``solve(objective_propagation=True)``).
    """

    def __init__(
        self, A, row_lower, row_upper, c, integrality, cutoff, tol=1e-6, backend="auto"
    ):
        super().__init__(
            sp.vstack(
                [sp.csr_matrix(A), sp.csr_matrix(np.asarray(c, np.float64)[None, :])]
            ),
            np.append(row_lower, -np.inf),
            np.append(row_upper, np.inf),
            integrality,
            tol=tol,
            backend=backend,
        )
        self.cutoff = cutoff

    def __call__(self, lower, upper, max_passes=5):
        value = float(self.cutoff())
        # A hair above the cutoff, against rounding in the row's activity.
        self.row_upper[-1] = (
            value + 1e-9 * (1.0 + abs(value)) if np.isfinite(value) else np.inf
        )
        return super().__call__(lower, upper, max_passes)


def _propagation_data(A, row_lower, row_upper, integrality, nnz_capacity=None):
    """The rows of ``BoundPropagator`` as float64 device arrays, for
    ``_propagate_jax`` and ``_lagrangian_terms_jax``. Call under x64.

    With ``nnz_capacity``, the entries are padded to at least that many,
    so rows with a few more or fewer nonzeros (a refreshed tree cut block)
    give arrays of the same shapes and a jitted kernel of them does not
    recompile. The padding is one extra row, free (``-inf`` to ``inf``),
    whose entries are 1 on column 0: it tightens no bound and is never
    violated, and a dual of 0 for it (``_StrongKernels``) adds nothing to
    a reduced cost. The arrays' row count, ``row_lower.shape[0]``, is then
    one more than ``A``'s."""
    A = sp.csc_matrix(A)
    A.eliminate_zeros()
    A.sort_indices()
    m, n = A.shape
    rows = A.indices.astype(np.int32)
    cols = np.repeat(np.arange(n), np.diff(A.indptr)).astype(np.int32)
    vals = A.data.astype(np.float64)
    row_lower = np.asarray(row_lower, np.float64)
    row_upper = np.asarray(row_upper, np.float64)
    if nnz_capacity is not None:
        pad = max(int(nnz_capacity), vals.size) - vals.size
        rows = np.concatenate([rows, np.full(pad, m, np.int32)])
        cols = np.concatenate([cols, np.zeros(pad, np.int32)])
        vals = np.concatenate([vals, np.ones(pad)])
        row_lower = np.append(row_lower, -np.inf)
        row_upper = np.append(row_upper, np.inf)
    return {
        "rows": jnp.asarray(rows, jnp.int32),
        "cols": jnp.asarray(cols, jnp.int32),
        "vals": jnp.asarray(vals, jnp.float64),
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
        min_c, max_c, min_inf, max_inf, min_act, max_act, n_min, n_max = activities(
            lo, up
        )
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
        bad = jnp.any(crossed & is_int) | jnp.any(
            lo2 > up2 + tol * (1.0 + jnp.abs(up2))
        )
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


class ProbeResult(
    collections.namedtuple(
        "ProbeResult", "lower upper infeasible fixed tightened probed cache"
    )
):
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
        propagator._activities(
            np.asarray(lower, np.float64), np.asarray(upper, np.float64)
        )
    )
    rows, cols = propagator.rows, propagator.cols
    ru, rl = propagator.row_upper[rows], propagator.row_lower[rows]
    with np.errstate(invalid="ignore"):
        flipped_min = min_act[rows] - np.where(min_inf, 0.0, min_c) + max_c
        flipped_max = max_act[rows] - np.where(max_inf, 0.0, max_c) + min_c
        over = np.where(
            (n_min[rows] - min_inf == 0) & np.isfinite(flipped_min),
            flipped_min - ru,
            -np.inf,
        )
        under = np.where(
            (n_max[rows] - max_inf == 0) & np.isfinite(flipped_max),
            rl - flipped_max,
            -np.inf,
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


# Probing (``probe``): propagation passes per side, and the implications the
# device backend keeps per side. On map06 a binary's implications reach
# continuous variables down chains of rows up to about 50 passes deep: 5
# passes found 897 of the 2,013 a fixed point finds for one binary, and the
# implied-bound cuts (exact LPs, with Gomory cuts) reached −401.8 at 5 passes
# and 64 implications against −395.8 at a fixed point with all of them.
_PROBE_MAX_PASSES = 100
# Variables per call of the numba probing kernel: the deadline is checked
# between calls.
_PROBE_NUMBA_CHUNK = 64
# ``None``: every implication; the outputs are no larger than the propagated
# boxes the chunk holds anyway.
_PROBE_MAX_IMPLICATIONS = None


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
    max_implications=_PROBE_MAX_IMPLICATIONS,
    tol=1e-6,
    max_useless=None,
    max_passes=_PROBE_MAX_PASSES,
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
    fiball all 32,399 variables took 11 s for one tightened bound. Under
    numba a probe that implies a bound on another variable also counts as
    useful (its implications go to the cache).
    ``backend`` is ``"numba"`` (``jaddle._probe_numba``: each side
    propagated incrementally from the probed variable, visiting only the
    rows whose variables changed, and undone after), ``"numpy"`` (one
    ``BoundPropagator`` call per side, each a full
    sweep from the box tightened so far), ``"jax"`` (chunks of variables at
    once, both sides of each propagated by ``_propagate_jax`` in one
    vmapped call of fixed shape) or ``"auto"`` (numba when it is installed,
    else numpy up to 10,000 nonzeros and JAX above).
    Propagation runs at most ``max_passes`` passes (it stops sooner at a
    fixed point), and the device backend keeps at most ``max_implications``
    implications per side in the cache (``None``: all; the box tightenings
    use them all either way).

    Returns a ``ProbeResult``.
    """
    lower = np.array(lower, np.float64)
    upper = np.array(upper, np.float64)
    integrality = np.asarray(integrality, bool)
    start_lower, start_upper = lower.copy(), upper.copy()
    cache = ProbingCache()
    propagator = functools.partial(
        BoundPropagator(A, row_lower, row_upper, integrality, tol=tol),
        max_passes=max_passes,
    )
    lower, upper, infeasible = propagator(lower, upper)

    def result(infeasible, probed):
        tightened = int(np.count_nonzero((lower > start_lower) | (upper < start_upper)))
        fixed = int(
            np.count_nonzero(
                integrality & (lower == upper) & (start_lower < start_upper)
            )
        )
        return ProbeResult(lower, upper, infeasible, fixed, tightened, probed, cache)

    if infeasible:
        return result(True, 0)
    order = _probe_order(A, row_lower, row_upper, integrality, lower, upper, tol)
    if max_variables is not None:
        order = order[: int(max_variables)]
    if backend not in ("auto", "numba", "numpy", "jax"):
        raise ValueError(
            f"backend must be 'auto', 'numba', 'numpy' or 'jax', not {backend!r}"
        )
    if backend == "auto":
        if _numba_probe() is not None:
            backend = "numba"
        else:
            backend = "numpy" if A.nnz <= _SMALL_MIP_NNZ else "jax"
    if backend == "numba" and _numba_probe() is None:
        raise ValueError("the numba probing backend needs numba: pip install jaddle[fast]")
    probed = 0
    useless = 0  # variables probed in a row that changed no bound

    def progress(old_lower, old_upper):
        return bool(np.any(lower != old_lower) or np.any(upper != old_upper))

    if backend == "numba":
        kernel = _numba_probe()
        csc = sp.csc_matrix(A, dtype=np.float64)
        csc.eliminate_zeros()
        csr = csc.tocsr()
        arrays = (
            csr.indptr.astype(np.int64),
            csr.indices.astype(np.int64),
            csr.data,
            csc.indptr.astype(np.int64),
            csc.indices.astype(np.int64),
            np.asarray(row_lower, np.float64),
            np.asarray(row_upper, np.float64),
            integrality,
        )
        limit = -1 if max_useless is None else int(max_useless)
        position = 0
        while position < order.size:
            if deadline is not None and time.time() >= deadline:
                break
            if limit >= 0 and useless >= limit:
                break
            (
                infeasible, consumed, count, useless, var, split, empty, ptr, idx, lo, up
            ) = kernel.probe_chunk(
                order[position : position + _PROBE_NUMBA_CHUNK].astype(np.int64),
                *arrays, lower, upper, int(max_passes), float(tol), _ROUND_SLACK,
                int(useless), limit,
            )
            position += consumed
            probed += count
            for p in range(count):
                sides = [
                    None
                    if empty[p, side]
                    else (
                        idx[ptr[2 * p + side] : ptr[2 * p + side + 1]].copy(),
                        lo[ptr[2 * p + side] : ptr[2 * p + side + 1]].copy(),
                        up[ptr[2 * p + side] : ptr[2 * p + side + 1]].copy(),
                    )
                    for side in range(2)
                ]
                cache.add(var[p], split[p], sides[0], sides[1])
            if infeasible:
                return result(True, probed)
            if consumed == 0:
                break
        return result(False, probed)

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
    k = n if max_implications is None else int(min(max_implications, n))
    with jax.enable_x64(True):
        data = _propagation_data(A, row_lower, row_upper, integrality)
        chunk = jax.jit(
            functools.partial(_probe_chunk_jax, m=m, k=k, max_passes=int(max_passes))
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
                chunk(
                    data, jnp.asarray(lower), jnp.asarray(upper), padded, split, valid
                )
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
                        (
                            idx,
                            out["implied_lower"][i, side, :kept],
                            out["implied_upper"][i, side, :kept],
                        )
                    )
                cache.add(cols[i], split[i], sides[0], sides[1])
    return result(False, probed)


def _probe_chunk_jax(data, lower, upper, cols, split, valid, m, k, max_passes=5):
    """Both sides of each variable in ``cols`` propagated at once (at most
    ``max_passes`` passes); returns the box that all of them together
    imply, whether some variable has both sides empty, and each side's
    emptiness and (at most ``k``) implied bounds. Runs under x64."""
    n = lower.shape[0]

    def sides(j, s):
        up_down = upper.at[j].min(s)
        lo_up = lower.at[j].max(s + 1.0)
        return jnp.stack([lower, lo_up]), jnp.stack([up_down, upper])

    lo, up = jax.vmap(sides)(cols, split)
    lo, up = lo.reshape(-1, n), up.reshape(-1, n)
    empty0 = jnp.any(lo > up, axis=1)
    lo, up, empty = jax.vmap(lambda a, b: _propagate_jax(data, a, b, m, max_passes))(
        lo, up
    )
    empty = (empty | empty0).reshape(-1, 2) & valid[:, None]
    lo, up = lo.reshape(-1, 2, n), up.reshape(-1, 2, n)
    e0, e1 = empty[:, 0][:, None], empty[:, 1][:, None]
    implied_lo = jnp.where(
        e0, lo[:, 1], jnp.where(e1, lo[:, 0], jnp.minimum(lo[:, 0], lo[:, 1]))
    )
    implied_up = jnp.where(
        e0, up[:, 1], jnp.where(e1, up[:, 0], jnp.maximum(up[:, 0], up[:, 1]))
    )
    both = empty[:, 0] & empty[:, 1]
    use = (valid & ~both)[:, None]
    new_lower = jnp.maximum(
        lower, jnp.max(jnp.where(use, implied_lo, -jnp.inf), axis=0)
    )
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


def _same_shapes(a, b):
    """Whether two pytrees of arrays have the same structure, shapes and
    dtypes (so a jitted function of one takes the other without a new
    compile)."""
    if jax.tree.structure(a) != jax.tree.structure(b):
        return False
    return all(
        np.shape(x) == np.shape(y) and jnp.result_type(x) == jnp.result_type(y)
        for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))
    )


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

    def __init__(self, mip, propagate, reduced_cost_tolerance, size, nnz_capacity=None):
        # Entries padded to ``nnz_capacity`` (``_propagation_data``): a
        # refreshed cut block of the same capacity keeps the compile.
        self.nnz_capacity = nnz_capacity
        self.data = self.data_for(mip)
        self.m, self.n = mip.A.shape
        self.propagate = bool(propagate)
        self.rc_tol = float(reduced_cost_tolerance)
        self.size = int(size)
        self._children = jax.jit(self._children_impl)

    def data_for(self, mip):
        """The device data of ``mip``'s rows (``_propagation_data``),
        padded to ``nnz_capacity``."""
        with jax.enable_x64(True):
            return _propagation_data(
                mip.A,
                mip.row_lower,
                mip.row_upper,
                mip.integrality,
                nnz_capacity=self.nnz_capacity,
            )

    def set_mip(self, mip):
        """Point the kernels at ``mip``'s rows, when its data has the shapes
        they were compiled for (a cut block of the same capacity): the
        compiled kernels are kept. Returns whether it did."""
        data = self.data_for(mip)
        if mip.A.shape != (self.m, self.n) or not _same_shapes(data, self.data):
            return False
        self.data = data
        return True

    def __call__(self, lower, upper, var, xv, r, constant):
        """Host arrays in (``size`` rows each), host arrays out: ``lower``,
        ``upper`` (``size, 2, n``), ``crossed`` (the branching emptied the
        box before propagation), ``empty``, ``bound``, ``rigorous``
        (``size, 2``); side 0 is the down child."""
        with jax.enable_x64(True):
            return jax.device_get(self.launch(lower, upper, var, xv, r, constant))

    def launch(self, lower, upper, var, xv, r, constant):
        """``__call__`` without waiting: the outputs stay on the device
        (read them with ``jax.device_get`` under x64)."""
        with jax.enable_x64(True):
            return self._children(self.data, lower, upper, var, xv, r, constant)

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
            rows = data["row_lower"].shape[0]  # the padding row too
            lo, up, empty = jax.vmap(lambda a, b: _propagate_jax(data, a, b, rows))(
                lo, up
            )
        else:
            empty = jnp.any(lo > up, axis=1)
        wlo, wup = jax.vmap(_widen_continuous, in_axes=(0, 0, None))(
            lo, up, data["is_int"]
        )
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

    def __init__(
        self,
        mip,
        propagate,
        reduced_cost_tolerance,
        screen_size,
        k,
        lp_dtype,
        branch_implied=False,
        nnz_capacity=None,
    ):
        self.lp_dtype = lp_dtype
        self.branch_implied = bool(branch_implied)
        # Entries padded as ``_ChildKernels``' (``_propagation_data``).
        self.nnz_capacity = nnz_capacity
        self.data = self.data_for(mip)
        self.m, self.n = mip.A.shape
        self.n_eq = mip.n_eq
        self.propagate = bool(propagate)
        self.rc_tol = float(reduced_cost_tolerance)
        self.screen_size, self.k = int(screen_size), int(k)
        self._screen = jax.jit(self._screen_impl)
        self._child_bounds = jax.jit(self._child_bounds_impl)

    def data_for(self, mip):
        """The device data of ``mip``: its rows (``_propagation_data``),
        costs, right-hand sides and bounds."""
        with jax.enable_x64(True):
            data = _propagation_data(
                mip.A,
                mip.row_lower,
                mip.row_upper,
                mip.integrality,
                nnz_capacity=self.nnz_capacity,
            )
            data["c"] = jnp.asarray(mip.c, jnp.float64)
            data["b"] = jnp.asarray(mip.b, jnp.float64)
            data["lower"] = jnp.asarray(mip.lower, jnp.float64)
            data["upper"] = jnp.asarray(mip.upper, jnp.float64)
            # Implied integers whose box bounds go into the child LPs,
            # rounded outward (``branch_implied``, ``_BranchAndBound.lp_lower``).
            data["implied_box"] = jnp.asarray(mip.implied & self.branch_implied)
        return data

    def set_mip(self, mip):
        """Point the kernels at ``mip``, when its data has the shapes they
        were compiled for (a cut block of the same capacity). Returns
        whether it did."""
        data = self.data_for(mip)
        if (
            mip.A.shape != (self.m, self.n)
            or mip.n_eq != self.n_eq
            or not _same_shapes(data, self.data)
        ):
            return False
        self.data = data
        return True

    def screen(self, *args):
        with jax.enable_x64(True):
            return self._screen(self.data, *args)

    def child_bounds(self, solution, lower, upper):
        with jax.enable_x64(True):
            return self._child_bounds(
                self.data, solution.dual_eq, solution.dual_ineq, lower, upper
            )

    def _screen_impl(
        self,
        data,
        lower,
        upper,
        cand,
        valid,
        xc,
        r,
        constant,
        est_down,
        est_up,
        eligible,
        bound,
        cutoff,
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
                lambda a, b: _propagate_jax(data, a, b, data["row_lower"].shape[0])
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
        implied = data["implied_box"]
        lp_lo = jnp.where(
            is_int,
            sel_lo,
            jnp.where(
                implied,
                jnp.maximum(jnp.floor(sel_lo + _ROUND_SLACK), data["lower"]),
                data["lower"],
            ),
        ).astype(self.lp_dtype)
        lp_up = jnp.where(
            is_int,
            sel_up,
            jnp.where(
                implied,
                jnp.minimum(jnp.ceil(sel_up - _ROUND_SLACK), data["upper"]),
                data["upper"],
            ),
        ).astype(self.lp_dtype)
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
        constant = -(y @ data["b"])
        # A dual of 0 for the padding row (``nnz_capacity``), if any.
        extra = data["row_lower"].shape[0] - self.m
        if extra:
            y = jnp.pad(y, ((0, 0), (0, extra)))
        rows, cols, vals = data["rows"], data["cols"], data["vals"]
        r = data["c"] + jax.vmap(
            lambda yy: jax.ops.segment_sum(vals * yy[rows], cols, num_segments=self.n)
        )(y)
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
        # One objective proof that holds over the whole global box, kept
        # apart from the learned rows and replaced as a whole
        # (``set_global``): the root's reduced-cost row.
        self.global_row = None

    def __len__(self):
        return len(self.rows) + (self.global_row is not None)

    def set_global(self, constant, r, global_lower, global_upper, keep=64):
        """Hold the objective proof ``constant + r·x <= cutoff`` for every
        ``x`` in the global box (Lagrangian terms over it: the root's), in
        place of the last one. Unlike ``add``, it need not exclude a box: it
        propagates where a node's bounds have used up part of the room
        ``cutoff − (constant + Σ min r_j x_j)``. Only variables with ``|r_j|``
        times their range at least ``room / keep`` stay; the rest go to the
        right-hand side at their least over the box (``r_j x_j`` is at least
        that, so the row stays valid). Returns whether a row is held."""
        self.global_row = None
        self._propagator = None
        r = np.asarray(r, np.float64)
        if not np.isfinite(constant) or not np.abs(r).max(initial=0.0) > 0.0:
            return False
        with np.errstate(invalid="ignore"):
            least = np.where(r > 0, r * global_lower, r * global_upper)
            span = np.abs(r) * (global_upper - global_lower)
        least = np.where(r != 0, least, 0.0)
        if not np.all(np.isfinite(least)):
            return False  # an unbounded term: no finite room
        room = self.cutoff_value() - constant - least.sum()
        if not (np.isfinite(room) and room >= 0):
            return False
        idx = np.flatnonzero((r != 0) & (span > 0) & (span * keep >= room))
        if idx.size == 0:
            return False
        rest = np.ones(r.size, bool)
        rest[idx] = False
        worst = least[rest]
        offset = -constant - worst.sum()
        offset -= 1e-9 * (1.0 + abs(constant) + np.abs(worst).sum())  # loosen: safe
        self.global_row = (idx, r[idx], float(offset), True)
        return True

    def add(
        self, constant, r, local_lower, local_upper, global_lower, global_upper, kind
    ):
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
        return self._add_row(
            idx, coef, offset, kind == "objective", kind, local_lower, local_upper
        )

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

    def _add_row(
        self, idx, coef, offset, objective, kind, local_lower=None, local_upper=None
    ):
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
        rows = list(self.rows) + (
            [self.global_row] if self.global_row is not None else []
        )
        if not rows:
            return lower, upper, False
        if self._propagator is None:
            idx, coef, offsets, objective = zip(*rows)
            lengths = [i.size for i in idx]
            A = sp.csr_matrix(
                (
                    np.concatenate(coef),
                    np.concatenate(idx),
                    np.concatenate([[0], np.cumsum(lengths)]),
                ),
                shape=(len(rows), self.n),
            )
            self._offsets = np.array(offsets)
            self._objective = np.array(objective)
            cutoff = self.cutoff_value()
            row_upper = self._offsets + np.where(self._objective, cutoff, 0.0)
            self._propagator = BoundPropagator(
                A, np.full(len(rows), -np.inf), row_upper, self.is_int, tol=self.tol
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
    return MIP(lp, integrality, mip.offset, mip.maximise, mip.implied), int(hit.sum())


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


def _exact_side_bound(mip, A_T, y_eq, y_ineq, lower, upper, is_int, reduced_cost_tol):
    """``_BranchAndBound.exact_bound`` from arrays, so the exact lane's
    threads can bound their own solutions: the Lagrangian bound of the
    duals ``y_eq``/``y_ineq`` (Jaddle's convention; ``y_eq`` padded with
    zeros or cut to ``mip.n_eq``) over ``mip``'s rows and the box, with
    round-off reduced costs at infinite bounds zeroed (``vertex_terms``)
    and continuous variables' bounds widened a little (``bound_over``).
    Returns ``(bound, rigorous)``, or ``None`` when ``y_ineq`` is shorter
    than ``mip``'s inequality rows."""
    n_eq = mip.n_eq
    m_ineq = mip.A.shape[0] - n_eq
    y_eq = np.asarray(y_eq, np.float64)
    y_ineq = np.asarray(y_ineq, np.float64)
    if y_ineq.size < m_ineq:
        return None
    kernel = _numba_bound()
    if kernel is not None and _csr_transpose(A_T, mip.c, n_eq + m_ineq):
        bound, rigorous = kernel.exact_side_bound(
            _f64(mip.c), A_T.indptr, A_T.indices, _f64(A_T.data), _f64(mip.b),
            n_eq, y_eq, y_ineq, _f64(lower), _f64(upper), np.asarray(is_int, bool),
            _EXACT_DUAL_ROUNDOFF, float(reduced_cost_tol),
        )
        return float(bound), bool(rigorous)
    y = np.zeros(n_eq + m_ineq)
    k = min(n_eq, y_eq.size)
    y[:k] = y_eq[:k]
    y[n_eq:] = y_ineq[:m_ineq]
    constant, r = lagrangian_terms(mip.c, mip.A, mip.b, n_eq, y, A_T=A_T)
    roundoff = (
        ((r > 0) & ~np.isfinite(lower)) | ((r < 0) & ~np.isfinite(upper))
    ) & (np.abs(r) <= _EXACT_DUAL_ROUNDOFF * (1.0 + np.abs(mip.c).max(initial=0.0)))
    r = np.where(roundoff, 0.0, r)
    cont = ~is_int
    lower = np.where(cont, lower - 1e-7 * (1.0 + np.abs(lower)), lower)
    upper = np.where(cont, upper + 1e-7 * (1.0 + np.abs(upper)), upper)
    return box_bound(constant, r, lower, upper, reduced_cost_tolerance=reduced_cost_tol)


def lagrangian_terms(c, A, b, n_eq, y, A_T=None):
    """The parts of ``lagrangian_bound`` that do not depend on the box:
    ``(−bᵀy, r)`` with ``r = c + Aᵀy`` (inequality duals clipped to
    ``y >= 0``). For every feasible ``x``, ``cᵀx >= −bᵀy + rᵀx``. With
    numba and ``A_T`` a scipy CSR matrix, one compiled pass
    (``jaddle._bound_numba``)."""
    kernel = _numba_bound()
    if kernel is not None and _csr_transpose(A_T, c, np.shape(y)[0]):
        constant, r = kernel.lagrangian_terms(
            _f64(c), A_T.indptr, A_T.indices, _f64(A_T.data), _f64(b), n_eq, _f64(y)
        )
        return float(constant), r
    y = np.array(y, dtype=np.float64)
    y[n_eq:] = np.maximum(y[n_eq:], 0.0)
    r = c + (A_T @ y if A_T is not None else A.T @ y)
    return float(-(b @ y)), r


def box_bound(constant, r, lower, upper, reduced_cost_tolerance=0.0):
    """``constant + Σⱼ min(rⱼlⱼ, rⱼuⱼ)``: the Lagrangian bound over the box
    ``[lower, upper]`` from ``lagrangian_terms``. One set of duals bounds
    any box this way, so a parent's duals bound its children. Returns
    ``(bound, rigorous)`` as ``lagrangian_bound`` does. With numba, one
    compiled pass (``jaddle._bound_numba``)."""
    kernel = _numba_bound()
    if kernel is not None:
        r, lower, upper = _f64(r), _f64(lower), _f64(upper)
        if r.ndim == 1 and lower.shape == r.shape == upper.shape:
            bound, rigorous = kernel.box_bound(
                float(constant), r, lower, upper, float(reduced_cost_tolerance)
            )
            return float(bound), bool(rigorous)
    return _box_bound_numpy(constant, r, lower, upper, reduced_cost_tolerance)


def _box_bound_numpy(constant, r, lower, upper, reduced_cost_tolerance=0.0):
    """``box_bound`` in numpy (the reference for ``jaddle._bound_numba``)."""
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


# ``DualRepair``: a continuous column's side is wide when it is more than
# this many (1 + |xⱼ|) from the node's primal; and the scaling passes.
_DUAL_REPAIR_WIDE = 10.0
_DUAL_REPAIR_PASSES = 10
# A wrong-signed reduced cost is repaired only when it costs the bound more
# than this (|rⱼ| times the room to the side it points at): round-off at
# exact duals is left alone.
_DUAL_REPAIR_MIN_LOSS = 1e-9


class DualRepair:
    """
    Sign repair of approximate duals before their Lagrangian bound
    (``box_bound``) is taken, for continuous columns with a wide box.

    The bound charges each reduced cost ``rⱼ`` at the bound it points at,
    so a PDHG dual whose ``rⱼ`` is off by ε on a column boxed to 1e6 loses
    ε·1e6 of bound. On CMS750_4 (2,734 continuous columns in difference
    rows, ``0 <= s <= ~1e6``, objective 0 on them) the device's node duals
    at ``node_lp_tolerance=1e-3`` had |rⱼ| <= 0.03 on those columns and
    bounds 50 to 43,000 below HiGHS's LP over the same box, almost all of
    it from rⱼ < 0 at the upper bound.

    A side of a continuous column is *wide* when it is more than
    ``wide·(1 + |xⱼ|)`` from the node's primal ``x``; ``rⱼ`` must not point
    at it (rⱼ >= 0 below a wide upper side, rⱼ <= 0 above a wide lower
    one). Lowering an inequality dual ``yᵢ >= 0`` raises ``rⱼ`` where
    ``Aᵢⱼ < 0`` and lowers it where ``Aᵢⱼ > 0``, and ``yᵢ = 0`` is always
    allowed, so each pass scales down, at each wrong-signed column, the
    duals of the rows that push it the wrong way just enough to zero it
    (a row takes the smallest factor of its columns). ``y`` only falls, so
    the passes settle; equality duals are left alone. On CMS750_4 ten
    passes (about 7 ms a node) recovered 98.7% of the shortfall.

    Any duals give a valid bound, so the caller keeps whichever of the two
    bounds is higher and the repair cannot cost rigour.
    """

    def __init__(self, mip, is_int, wide=_DUAL_REPAIR_WIDE, passes=_DUAL_REPAIR_PASSES):
        A = sp.coo_matrix(mip.A)
        cont = ~np.asarray(is_int, bool)
        keep = (A.row >= mip.n_eq) & cont[A.col] & (A.data != 0)
        self.rows = A.row[keep].astype(np.int64)
        self.cols = A.col[keep].astype(np.int64)
        self.vals = np.asarray(A.data[keep], np.float64)
        self.negative = self.vals < 0
        self.cont = cont
        self.n_eq = mip.n_eq
        self.m = A.shape[0]
        self.n = A.shape[1]
        self.wide = float(wide)
        self.passes = int(passes)

    def __bool__(self):
        return self.rows.size > 0

    def wide_sides(self, x, lower, upper):
        """The continuous columns whose upper and lower sides are wide at
        ``x`` (``(up_wide, lo_wide)``)."""
        room = self.wide * (1.0 + np.abs(x))
        with np.errstate(invalid="ignore"):
            up_wide = self.cont & (upper - x > room)
            lo_wide = self.cont & (x - lower > room)
        return up_wide, lo_wide

    def __call__(self, c, A_T, y, x, lower, upper):
        """Repaired duals for ``y`` (Jaddle's convention, inequality duals
        clipped to ``>= 0``), or ``None`` when no wide side is pointed at."""
        x = np.asarray(x, np.float64)
        if x.shape != (self.n,) or not np.all(np.isfinite(x)):
            return None
        up_wide, lo_wide = self.wide_sides(x, lower, upper)
        if not (up_wide.any() or lo_wide.any()):
            return None
        y = np.array(y, np.float64)
        y[self.n_eq :] = np.maximum(y[self.n_eq :], 0.0)
        rows, cols, vals, negative = self.rows, self.cols, self.vals, self.negative
        changed = False
        x_up = upper - x  # the room each side would charge rⱼ for
        x_down = x - lower
        for _ in range(self.passes):
            r = c + A_T @ y
            with np.errstate(invalid="ignore"):
                need_up = up_wide & (r < 0) & (-r * x_up > _DUAL_REPAIR_MIN_LOSS)
                need_down = lo_wide & (r > 0) & (r * x_down > _DUAL_REPAIR_MIN_LOSS)
            if not (need_up.any() or need_down.any()):
                break
            mass = vals * y[rows]
            # What pushes each column's rⱼ down (Aᵢⱼ < 0 rows) and up.
            down_mass = np.bincount(
                cols, weights=np.where(negative, -mass, 0.0), minlength=self.n
            )
            up_mass = np.bincount(
                cols, weights=np.where(negative, 0.0, mass), minlength=self.n
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                f_up = np.where(need_up, 1.0 + r / down_mass, 1.0)
                f_down = np.where(need_down, 1.0 - r / up_mass, 1.0)
            f_up = np.clip(np.nan_to_num(f_up, nan=0.0), 0.0, 1.0)
            f_down = np.clip(np.nan_to_num(f_down, nan=0.0), 0.0, 1.0)
            factor = np.ones(self.m)
            np.minimum.at(factor, rows, np.where(negative, f_up[cols], f_down[cols]))
            if np.all(factor >= 1.0):
                break
            y *= factor
            changed = True
        return y if changed else None


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


def branching_impact(mip, integrality=None):
    """
    A prior on how much branching on each integer variable can move the
    bound, for picking strong-branching candidates before any pseudocost is
    known: its column length (the rows a change to it reaches) times
    ``1 + |cⱼ|/c̄``, with ``c̄`` the mean nonzero ``|c|``. Zero for
    continuous variables, at least 1 for integers.

    On buildingenergy the 7 integer sizing variables (all the integer cost,
    3,700–35,000 rows each) score 10⁴–10⁵ and the 26,280 hourly integers
    (3 rows, no cost) score 3. At the root, branching on the one fractional
    sizing variable raises the LP bound to the optimum (+37.6 up, +93.8
    down); branching on an hourly integer moves it by about 0.
    """
    c = np.abs(np.asarray(mip.c, np.float64))
    scale = c[c > 0].mean() if np.any(c > 0) else 1.0
    length = np.diff(sp.csc_matrix(mip.A).indptr).astype(np.float64)
    impact = np.maximum(length, 1.0) * (1.0 + c / scale)
    # ``integrality``: the columns that may be branched on (default the
    # integers).
    integrality = (
        mip.integrality if integrality is None else np.asarray(integrality, bool)
    )
    return np.where(integrality, impact, 0.0)


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
# Knapsack covers (``_knapsack``): slack on a row's capacity relative to
# ``1 + |b|``, as the feasibility checks (``feasibility_tolerance``) allow.
_KNAPSACK_TOL = 1e-6
# ``_substituted_cmir``: a variable bound and a simple one this close to the
# LP point tie (the coefficient's sign then picks the side).
_VB_TIE = 1e-6
# At most this many rows start a separation per round.
_CUT_MAX_ROWS = 2000
# Rows aggregated into one c-MIR base inequality, beyond the first.
_CUT_MAX_AGGREGATION = 2
# The search's defaults for ``CutSeparator``'s ``aggregation_width`` (first
# aggregation steps tried per row) and ``cmir_per_row`` (c-MIR cuts kept per
# row), and for ``select_cuts``'s ``weights``.
_CUT_AGGREGATION_WIDTH = 3
_CUT_CMIR_PER_ROW = 3
_CUT_SCORE_WEIGHTS = (0.1, 0.1, 0.5)
# Two cuts this parallel (cosine) are duplicates.
_CUT_MAX_PARALLELISM = 0.99
# Tree cuts (``tree_cuts``): the least room reserved for them (rows) beyond
# the root's cuts; at least two rounds' worth (``max_tree_cuts_per_round``).
_TREE_CUT_MIN_ROWS = 32
# Reduced costs at an exact (crossover) vertex this small, relative to
# 1 + max|c|, are round-off (``root_exact_terms``).
_EXACT_DUAL_ROUNDOFF = 1e-9
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
    LP point ``z`` in complemented form, or ``None`` when no cover exists.

    The capacity has ``_KNAPSACK_TOL`` of slack, as the feasibility checks
    allow, so a cover's items overfill the row by more than rounding. A
    cover is a jump of a whole unit: rows stored at float32 (a cut's
    right-hand side rounded down by 1e-7) once made a set the optimum
    fills exactly into a cover, and cut off the optimum (a
    generalised-assignment knapsack, ``precision="float32"``)."""
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
    capacity += _KNAPSACK_TOL * (1.0 + max(abs(b), abs(capacity)))
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
    outside = np.ones(w.size, dtype=bool)
    outside[cover] = False
    rest = np.flatnonzero(outside)  # sorted, as np.setdiff1d gave, without its sort
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
    # An integral point inside the knapsack satisfies every valid cover
    # inequality, so nothing can cut it off (on bab5's root most tight rows
    # are like this).
    z_int = np.round(z)
    if np.all(np.abs(z - z_int) <= 1e-9) and w @ z_int <= threshold:
        return []
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
        rhs_z = cover.size - 1.0
        # Lifting gives each item outside the cover at most |C| − 1, so a
        # cover that no lifting can make violated is dropped before the
        # lifting's dynamic program (most covers at a node point).
        in_cover = z[cover].sum()
        positive = np.maximum(z, 0.0)
        outside = positive.sum() - positive[cover].sum()
        if in_cover - rhs_z + rhs_z * outside <= 1e-6:
            continue
        pi = _lift_cover(cover, w, z, capacity)
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


def cmir_cut(idx, a, b, x, lower, upper, is_int, max_divisors=64, variable_bounds=None):
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


def _nearest_variable_bound(kinds, y, x, lower, upper, positive):
    """The variable bound ``(kind, z, c, d)`` of ``kinds`` (``y``'s entry of
    ``variable_bounds``) that ``_substituted_cmir`` shifts ``y`` to at
    ``x``, or ``None`` to keep its simple bounds: the nearest to ``x``, a
    variable one only when no simple bound is nearer. Bounds within
    ``_VB_TIE`` tie, and a tie goes to the side whose shifted term the MIR
    drops: a lower bound when the row's coefficient on ``y`` is positive
    (``positive``), an upper one when it is negative, as HiGHS's transformed
    LP does. Unused arcs of a fixed-charge network sit at y = 0 = u·z, at
    both bounds: shifting an outflow to its VUB put u·z terms into the
    cut-set inequalities (beasleyC3: x₁ + x₂ >= 1 became x₁ + x₂ + 81·x₃ +
    81·x₄ − y₃ − y₄ >= 1), too shallow to be chosen."""
    prefer = "lb" if positive else "ub"
    x_y = x[y]
    lb_distance, ub_distance = x_y - lower[y], upper[y] - x_y
    best_distance = min(lb_distance, ub_distance)
    held = "lb" if lb_distance <= ub_distance else "ub"
    best = None
    for kind in ("ub", "lb"):
        for z, c, d in kinds[kind]:
            implied = d + c * x[z]
            distance = implied - x_y if kind == "ub" else x_y - implied
            if distance < best_distance - _VB_TIE:
                better = True
            elif distance <= best_distance + _VB_TIE:
                better = kind == prefer or held != prefer
            else:
                better = False
            if better:
                best_distance, best, held = distance, (kind, z, c, d), kind
    return best


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
        best = _nearest_variable_bound(bounds[y], y, x, lower, upper, a_y > 0)
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
        inside = (
            integer
            & np.isfinite(lo)
            & np.isfinite(up)
            & (xv > lo + 1e-6)
            & (xv < up - 1e-6)
        )
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
        if (
            not is_int[j]
            and lower[j] == 0.0
            and variable_bounds
            and j in variable_bounds
        ):
            for z, c, d in variable_bounds[j]["ub"]:
                if (
                    d == 0.0
                    and c > 0
                    and is_int[z]
                    and lower[z] == 0.0
                    and upper[z] == 1.0
                ):
                    if vub is None or c * x[z] < vub[1] * x[vub[0]]:
                        vub = (z, c)  # the tightest at x
        if vub is not None:
            z, c = vub
            arcs.append(
                dict(
                    sign=sign,
                    u=mag * c,
                    f=[(j, mag)],
                    f0=0.0,
                    w=z,
                    fval=mag * x[j],
                    wval=x[z],
                )
            )
        elif binary:
            arcs.append(
                dict(
                    sign=sign,
                    u=mag,
                    f=[(j, mag)],
                    f0=0.0,
                    w=j,
                    fval=mag * x[j],
                    wval=x[j],
                )
            )
        else:
            lo, up = lower[j], upper[j]
            if np.isfinite(up) and (not np.isfinite(lo) or up - x[j] < x[j] - lo):
                # y = up − t: the term flips to the other side.
                b -= aj * up
                arcs.append(
                    dict(
                        sign=-sign,
                        u=mag * (up - lo),
                        f=[(j, -mag)],
                        f0=mag * up,
                        w=None,
                        fval=mag * (up - x[j]),
                        wval=1.0,
                    )
                )
            elif np.isfinite(lo):
                b -= aj * lo
                arcs.append(
                    dict(
                        sign=sign,
                        u=mag * (up - lo),
                        f=[(j, mag)],
                        f0=-mag * lo,
                        w=None,
                        fval=mag * (x[j] - lo),
                        wval=1.0,
                    )
                )
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
    plus = [
        i for i, arc in enumerate(arcs) if arc["sign"] > 0 and np.isfinite(arc["u"])
    ]
    minus = [
        i for i, arc in enumerate(arcs) if arc["sign"] < 0 and np.isfinite(arc["u"])
    ]
    saturated = [
        i
        for i in minus
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
# Rows per kernel call, fewer for wide rows: a call holds at most
# _SEPARATION_ENTRIES padded entries. At bab5's root one 5,538-term base
# padded its call to 256 × 16,384 entries, nine times all the others'.
_SEPARATION_CHUNK = 256
_SEPARATION_ENTRIES = 256 * 1024


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
    return np.maximum(
        16, 4 ** np.ceil(np.log(lengths) / np.log(4) - 1e-9).astype(np.int64)
    )


def _run_batched(
    kernel, row, slot, lengths, fields, fills, b, n_slot_outputs, **static
):
    """Run ``kernel(*padded fields, valid, b, **static) -> (ok, rhs,
    *per_slot)`` over rows given as flat entries (``row``, ``slot``), in
    chunks of ``_SEPARATION_CHUNK`` rows (fewer for wide rows, at most
    ``_SEPARATION_ENTRIES`` entries a chunk) padded to a width from
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
    for width in np.unique(width_of):
        width = int(width)
        # Widths are powers of two, so this is one too (and one shape a width).
        chunk = max(1, min(_SEPARATION_CHUNK, _SEPARATION_ENTRIES // width))
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
                    kernel(
                        *(array[part] for array in padded),
                        valid[part],
                        b_group[part],
                        **static,
                    )
                )
            stacked = [
                np.concatenate([np.asarray(res[k]) for res in results])
                for k in range(len(results[0]))
            ]
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
        return (
            has | take,
            jnp.where(take, eff, best),
            jnp.where(take, delta, best_delta),
        )

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
        integer
        & jnp.isfinite(lo)
        & jnp.isfinite(up)
        & (xv > lo + 1e-6)
        & (xv < up - 1e-6)
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
    and ``z``, ``c``, ``d`` arrays of shape ``(2, n)``, row 0 for a negative
    coefficient on the column and row 1 for a positive one (they differ
    only where bounds tie, ``_nearest_variable_bound``)."""
    n = x.size
    kind_of = np.zeros((2, n), np.int64)
    z_of = np.zeros((2, n), np.int64)
    c_of, d_of = np.zeros((2, n)), np.zeros((2, n))
    for y, kinds in bounds.items():
        for sign in (0, 1):  # the row's coefficient on y: < 0, > 0
            best = _nearest_variable_bound(kinds, y, x, lower, upper, sign == 1)
            if best is not None:
                kind, z, c, d = best
                kind_of[sign, y] = 1 if kind == "ub" else 2
                z_of[sign, y], c_of[sign, y], d_of[sign, y] = z, c, d
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
    sign = (a > 0).astype(np.int64)  # the tables' row for each term
    sub = (kind_of[sign, cols] > 0) & ~is_int[cols]
    if not sub.any():
        return None
    rows = np.unique(row[sub])
    take = np.isin(row, rows)
    row, slot, cols, a, sub = row[take], slot[take], cols[take], a[take], sub[take]
    sign = sign[take]
    kept, s_idx = ~sub, np.flatnonzero(sub)
    s_sign = sign[s_idx]
    y, z = cols[s_idx], z_of[s_sign, cols[s_idx]]
    c, d = c_of[s_sign, cols[s_idx]], d_of[s_sign, cols[s_idx]]
    kind = kind_of[s_sign, cols[s_idx]]
    a_y = a[s_idx]
    # Terms over the variables: the kept ones and each substituted y's z.
    v_row = np.concatenate([row[kept], row[s_idx]])
    v_col = np.concatenate([cols[kept], z])
    v_a = np.concatenate([a[kept], a_y * c])
    v_k1 = np.concatenate(
        [np.zeros(kept.sum(), np.int64), np.ones(s_idx.size, np.int64)]
    )
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
    scale = np.maximum(
        np.maximum.reduceat(
            np.abs(e_a), np.flatnonzero(np.r_[True, e_row[1:] != e_row[:-1]])
        ),
        1.0,
    )
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
        rows=rows,
        local=local,
        lengths=lengths,
        b=b_new,
        a=e_a,
        col=e_col,
        t=e_t,
        lo=np.where(real, lower[safe], 0.0),
        up=np.where(real, upper[safe], np.inf),
        x=np.where(real, x[safe], t_value[t_safe]),
        integer=real & is_int[safe],
        t_kind=kind,
        t_y=y,
        t_z=z,
        t_c=c,
        t_d=d,
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
        batch = _substituted_batch(
            row, slot, cols, a, b, x, lower, upper, is_int, tables
        )
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
        _cmir_kernel,
        all_row,
        all_slot,
        all_lengths,
        fields,
        [0.0, 0.0, 0.0, 0.0, False],
        all_b,
        1,
        max_divisors=max_divisors,
    )
    ok, rhs, coef = ok_all[:count], rhs_all[:count], coef_all[: row.size]
    for i in np.flatnonzero(ok):
        cuts[i] = (
            bases[i][0],
            coef[starts[i] : starts[i] + lengths[i]].copy(),
            float(rhs[i]),
        )
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
    rhs_t = s_rhs - np.bincount(
        local[is_t], sign * g * batch["t_d"][ti], minlength=s_rhs.size
    )
    found = _cuts_from_entries(
        np.arange(s_rhs.size),
        np.concatenate([local[real], local[is_t], local[is_t]]),
        np.concatenate([col[real], batch["t_z"][ti], batch["t_y"][ti]]),
        np.concatenate([s_coef[real], sign * g * batch["t_c"][ti], -sign * g]),
        rhs_t,
        n,
        relative=True,
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
def _flow_kernel(
    sign, u, fval, wval, f_mult, f0, w_col, x_f, x_w, f_col, live, valid, b
):
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
    prefix = jnp.cumsum(
        jnp.take_along_axis(jnp.where(plus, u0, 0.0), order, axis=1), axis=1
    )

    # The norm of a cut over merged columns: entries f (f_col) and w (w_col).
    cols = jnp.concatenate(
        [jnp.where(valid, f_col, -1), jnp.where(valid, w_col, -1)], axis=1
    )
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
            b
            + jnp.sum(jnp.where(in_minus, u0, 0.0), axis=1)
            - jnp.sum(excess, axis=1)
            - jnp.sum(cf * f0, axis=1)
            - jnp.sum(jnp.where(w_col < 0, cw, 0.0), axis=1)
        )
        values = jnp.concatenate([cf * f_mult, jnp.where(w_col >= 0, cw, 0.0)], axis=1)
        merged = jax.ops.segment_sum(
            jnp.where(
                real_col, jnp.take_along_axis(values, col_order, axis=1), 0.0
            ).ravel(),
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
        jnp.zeros(rows, bool),
        jnp.full(rows, 1e-6),
        jnp.zeros_like(u0),
        jnp.zeros_like(u0),
        jnp.zeros(rows),
    )
    for in_minus in (jnp.zeros_like(valid), saturated):
        need = b + jnp.sum(jnp.where(in_minus, u0, 0.0), axis=1)
        crossed = (prefix > (need + 1e-9)[:, None]) & (slots[None, :] < n_plus[:, None])
        size = jnp.where(
            jnp.any(crossed, axis=1), jnp.argmax(crossed, axis=1) + 1, n_plus
        )
        total = jnp.where(
            size > 0,
            jnp.take_along_axis(prefix, jnp.maximum(size - 1, 0)[:, None], axis=1)[
                :, 0
            ],
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
        fval = np.where(
            use_up, mag * (up - xv), np.where(use_lo, mag * (xv - lo), mag * xv)
        )
    bound = np.where(use_up, up, np.where(use_lo, lo, 0.0))
    shift = np.where(live & (use_up | use_lo), a * bound, 0.0)
    b_flow = b - np.bincount(row, shift, minlength=count)
    f_mult = np.where(use_up, -mag, mag)
    ok, rhs, (cf, cw) = _run_batched(
        _flow_kernel,
        row,
        slot,
        lengths,
        [
            np.where(use_up, -sgn, sgn),
            u,
            fval,
            np.where(vub, x[np.maximum(w_col, 0)], np.where(binary, xv, 1.0)),
            f_mult,
            np.where(use_up, mag * up, np.where(use_lo, -mag * lo, 0.0)),
            w_col,
            xv,
            np.where(w_col >= 0, x[np.maximum(w_col, 0)], 0.0),
            col,
            live,
        ],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1, 0.0, 0.0, -1, False],
        b_flow,
        2,
    )
    ok &= ~free
    live_f = ok[row] & (cf != 0)
    live_w = ok[row] & (cw != 0) & (w_col >= 0)
    found = _cuts_from_entries(
        np.arange(count),
        np.concatenate([row[live_f], row[live_w]]),
        np.concatenate([col[live_f], w_col[live_w]]),
        np.concatenate([cf[live_f] * f_mult[live_f], cw[live_w]]),
        rhs,
        x.size,
        relative=False,
    )
    for r, cut in found.items():
        out[r] = cut
    return out


# Tableau rows computed together by ``gomory_cuts``.
_GOMORY_BLOCK = 64


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
                (
                    np.ones(basic_rows.size),
                    (basic_rows, np.arange(basic_rows.size)),
                ),
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

    # The tableau rows of a block of candidates at once: one solve with the
    # factors and one product with Aᵀ for the block, instead of one each
    # per row. On bab5's root, with the cut rows in the LP (1.4 million
    # nonzeros), the 400 products took 0.75 s of a 1.6 s call.
    selected = candidates[:max_cuts]
    integer_col = integrality & ~col_basic
    sign = np.where(at_upper, -1.0, 1.0)
    for first in range(0, len(selected), _GOMORY_BLOCK):
        block = selected[first : first + _GOMORY_BLOCK]
        q = len(block)
        P = np.array([p for _, p, _ in block])
        K = np.array([k for _, _, k in block])
        E = np.zeros((m, q))
        E[P, np.arange(q)] = 1.0
        U = lu.solve(E, trans="T").reshape(m, q)
        ABAR = AT @ U  # tableau entries of the columns, one column a row
        expected = np.zeros((basic_cols.size, q))
        expected[P, np.arange(q)] = 1.0
        reliable = np.ones(q, bool)
        if basic_cols.size:
            reliable &= np.abs(ABAR[basic_cols] - expected).max(axis=0) <= 1e-7
        nonbasic = ~col_basic[:, None] & (np.abs(ABAR) > 0)
        reliable &= ~np.any(free[:, None] & nonbasic, axis=0)
        F0 = frac[K]
        # Shifted coefficients a'ⱼ of tⱼ: t = x − l (a' = ā) or u − x (a' = −ā).
        A_COL = np.where(at_upper[:, None], -ABAR, ABAR)
        A_COL[col_basic] = 0.0
        FJ = A_COL - np.floor(A_COL)
        PI_COL = np.where(
            integer_col[:, None],
            np.where(FJ <= F0, FJ / F0, (1.0 - FJ) / (1.0 - F0)),
            np.where(A_COL >= 0, A_COL / F0, -A_COL / (1.0 - F0)),
        )
        PI_COL[col_basic] = 0.0
        A_SLACK = np.where(nonbasic_slack[:, None], U, 0.0)
        PI_SLACK = np.where(A_SLACK >= 0, A_SLACK / F0, -A_SLACK / (1.0 - F0))
        PI_SLACK[~nonbasic_slack] = 0.0
        # Σ π t >= 1 in x: t = ±(x − bound), s = b − A x.
        SIGNED = PI_COL * sign[:, None]
        COEF = SIGNED - AT @ PI_SLACK
        CONSTANT = -(SIGNED.T @ bound) + PI_SLACK.T @ b
        for j in range(q):
            if not reliable[j]:
                continue
            abar, u = ABAR[:, j], U[:, j]
            coef_ge = COEF[:, j]
            rhs_ge = 1.0 - CONSTANT[j]
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
                        row_idx,
                        sense * row[row_idx],
                        sense * (u @ b),
                        ext_x,
                        ext_lower,
                        ext_upper,
                        ext_int,
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


# Clique cuts (``ConflictGraph``): conflicts a row may add beyond the clique
# it stores, literals that start a clique search per round, and adjacency
# checks spent lifting one clique.
_CLIQUE_MAX_EXTRA_EDGES = 10_000
# Conflicts from probing (``ConflictGraph.add_implications``) kept at most:
# on presolved mzzv11 probing implies about a million pairs of binaries.
_CLIQUE_MAX_PROBING_EDGES = 2_000_000
_CLIQUE_MAX_STARTS = 2000
_CLIQUE_LIFT_CHECKS = 20_000
_CLIQUE_ROUND_CHECKS = 1_000_000  # and in one round, over all its cliques


class ConflictGraph:
    """
    Conflicts between binary literals: pairs that are never both 1 in a
    point of the rows within the box ``lower <= x <= upper``. Literal ``2j``
    is ``x_j`` and ``2j + 1`` is ``1 − x_j``, for the binaries of the box.

    A row ``a·x <= b`` of any of ``mips`` (an equality row in both senses)
    puts two literals in conflict when raising both from the row's least
    activity over the box (``x_j`` to 1 when ``a_j > 0``, to 0 when
    ``a_j < 0``) passes ``b``: ``|a_p| + |a_q| > b − minact``. With the
    terms sorted by ``|a_j|``, the longest prefix whose last two terms
    conflict is a clique (every pair in it conflicts) and is stored as one;
    the other conflicts, of a prefix term with a later one, are stored as
    edges, at most ``_CLIQUE_MAX_EXTRA_EDGES`` per row (leaving conflicts
    out only weakens the cuts). Rows with an unbounded least activity give
    none.

    ``cuts(x)`` separates clique cuts ``Σ_{l in C} l <= 1``. ``num_edges``
    counts the stored conflicts, a pair once for each row that gives it.
    """

    def __init__(self, mips, lower, upper, tol=1e-6, integrality=None):
        # ``integrality``: the columns that count as integers (default each
        # MIP's own), as the search's cut separation counts them.
        lower = np.asarray(lower, np.float64)
        upper = np.asarray(upper, np.float64)
        self.n = lower.size
        cliques = {}  # sorted literal tuple -> None, in insertion order
        extra = collections.defaultdict(set)
        for mip in mips:
            self._add_rows(mip, lower, upper, tol, cliques, extra, integrality)
        self.cliques = [np.array(c, np.int64) for c in cliques]
        literal_cliques = collections.defaultdict(set)
        for k, clique in enumerate(cliques):
            for literal in clique:
                literal_cliques[literal].add(k)
        self.literal_cliques = dict(literal_cliques)
        self.extra = dict(extra)
        self.num_edges = (
            sum(len(c) * (len(c) - 1) // 2 for c in cliques)
            + sum(len(e) for e in extra.values()) // 2
        )

    def __len__(self):
        return len(self.cliques)

    def add_implications(self, cache, lower, upper, integrality, max_edges=None):
        """
        Conflicts from probing (``ProbingCache``): when fixing binary ``z``
        to ``d`` forces binary ``y`` to ``w``, literal ``z = d`` conflicts
        with literal ``y = 1 − w``. These need not show in any single row:
        propagation finds them through chains of rows, so clique cuts can
        grow across rows (HiGHS and SCIP fill their clique tables this way).
        ``lower``/``upper`` are the graph's box and ``integrality`` its
        integers; probing's implications hold over a box at least as wide.
        Conflicts the rows' cliques already hold are skipped, and at most
        ``max_edges`` new ones are kept (``None``:
        ``_CLIQUE_MAX_PROBING_EDGES``), in probing order. Returns the number
        added.
        """
        lower = np.asarray(lower, np.float64)
        upper = np.asarray(upper, np.float64)
        is_binary = np.asarray(integrality, bool) & (lower == 0.0) & (upper == 1.0)
        max_edges = _CLIQUE_MAX_PROBING_EDGES if max_edges is None else int(max_edges)
        first, second = [], []
        for z, sides in cache.sides.items():
            if cache.split[z] != 0.0 or not is_binary[z]:
                continue
            for d, side in enumerate(sides):
                if side is None:
                    continue
                idx, lo, up = side
                binary = is_binary[idx] & (idx != z)
                to_zero = binary & (up <= 0.0)
                to_one = binary & (lo >= 1.0)
                if not (to_zero.any() or to_one.any()):
                    continue
                own = 2 * z + (0 if d == 1 else 1)  # literal z = d
                # y forced to 0 conflicts with y = 1 (literal 2y); forced to
                # 1, with 1 − y (literal 2y + 1).
                other = np.concatenate([2 * idx[to_zero], 2 * idx[to_one] + 1])
                first.append(np.full(other.size, own, np.int64))
                second.append(other.astype(np.int64))
        if not first:
            return 0
        u, v = np.concatenate(first), np.concatenate(second)
        u, v = np.minimum(u, v), np.maximum(u, v)
        key = u * (2 * self.n) + v
        _, keep = np.unique(key, return_index=True)
        keep.sort()  # probing order
        u, v = u[keep], v[keep]
        if self.cliques:
            # Pairs a row's clique already holds.
            members = [
                (literal, k)
                for literal, ks in self.literal_cliques.items()
                for k in ks
            ]
            lit, clq = np.array(members, np.int64).T
            M = sp.csr_matrix(
                (np.ones(lit.size), (lit, clq)), shape=(2 * self.n, len(self.cliques))
            )
            covered = np.asarray(M[u].multiply(M[v]).sum(axis=1)).ravel() > 0
            u, v = u[~covered], v[~covered]
        added = 0
        for a, b in zip(u.tolist(), v.tolist()):
            if added >= max_edges:
                break
            neighbours = self.extra.setdefault(a, set())
            if b in neighbours:
                continue
            neighbours.add(b)
            self.extra.setdefault(b, set()).add(a)
            added += 1
        self.num_edges += added
        return added

    @staticmethod
    def _add_rows(mip, lower, upper, tol, cliques, extra, integrality=None):
        A = sp.csr_matrix(mip.A)
        b = np.asarray(mip.b, np.float64)
        m = A.shape[0]
        if m == 0 or A.nnz == 0:
            return
        ints = mip.integrality if integrality is None else integrality
        is_binary = np.asarray(ints, bool) & (lower == 0.0) & (upper == 1.0)
        rows = np.repeat(np.arange(m), np.diff(A.indptr))
        a, cols = A.data, A.indices
        with np.errstate(invalid="ignore"):
            low = np.where(a > 0, a * lower[cols], a * upper[cols])
            high = np.where(a > 0, a * upper[cols], a * lower[cols])
        low[a == 0] = 0.0
        high[a == 0] = 0.0
        min_act = np.bincount(rows, low, minlength=m)
        max_act = np.bincount(rows, high, minlength=m)
        # The largest binary |a_j| of each row screens out rows with no
        # conflict: two terms can pass the slack only if twice it does.
        biggest = np.zeros(m)
        np.maximum.at(biggest, rows, np.where(is_binary[cols], np.abs(a), 0.0))
        senses = [(1.0, b - min_act, b)]
        if mip.n_eq:
            eq = np.arange(m) < mip.n_eq
            senses.append((-1.0, np.where(eq, max_act - b, np.inf), -b))
        for sign, slack, rhs in senses:
            threshold = slack + tol * (1.0 + np.abs(rhs))
            with np.errstate(invalid="ignore"):
                live = (
                    np.isfinite(slack) & (slack >= -tol) & (2.0 * biggest > threshold)
                )
            for r in np.flatnonzero(live):
                start, end = A.indptr[r], A.indptr[r + 1]
                idx, coef = cols[start:end], sign * a[start:end]
                keep = is_binary[idx] & (coef != 0.0)
                idx, coef = idx[keep], coef[keep]
                cost = np.abs(coef)
                literal = np.where(coef > 0, 2 * idx, 2 * idx + 1)
                order = np.argsort(-cost, kind="stable")
                cost, literal = cost[order], literal[order]
                limit = threshold[r]
                k = 1 + int(np.count_nonzero(cost[:-1] + cost[1:] > limit))
                if k < 2:
                    continue
                cliques.setdefault(tuple(sorted(literal[:k].tolist())), None)
                budget = _CLIQUE_MAX_EXTRA_EDGES
                for p in range(k):
                    if k >= cost.size or cost[p] + cost[k] <= limit or budget <= 0:
                        break
                    later = k + int(np.count_nonzero(cost[k:] > limit - cost[p]))
                    for q in literal[k:later][:budget].tolist():
                        extra[int(literal[p])].add(q)
                        extra[q].add(int(literal[p]))
                    budget -= later - k

    def adjacent(self, u, v):
        """Whether literals ``u`` and ``v`` conflict."""
        if v in self.extra.get(u, ()):
            return True
        cu, cv = self.literal_cliques.get(u), self.literal_cliques.get(v)
        return cu is not None and cv is not None and not cu.isdisjoint(cv)

    def neighbours(self, u):
        """The literals in conflict with ``u``."""
        out = set(self.extra.get(u, ()))
        for k in self.literal_cliques.get(u, ()):
            out.update(self.cliques[k].tolist())
        out.discard(u)
        return out

    def cuts(self, x, max_cuts=200, min_violation=1e-6):
        """Clique cuts violated by ``x``, as ``("clique", idx, coef, rhs)``.
        From each literal with a positive value (largest first, at most
        ``_CLIQUE_MAX_STARTS``), a clique grows greedily through its
        neighbours with positive values, largest first; a violated one is
        then lifted with neighbours at zero (at most
        ``_CLIQUE_LIFT_CHECKS`` adjacency checks, ``_CLIQUE_ROUND_CHECKS``
        over the call), which keeps the cut violated and makes it stronger.
        A clique grown before from another start is not lifted again."""
        x = np.asarray(x, np.float64)
        value = np.empty(2 * self.n)
        value[0::2] = x
        value[1::2] = 1.0 - x
        in_graph = np.zeros(2 * self.n, bool)
        in_graph[list(self.literal_cliques)] = True
        in_graph[list(self.extra)] = True
        starts = np.flatnonzero(in_graph & (value > 1e-9))
        starts = starts[np.argsort(-value[starts], kind="stable")][:_CLIQUE_MAX_STARTS]
        seen, cuts = set(), []
        round_checks = _CLIQUE_ROUND_CHECKS
        cache = {}

        def neighbours_of(u):
            if u not in cache:
                cache[u] = self.neighbours(u)
            return cache[u]

        for s in starts.tolist():
            neighbours = neighbours_of(s)
            positive = sorted(
                (l for l in neighbours if value[l] > 1e-9), key=lambda l: (-value[l], l)
            )
            clique = [s]
            # The literals in conflict with every member so far: a literal
            # joins when it is one of them. Checking each candidate against
            # each member instead took 11–21 million checks a call on
            # 30n20b8, whose short PDHG node solves leave 1,400–2,600
            # binaries just off zero.
            common = set(neighbours)
            for l in positive:
                if l in common:
                    clique.append(l)
                    common &= neighbours_of(l)
            if value[clique].sum() <= 1.0 + min_violation:
                continue
            key = frozenset(clique)
            if key in seen:
                continue
            seen.add(key)
            checks = min(_CLIQUE_LIFT_CHECKS, round_checks)
            for l in sorted(neighbours - key):
                if checks <= 0:
                    break
                if value[l] > 1e-9:
                    continue
                checks -= len(clique)
                if l in common:
                    clique.append(l)
                    common &= neighbours_of(l)
            round_checks -= min(_CLIQUE_LIFT_CHECKS, round_checks) - max(checks, 0)
            literals = np.array(sorted(clique), np.int64)
            idx = literals // 2
            coef = np.where(literals % 2 == 0, 1.0, -1.0)
            rhs = 1.0 - float(np.count_nonzero(literals % 2))
            cuts.append(("clique", idx, coef, rhs))
            if len(cuts) >= max_cuts:
                break
        return cuts


class ImpliedBounds:
    """
    Implied-bound cuts from probing (``ProbingCache``). When fixing a binary
    ``z`` to 0 implies ``x_j <= u0`` and fixing it to 1 implies
    ``x_j <= u1`` (each no looser than ``upper[j]``), every feasible point
    has ``x_j <= u0 + (u1 − u0)·z``; likewise ``x_j >= l0 + (l1 − l0)·z``
    for lower bounds. The LP sees neither side alone, so a fractional ``z``
    can violate them: on facility location, ``y_ij <= x_j`` style links
    that the rows state only in aggregate.

    ``lower``/``upper`` are bounds that hold for every feasible point (the
    root box); they only tighten later, so the cuts stay valid. Only
    binaries probed with both sides feasible give cuts.
    """

    def __init__(self, cache, lower, upper, integrality):
        lower, upper = np.asarray(lower, np.float64), np.asarray(upper, np.float64)
        is_int = np.asarray(integrality, bool)
        z_all, j_all, a_all, b_all, sense_all = [], [], [], [], []
        for z, (down, up) in cache.sides.items():
            if down is None or up is None or cache.split[z] != 0.0:
                continue
            if not (is_int[z] and lower[z] == 0.0 and upper[z] == 1.0):
                continue
            # Each column changed on either side, with both sides' bounds
            # (the box's where a side leaves it alone).
            cols = np.union1d(down[0], up[0])
            cols = cols[cols != z]
            if cols.size == 0:
                continue
            bounds = []
            for idx, lo, hi in (down, up):
                side_lo, side_hi = lower[cols].copy(), upper[cols].copy()
                where = np.searchsorted(cols, idx)
                keep = (where < cols.size) & (
                    cols[np.minimum(where, cols.size - 1)] == idx
                )
                np.maximum.at(side_lo, where[keep], np.asarray(lo, np.float64)[keep])
                np.minimum.at(side_hi, where[keep], np.asarray(hi, np.float64)[keep])
                bounds.append((side_lo, side_hi))
            (l0, u0), (l1, u1) = bounds
            # Upper: x_j + (u0 − u1)·z <= u0. Lower: −x_j + (l1 − l0)·z <= −l0.
            for sense, b0, b1 in ((1.0, u0, u1), (-1.0, l0, l1)):
                ok = np.isfinite(b0) & np.isfinite(b1) & (np.abs(b1 - b0) > 1e-9)
                z_all.append(np.full(int(ok.sum()), z, np.int64))
                j_all.append(cols[ok])
                a_all.append(sense * (b0[ok] - b1[ok]))
                b_all.append(sense * b0[ok])
                sense_all.append(np.full(int(ok.sum()), sense))
        cat = lambda parts, dtype: (
            np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype)
        )
        self.z = cat(z_all, np.int64)
        self.j = cat(j_all, np.int64)
        self.coef_z = cat(a_all, np.float64)  # z's coefficient
        self.rhs = cat(b_all, np.float64)
        self.sense = cat(sense_all, np.float64)  # x_j's coefficient
        self.is_int = is_int
        self._variable_bounds = None  # ``variable_bounds()``, once built

    def __len__(self):
        return int(self.z.size)

    def variable_bounds(self):
        """The implied bounds of continuous columns in ``variable_bounds``'s
        form, ``{x_j: {"ub": [(z, c, d), ...], "lb": [...]}}`` for
        ``x_j <= d + c·z`` and ``x_j >= d + c·z``, so c-MIR and flow cover
        separation can shift ``x_j`` to them as to the rows' own variable
        bounds. On map06 the rows bound a continuous ``x`` by its binary
        only through chains of continuous rows; probing finds ``x <= z``.
        Built once; callers must not change it."""
        if self._variable_bounds is not None:
            return self._variable_bounds
        out = self._variable_bounds = {}
        for z, j, coef_z, rhs, sense in zip(
            self.z.tolist(),
            self.j.tolist(),
            self.coef_z.tolist(),
            self.rhs.tolist(),
            self.sense.tolist(),
        ):
            if self.is_int[j]:
                continue
            bounds = out.setdefault(j, {"ub": [], "lb": []})
            if sense > 0:  # x_j + coef_z·z <= rhs
                bounds["ub"].append((z, -coef_z, rhs))
            else:  # −x_j + coef_z·z <= rhs
                bounds["lb"].append((z, coef_z, -rhs))
        return out

    def cuts(self, x, max_cuts=200, min_violation=1e-6):
        """The most efficacious implied-bound cuts violated by ``x``, as
        ``("impliedbound", idx, coef, rhs)``."""
        if not len(self):
            return []
        x = np.asarray(x, np.float64)
        activity = self.sense * x[self.j] + self.coef_z * x[self.z]
        violation = activity - self.rhs
        hit = np.flatnonzero(violation > min_violation * (1.0 + np.abs(self.rhs)))
        if hit.size == 0:
            return []
        efficacy = violation[hit] / np.sqrt(1.0 + self.coef_z[hit] ** 2)
        hit = hit[np.argsort(-efficacy, kind="stable")][:max_cuts]
        return [
            (
                "impliedbound",
                np.array([self.j[k], self.z[k]], np.int64),
                np.array([self.sense[k], self.coef_z[k]]),
                float(self.rhs[k]),
            )
            for k in hit
        ]


# Zero-half cuts (``zero_half_cuts``): the most rows (the tightest) in the
# search, and the 64-bit word operations its elimination may spend.
_ZERO_HALF_MAX_ROWS = 10_000
_ZERO_HALF_WORK = 200_000_000


def _bit_rows(rows, positions, n_rows, n_bits):
    """A packed bit matrix (``uint64`` words) with bit ``positions[i]`` of
    row ``rows[i]`` set; the pairs must be distinct."""
    bits = np.zeros((n_rows, max(1, -(-n_bits // 64))), np.uint64)
    positions = np.asarray(positions, np.uint64)
    np.bitwise_or.at(
        bits,
        (np.asarray(rows, np.int64), (positions >> np.uint64(6)).astype(np.int64)),
        np.uint64(1) << (positions & np.uint64(63)),
    )
    return bits


def _unpack_bits(bits, n_bits):
    return np.unpackbits(bits.view(np.uint8), axis=1, bitorder="little")[:, :n_bits]


def zero_half_cuts(A, b, n_eq, x, lower, upper, integrality, max_cuts=200, eps=1e-6):
    """
    Zero-half cuts ({0, ½}-Chvátal–Gomory cuts) violated by ``x``, from the
    rows ``A[:n_eq] x = b[:n_eq]``, ``A[n_eq:] x <= b[n_eq:]`` over the box
    ``lower <= x <= upper``. Returns ``("zerohalf", idx, coef, rhs)`` for
    ``coef · x[idx] <= rhs``, at most ``max_cuts``.

    Only rows with integer coefficients on integer variables with integral
    bounds take part (an integral right-hand side within 1e-6 is rounded:
    the row's activity is an integer). Each variable is measured from its
    nearer bound at ``x``, ``x̃_j = x_j − l_j`` or ``u_j − x_j``, so
    ``x̃ >= 0`` is integral and small at ``x``. Half the sum of a set ``S``
    of rows, ``½ Σ_S ã·x̃ <= ½ Σ_S b̃``, rounds down to the valid cut
    ``Σ_j ⌊ã_j/2⌋ x̃_j <= ⌊Σ_S b̃/2⌋``. When ``Σ_S b̃`` is odd, ``x``
    violates it by ``(1 − Σ_S s_r − Σ_{odd j} x̃_j) / 2``, with ``s_r`` the
    rows' slacks and the odd columns those with an odd coefficient in the
    sum. So a useful ``S`` has small slacks and an even sum on every column
    away from its bound.

    The search follows HiGHS's mod-k separator with k = 2: Gaussian
    elimination over GF(2) on the coefficient parities of the rows with
    slack below 1 (at most ``_ZERO_HALF_MAX_ROWS``, tightest first), over
    the columns with ``x̃_j > 0``, largest first, pivoting on the row with
    the least slack. Every combination left with an odd right-hand side and
    a total cost (slacks plus odd columns' ``x̃``) below 1 gives a cut. The
    elimination stops after ``_ZERO_HALF_WORK`` word operations.

    On bab5's root, with exact LPs, these cuts alone take the bound from
    −115,370 to −111,545 in 20 rounds (4 s); separated from the earlier
    cuts as well as the rows, they take the other families' 20-round bound
    from −111,209 to −107,410, past HiGHS's root bound (−107,454).
    """
    A = sp.csr_matrix(A, dtype=np.float64)
    A.sum_duplicates()
    m, n = A.shape
    if m == 0 or A.nnz == 0:
        return []
    x = np.asarray(x, np.float64)
    lower = np.asarray(lower, np.float64)
    upper = np.asarray(upper, np.float64)
    b = np.asarray(b, np.float64)
    integrality = np.asarray(integrality, bool)

    # Each variable from its nearer bound: x̃ = sign · (x − shift).
    with np.errstate(invalid="ignore"):
        flip = np.isfinite(upper) & (upper - x < x - lower)
    shift = np.where(flip, upper, lower)
    sign = np.where(flip, -1.0, 1.0)
    with np.errstate(invalid="ignore"):
        shift_ok = np.isfinite(shift) & (np.abs(shift - np.round(shift)) <= 1e-9)
    shift = np.where(shift_ok, np.round(shift), 0.0)
    x_tilde = np.maximum(sign * (x - shift), 0.0)

    rows = np.repeat(np.arange(m), np.diff(A.indptr))
    cols, a = A.indices, A.data
    bad = (np.abs(a - np.round(a)) > 1e-9) | ~integrality[cols] | ~shift_ok[cols]
    b_round = np.round(b)
    eligible = (
        (np.bincount(rows, bad, minlength=m) == 0)
        & (np.diff(A.indptr) > 0)
        & (np.abs(b - b_round) <= 1e-6)
    )
    slack = b_round - A @ x
    slack[:n_eq] = np.abs(slack[:n_eq])
    slack = np.maximum(slack, 0.0)
    eligible &= slack < 1.0 - eps
    chosen = np.flatnonzero(eligible)
    chosen = chosen[np.argsort(slack[chosen], kind="stable")][:_ZERO_HALF_MAX_ROWS]
    k = chosen.size
    if k == 0:
        return []
    sub = A[chosen]
    sub.data = np.round(sub.data)
    rhs_tilde = b_round[chosen] - sub @ shift  # integral
    parity = (np.round(rhs_tilde).astype(np.int64) % 2).astype(np.uint8)
    slack = slack[chosen]

    # Parities on the columns away from their bounds, largest x̃ first.
    sub_rows = np.repeat(np.arange(k), np.diff(sub.indptr))
    odd = (np.abs(sub.data).astype(np.int64) % 2 == 1) & (x_tilde[sub.indices] > eps)
    active = np.unique(sub.indices[odd])
    active = active[np.argsort(-x_tilde[active], kind="stable")]
    position = np.full(n, -1, np.int64)
    position[active] = np.arange(active.size)
    C = _bit_rows(sub_rows[odd], position[sub.indices[odd]], k, active.size)
    members = _bit_rows(np.arange(k), np.arange(k), k, k)
    R = parity.copy()
    pivoted = np.zeros(k, bool)
    work = 0
    for p in range(active.size):
        word, bit = p >> 6, np.uint64(1) << np.uint64(p & 63)
        has = (C[:, word] & bit) != 0
        free = np.flatnonzero(has & ~pivoted)
        if free.size == 0:
            continue
        pivot = free[np.argmin(slack[free])]
        pivoted[pivot] = True
        others = np.flatnonzero(has)
        others = others[others != pivot]
        if others.size:
            C[others] ^= C[pivot]
            R[others] ^= R[pivot]
            members[others] ^= members[pivot]
            work += others.size * (C.shape[1] + members.shape[1])
            if work > _ZERO_HALF_WORK:
                break

    odd_rhs = np.flatnonzero(R == 1)
    if odd_rhs.size == 0:
        return []
    in_set = _unpack_bits(members[odd_rhs], k).astype(bool)
    cost = in_set @ slack + _unpack_bits(C[odd_rhs], active.size) @ x_tilde[active]
    good = np.flatnonzero(cost < 1.0 - 1e-3)
    good = good[np.argsort(cost[good], kind="stable")]

    cuts, seen = [], set()
    for i in good:
        key = members[odd_rhs[i]].tobytes()
        if key in seen:
            continue
        seen.add(key)
        total = np.asarray(sub[in_set[i]].sum(axis=0)).ravel()
        idx = np.flatnonzero(total)
        b_tilde = round(rhs_tilde[in_set[i]].sum())
        if b_tilde % 2 == 0 or idx.size == 0:
            continue
        # ⌊ã/2⌋ x̃ <= ⌊b̃/2⌋, back in x.
        c = np.floor(total[idx] * sign[idx] / 2.0)
        keep = c != 0.0
        idx, c = idx[keep], c[keep]
        if idx.size == 0:
            continue
        coef = c * sign[idx]
        rhs = np.floor(b_tilde / 2.0) + coef @ shift[idx]
        cleaned = _clean_cut(idx, coef, float(rhs), lower, upper)
        if cleaned is not None:
            cuts.append(("zerohalf", *cleaned))
            if len(cuts) >= max_cuts:
                break
    return cuts


class CutSeparator:
    """Cover and c-MIR separation from the rows of a ``MIP`` over bounds that
    hold for every feasible point (the root's, after propagation)."""

    def __init__(
        self,
        mip,
        lower,
        upper,
        aggregate_integers=False,
        max_aggregation=_CUT_MAX_AGGREGATION,
        flow_covers=True,
        backend="auto",
        min_batch=_SEPARATION_MIN_BATCH,
        conflict_graph=None,
        integrality=None,
        implied_bounds=None,
        implied_variable_bounds=True,
        device=None,
        aggregation_width=1,
        cmir_per_row=1,
        mir_cuts=True,
    ):
        # The JAX device the batched searches run on (``None``: the default
        # one where the call is made).
        self.device = device
        # ``aggregation_width``: the first aggregation step of a row tries
        # the partners of its ``aggregation_width`` best columns to cancel,
        # each continuing greedily (``_aggregation_path``). ``cmir_per_row``:
        # the c-MIR cuts kept per row and sense, the most efficacious over
        # its aggregates (``_path_cuts``).
        self.aggregation_width = max(1, int(aggregation_width))
        self.cmir_per_row = max(1, int(cmir_per_row))
        # c-MIR cuts at all (``mir_cuts``): without them the aggregation
        # paths still give their flow cover and cover cuts.
        self.mir_cuts = bool(mir_cuts)
        self.flow_covers = flow_covers  # also flow cover cuts (``flow_cover_cut``)
        # Also implied-bound cuts from this ``ImpliedBounds`` (built over
        # bounds no looser than ``lower``/``upper``), and with
        # ``implied_variable_bounds`` its bounds on continuous columns as
        # variable bounds for c-MIR and flow cover cuts.
        self.implied_bounds = implied_bounds
        # Also clique cuts from this ``ConflictGraph`` (built over bounds no
        # looser than ``lower``/``upper``).
        self.conflict_graph = conflict_graph
        # ``backend``: the whole round compiled with numba
        # (``jaddle._separate_numba``), batched JAX searches, or the per-row
        # numpy ones (the reference); ``"auto"`` is numba when it is
        # installed, else JAX. Under JAX, rounds with fewer than
        # ``min_batch`` base inequalities use numpy, where compiling would
        # cost more than it saves.
        self.backend, self.min_batch = _separation_backend(backend), int(min_batch)
        self._numba_inputs = None
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
        # ``integrality``: the columns the cuts count as integers (default
        # the MIP's own; the search adds its implied integers).
        self.is_int = (
            mip.integrality if integrality is None else np.asarray(integrality, bool)
        )
        self.is_binary = self.is_int & (self.lower == 0.0) & (self.upper == 1.0)
        # Variable bounds for c-MIR (``None`` without any, as on a pure
        # integer MIP): the rows' own, then probing's.
        bounds = variable_bounds(mip)
        if implied_bounds is not None and implied_variable_bounds:
            for j, implied in implied_bounds.variable_bounds().items():
                if self.is_int[j]:
                    continue
                kinds = bounds.setdefault(j, {"ub": [], "lb": []})
                for kind in ("ub", "lb"):
                    kinds[kind] += [b for b in implied[kind] if self.is_int[b[0]]]
        self.variable_bounds = bounds or None

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
        (``batched_cmir_cuts``, ``batched_flow_cover_cuts``), on ``device``
        when one is given, and give the same cuts as the per-row numpy
        searches."""
        if self.device is None:
            return self._separate(x)
        with jax.default_device(self.device):
            return self._separate(x)

    def _separate(self, x):
        x = np.asarray(x, np.float64)
        if self.backend == "numba":
            return self._separate_numba(x)
        jobs = []  # per (row, sense): its covers and its aggregation path
        for r in self.tight_rows(x):
            idx, a = self.row(r)
            if idx.size == 0:
                continue
            for s in (1.0, -1.0) if self.is_eq[r] else (1.0,):
                covers = [
                    ("cover", *found)
                    for found in cover_cuts(
                        idx,
                        s * a,
                        s * self.b[r],
                        x,
                        self.lower,
                        self.upper,
                        self.is_binary,
                    )
                ]
                jobs.append((covers, self._aggregation_path(r, s, x)))
        bases = [base for _, path in jobs for base in path]
        if self.backend == "jax" and len(bases) >= self.min_batch:
            mir = (
                batched_cmir_cuts(
                    bases, x, self.lower, self.upper, self.is_int, self.variable_bounds
                )
                if self.mir_cuts
                else [None] * len(bases)
            )
            flows = [None] * len(bases)
            if self.flow_covers:
                which = [
                    i
                    for i, (idx, _, _) in enumerate(bases)
                    if not self.is_int[idx].all()
                ]
                found = batched_flow_cover_cuts(
                    [bases[i] for i in which],
                    x,
                    self.lower,
                    self.upper,
                    self.is_int,
                    self.variable_bounds,
                )
                for i, flow in zip(which, found):
                    flows[i] = flow
        else:
            mir = [
                (
                    cmir_cut(
                        idx,
                        a,
                        b,
                        x,
                        self.lower,
                        self.upper,
                        self.is_int,
                        variable_bounds=self.variable_bounds,
                    )
                    if self.mir_cuts
                    else None
                )
                for idx, a, b in bases
            ]
            flows = [
                (
                    flow_cover_cut(
                        idx,
                        a,
                        b,
                        x,
                        self.lower,
                        self.upper,
                        self.is_int,
                        self.variable_bounds,
                    )
                    if self.flow_covers and not self.is_int[idx].all()
                    else None
                )
                for idx, a, b in bases
            ]
        cuts = [] if self.conflict_graph is None else self.conflict_graph.cuts(x)
        if self.implied_bounds is not None:
            cuts += self.implied_bounds.cuts(x)
        position = 0
        for covers, path in jobs:
            cuts += covers
            cuts += self._path_cuts(
                path,
                x,
                mir[position : position + len(path)],
                flows[position : position + len(path)],
            )
            position += len(path)
        out = []
        for family, idx, coef, rhs in cuts:
            cleaned = _clean_cut(idx, coef, rhs, self.lower, self.upper)
            if cleaned is not None:
                out.append((family, *cleaned))
        return out

    def _separate_numba(self, x):
        """``_separate`` with the rows' cuts from one compiled call
        (``jaddle._separate_numba.separate``): the same cuts in the same
        order, up to rounding in the last bits."""
        if self._numba_inputs is None:
            self._numba_inputs = self._numba_arrays()
        arrays, vb = self._numba_inputs
        lower, upper, is_int, is_binary, A_ptr, A_idx, A_val, b, is_eq, C_ptr, C_row, C_val = arrays
        rows = self.tight_rows(x).astype(np.int64)
        cuts = [] if self.conflict_graph is None else self.conflict_graph.cuts(x)
        if self.implied_bounds is not None:
            cuts += self.implied_bounds.cuts(x)
        out = []
        for family, idx, coef, rhs in cuts:
            cleaned = _clean_cut(idx, coef, rhs, self.lower, self.upper)
            if cleaned is not None:
                out.append((family, *cleaned))
        family, ptr, idx, coef, rhs = _numba_separate().separate(
            rows, x, lower, upper, is_int, is_binary,
            A_ptr, A_idx, A_val, b, is_eq, C_ptr, C_row, C_val,
            _f64(self.relative), self.variable_bounds is not None, *vb,
            int(self.max_aggregation), bool(self.aggregate_integers),
            bool(self.flow_covers), 64, _COVER_ORDERS, _KNAPSACK_TOL, _CUT_TINY,
            # None of the c-MIR cuts kept without ``mir_cuts``.
            int(self.aggregation_width), int(self.cmir_per_row) if self.mir_cuts else 0,
        )
        names = ("cover", "mir", "flowcover")
        for k in range(family.size):
            start, end = ptr[k], ptr[k + 1]
            out.append(
                (names[family[k]], idx[start:end].copy(), coef[start:end].copy(), float(rhs[k]))
            )
        return out

    def _numba_arrays(self):
        """The separator's rows, columns, box and variable bounds as the
        arrays ``jaddle._separate_numba.separate`` takes."""
        n = self.A.shape[1]
        A, A_csc = self.A, self.A_csc
        arrays = (
            _f64(self.lower).copy(),
            _f64(self.upper).copy(),
            np.asarray(self.is_int, bool).copy(),
            np.asarray(self.is_binary, bool).copy(),
            A.indptr.astype(np.int64),
            A.indices.astype(np.int64),
            _f64(A.data).copy(),
            _f64(self.b).copy(),
            np.asarray(self.is_eq, bool),
            A_csc.indptr.astype(np.int64),
            A_csc.indices.astype(np.int64),
            _f64(A_csc.data).copy(),
        )
        # Variable bounds by column, upper ones first, as the dict lists them.
        counts = np.zeros(n + 1, np.int64)
        kind, z, c, d = [], [], [], []
        for y in range(n) if self.variable_bounds else ():
            bounds = self.variable_bounds.get(y)
            if not bounds:
                continue
            for k, name in enumerate(("ub", "lb")):
                for zb, cb, db in bounds[name]:
                    kind.append(k)
                    z.append(zb)
                    c.append(cb)
                    d.append(db)
                    counts[y + 1] += 1
        vb = (
            np.cumsum(counts),
            np.array(kind, np.int64),
            np.array(z, np.int64),
            np.array(c, np.float64),
            np.array(d, np.float64),
        )
        return arrays, vb

    def _path_cuts(self, path, x, mir, flows):
        """The cuts of one aggregation path from its levels' c-MIR and flow
        cover cuts: the ``cmir_per_row`` most efficacious c-MIR cuts over
        the levels (most efficacious first, ties in level order), each
        level's flow cover cut, and the cover cuts of the levels beyond the
        row."""
        mirs = []  # (efficacy, level, cut)
        extra = []
        for level, ((idx, a, b), cut, flow) in enumerate(zip(path, mir, flows)):
            if cut is not None:
                c_idx, c_coef, c_rhs = cut
                efficacy = (c_coef @ x[c_idx] - c_rhs) / np.linalg.norm(c_coef)
                if efficacy > 0.0:
                    mirs.append((efficacy, level, cut))
            if flow is not None:
                extra.append(("flowcover", *flow))
            if level > 0:
                extra += [
                    ("cover", *found)
                    for found in cover_cuts(
                        idx, a, b, x, self.lower, self.upper, self.is_binary
                    )
                ]
        mirs.sort(key=lambda item: (-item[0], item[1]))
        best = [("mir", *cut) for _, _, cut in mirs[: self.cmir_per_row]]
        return best + extra

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
                idx,
                a,
                b,
                x,
                self.lower,
                self.upper,
                self.is_int,
                variable_bounds=self.variable_bounds,
            )
            if self.mir_cuts
            else None
            for idx, a, b in path
        ]
        flows = [
            (
                flow_cover_cut(
                    idx,
                    a,
                    b,
                    x,
                    self.lower,
                    self.upper,
                    self.is_int,
                    self.variable_bounds,
                )
                if self.flow_covers and not self.is_int[idx].all()
                else None
            )
            for idx, a, b in path
        ]
        return self._path_cuts(path, x, mir, flows)

    def _aggregation_path(self, r, s, x):
        """Row ``r`` (times ``s``) and its aggregates with up to
        ``max_aggregation`` more rows, as ``[(idx, a, b), ...]``: each step
        adds the row that cancels the continuous variable furthest from its
        bounds (or, with ``aggregate_integers`` and none left, the integer
        term ``|aⱼ|·min(fⱼ, 1 − fⱼ)`` largest at ``x``), preferring the
        tightest rows (``_cancel_order``, ``_partner``).

        With ``aggregation_width`` above 1 the first step also cancels the
        next best columns, up to that many in all, each with its own partner
        row, and each of these branches continues greedily. The row comes
        first, then each branch's aggregates in turn; a branch that would
        add the same multiple of the same row as an earlier one is
        skipped."""
        idx, a = self.row(r)
        a, b = s * a, s * self.b[r]
        path = [(idx, a, b)]
        if self.max_aggregation <= 0 or idx.size == 0:
            return path
        starts = []
        for pos in self._cancel_order(idx, a, x)[: self.aggregation_width]:
            partner = self._partner(idx[pos], a[pos], {r})
            if partner is not None and partner not in [p for _, p in starts]:
                starts.append((idx[pos], partner))
        for col, (r2, lam) in starts:
            used = {r}
            level_idx, level_a, level_b = idx, a, b
            for level in range(self.max_aggregation):
                if level > 0:
                    order = self._cancel_order(level_idx, level_a, x)
                    if order.size == 0:
                        break
                    pos = order[0]
                    col = level_idx[pos]
                    partner = self._partner(col, level_a[pos], used)
                    if partner is None:
                        break
                    r2, lam = partner
                level_idx, level_a, level_b = self._add_row(
                    level_idx, level_a, level_b, col, r2, lam
                )
                used.add(r2)
                if level_idx.size == 0:
                    break
                path.append((level_idx, level_a, level_b))
        return path

    def _cancel_order(self, idx, a, x):
        """The positions in ``idx`` worth cancelling, best first: continuous
        columns by their distance from their nearer bound at ``x``; when no
        continuous one is more than 1e-6 from a bound, with
        ``aggregate_integers``, integer ones by ``|aⱼ|·min(fⱼ, 1 − fⱼ)``.
        Scores of 1e-6 or less are left out; ties keep the row's order."""
        continuous = ~self.is_int[idx]
        with np.errstate(invalid="ignore"):
            distance = np.minimum(x[idx] - self.lower[idx], self.upper[idx] - x[idx])
        score = np.where(continuous, np.nan_to_num(distance, posinf=1e30), -1.0)
        if score.max(initial=-1.0) <= 1e-6:
            if not self.aggregate_integers:
                return np.empty(0, np.int64)
            frac = x[idx] - np.floor(x[idx])
            score = np.where(
                self.is_int[idx], np.abs(a) * np.minimum(frac, 1.0 - frac), -1.0
            )
        order = np.argsort(-score, kind="stable")
        return order[score[order] > 1e-6]

    def _partner(self, col, coef, used):
        """The tightest row outside ``used`` that cancels the term
        ``coef·x[col]``, with its weight, as ``(row, weight)``; ``None``
        when none can (a <= row only takes a weight >= 0)."""
        start, end = self.A_csc.indptr[col], self.A_csc.indptr[col + 1]
        for r2, a2 in sorted(
            zip(self.A_csc.indices[start:end], self.A_csc.data[start:end]),
            key=lambda item: self.relative[item[0]],
        ):
            if r2 in used or a2 == 0:
                continue
            lam = -coef / a2
            if lam < 0 and not self.is_eq[r2]:
                continue  # a <= row can only be added with a weight >= 0
            return int(r2), float(lam)
        return None

    def _add_row(self, idx, a, b, col, r2, lam):
        """``a · x[idx] <= b`` plus ``lam`` times row ``r2``, with the
        cancelled column ``col`` and near-zero terms removed."""
        idx2, a2 = self.row(r2)
        merged_idx, inverse = np.unique(np.concatenate([idx, idx2]), return_inverse=True)
        merged = np.bincount(
            inverse, np.concatenate([a, lam * a2]), minlength=merged_idx.size
        )
        merged[merged_idx == col] = 0.0
        nonzero = np.abs(merged) > 1e-12 * np.abs(merged).max(initial=1.0)
        return merged_idx[nonzero], merged[nonzero], b + lam * self.b[r2]


def violated_cuts(cuts, x, min_efficacy=1e-4):
    """The ``cuts`` (``(family, idx, coef, rhs)``) that ``x`` violates by an
    efficacy (distance) of at least ``min_efficacy``, in the order given:
    ``select_cuts``'s candidates before its limit and parallelism test."""
    if not cuts:
        return []
    x = np.asarray(x, np.float64)
    lengths = np.array([cut[1].size for cut in cuts])
    k = len(cuts)
    rows = np.repeat(np.arange(k), lengths)
    cols = np.concatenate([np.asarray(cut[1], np.int64) for cut in cuts])
    vals = np.concatenate([np.asarray(cut[2], np.float64) for cut in cuts])
    rhs = np.array([cut[3] for cut in cuts], np.float64)
    norm = np.sqrt(np.bincount(rows, vals * vals, minlength=k))
    with np.errstate(divide="ignore", invalid="ignore"):
        efficacy = (np.bincount(rows, vals * x[cols], minlength=k) - rhs) / norm
    return [cuts[i] for i in np.flatnonzero(efficacy >= min_efficacy)]


def select_cuts(
    cuts,
    x,
    n,
    min_efficacy=1e-4,
    max_cuts=100,
    weights=None,
    objective=None,
    is_int=None,
    incumbent=None,
):
    """The cuts that ``x`` violates by an efficacy (distance) of at least
    ``min_efficacy``, best first, skipping near-duplicates, at most
    ``max_cuts`` of them.

    Without ``weights``, best is most efficacious. With ``weights =
    (w_obj, w_int, w_dir)``, as SCIP's hybrid cut selector, a cut ``a·x <=
    b`` scores

        efficacy / max efficacy
            + w_obj · |cos(a, objective)|
            + w_int · (share of its terms on ``is_int`` columns)
            + w_dir · (directed cutoff distance / its max),

    each term in [0, 1]. The directed cutoff distance is how far the cut
    lies from ``x`` along the segment to ``incumbent``, ``(a·x − b)·‖d‖ /
    (−a·d)`` with ``d = incumbent − x`` (at least the efficacy); it is the
    efficacy where ``a·d >= 0``. A term whose data is not given (or zero)
    is left out.

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
    score = efficacy
    if weights is not None and any(weights):
        score = _cut_scores(
            efficacy, keep, rows, cols, vals, norm, lengths, x, weights,
            objective, is_int, incumbent,
        )
    order = keep[np.argsort(-score[keep], kind="stable")]
    with np.errstate(divide="ignore", invalid="ignore"):
        directions = sp.csr_matrix((vals / norm[rows], (rows, cols)), shape=(k, n))
    return [cuts[i] for i in _spread_out(directions, order, max_cuts)]


def _cut_scores(
    efficacy, keep, rows, cols, vals, norm, lengths, x, weights, objective,
    is_int, incumbent,
):
    """``select_cuts``'s scores (meaningful at ``keep`` only)."""
    w_obj, w_int, w_dir = weights
    k = efficacy.size
    score = np.zeros(k)
    score[keep] = efficacy[keep] / efficacy[keep].max()
    if w_obj and objective is not None and np.shape(objective) == x.shape:
        objective = np.asarray(objective, np.float64)
        c_norm = np.linalg.norm(objective)
        if c_norm > 0:
            dot = np.bincount(rows, vals * objective[cols], minlength=k)
            score[keep] += w_obj * np.abs(dot[keep]) / (norm[keep] * c_norm)
    if w_int and is_int is not None and np.shape(is_int) == x.shape:
        integer = np.bincount(
            rows, np.asarray(is_int, bool)[cols].astype(np.float64), minlength=k
        )
        score[keep] += w_int * integer[keep] / np.maximum(lengths[keep], 1)
    if w_dir and incumbent is not None and np.shape(incumbent) == x.shape:
        d = np.asarray(incumbent, np.float64) - x
        length = np.linalg.norm(d)
        if length > 0:
            toward = np.bincount(rows, vals * d[cols], minlength=k)[keep]
            violation = efficacy[keep] * norm[keep]
            with np.errstate(divide="ignore", invalid="ignore"):
                directed = np.where(
                    toward < 0, violation * length / -toward, efficacy[keep]
                )
            directed = np.maximum(np.nan_to_num(directed), efficacy[keep])
            score[keep] += w_dir * directed / directed.max()
    return score


def _spread_out(directions, order, max_cuts, held=None):
    """The greedy pass of ``select_cuts``: the rows of ``directions`` (CSR,
    unit rows) taken in ``order``, skipping each whose |cosine| with one
    taken before it, or with a row of ``held`` (CSR, unit rows: cuts in the
    LP already), is over ``_CUT_MAX_PARALLELISM``; it stops at ``max_cuts``.
    Returns the rows taken.

    It works in blocks of the candidates: a chosen cut's cosines are taken
    with its block only, and a later block's with all the cuts chosen
    before it. Same cuts as one pass over all of them, without a product
    over every candidate per cut chosen (a third of a second a round on
    bab5, with about 1,000 candidates)."""
    block_size = max(4 * max_cuts, 256)
    chosen = []
    for start in range(0, order.size, block_size):
        block = order[start : start + block_size]
        D = directions[block]
        # The largest |cosine| of each candidate with a cut chosen so far.
        nearest = np.zeros(block.size)
        if chosen:
            nearest = np.abs((D @ directions[chosen].T).toarray()).max(axis=1)
        if held is not None and held.shape[0]:
            np.maximum(
                nearest, np.abs((D @ held.T).toarray()).max(axis=1), out=nearest
            )
        cosines = None  # the block's |cosines|, once a cut in it is chosen
        for j, i in enumerate(block):
            if nearest[j] > _CUT_MAX_PARALLELISM:
                continue
            chosen.append(i)
            if len(chosen) >= max_cuts:
                return chosen
            if cosines is None:
                cosines = np.abs(_gram(D))
            np.maximum(nearest, cosines[j], out=nearest)
    return chosen


# ``_gram``: dense products do this many times the work of sparse ones in
# the same time (on bab5's dense Gomory cuts, about 60); entries of a dense
# column chunk.
_GRAM_DENSE_SPEEDUP = 50
_GRAM_CHUNK_ENTRIES = 4_000_000


def _gram(D):
    """``D @ D.T`` (CSR ``D``) as a dense array: one sparse product, or,
    when the rows are dense enough that BLAS is faster, dense products over
    chunks of the columns ``D`` uses. Per chosen cut, a sparse product
    with the block walked all of the block's nonzeros: on bab5's root, with
    Gomory cuts on half of its 20,508 columns, 69 s over 20 rounds."""
    q = D.shape[0]
    counts = np.bincount(D.indices, minlength=D.shape[1]).astype(np.float64)
    used = np.flatnonzero(counts)
    sparse_work = counts @ counts
    dense_work = float(q) * q * used.size
    if dense_work > _GRAM_DENSE_SPEEDUP * sparse_work:
        return (D @ D.T).toarray()
    columns = D[:, used].tocsc()
    G = np.zeros((q, q))
    step = max(256, _GRAM_CHUNK_ENTRIES // max(q, 1))
    for first in range(0, used.size, step):
        part = columns[:, first : first + step].toarray()
        G += part @ part.T
    # A row with a NaN (a zero cut's direction) spreads it to every entry
    # here, through its zeros; the sparse product only to rows it overlaps.
    bad = np.flatnonzero(
        ~np.isfinite(
            np.bincount(
                np.repeat(np.arange(q), np.diff(D.indptr)), np.abs(D.data), minlength=q
            )
        )
    )
    if bad.size:
        exact = (D[bad] @ D.T).toarray()
        G[bad, :] = exact
        G[:, bad] = exact.T
    return G


def _next_power_of_two(k):
    return 1 << max(int(k) - 1, 0).bit_length()


# Exact root cut rounds (``exact_cut_loop``): the most rounds a cut stays in
# the LP slack with a zero dual (as HiGHS keeps them; ``cut_age``, tuned for
# PDHG's inexact duals, may allow more), and the share of ``max_seconds``
# the rounds may take.
_EXACT_CUT_AGE = 10
_EXACT_CUT_SHARE = 0.25


def _cut_rows(cuts, n):
    """The cuts (``cut_list`` dicts) as an unpadded CSR block ``A x <= b``."""
    if not cuts:
        return sp.csr_matrix((0, n)), np.zeros(0)
    lengths = [cut["idx"].size for cut in cuts]
    A = sp.csr_matrix(
        (
            np.concatenate([np.asarray(cut["coef"], np.float64) for cut in cuts]),
            np.concatenate([np.asarray(cut["idx"], np.int64) for cut in cuts]),
            np.concatenate([[0], np.cumsum(lengths)]),
        ),
        shape=(len(cuts), n),
    )
    return A, np.array([cut["rhs"] for cut in cuts], np.float64)


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
            (
                np.concatenate(indices).astype(np.int64)
                if indices
                else np.zeros(0, np.int64)
            ),
            indptr,
        ),
        shape=(rows, n),
    )
    b = np.zeros(rows)
    b[: len(cuts)] = [cut["rhs"] for cut in cuts]
    return A, b


# %%
# Symmetry
def _colours(keys):
    """Integer colours for a sequence of hashable keys, equal keys equal."""
    table = {}
    return [table.setdefault(key, len(table)) for key in keys]


def symmetry_generators(mip, lower=None, upper=None):
    """
    Generators of a group of column permutations that map ``mip`` (with
    the box ``[lower, upper]``, default its own bounds) to itself: costs,
    bounds, integrality and every row (up to permuting the rows) kept
    exactly. Found as the automorphisms of a coloured graph by bliss, through
    igraph (``pip install jaddle[symmetry]``): a vertex per column (coloured
    by cost, bounds and integrality), a vertex per row (by its bounds), and a
    vertex per distinct coefficient of each row (by its value), joined to
    its row and to the columns with that coefficient there. Each generator
    found is checked exactly against the rows before it is returned.
    Returns a list of permutations of the columns (``x → x[perm]`` keeps
    the MIP), without the identity.

    On bab5 the graph has 34,224 vertices, bliss takes 0.02 s, and 5
    generators come back, each swapping two blocks of 440 or 446 binaries.
    """
    try:
        import igraph
    except ImportError as error:
        raise ImportError(
            "symmetry detection needs igraph: pip install jaddle[symmetry]"
        ) from error
    A = sp.csr_matrix(mip.A)
    A.sort_indices()
    m, n = A.shape
    lower = mip.lower if lower is None else np.asarray(lower, np.float64)
    upper = mip.upper if upper is None else np.asarray(upper, np.float64)
    rows = np.repeat(np.arange(m), np.diff(A.indptr))
    values = A.data.astype(np.float64)
    # One vertex per distinct (row, coefficient).
    pairs, slot = np.unique(
        np.stack([rows, values.view(np.int64)]), axis=1, return_inverse=True
    )
    slot = slot.ravel()
    n_aux = pairs.shape[1]
    keys = (
        [("column", *key) for key in zip(mip.c, lower, upper, mip.integrality)]
        + [("row", *key) for key in zip(mip.row_lower, mip.row_upper)]
        + [("coefficient", value) for value in pairs[1].view(np.float64)]
    )
    edges = np.concatenate(
        [
            np.stack([n + pairs[0], n + m + np.arange(n_aux)], axis=1),
            np.stack([A.indices, n + m + slot], axis=1),
        ]
    )
    graph = igraph.Graph(n=n + m + n_aux, edges=edges.tolist())
    found = graph.automorphism_group(color=_colours(keys))
    out = []
    for perm in found:
        perm = np.asarray(perm, np.int64)
        columns, row_map = perm[:n], perm[n : n + m] - n
        if np.all(columns == np.arange(n)):
            continue
        # The exact check: row i, column j of A is row row_map[i], column
        # columns[j] of A again.
        image = sp.csr_matrix(
            (values, (row_map[rows], columns[A.indices])), shape=(m, n)
        )
        image.sort_indices()
        same = (
            image.nnz == A.nnz
            and np.array_equal(image.indptr, A.indptr)
            and np.array_equal(image.indices, A.indices)
            and np.array_equal(image.data, A.data)
        )
        kept = (
            np.array_equal(mip.c[columns], mip.c)
            and np.array_equal(lower[columns], lower)
            and np.array_equal(upper[columns], upper)
            and np.array_equal(mip.integrality[columns], mip.integrality)
            and np.array_equal(mip.row_lower[row_map], mip.row_lower)
            and np.array_equal(mip.row_upper[row_map], mip.row_upper)
        )
        if same and kept:
            out.append(columns)
    return out


def orbitopes(generators, n):
    """
    Full orbitopes among ``generators`` (column permutations of ``n``
    columns): matrices of columns (rows × p, an ``int`` array) whose p
    matrix columns the generators permute in every way, row by row. They
    are built from the generators that swap two blocks of columns
    elementwise (involutions): one gives two matrix columns; another that
    swaps a matrix column already found with a new block adds that block,
    its row ``r`` the image of the column's row ``r``. Each added swap joins
    the new column to the others, so the swaps generate every permutation
    of the matrix columns. Orbitopes act on disjoint columns, so breaking
    each separately is valid for all at once. Generators that fit no
    orbitope are left out (breaking a subgroup's symmetry is still valid).
    """
    ident = np.arange(n)
    swaps = []
    for perm in generators:
        perm = np.asarray(perm)
        if np.array_equal(perm[perm], ident) and np.any(perm != ident):
            swaps.append(perm)
    used = [False] * len(swaps)
    taken = np.zeros(n, bool)
    out = []
    for i, perm in enumerate(swaps):
        if used[i]:
            continue
        first = np.flatnonzero((perm != ident) & (ident < perm))
        if np.any(taken[first]) or np.any(taken[perm[first]]):
            used[i] = True
            continue
        columns = [first, perm[first]]
        used[i] = True
        grown = True
        while grown:
            grown = False
            members = np.concatenate(columns)
            for j, other in enumerate(swaps):
                if used[j]:
                    continue
                moved = other != ident
                if moved.sum() != 2 * first.size:
                    continue
                for column in columns:
                    image = other[column]
                    if (
                        np.all(moved[column])
                        and not np.any(np.isin(image, members))
                        and not np.any(taken[image])
                    ):
                        columns.append(image)
                        used[j] = grown = True
                        break
                if grown:
                    break
        matrix = np.stack(columns, axis=1)
        taken[matrix.ravel()] = True
        out.append(matrix)
    return out


def _packing_rows(mip, matrix, lower):
    """Whether each row of an orbitope ``matrix`` lies in a row of ``mip``
    that lets at most one of its entries be 1: binaries with coefficient 1
    there, every other coefficient nonnegative on a nonnegative variable,
    and a right-hand side below 2."""
    A = sp.csr_matrix(mip.A)
    At = sp.csc_matrix(A)
    for entries in matrix:
        common = None
        for j in entries:
            rows = set(At.indices[At.indptr[j] : At.indptr[j + 1]].tolist())
            common = rows if common is None else common & rows
        found = False
        for i in common or ():
            cols = A.indices[A.indptr[i] : A.indptr[i + 1]]
            coef = A.data[A.indptr[i] : A.indptr[i + 1]]
            here = dict(zip(cols.tolist(), coef.tolist()))
            if (
                all(here[j] == 1.0 for j in entries)
                and np.all(coef >= 0)
                and np.all(lower[cols] >= 0)
                and mip.row_upper[i] < 2.0
            ):
                found = True
                break
        if not found:
            return False
    return True


def symmetry_rows(mip, matrices, lower, upper):
    """
    Symmetry-breaking rows and fixings for orbitopes (``orbitopes``) of
    ``mip`` over the box ``[lower, upper]``, each valid for at least one
    optimal solution (and for all orbitopes together, as they act on
    disjoint columns). Returns ``(A_rows, b_rows, upper)``: rows
    ``A_rows x <= b_rows`` and the box's upper bounds with fixings.

    * A packing orbitope (binaries, each row at most one 1: ``_packing_rows``)
      has its columns put in lexicographic order, rows top down: its
      columns' first 1s then go down strictly, so entry ``(i, j)`` is 0 for
      ``j > i`` (a fixing), and an entry ``(i, j)`` can be 1 only if column
      ``j − 1`` has a 1 above row ``i``:
      ``x[i, j] − Σ_{k<i} x[k, j−1] <= 0``.
    * Any other orbitope has its columns in decreasing order of their sums:
      ``Σ_r x[r, j+1] − Σ_r x[r, j] <= 0`` (a column permutation sorts any
      solution's columns so, at the same cost). On bab5 that is one row for
      each of its 5 two-column orbitopes.
    """
    n = mip.num_variables
    upper = np.asarray(upper, np.float64).copy()
    data, indices, indptr, rhs = [], [], [0], []

    def add(cols, coefs):
        indices.append(np.asarray(cols, np.int64))
        data.append(np.asarray(coefs, np.float64))
        indptr.append(indptr[-1] + len(cols))
        rhs.append(0.0)

    binary = mip.integrality & (lower >= 0) & (upper <= 1)
    for matrix in matrices:
        rows, p = matrix.shape
        if np.all(binary[matrix]) and _packing_rows(mip, matrix, lower):
            for i in range(rows):
                for j in range(1, p):
                    if j > i:
                        upper[matrix[i, j]] = 0.0
                    elif i > 0:
                        add(
                            np.concatenate([[matrix[i, j]], matrix[:i, j - 1]]),
                            np.concatenate([[1.0], -np.ones(i)]),
                        )
        else:
            for j in range(p - 1):
                add(
                    np.concatenate([matrix[:, j + 1], matrix[:, j]]),
                    np.concatenate([np.ones(rows), -np.ones(rows)]),
                )
    if not rhs:
        return sp.csr_matrix((0, n)), np.zeros(0), upper
    A_rows = sp.csr_matrix(
        (np.concatenate(data), np.concatenate(indices), np.asarray(indptr)),
        shape=(len(rhs), n),
    )
    return A_rows, np.asarray(rhs), upper


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
        lo, up, empty = propagator(
            np.asarray(lo, np.float64), np.asarray(up, np.float64)
        )
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
                    # The data's rows: ``A``'s, plus a padding row when
                    # padded (``_propagation_data(nnz_capacity=...)``).
                    int(data["row_lower"].shape[0]),
                )
            )
        return [
            (lo[i], up[i], bool(empty[i]) or bool(np.any(lo[i] > up[i])))
            for i in range(len(boxes))
        ]

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
                    raised = (
                        np.clip(np.round(state["x"][free]), lo[free], up[free])
                        > lo[free]
                    )
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
                        cache.implied(j, v, new_lo, new_up)
                        for j, v in zip(bulk, values)
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
            states[p]["lower"], states[p]["upper"] = np.asarray(new_lo), np.asarray(
                new_up
            )
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
    lower, upper, is_int, tol = (
        data["lower"],
        data["upper"],
        data["is_int"],
        data["tol"],
    )
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
        random_in_row = jnp.argmax(
            jnp.where(in_row, jax.random.uniform(k_pick, (n,)), -1.0)
        )
        forced_var = jnp.where(
            jax.random.uniform(k_coin) < 0.01, random_in_row, best_in_row
        )
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


def _jump_rows(mip, tol, cutoff=None):
    """The rows the jump works on: the MIP's (as given, no cuts) scaled to
    unit ∞-norm, with a per-row tolerance matching ``MIP.violation``'s. With
    a ``cutoff``, the row ``c·x <= cutoff`` (its nonzero ``c`` entries) is
    appended last, so a point meets every row only when it also improves on
    the cutoff. Returns ``(A, row_lower, row_upper, row_tol, scale)``, all
    scaled, with ``A`` in CSR."""
    A = sp.csr_matrix(mip.A)
    row_lower, row_upper = mip.row_lower, mip.row_upper
    if cutoff is not None:
        A = sp.vstack(
            [A, sp.csr_matrix(np.asarray(mip.c, np.float64)[None, :])], format="csr"
        )
        A.eliminate_zeros()
        row_lower = np.append(row_lower, -np.inf)
        row_upper = np.append(row_upper, cutoff)
    scale = np.abs(A).max(axis=1).toarray().ravel()
    scale = np.where(scale > 0, scale, 1.0)
    row_tol = 0.5 * tol * (1.0 + np.abs(mip.b)) / scale[: mip.b.shape[0]]
    if cutoff is not None:
        row_tol = np.append(row_tol, 0.0)
        size = abs(cutoff) if np.isfinite(cutoff) else 0.0
        row_tol[-1] = 0.5 * tol * (1.0 + size) / scale[-1]
    A = sp.csr_matrix(A, dtype=np.float64, copy=True)
    A.data = A.data / np.repeat(scale, np.diff(A.indptr))
    return A, row_lower / scale, row_upper / scale, row_tol, scale


def _jump_data(mip, lower, upper, tol, cutoff=None):
    """The device arrays ``_jump_chunk`` reads: the rows of ``_jump_rows``
    as COO entries, and the box."""
    A, row_lower, row_upper, row_tol, scale = _jump_rows(mip, tol, cutoff)
    exclusive = np.ones(A.shape[0], dtype=bool)
    if cutoff is not None:
        exclusive[-1] = False
    dtype = mip.lp.c.dtype
    A = A.tocoo()

    def put(v):
        return jax.device_put(np.asarray(v).astype(dtype))

    return dict(
        rows=jax.device_put(A.row.astype(np.int32)),
        cols=jax.device_put(A.col.astype(np.int32)),
        a=put(A.data),
        row_lower=put(row_lower),
        row_upper=put(row_upper),
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


@functools.lru_cache(maxsize=None)
def _numba_propagate():
    """``jaddle._propagate_numba``, or ``None`` without numba."""
    try:
        from jaddle import _propagate_numba
    except ImportError:
        return None
    return _propagate_numba


@functools.cache
def _numba_bound():
    """``jaddle._bound_numba``, or ``None`` without numba."""
    try:
        from jaddle import _bound_numba
    except ImportError:
        return None
    return _bound_numba


@functools.cache
def _numba_probe():
    """``jaddle._probe_numba``, or ``None`` without numba."""
    try:
        from jaddle import _probe_numba
    except ImportError:
        return None
    return _probe_numba


@functools.cache
def _numba_separate():
    """``jaddle._separate_numba``, or ``None`` without numba."""
    try:
        from jaddle import _separate_numba
    except ImportError:
        return None
    return _separate_numba


def _f64(values):
    """``values`` as a float64 numpy array (no copy when it already is)."""
    return np.asarray(values, np.float64)


def _csr_transpose(A_T, c, m):
    """Whether ``A_T`` is a scipy CSR matrix of shape ``(len(c), m)``, as the
    compiled Lagrangian kernels take it."""
    return (
        A_T is not None
        and sp.issparse(A_T)
        and A_T.format == "csr"
        and A_T.shape == (np.shape(c)[0], m)
    )


def _lp_solver(name, option):
    """The LP solver an option names: ``"jaddle"`` (PDHG) or ``"highs"``;
    ``"pdhg"`` and ``"exact"`` are the earlier names of the two."""
    name = {"pdhg": "jaddle", "exact": "highs"}.get(name, name)
    if name not in ("jaddle", "highs"):
        raise ValueError(f"{option} must be 'jaddle' or 'highs', not {name!r}")
    return name


def _separation_backend(backend):
    """The cut separation backend for ``backend`` (``"auto"``, ``"numba"``,
    ``"jax"`` or ``"numpy"``): ``"auto"`` is numba when it is installed,
    else JAX."""
    if backend not in ("auto", "numba", "jax", "numpy"):
        raise ValueError(
            f"separation backend must be 'auto', 'numba', 'jax' or 'numpy', not {backend!r}"
        )
    if backend == "auto":
        return "numba" if _numba_separate() is not None else "jax"
    if backend == "numba" and _numba_separate() is None:
        raise ValueError(
            "the numba separation backend needs numba: pip install jaddle[fast]"
        )
    return backend


def _numba_jump():
    """``jaddle._jump_numba``, or ``None`` without numba."""
    try:
        from jaddle import _jump_numba
    except ImportError:
        return None
    return _jump_numba


def _jump_backend(backend, data=None):
    """The feasibility jump backend for ``backend`` (``"auto"``, ``"numba"``
    or ``"jax"``); device ``data`` means JAX."""
    if backend not in ("auto", "numba", "jax"):
        raise ValueError(
            f"jump backend must be 'auto', 'numba' or 'jax', not {backend!r}"
        )
    if backend == "auto":
        return "jax" if data is not None or _numba_jump() is None else "numba"
    if backend == "numba":
        if data is not None:
            raise ValueError("data (device arrays) is for the jax jump backend")
        if _numba_jump() is None:
            raise ImportError(
                "the numba jump backend needs numba: pip install jaddle[fast]"
            )
    return backend


_JUMP_CHUNK_MAX = 100_000  # moves per call of the numba jump


def _feasibility_jump_host(
    mip,
    x,
    lower,
    upper,
    tol,
    max_iterations,
    deadline,
    seed,
    cutoff,
    improve=False,
    improve_step=None,
    stop=None,
):
    """``feasibility_jump`` with the numba backend (``x`` already rounded
    into the box). Runs in chunks of 1,000 moves, doubling up to
    ``_JUMP_CHUNK_MAX`` (about 0.2 s on academictimetablesmall), with the
    deadline checked between them; each chunk recomputes the activities, which
    keeps rounding error from building up. With ``improve``, the search goes
    on after each point it finds, for better ones (see ``feasibility_jump``),
    and returns the best. ``stop`` (a ``threading.Event``) ends it between
    chunks too."""
    fj = _numba_jump()
    if improve and cutoff is None:
        cutoff = np.inf  # an objective row to tighten once a point is found
    A, row_lower, row_upper, row_tol, scale = _jump_rows(mip, tol, cutoff)
    C = A.tocsc()
    x = np.array(x, np.float64)
    w = np.ones(A.shape[0])
    lower = np.asarray(lower, np.float64)
    upper = np.asarray(upper, np.float64)
    is_int = np.asarray(mip.integrality, bool)
    if improve:
        # Each better point must improve by the objective step (objectives
        # that are multiples of one), else by 1e-4 relative.
        step = (
            objective_step(mip.c, mip.integrality | mip.implied)
            if improve_step is None
            else improve_step
        )
        improvement = (
            A.shape[0] - 1,
            0.0 if step is None else float(step) / scale[-1],
            1e-4 if step is None else 0.0,
            np.empty_like(x),
        )
    else:
        improvement = (-1, 0.0, 0.0, np.empty(0))
    steps, size, chunk = 0, 1000, 0
    best, best_count = x.copy(), np.inf
    found_x = None
    while steps < max_iterations:
        if deadline is not None and time.time() >= deadline:
            break
        if stop is not None and stop.is_set():
            break
        size = int(min(size, max_iterations - steps))
        moves, found, _, _ = fj.run(
            x,
            w,
            A.indptr,
            A.indices,
            A.data,
            C.indptr,
            C.indices,
            C.data,
            row_lower,
            row_upper,
            row_tol,
            lower,
            upper,
            is_int,
            size,
            (seed * 1_000_003 + chunk) % 2**32,
            *improvement,
        )
        steps += int(moves)
        if found:
            if not improve:
                return x, True, steps
            found_x = improvement[3].copy()
            if moves < size:
                break  # no better point can exist within the row's tolerance
        elif found_x is None:
            # Keep the near-miss with the fewest violated rows, for ``repair``.
            count = _violated_rows(mip, x, tol).size
            if count < best_count:
                best, best_count = x.copy(), count
        # A fixed schedule, not one timed to the clock, so a run depends
        # only on its inputs (unless the deadline cuts it short).
        size = min(2 * size, _JUMP_CHUNK_MAX)
        chunk += 1
    if found_x is not None:
        return found_x, True, steps
    return best, False, steps


def _jump_substitution(mip):
    """``mip``'s rows with its defined integers substituted out
    (``jaddle.presolve.defined_integers``), or ``None``; built once per MIP
    (a race between threads only builds it twice)."""
    defined = getattr(mip, "_defined_integers", False)
    if defined is False:
        defined = jaddle_presolve.defined_integers(
            mip.A, mip.b, mip.n_eq, mip.c, mip.integrality, mip.lower, mip.upper
        )
        mip._defined_integers = defined
    return defined


class _JumpRows:
    """The rows feasibility jump searches in place of ``mip``'s, with its
    defined integers substituted out (``defined``, a
    ``jaddle.presolve.DefinedIntegers``) and their bound rows set from the
    box ``lower``/``upper``: what ``_jump_rows``, ``_violated_rows`` and
    ``_feasibility_jump_host`` read of a MIP, over the kept columns."""

    def __init__(self, mip, defined, lower, upper):
        keep = defined.keep
        self.A = defined.A
        self.b = defined.b
        self.row_lower, self.row_upper = defined.rows(lower, upper)
        self.c = mip.c[keep]
        self.integrality = mip.integrality[keep]
        self.implied = mip.implied[keep]
        self.num_variables = keep.size


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
    backend="auto",
    improve=False,
    improve_step=None,
    stop=None,
    substitute=True,
):
    """
    Feasibility Jump (Luteberget & Sartor, 2023): an LP-free local search
    for a point that meets ``mip``'s rows, from ``x0``.

    ``substitute`` (numba only): integers that an equality row defines
    (``jaddle.presolve.defined_integers``: ``z = Σ a_j x_j`` over integers,
    ``|a_z| = 1``, zero cost) are substituted out of the rows the search
    moves in, and recomputed from the point it finds. One move cannot change
    such a row's ``x`` without breaking it; on 30n20b8 (57 start times
    ``S_j = Σ_t t·x_jt``) the plain jump stalled with a few of those rows
    violated, and with them substituted it found points on every seed in
    1–3 s. ``z``'s bounds become rows over the box searched.

    ``backend``: ``"numba"`` runs the incremental jump on the host
    (``jaddle._jump_numba``; needs numba, ``pip install jaddle[fast]``). It
    moves one variable a step and updates only what that variable's rows
    touch, as HiGHS does, so a step costs the nonzeros of those rows. On
    academictimetablesmall that is about 555 nonzeros of 230,000.
    ``"jax"`` is the device version described below, where every step is
    a few passes (and a sort) over all the nonzeros. ``"auto"`` (default)
    picks numba when it is installed and ``data`` is not given. Under numba
    ``steps`` counts single moves; under JAX each step may move many
    variables.

    ``improve`` (numba only): after each point it finds, the search asks
    for a better one. The objective row ``c·x <= cutoff`` (added when
    there is no ``cutoff``) drops below that point's objective by
    ``improve_step`` (``None``: the objective step when objective values
    are multiples of one, else 1e-4 relative). The moves then go on with
    the weights as they are, until ``max_iterations`` or ``deadline``. It
    returns the best point found. On academictimetablesmall, a loop of
    separate cutoff jumps took the objective from 4,623 to 282 in 27 s,
    mostly in rebuilding the rows each call; this does it within one call.
    ``stop`` (numba only, a ``threading.Event``) ends the search early when
    set, as the deadline does.

    The device version:

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
    backend = _jump_backend(backend, data)
    if max_iterations is None:
        if backend == "numba":
            max_iterations = max(100_000, 50 * (n + m))  # single moves
        else:
            max_iterations = max(10_000, 20 * (n + m))
    x = np.clip(np.asarray(x0, np.float64), lower, upper)
    ints = mip.integrality
    x[ints] = np.clip(np.round(x[ints]), lower[ints], upper[ints])
    if backend == "numba":
        defined = _jump_substitution(mip) if substitute else None
        if defined is None:
            return _feasibility_jump_host(
                mip,
                x,
                lower,
                upper,
                tol,
                max_iterations,
                deadline,
                seed,
                cutoff,
                improve,
                improve_step,
                stop,
            )
        keep = defined.keep
        point, feasible, steps = _feasibility_jump_host(
            _JumpRows(mip, defined, lower, upper),
            x[keep],
            lower[keep],
            upper[keep],
            tol,
            max_iterations,
            deadline,
            seed,
            cutoff,
            improve,
            improve_step,
            stop,
        )
        point = defined.primal(point)
        if feasible:
            # The substituted rows are the MIP's, exactly at integer points;
            # check the point in the MIP's own rows and box all the same.
            feasible = _violated_rows(mip, point, tol).size == 0 and bool(
                np.all(point >= lower - tol) and np.all(point <= upper + tol)
            )
        return point, feasible, steps
    if improve:
        raise ValueError("improve needs the numba jump backend")
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


def integer_roundings(x, integrality, lower, upper, snap=1e-6):
    """
    Up to three roundings of ``x``'s integers, in the order round and repair
    tries them (``_BranchAndBound.round_repair``): up (``ceil(x − snap)``), nearest and down
    (``floor(x + snap)``), each within ``[lower, upper]``, duplicates
    dropped. ``snap`` puts values within it of an integer on that integer
    first, so the noise of a PDHG point does not round them away. Returns a
    list of arrays over the integer variables.
    """
    xi = np.asarray(x, np.float64)[integrality]
    lo = np.asarray(lower, np.float64)[integrality]
    up = np.asarray(upper, np.float64)[integrality]
    out, seen = [], set()
    for values in (np.ceil(xi - snap), np.round(xi), np.floor(xi + snap)):
        values = np.clip(values, lo, up)
        key = values.tobytes()
        if key not in seen:
            seen.add(key)
            out.append(values)
    return out


def _relative_gap(value, bound):
    """The relative gap between an incumbent's min-form ``value`` and a
    lower ``bound`` (both with the objective's offset)."""
    return max(value - bound, 0.0) / max(abs(value), 1e-10)


def _complete_start(mip, x, tol, max_seconds):
    """The user's start ``x`` (``check_mip_start``) as a feasible point of
    ``mip``: its integers rounded and, when it is not feasible as it is and
    ``mip`` has continuous variables, those solved for by a Jaddle LP with
    the integers fixed (as ``_BranchAndBound.try_incumbent`` repairs a
    rounding). ``None`` when it cannot be."""
    ints = mip.integrality
    point = _fill_start(x, mip.lower, mip.upper)
    point[ints] = np.round(point[ints])
    if mip.is_feasible(point, tol):
        return point
    if ints.all():
        return None
    fixed_lower = np.where(ints, point, mip.lower)
    fixed_upper = np.where(ints, point, mip.upper)
    if np.any(fixed_lower > fixed_upper):
        return None
    dtype = mip.lp.c.dtype
    lp = mip.lp.with_values(
        mip.lp.values()._replace(
            lower_bounds=jnp.asarray(fixed_lower, dtype),
            upper_bounds=jnp.asarray(fixed_upper, dtype),
        )
    )
    result = jl.solve(
        lp,
        max_seconds=max_seconds,
        primal_feasibility_tolerance=0.1 * tol,
        termination_norm="inf",
    )
    repaired = np.clip(
        np.asarray(result["solution"].primal, np.float64), fixed_lower, fixed_upper
    )
    repaired[ints] = point[ints]
    return repaired if mip.is_feasible(repaired, tol) else None


def check_mip_start(x, mip):
    """A user's starting point for ``mip`` (``solve(mip_start=...)``) as a
    float64 array, or ``None``. It has one entry per column; continuous
    entries may be NaN (the LP over the continuous columns fills them in),
    integer entries may not."""
    if x is None:
        return None
    x = np.array(x, dtype=np.float64).reshape(-1)
    if x.shape[0] != mip.num_variables:
        raise ValueError(
            f"mip_start has {x.shape[0]} entries for {mip.num_variables} variables"
        )
    if np.isnan(x[mip.integrality]).any():
        raise ValueError("mip_start has NaN for an integer variable")
    if np.isinf(x).any():
        raise ValueError("mip_start has an infinite entry")
    return x


def _fill_start(x, lower, upper):
    """``x`` with its NaN entries set to the value nearest zero in their
    bounds, and clipped to the bounds: a point to round and repair."""
    x = np.where(np.isnan(x), 0.0, x)
    return np.clip(x, lower, upper)


def one_opt(mip, x, lower=None, upper=None, tol=1e-9, A_csc=None):
    """
    1-opt: move integer variables of the feasible point ``x`` one at a time
    in the direction that improves the objective, each as far as its bounds
    (``lower``/``upper``, default ``mip``'s) and every row's slack allow, in
    whole steps. The largest gains ``|c_j|·t`` go first, and each step is
    recomputed against the slack the earlier ones left. Continuous variables
    stay put, so the point stays feasible. One pass, as in SCIP's oneopt
    without its LP.

    ``A_csc``: ``mip.A`` in CSC form, for callers that keep one. Returns the
    improved point, or ``None`` when no variable can move.
    """
    lower = mip.lower if lower is None else np.asarray(lower, np.float64)
    upper = mip.upper if upper is None else np.asarray(upper, np.float64)
    x = np.array(x, dtype=np.float64)
    cand = np.flatnonzero(mip.integrality & (mip.c != 0.0))
    if cand.size == 0:
        return None
    A = mip.A.tocsc() if A_csc is None else A_csc
    act = mip.A @ x
    room_up = np.maximum(mip.row_upper - act, 0.0)  # inf on rows without one
    room_down = np.maximum(act - mip.row_lower, 0.0)

    def limits(cols):
        """The largest whole step of each column in ``cols``."""
        d = -np.sign(mip.c[cols])
        t = np.where(d > 0, upper[cols] - x[cols], x[cols] - lower[cols])
        sub = A[:, cols]
        counts = np.diff(sub.indptr)
        col = np.repeat(np.arange(cols.size), counts)
        a = sub.data * d[col]
        rows = sub.indices
        with np.errstate(divide="ignore", invalid="ignore"):
            lim = np.where(
                a > 0, room_up[rows] / a, np.where(a < 0, room_down[rows] / -a, np.inf)
            )
        np.minimum.at(t, col, lim)
        t = np.where(np.isfinite(t), np.floor(t + tol), 0.0)  # unbounded: skip
        return np.maximum(t, 0.0), d

    t, _ = limits(cand)
    order = cand[t > 0]
    if order.size == 0:
        return None
    gains = np.abs(mip.c[order]) * t[t > 0]
    moved = False
    for j in order[np.argsort(-gains, kind="stable")]:
        # ``limits`` for one column, on the CSC arrays directly.
        d = -1.0 if mip.c[j] > 0 else 1.0
        step = upper[j] - x[j] if d > 0 else x[j] - lower[j]
        start, end = A.indptr[j], A.indptr[j + 1]
        rows, a = A.indices[start:end], A.data[start:end] * d
        up, down = a > 0, a < 0
        if up.any():
            step = min(step, float((room_up[rows[up]] / a[up]).min()))
        if down.any():
            step = min(step, float((room_down[rows[down]] / -a[down]).min()))
        step = np.floor(step + tol) if np.isfinite(step) else 0.0
        if step <= 0:
            continue
        change = A.data[start:end] * d * step
        x[j] += d * step
        act[rows] += change
        room_up[rows] = np.maximum(mip.row_upper[rows] - act[rows], 0.0)
        room_down[rows] = np.maximum(act[rows] - mip.row_lower[rows], 0.0)
        moved = True
    return x if moved else None


def _violated_rows(mip, x, tol):
    """The rows ``x`` violates by more than ``tol`` (relative to each row's
    ``1 + |b_i|``, as ``MIP.violation`` measures)."""
    Ax = mip.A @ x
    excess = np.maximum(np.maximum(mip.row_lower - Ax, Ax - mip.row_upper), 0.0)
    return np.flatnonzero(excess > tol * (1.0 + np.abs(mip.b)))


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


def trivial_points(mip, lower=None, upper=None):
    """
    The trivial heuristic's points (as SCIP's ``trivial``), each one matvec
    to check: every variable at zero, at its lower bound, at its upper
    bound, and at the bound its locks allow (the lock point). The lock
    point puts a variable at its lower bound when at most as many rows lock
    it from going down as from going up (an equality locks both ways), and
    at its upper bound otherwise. An infinite bound is replaced by zero,
    clipped to the box. Integers are rounded. Cheap insurance on covering
    (all ones), packing and knapsack (all zeros) and many big-M models.
    """
    lower = mip.lower if lower is None else np.asarray(lower, np.float64)
    upper = mip.upper if upper is None else np.asarray(upper, np.float64)
    zero = np.clip(np.zeros(lower.shape), lower, upper)
    at_lower = np.where(np.isfinite(lower), lower, zero)
    at_upper = np.where(np.isfinite(upper), upper, zero)
    A = mip.A
    rows = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
    eq = rows < mip.n_eq
    n = lower.size
    # In a ``<=`` row, a positive entry locks its variable from going up and
    # a negative one from going down.
    up_locks = np.bincount(A.indices[eq | (A.data > 0)], minlength=n)
    down_locks = np.bincount(A.indices[eq | (A.data < 0)], minlength=n)
    locks = np.where(down_locks <= up_locks, at_lower, at_upper)
    points = []
    for point in (zero, at_lower, at_upper, locks):
        point = point.copy()
        point[mip.integrality] = np.round(point[mip.integrality])
        if not any(np.array_equal(point, seen) for seen in points):
            points.append(point)
    return points


# %%
# Block decomposition (``block_decomposition``)
class LinkingBlocks(
    collections.namedtuple("LinkingBlocks", "linking blocks block_rows")
):
    """A MIP's block structure (``linking_blocks``): ``linking``, the
    indices of the linking rows (``<=`` rows); ``blocks``, one array of
    column indices per block, largest first; ``block_rows``, the rows of
    each block (every other row lies inside one block)."""

    __slots__ = ()


def linking_blocks(
    mip,
    min_length=20,
    quantile=99.0,
    max_linking_share=0.2,
    max_block_share=0.5,
):
    """
    Blocks that only a few long ``<=`` rows join, or ``None``.

    The candidate linking rows are the inequality rows with at least
    ``max(min_length, quantile-th percentile)`` entries. Without them the
    columns fall into connected components (of the bipartite row-column
    graph); a candidate whose columns all lie in one component goes back
    into it. The structure is accepted when there are at least two blocks
    of two or more columns, the linking rows are at most
    ``max_linking_share`` of the rows, and no block has more than
    ``max_block_share`` of the columns. Columns in components of their own
    form one block together (it separates, so HiGHS solves it at once).

    On stp3d (Steiner tree packing, 204,880 binaries) the 2,171 node
    capacity rows join 24 blocks, one a net; after HiGHS's presolve, 1,936
    of them join 21.
    """
    from scipy.sparse import csgraph

    A = mip.A
    m, n = A.shape
    if m < 2 or n < 2:
        return None
    lengths = np.diff(A.indptr)
    threshold = max(float(min_length), float(np.percentile(lengths, quantile)))
    linking = (np.arange(m) >= mip.n_eq) & (lengths >= threshold)
    if not linking.any() or linking.sum() > max_linking_share * m:
        return None
    inner = np.flatnonzero(~linking & (lengths > 0))
    K = A[inner]
    pattern = sp.csr_matrix(
        (np.ones(K.nnz, np.int8), K.indices, K.indptr), shape=(inner.size, n)
    )
    graph = sp.bmat([[None, pattern], [pattern.T, None]], format="csr")
    _, label = csgraph.connected_components(graph, directed=False)
    label = label[inner.size :]  # each column's component
    # Candidates inside one component are rows of it, not links.
    rows = np.flatnonzero(linking)
    L = A[rows]
    spans = np.array(
        [
            np.unique(label[L.indices[L.indptr[i] : L.indptr[i + 1]]]).size
            for i in range(rows.size)
        ]
    )
    linking[rows[spans < 2]] = False
    rows = rows[spans >= 2]
    if rows.size == 0:
        return None
    sizes = np.bincount(label)
    real = sizes >= 2
    if real.sum() < 2 or sizes.max() > max_block_share * n:
        return None
    # Singleton columns share one block, labelled past the real ones.
    loose = ~real[label]
    _, block_of = np.unique(np.where(loose, -1, label), return_inverse=True)
    block_of = block_of.reshape(-1)
    n_blocks = int(block_of.max()) + 1
    columns = [np.flatnonzero(block_of == b) for b in range(n_blocks)]
    inner = np.flatnonzero(~linking & (lengths > 0))
    row_block = block_of[A.indices[A.indptr[inner]]]
    row_lists = [inner[row_block == b] for b in range(n_blocks)]
    order = sorted(range(n_blocks), key=lambda b: -columns[b].size)
    return LinkingBlocks(
        rows, [columns[b] for b in order], [row_lists[b] for b in order]
    )


class _HighsHolder:
    """The HiGHS instance a background job is running, so another thread
    can stop it (``cancel``): a block's solve runs to its own time limit
    otherwise."""

    def __init__(self):
        self.current = None
        self.cancelled = False
        self.lock = threading.Lock()
        # Progress, for the search's result (``negotiate_blocks`` fills it).
        self.stats = {}

    def run(self, highs):
        with self.lock:
            if self.cancelled:
                return False
            self.current = highs
        try:
            highs.run()
        finally:
            with self.lock:
                self.current = None
        return not self.cancelled

    def cancel(self):
        with self.lock:
            self.cancelled = True
            if self.current is not None:
                try:
                    self.current.cancelSolve()
                except Exception:
                    pass


def _block_highs(mip, cols, rows, lower, upper, gap, relax=False):
    """A silent one-thread ``highspy.Highs`` holding the block of ``mip``
    on the columns ``cols`` and rows ``rows`` (``linking_blocks``), with
    HiGHS's presolve and MIP solver, or (``relax``) its LP relaxation for
    the simplex method without presolve, so each run starts from the last
    basis; ``cancelSolve`` stops a run."""
    import highspy

    A = mip.A[rows][:, cols].tocsc()
    A.sort_indices()
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = cols.size, rows.size
    lp.col_cost_ = np.asarray(mip.c[cols], np.float64)
    lp.col_lower_ = np.asarray(lower[cols], np.float64)
    lp.col_upper_ = np.asarray(upper[cols], np.float64)
    lp.row_lower_ = np.asarray(mip.row_lower[rows], np.float64)
    lp.row_upper_ = np.asarray(mip.row_upper[rows], np.float64)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    highs = highspy.Highs()
    # No ``threads`` option: HiGHS's scheduler is global, and once another
    # instance (presolve) has started it, setting it fails every run.
    options = [("output_flag", False), ("parallel", "off")]
    if relax:
        options += [("presolve", "off"), ("solver", "simplex")]
    else:
        options += [("mip_rel_gap", float(gap))]
    for option, value in options:
        highs.setOptionValue(option, value)
    highs.passModel(lp)
    ints = np.flatnonzero(mip.integrality[cols]).astype(np.int32)
    if ints.size and not relax:
        highs.changeColsIntegrality(
            ints.size, ints, np.full(ints.size, highspy.HighsVarType.kInteger)
        )
    highs.HandleUserInterrupt = True
    return highs


def negotiate_blocks(
    mip,
    blocks=None,
    lower=None,
    upper=None,
    deadline=None,
    stop=None,
    tol=1e-6,
    block_seconds=10.0,
    gap=0.02,
    present_start=0.5,
    present_growth=1.6,
    present_max=1e4,
    history_weight=0.1,
    stall_sweeps=3,
    joint_seconds=0.0,
    max_window_share=0.1,
    lp_first=True,
    noise=0.05,
    shake=0.0,
    seed=0,
    holder=None,
    log=None,
):
    """
    A feasible point by negotiated congestion (PathFinder, McMurchie and
    Ebeling, 1995, the rip-up-and-reroute of VLSI routers) over the blocks
    of ``linking_blocks``, or ``None``. It solves no LP of the whole MIP.

    Each block is solved alone by HiGHS, one after another, with its own
    rows and its objective plus prices on the linking rows, so a block pays
    to use capacity the others already hold. A linking row ``a·x <= b``
    with usage ``u`` by the other blocks has the price
    ``(1 + h)·p·max(0, u - b + s)/s + w·h``, in units of the objective's
    typical cost per unit ``s`` of the row's typical entry: ``h`` the row's
    history (its overuse summed over the sweeps), ``p`` the present factor
    (``present_start``, times ``present_growth`` each sweep, at most
    ``present_max``: larger costs only blunt HiGHS's relative gap), and
    ``w`` ``history_weight``. Each sweep solves every block again (a block
    that keeps its route still holds capacity others need).

    With ``joint_seconds`` (default 0: off; on stp3d no window ever found
    a point, and their time slowed the negotiation), when the overuse has
    not fallen for ``stall_sweeps`` sweeps, a window
    around the conflicts is searched as one sub-MIP of the whole model
    (as a router rips up a region): the linking rows within ``r`` steps of
    the overused ones (``_linking_adjacency``; ``r`` is 1, then one more at
    each attempt), their columns, and the columns that share a block row
    with those (a path's flow variables beside its arcs) are freed, every
    other column fixed, and HiGHS searches for up to ``joint_seconds``. It
    settles the last few conflicts the prices circle around. A window
    wider than ``max_window_share`` of the columns is skipped and the next
    starts at ``r = 1`` again (wide ones ran far past their time limit in
    HiGHS's presolve). If it fails, each history is scaled by a random
    factor in ``1 ± shake`` (default 0: on presolved stp3d a shake of 0.5
    undid the negotiation's progress) and the negotiation goes on.

    On stp3d (no presolve) the overuse fell from 29 rows to 2 in seven
    sweeps of about 3 s, and the prices alone settled the last two by 68 s
    at 538.8 (best known 493.7).

    ``holder`` (a ``_HighsHolder``) lets another thread stop a running
    HiGHS solve; ``stop`` (an event) is checked between solves; ``log``
    (a function of a string) gets a line a sweep. Returns
    ``(x, feasible, "K")``, the shape of the background jobs' results.
    """
    import highspy

    started = time.time()
    holder = _HighsHolder() if holder is None else holder
    say = log or (lambda line: None)
    if blocks is None:
        blocks = linking_blocks(mip)
    if blocks is None:
        return None, False, "K"
    lower = mip.lower if lower is None else np.asarray(lower, np.float64)
    upper = mip.upper if upper is None else np.asarray(upper, np.float64)
    rng = np.random.default_rng(seed)

    def halted():
        return (
            holder.cancelled
            or (stop is not None and stop.is_set())
            or (deadline is not None and time.time() >= deadline)
        )

    L = mip.A[blocks.linking]
    capacity = mip.row_upper[blocks.linking]
    size = np.ones(L.shape[0])
    for i in range(L.shape[0]):
        entries = np.abs(L.data[L.indptr[i] : L.indptr[i + 1]])
        if entries.size:
            size[i] = max(float(np.median(entries)), 1e-9)
    L_csc = L.tocsc()
    costs = np.abs(mip.c[mip.c != 0])
    unit = float(np.median(costs)) if costs.size else 1.0
    # Each block: its columns, its linking entries, its LP relaxation in
    # HiGHS (``lp_first``; warm from the last sweep's basis, as only the
    # costs change) and its MIP, built when the LP's point is fractional.
    # On a network block (a path, a flow) the LP's vertex is integral: on
    # stp3d every block's was, and a sweep of warm LPs took 1.2 s against
    # 7.4 s for the MIPs. Alone, warm LPs kept their old routes on ties and
    # needed more sweeps (152 s to a point, against 68-74 s with the MIPs);
    # with ``noise`` (random costs up to 5% of a typical cost, new each
    # sweep) they took 37-146 s (median 52 s, five runs).
    parts = []
    for cols, rows in zip(blocks.blocks, blocks.block_rows):
        parts.append(
            [
                cols,
                L_csc[:, cols].tocsr(),
                (
                    _block_highs(mip, cols, rows, lower, upper, gap, relax=True)
                    if lp_first
                    else None
                ),
                None,
                rows,
            ]
        )
    holder.stats.setdefault("mip_solves", 0)

    def run_block(highs, cost):
        """``(status, point)``: ``"ok"`` with the point, ``"none"`` (no
        point in the time), ``"infeasible"``, ``"stopped"`` or ``"error"``."""
        n_cols = cost.size
        highs.changeColsCost(n_cols, np.arange(n_cols, dtype=np.int32), cost)
        budget = block_seconds
        if deadline is not None:
            budget = min(budget, deadline - time.time())
        highs.setOptionValue("time_limit", highs.getRunTime() + max(budget, 1e-3))
        if not holder.run(highs):
            return "stopped", None
        status = highs.getModelStatus()
        if status == highspy.HighsModelStatus.kNotset:
            return "error", None
        if status == highspy.HighsModelStatus.kInfeasible:
            return "infeasible", None
        if highs.getInfo().primal_solution_status != 2:  # kSolutionStatusFeasible
            return "none", None
        return "ok", np.array(highs.getSolution().col_value, np.float64)
    x = np.clip(np.zeros(lower.shape), lower, upper)
    solved = np.zeros(len(parts), bool)
    usage = L @ x
    history = np.zeros(L.shape[0])
    present = float(present_start)
    best_over, stalled, joints = np.inf, 0, 0
    joint = None  # the whole MIP in HiGHS, built at the first stall
    adjacent = None  # the linking rows' adjacency (``_linking_adjacency``)
    todo = list(range(len(parts)))
    sweep = 0
    while not halted():
        sweep += 1
        for b in todo:
            if halted():
                break
            cols, Lb, relaxed, exact, rows = parts[b]
            own = Lb @ x[cols]
            other = usage - own
            crowd = np.maximum(other - capacity + size, 0.0) / size
            price = unit * ((1.0 + history) * present * crowd + history_weight * history) / size
            cost = mip.c[cols] + Lb.T @ price
            if noise > 0:  # breaks ties afresh each sweep
                cost = cost + unit * noise * rng.random(cols.size)
            ints = mip.integrality[cols]
            status, xb = ("none", None) if relaxed is None else run_block(relaxed, cost)
            if status == "ok" and np.any(np.abs(xb[ints] - np.round(xb[ints])) > 1e-6):
                status = "fractional"
            if status in ("fractional", "none"):
                if exact is None:
                    exact = parts[b][3] = _block_highs(mip, cols, rows, lower, upper, gap)
                if relaxed is not None:
                    holder.stats["mip_solves"] += 1
                status, xb = run_block(exact, cost)
            if status == "stopped":
                break
            if status == "error":
                say(f"block {b}: HiGHS did not run")
                return None, False, "K"
            if status == "infeasible":
                say(f"block {b} ({cols.size} columns) is infeasible on its own")
                return None, False, "K"
            if status == "ok":
                xb[ints] = np.round(xb[ints])
                x[cols] = np.clip(xb, lower[cols], upper[cols])
                solved[b] = True
                usage = other + Lb @ x[cols]
        if halted():
            break
        over = np.maximum(usage - capacity - tol * (1.0 + np.abs(capacity)), 0.0)
        overused = over > 0
        total = float((over / size).sum())
        say(
            f"sweep {sweep}: {int(overused.sum())} overused linking rows "
            f"(overuse {total:.4g}), objective {float(mip.c @ x):.6g}, "
            f"{time.time() - started:.1f}s"
        )
        holder.stats.update(
            sweeps=sweep,
            overused_rows=int(overused.sum()),
            seconds=time.time() - started,
        )
        if solved.all() and not overused.any():
            if mip.is_feasible(x, tol):
                return x, True, "K"
        history += over / size
        present = min(present * present_growth, present_max)
        if total < best_over - 1e-12:
            best_over, stalled = total, 0
        else:
            stalled += 1
        if stalled >= stall_sweeps and solved.all() and joint_seconds > 0:
            stalled, joints = 0, joints + 1
            if joint is None:
                joint = _highs_submip(mip, mip.integrality)
            if adjacent is None:
                adjacent = _linking_adjacency(mip, blocks)
                inner = mip.A[np.concatenate(blocks.block_rows)]
                inner = sp.csr_matrix(
                    (np.ones(inner.nnz, np.int32), inner.indices, inner.indptr),
                    shape=inner.shape,
                )
                inner_T = inner.T.tocsr()
                linking_T = (L != 0).T.tocsr().astype(np.int32)
            near = overused.copy()
            for _ in range(joints):
                near = near | (adjacent @ near.astype(np.int32) > 0)
            free = linking_T @ near.astype(np.int32) > 0
            free = free | (inner_T @ (inner @ free.astype(np.int32) > 0).astype(np.int32) > 0)
            if free.sum() > max_window_share * x.size:
                joints = 0  # too wide to search in time: start small again
                continue
            box_lower = np.where(free, lower, x)
            box_upper = np.where(free, upper, x)
            seconds = joint_seconds
            if deadline is not None:
                seconds = min(seconds, deadline - time.time())
            joint.changeColsBounds(x.size, np.arange(x.size, dtype=np.int32), box_lower, box_upper)
            joint.setOptionValue("objective_bound", np.inf)
            joint.setOptionValue("time_limit", joint.getRunTime() + max(seconds, 1e-3))
            if holder.run(joint) and joint.getInfo().primal_solution_status == 2:
                point = np.array(joint.getSolution().col_value, np.float64)
                point[mip.integrality] = np.round(point[mip.integrality])
                point = np.clip(point, lower, upper)
                holder.stats["windows"] = holder.stats.get("windows", 0) + 1
                say(f"window of {joints} steps ({int(free.sum())} columns): found a point")
                if mip.is_feasible(point, tol):
                    return point, True, "K"
            else:
                holder.stats["windows"] = holder.stats.get("windows", 0) + 1
                say(f"window of {joints} steps ({int(free.sum())} columns): nothing")
            if shake > 0:
                history *= rng.uniform(1.0 - shake, 1.0 + shake, history.shape)
    return None, False, "K"


def _linking_adjacency(mip, blocks):
    """The linking rows' adjacency (a boolean CSR matrix): two rows are
    adjacent when one block row has columns of both. On a routing model
    (stp3d) these are neighbouring nodes."""
    def pattern(rows):
        M = mip.A[rows]
        return sp.csr_matrix(
            (np.ones(M.nnz, np.int32), M.indices, M.indptr), shape=M.shape
        )

    inner = np.concatenate(blocks.block_rows) if blocks.block_rows else []
    R = pattern(blocks.linking) @ pattern(np.asarray(inner, np.int64)).T
    adjacent = (R @ R.T).astype(bool).tocsr()
    adjacent.setdiag(False)
    adjacent.eliminate_zeros()
    return adjacent


# ``device="auto"`` runs MIPs with at most this many nonzeros on the CPU:
# their node LPs are latency-bound on an accelerator. Measured (ms per LP
# epoch, CPU vs GPU): 98 nonzeros 0.7 vs 4.2, 8,937 nonzeros 4.2 vs 5.1,
# 100,933 nonzeros 73 vs 7.5; the crossover is near 10k.
_SMALL_MIP_NNZ = 10_000


def _numba_short_jump(mip):
    """The default moves of the pump's and the tree's short jumps with the
    numba backend: ``5·(n + m)``, at least 10,000. About 200,000 moves (half
    a second) on academictimetablesmall, and milliseconds on small MIPs,
    where a failing jump (one with an unreachable cutoff) must stay cheap."""
    return max(10_000, 5 * (mip.num_variables + mip.A.shape[0]))


def _search_device(device, nnz):
    """The JAX device for the search (``None``: JAX's default)."""
    if isinstance(device, str) and device == "auto":
        if nnz > _SMALL_MIP_NNZ:
            return None
        device = "cpu"
    return jo._resolve_device(device)


# %%
# Branch and bound
def _cast_warm(warm, dtype):
    """A warm start ``(solution, opt_state)`` with its float arrays in
    ``dtype``, so an LP at one precision can start from another's point."""

    def cast(leaf):
        leaf = jnp.asarray(leaf)
        if jnp.issubdtype(leaf.dtype, jnp.floating) and leaf.dtype != dtype:
            return leaf.astype(dtype)
        return leaf

    return jax.tree.map(cast, warm)


def _fit_warm(warm, lp):
    """``warm`` for ``lp``: as it is when its duals match ``lp``'s rows,
    else its primal point with zero duals and only the step sizes of its
    optimiser state (its Halpern anchor is sized to the old rows). A
    background heuristic's warm start is the root LP's from when the job
    was queued; the cut rounds may have changed the rows since (on map06,
    31,286 duals for 30,262 rows)."""
    solution, opt_state = warm
    rows = (lp.A_eq.shape[0], lp.A_ineq.shape[0])
    have = (np.shape(solution.dual_eq)[0], np.shape(solution.dual_ineq)[0])
    # ``jl.solve`` itself gives a zero dual to an empty block's padding row.
    if all(h == r or (h == 0 and r == 1) for h, r in zip(have, rows)):
        return warm
    dtype = jnp.asarray(solution.primal).dtype
    solution = solution._replace(
        dual_eq=jnp.zeros(rows[0], dtype), dual_ineq=jnp.zeros(rows[1], dtype)
    )
    return solution, None if opt_state is None else tuple(opt_state)[:2]


def _step_sizes(opt_state):
    """The step sizes ``(k, eta)`` of a solve's ``opt_state`` as host
    floats (``None`` without one): what a batched node LP takes, without
    the Halpern anchor some solves also carry."""
    if opt_state is None:
        return None
    k, eta = tuple(opt_state)[:2]
    return float(k), float(eta)


def _raw_of(child):
    """A strong child's unrounded bound."""
    return child.bound if child.raw is None else child.raw


def _narrow(values):
    """``values`` (``float64``) as ``float32`` when every one is exactly
    representable there (``expand`` widens them back exactly), else as
    they are."""
    narrow = values.astype(np.float32)
    return narrow if np.array_equal(narrow, values) else values


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
    collections.namedtuple(
        "_StrongChild", "bound closed solution raw exact", defaults=(None, False)
    )
):
    """A strong-branching child: its bound (rounded as node bounds are),
    whether it is closed (empty or at the cutoff), its LP solution for a
    warm start (``None`` when closed or not solved, or a function giving it)
    and its unrounded bound, for scores and pseudocosts (``None``: the
    bound). Rounding to an objective step would make most gains 0 or one
    step. ``exact``: its solution is an optimal vertex of its LP (the exact
    lane's strong branching), so its node needs no LP solve of its own."""

    __slots__ = ()


# The progress table (``verbose``), laid out as HiGHS's MIP log.
_LOG_SOURCES = (
    ("B", "Bulk rounding"),
    ("D", "Dive"),
    ("E", "RENS"),
    ("F", "Feasibility pump"),
    ("H", "Rounding"),
    ("J", "Feasibility jump"),
    ("K", "Block decomposition"),
    ("L", "Objective levels"),
    ("N", "RINS"),
    ("P", "Fix-and-propagate"),
    ("R", "Repair"),
    ("S", "User start"),
    ("T", "Evaluate node"),
    ("U", "Round and repair"),
    ("Z", "Trivial"),
    ("C", "Crossover"),
    ("X", "Proximity"),
    ("Q", "Presolved restart"),
    ("W", "HiGHS first solution"),
    ("1", "1-opt"),
)


def _log_legend(sources, width=80):
    """The progress table's source legend, for the marks in ``sources``
    only (the heuristics the solve has on), wrapped to ``width``."""
    entries = [f"{mark} => {name}" for mark, name in _LOG_SOURCES if mark in sources]
    lines, line = [], "Src:"
    for i, entry in enumerate(entries):
        entry += ";" if i < len(entries) - 1 else ""
        if len(line) > 5 and len(line) + 1 + len(entry) > width:
            lines.append(line)
            line = "    "
        line += " " + entry
    lines.append(line)
    return "\n".join(lines)


_LOG_HEADER = (
    "        Nodes      |    B&B Tree     |            Objective Bounds"
    "              |  Dynamic Constraints |       Work      \n"
    "Src  Proc. InQueue |  Leaves   Expl. | BestBound       BestSol"
    "              Gap |   Cuts   InLp Confl. |  Epochs     Time"
)
# Status names for the solving report.
_STATUS_TEXT = {
    "optimal": "Optimal",
    "infeasible": "Infeasible",
    "time_limit": "Time limit reached",
    "node_limit": "Node limit reached",
    "interrupted": "Interrupted",
    "unresolved": "Unresolved",
    "unbounded_or_infeasible": "Unbounded or infeasible",
}


def _log_count(value):
    """A count in at most 7 characters, as HiGHS's log (``12345k``)."""
    value = int(value)
    if value < 10_000_000:
        return str(value)
    if value < 10_000_000_000:
        return f"{value // 1000}k"
    return f"{value // 1_000_000}m"


def _log_value(value):
    """An objective value in at most 15 characters."""
    if not np.isfinite(value):
        return "inf" if value > 0 else "-inf"
    text = f"{value:.10g}"
    return text if len(text) <= 15 else f"{value:.8e}"


class _SparseLeaf:
    """A warm start array kept as its nonzeros (``_pack_leaf``)."""

    __slots__ = ("shape", "dtype", "index", "values")

    def __init__(self, shape, dtype, index, values):
        self.shape, self.dtype, self.index, self.values = shape, dtype, index, values

    def dense(self):
        out = np.zeros(self.shape, self.dtype)
        np.put(out, self.index, self.values)
        return out


def _pack_leaf(leaf):
    """A warm start leaf as an open node keeps it: a host array that is
    mostly zeros as its nonzeros (``_SparseLeaf``: an inequality's dual is
    zero while its row is slack; on ns1830653 18% of ``dual_ineq`` and 35%
    of ``primal`` were nonzero), any other host array as its own copy (a
    batch member's point is a view of the whole batch's block, and a view
    on an open node keeps the block alive: on ns1830653 the open nodes'
    views held 2.3 times their own bytes), anything else (a device array)
    as it is. Lossless: -0.0 counts as a nonzero, and the dtype is kept."""
    if not isinstance(leaf, np.ndarray) or leaf.dtype.kind != "f":
        return leaf
    keep = np.flatnonzero((leaf != 0) | np.signbit(leaf))
    index = np.int32 if leaf.size <= np.iinfo(np.int32).max else np.int64
    if keep.size * (np.dtype(index).itemsize + leaf.itemsize) < leaf.nbytes:
        return _SparseLeaf(
            leaf.shape, leaf.dtype, keep.astype(index), leaf.reshape(-1)[keep]
        )
    return leaf.copy() if leaf.base is not None else leaf


class _PackedWarm:
    """An open node's warm start ``(solution, opt_state)`` with the
    solution's leaves packed (``_pack_leaf``); ``unpack`` gives it back.
    A solution none of whose leaves packing changes (device arrays, host
    arrays of their own) is kept as it is, and given back as itself."""

    __slots__ = ("treedef", "leaves", "opt_state", "solution")

    def __init__(self, warm):
        solution, self.opt_state = warm
        leaves, self.treedef = jax.tree.flatten(solution)
        self.leaves = [_pack_leaf(leaf) for leaf in leaves]
        self.solution = None
        if all(packed is leaf for packed, leaf in zip(self.leaves, leaves)):
            self.solution, self.leaves = solution, None

    def has_solution(self):
        return self.treedef.num_leaves > 0

    def unpack(self):
        if self.leaves is None:
            return self.solution, self.opt_state
        leaves = [
            leaf.dense() if isinstance(leaf, _SparseLeaf) else leaf
            for leaf in self.leaves
        ]
        return jax.tree.unflatten(self.treedef, leaves), self.opt_state


# The last warm start packed, as ``(warm, packed)``: a node's two children
# start from the same one (``_BranchAndBound.branch``), and share its packing.
_last_packed = None


def _pack_warm(warm):
    """``warm`` as ``_Node`` keeps it (``_PackedWarm``); ``None`` as it is."""
    global _last_packed
    if warm is None or isinstance(warm, _PackedWarm):
        return warm
    last = _last_packed
    if last is not None and last[0] is warm:
        return last[1]
    packed = _PackedWarm(warm)
    _last_packed = (warm, packed)
    return packed


class _Node:
    """An open node: its bound, depth, bound changes against the root (as
    ``(index, lower, upper)``), the parent's warm start, the branching
    that made it (``(var, down, distance, parent_value)``) and the LP
    epochs it has had in earlier batches (``epochs``). The warm start is
    kept packed (``_PackedWarm``): ``warm`` reads it back as
    ``(solution, opt_state)``."""

    __slots__ = (
        "bound",
        "depth",
        "changes",
        "_warm",
        "branch",
        "epochs",
        "inferences",
        "cut_version",
        "cut_rounds",
        "branch_value",
        "exact_tried",
        "exact_known",
        "converge",
        "exact_basis",
        "exact_sides",
        "band_edge",
    )

    def __init__(
        self,
        bound,
        depth,
        changes,
        warm,
        branch,
        epochs=0,
        inferences=0,
        cut_rounds=0,
        branch_value=None,
    ):
        self.bound, self.depth, self.changes = bound, depth, changes
        self.warm, self.branch = warm, branch
        self.epochs = epochs
        # Bounds that propagation tightened beyond the split that made it.
        self.inferences = inferences
        # The cut block version its warm start is for (set when it is
        # queued), the re-solves it has had for new tree cuts, and its LP's
        # value before them (for its pseudocost).
        self.cut_version = None
        self.cut_rounds = cut_rounds
        self.branch_value = branch_value
        # Whether the exact lane (``exact_nodes``) has taken its LP once.
        self.exact_tried = False
        # Whether ``warm`` is an optimal vertex of its LP (from the exact
        # lane's strong branching): the lane evaluates it without a solve.
        self.exact_known = False
        # Whether its LP runs to convergence on the device, over batches,
        # instead of being truncated (``exact_overflow_share``).
        self.converge = False
        # ``"exact"`` or ``"device"`` when it was taken near the top of the
        # auto band, for ``tune_band_scale``.
        self.band_edge = None
        # A key in ``_BranchAndBound.basis_store``: the HiGHS basis its exact
        # LP starts from (its parent's; ``None``: none), and once the lane
        # has solved it, its own optimal one, for its children; with its
        # strong sides' keys by ``(var, down)``.
        self.exact_basis = None
        self.exact_sides = None

    @property
    def warm(self):
        packed = self._warm
        return packed.unpack() if isinstance(packed, _PackedWarm) else packed

    @warm.setter
    def warm(self, warm):
        self._warm = _pack_warm(warm)

    def has_warm_solution(self):
        """Whether its warm start has a solution (without unpacking it)."""
        packed = self._warm
        if isinstance(packed, _PackedWarm):
            return packed.has_solution()
        return packed is not None and packed[0] is not None


_LP_RESERVED = (
    "initial_solution",
    "initial_opt_state",
    "max_epochs",
    "max_seconds",
    "device",
    "precision",
)


# With ``parallel_heuristics``, the improving jump runs in slices of this
# many seconds, each from the incumbent of its time (see
# ``_BranchAndBound.schedule_improve_jump``).
_JUMP_SLICE_SECONDS = 5.0
# A restart slice from a perturbed incumbent or a node point (``jump_restarts``).
_JUMP_RESTART_SECONDS = 2.0


def _pseudocost_estimates(pc_sum, pc_count, x, candidates):
    """Each candidate's estimated down and up bound gains: fraction times
    pseudocost (``pc_sum / pc_count``, shape ``(2, n)``: down, up), the mean
    pseudocost where a direction has no observation yet."""
    f = x[candidates] - np.floor(x[candidates])
    known = pc_count > 0
    mean = np.array(
        [
            (
                pc_sum[d][known[d]].sum() / pc_count[d][known[d]].sum()
                if known[d].any()
                else 1.0
            )
            for d in range(2)
        ]
    )
    pc = np.where(
        known[:, candidates],
        pc_sum[:, candidates] / np.maximum(pc_count[:, candidates], 1),
        mean[:, None],
    )
    return f * pc[0], (1.0 - f) * pc[1]


def _exact_node_lp(highs, mip, lower, upper, seconds, basis=None):
    """The LP of ``mip``'s rows over ``[lower, upper]`` solved by HiGHS's
    dual simplex in ``highs`` (which holds ``mip``'s rows and keeps its
    basis from the last call), from ``basis`` (a ``highspy.HighsBasis``
    for those rows, such as the parent node's optimal one) when given,
    else from the last call's. Returns ``(status, x, y_eq, y_ineq)``:
    ``status`` is ``"optimal"``, ``"infeasible"`` or ``"failed"`` (time
    limit or another status), and the duals are in Jaddle's convention (``L = cᵀx +
    yᵀ(Ax − b)``, inequality duals ``>= 0``)."""
    import highspy

    n = lower.size
    highs.changeColsBounds(
        n,
        np.arange(n, dtype=np.int32),
        np.asarray(lower, np.float64),
        np.asarray(upper, np.float64),
    )
    if basis is not None:
        highs.setBasis(basis)
    # ``time_limit`` counts the instance's whole run time.
    highs.setOptionValue("time_limit", highs.getRunTime() + float(seconds))
    highs.run()
    status = highs.getModelStatus()
    MS = highspy.HighsModelStatus
    if status == MS.kInfeasible:
        return "infeasible", None, None, None
    if status != MS.kOptimal:
        return "failed", None, None, None
    y_eq, y_ineq = _jaddle_duals(highs, mip)
    x = np.asarray(highs.getSolution().col_value, np.float64)
    return "optimal", x, y_eq, y_ineq


def _shaped_like(solution, template):
    """``solution`` (a ``SaddleState``) with each dual vector padded with
    zeros, or cut, to its length in ``template``: ``jaddle_linear.crossover``
    gives duals for the LP's rows only, while ``solve()`` gives an empty row
    block one zero row, and a batch's warm starts must share their shapes."""

    def fit(values, like):
        values = np.asarray(values, np.float64)
        size = np.shape(like)[0]
        if values.shape[0] == size:
            return values
        if values.shape[0] > size:
            return values[:size]
        return np.concatenate([values, np.zeros(size - values.shape[0])])

    return solution._replace(
        dual_eq=fit(solution.dual_eq, template.dual_eq),
        dual_ineq=fit(solution.dual_ineq, template.dual_ineq),
    )


def _jaddle_duals(highs, mip):
    """HiGHS's row duals in Jaddle's convention: ``(y_eq, y_ineq)``."""
    row_dual = np.asarray(highs.getSolution().row_dual, np.float64)
    return -row_dual[: mip.n_eq], np.maximum(-row_dual[mip.n_eq :], 0.0)


def _exact_strong(
    highs, mip, lower, upper, x, request, bases=None, candidates=None, deadline=None,
    lookahead=None, objective_bound=None,
):
    """
    Strong branching on a node HiGHS has just solved in ``highs`` (box
    ``[lower, upper]``, vertex ``x``, its basis loaded), for
    ``_ExactLane``. ``request`` is ``(is_int, tol, pc_sum, pc_count, k,
    iterations, reliable)``: of the fractional integers whose exact
    pseudocosts (``pc_sum``/``pc_count``) have fewer than ``reliable``
    observations a side, the ``k`` with the largest pseudocost product are
    split, and each side is re-solved from the node's basis for at most
    ``iterations`` dual simplex pivots (``None``: no limit). Dual simplex
    keeps its duals feasible, so a side stopped early (by the pivot or the
    time limit) still has duals whose Lagrangian bound holds (lower than
    the side's LP value). ``candidates`` (variable indices) names the
    ones to split instead, in that order (those fractional at ``x`` with
    room in the box; the pseudocosts are not used). With ``deadline`` (a
    ``time.time()`` value) no side runs past it, and the candidates left
    once it has passed are not split.

    With ``lookahead`` (and no ``candidates``), the candidates after that
    many in a row whose gain product (from the sides' objective values, an
    empty side counting as infinite) did not beat the best so far are not
    split, as SCIP's strong branching looks ahead: they come in order of
    their pseudocost product, so the later ones rarely win. With
    ``objective_bound`` (a value of ``mip.c @ x``: the cutoff), HiGHS stops
    a side once its dual objective passes it: the side is ``"partial"``,
    and its duals' bound closes it.

    Returns ``[(var, down, status, x, y_eq, y_ineq), ...]``, ``status``
    ``"optimal"``, ``"partial"`` (stopped early), ``"infeasible"`` or
    ``"failed"``; ``highs`` is left with the node's bounds and basis. With
    a dict ``bases``, each optimal or partial side's final basis goes in
    it as ``bases[(var, down)]``, to warm-start that child's own LP.
    """
    import highspy

    is_int, tol, pc_sum, pc_count, k, iterations, reliable = request
    frac = x - np.floor(x)
    branchable = is_int & (frac > tol) & (frac < 1.0 - tol) & (lower < upper)
    if candidates is not None:
        chosen = np.asarray(candidates, np.int64)
        chosen = chosen[branchable[chosen]][: max(int(k), 0)]
        if chosen.size == 0:
            return []
    else:
        found = np.flatnonzero(branchable)
        if k <= 0 or found.size == 0:
            return []
        unreliable = found[pc_count[:, found].min(axis=0) < reliable]
        if unreliable.size == 0:
            return []
        down, up = _pseudocost_estimates(pc_sum, pc_count, x, unreliable)
        score = np.maximum(down, 1e-6) * np.maximum(up, 1e-6)
        chosen = unreliable[np.argsort(-score, kind="stable")[:k]]
    MS = highspy.HighsModelStatus
    basis = highs.getBasis()
    _, limit = highs.getOptionValue("simplex_iteration_limit")
    _, time_limit = highs.getOptionValue("time_limit")
    _, bound_option = highs.getOptionValue("objective_bound")
    if iterations is not None:
        highs.setOptionValue("simplex_iteration_limit", int(iterations))
    if objective_bound is not None and np.isfinite(objective_bound):
        highs.setOptionValue("objective_bound", float(objective_bound))
    looking = bool(lookahead) and candidates is None
    node_value = highs.getInfo().objective_function_value if looking else 0.0
    best_score, behind = -np.inf, 0
    out = []
    try:
        for var in chosen.tolist():
            if deadline is not None and time.time() >= deadline:
                break
            if looking and behind >= lookahead:
                break
            lo, up = float(lower[var]), float(upper[var])
            gains = []
            for is_down in (True, False):
                side = (lo, math.floor(x[var])) if is_down else (math.ceil(x[var]), up)
                highs.changeColBounds(var, *side)
                highs.setBasis(basis)
                if deadline is not None:
                    # ``time_limit`` counts the instance's whole run time.
                    highs.setOptionValue(
                        "time_limit",
                        highs.getRunTime() + max(deadline - time.time(), 0.0),
                    )
                highs.run()
                status = highs.getModelStatus()
                if status == MS.kInfeasible:
                    out.append((var, is_down, "infeasible", None, None, None))
                    gains.append(np.inf)
                elif (
                    status
                    in (MS.kOptimal, MS.kIterationLimit, MS.kTimeLimit, MS.kObjectiveBound)
                    and highs.getSolution().dual_valid
                ):
                    if looking:
                        value = highs.getInfo().objective_function_value
                        gains.append(max(value - node_value, 0.0))
                    y_eq, y_ineq = _jaddle_duals(highs, mip)
                    if bases is not None:
                        bases[(var, is_down)] = highs.getBasis()
                    out.append(
                        (
                            var,
                            is_down,
                            "optimal" if status == MS.kOptimal else "partial",
                            np.asarray(highs.getSolution().col_value, np.float64),
                            y_eq,
                            y_ineq,
                        )
                    )
                else:
                    out.append((var, is_down, "failed", None, None, None))
                    gains.append(0.0)
                highs.changeColBounds(var, lo, up)
            if looking and len(gains) == 2:
                score = max(gains[0], 1e-6) * max(gains[1], 1e-6)
                if score > best_score:
                    best_score, behind = score, 0
                else:
                    behind += 1
    finally:
        highs.setOptionValue("simplex_iteration_limit", limit)
        highs.setOptionValue("time_limit", time_limit)
        highs.setOptionValue("objective_bound", bound_option)
        highs.setBasis(basis)
    return out


def _highs_rows(mip):
    """A ``highspy.Highs`` holding ``mip``'s rows (equalities, then ``<=``
    rows) for ``_exact_node_lp``: dual simplex, no presolve (it would drop
    the basis between calls), one thread, silent. The tree cut block's
    explicit zero entries (its padding) are dropped."""
    import highspy

    A = sp.csc_matrix(mip.A, dtype=np.float64, copy=True)
    A.eliminate_zeros()
    A.sort_indices()
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
    lp.col_cost_ = np.asarray(mip.c, np.float64)
    lp.col_lower_ = np.asarray(mip.lower, np.float64)
    lp.col_upper_ = np.asarray(mip.upper, np.float64)
    b = np.asarray(mip.b, np.float64)
    lp.row_lower_ = np.concatenate([b[: mip.n_eq], np.full(b.size - mip.n_eq, -np.inf)])
    lp.row_upper_ = b
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = A.indptr
    lp.a_matrix_.index_ = A.indices
    lp.a_matrix_.value_ = A.data
    highs = highspy.Highs()
    for option, value in (
        ("output_flag", False),
        ("presolve", "off"),
        ("solver", "simplex"),
        ("simplex_strategy", 1),  # dual
        ("parallel", "off"),
    ):
        highs.setOptionValue(option, value)
    highs.passModel(lp)
    return highs


def _highs_submip(mip, is_int):
    """A ``highspy.Highs`` holding ``mip`` as a MIP over the integers
    ``is_int`` for ``_submip_job``: ``_highs_rows``'s rows with HiGHS's
    presolve and MIP solver, one thread, silent. Implied integers stay
    continuous. ``cancelSolve`` stops a run (``_SubMipLane.shutdown``)."""
    import highspy

    highs = _highs_rows(mip)
    cols = np.flatnonzero(is_int).astype(np.int32)
    if cols.size:
        highs.changeColsIntegrality(
            cols.size, cols, np.full(cols.size, highspy.HighsVarType.kInteger)
        )
    highs.setOptionValue("presolve", "on")
    highs.setOptionValue("solver", "choose")
    highs.HandleUserInterrupt = True
    return highs


def _submip_job(highs, mip, lower, upper, cutoff, seconds):
    """Search the MIP in ``highs`` (``_highs_submip``) over the box
    ``[lower, upper]`` for up to ``seconds``, pruned at ``cutoff``
    (HiGHS's ``objective_bound``, in ``mip.c``'s units, offset excluded), so
    only points better than the cutoff come back. Returns HiGHS's best
    point, or ``None`` when it found none."""
    n = lower.size
    highs.changeColsBounds(n, np.arange(n, dtype=np.int32), lower, upper)
    highs.setOptionValue("objective_bound", float(cutoff))
    highs.setOptionValue("time_limit", highs.getRunTime() + float(seconds))
    highs.run()
    if highs.getInfo().primal_solution_status != 2:  # kSolutionStatusFeasible
        return None
    x = np.array(highs.getSolution().col_value, np.float64)
    if float(np.asarray(mip.c, np.float64) @ x) >= cutoff:
        return None
    return x


def _highs_proximity(mip, is_int):
    """``_highs_submip``'s MIP with one more row, ``c·x``, for proximity
    search (``_proximity_job``) to hold under the cutoff."""
    highs = _highs_submip(mip, is_int)
    c = np.asarray(mip.c, np.float64)
    cols = np.flatnonzero(c).astype(np.int32)
    highs.addRow(-np.inf, np.inf, cols.size, cols, c[cols])
    return highs


def _proximity_distance(center, lower, upper, is_int):
    """Proximity search's cost: the L1 distance to ``center`` over the
    integers it has at a bound of ``[lower, upper]`` (all of a binary's),
    as the feasibility pump's. Integers strictly inside their box get no
    cost, as they would need an extra column each."""
    at_lower = is_int & (center <= lower + 0.5)
    at_upper = is_int & ~at_lower & (center >= upper - 0.5)
    return np.where(at_lower, 1.0, np.where(at_upper, -1.0, 0.0))


def _proximity_job(highs, mip, is_int, center, lower, upper, cutoff, seconds):
    """
    Proximity search (Fischetti & Monaci, 2014) in ``highs``
    (``_highs_proximity``): over ``[lower, upper]`` with ``c·x ≤ cutoff``,
    minimise the distance to ``center`` (``_proximity_distance``), and
    stop at HiGHS's first point. Then, with its integers fixed, the
    continuous variables are re-solved for the true objective (the
    distance leaves them anywhere feasible). Returns ``(x, empty)``: the
    point (``None`` when none was found) and whether HiGHS showed the box
    holds no point under the cutoff.
    """
    import highspy

    n = lower.size
    cols = np.arange(n, dtype=np.int32)
    row = highs.getNumRow() - 1
    c = np.asarray(mip.c, np.float64)
    deadline = highs.getRunTime() + float(seconds)
    highs.changeColsBounds(n, cols, lower, upper)
    highs.changeColsCost(n, cols, _proximity_distance(center, lower, upper, is_int))
    highs.changeRowBounds(row, -np.inf, float(cutoff))
    highs.setOptionValue("objective_bound", np.inf)
    highs.setOptionValue("mip_max_improving_sols", 1)
    highs.setOptionValue("time_limit", deadline)
    highs.run()
    status = highs.getModelStatus()
    if highs.getInfo().primal_solution_status != 2:  # kSolutionStatusFeasible
        return None, status == highspy.HighsModelStatus.kInfeasible
    x = np.array(highs.getSolution().col_value, np.float64)
    # Re-solve the continuous variables with the integers fixed.
    fixed_lower, fixed_upper = lower.copy(), upper.copy()
    fixed_lower[is_int] = fixed_upper[is_int] = np.round(x[is_int])
    highs.changeColsBounds(n, cols, fixed_lower, fixed_upper)
    highs.changeColsCost(n, cols, c)
    highs.changeRowBounds(row, -np.inf, np.inf)
    highs.setOptionValue("mip_max_improving_sols", 2147483647)
    highs.setOptionValue("time_limit", max(deadline, highs.getRunTime() + 1.0))
    highs.run()
    if highs.getInfo().primal_solution_status == 2:
        polished = np.array(highs.getSolution().col_value, np.float64)
        if float(c @ polished) <= float(c @ x):
            x = polished
    if float(c @ x) >= cutoff:
        return None, False
    return x, False


# The solutions crossover draws from (``_BranchAndBound.solution_pool``).
_SOLUTION_POOL_SIZE = 10

# RINS's fixing rate on the sub-MIP lane (``adapt_rins_fixing``): the factor
# it moves by, its floor, and the share of a job's seconds under which a job
# that found nothing counts as having ended early.
_RINS_FIXING_STEP = 0.8
_RINS_MIN_FIXING = 0.4
_RINS_EARLY_SHARE = 0.5


class _SubMipLane:
    """
    HiGHS sub-MIPs on a CPU thread (``submip_solver="highs"``): RINS's and
    crossover's neighbourhoods, a box with most integers fixed, searched by
    HiGHS's branch and cut while the device keeps solving node LPs. Jaddle's
    own sub-search solves its node LPs one at a time by PDHG, the slowest
    way to use it; HiGHS's dual simplex is at home on a small sub-MIP. One
    job at a time; the thread keeps one HiGHS instance for the rows it last
    searched (``self.base`` changes after cut rounds and restarts). HiGHS
    releases the GIL while it solves. Jobs are functions of NumPy and SciPy
    arrays only.
    """

    def __init__(self):
        from concurrent.futures import ThreadPoolExecutor

        self.pool = ThreadPoolExecutor(1, thread_name_prefix="jaddle-submip")
        self.held = None  # (mip, Highs), used on the lane's thread only
        # Proximity search's own copy, with the objective row added.
        self.held_proximity = None
        self.job = None  # (future, kind)

    def busy(self):
        return self.job is not None

    def submit(self, kind, mip, is_int, lower, upper, cutoff, seconds, options=None):
        """Queue ``kind``'s (``"rins"``, ``"crossover"``, ``"rens"`` or
        ``"first"``) search of ``mip`` over ``[lower, upper]``
        (``_submip_job``), with HiGHS ``options`` for this job only."""
        lower, upper = np.array(lower, np.float64), np.array(upper, np.float64)
        is_int = np.array(is_int, bool)
        options = dict(options or {})

        def job():
            started = time.time()
            if self.held is None or self.held[0] is not mip:
                self.held = (mip, _highs_submip(mip, is_int))
            highs = self.held[1]
            saved = {key: highs.getOptionValue(key)[1] for key in options}
            try:
                for key, value in options.items():
                    highs.setOptionValue(key, value)
                x = _submip_job(highs, mip, lower, upper, cutoff, seconds)
            finally:
                for key, value in saved.items():
                    highs.setOptionValue(key, value)
            return x, False, time.time() - started

        self.job = (self.pool.submit(job), kind)

    def submit_proximity(self, mip, is_int, center, lower, upper, cutoff, seconds):
        """Queue proximity search around ``center`` (``_proximity_job``)."""
        lower, upper = np.array(lower, np.float64), np.array(upper, np.float64)
        is_int = np.array(is_int, bool)
        center = np.array(center, np.float64)

        def job():
            started = time.time()
            if self.held_proximity is None or self.held_proximity[0] is not mip:
                self.held_proximity = (mip, _highs_proximity(mip, is_int))
            x, empty = _proximity_job(
                self.held_proximity[1], mip, is_int, center, lower, upper, cutoff, seconds
            )
            return x, empty, time.time() - started

        self.job = (self.pool.submit(job), "proximity")

    def finished(self):
        """The finished job as ``(kind, x, empty, seconds)`` (``x`` ``None``
        when it found nothing or raised; ``empty``: HiGHS showed the box
        holds nothing under the cutoff, proximity only), or ``None`` while
        it runs or with no job."""
        if self.job is None or not self.job[0].done():
            return None
        future, kind = self.job
        self.job = None
        try:
            x, empty, seconds = future.result()
        except Exception:
            x, empty, seconds = None, False, 0.0
        return kind, x, empty, seconds

    def shutdown(self):
        """Stop: a running HiGHS search is cancelled, a queued one dropped."""
        if self.job is not None:
            self.job[0].cancel()
            self.job = None
        for held in (self.held, self.held_proximity):
            if held is not None:
                try:
                    held[1].cancelSolve()
                except Exception:
                    pass
        self.pool.shutdown(wait=False, cancel_futures=True)


# Objective levels (``objective_levels``): the most integer columns the
# objective may rest on, and the most assignments of them the box may hold.
_LEVEL_MAX_COLUMNS = 3
_LEVEL_MAX_LEVELS = 10_000


def objective_level_table(
    c,
    integrality,
    lower,
    upper,
    max_columns=_LEVEL_MAX_COLUMNS,
    max_levels=_LEVEL_MAX_LEVELS,
):
    """
    The objective's levels: when every nonzero cost is on one of at most
    ``max_columns`` integer columns with finite bounds, and the box holds
    at most ``max_levels`` (and at least two) assignments of them, returns
    ``(columns, values, levels)``: each assignment as a row of ``values``
    (``(k, len(columns))``), sorted by its objective value ``levels =
    values @ c[columns]``. ``None`` otherwise.

    Every solution lies in exactly one level, and its objective value is
    that level's. A level shown empty can be dropped from the bound, so the
    bound is the lowest level not yet shown empty. On 30n20b8
    (``100·R1 + 51·R2``) the LP stays at 151 deep in the tree, while HiGHS
    shows each level below the optimum, 302, empty in at most 2 s.
    """
    c = np.asarray(c, np.float64)
    columns = np.flatnonzero(c)
    is_int = np.asarray(integrality, bool)
    if columns.size == 0 or columns.size > max_columns or not is_int[columns].all():
        return None
    lo = np.asarray(lower, np.float64)[columns]
    up = np.asarray(upper, np.float64)[columns]
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(up))) or np.any(lo > up):
        return None
    count = float(np.prod(up - lo + 1.0))
    if count < 2 or count > max_levels:
        return None
    grids = np.meshgrid(*[np.arange(a, b + 1.0) for a, b in zip(lo, up)], indexing="ij")
    values = np.stack([grid.ravel() for grid in grids], axis=1)
    levels = values @ c[columns]
    order = np.argsort(levels, kind="stable")
    return columns, values[order], levels[order]


class _LevelLane:
    """
    The objective's levels (``objective_levels``, ``objective_level_table``)
    searched by HiGHS on a CPU thread of their own, lowest first (from the
    search's bound up): the box the search gives it (``update``) with the
    level's columns fixed, for
    ``seconds`` at first and twice as long at each new try. A level HiGHS
    shows infeasible is ``empty``; a point it finds waits in ``found`` for
    the search to check. The thread works on the lowest level that may still
    hold a solution better than the cutoff, so the bound (``floor``) rises
    level by level, and a feasible level found that way is optimal.

    The thread keeps one HiGHS instance for the rows it last searched (the
    search's base MIP changes at restarts). HiGHS releases the GIL while it
    solves. The search reads ``empty`` and ``found``; the thread writes
    them.
    """

    def __init__(self, columns, values, levels, seconds):
        self.columns, self.values, self.levels = columns, values, levels
        self.seconds = float(seconds)
        self.empty = np.zeros(levels.size, bool)  # shown infeasible
        self.settled = np.zeros(levels.size, bool)  # empty, or a point found
        self.tries = np.zeros(levels.size, np.int64)
        self.proved = 0  # levels shown empty
        self.jobs = 0
        self.busy_seconds = 0.0
        self.best_found = np.inf  # the lowest level with a point found
        self.found = collections.deque()  # (level, x) for the search to check
        self.box = None  # (mip, is_int, lower, upper), set by the search
        self.cutoff = np.inf
        self.known = -np.inf  # the search's own bound: lower levels are empty
        self.deadline = None
        self.stop = threading.Event()
        self.thread = None
        self.highs = None

    def update(self, mip, is_int, lower, upper, cutoff, deadline, known=-np.inf):
        """The search's base MIP, root box, cutoff, deadline and bound
        (``known``: the levels below it hold no solution, so the thread
        skips them); starts the thread on the first call."""
        self.box = (mip, is_int, lower, upper)
        self.cutoff, self.deadline = float(cutoff), deadline
        self.known = float(known)
        if self.thread is None:
            self.thread = threading.Thread(
                target=self._run, name="jaddle-levels", daemon=True
            )
            self.thread.start()

    def open_levels(self, cutoff, lower, upper):
        """The levels that may still hold a solution better than
        ``cutoff`` in the box ``[lower, upper]``."""
        cols = self.columns
        inside = np.all(
            (self.values >= lower[cols] - 1e-9) & (self.values <= upper[cols] + 1e-9),
            axis=1,
        )
        return ~self.empty & (self.levels < cutoff) & inside

    def floor(self, cutoff, lower, upper):
        """A bound on every solution better than ``cutoff`` in the box: the
        lowest open level (``inf`` when none is open)."""
        open_ = self.open_levels(cutoff, lower, upper)
        i = int(np.argmax(open_))
        return float(self.levels[i]) if open_[i] else np.inf

    def _run(self):
        import highspy

        infeasible = highspy.HighsModelStatus.kInfeasible
        held = None
        while not self.stop.is_set():
            mip, is_int, lower, upper = self.box
            cutoff = min(self.cutoff, self.best_found)
            pick = self.open_levels(cutoff, lower, upper) & ~self.settled
            pick &= self.levels >= self.known
            seconds = 0.0
            if pick.any():
                i = int(np.argmax(pick))
                seconds = self.seconds * 2.0 ** int(self.tries[i])
                if self.deadline is not None:
                    seconds = min(seconds, self.deadline - time.time())
            if seconds <= 0.0:
                self.stop.wait(0.1)  # nothing to search until the search moves
                continue
            if held is None or held[0] is not mip:
                held = (mip, _highs_submip(mip, is_int))
                self.highs = held[1]
            lo = np.array(lower, np.float64)
            up = np.array(upper, np.float64)
            lo[self.columns] = up[self.columns] = self.values[i]
            started = time.time()
            try:
                x = _submip_job(held[1], mip, lo, up, np.inf, seconds)
                status = held[1].getModelStatus()
            except Exception:
                x, status = None, None
            self.busy_seconds += time.time() - started
            self.jobs += 1
            if self.stop.is_set():
                break
            if x is not None:
                self.settled[i] = True
                self.best_found = min(self.best_found, float(self.levels[i]))
                self.found.append((i, x))
            elif status == infeasible:
                self.empty[i] = self.settled[i] = True
                self.proved += 1
            else:
                self.tries[i] += 1

    def shutdown(self):
        """Stop the thread; a running HiGHS search is cancelled."""
        self.stop.set()
        if self.highs is not None:
            try:
                self.highs.cancelSolve()
            except Exception:
                pass


def exact_lane_bounds(mip, terms, box, out, children):
    """The bounds the search's ``exact_bound`` gives a lane job's node and
    its strong sides, computed on the lane's thread: ``{"node": (bound,
    rigorous), (var, down): (bound, rigorous)}`` over ``mip``'s rows
    (``terms``: their ``A_T``, the integrality and the reduced-cost
    tolerance), the node's ``box`` and each side's box (``box`` with
    ``var`` rounded down or up from the node's vertex, as
    ``exact_strong_children`` builds it). Only sides HiGHS solved
    (``"optimal"``, ``"partial"``) are bounded."""
    A_T, is_int, tol = terms
    lower, upper = box
    _, x, y_eq, y_ineq = out
    x = np.asarray(x, np.float64)
    bounds = {"node": _exact_side_bound(mip, A_T, y_eq, y_ineq, lower, upper, is_int, tol)}
    for var, down, status, _, cy_eq, cy_ineq in children:
        if status not in ("optimal", "partial"):
            continue
        lo, up = lower.copy(), upper.copy()
        if down:
            up[var] = min(up[var], math.floor(x[var]))
        else:
            lo[var] = max(lo[var], math.ceil(x[var]))
        if lo[var] > up[var]:
            continue
        bounds[(var, down)] = _exact_side_bound(mip, A_T, cy_eq, cy_ineq, lo, up, is_int, tol)
    return bounds


class _ExactLane:
    """
    Exact node LPs on CPU threads (``exact_nodes``): HiGHS's dual simplex
    solves the LPs of the nodes that hold the global bound while the device
    solves the rest in batches. A truncated PDHG node LP gives a valid
    Lagrangian bound, but often one no higher than its parent's, so the
    nodes at the bottom of the queue never lift the global bound (map06:
    7,800 nodes moved it 3 units). An exact LP's duals give the node its
    LP bound, its children their bounds over their boxes, and a vertex to
    branch on.

    Each thread keeps one HiGHS instance for the rows it last solved (a tree
    cut refresh changes them). A node re-solves from the basis it is given
    (its parent's optimal one, ``exact_basis_memory``), else from the
    thread's last basis, usually an unrelated node's: on presolved map06 a
    child took 0.5–1.6 s from its parent's basis (870–3,100 pivots) against
    1.8–2.9 s (3,400–5,200) from another node's; the root LP takes 3 s.
    HiGHS releases the GIL while it solves. Jobs are functions of NumPy and
    SciPy arrays only (JAX's settings are per thread).
    """

    def __init__(self, threads, seconds):
        from concurrent.futures import ThreadPoolExecutor

        self.threads = max(1, int(threads))
        self.seconds = float(seconds)
        self.pool = ThreadPoolExecutor(self.threads, thread_name_prefix="jaddle-exact")
        self.local = threading.local()  # each thread's (mip, Highs)
        # Queued and running jobs, in order: (future, node, box, cut version).
        self.jobs = []

    def room(self):
        """Whether another node may be queued: up to two a thread, so a
        thread never waits for the search to hand it the next."""
        return len(self.jobs) < 2 * self.threads

    def nodes(self):
        return [job[1] for job in self.jobs]

    def submit(
        self, node, box, lp_box, mip, version, strong=None, basis=None, keep=False,
        bound_terms=None, strong_options=None,
    ):
        """Queue ``node``'s LP over ``lp_box`` (the LP's bounds over its box
        ``box``) for ``mip``'s rows as of cut block ``version``, from
        ``basis`` (a ``highspy.HighsBasis`` for ``mip``'s rows) when given,
        then, with a ``strong`` request, strong branching on it
        (``_exact_strong``, with ``strong_options``: its ``lookahead`` and
        ``objective_bound``). With ``keep``, the job also returns the node's
        optimal basis and its strong sides' bases, for the children. With
        ``bound_terms`` (``exact_lane_bounds``'s ``terms``), it also bounds
        the node and its sides on the thread (``extra["bounds"]``)."""
        lower, upper = (np.array(b, np.float64) for b in lp_box)
        node_box = (np.array(box[0], np.float64), np.array(box[1], np.float64))

        def job():
            started = time.time()
            highs = self._held(mip)
            out = _exact_node_lp(highs, mip, lower, upper, self.seconds, basis)
            extra = dict(
                pivots=int(highs.getInfo().simplex_iteration_count),
                warm=basis is not None,
                mip=mip,
                basis=None,
                sides={},
            )
            if keep and out[0] == "optimal":
                extra["basis"] = highs.getBasis()
            children, strong_seconds = [], 0.0
            if strong is not None and out[0] == "optimal":
                strong_started = time.time()
                children = _exact_strong(
                    highs,
                    mip,
                    lower,
                    upper,
                    out[1],
                    strong,
                    extra["sides"] if keep else None,
                    **(strong_options or {}),
                )
                strong_seconds = time.time() - strong_started
            if bound_terms is not None and out[0] == "optimal":
                extra["bounds"] = exact_lane_bounds(
                    mip, bound_terms, node_box, out, children
                )
            return out, children, strong_seconds, time.time() - started, extra

        self.jobs.append((self.pool.submit(job), node, box, version))

    def _held(self, mip):
        """This thread's HiGHS instance for ``mip``'s rows (built when the
        rows changed since its last job)."""
        held = getattr(self.local, "held", None)
        if held is None or held[0] is not mip:
            held = self.local.held = (mip, _highs_rows(mip))
        return held[1]

    def root_lp(self, mip, lp_box, seconds):
        """The root LP over ``lp_box`` on one of the threads, waited for
        (the lane is idle at the root): ``((status, x, y_eq, y_ineq),
        basis, pivots)``, ``basis`` the optimal one (else ``None``)."""
        lower, upper = (np.array(b, np.float64) for b in lp_box)

        def job():
            highs = self._held(mip)
            out = _exact_node_lp(highs, mip, lower, upper, seconds)
            pivots = int(highs.getInfo().simplex_iteration_count)
            return out, highs.getBasis() if out[0] == "optimal" else None, pivots

        try:
            return self.pool.submit(job).result()
        except Exception:
            return ("failed", None, None, None), None, 0

    def split(self, mip, lp_box, x, basis, chunks, request, deadline):
        """Strong branching on the root (vertex ``x``, optimal ``basis``,
        box ``lp_box``) over all the threads at once: one job a chunk of
        candidates (``_exact_strong(candidates=chunk)``), each from
        ``basis`` on its thread's HiGHS, until ``deadline``, waited for.
        Returns ``(children, bases, seconds)``: ``_exact_strong``'s lists
        joined, the sides' bases by ``(var, down)`` and the threads' time."""
        lower, upper = (np.array(b, np.float64) for b in lp_box)
        n = lower.size

        def job(chunk):
            started = time.time()
            highs = self._held(mip)
            highs.changeColsBounds(n, np.arange(n, dtype=np.int32), lower, upper)
            highs.setBasis(basis)
            bases = {}
            children = _exact_strong(
                highs, mip, lower, upper, x, request, bases,
                candidates=chunk, deadline=deadline,
            )
            return children, bases, time.time() - started

        futures = [self.pool.submit(job, chunk) for chunk in chunks if len(chunk)]
        children, bases, seconds = [], {}, 0.0
        for future in futures:
            try:
                part, part_bases, part_seconds = future.result()
            except Exception:
                continue
            children += part
            bases.update(part_bases)
            seconds += part_seconds
        return children, bases, seconds

    def finished(self):
        """The jobs done since the last call, as ``(node, box, version,
        (status, x, y_eq, y_ineq), children, strong_seconds, seconds,
        extra)`` (``children`` from ``_exact_strong``; ``extra``: the node
        LP's ``pivots``, whether it started from a given basis (``warm``),
        the ``mip`` whose rows it solved and, kept with
        ``submit(keep=True)``, its optimal ``basis`` and its strong sides'
        bases (``sides``, by ``(var, down)``)). A job that
        raised comes back ``"failed"``."""
        out, waiting = [], []
        for future, node, box, version in self.jobs:
            if not future.done():
                waiting.append((future, node, box, version))
                continue
            try:
                result, children, strong_seconds, seconds, extra = future.result()
            except Exception:
                result, children, strong_seconds, seconds, extra = (
                    ("failed", None, None, None),
                    [],
                    0.0,
                    0.0,
                    dict(pivots=0, warm=False, mip=None, basis=None, sides={}),
                )
            out.append(
                (node, box, version, result, children, strong_seconds, seconds, extra)
            )
        self.jobs = waiting
        return out

    def wait(self, timeout):
        """Block until a job finishes or ``timeout`` seconds pass."""
        from concurrent.futures import FIRST_COMPLETED, wait

        if self.jobs:
            wait([job[0] for job in self.jobs], timeout, FIRST_COMPLETED)

    def drop(self):
        """Forget every job (queued ones are cancelled; running ones finish
        on their own) and return their nodes, unsolved."""
        nodes = []
        for future, node, _, _ in self.jobs:
            future.cancel()
            nodes.append(node)
        self.jobs = []
        return nodes

    def shutdown(self):
        self.drop()
        self.pool.shutdown(wait=False, cancel_futures=True)


# Cutoff probing (``cutoff_probing``): the fixes a thread finds before it
# re-solves its LP, for a fresh vertex and basis.
_PROBE_REFRESH_FIXES = 32
# Another round at the same cutoff while the last fixed at least this share
# of the integers it probed (its fixes tighten the next round's LP).
_PROBE_AGAIN_SHARE = 0.05
# A round at a new cutoff waits until the room it has to close sides in
# (the cutoff less the global bound) is at most this share of the last
# round's, unless that round fixed ``_PROBE_AGAIN_SHARE``. On eil33-2 each
# feasibility jump point started a 28 s round that fixed 41 of 4,516
# binaries (cutoff 1,924, bound 820), against 4,046 at the optimum's.
_PROBE_ROOM_SHRINK = 0.75
# A restart presolve (``restart_presolve``) goes on in the reduced MIP only
# when it removed at least this share of the columns.
_RESTART_PRESOLVE_MIN_SHARE = 0.1
# RENS around a node point (``tree_rens``) needs at least this share of the
# integers integral there (the root's needs 0.3 from a vertex, 0.5 else).
_TREE_RENS_MIN_FIXED = 0.3


def _cutoff_probe(highs, mip, lower, upper, cols, cutoff, terms, iterations, deadline):
    """
    Cutoff probing on one thread (``_ProbeLane``): the LP of ``mip``'s rows
    (``highs``: ``_highs_rows``) over ``[lower, upper]``, then for each
    integer of ``cols`` in turn, each side its vertex is not on (``x_j >=
    l_j + 1`` when ``x_j < l_j + 1``, ``x_j <= u_j − 1`` when ``x_j > u_j −
    1``; for a binary at 0 the side ``x_j = 1``), re-solved from the LP's
    basis by dual simplex for at most ``iterations`` pivots, stopped once
    its dual objective passes ``cutoff`` (HiGHS's ``objective_bound``). A
    side HiGHS shows infeasible, or whose duals' Lagrangian bound over the
    side's box (``_exact_side_bound`` with ``terms``: the rows' ``A_T``,
    the integrality and the reduced-cost tolerance) is rigorous and above
    ``cutoff``, holds no solution with ``c·x <= cutoff``, so the integer
    takes the other side's value: fixed. Fixes stay in the thread's LP for
    the rest of ``cols`` (they hold for every solution searched for), and
    every ``_PROBE_REFRESH_FIXES`` of them the LP is re-solved for a new
    vertex and basis. Stops at ``deadline`` (a ``time.time()`` value).

    Returns ``dict(status, upper, lower, probed, lps, pivots)``:
    ``status`` ``"done"``, ``"empty"`` (the box holds nothing at or below
    ``cutoff``), ``"failed"`` (the LP did not solve) or ``"time"``;
    ``upper``/``lower``: ``(columns, values)`` of the new upper bounds (up
    side closed) and lower bounds (down side closed); ``probed``: the
    columns of ``cols`` whose sides were all tried.
    """
    import highspy

    MS = highspy.HighsModelStatus
    A_T, is_int, tol = terms
    n = lower.size
    lower = np.array(lower, np.float64)
    upper = np.array(upper, np.float64)
    out = dict(status="done", upper=([], []), lower=([], []), probed=0, lps=0, pivots=0)
    saved = {
        option: highs.getOptionValue(option)[1]
        for option in ("objective_bound", "simplex_iteration_limit", "time_limit")
    }
    _, unlimited = highs.getOptionValue("simplex_iteration_limit")

    def solve(limit):
        highs.setOptionValue("simplex_iteration_limit", limit)
        highs.setOptionValue(
            "time_limit", highs.getRunTime() + max(deadline - time.time(), 0.0)
        )
        highs.run()
        out["pivots"] += int(highs.getInfo().simplex_iteration_count)
        return highs.getModelStatus()

    def lp():
        """The LP over the current box: ``(x, basis)``, or ``None`` with
        ``out["status"]`` set."""
        highs.changeColsBounds(n, np.arange(n, dtype=np.int32), lower, upper)
        status = solve(unlimited)
        if status in (MS.kInfeasible, MS.kObjectiveBound):
            out["status"] = "empty"
            return None
        if status != MS.kOptimal:
            out["status"] = "time" if time.time() >= deadline else "failed"
            return None
        return np.asarray(highs.getSolution().col_value, np.float64), highs.getBasis()

    try:
        highs.setOptionValue("objective_bound", float(cutoff))
        state = lp()
        if state is None:
            return out
        x, basis = state
        fresh = 0
        side_iterations = unlimited if iterations is None else int(iterations)
        for j in np.asarray(cols, np.int64).tolist():
            if time.time() >= deadline:
                out["status"] = "time"
                break
            lo, up = float(lower[j]), float(upper[j])
            if lo >= up:
                out["probed"] += 1
                continue
            closed = []
            for is_up in (True, False):
                if is_up:
                    if not (np.isfinite(lo) and x[j] < lo + 1.0 - 1e-9):
                        continue
                    side = (lo + 1.0, up)
                else:
                    if not (np.isfinite(up) and x[j] > up - 1.0 + 1e-9):
                        continue
                    side = (lo, up - 1.0)
                highs.changeColBounds(j, *side)
                highs.setBasis(basis)
                status = solve(side_iterations)
                out["lps"] += 1
                shut = status == MS.kInfeasible
                if (
                    not shut
                    and status
                    in (MS.kOptimal, MS.kIterationLimit, MS.kTimeLimit, MS.kObjectiveBound)
                    and highs.getSolution().dual_valid
                ):
                    side_lower, side_upper = lower.copy(), upper.copy()
                    side_lower[j], side_upper[j] = side
                    y_eq, y_ineq = _jaddle_duals(highs, mip)
                    bounded = _exact_side_bound(
                        mip, A_T, y_eq, y_ineq, side_lower, side_upper, is_int, tol
                    )
                    shut = bounded is not None and bounded[1] and bounded[0] > cutoff
                highs.changeColBounds(j, lo, up)
                if shut:
                    closed.append(is_up)
            if time.time() >= deadline and not closed:
                # The sides may have stopped on the deadline unproved.
                out["status"] = "time"
                break
            out["probed"] += 1
            if not closed:
                continue
            if len(closed) == 2:
                out["status"] = "empty"
                break
            if closed[0]:  # x_j >= lo + 1 holds nothing: x_j = lo
                upper[j] = lo
                out["upper"][0].append(j)
                out["upper"][1].append(lo)
            else:  # x_j <= up − 1 holds nothing: x_j = up
                lower[j] = up
                out["lower"][0].append(j)
                out["lower"][1].append(up)
            highs.changeColBounds(j, lower[j], upper[j])
            fresh += 1
            if fresh >= _PROBE_REFRESH_FIXES:
                fresh = 0
                state = lp()
                if state is None:
                    break
                x, basis = state
    finally:
        for option, value in saved.items():
            highs.setOptionValue(option, value)
    return out


class _ProbeLane:
    """
    Cutoff probing (``cutoff_probing``) on CPU threads: once there is an
    incumbent, every free integer of the root box is tried on the side of
    it the root LP's vertex is not on (a binary at 0: ``x_j = 1``), by
    HiGHS's dual simplex over the search's rows, stopped at the cutoff
    (``_cutoff_probe``). A side with no LP point below the cutoff holds no
    better solution, so the integer is fixed in the root box for the rest
    of the search. It is strong branching on every integer, not only the
    fractional ones, with the cutoff as the only question asked. Reduced-
    cost fixing estimates each side from one set of duals; probing solves
    it. On eil33-2 (4,516 binaries, 32 rows, optimum 934.008) the root's
    reduced costs fix 1,005 binaries at the optimum's cutoff, probing
    4,014 and a second round 382 more; HiGHS then shows the 120 binaries
    left hold nothing better in 1 s, against 96 s for the whole MIP.

    A round deals the candidates out to the threads, one job each, and is
    collected once every job is done. Each thread keeps a HiGHS instance
    for the rows it last probed. Jobs are functions of NumPy and SciPy
    arrays only.
    """

    def __init__(self, threads):
        from concurrent.futures import ThreadPoolExecutor

        self.threads = max(1, int(threads))
        self.pool = ThreadPoolExecutor(self.threads, thread_name_prefix="jaddle-probe")
        self.local = threading.local()
        self.jobs = []
        self.started = None

    def busy(self):
        return bool(self.jobs)

    def submit(self, mip, lp_box, cols, cutoff, terms, iterations, deadline):
        """Start a round: ``cols`` (in priority order) dealt out to the
        threads, each probing ``mip``'s LP over ``lp_box`` against
        ``cutoff`` until ``deadline``."""
        lower, upper = (np.array(b, np.float64) for b in lp_box)
        cols = np.asarray(cols, np.int64)

        def job(chunk):
            held = getattr(self.local, "held", None)
            if held is None or held[0] is not mip:
                held = self.local.held = (mip, _highs_rows(mip))
            return _cutoff_probe(
                held[1], mip, lower, upper, chunk, cutoff, terms, iterations, deadline
            )

        self.started = time.time()
        self.jobs = [
            self.pool.submit(job, cols[i :: self.threads])
            for i in range(min(self.threads, cols.size))
        ]

    def finished(self):
        """The round's job results and its wall time once every job is
        done (a job that raised comes back ``"failed"``), else ``None``."""
        if not self.jobs or not all(job.done() for job in self.jobs):
            return None
        results = []
        for job in self.jobs:
            try:
                results.append(job.result())
            except Exception:
                results.append(
                    dict(status="failed", upper=([], []), lower=([], []),
                         probed=0, lps=0, pivots=0)
                )
        self.jobs = []
        return results, time.time() - self.started

    def shutdown(self):
        for job in self.jobs:
            job.cancel()
        self.jobs = []
        self.pool.shutdown(wait=False, cancel_futures=True)


class _HeuristicWorkers:
    """Background threads for the CPU heuristics (``parallel_heuristics``),
    so they run while the main thread keeps the device busy with node LPs.

    Two lanes, each a thread running one job at a time in the order given:
    ``"jump"``, the numba feasibility jump (the root's first jumps, then the
    improving jump), and ``"round"``, fix-and-propagate roundings of the
    root LP's, cut rounds' and node LPs' points (``parallel_rounding``). A job is a function of NumPy arrays only, never
    of JAX arrays: JAX's precision settings are per thread, and the search
    may be inside ``local_settings``. numba releases the GIL while it
    works. The search collects finished jobs with ``finished`` and checks
    their points itself."""

    LANES = ("jump", "round")

    def __init__(self, jump_lanes=1):
        from concurrent.futures import ThreadPoolExecutor

        # ``jump_lanes`` jump threads (``jump_threads``): ``"jump"``, then
        # ``"jump1"``, ``"jump2"``, ...
        self.jump_lanes = ("jump",) + tuple(
            f"jump{i}" for i in range(1, max(1, int(jump_lanes)))
        )
        self.LANES = self.jump_lanes + ("round",)
        self.pools = {
            lane: ThreadPoolExecutor(1, thread_name_prefix=f"jaddle-{lane}")
            for lane in self.LANES
        }
        self.jobs = {lane: [] for lane in self.LANES}
        self.stop = threading.Event()  # set when the search ends

    def idle(self, lane):
        """Whether ``lane`` has no job running or waiting."""
        return not self.jobs[lane]

    def submit(self, lane, kind, fn, *args, info=None, **kwargs):
        """Queue ``fn(*args, stop=..., **kwargs)`` on ``lane``. ``kind`` and
        ``info`` come back with its result."""

        def job():
            started = time.time()
            out = fn(*args, stop=self.stop, **kwargs)
            return out, time.time() - started

        self.jobs[lane].append((kind, self.pools[lane].submit(job), info))

    def finished(self):
        """The jobs done since the last call, as ``(kind, result, seconds,
        info)`` in the order they were queued per lane (``seconds``: the
        job's own run time). A job that raised re-raises here, on the main
        thread."""
        out = []
        for lane in self.LANES:
            jobs = self.jobs[lane]
            while jobs and jobs[0][1].done():
                kind, future, info = jobs.pop(0)
                result, seconds = future.result()
                out.append((kind, result, seconds, info))
        return out

    def shutdown(self):
        """Stop every job (HiGHS within about a second, the jump within a
        chunk) and wait for the threads. Results not yet collected are
        dropped."""
        self.stop.set()
        for lane in self.LANES:
            for _, future, _ in self.jobs[lane]:
                future.cancel()
            self.pools[lane].shutdown(wait=True)
            self.jobs[lane] = []


def _node_heuristic_job(
    x, lower, upper, integrality, propagator, jump=None, stop=None
):
    """A node LP point's background heuristics (``_HeuristicWorkers``):
    fix-and-propagate, and if that fails and ``jump`` is given (keyword
    arguments for ``feasibility_jump``: the MIP, ``seconds``, ...), an
    improving feasibility jump from the point for ``jump["seconds"]``.
    Returns ``(point, found, source)``, ``source`` the log's mark (``"P"``
    or ``"J"``)."""
    point = fix_and_propagate(x, lower, upper, integrality, propagator)
    if point is not None or not jump:
        return point, point is not None, "P"
    jump = dict(jump)
    mip = jump.pop("mip")
    deadline = time.time() + jump.pop("seconds")
    if jump.get("deadline") is not None:
        deadline = min(deadline, jump["deadline"])
    jump["deadline"] = deadline
    point, found, _ = feasibility_jump(
        mip, x, lower, upper, stop=stop, backend="numba", improve=True, **jump
    )
    return point, bool(found), "J"


def _jump_job(skip, *args, stop=None, **kwargs):
    """``feasibility_jump`` as a background job, unless ``skip()`` holds
    when it starts (then ``(None, False, None)``)."""
    if skip():
        return None, False, None
    return feasibility_jump(*args, stop=stop, **kwargs)


class _EitherEvent:
    """Set when either of two ``threading.Event``s is (``None`` never is)."""

    def __init__(self, first, second):
        self.events = [e for e in (first, second) if e is not None]

    def is_set(self):
        return any(e.is_set() for e in self.events)


def _early_jump_job(skip, done, *args, stop=None, **kwargs):
    """A root jump started before the root LP (``root_jump_seconds``): as
    ``_jump_job``, and it also stops (between chunks) once ``done`` is set,
    when the search has its first incumbent, so the lanes free up for the
    improving jump."""
    if skip():
        return None, False, None
    return feasibility_jump(*args, stop=_EitherEvent(stop, done), **kwargs)


# Background jobs' kinds (``_HeuristicWorkers.submit``) and the source
# their incumbents are marked with.
_JOB_SOURCES = {"jump": "J", "jumprestart": "J", "fixprop": "P"}

# ``batch_size="auto"`` (``auto_batch_size``): nodes a batch on a CPU; at
# most this many on a GPU or TPU (and this many when the device reports no
# memory); the share of the device's free memory the batches may take; and
# the per-member memory envelope: node LP words (of the node precision) per
# variable or row and per nonzero, and children-kernel bytes per padded
# nonzero.
_AUTO_BATCH_CPU = 32
_AUTO_BATCH_ACCELERATOR = 128
_AUTO_BATCH_MEMORY_SHARE = 0.75
_AUTO_BATCH_LP_WORDS = (16, 4)
_AUTO_BATCH_CHILD_BYTES = 64


def auto_batch_size(
    n,
    m,
    nnz,
    free_bytes,
    itemsize=4,
    kernel_nnz=None,
    children=lambda batch: False,
    largest=_AUTO_BATCH_ACCELERATOR,
):
    """
    The largest power of two, at most ``largest``, whose batch of node
    LPs fits in ``_AUTO_BATCH_MEMORY_SHARE`` of ``free_bytes`` of device
    memory (at least 1): ``solve(batch_size="auto")`` on a GPU or TPU.

    A member costs ``16·(n + m) + 4·nnz`` words of ``itemsize`` bytes (the
    rows and nonzeros of the node LP, tree cut block included), and, when
    ``children(batch)`` says the batch would build its children on the
    device (``_ChildKernels``, in float64), ``64·kernel_nnz`` bytes more.
    These are envelopes over peak memory measured on a 12 GB GPU across
    batch sizes (pipelined batches included): a member's node LPs took
    0.19 MB on 30n20b8, 2.7 MB on academictimetablesmall, 3.0 MB on bab5
    and 117 MB on neos-5114902-kasavu (1.3–2.5 times under the envelope),
    and its children 1.2–4 float64 words per padded nonzero a box (bab5:
    2 GB at 128 a batch). The share leaves room for the rest of the
    search (heuristics, strong branching, the root's own peak).

    ``largest`` caps the batch: a short batch is padded to full size, so a
    batch larger than the open nodes costs device time for nothing.
    """
    budget = _AUTO_BATCH_MEMORY_SHARE * max(float(free_bytes), 0.0)
    per_lp, per_nnz = _AUTO_BATCH_LP_WORDS
    member = itemsize * (per_lp * (n + m) + per_nnz * nnz)
    kernel = _AUTO_BATCH_CHILD_BYTES * (nnz if kernel_nnz is None else kernel_nnz)
    batch = 1 << max(0, int(largest).bit_length() - 1)
    while batch > 1:
        need = batch * (member + (kernel if children(batch) else 0))
        if need <= budget:
            break
        batch //= 2
    return batch

# The epoch budget of a node LP outside batched search (the serial
# search, dives, the pump) when ``node_max_epochs`` is ``None``.
_NODE_MAX_EPOCHS = 200
# The exact lane's band (``exact_overflow_share``): reserved nodes kept for
# each lane thread before the top of the band overflows to the device.
_EXACT_OVERFLOW_KEEP = 4
# ``exact_strong_iterations="auto"``: the least and most pivots a strong
# side gets (SCIP's limits for its strong branching LPs).
_EXACT_STRONG_AUTO = (10, 500)
# The adaptive band (``exact_band="auto"``): its limits, and the scale it
# grows by as the relative gap closes (``band_scale / gap``; at the starting
# scale, half the gap at a 50% gap, all of it from 25% down).
_EXACT_BAND_MIN = 0.001
_EXACT_BAND_MAX = 1.0
_EXACT_BAND_GAP_SCALE = 0.25
# The band's cap (``band_capacity_top``): the lane's threads keep the nodes
# each clears in this many batches, within these per thread (before any
# lane job has finished: ``_EXACT_OVERFLOW_KEEP``).
_EXACT_BAND_HORIZON_BATCHES = 2.0
_EXACT_BAND_CAP_PER_THREAD = (1, 16)
# Tuning the scale (``tune_band_scale``): the factor it moves by, its
# limits, and the samples each side of the band's edge needs first.
_EXACT_BAND_SCALE_STEP = 1.25
_EXACT_BAND_SCALE_LIMITS = (0.01, 4.0)
_EXACT_BAND_TUNE_SAMPLES = 8
# The edge of the band whose lift is measured: its top half for the lane,
# and as far again above it for the device.
_EXACT_BAND_EDGE = 0.5

# Root cut rounds (``cut_age``): the rounds in a row a cut stays in the LP
# while idle (slack, with a zero dual), as HiGHS's LP age limit.
_CUT_AGE = 10
# The most cuts kept for propagation (``cut_pool_propagation``).
_CUT_POOL_MAX_ROWS = 10_000
# In-out separation (``inout_separation``): once λ would pass this, the
# cut rounds stop separating at the in-out point until the incumbent
# changes (the point is all but the LP point by then).
_INOUT_MAX_LAMBDA = 0.9
# The global cut pool (``_CutPool``): the most cuts kept (newest), and the
# least efficacy for a pooled cut to come back into the LP.
_CUT_POOL_SIZE = 10_000
_CUT_POOL_MIN_EFFICACY = 1e-4
# Tree cut refreshes (``tree_cuts``): the tree's first seconds refresh
# freely, then within ``tree_cut_share`` of its time; a slot idle for this
# many refreshes in a row may take a new cut.
_TREE_CUT_GRACE_SECONDS = 2.0
_TREE_CUT_IDLE_REFRESHES = 2
# Hybrid branching scores (``branching_scores``): the weights of the
# conflict, cutoff and inference parts beside the bound gains (SCIP's
# defaults, which HiGHS also uses), and the conflict scores' decay per
# conflict (VSIDS).
_CONFLICT_WEIGHT = 1e-2
_CUTOFF_WEIGHT = 1e-4
_INFERENCE_WEIGHT = 1e-4
_CONFLICT_DECAY = 0.95
# Restarts (``restarts``), as HiGHS restarts: the share of the integers free
# at the last (re)start that reduced-cost fixing of the root box has fixed
# since, at the root (after its heuristics) and in the tree.
_RESTART_ROOT_SHARE = 0.025
_RESTART_TREE_SHARE = 0.10
class _CutPool:
    """
    The global cut pool: every cut the root and tree rounds have found, the
    newest ``size`` of them: ``coef·x[idx] <= rhs`` by key (``_cut_key``).
    Each was separated over the root box of its time, which only shrinks,
    so each holds at every later node, across restarts too. ``violated``
    finds those an LP point cuts off, to go back into the LP.
    """

    def __init__(self, size):
        self.size = int(size)
        self.cuts = {}  # key -> (family, idx, coef, rhs), oldest first
        self._block = None  # (keys, CSR rows, rhs) of ``cuts``

    def __len__(self):
        return len(self.cuts)

    def add(self, cuts):
        """Keep ``cuts`` (``(family, idx, coef, rhs)``); returns their keys."""
        keys = []
        changed = False
        for cut in cuts:
            key = _cut_key(cut)
            changed |= self.cuts.pop(key, None) is None
            self.cuts[key] = cut  # the newest now
            keys.append(key)
        while len(self.cuts) > self.size:
            self.cuts.pop(next(iter(self.cuts)))
            changed = True
        if changed:
            # Only a new or evicted cut stales the block (``violated``):
            # cuts re-added (moved to the newest) keep their rows there.
            self._block = None
        return keys

    def violated(self, x, exclude=(), min_efficacy=_CUT_POOL_MIN_EFFICACY):
        """The kept cuts that ``x`` violates by an efficacy of at least
        ``min_efficacy``, other than those keyed in ``exclude``."""
        if not self.cuts:
            return []
        if self._block is None:
            keys = list(self.cuts)
            cuts = [self.cuts[key] for key in keys]
            lengths = [cut[1].size for cut in cuts]
            rows = sp.csr_matrix(
                (
                    np.concatenate([np.asarray(cut[2], np.float64) for cut in cuts]),
                    np.concatenate([cut[1] for cut in cuts]).astype(np.int64),
                    np.concatenate([[0], np.cumsum(lengths)]),
                ),
                shape=(len(cuts), x.shape[0]),
            )
            norms = np.sqrt(rows.multiply(rows).sum(axis=1)).A1
            rhs = np.array([cut[3] for cut in cuts], np.float64)
            self._block = (keys, rows, rhs, np.maximum(norms, 1e-12))
        keys, rows, rhs, norms = self._block
        efficacy = (rows @ x - rhs) / norms
        exclude = set(exclude)
        return [
            self.cuts[keys[i]]
            for i in np.flatnonzero(efficacy >= min_efficacy)
            if keys[i] not in exclude
        ]


def _as_cut(cut):
    """A ``cut_list`` entry (a dict) as ``(family, idx, coef, rhs)``."""
    return (cut["family"], cut["idx"], cut["coef"], cut["rhs"])


def _cut_key(cut):
    """A cut's identity: its columns, coefficients and right-hand side."""
    _, idx, coef, rhs = cut
    return (
        np.asarray(idx).tobytes(),
        np.asarray(coef, np.float64).tobytes(),
        float(rhs),
    )


class _BranchAndBound:
    # Where batched search spends the main thread (``result["batch_timing"]``):
    # blocked on the device for a batch's LPs or its children, evaluating
    # the members on the host, and the host's Lagrangian bounds within that
    # (``dual_terms`` and ``bound_over``, all callers).
    batch_wait_seconds = 0.0
    batch_evaluate_seconds = 0.0
    bound_seconds = 0.0
    # Within or beside that: tree cut refreshes (in the evaluation), the
    # exact lane's results, and branching from device-built children.
    cut_refresh_seconds = 0.0
    exact_collect_seconds = 0.0
    # Of it, the time servicing the lane during device waits
    # (``await_device``), and whether the lane is being collected or fed
    # (a wait nested in that does not service it again).
    exact_service_seconds = 0.0
    _lane_busy = False
    branch_seconds = 0.0
    _cut_worker = None  # the tree cut worker (``tree_cut_async``), if made
    exact_bounds_reused = 0  # lane jobs whose own bounds were used

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
        max_tree_cuts_per_round,
        tree_cut_rows,
        tree_cut_nnz,
        fix_and_propagate,
        feasibility_jump,
        jump_iterations,
        repair,
        repair_max_epochs,
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
        mir_cuts=True,
        clique_cuts=True,
        zero_half_cuts=True,
        cut_age=_CUT_AGE,
        max_cut_rows=None,
        gomory_candidates=None,
        inout_separation=None,
        tree_cut_points=4,
        tree_cut_async=True,
        tree_cut_rounds=1,
        tree_cut_share=0.25,
        tree_cut_max_nnz=None,
        cut_pool_size=_CUT_POOL_SIZE,
        cutoff_from_start=False,
        root_redcost_propagation=False,
        global_redcost_fixing=False,
        cut_pool_propagation=False,
        plunge_child="score",
        implied_integer_cuts=True,
        separation_backend="auto",
        separation_device="cpu",
        node_bound_stop=True,
        dual_repair=True,
        node_lp_tolerance=None,
        heuristic_effort=0.1,
        reliability_threshold=4,
        strong_candidates=8,
        strong_epochs=50,
        strong_max_depth=None,
        strong_screen=32,
        strong_backend="jax",
        root_strong_candidates="auto",
        root_strong_epochs=None,
        root_crossover="auto",
        round_repair=True,
        round_repair_every=None,
        round_repair_seconds=10.0,
        pump_jump_iterations=None,
        pump_jump_share=0.2,
        probing=True,
        probing_seconds=None,
        probing_max_variables=None,
        probing_backend="auto",
        probing_max_useless=1000,
        probing_passes=_PROBE_MAX_PASSES,
        probing_max_implications=_PROBE_MAX_IMPLICATIONS,
        bulk_rounding=True,
        bulk_starts=4,
        repair_share=0.2,
        batch_epochs=2,
        children_backend="auto",
        batch_roundings=False,
        node_lp_options=None,
        tree_heuristics=False,
        tree_heuristic_every=20,
        tree_heuristic_share=0.05,
        tree_jump_iterations=None,
        node_crossover=False,
        node_crossover_every=1,
        node_crossover_seconds=2.0,
        node_precision=None,
        jump_backend="auto",
        parallel_heuristics=False,
        symmetry="auto",
        restarts=5,
        restart_presolve=False,
        mip_start=None,
        implied_bound_cuts=True,
        probing_cliques=True,
        implied_variable_bounds=True,
        max_aggregation=None,
        aggregation_width=_CUT_AGGREGATION_WIDTH,
        cmir_per_row=_CUT_CMIR_PER_ROW,
        cut_score_weights=_CUT_SCORE_WEIGHTS,
        exact_nodes=0,
        exact_node_seconds=10.0,
        exact_band=0.005,
        exact_strong_candidates=4,
        exact_strong_iterations=1000,
        exact_overflow_share=0.5,
        exact_overflow_epochs=None,
        exact_basis_memory=2e9,
        exact_strong_lookahead=None,
        exact_strong_cutoff=False,
        exact_root_candidates=128,
        exact_root_seconds=60.0,
        exact_root_iterations=None,
        cutoff_probing=True,
        cutoff_probing_threads=4,
        cutoff_probing_seconds=60.0,
        cutoff_probing_iterations=1000,
        one_opt=True,
        objective_propagation=False,
        plunge_gap=0.25,
        branch_implied=False,
        near_integral_tolerance=1e-2,
        tree_pool_every=1,
        parallel_rounding=True,
        node_jump_seconds=1.0,
        pipeline_batches=True,
        gpu_dive_every=0,
        gpu_dive_steps=30,
        gpu_dive_epochs=8,
        gpu_dive_share=0.2,
        gpu_dive_snap=0.1,
        gpu_dive_fix_share=0.1,
        gpu_dive_crossover=4,
        exact_plunges=False,
        exact_plunge_share=0.25,
        jump_threads=4,
        jump_restarts=True,
        root_jump_seconds=60.0,
        block_decomposition=False,
        block_decomposition_seconds=300.0,
        trivial_heuristic=True,
        submip_solver="highs",
        rins_tolerance=0.1,
        crossover_seconds=10.0,
        crossover_solutions=3,
        crossover_min_fixed=0.5,
        proximity_seconds=10.0,
        first_solution_seconds=0.0,
        objective_levels=True,
        objective_level_seconds=2.0,
        cut_round_lp="jaddle",
        root_lp="jaddle",
        interrupt=None,
    ):
        # Every setting, for sub-searches (``jaddle_rins``).
        self.settings = {
            key: value for key, value in locals().items() if key not in ("self", "mip")
        }
        # Ctrl-C (``request_interrupt``): an event set at the first press,
        # which every time check treats as the deadline passed. A sub-search
        # shares its parent's (``interrupt``); the search that made it
        # catches SIGINT while it runs (``run``).
        self.owns_interrupt = interrupt is None
        self.interrupt = threading.Event() if interrupt is None else interrupt
        self.rins_nodes = rins_nodes
        if symmetry not in ("auto", True, False):
            raise ValueError(
                f"symmetry must be 'auto', True or False, not {symmetry!r}"
            )
        self.symmetry = symmetry
        self.symmetry_stats = None
        if int(restarts) < 0:
            raise ValueError(f"restarts must be at least 0, not {restarts!r}")
        self.max_restarts = int(restarts)
        # HiGHS's presolve at a restart, the search going on in the reduced
        # MIP (``presolved_round``).
        self.restart_presolve = bool(restart_presolve)
        self.restart_presolve_log = []
        self.restart_log = []
        # The integers free in the root box when the current round started,
        # and the best bound of the rounds before (a floor for the bound).
        self.restart_free = None
        self.bound_floor = -np.inf
        self.rins_seen = set()  # neighbourhoods searched by jaddle_rins
        # A warm start for the root LP (a sub-search's, from its parent).
        self.root_warm = None
        self.rins_skipped = 0
        # A cutoff known before the search (a sub-MIP's parent's).
        self.initial_cutoff = (
            np.inf if initial_cutoff is None else float(initial_cutoff)
        )
        # Above the largest objective over the root box (set in ``_run``):
        # no feasible point reaches it.
        self.box_cutoff = np.inf
        if branching not in ("auto", "pseudocost", "most_fractional", "reliability"):
            raise ValueError(f"unknown branching rule {branching!r}")
        if branching == "auto":
            # PDHG's truncated child LPs make reliability branching's strong
            # branching too dear in the tree; the root sweep seeds the
            # pseudocosts instead.
            branching = "pseudocost"
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
        self.strong_backend = strong_backend
        self._strong_kernels = None  # (mip, _StrongKernels)
        self.strong_branch_nodes = 0
        self.strong_branch_lps = 0
        self.strong_branch_seconds = 0.0
        self.strong_fixings = 0
        # The root sweep (``root_strong_candidates``): strong branching at
        # the root on the candidates with the largest ``impact``, which seeds
        # the pseudocosts. Off for most-fractional branching.
        if root_strong_candidates == "auto":
            root_strong_candidates = self.strong_candidates
        if branching == "most_fractional":
            root_strong_candidates = 0
        self.root_strong_candidates = max(int(root_strong_candidates), 0)
        self.root_strong_epochs = (
            self.strong_epochs
            if root_strong_epochs is None
            else int(root_strong_epochs)
        )
        self.root_strong_lps = 0
        self.root_strong_seconds = 0.0
        if root_crossover not in ("auto", True, False):
            raise ValueError(
                f"root_crossover must be 'auto', True or False, not {root_crossover!r}"
            )
        self.root_crossover = root_crossover
        self.root_crossover_used = False
        # The last root crossover: (the MIP it was for, its SaddleState).
        self.root_vertex_state = None
        # Round and repair (``round_repair``): roundings of LP points whose
        # integers HiGHS fixes, solving the rest exactly.
        self.use_round_repair = bool(round_repair)
        self.round_repair_every = (
            heuristic_every if round_repair_every is None else int(round_repair_every)
        )
        self.round_repair_limit = float(round_repair_seconds)
        self.round_repair_calls = 0
        self.round_repair_finds = 0
        self.round_repair_seconds = 0.0
        self.round_repair_seen = set()  # integer parts already tried
        # Snapping for PDHG points (vertices are exact): their noise is about
        # the LP tolerance, which would round near-integral values away.
        tolerances = [lp_tolerance] + ([node_lp_tolerance] if node_lp_tolerance else [])
        self.round_snap = min(
            0.1, max(10 * integrality_tolerance, 100 * max(tolerances))
        )
        if node_selection not in ("hybrid", "best_first", "depth_first"):
            raise ValueError(f"unknown node selection {node_selection!r}")
        lp_options = dict(lp_options or {})
        reserved = [key for key in _LP_RESERVED if key in lp_options]
        if reserved:
            raise ValueError(f"lp_options cannot set {', '.join(reserved)}")
        node_lp_options = dict(node_lp_options or {})
        # ``node_lp_options["precision"]`` is another way to set ``node_precision``.
        if "precision" in node_lp_options:
            option = node_lp_options.pop("precision")
            if node_precision is not None and option != node_precision:
                raise ValueError(
                    f"node_precision={node_precision!r} and "
                    f"node_lp_options['precision']={option!r} disagree"
                )
            node_precision = option
        reserved = [key for key in _LP_RESERVED if key in node_lp_options]
        if reserved:
            raise ValueError(f"node_lp_options cannot set {', '.join(reserved)}")
        # The precision profile of node LPs (``None``: the search's own).
        self.node_precision = (
            None if node_precision is None else jo._active_profile(node_precision)
        )

        self.original = mip  # the MIP as given: incumbents are checked on it
        self.device = _search_device(device, mip.A.nnz)
        # The MIP searched: the original, or it with tightened coefficients;
        # cuts extend ``self.mip``.
        self.base = mip
        self.use_coefficient_tightening = coefficient_tightening
        self.coefficients_tightened = 0
        self.cuts, self.cut_rounds = cuts, cut_rounds
        # The LP the root cut rounds re-solve (``cut_round_lp``): Jaddle's
        # PDHG (``"jaddle"``), or
        # HiGHS's dual simplex warm from the last basis (``exact_cut_loop``).
        self.cut_round_lp = _lp_solver(cut_round_lp, "cut_round_lp")
        # The solver of the first root LP (``root_lp``): Jaddle's PDHG, or HiGHS with
        # PDHG warm from its vertex (``highs_root_lp``); and its time.
        self.root_lp = _lp_solver(root_lp, "root_lp")
        self.root_lp_stats = {"solver": self.root_lp, "seconds": 0.0, "highs_seconds": 0.0}
        self.exact_cut_stats = None  # ``result["exact_cut_rounds"]``
        self.max_cuts_per_round, self.cut_min_progress = (
            max_cuts_per_round,
            cut_min_progress,
        )
        # Small MIPs get Gomory cuts by default: their crossovers take
        # milliseconds.
        small = mip.A.nnz <= _SMALL_MIP_NNZ
        if gomory_cuts not in ("auto", True, False):
            raise ValueError(
                f"gomory_cuts must be 'auto', True or False, not {gomory_cuts!r}"
            )
        self.gomory = small if gomory_cuts == "auto" else bool(gomory_cuts)
        if aggregate_integers not in ("auto", True, False):
            raise ValueError(
                f"aggregate_integers must be 'auto', True or False, not {aggregate_integers!r}"
            )
        # ``"auto"``: on for small MIPs, and for every MIP with the numba
        # separation backend, where a round costs milliseconds anyway.
        self.aggregate_integers = (
            (
                small
                or separation_backend in ("auto", "numba")
                and _numba_separate() is not None
            )
            if aggregate_integers == "auto"
            else bool(aggregate_integers)
        )
        # Rows added to a c-MIR aggregate (``max_aggregation``; ``None``: 4
        # with integer aggregation, else ``_CUT_MAX_AGGREGATION``).
        self.max_aggregation = (
            (4 if self.aggregate_integers else _CUT_MAX_AGGREGATION)
            if max_aggregation is None
            else int(max_aggregation)
        )
        # The first aggregation steps tried per row, and the c-MIR cuts kept
        # per row (``CutSeparator``).
        self.aggregation_width = max(1, int(aggregation_width))
        self.cmir_per_row = max(1, int(cmir_per_row))
        # ``select_cuts``'s weights for objective parallelism, integer
        # support and directed cutoff distance (``None``: efficacy alone).
        self.cut_score_weights = (
            None if cut_score_weights is None else tuple(map(float, cut_score_weights))
        )
        if self.cut_score_weights is not None and len(self.cut_score_weights) != 3:
            raise ValueError(
                "cut_score_weights must be None or three weights (objective "
                f"parallelism, integer support, directed cutoff distance), not "
                f"{cut_score_weights!r}"
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
        self.use_mir_cuts = bool(mir_cuts)
        # Clique cuts (``clique_cuts``): one conflict graph over the root
        # box, built at the first separation and kept for the tree rounds.
        self.use_clique_cuts = bool(clique_cuts)
        # Zero-half cuts (``zero_half_cuts``) from the LP's rows, its cuts
        # included, at each root and tree round.
        self.use_zero_half = bool(zero_half_cuts)
        # Implied-bound cuts (``implied_bound_cuts``) from what probing learnt.
        self.use_implied_bound_cuts = bool(implied_bound_cuts)
        # Conflicts from probing in the conflict graph (``probing_cliques``):
        # the cache whose implications it holds.
        self.use_probing_cliques = bool(probing_cliques)
        self._graph_probe_cache = None
        # Probing's bounds on continuous columns as variable bounds for c-MIR
        # and flow cover cuts (``implied_variable_bounds``).
        self.use_implied_variable_bounds = bool(implied_variable_bounds)
        self._implied_bounds = None  # (ProbingCache, ImpliedBounds)
        # 1-opt (``one_opt``) on each new incumbent.
        self.use_one_opt = bool(one_opt)
        self.one_opt_finds = 0
        self._original_csc = None  # the MIP as given, in CSC form, for 1-opt
        # The root cut pool's growth: rounds a slack cut stays in the
        # simplex rounds, the most cut rows in the root LP (``None``: no
        # limit) and the tableau rows tried per round (``None``: twice
        # ``max_cuts_per_round``).
        for name, value in (
            ("cut_age", cut_age),
            ("max_cut_rows", max_cut_rows),
            ("gomory_candidates", gomory_candidates),
        ):
            if value is not None and (int(value) != value or value < 1):
                raise ValueError(
                    f"{name} must be a positive integer or None, not {value!r}"
                )
        if cut_age is None:
            raise ValueError("cut_age must be a positive integer, not None")
        self.cut_age = int(cut_age)
        self.max_cut_rows = None if max_cut_rows is None else int(max_cut_rows)
        self.gomory_candidates = (
            None if gomory_candidates is None else int(gomory_candidates)
        )
        # The incumbent's use for the bound: the root's reduced-cost row in
        # the conflict pool, and root reduced-cost fixing from every LP's
        # duals.
        self.root_redcost_propagation = bool(root_redcost_propagation)
        self.global_redcost_fixing = bool(global_redcost_fixing)
        self.global_redcost_fixed = 0
        # In-out separation (``inout_separation``): the cut rounds also
        # separate at ``λ·x + (1 − λ)·incumbent``, λ starting here.
        if inout_separation is not None and not 0.0 < inout_separation < 1.0:
            raise ValueError(
                f"inout_separation must be in (0, 1) or None, not {inout_separation!r}"
            )
        self.inout_separation = (
            None if inout_separation is None else float(inout_separation)
        )
        self.inout_lambda = self.inout_separation
        self.inout_at = None  # the incumbent value λ was last reset for
        self.inout_cuts = 0
        self.inout_rounds = 0
        # Tree cuts (``tree_cuts``): refreshes of the LP's cut block between
        # node solves (``refresh_cuts``), separating at up to
        # ``tree_cut_points`` points, re-solving members their new cuts cut
        # off up to ``tree_cut_rounds`` times, within ``tree_cut_share`` of
        # the tree's time; cuts denser than ``tree_cut_max_nnz`` are left out.
        for name, value, least in (
            ("tree_cut_points", tree_cut_points, 1),
            ("tree_cut_rounds", tree_cut_rounds, 0),
            ("cut_pool_size", cut_pool_size, 0),
        ):
            if int(value) != value or value < least:
                raise ValueError(
                    f"{name} must be an integer of at least {least}, not {value!r}"
                )
        self.tree_cut_points = int(tree_cut_points)
        # Tree cut separation on a worker thread (``tree_cut_async``): a
        # full refresh hands its points to the worker and the next refresh
        # takes the cuts (``submit_cut_job``, ``collect_cut_job``).
        self.tree_cut_async = bool(tree_cut_async)
        self._cut_worker = None  # its ThreadPoolExecutor, made on first use
        self._cut_job = None  # (future, base) of the separation in flight
        self.tree_cut_async_jobs = 0
        self.tree_cut_rounds = int(tree_cut_rounds)
        if not 0.0 <= tree_cut_share <= 1.0:
            raise ValueError(
                f"tree_cut_share must be in [0, 1], not {tree_cut_share!r}"
            )
        self.tree_cut_share = float(tree_cut_share)
        # The densest cut kept for the tree (``None``: ``max(64, n / 8)``):
        # dense rows make every iteration of the node LPs dearer.
        if tree_cut_max_nnz is not None and (
            int(tree_cut_max_nnz) != tree_cut_max_nnz or tree_cut_max_nnz < 1
        ):
            raise ValueError(
                f"tree_cut_max_nnz must be a positive integer or None, "
                f"not {tree_cut_max_nnz!r}"
            )
        self.tree_cut_max_nnz = (
            max(64, mip.num_variables // 8)
            if tree_cut_max_nnz is None
            else int(tree_cut_max_nnz)
        )
        # The root's separators over the root box, for the tree's refreshes
        # (``node_separators``), and a lock, since a separation keeps state.
        self._node_separators = None  # (key, [CutSeparator, ...])
        self._separator_lock = threading.Lock()
        # The cut block's version (``refresh_cuts``): it goes up whenever a
        # refresh changes the block, and each slot keeps the version its
        # cut came in at, so a warm start from an older block has the duals
        # of replaced slots zeroed (``fit_warm``).
        self.cut_version = 0
        self.slot_version = []
        self.tree_cut_refreshes = 0
        self.tree_cut_requeued = 0
        self.tree_cut_seconds = 0.0
        self.tree_cut_pool_hits = 0
        self.tree_cut_skipped = 0
        self.tree_cut_families = collections.Counter()
        self.last_refresh_nodes = 0
        self.last_pool_nodes = 0  # the last pool refresh (``tree_pool_every``)
        self.tree_pool_refreshes = 0
        # The heuristic clock (``heuristic_due``): the multiple of each
        # schedule's period it last ran at.
        self.heuristic_fired = {}
        # A lock for what background work shares with the search (the cut
        # pool, the archive, the counters).
        self._shared_lock = threading.Lock()
        self.cut_pool = _CutPool(int(cut_pool_size))
        # A solution is known before the search (``mip_start`` with
        # presolve: its value is ``initial_cutoff``, but the point is not in
        # this MIP's columns), so hybrid node selection need not dive.
        self.cutoff_from_start = bool(cutoff_from_start)
        self.tree_clock = None  # (start, strong seconds then) while in the tree
        self.tree_seconds = 0.0
        self.tree_strong_seconds = 0.0
        # Every cut separated (root rounds and tree rounds), kept for
        # propagation at every node once it has left the LP
        # (``cut_pool_propagation``): {key: (idx, coef, rhs)}, oldest first.
        self.use_cut_pool = bool(cut_pool_propagation)
        # The objective row c·x <= cutoff in the search's propagation
        # (``objective_propagation``; the heuristics always have it).
        self.use_tree_objective = bool(objective_propagation)
        # How a plunge (or dive) picks its next child (``branch``).
        if plunge_child not in ("score", "estimate", "rounding"):
            raise ValueError(
                f"plunge_child must be 'score', 'estimate' or 'rounding', not {plunge_child!r}"
            )
        self.plunge_child = plunge_child
        self.cut_archive = {}
        self.conflict_graph = None
        if separation_backend not in ("auto", "numba", "jax", "numpy"):
            raise ValueError(
                "separation_backend must be 'auto', 'numba', 'jax' or 'numpy', "
                f"not {separation_backend!r}"
            )
        self.separation_backend = _separation_backend(separation_backend)
        # Where the batched separation kernels run (``separation_device``):
        # they run in float64, which consumer GPUs are slow at, on rows
        # small enough that the transfers cost more than the work.
        if separation_device not in ("cpu", "search"):
            raise ValueError(
                f"separation_device must be 'cpu' or 'search', not {separation_device!r}"
            )
        self.separation_device = (
            jax.devices("cpu")[0] if separation_device == "cpu" else None
        )
        # Node LPs stop once their Lagrangian bound passes the cutoff.
        self.node_bound_stop = bool(node_bound_stop)
        self.bound_stops = 0
        # Sign repair of node duals on wide continuous columns (``DualRepair``).
        self.use_dual_repair = bool(dual_repair)
        self.dual_repairer = None
        self.dual_repair_stats = collections.Counter(tried=0, improved=0, seconds=0.0)
        self.use_tree_cuts = bool(tree_cuts)
        self.tree_cut_every = tree_cut_every
        if int(tree_pool_every) < 0:
            raise ValueError(
                f"tree_pool_every must be at least 0, not {tree_pool_every!r}"
            )
        self.tree_pool_every = int(tree_pool_every)
        if int(max_tree_cuts_per_round) < 1:
            raise ValueError(
                f"max_tree_cuts_per_round must be at least 1, not {max_tree_cuts_per_round!r}"
            )
        self.max_tree_cuts_per_round = int(max_tree_cuts_per_round)
        for name, value in (
            ("tree_cut_rows", tree_cut_rows),
            ("tree_cut_nnz", tree_cut_nnz),
        ):
            if value is not None and int(value) < 1:
                raise ValueError(f"{name} must be None or at least 1, not {value!r}")
        self.tree_cut_rows = None if tree_cut_rows is None else int(tree_cut_rows)
        self.tree_cut_nnz = None if tree_cut_nnz is None else int(tree_cut_nnz)
        self.tree_cuts_added = 0
        self.tree_cuts_dropped = 0
        self.cut_list = []
        self.root_bound_before_cuts = None
        self.use_fix_and_propagate = fix_and_propagate
        self.use_jump, self.jump_iterations = feasibility_jump, jump_iterations
        # ``"numba"`` (the incremental host jump) or ``"jax"`` (the device
        # one). A numba step is one move, so the short jumps of the pump and
        # the tree get ``_numba_short_jump`` of them by default (``None``).
        self.jump_backend = _jump_backend(jump_backend)
        numba = self.jump_backend == "numba"
        # CPU heuristics on background threads (``_HeuristicWorkers``); their
        # threads start with the first job.
        self.workers = (
            _HeuristicWorkers(jump_threads) if parallel_heuristics else None
        )
        # Exact node LPs on CPU threads (``exact_nodes``) for the nodes
        # that hold the global bound; batched search only.
        self.exact_lane = (
            _ExactLane(exact_nodes, exact_node_seconds)
            if exact_nodes and batch_size != 1
            else None
        )
        self.exact_solved = 0
        self.exact_infeasible = 0
        self.exact_failed = 0
        self.exact_seconds = 0.0
        self.exact_gain = 0.0  # bound gained over the nodes' own, summed
        # HiGHS bases of the lane's solved nodes and strong sides, for their
        # children's LPs (``exact_basis_memory`` bytes at most, about one
        # byte a column and row each): key -> [shape, basis, holders],
        # oldest first; the oldest go when it is full.
        self.exact_basis_memory = float(exact_basis_memory or 0)
        self.basis_store = collections.OrderedDict()
        self.basis_keys = itertools.count()
        self.exact_pivots = 0
        self.exact_warm_bases = 0
        # The band of the queue kept for the lane (``exact_band``): nodes
        # within this share of the gap above the global bound wait in
        # ``exact_reserved`` (a heap like ``open``) for an exact LP, and the
        # device never takes them. ``0``: the lane takes the heap's top
        # whenever it has room, and the device the rest. ``"auto"``: the
        # band widens as the gap closes (``adapt_exact_band``).
        self.exact_band_auto = isinstance(exact_band, str)
        if self.exact_band_auto and exact_band != "auto":
            raise ValueError(f"exact_band must be a number or 'auto', not {exact_band!r}")
        self.exact_band = _EXACT_BAND_MIN if self.exact_band_auto else float(exact_band)
        self.exact_band_range = [self.exact_band, self.exact_band]
        # The auto band's scale (``tune_band_scale``) and cap
        # (``band_capacity_top``): the lift and seconds of the nodes each
        # side solved near the band's top since the last tune, as
        # ``[lift, seconds, nodes]``, the last rates compared, and the
        # time between batch launches (an average).
        self.band_scale = _EXACT_BAND_GAP_SCALE
        self.band_scale_range = [self.band_scale, self.band_scale]
        self.band_edge_sums = {"exact": [0.0, 0.0, 0], "device": [0.0, 0.0, 0]}
        self.band_rates = (None, None)
        self.band_tunes = 0
        self.band_capped = False
        self.band_capped_batches = 0
        self.band_cap_nodes = None
        self.batch_period = None
        self.last_launch = None
        self.last_raw_bound = None
        self.band_bound = -np.inf  # the global bound ``exact_band_top`` saw
        self.exact_reserved = []
        # Nodes the lane has finished that ``collect_exact`` has not yet
        # evaluated: in no other list, so the global bound counts them here.
        self.exact_unevaluated = []
        # Strong branching in the lane (``exact_strong_candidates``,
        # ``exact_strong_iterations``).
        self.exact_strong_candidates = int(exact_strong_candidates)
        self.exact_strong_auto = isinstance(exact_strong_iterations, str)
        if self.exact_strong_auto and exact_strong_iterations != "auto":
            raise ValueError(
                "exact_strong_iterations must be a number or 'auto', not "
                f"{exact_strong_iterations!r}"
            )
        self.exact_strong_iterations = (
            _EXACT_STRONG_AUTO[1]
            if self.exact_strong_auto
            else int(exact_strong_iterations)
        )
        self.exact_jobs = 0  # lane jobs collected (``exact_pivots`` is theirs)
        # ``_exact_strong``'s lookahead (``None``/0: every candidate) and
        # whether its sides stop at the cutoff.
        self.exact_strong_lookahead = int(exact_strong_lookahead or 0)
        self.exact_strong_cutoff = bool(exact_strong_cutoff)
        self.exact_strong_lps = 0
        self.exact_strong_seconds = 0.0
        self.exact_reused = 0
        # Exact strong branching at the root over the lane's threads
        # (``exact_root_candidates``: ``None`` every fractional integer, 0
        # off; ``exact_root_seconds``; ``exact_root_iterations``).
        self.exact_root_candidates = (
            None if exact_root_candidates is None else int(exact_root_candidates)
        )
        self.exact_root_seconds = float(exact_root_seconds)
        self.exact_root_iterations = (
            None if exact_root_iterations is None else int(exact_root_iterations)
        )
        self.root_exact_stats = dict(
            candidates=0, lps=0, fixed=0, pivots=0, seconds=0.0, bound=None
        )
        # Cutoff probing (``_ProbeLane``): rounds over the root box's free
        # integers on CPU threads, once per cutoff, again while a round
        # fixes ``_PROBE_AGAIN_SHARE`` of what it probed.
        self.probe_lane = (
            _ProbeLane(cutoff_probing_threads)
            if cutoff_probing and cutoff_probing_threads
            else None
        )
        self.probe_seconds = float(cutoff_probing_seconds or 0.0)
        self.probe_iterations = (
            None if cutoff_probing_iterations is None else int(cutoff_probing_iterations)
        )
        self.probe_cutoff = None  # the cutoff of the last round
        self.probe_again = False
        self.probe_room = None  # the last round's cutoff less the global bound
        self.probe_checked = 0.0  # when ``feed_probing`` last took the bound
        self.probe_stats = dict(
            rounds=0, probed=0, lps=0, pivots=0, fixed=0, seconds=0.0, empty=False
        )
        # The band's overflow (``exact_overflow_share``): past
        # ``_EXACT_OVERFLOW_KEEP`` nodes a lane thread, the top of the band
        # goes to the device, solved to convergence within
        # ``exact_overflow_epochs`` (``None``: ``_NODE_MAX_EPOCHS``).
        if not 0.0 <= float(exact_overflow_share) <= 1.0:
            raise ValueError(
                f"exact_overflow_share must be in [0, 1], not {exact_overflow_share!r}"
            )
        self.exact_overflow_share = float(exact_overflow_share)
        self.exact_overflow_epochs = int(
            _NODE_MAX_EPOCHS if exact_overflow_epochs is None else exact_overflow_epochs
        )
        self.overflow_nodes = 0
        self.overflow_converged = 0
        self.worker_jobs = 0
        self.worker_seconds = 0.0
        self.worker_finds = 0
        self.jump_slices = 0
        if int(jump_threads) < 1:
            raise ValueError(f"jump_threads must be at least 1, not {jump_threads!r}")
        self.use_jump_restarts = bool(jump_restarts)
        # Jumps before the root LP (``root_jump_seconds``), one a jump lane,
        # and the event that stops them at the first incumbent.
        self.root_jump_seconds = float(root_jump_seconds or 0.0)
        self.early_jump = None
        # Block decomposition before the root LP (``block_decomposition``).
        if not isinstance(block_decomposition, (bool, np.bool_)):
            raise ValueError(
                f"block_decomposition must be True or False, not {block_decomposition!r}"
            )
        self.use_block_decomposition = bool(block_decomposition)
        self.block_decomposition_seconds = float(block_decomposition_seconds or 0.0)
        self.block_holder = None
        self.linking_blocks = None  # ``linking_blocks``, once found
        self.use_trivial = bool(trivial_heuristic)
        self.jump_restarts = 0
        self.jump_restart_finds = 0
        self.recent_points = collections.deque(maxlen=64)
        # The incumbent value the last improving-jump slice started from, and
        # the slices' box and warm start (the root's, once it has one).
        self.jump_slice_from = None
        self.jump_box = None
        if pump_jump_iterations is None:
            pump_jump_iterations = _numba_short_jump(mip) if numba else 3000
        if tree_jump_iterations is None:
            tree_jump_iterations = _numba_short_jump(mip) if numba else 3000
        self.use_repair = repair
        if not 0.0 <= float(repair_share) <= 1.0:
            raise ValueError(f"repair_share must be in [0, 1], not {repair_share!r}")
        self.repair_share = float(repair_share)
        self.repair_max_epochs = repair_max_epochs
        # RENS at the root (``None``: auto): a Jaddle sub-search, on small
        # MIPs only (5 s).
        if rens_seconds is None:
            rens_seconds = 5.0 if small else 0.0
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
        # HiGHS sub-MIPs on a CPU thread (``submip_solver="highs"``) for
        # RINS and crossover; ``"jaddle"``: RINS by Jaddle's sub-search.
        if submip_solver not in ("highs", "jaddle"):
            raise ValueError(
                f"submip_solver must be 'highs' or 'jaddle', not {submip_solver!r}"
            )
        if not 0.0 <= float(crossover_min_fixed) <= 1.0:
            raise ValueError(
                f"crossover_min_fixed must be in [0, 1], not {crossover_min_fixed!r}"
            )
        if int(crossover_solutions) < 2:
            raise ValueError(
                f"crossover_solutions must be at least 2, not {crossover_solutions!r}"
            )
        self.rins_tolerance = float(rins_tolerance)
        self.crossover_seconds = float(crossover_seconds or 0.0)
        self.crossover_solutions = int(crossover_solutions)
        self.crossover_min_fixed = float(crossover_min_fixed)
        self.proximity_seconds = float(proximity_seconds or 0.0)
        self.submip_lane = (
            _SubMipLane()
            if submip_solver == "highs"
            and (
                rins_seconds
                or self.crossover_seconds
                or self.proximity_seconds
                or first_solution_seconds
            )
            else None
        )
        # The objective's levels (``objective_levels``): when it rests on a
        # few integer columns, HiGHS shows their assignments below the
        # cutoff empty, lowest first, on a thread of its own, and the bound
        # rises to the lowest level left (``level_floor``).
        self.level_lane = None
        if objective_levels:
            table = objective_level_table(mip.c, mip.integrality, mip.lower, mip.upper)
            if table is not None:
                self.level_lane = _LevelLane(*table, objective_level_seconds)
        self.level_finds = 0
        self.level_floor_cache = (None, -np.inf)
        self.submip_last = None  # the kind of the lane's last job
        self.submip_budget = 0.0  # the lane's last job's seconds
        self.submip_lane_seconds = 0.0
        # The share of the agreeing integers RINS fixes on the lane
        # (``adapt_rins_fixing``), and the draws that pick them.
        self.rins_fixing_rate = 1.0
        self.rins_rng = np.random.default_rng(0)
        self.rins_finds = 0
        self.crossover_calls = 0
        self.crossover_finds = 0
        self.proximity_calls = 0
        self.proximity_finds = 0
        self.proximity_empty = 0  # calls HiGHS showed had nothing to find
        # Feasible points with distinct integers, best first: (value, key, x).
        self.solution_pool = []
        self.rens_seconds, self.rens_nodes = rens_seconds, rens_nodes
        self.dive_every, self.dive_depth = dive_every, dive_depth
        self.dives = 0
        self.use_objective_propagation = True
        self.rins_calls = 0
        # RENS on the sub-MIP lane (``rens``, ``tree_rens``): the root's
        # neighbourhood waiting for the lane, and the calls and finds.
        self.rens_pending = None
        self.rens_calls = 0
        self.rens_finds = 0
        # HiGHS on the whole root box before the first incumbent
        # (``submit_first_solution``): the next call's time, doubling.
        self.first_solution_seconds = float(first_solution_seconds or 0.0)
        self.first_solution_budget = self.first_solution_seconds
        self.first_solution_calls = 0
        self.first_solution_finds = 0
        self.plunge_depth = plunge_depth
        self.plunge_node = None  # the child the next step plunges into
        self.plunge_left = 0
        # A plunge stops at a child whose bound is past this share of the
        # way from the best open bound to the cutoff (``plunge_gap``).
        if not 0.0 <= float(plunge_gap) <= 1.0:
            raise ValueError(f"plunge_gap must be in [0, 1], not {plunge_gap!r}")
        self.plunge_gap = float(plunge_gap)
        self.plunges_stopped = 0
        if isinstance(batch_size, str):
            if batch_size != "auto":
                raise ValueError(
                    f"batch_size must be 'auto' or a positive int, not {batch_size!r}"
                )
            batch_size = self.auto_batch_size(mip, children_backend)
            self.batch_size_auto = True
        else:
            self.batch_size_auto = False
        if int(batch_size) < 1:
            raise ValueError("batch_size must be at least 1")
        self.batch_size = int(batch_size)
        self.batches = 0
        # Overlap the host's evaluation of one batch with the device's LPs
        # for the next (``pipelined_step``).
        self.pipeline_batches = bool(pipeline_batches) and self.batch_size > 1
        # Batched node LPs run at most this many epochs a call (``None``:
        # ``node_max_epochs``). A node gets ``node_max_epochs`` in all
        # when that is given, its unfinished LP going back to the queue
        # between calls; by default (``None``) just the one call: it
        # branches on its point after ``batch_epochs`` (truncated node
        # LPs, for throughput; its Lagrangian bound holds at any duals).
        given_node_epochs = node_max_epochs
        if node_max_epochs is None:
            node_max_epochs = _NODE_MAX_EPOCHS
        self.batch_epochs = int(
            node_max_epochs if batch_epochs is None else batch_epochs
        )
        if self.batch_epochs < 1:
            raise ValueError("batch_epochs must be at least 1")
        self.batch_node_epochs = int(
            self.batch_epochs if given_node_epochs is None else given_node_epochs
        )
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
        self.dive_starts = None  # this batch's GPU dive starts, when due
        self.in_flight = None  # the batch on the device (``pipeline_batches``)
        self._device_thread = None  # dispatches its LPs (``on_device_thread``)
        self.pending_children = None  # its children, on the device
        self.deferred_roundings = []
        self.batched_children = 0
        self.batched_roundings = 0
        # The batched node LPs' options (``node_batch``), built on first use.
        self._node_batch_settings = None
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
            probing_seconds = (
                10.0
                if deadline is None
                else min(30.0, 0.05 * max(deadline - start, 0.0))
            )
        self.probing_budget = float(probing_seconds)
        self.probing_max_variables = probing_max_variables
        self.probing_backend = probing_backend
        self.probing_max_useless = probing_max_useless
        self.probing_passes = int(probing_passes)
        self.probing_max_implications = (
            None if probing_max_implications is None else int(probing_max_implications)
        )
        self.probe_cache = None
        # Bulk fix-and-propagate (``bulk_fix_and_propagate``) from
        # ``bulk_starts`` starts at once, after the classic one at the root
        # and in the pump.
        self.use_bulk_rounding = bool(bulk_rounding)
        self.bulk_starts = max(1, int(bulk_starts))
        self._bulk_data = None  # (mip, device rows) for the jax backend
        self.probing_stats = {
            "probed": 0, "fixed": 0, "tightened": 0, "seconds": 0.0, "conflicts": 0,
        }
        self.heuristic_budget = heuristic_seconds
        self.heuristic_seconds = 0.0
        self.root_heuristic_used = 0.0  # by ``root_heuristics``' calls
        self.rel_gap, self.abs_gap = mip_rel_gap, mip_abs_gap
        self.max_nodes, self.deadline = max_nodes, deadline
        self.branching, self.node_selection = branching, node_selection
        self.int_tol = integrality_tolerance
        # Node LP points this close to integral are rounded and tried
        # (``near_integral_tolerance``), each rounding once.
        if not near_integral_tolerance >= 0:
            raise ValueError(
                "near_integral_tolerance must be at least 0, not "
                f"{near_integral_tolerance!r}"
            )
        self.near_integral_tol = float(near_integral_tolerance)
        # Background fix-and-propagate of root, cut round and node points
        # (``parallel_rounding``, with ``parallel_heuristics``).
        self.use_parallel_rounding = bool(parallel_rounding)
        self.background_roundings = 0
        # Batched GPU dives (``gpu_dives``).
        for name, value, low in (
            ("gpu_dive_every", gpu_dive_every, 0),
            ("gpu_dive_steps", gpu_dive_steps, 1),
            ("gpu_dive_epochs", gpu_dive_epochs, 1),
            ("gpu_dive_crossover", gpu_dive_crossover, 0),
        ):
            if int(value) < low:
                raise ValueError(f"{name} must be at least {low}, not {value!r}")
        for name, value in (
            ("gpu_dive_share", gpu_dive_share),
            ("gpu_dive_snap", gpu_dive_snap),
            ("gpu_dive_fix_share", gpu_dive_fix_share),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], not {value!r}")
        self.gpu_dive_every = int(gpu_dive_every)
        self.gpu_dive_steps = int(gpu_dive_steps)
        self.gpu_dive_epochs = int(gpu_dive_epochs)
        self.gpu_dive_share = float(gpu_dive_share)
        self.gpu_dive_snap = float(gpu_dive_snap)
        self.gpu_dive_fix_share = float(gpu_dive_fix_share)
        self.gpu_dive_crossover = int(gpu_dive_crossover)
        # Exact plunges beside the batches (``exact_plunge``).
        if not 0.0 <= float(exact_plunge_share) <= 1.0:
            raise ValueError(
                f"exact_plunge_share must be in [0, 1], not {exact_plunge_share!r}"
            )
        self.use_exact_plunges = bool(exact_plunges) and float(exact_plunge_share) > 0
        self.exact_plunge_share = float(exact_plunge_share)
        self.exact_plunge_active = False
        self.exact_plunge_seconds = 0.0
        self.exact_plunge_stats = collections.Counter(
            plunges=0, nodes=0, **{"max depth": 0}
        )
        self.gpu_dive_seconds = 0.0
        self.gpu_dive_crossover_seconds = 0.0
        self.gpu_dive_stats = collections.Counter(
            runs=0, dives=0, steps=0, leaves=0, finds=0, fixed=0, infeasible=0,
            backtracks=0, flips=0, crossovers=0,
            **{"cut off": 0, "out of steps": 0},
        )
        if not node_jump_seconds >= 0:
            raise ValueError(
                f"node_jump_seconds must be at least 0, not {node_jump_seconds!r}"
            )
        self.node_jump_seconds = float(node_jump_seconds)
        self.background_jumps = 0
        self.near_integral_seen = set()
        self.near_integral_tries = 0
        self.near_integral_finds = 0
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
        if self.node_precision is not None:
            self.node_lp_kwargs["precision"] = self.node_precision
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
        # Implied integers as branching candidates (``branch_implied``):
        # their columns, or ``None``.
        self.branch_implied_cols = (
            np.flatnonzero(mip.implied)
            if branch_implied and mip.implied.any()
            else None
        )
        self.implied_branchings = 0
        self.impact = branching_impact(
            mip,
            None if self.branch_implied_cols is None else mip.integrality | mip.implied,
        )

        self.incumbent = None  # (min-form value, x)
        # The user's starting point (``mip_start``), tried at the first root.
        self.mip_start = check_mip_start(mip_start, mip)
        self.has_mip_start = self.mip_start is not None  # for the log legend
        self.mip_start_accepted = None
        self.closed_floor = np.inf  # least bound of the nodes closed so far
        # Least bound of the nodes closed without proof: an LP that did not
        # converge, or an integral LP point that failed the feasibility check.
        self.unresolved_floor = np.inf
        self.bound_rigorous = True
        self.nodes = 0
        self.lp_epochs = 0
        self.infeasible_nodes = 0
        self.unresolved_nodes = 0
        # Nodes with an integral LP point that did not settle them, split on
        # a free integer (``integral_split_variable``).
        self.integral_splits = 0
        self.history = []
        self.open = []
        self.exact_reserved = []
        self.counter = itertools.count()
        self.pc_sum = np.zeros((2, mip.num_variables))  # [down, up]
        # The exact lane's own pseudocosts (``exact_strong_candidates``):
        # from exact LPs, which truncated node LPs underestimate.
        self.exact_pc_sum = np.zeros((2, mip.num_variables))
        self.exact_pc_count = np.zeros((2, mip.num_variables))
        self.pc_count = np.zeros((2, mip.num_variables))
        # Branching history beside the pseudocosts (``branching_scores``):
        # per variable and direction [down, up], the children made and those
        # that closed at once (``branch_cutoffs``), and the bounds their
        # propagation fixed beyond the split (``inference_sum``); per
        # variable, a decaying count of the closed boxes it was tightened
        # in (``conflict_score``, bumped by ``conflict_bump``).
        self.branch_children = np.zeros((2, mip.num_variables))
        self.branch_cutoffs = np.zeros((2, mip.num_variables))
        self.inference_sum = np.zeros((2, mip.num_variables))
        self.inference_count = np.zeros((2, mip.num_variables))
        self.conflict_score = np.zeros(mip.num_variables)
        self.conflict_bump = 1.0
        self.last_log = -np.inf
        self.log_header_printed = False
        # The progress table is also printed by the heartbeat thread
        # (``log_heartbeat``): the lock keeps lines whole, ``log_closed``
        # (the solving report has begun) and ``log_paused`` (an inner
        # search logs instead, ``presolved_round``) silence it.
        self.log_lock = threading.RLock()
        self.log_closed = False
        self.log_paused = False
        self.tree_started = False  # for ``explored``
        self.branched_nodes = 0  # nodes split (the rest of ``nodes`` are leaves)
        # The bound of the node being evaluated (it is in no list meanwhile).
        self.active_bound = np.inf
        # A batch's members still to be evaluated (also in no list meanwhile).
        self.batch_waiting = []
        # Every objective value is a multiple of this (None: no such step).
        self.objective_step = objective_step(mip.c, mip.integrality | mip.implied)
        # The columns cut separation counts as integers: with
        # ``implied_integer_cuts``, the implied integers too, as HiGHS's cuts
        # do. Presolve showed that each choice of the integers has a best
        # completion with them integral, which no such cut removes, so node
        # bounds hold; node points with them fractional still count.
        self.cut_int = mip.integrality | (
            mip.implied if implied_integer_cuts else False
        )
        # The root's Lagrangian terms, for reduced-cost fixing of the root box
        # when the incumbent improves, and the incumbent last used for it.
        self.root_terms = None
        self.root_terms_rigorous = False
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

    def set_mip(self, mip, rows_only=False):
        """Search ``mip`` (the MIP as given, or it with cuts). ``rows_only``
        (a tree cut refresh, ``refresh_cuts``): only the LP and its rows
        change; the propagators keep the cut rows they were built with,
        which hold everywhere, so they stay valid."""
        self.mip = mip
        self.A_T = mip.A.T.tocsr()
        self.m_ineq = mip.A.shape[0] - mip.n_eq
        self.values = mip.lp.values()
        self.dtype = mip.lp.c.dtype
        # The LP at the node LPs' precision (``node_precision``), cast once.
        if self.node_precision is None:
            self.node_dtype, self.node_mip_lp = self.dtype, mip.lp
        else:
            with self.node_settings():
                self.node_dtype = _wide(jo.jaddle_dtype())
                self.node_mip_lp = jl._cast_lp(mip.lp, self.node_dtype)
        self.node_values = self.node_mip_lp.values()
        if self.use_dual_repair:
            repairer = DualRepair(mip, self.is_int)
            self.dual_repairer = repairer if repairer else None
        if rows_only:
            return
        # The heuristics propagate even when the search does not.
        self.row_propagator = BoundPropagator(
            mip.A, mip.row_lower, mip.row_upper, self.is_int
        )
        self.propagator = self.row_propagator if self.propagate else None
        if self.propagate and self.use_cut_pool and self.cut_archive:
            # The search's propagation also runs the archived cuts.
            self.propagator = self.pooled_propagator(mip)
        elif self.propagate and self.use_tree_objective:
            self.propagator = self.search_propagator(
                mip.A, mip.row_lower, mip.row_upper
            )
        # The rows plus the objective row c·x <= cutoff, for heuristics that
        # should only find improving points (``heuristic_propagator``).
        self.objective_propagator = BoundPropagator(
            sp.vstack([mip.A, sp.csr_matrix(mip.c[None, :])], format="csr"),
            np.append(mip.row_lower, -np.inf),
            np.append(mip.row_upper, np.inf),
            self.is_int,
        )


    def pool_cuts(self, cuts):
        """Put root, tree and node cuts (``(family, idx, coef, rhs)``) into
        the global cut pool (``cut_pool``), so the tree can take them
        back once violated. Each was separated over the root box, which only
        shrinks, so it holds at every node. A cut that leaves the LP for
        being idle stays here."""
        cuts = [cut for cut in cuts if 0 < cut[1].size <= self.tree_cut_max_nnz]
        if cuts:
            with self._shared_lock:
                self.cut_pool.add(cuts)

    def archive_cuts(self, cuts):
        """Keep ``cuts`` (``(family, idx, coef, rhs)``) for propagation
        (``cut_pool_propagation``): those with at most ``max(32, n / 8)``
        nonzeros (dense ones propagate little and cost every node), the
        newest ``_CUT_POOL_MAX_ROWS``. Every cut holds over the root box at
        its separation, and the root box only shrinks, so each stays valid,
        across restarts too."""
        if not self.use_cut_pool:
            return
        limit = max(32, self.base.num_variables // 8)
        for _, idx, coef, rhs in cuts:
            if idx.size == 0 or idx.size > limit:
                continue
            key = (idx.tobytes(), np.asarray(coef, np.float64).tobytes(), float(rhs))
            self.cut_archive.pop(key, None)
            self.cut_archive[key] = (idx, np.asarray(coef, np.float64), float(rhs))
        while len(self.cut_archive) > _CUT_POOL_MAX_ROWS:
            self.cut_archive.pop(next(iter(self.cut_archive)))

    def pooled_propagator(self, mip):
        """A ``BoundPropagator`` over ``mip``'s rows and the archived cuts."""
        cuts = list(self.cut_archive.values())
        lengths = [idx.size for idx, _, _ in cuts]
        pool = sp.csr_matrix(
            (
                np.concatenate([coef for _, coef, _ in cuts]),
                np.concatenate([idx for idx, _, _ in cuts]).astype(np.int64),
                np.concatenate([[0], np.cumsum(lengths)]),
            ),
            shape=(len(cuts), mip.num_variables),
        )
        return self.search_propagator(
            sp.vstack([mip.A, pool], format="csr"),
            np.concatenate([mip.row_lower, np.full(len(cuts), -np.inf)]),
            np.concatenate([mip.row_upper, [rhs for _, _, rhs in cuts]]),
        )

    def search_propagator(self, A, row_lower, row_upper):
        """The search's propagator over these rows: with the objective row
        at the cutoff (``proof_cutoff``, read at each call) when
        ``objective_propagation`` is on (``CutoffPropagator``), else the
        rows alone."""
        if self.use_tree_objective:
            return CutoffPropagator(
                A, row_lower, row_upper, self.mip.c, self.is_int, self.proof_cutoff
            )
        return BoundPropagator(A, row_lower, row_upper, self.is_int)

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
                # Padded to the kernels' capacity (and the objective row's
                # nonzeros): a tree cut refresh changes the rows' nonzeros,
                # and unpadded every one recompiled the bulk propagation.
                capacity = self.kernel_nnz_capacity() + (
                    mip.num_variables if objective else 0
                )
                with jax.enable_x64(True):
                    rows = _propagation_data(
                        A,
                        row_lower,
                        row_upper,
                        self.is_int,
                        nnz_capacity=1 << math.ceil(math.log2(capacity)),
                    )
                self._bulk_data = (key, rows)
            data = self._bulk_data[1]
            if objective:
                # Only the cutoff moves between calls (the padding row,
                # last, stays free).
                upper_rows = np.full(data["row_upper"].shape[0], np.inf)
                upper_rows[: len(row_upper)] = row_upper
                with jax.enable_x64(True):
                    data = dict(data, row_upper=jnp.asarray(upper_rows, jnp.float64))
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
            deadline=(
                None if deadline is None or not np.isfinite(deadline) else deadline
            ),
            propagation_data=data,
            boxes=boxes,
            pad_to=pad_to,
            propagator=propagator,
        )

    # -- bookkeeping ---------------------------------------------------------
    def seconds_left(self):
        """Seconds to the deadline (``None``: no deadline), at least 1ms:
        the deadline can pass between a time check and an LP solve, and
        ``jl.solve`` needs a positive limit (it then stops at once). An
        interrupt (``request_interrupt``) leaves 1ms."""
        if self.interrupt.is_set():
            return 1e-3
        if self.deadline is None:
            return None
        return max(self.deadline - time.time(), 1e-3)

    def out_of_time(self):
        """Whether the deadline has passed, or the search was interrupted:
        an interrupt takes the time-limit path everywhere, which puts nodes
        in flight back in the queue, so the bound reported stays valid."""
        if self.interrupt.is_set():
            return True
        return self.deadline is not None and time.time() >= self.deadline

    def request_interrupt(self):
        """Stop the search as a time limit would, with the best point so
        far (the first Ctrl-C, ``run``): the interrupt event is set (the LP
        being solved stops at its next chunk, ``jl.solve(cancel=...)``;
        sub-searches share the event), the deadline moves to now (budgets
        drawn from it shrink to nothing), and block decomposition's HiGHS
        solve is cancelled. The status is then ``"interrupted"``."""
        self.interrupt.set()
        now = time.time()
        self.deadline = now if self.deadline is None else min(self.deadline, now)
        if self.block_holder is not None:
            self.block_holder.cancel()

    def cutoff(self, box=True):
        """Nodes whose bound reaches this hold no solution worth finding:
        below the incumbent by the gaps, and (``box``) below the largest
        objective any point of the root box has (``box_cutoff``), which
        lets infeasible node LPs stop on their bound before an incumbent."""
        if self.incumbent is None:
            return (
                min(self.initial_cutoff, self.box_cutoff)
                if box
                else self.initial_cutoff
            )
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

    def root_closed(self):
        """Whether, before the tree, the global bound (the root's rounds'
        best, ``active_bound``, raised by the objective levels shown empty
        and the restart floor) reaches the cutoff, so the incumbent is
        within the gaps and the search is over. The rounds' re-solves stop
        on their own bound only (``bound_target``): on 30n20b8 the levels
        lane had the bound at 302 with the root LP's at 129, and the rounds,
        root heuristics and root strong branching ran 22 s past an
        incumbent of 302."""
        return self.incumbent is not None and self.global_bound() >= self.cutoff()

    def global_bound(self):
        bound = min((node.bound for node in self.open_nodes()), default=np.inf)
        # Nodes a batch has evaluated but not yet split are in no list.
        bound = min(
            bound, min((d["bound"] for d in self.deferred_branches), default=np.inf)
        )
        bound = min(
            bound, min((node.bound for node in self.batch_waiting), default=np.inf)
        )
        bound = min(
            bound, min((node.bound for node in self.pipeline_nodes()), default=np.inf)
        )
        bound = min(bound, self.closed_floor, self.active_bound)
        # The rounds before a restart bounded every solution below the
        # cutoff, and the restarted root box keeps all of them.
        bound = max(bound, self.bound_floor)
        # Every solution is in an objective level not yet shown empty
        # (``objective_levels``).
        bound = max(bound, self.level_floor())
        if self.incumbent is not None:
            bound = min(bound, self.incumbent[0])
        return bound

    def close(self, bound, rigorous=True):
        self.closed_floor = min(self.closed_floor, bound)
        self.bound_rigorous &= rigorous

    def gap(self, bound):
        if self.incumbent is None:
            return np.inf
        offset = self.mip.offset
        return _relative_gap(self.incumbent[0] + offset, bound + offset)

    def model_value(self, value):
        """A min-form value (without offset) in the model's own sense."""
        return self.mip.sign * (value + self.mip.offset)

    # -- node keys -----------------------------------------------------------
    def diving(self):
        return self.node_selection == "depth_first" or (
            self.node_selection == "hybrid"
            and self.incumbent is None
            and not self.cutoff_from_start
        )

    def key(self, node, order):
        if self.diving():
            return (-order,)
        return (node.bound, -node.depth, order)

    def push(self, node):
        if node.cut_version is None:
            node.cut_version = self.cut_version
        order = next(self.counter)
        heapq.heappush(self.open, (self.key(node, order), order, node))

    def rekey(self):
        self.open = [(self.key(n, o), o, n) for _, o, n in self.open]
        heapq.heapify(self.open)

    def open_nodes(self):
        nodes = [entry[2] for entry in self.open]
        if self.plunge_node is not None:
            nodes.append(self.plunge_node)
        # The exact lane's (``exact_nodes``): reserved, queued or running,
        # and finished but not yet evaluated.
        if self.exact_lane is not None:
            nodes += [entry[2] for entry in self.exact_reserved]
            nodes += self.exact_lane.nodes()
            nodes += self.exact_unevaluated
        return iter(nodes)

    def plunging(self):
        """Best-bound search with plunges: after each best-bound node, up to
        ``plunge_depth`` steps follow its preferred child, so the search
        keeps reaching leaves (and incumbents) while it raises the bound."""
        return (
            self.node_selection == "hybrid"
            and (self.incumbent is not None or self.cutoff_from_start)
            and self.plunge_depth > 0
        )

    # -- LP solves -----------------------------------------------------------
    def node_settings(self):
        """The precision context of node LPs (``node_precision``)."""
        return jo.local_settings(precision=self.node_precision)

    def node_lp(self, lower, upper):
        """The LP over a box, at the search's precision (root, cut rounds,
        heuristics, infeasibility checks)."""
        return self._box_lp(self.mip.lp, self.values, self.dtype, lower, upper)

    def tree_lp(self, lower, upper):
        """The LP over a box, at the node LPs' precision (``node_precision``)."""
        with self.node_settings():
            return self._box_lp(
                self.node_mip_lp, self.node_values, self.node_dtype, lower, upper
            )

    def lp_lower(self, lower):
        """The lower bounds an LP over the box takes: the integers' from the
        box; with ``branch_implied``, the implied integers' rounded down to
        a whole number (so branching bounds, which are whole, go in exactly,
        and rounding error in a propagated one cannot cut the LP off); the
        MIP's for the other continuous variables."""
        out = np.where(self.is_int, lower, self.mip.lower)
        k = self.branch_implied_cols
        if k is not None:
            out[k] = np.maximum(np.floor(lower[k] + _ROUND_SLACK), self.mip.lower[k])
        return out

    def lp_upper(self, upper):
        """The upper bounds an LP over the box takes (``lp_lower``)."""
        out = np.where(self.is_int, upper, self.mip.upper)
        k = self.branch_implied_cols
        if k is not None:
            out[k] = np.minimum(np.ceil(upper[k] - _ROUND_SLACK), self.mip.upper[k])
        return out

    def _box_lp(self, lp, values, dtype, lower, upper):
        # Integer tightenings go into the LP; continuous ones only into the
        # Lagrangian bound, so rounding error in them cannot cut off the LP
        # (branched implied integers: rounded outward, ``lp_lower``).
        lp_lower = self.lp_lower(lower)
        lp_upper = self.lp_upper(upper)
        return lp.with_values(
            values._replace(
                lower_bounds=jnp.asarray(lp_lower, dtype),
                upper_bounds=jnp.asarray(lp_upper, dtype),
            )
        )

    def solve_lp(self, lp, warm, max_epochs, **overrides):
        kwargs = dict(self.lp_kwargs, **overrides)
        if warm is not None:
            warm = _fit_warm(warm, lp)
            # A warm start from an LP at another precision (root or node).
            with jo.local_settings(precision=kwargs.get("precision")):
                warm = _cast_warm(warm, _wide(lp.c.dtype))
            kwargs.update(initial_solution=warm[0], initial_opt_state=warm[1])
        result = jl.solve(
            lp,
            max_epochs=max_epochs,
            max_seconds=self.seconds_left(),
            cancel=self.interrupt,
            **kwargs,
        )
        if result["stop_reason"] == "cancelled":
            # Stopped by ``request_interrupt``: as ``jl.solve`` reports Ctrl-C.
            result["stop_reason"] = "interrupted"
        self.lp_epochs += result["epochs"]
        return result

    def result_terms(self, result, lower, upper):
        """The Lagrangian terms of a node LP's result (``dual_terms``; an
        exact lane result's as an exact vertex's, ``vertex_terms``)."""
        if result.get("stop_reason") == "exact":
            return self.vertex_terms(result["solution"], lower, upper)
        return self.dual_terms(result["solution"])

    def exact_bound(self, solution, lower, upper):
        """``lp_bound`` of a HiGHS solution (the exact lane's): its reduced
        costs at infinite bounds that are round-off are zeroed first
        (``vertex_terms``), or a −1e-17 one makes the bound −inf. The same
        function (``_exact_side_bound``) bounds the lane's solutions on its
        threads (``exact_lane_bounds``)."""
        started = time.perf_counter()
        out = _exact_side_bound(
            self.mip, self.A_T, solution.dual_eq, solution.dual_ineq,
            lower, upper, self.is_int, self.reduced_cost_tol,
        )
        self.bound_seconds += time.perf_counter() - started
        if out is None:
            return self.bound_over(self.vertex_terms(solution, lower, upper), lower, upper)
        return out

    def solve_node_lp(self, lower, upper, warm, max_epochs):
        """A node's LP over its box by PDHG at ``node_precision``, with the
        node options (``node_lp_tolerance``, ``node_lp_options``) and, with
        ``node_bound_stop``, the bound target."""
        target = self.node_bound_target()
        return self.solve_lp(
            self.tree_lp(lower, upper),
            warm,
            max_epochs,
            **self.node_lp_kwargs,
            **({} if target is None else {"bound_target": target}),
        )

    def detect(self, lp, result, max_epochs):
        """``detect_infeasibility`` warm-started from a solve's last point."""
        warm = (None, None)
        if result is not None:
            warm = _cast_warm(
                (result["solution"], result["opt_state"]), _wide(lp.c.dtype)
            )
        out = jl.detect_infeasibility(
            lp,
            tol=self.infeas_tol,
            max_epochs=max_epochs,
            max_seconds=self.seconds_left(),
            initial_solution=warm[0],
            initial_opt_state=warm[1],
            **self.lp_kwargs,
        )
        self.lp_epochs += out["epochs"]
        return out

    def duals(self, solution, m_ineq=None):
        m_ineq = self.m_ineq if m_ineq is None else m_ineq
        return np.concatenate(
            [
                np.asarray(solution.dual_eq, np.float64)[: self.mip.n_eq],
                np.asarray(solution.dual_ineq, np.float64)[:m_ineq],
            ]
        )

    def dual_terms(self, solution):
        """``lagrangian_terms`` of a node's duals over the searched rows."""
        started = time.perf_counter()
        terms = lagrangian_terms(
            self.mip.c,
            self.mip.A,
            self.mip.b,
            self.mip.n_eq,
            self.duals(solution, self.m_ineq),
            A_T=self.A_T,
        )
        self.bound_seconds += time.perf_counter() - started
        return terms

    def vertex_terms(self, vertex, lower, upper):
        """``dual_terms`` of an exact vertex (a crossover HiGHS proved
        optimal). Its
        reduced costs that point at an infinite bound of the box are
        round-off in ``c + Aᵀy`` (on buildingenergy 14,432 of them, at most
        1.8e-11), which would make the bound drop them and lose its rigour:
        they are zeroed, as a simplex solver does."""
        constant, r = self.dual_terms(vertex)
        roundoff = (
            ((r > 0) & ~np.isfinite(lower)) | ((r < 0) & ~np.isfinite(upper))
        ) & (
            np.abs(r)
            <= _EXACT_DUAL_ROUNDOFF * (1.0 + np.abs(self.mip.c).max(initial=0.0))
        )
        return constant, np.where(roundoff, 0.0, r)

    def bound_over(self, terms, lower, upper):
        """The Lagrangian bound of ``terms`` over a propagated box, widened a
        little on continuous variables against rounding error."""
        started = time.perf_counter()
        kernel = _numba_bound()
        if kernel is not None:
            r = _f64(terms[1])
            lower, upper = _f64(lower), _f64(upper)
            if r.ndim == 1 and lower.shape == r.shape == upper.shape:
                bound, rigorous = kernel.widened_box_bound(
                    float(terms[0]), r, lower, upper, self.is_int,
                    float(self.reduced_cost_tol),
                )
                self.bound_seconds += time.perf_counter() - started
                return float(bound), bool(rigorous)
        cont = ~self.is_int
        lower = np.where(cont, lower - 1e-7 * (1.0 + np.abs(lower)), lower)
        upper = np.where(cont, upper + 1e-7 * (1.0 + np.abs(upper)), upper)
        bound = box_bound(
            *terms, lower, upper, reduced_cost_tolerance=self.reduced_cost_tol
        )
        self.bound_seconds += time.perf_counter() - started
        return bound

    def lp_bound(self, solution, lower, upper):
        """The Lagrangian bound of a node's duals over its propagated box."""
        return self.bound_over(self.dual_terms(solution), lower, upper)

    def repair_terms(self, result, terms, raw, rigorous, lower, upper):
        """A device LP's ``(terms, raw, rigorous)`` with its duals repaired
        (``DualRepair``, ``dual_repair``) when that bounds the box higher;
        as given otherwise, and for an exact lane result (its duals are a
        vertex's) or a box its bound already closes."""
        repairer = self.dual_repairer
        if (
            repairer is None
            or result.get("stop_reason") == "exact"
            or raw >= self.cutoff()
        ):
            return terms, raw, rigorous
        started = time.perf_counter()
        solution = result["solution"]
        y = repairer(
            self.mip.c, self.A_T, self.duals(solution, self.m_ineq),
            solution.primal, lower, upper,
        )
        stats = self.dual_repair_stats
        stats["tried"] += 1
        if y is not None:
            new_terms = lagrangian_terms(
                self.mip.c, self.mip.A, self.mip.b, self.mip.n_eq, y, A_T=self.A_T
            )
            new_raw, new_rigorous = self.bound_over(new_terms, lower, upper)
            # A rigorous bound only gives way to a higher rigorous one.
            if new_raw > raw and (new_rigorous or not rigorous):
                stats["improved"] += 1
                terms, raw, rigorous = new_terms, new_raw, new_rigorous
        stats["seconds"] += time.perf_counter() - started
        return terms, raw, rigorous

    def reduced_cost_fix(self, terms, lower, upper):
        """``reduced_cost_tightening`` of a box against ``proof_cutoff`` (the
        largest objective a better solution can have: with an objective
        step, the multiple of it at or below the cutoff), then propagation.
        Returns ``(lower, upper, infeasible)``, where ``infeasible`` means
        the box holds no solution below the cutoff.

        The step matters when reduced costs are about one step: on
        CMS750_4 (costs ±1, incumbent 252) every nonzero root reduced cost
        is exactly 1, so with the cutoff's room (251.5 − bound) a node
        fixed nothing until its bound passed 250.5, and with the step's
        (251 − bound) it fixes them all once its bound passes 250."""
        cutoff = self.proof_cutoff()
        if not np.isfinite(cutoff):
            return lower, upper, False
        bound, rigorous = self.bound_over(terms, lower, upper)
        if not rigorous:
            return lower, upper, False
        new_lower, new_upper, changed = reduced_cost_tightening(
            terms[1], bound, cutoff, lower, upper, self.is_int
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
        if self.root_redcost_propagation and self.conflicts is not None:
            # The root's reduced-cost row, at this cutoff, for every node.
            self.conflicts.set_cutoff(self.proof_cutoff())
            self.conflicts.set_global(
                *self.root_terms, self.root_lower, self.root_upper
            )
        return True

    def global_fix(self, terms):
        """Reduced-cost fixing of the root box with any LP's Lagrangian terms
        (``global_redcost_fixing``): ``c·x >= constant + r·x`` holds for
        every feasible ``x`` whatever the duals, so its fixings hold for
        every solution better than the cutoff. An emptied root box closes
        every node as it is expanded."""
        if self.incumbent is None:
            return
        before = self.reduced_cost_tightenings
        lower, upper, _ = self.reduced_cost_fix(terms, self.root_lower, self.root_upper)
        if lower is self.root_lower and upper is self.root_upper:
            return
        self.global_redcost_fixed += self.reduced_cost_tightenings - before
        self.root_lower, self.root_upper = lower, upper

    def fixed_share(self):
        """The share of the integers free at the start of this round that
        the root box has fixed since (``root_lower == root_upper``)."""
        if not self.restart_free:
            return 0.0
        free = np.count_nonzero(self.is_int & (self.root_lower < self.root_upper))
        return (self.restart_free - free) / self.restart_free

    def restart_due(self, share):
        """Whether to restart: restarts left (``restarts``), time left, and
        at least ``share`` of this round's free integers fixed since."""
        return (
            len(self.restart_log) < self.max_restarts
            and not self.out_of_time()
            and self.fixed_share() >= share
        )

    def restart(self):
        """
        End this round and start the next from the root box reduced-cost
        fixing has left: the tree and the root's LP state go, and probing,
        coefficient tightening and the cut rounds are redone over the
        smaller box. The cuts in the pool that were tight or had a dual at
        their last solve (``age`` 0) join the searched MIP's rows
        (``self.base``) for good: separated over the root box, they are
        valid for every solution the restarted box holds, so the new root
        starts from about the bound they gave rather than cutting again
        from the LP alone (on bab5 twenty rounds take 50–60 s and land
        anywhere from −107,266 to −108,928). The new rounds' pool starts
        empty, on top of them. The incumbent, pseudocosts, conflict pool,
        symmetry rows and background heuristics stay, and the round's bound
        stays as a floor (``bound_floor``). Returns the box for
        ``_prepare``.
        """
        share = self.fixed_share()
        free = int(np.count_nonzero(self.is_int & (self.root_lower < self.root_upper)))
        self.bound_floor = max(self.bound_floor, self.global_bound())
        kept = [cut for cut in self.cut_list if cut["age"] == 0]
        if kept:
            lengths = [cut["idx"].size for cut in kept]
            A_rows = sp.csr_matrix(
                (
                    np.concatenate(
                        [np.asarray(cut["coef"], np.float64) for cut in kept]
                    ),
                    np.concatenate([np.asarray(cut["idx"], np.int64) for cut in kept]),
                    np.concatenate([[0], np.cumsum(lengths)]),
                ),
                shape=(len(kept), self.base.num_variables),
            )
            self.base = self.base.with_rows(A_rows, [cut["rhs"] for cut in kept])
        self.restart_log.append(
            {
                "seconds": time.time() - self.start,
                "nodes": self.nodes,
                "fixed_share": share,
                "free_integers": free,
                "cuts_kept": len(kept),
                "bound": self.model_value(self.bound_floor),
            }
        )
        if self.verbose:
            print(
                f"Restart {len(self.restart_log)}: {100 * share:.1f}% of the free "
                f"integers fixed ({self.restart_free - free}), {free} left free, "
                f"{len(kept)} of {len(self.cut_list)} cuts kept as rows, "
                f"{self.nodes} nodes, {time.time() - self.start:.1f}s"
            )
        # The old tree.
        self.open = []
        self.exact_reserved = []
        self.plunge_node = None
        self.plunge_left = 0
        self.deferring = False
        self.deferred_branches = []
        self.deferred_roundings = []
        self.batch_waiting = []
        self.closed_floor = np.inf
        self.unresolved_floor = np.inf
        self.active_bound = -np.inf
        # The cut pool and the root's LP state.
        self.cut_list = []
        self.set_mip(self.base)
        self.root_vertex_state = None
        self.root_vertex_x = None
        self.root_terms = None
        self.root_terms_rigorous = False
        self.root_fixed_at = None
        return self.root_lower.copy(), self.root_upper.copy()

    def presolved_round(self, lower, upper):
        """
        The rest of the search on a presolved MIP (``restart_presolve``): at
        a restart, HiGHS's presolve runs on the searched MIP
        (``self.base``: the kept cuts and symmetry rows included) over the
        restarted root box ``[lower, upper]``, whose fixed columns and the
        rows they empty a restart alone keeps. When it removes at least
        ``_RESTART_PRESOLVE_MIN_SHARE`` of the columns, a new search of the
        reduced MIP runs to the deadline with this one's cutoff (HiGHS
        cannot map the incumbent into the reduced columns, so the search
        starts from the cutoff alone, as ``solve(presolve=True)`` does
        from a ``mip_start``), and the restarts and nodes left. Its points
        come back through HiGHS's postsolve (marked ``"Q"``), and its bound
        and status are this search's. Returns the result, or ``None`` to
        restart as usual (presolve reduced too little, or failed).
        """
        import highspy

        started = time.time()
        base = self.base
        dtype = base.lp.c.dtype
        lp = base.lp.with_values(
            base.lp.values()._replace(
                lower_bounds=jnp.asarray(self.lp_lower(lower), dtype),
                upper_bounds=jnp.asarray(self.lp_upper(upper), dtype),
            )
        )
        # Min form with the base's offset: HiGHS's objective is this
        # search's value plus ``base.offset``, and so is the reduced MIP's
        # (its own value plus its offset).
        boxed = MIP(lp, self.is_int, base.offset, False, base.implied)
        try:
            model = to_highs(boxed)
            # Named by index: the reduced MIP's names say which of these
            # columns each of its own is (``reduced_start``).
            model.col_names_ = [f"c{j}" for j in range(base.num_variables)]
            highs = _highs_mip(model)
            status = _presolve(highs)
        except Exception:
            return None
        PS = highspy.HighsPresolveStatus
        n, m = base.num_variables, base.A.shape[0]
        entry = dict(
            presolve=status.name,
            columns=(n, None),
            rows=(m, None),
            presolve_seconds=time.time() - started,
        )

        def postsolved(reduced_x):
            solution = highspy.HighsSolution()
            solution.col_value = list(np.asarray(reduced_x, np.float64))
            solution.value_valid = True
            solution.dual_valid = False
            highs.postsolve(solution)
            return np.array(highs.getSolution().col_value, dtype=np.float64)

        def no_better_than(value):
            """Whether the incumbent is at least as good as ``value`` (up
            to round-off in postsolve and in the start's completion)."""
            return self.incumbent is not None and self.incumbent[0] <= value + 1e-6 * (
                1.0 + abs(value)
            )

        if status == PS.kInfeasible:
            # The restarted box holds only solutions better than the
            # incumbent (reduced-cost fixing): none is left.
            self.restart_presolve_log.append(entry)
            if self.verbose:
                print("Restart presolve: the root box is infeasible")
            return self.empty_root()
        if status == PS.kReducedToEmpty:
            # Presolve solved the box: its point is the best in it.
            x = postsolved(np.zeros(0))
            self.try_incumbent(x, "Q", None)
            entry["columns"], entry["rows"] = (n, 0), (m, 0)
            self.restart_presolve_log.append(entry)
            if no_better_than(float(base.c @ x)):
                return self.empty_root()
            return None
        if status != PS.kReduced:
            return None
        reduced = _mip_from_highs_lp(highs.getPresolvedLp(), maximise=False)
        entry["columns"] = (n, reduced.num_variables)
        entry["rows"] = (m, reduced.A.shape[0])
        if reduced.num_variables > (1.0 - _RESTART_PRESOLVE_MIN_SHARE) * n:
            entry["used"] = False
            self.restart_presolve_log.append(entry)
            if self.verbose:
                print(
                    f"Restart presolve: {n} -> {reduced.num_variables} variables, "
                    f"too few removed; restarting as usual"
                )
            return None
        entry["used"] = True
        if self.verbose:
            print(
                f"Restart presolve: {n} -> {reduced.num_variables} variables, "
                f"{m} -> {reduced.A.shape[0]} rows ({entry['presolve_seconds']:.1f}s); "
                "the search continues on the reduced MIP"
            )
        # The background lanes are this search's: the new one has its own.
        self.stop_lanes()
        shift = base.offset - reduced.offset  # this search's value -> reduced's
        start = self.reduced_start(highs.getPresolvedLp(), reduced)
        initial_cutoff = self.cutoff(box=False) + shift
        if start is not None:
            # Room for the start itself, at the incumbent's value: once it
            # is in, the reduced search's cutoff is this one's again.
            value = self.incumbent[0] + shift
            initial_cutoff = value + 1e-6 * (1.0 + abs(value))
        settings = dict(
            self.settings,
            deadline=self.deadline,
            start=self.start,
            max_nodes=None if self.max_nodes is None else max(self.max_nodes - self.nodes, 0),
            initial_cutoff=initial_cutoff,
            cutoff_from_start=self.incumbent is not None,
            mip_start=start,
            # ``self.base`` has the symmetry rows already.
            symmetry=False,
            restarts=max(self.max_restarts - len(self.restart_log), 0),
            interrupt=self.interrupt,
        )
        # The inner search prints its own table meanwhile.
        self.log_paused = True
        try:
            inner = _BranchAndBound(reduced, **settings).run()
        except KeyboardInterrupt:
            return self.result("interrupted")
        finally:
            self.log_paused = False
        self.nodes += inner["nodes"]
        # Its restarts are this search's too (``result["restarts"]``).
        self.restart_log += inner.get("restart_log", [])
        self.restart_presolve_log += inner.get("restart_presolve", [])
        entry.update(
            status=inner["status"], nodes=inner["nodes"], seconds=time.time() - started
        )
        self.restart_presolve_log.append(entry)
        found = None
        if inner["x"] is not None:
            x = postsolved(inner["x"])
            found = float(base.c @ x)
            self.try_incumbent(x, "Q", None)
        inner_bound = inner["bound"]
        bound = (
            -np.inf
            if inner_bound is None or not np.isfinite(inner_bound)
            else float(inner_bound) - base.offset
        )
        proved = inner["status"] in ("optimal", "infeasible")
        if proved and (found is None or no_better_than(found)):
            # Nothing in the box beats the incumbent (or the reduced
            # search's point, now the incumbent).
            return self.empty_root()
        # The reduced search's bound holds for every solution in the box.
        self.active_bound = np.inf
        self.close(bound, inner.get("bound_rigorous", True))
        status = inner["status"]
        if proved or status == "unresolved":
            # Proved for a point postsolve did not give back feasible.
            self.unresolved_floor = min(self.unresolved_floor, bound)
            status = "optimal"
        return self.result(status)

    def reduced_start(self, presolved_lp, reduced):
        """The incumbent as a start (``mip_start``) for the reduced MIP of
        a presolved restart (``presolved_round``), or ``None`` (no
        incumbent, or the columns' names lost). HiGHS's presolve keeps the
        names, which ``presolved_round`` sets to the base's column indices,
        so each reduced column takes the incumbent's value there. Presolve
        moves the costs and offset of the columns it keeps (substitutions,
        dual fixings), so the continuous values need not fit the reduced
        rows; the search's start goes through ``try_incumbent``, which keeps
        the integers and re-solves the continuous variables. On pg5_34 the
        reduced search had only the cutoff, 0.3% from its bound, and no
        heuristic found a point below it in 50 s, so RINS, proximity search
        and crossover, which had found the incumbents, never ran."""
        if self.incumbent is None:
            return None
        try:
            names = list(presolved_lp.col_names_)
            cols = np.array([int(name[1:]) for name in names], np.int64)
        except (ValueError, TypeError, IndexError):
            return None
        if cols.size != reduced.num_variables or np.any(cols >= self.base.num_variables):
            return None
        return np.asarray(self.incumbent[1], np.float64)[cols]

    def stop_lanes(self):
        """Shut the background lanes down for good (``run`` does the same
        when it returns)."""
        for name in ("workers", "exact_lane", "submip_lane", "level_lane", "probe_lane"):
            lane = getattr(self, name)
            if lane is not None:
                lane.shutdown()
                setattr(self, name, None)

    def close_all(self):
        """Close every open node: none holds a solution below the cutoff.
        So do the pipeline's (``pipeline_batches``): its batch on the device
        is dropped, and its children are dropped unread; so are the exact
        lane's (``exact_nodes``)."""
        self.close(self.cutoff_bound())
        self.open = []
        self.plunge_node = None
        self.in_flight = None
        self.pending_children = None
        self.exact_reserved = []
        if self.exact_lane is not None:
            self.exact_lane.drop()

    def empty_root(self):
        """The result when the root box holds no solution below the cutoff
        (propagation, probing, the root LP or reduced-cost fixing shows it
        empty): every node closes, and the incumbent, if any, is optimal.
        After a restart the box only ever held the solutions better than
        the incumbent (reduced-cost fixing, and with
        ``objective_propagation`` the objective row), so an empty one there
        is a proof of optimality, not of infeasibility."""
        self.close_all()
        self.active_bound = np.inf
        return self.result("optimal")

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
        self.conflicts.add(*terms, lower, upper, self.root_lower, self.root_upper, kind)

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
        self.bump_conflict(lower, upper, cols=kept)
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
            changed = int(np.count_nonzero((new_lower != lower) | (new_upper != upper)))
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
            repaired = np.asarray(result["solution"].primal, np.float64)
            candidate = np.clip(repaired, fixed_lower, fixed_upper)
            candidate[self.is_int] = fixed_lower[self.is_int]
            if not self.original.is_feasible(candidate, self.feas_tol):
                self.failed_roundings.add(candidate[self.is_int].tobytes())
                return False
        value = float(mip.c @ candidate)
        if self.crossover_seconds and self.submip_lane is not None:
            self.remember_solution(value, candidate)
        if value >= self.initial_cutoff:
            return True  # feasible, but no better than the parent search's cutoff
        if self.incumbent is not None and value >= self.incumbent[0] - 1e-12 * (
            1.0 + abs(self.incumbent[0])
        ):
            return True
        switch = self.incumbent is None
        self.incumbent = (value, candidate)
        if self.early_jump is not None:
            self.early_jump.set()  # the early root jumps stop
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
        self.log(source, bound, force=True)
        if switch and self.node_selection == "hybrid":
            self.rekey()
        if self.use_one_opt and source != "1":
            # Move single integers to better values while the rows allow
            # (``one_opt``); against the MIP as given, so the point stays
            # feasible.
            if self._original_csc is None:
                self._original_csc = self.original.A.tocsc()
            better = one_opt(self.original, candidate, A_csc=self._original_csc)
            if better is not None:
                before = self.incumbent[0]
                self.try_incumbent(better, "1", warm)
                if self.incumbent[0] < before:
                    self.one_opt_finds += 1
        return True

    # -- branching -----------------------------------------------------------
    def pseudocost_scores(self, x, candidates):
        """The product score of each candidate's estimated gains."""
        down, up = self.pseudocost_estimates(x, candidates)
        return np.maximum(down, 1e-6) * np.maximum(up, 1e-6)

    def pseudocost_estimates(self, x, candidates):
        """Each candidate's estimated down and up bound gains
        (``_pseudocost_estimates`` over this search's pseudocosts)."""
        return _pseudocost_estimates(self.pc_sum, self.pc_count, x, candidates)

    def choose_variable(self, x, fractional):
        candidates = np.flatnonzero(fractional)
        if self.branching == "most_fractional" or not self.pc_count.any():
            f = x[candidates] - np.floor(x[candidates])
            return candidates[np.argmax(np.minimum(f, 1.0 - f))]
        scores = self.branching_scores(
            candidates, self.pseudocost_scores(x, candidates)
        )
        return candidates[np.argmax(scores)]

    def branching_scores(self, candidates, gains):
        """
        The hybrid branching score of each candidate, as SCIP's and HiGHS's:
        its bound-gain product (``gains``: pseudocost estimates, or strong
        branching's child gains) first, then its conflict, cutoff and
        inference history. Each part ``v`` is mapped to ``1 − 1/(1 + v/v̄)``
        against its mean ``v̄`` over the candidates, and they are weighed
        1, ``_CONFLICT_WEIGHT``, ``_CUTOFF_WEIGHT`` and
        ``_INFERENCE_WEIGHT``. The gains decide; the history settles near
        ties (on ns1830653 most branchings are at x = 0.5).

        * conflict: ``conflict_score``, how often (decaying) the variable
          was tightened in a box that closed;
        * cutoff: the share of its children, either way, that closed at
          once (infeasible after propagation, past the cutoff, or by their
          LP), summed over the two directions;
        * inference: the product over the directions of the mean bounds
          propagation fixed in its children beyond the split.

        ``-inf`` gains (fixed candidates) stay ``-inf``.
        """
        gains = np.asarray(gains, np.float64)

        def mapped(values):
            finite = values[np.isfinite(values) & (values > 0)]
            mean = finite.mean() if finite.size else 0.0
            if mean <= 0.0:
                return np.zeros_like(values)
            return 1.0 - 1.0 / (1.0 + np.maximum(values, 0.0) / mean)

        made = self.branch_children[:, candidates]
        cutoff = (self.branch_cutoffs[:, candidates] / np.maximum(made, 1.0)).sum(
            axis=0
        )
        count = self.inference_count[:, candidates]
        mean_inferences = self.inference_sum[:, candidates] / np.maximum(count, 1.0)
        inference = np.prod(
            np.where(count > 0, np.maximum(mean_inferences, 1e-6), 1e-6), axis=0
        )
        score = (
            mapped(np.where(np.isfinite(gains), gains, 0.0))
            + _CONFLICT_WEIGHT * mapped(self.conflict_score[candidates])
            + _CUTOFF_WEIGHT * mapped(cutoff)
            + _INFERENCE_WEIGHT * mapped(inference)
        )
        return np.where(np.isfinite(gains), score, -np.inf)

    def record_cutoff(self, branch):
        """A child of the branching ``branch`` (``_Node.branch``: var,
        down, ...) closed at once."""
        if branch is None:
            return
        self.branch_cutoffs[0 if branch[1] else 1, branch[0]] += 1

    def bump_conflict(self, lower, upper, cols=None):
        """The variables tightened (against the root box) in a box that just
        closed, or ``cols``, gain ``conflict_bump``, which then grows by
        ``1/_CONFLICT_DECAY``: newer conflicts weigh more (VSIDS)."""
        if cols is None:
            cols = np.flatnonzero(
                (lower > self.root_lower) | (upper < self.root_upper)
            )
        if np.size(cols) == 0:
            return
        self.conflict_score[cols] += self.conflict_bump
        self.conflict_bump /= _CONFLICT_DECAY
        if self.conflict_bump > 1e100:  # rescale before overflow
            self.conflict_score /= self.conflict_bump
            self.conflict_bump = 1.0

    def integral_split_variable(self, x, lower, upper, terms):
        """The integer to split at a node whose LP point is integral but
        did not settle it, or ``None`` when every integer is fixed in the
        box. Preferred: a free integer in the rows of the MIP as given that
        the rounded point breaks, with the largest ``|coefficient|`` there
        (moving it can mend them); else the free integer with the largest
        ``|reduced cost|``."""
        free = self.is_int & (lower < upper)
        if not free.any():
            return None
        point = np.clip(np.asarray(x, np.float64), lower, upper)
        point[self.is_int] = np.round(point[self.is_int])
        rows = _violated_rows(self.original, point, self.feas_tol)
        if rows.size:
            weight = np.asarray(
                abs(self.original.A[rows]).max(axis=0).todense()
            ).ravel()
            weight = np.where(free, weight, 0.0)
            if weight.max() > 0.0:
                return int(np.argmax(weight))
        score = np.where(free, np.abs(terms[1]), -1.0)
        return int(np.argmax(score))

    def record_gain(self, var, down, gain):
        self.pc_sum[0 if down else 1, var] += gain
        self.pc_count[0 if down else 1, var] += 1

    def update_pseudocost(self, node, value):
        var, down, distance, parent_value = node.branch
        if parent_value is None or distance <= 0:
            return
        self.record_gain(var, down, max(value - parent_value, 0.0) / distance)

    def tree_elapsed(self):
        """Seconds spent in the tree so far, over every round between
        restarts (0 outside it)."""
        if self.tree_clock is None:
            return self.tree_seconds
        return self.tree_seconds + time.time() - self.tree_clock[0]

    def heuristic_clock(self):
        """The node count the tree heuristics are scheduled by: the nodes
        finished, plus the time tree cut refreshes (``tree_cuts``) took
        counted as the nodes it would have been without them (at the tree's
        mean time a node without them). Cut rounds slow the nodes, and the
        heuristics run every so many nodes: on ns1830653 three rounds a node
        took the search from 1,173 nodes in 81 s to 165 in 140 s, and the
        incumbent RINS had found at node 1,638 was late."""
        if not self.use_tree_cuts or self.tree_cut_seconds <= 0.0 or self.nodes <= 0:
            return float(self.nodes)
        plain = self.tree_elapsed() - self.tree_cut_seconds
        if plain <= 0.0:
            return float(self.nodes)
        return self.nodes + self.tree_cut_seconds * self.nodes / plain

    def heuristic_due(self, name, every):
        """Whether the heuristic ``name``, run every ``every`` nodes, is due:
        ``nodes % every == 0`` without tree cuts; with them, once
        ``heuristic_clock`` passes the next multiple of ``every`` (at the
        first node that asks after it)."""
        if not every:
            return False
        if not self.use_tree_cuts:
            return self.nodes % every == 0
        period = int(self.heuristic_clock() // every)
        if period <= self.heuristic_fired.get(name, 0):
            return False
        self.heuristic_fired[name] = period
        return True

    def strong_branching_on(self, node):
        """Whether reliability branching strong-branches at ``node``: within
        ``strong_max_depth``."""
        return (
            self.branching == "reliability"
            and self.strong_candidates > 0
            and (self.strong_max_depth is None or node.depth <= self.strong_max_depth)
            and not self.out_of_time()
        )

    def choose_branch(
        self,
        node,
        x,
        fractional,
        lower,
        upper,
        warm,
        terms,
        bound,
        pending=None,
        raw=None,
        exact_strong=None,
        exact=False,
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
        if exact:
            # An exact lane node (``exact_nodes``): scored by the lane's
            # exact pseudocosts, with its strong-branching children
            # (``exact_strong``: ``({var: (down, up)}, solved)``) if it has
            # any.
            est_down, est_up = _pseudocost_estimates(
                self.exact_pc_sum, self.exact_pc_count, x, candidates
            )
            score = np.maximum(est_down, 1e-6) * np.maximum(est_up, 1e-6)
            if exact_strong is None:
                best = np.argmax(self.branching_scores(candidates, score))
                return candidates[best], None, lower, upper, bound
            children, solved = exact_strong
            return self.finish_choice(
                node, children, solved, candidates, score, est_down, est_up,
                x, lower, upper, warm, bound, raw,
            )
        if not self.strong_branching_on(node):
            return self.choose_variable(x, fractional), None, lower, upper, bound
        est_down, est_up = self.pseudocost_estimates(x, candidates)
        score = np.maximum(est_down, 1e-6) * np.maximum(est_up, 1e-6)
        started = time.time()
        if self.strong_backend == "jax":
            if pending is None:  # every candidate was reliable
                best = np.argmax(self.branching_scores(candidates, score))
                return candidates[best], None, lower, upper, bound
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
                best = np.argmax(self.branching_scores(candidates, score))
                return candidates[best], None, lower, upper, bound
            # High-impact variables first (``branching_impact``).
            priority = score[unreliable] * self.impact[candidates[unreliable]]
            chosen = unreliable[np.argsort(-priority, kind="stable")]
            chosen = candidates[chosen[: self.strong_candidates]]
            self.strong_branch_nodes += 1
            try:
                children = self.strong_branch(
                    chosen, x, lower, upper, warm, terms, bound, raw
                )
            finally:
                self.strong_branch_seconds += time.time() - started
            solved = set(chosen.tolist())
        return self.finish_choice(
            node, children, solved, candidates, score, est_down, est_up,
            x, lower, upper, warm, bound, raw,
        )

    def finish_choice(
        self, node, children, solved, candidates, score, est_down, est_up,
        x, lower, upper, warm, bound, raw,
    ):
        """``choose_branch``'s end, given strong-branching ``children``
        (``{var: (down, up)}`` of ``_StrongChild``) and the candidates they
        solved: ``apply_strong``, then the best-scoring candidate. Returns
        ``choose_branch``'s result."""
        applied = self.apply_strong(
            children,
            solved,
            candidates,
            score,
            est_down,
            est_up,
            x,
            lower,
            upper,
            bound,
            raw,
        )
        if applied is None:
            return None
        lower, upper, strong_bound, results = applied
        score = self.branching_scores(candidates, score)
        best = int(np.argmax(score))
        if not np.isfinite(score[best]):
            # Every candidate was fixed: solve the node again in its box.
            self.push(
                _Node(
                    strong_bound,
                    node.depth,
                    self.compress(lower, upper),
                    warm,
                    None,
                )
            )
            return None
        var = candidates[best]
        return var, results.get(var), lower, upper, strong_bound

    def apply_strong(
        self,
        children,
        solved,
        candidates,
        score,
        est_down,
        est_up,
        x,
        lower,
        upper,
        bound,
        raw,
    ):
        """
        Use strong-branching ``children`` (``{var: (down, up)}`` of
        ``_StrongChild``) at a node: a candidate with both children closed
        closes the node, one with one child closed is fixed to its other
        side, and the node's bound rises to the best lesser child bound.
        ``score`` (over ``candidates``) is updated in place: the children's
        gains for the solved candidates (``solved``), the screened gains
        (at least the pseudocost estimates) for the others, and ``-inf``
        for fixed candidates and those left outside the tightened box.
        Returns ``(lower, upper, bound, {var: {down: child}})``, or
        ``None`` when the node is closed.
        """
        lower, upper = lower.copy(), upper.copy()
        strong_bound, fixed = bound, False
        results = {}
        position = {var: k for k, var in enumerate(candidates.tolist())}
        for var, (down, up) in children.items():
            k = position.get(var)
            if k is None:  # screened, but reduced-cost fixing took it out
                continue
            self.branch_children[:, var] += 1
            self.branch_cutoffs[0, var] += down.closed
            self.branch_cutoffs[1, var] += up.closed
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
                gain_down, gain_up = max(gain_down, est_down[k]), max(
                    gain_up, est_up[k]
                )
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
        return lower, upper, strong_bound, results

    def root_sweep(self, x, candidates, lower, upper, warm, terms, bound, raw):
        """
        Strong branching at the root on the ``root_strong_candidates``
        fractional integers with the largest ``impact · f(1 − f)``
        (``branching_impact``; ``f`` is the fractional part), for
        ``root_strong_epochs``, in one vmapped call (``strong_branch``). It
        seeds their pseudocosts, so pseudocost branching sees which
        variables move the bound before it has branched on them; otherwise
        it starts most fractional and keeps to low-impact variables whose
        gains are all about zero. ``candidates`` is a mask of the branchable
        integers. Returns ``(lower, upper, bound, children)`` as
        ``apply_strong`` does (``children``: ``{var: {down: child}}``), or
        ``None`` when the root is closed.
        """
        cand = np.flatnonzero(candidates)
        if cand.size == 0 or self.out_of_time():
            return lower, upper, bound, {}
        if raw is None or not np.isfinite(raw):
            raw = bound
        f = x[cand] - np.floor(x[cand])
        prior = self.impact[cand] * f * (1.0 - f)
        chosen = cand[np.argsort(-prior, kind="stable")[: self.root_strong_candidates]]
        started = time.time()
        lps = self.strong_branch_lps
        epochs, self.strong_epochs = self.strong_epochs, self.root_strong_epochs
        try:
            children = self.strong_branch(
                chosen, x, lower, upper, warm, terms, bound, raw, trust=False
            )
        finally:
            self.strong_epochs = epochs
            self.root_strong_seconds += time.time() - started
            self.root_strong_lps += self.strong_branch_lps - lps
            self.strong_branch_lps = lps
        est_down, est_up = self.pseudocost_estimates(x, cand)
        score = np.maximum(est_down, 1e-6) * np.maximum(est_up, 1e-6)
        applied = self.apply_strong(
            children,
            set(chosen.tolist()),
            cand,
            score,
            est_down,
            est_up,
            x,
            lower,
            upper,
            bound,
            raw,
        )
        return applied

    def strong_kernels(self):
        """The device kernels for the current LP, built once per shape."""
        held = self._strong_kernels
        if held is not None and held[0] is not self.mip and held[1].set_mip(self.mip):
            self._strong_kernels = (self.mip, held[1])  # same shapes: kept
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
                self.node_dtype,
                branch_implied=self.branch_implied_cols is not None,
                nnz_capacity=self.kernel_nnz_capacity(),
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
        unreliable = (
            self.pc_count[:, candidates].min(axis=0) < self.reliability_threshold
        )
        if not unreliable.any():
            return None
        started = time.time()
        kernels = self.strong_kernels()
        size = kernels.screen_size
        est_down, est_up = self.pseudocost_estimates(x, candidates)
        if candidates.size > size:
            # The unreliable first, then by pseudocost score times impact
            # (``branching_impact``), so high-impact variables get screened.
            score = np.maximum(est_down, 1e-6) * np.maximum(est_up, 1e-6)
            keep = np.lexsort((-score * self.impact[candidates], ~unreliable))[:size]
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
        width = 2 * kernels.k
        with self.node_settings():
            warm_starts = jax.tree.map(
                lambda leaf: jnp.broadcast_to(leaf, (width,) + jnp.shape(leaf)),
                _cast_warm(warm[0], self.node_dtype),
            )
        steps = _step_sizes(warm[1])
        if steps is not None:
            # The parent's step sizes, for every child.
            dtype = _wide(self.node_dtype)
            steps = tuple(jnp.full(width, value, dtype) for value in steps)
        out, _ = self.node_batch(
            screened["lp_lower"],
            screened["lp_upper"],
            warm_starts,
            self.strong_epochs,
            steps,
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
                children[var][side] = _StrongChild(
                    child_bound, closed, solution, child_raw
                )
        return {var: tuple(sides) for var, sides in children.items()}, solved

    def strong_branch(
        self, variables, x, lower, upper, warm, terms, bound, raw=None, trust=True
    ):
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
        bounds, from ``raw`` (the node's; ``None``: ``bound``). With
        ``trust=False`` a child LP's bound that is not rigorous (it dropped
        reduced costs at infinite bounds) still scores the variable and
        records its pseudocost, but neither raises the child's bound nor
        closes it: on buildingenergy such bounds after 50–200 epochs passed
        the child's true LP bound by up to 5.5.
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
            child_terms = self.result_terms(result, lo, up)
            child_raw, rigorous = self.bound_over(child_terms, lo, up)
            child_terms, child_raw, rigorous = self.repair_terms(
                result, child_terms, child_raw, rigorous, lo, up
            )
            child_raw = max(child_raw, pre_raw)
            score_raw = child_raw  # for the score and pseudocost either way
            if not (rigorous or trust):
                child_raw = pre_raw
            child_bound = max(self.round_bound(child_raw), pre)
            if child_bound > pre:
                self.bound_rigorous &= rigorous
            distance = (
                x[var] - math.floor(x[var]) if down else math.ceil(x[var]) - x[var]
            )
            if distance > 0 and np.isfinite(score_raw):
                self.record_gain(var, down, (score_raw - raw) / distance)
            closed = child_bound >= cutoff
            if closed and rigorous:
                self.learn(child_terms, lo, up)
            children[var][down] = _StrongChild(
                child_bound, closed, None if closed else solution, score_raw
            )
        return {var: (sides[True], sides[False]) for var, sides in children.items()}


    def solve_children(self, boxes, warm):
        """The LPs of ``boxes`` for ``strong_epochs``, warm-started from
        ``warm``, in one batched call (padded to a power of two)."""
        if not boxes:
            return []
        # One batch shape for the whole search, so one compile.
        size = _next_power_of_two(max(2 * self.strong_candidates, len(boxes)))
        results = self.solve_batch(
            boxes,
            [warm[0]] * len(boxes),
            [warm[1]] * len(boxes),
            self.strong_epochs,
            size,
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
        """A box's changes against the root box, as ``expand`` takes them:
        ``int32`` indices, and bounds in ``float32`` where that is exact
        (integer bounds and infinities: half the bytes of the open nodes'
        changes on ns1830653)."""
        idx = np.flatnonzero((lower != self.root_lower) | (upper != self.root_upper))
        if lower.size <= np.iinfo(np.int32).max:
            idx = idx.astype(np.int32)
        return idx, _narrow(lower[idx]), _narrow(upper[idx])

    def _child_estimate(self, child, xv, var):
        """A child's estimated bound: its bound plus its pseudocost gain
        once both directions of ``var`` have observations."""
        down = child.branch[1]
        gain = 0.0
        if np.all(self.pc_count[:, var] > 0):
            side = 0 if down else 1
            f = xv - math.floor(xv)
            mean = self.pc_sum[side, var] / self.pc_count[side, var]
            gain = (f if down else 1.0 - f) * mean
        return child.bound + gain

    def _child_score(self, child, xv, var, bound):
        """``(1 + inferences) / (1 + cost / unit)`` for choosing the child
        to plunge into: ``cost`` the child's estimated bound gain over the
        parent's ``bound``, ``unit`` the mean pseudocost gain observed so far
        (1 before any), so a gain of a typical branching costs as much as one
        more fixing gains."""
        cost = max(self._child_estimate(child, xv, var) - bound, 0.0)
        known = self.pc_count > 0
        unit = (
            self.pc_sum[known].sum() / self.pc_count[known].sum()
            if known.any()
            else 1.0
        )
        unit = unit if np.isfinite(unit) and unit > 0 else 1.0
        return (1.0 + child.inferences) / (1.0 + cost / unit)

    def branch(
        self,
        node,
        lower,
        upper,
        x,
        var,
        bound,
        value,
        warm,
        terms,
        strong=None,
        precomputed=None,
    ):
        """Split on ``var``. Each child is bounded by the parent's duals
        (``terms``) over its own propagated box, and closed at once when
        that reaches the cutoff. ``strong`` (``{down: _StrongChild}``, from
        strong branching) gives the children a bound and a warm start of
        their own. ``precomputed`` is this node's row of ``_ChildKernels``
        (the children already propagated and bounded on the device)."""
        self.branched_nodes += 1
        xv = x[var]
        if not self.is_int[var]:
            self.implied_branchings += 1  # an implied integer (``branch_implied``)
        children = []
        for side, down in enumerate((True, False)):
            # The split alone, to count the bounds propagation tightens
            # beyond it (the child's inferences).
            split_lo, split_up = _child_box((lower, upper), var, xv, down)
            if precomputed is not None:
                if precomputed["crossed"][side]:
                    continue
                if precomputed["empty"][side]:
                    self.infeasible_nodes += 1
                    self.branch_children[side, var] += 1
                    self.branch_cutoffs[side, var] += 1
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
                self.branch_children[side, var] += 1
                if self.propagator is not None:
                    branched = (lo, up)
                    lo, up, infeasible = self.propagator(lo, up)
                    if infeasible:
                        self.infeasible_nodes += 1
                        self.branch_cutoffs[side, var] += 1
                        self.learn_nogood(*branched)
                        continue
            if precomputed is not None:
                self.branch_children[side, var] += 1
            lo, up, infeasible = self.conflict_propagate(lo, up)
            if infeasible:
                self.conflict_pruned += 1
                self.branch_cutoffs[side, var] += 1
                self.bump_conflict(lo, up)
                self.close(max(bound, self.cutoff_bound()))
                continue
            inferences = int(np.count_nonzero((lo != split_lo) | (up != split_up)))
            self.inference_sum[side, var] += inferences
            self.inference_count[side, var] += 1
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
                    child_warm = (
                        solution() if callable(solution) else solution,
                        warm[1],
                    )
            if child_bound >= self.cutoff():
                self.children_pruned += 1
                self.branch_cutoffs[side, var] += 1
                self.close(child_bound, rigorous)
                if rigorous:
                    self.learn(terms, lo, up)
                continue
            distance = xv - math.floor(xv) if down else math.ceil(xv) - xv
            child = _Node(
                child_bound,
                node.depth + 1,
                self.compress(lo, up),
                child_warm,
                (var, down, distance, value),
                inferences=inferences,
            )
            # The exact lane's strong child solved its LP already.
            child.exact_known = strong is not None and strong[down].exact
            # Its exact LP starts from its strong side's basis, else from
            # this node's (``basis_store``).
            child.exact_basis = (node.exact_sides or {}).get(
                (var, down), node.exact_basis
            )
            self.hold_basis(child.exact_basis)
            children.append(child)
        self.release_bases(node)  # the children hold what they start from
        # The child to plunge into (``children.pop()``) or, in a dive, to pop
        # next (pushed last) sorts last: the higher ``(1 + inferences) /
        # (1 + cost / unit)`` (``_child_score``), the bounds its propagation
        # fixed against its estimated bound gain, as HiGHS weighs inferences
        # against cost. A child that fixes many variables reaches an
        # integral point in fewer steps; on ns1830653 plunges that kept to
        # the cheaper child (down at x = 0.5) still had about 100 fractional
        # integers at depth 27. Equal scores go to the lower estimated bound,
        # then to the LP point's rounding direction.
        prefer_up = xv - math.floor(xv) >= 0.5
        # ``branch[1]`` is ``down``. (Before, it was compared with
        # ``prefer_up``, which put the child against the LP point last: on
        # ns1830653 plunges went up at x = 0.08.)
        children.sort(key=lambda child: (not child.branch[1]) == prefer_up)
        # ``plunge_child``: "score" as above, "estimate" (the lower estimated
        # bound, then rounding) or "rounding" (the LP point's direction only).
        if len(children) == 2 and self.plunge_child != "rounding":
            score_first = score_last = 0.0
            if self.plunge_child == "score":
                score_first, score_last = (
                    self._child_score(c, xv, var, bound) for c in children
                )
            tol = 1e-9 * (1.0 + abs(score_first) + abs(score_last))
            if score_first > score_last + tol:
                children.reverse()
            elif abs(score_first - score_last) <= tol:
                est_first, est_last = (
                    self._child_estimate(c, xv, var) for c in children
                )
                if est_first < est_last - 1e-9 * (1.0 + abs(est_first) + abs(est_last)):
                    children.reverse()
        if (
            children
            and (self.plunging() or self.exact_plunge_active)
            and self.plunge_left > 0
            and self.plunge_node is None
        ):
            if self.plunge_reaches(children[-1], bound):
                # Plunge: the next step continues into the preferred child.
                self.plunge_node = children.pop()
                self.plunge_node.cut_version = self.cut_version
                self.plunge_left -= 1
            else:
                self.plunges_stopped += 1
        for child in children:
            self.push(child)

    def plunge_reaches(self, child, bound):
        """Whether a plunge may continue into ``child`` (``plunge_gap``):
        its bound is within ``plunge_gap`` of the way from the best open
        bound (the heap's top, or ``bound``, the node being split) to the
        cutoff. A child past that is close to being pruned, and its
        plunge would leave the nodes that hold the global bound waiting."""
        if self.plunge_gap >= 1.0:
            return True  # children at or past the cutoff are pruned already
        cutoff = self.cutoff()
        best = min(self.open[0][2].bound if self.open else np.inf, bound)
        reserved = getattr(self, "exact_reserved", None)  # (``exact_band``)
        if reserved:
            best = min(best, reserved[0][2].bound)
        if not (np.isfinite(cutoff) and np.isfinite(best)):
            return True
        return child.bound <= best + self.plunge_gap * max(cutoff - best, 0.0)

    # -- the search ------------------------------------------------------------
    def evaluate(self, node, lower, upper, lp, result):
        """Bound, prune, accept or branch on a solved node."""
        self.active_bound = node.bound
        try:
            self._evaluate(node, lower, upper, lp, result)
        finally:
            self.active_bound = np.inf

    def _evaluate(self, node, lower, upper, lp, result):
        terms = self.result_terms(result, lower, upper)
        raw, rigorous = self.bound_over(terms, lower, upper)
        terms, raw, rigorous = self.repair_terms(
            result, terms, raw, rigorous, lower, upper
        )
        self.last_raw_bound = raw
        bound = max(self.round_bound(raw), node.bound)
        self.active_bound = bound
        spent = node.epochs + result["epochs"]
        # A node with every integer fixed has nothing to branch on: its LP
        # gets the full ``node_max_epochs`` to converge, even when the
        # others are truncated (``batch_epochs``), or it would close
        # unproven ("unresolved").
        budget = self.batch_node_epochs
        if budget < self.node_max_epochs and np.all(
            lower[self.is_int] == upper[self.is_int]
        ):
            budget = self.node_max_epochs
        if node.converge:
            # From the exact lane's band (``overflow_band``): solved to
            # convergence over as many batches as it takes.
            budget = max(budget, self.exact_overflow_epochs)
            self.overflow_converged += bool(result["converged"])
        if (
            not result["converged"]
            and bound < self.cutoff()
            and not self.out_of_time()
            and self.batch_size > 1
            and not self.exact_plunge_active
            and spent < budget
        ):
            # Continuous batching: a batched node LP runs at most
            # ``batch_epochs`` a call, so one slow member cannot hold the
            # batch. An unfinished one goes back, warm-started where it
            # stopped, for its next epochs in a later batch (until
            # ``node_max_epochs`` in all, when given; by default it
            # branches after its first call). Its Lagrangian bound holds
            # whatever the duals, and an infeasible LP's bound passes the
            # cutoff along its Farkas ray (``node_bound_stop``).
            self.batch_requeues += 1
            again = _Node(
                bound,
                node.depth,
                self.compress(lower, upper),
                (result["solution"], result["opt_state"]),
                node.branch,
                spent,
                cut_rounds=node.cut_rounds,
                branch_value=node.branch_value,
            )
            if node.converge:
                # Out of the lane's band for good, so it keeps its progress.
                again.converge = again.exact_tried = True
            # The same box: it keeps its hold on the basis it would start from.
            again.exact_basis, node.exact_basis = node.exact_basis, None
            self.push(again)
            return
        # An LP that did not converge keeps its Lagrangian bound (valid at
        # any duals) and branches on its point. An infeasible one has duals
        # that diverge along a Farkas ray, so its bound passes the cutoff and
        # closes it (``node_bound_stop``) without infeasibility detection.
        root = node.depth == 0 and node.branch is None
        if root and self.root_terms is not None and self.root_terms_rigorous:
            # Exact duals from a root crossover (``root_exact_terms``): a
            # rigorous bound, which reduced-cost fixing and proofs need.
            exact_raw, _ = self.bound_over(self.root_terms, lower, upper)
            if not rigorous or exact_raw >= raw:
                terms, raw, rigorous = self.root_terms, exact_raw, True
                bound = max(self.round_bound(raw), node.bound)
                self.active_bound = bound
        if root and not self.root_terms_rigorous:
            self.root_terms = terms
            self.root_terms_rigorous = rigorous and np.isfinite(raw)
        if self.global_redcost_fixing and not root:
            self.global_fix(terms)  # this LP's duals over the root box
        converged = result["converged"]
        x = np.asarray(result["solution"].primal, np.float64)
        if converged:
            value = float(self.mip.c @ x)
        else:
            # A truncated LP (``batch_epochs``): its Lagrangian bound stands
            # in for the LP value, for the pseudocosts and its children's
            # parent value, which would otherwise rarely learn.
            value = float(raw) if np.isfinite(raw) else None
        if node.branch is not None and value is not None:
            # The gain of the branching alone: the LP before this node's
            # re-solve with new tree cuts (``refresh_cuts``), whose gain is
            # the cuts', not the branched variable's.
            self.update_pseudocost(
                node, value if node.branch_value is None else node.branch_value
            )
        if bound >= self.cutoff():
            self.record_cutoff(node.branch)
            self.bump_conflict(lower, upper)
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
                new_terms = self.vertex_terms(vertex, lower, upper)
                new_raw, new_rigorous = self.bound_over(new_terms, lower, upper)
                # A rigorous bound wins over one that dropped reduced costs,
                # which can overshoot the LP's optimum.
                exact = new_rigorous and not rigorous and np.isfinite(new_raw)
                if new_raw > raw or exact:
                    terms, raw, rigorous = new_terms, new_raw, new_rigorous
                    bound = max(self.round_bound(raw), node.bound)
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
            if converged and feasible:
                self.close(bound, rigorous)
                return
            # The LP point is integral but did not settle the node (it failed
            # the check, or the LP did not converge). Closing the node on its
            # LP bound would hold the global bound there for good (on bab5,
            # 0.14% below the incumbent) and leave its other integer points
            # unsearched, so split a free integer instead, as SCIP does.
            var = self.integral_split_variable(x, lower, upper, terms)
            if var is None:  # every integer fixed: nothing left to split
                self.unresolved_nodes += 1
                self.unresolved_floor = min(self.unresolved_floor, bound)
                self.close(bound, rigorous)
                return
            self.integral_splits += 1
            self.bound_rigorous &= rigorous
            # A split point half a unit from the point's integral value, inside
            # the box; no pseudocost is learned from it (``value=None``).
            v = round(x[var])
            split = x.copy()
            split[var] = v + 0.5 if v < upper[var] else v - 0.5
            self.branch(node, lower, upper, split, var, bound, None, warm, terms)
            return
        self.bound_rigorous &= rigorous
        self.try_near_integral(x, xi, warm)
        # Every node's LP point is rounded in the background, beside the
        # node LPs (``parallel_rounding``), then jumped from if that fails
        # (``node_jump_seconds``): node points are often far from integral
        # (ns1830653: ~200 of 458 integers at 0.5), where rounding fails
        # and the jump still finds better points.
        rounding_queued = self.background_rounding(
            x, lower, upper, warm, jump_seconds=self.node_jump_seconds
        )
        if self.branch_implied_cols is not None:
            # Fractional implied integers join the candidates (not the
            # integrality test): on ns1830653 they carry the objective.
            k = self.branch_implied_cols
            fractional = fractional.copy()
            fractional[k] = np.abs(x[k] - np.round(x[k])) > self.int_tol
        # Strong branching runs on the device while the host does the
        # heuristics and reduced-cost fixing below.
        pending = None
        if self.strong_backend == "jax" and self.strong_branching_on(node):
            inside = fractional & (x > lower) & (x < upper)
            pending = self.launch_strong(
                x,
                inside if inside.any() else fractional,
                lower,
                upper,
                warm,
                terms,
                raw,
            )
        if self.heuristic_every and (self.nodes - 1) % self.heuristic_every == 0:
            if (
                self.try_incumbent(x, "H", warm)
                or not self.use_fix_and_propagate
                or rounding_queued
            ):
                pass  # (the background lane rounds this point)
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
        if not root and self.round_repair_due():
            self.round_repair(x, lower, upper, warm)
            if bound >= self.cutoff():
                self.close(bound, rigorous)
                return
        if (
            self.rins_seconds
            and self.incumbent is not None
            and (self.rins_pending or self.heuristic_due("rins", self.rins_every))
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
        if (
            self.dive_every
            and not self.out_of_time()
            and self.heuristic_due("dive", self.dive_every)
        ):
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
            self.push(
                _Node(
                    bound,
                    node.depth,
                    self.compress(lower, upper),
                    warm,
                    None,
                )
            )
            return
        else:
            candidates = fractional
        root_children = {}
        if (
            root
            and self.root_strong_candidates
            and result.get("exact_strong") is None
        ):
            swept = self.root_sweep(
                x, candidates, lower, upper, warm, terms, bound, raw
            )
            if swept is None:  # the sweep closed the root
                return
            lower, upper, bound, root_children = swept
            self.active_bound = bound
            inside = candidates & (x > lower) & (x < upper)
            if not inside.any():
                # The sweep fixed every candidate: solve the root again.
                self.push(
                    _Node(
                        bound,
                        node.depth,
                        self.compress(lower, upper),
                        warm,
                        None,
                    )
                )
                return
            candidates = inside
        choice = self.choose_branch(
            node,
            x,
            candidates,
            lower,
            upper,
            warm,
            terms,
            bound,
            pending,
            raw,
            exact_strong=result.get("exact_strong"),
            exact=result.get("stop_reason") == "exact",
        )
        if choice is None:  # strong branching closed or requeued the node
            return
        var, strong, lower, upper, bound = choice
        if strong is None and var in root_children:
            # The sweep's children: their bounds and warm starts.
            strong = root_children[var]
        if root and strong is not None and self.use_round_repair:
            # Round and repair from the branching variable's child LPs, up
            # first: on buildingenergy the sizing variable's up child rounds
            # up to within 0.005% of the optimum.
            for down in (False, True):
                solution = strong[down].solution
                if solution is None:
                    continue
                if callable(solution):
                    solution = solution()
                self.round_repair(
                    np.asarray(solution.primal, np.float64), lower, upper, warm
                )
            if bound >= self.cutoff():
                self.close(bound, rigorous)
                return
        self.active_bound = bound
        # Recent node points, restarts for the background jumps.
        self.recent_points.append(x)
        if self.dive_starts is not None:
            # A start for this batch's GPU dives (``gpu_dives``).
            self.dive_starts.append((bound, x, lower, upper, warm))
        if self.deferring:
            # Batched on the device: split with the batch's other members.
            self.deferred_branches.append(
                dict(
                    node=node,
                    lower=lower,
                    upper=upper,
                    x=x,
                    var=var,
                    bound=bound,
                    value=value,
                    warm=warm,
                    terms=terms,
                    strong=strong,
                )
            )
            return
        self.branch(node, lower, upper, x, var, bound, value, warm, terms, strong)

    def try_near_integral(self, x, xi, warm):
        """A node LP point whose integers ``xi`` are all within
        ``near_integral_tolerance`` of integers, though not within
        ``integrality_tolerance``, is rounded and tried (``"T"``). A loose
        PDHG point (float32 nodes at 1e-4, no node crossover) rarely passes
        the integrality test even at an integral optimum (a binary at
        0.9997). The node branches as usual; each rounding is tried once."""
        tol = self.near_integral_tol
        if not tol or not xi.size:
            return
        rounded = np.round(xi)
        if np.abs(xi - rounded).max() > tol:
            return
        key = rounded.tobytes()
        if key in self.near_integral_seen:
            return
        if len(self.near_integral_seen) >= 100_000:
            self.near_integral_seen.clear()
        self.near_integral_seen.add(key)
        before = None if self.incumbent is None else self.incumbent[0]
        self.near_integral_tries += 1
        self.try_incumbent(x, "T", warm)
        self.near_integral_finds += self.incumbent is not None and (
            before is None or self.incumbent[0] < before
        )

    def explored(self):
        """The share of the tree closed, as HiGHS's ``Expl.``: one less the
        weight (``2^-depth``) of the nodes still open, waiting in a batch or
        evaluated but not yet split. 0 before the tree starts."""
        if not self.tree_started:
            return 0.0
        nodes = list(self.open_nodes()) + list(self.batch_waiting)
        nodes += [entry["node"] for entry in self.deferred_branches]
        nodes += self.pipeline_nodes()
        weight = math.fsum(0.5 ** node.depth for node in nodes)
        return min(1.0, max(0.0, 1.0 - weight))

    def log_sources(self):
        """The progress table's source marks this solve can produce: those
        of the heuristics its options turn on (``"T"``, the node LP points,
        always)."""
        background = self.workers is not None
        on = {
            "B": self.use_bulk_rounding,
            "D": bool(self.dive_every)
            or bool(self.gpu_dive_every and self.batch_size > 1),
            "E": bool(self.rens_seconds),
            "F": bool(self.use_pump),
            "H": bool(self.heuristic_every),
            "J": bool(self.use_jump),
            "N": bool(self.rins_seconds),
            "P": bool(self.use_fix_and_propagate)
            or (background and self.use_parallel_rounding),
            "R": bool(self.use_repair and self.use_jump),
            "S": self.has_mip_start,
            "T": True,
            "U": self.use_round_repair,
            "C": bool(self.crossover_seconds and self.submip_lane is not None),
            "X": bool(self.proximity_seconds and self.submip_lane is not None),
            "Q": self.restart_presolve and self.max_restarts > 0,
            "W": bool(self.first_solution_seconds and self.submip_lane is not None),
            "L": self.level_lane is not None,
            "1": self.use_one_opt,
            "K": self.linking_blocks is not None,
            "Z": self.use_trivial,
        }
        return {mark for mark, enabled in on.items() if enabled}

    def log(self, source, bound, force=False):
        """A progress line, as HiGHS's (``bound=None``: the global bound).
        The first one prints the source legend and the header."""
        if not self.verbose:
            return
        with self.log_lock:
            if self.log_closed:
                return
            self._log(source, bound, force)

    def _log(self, source, bound, force):
        now = time.time()
        if not force and now - self.last_log < self.log_every:
            return
        self.last_log = now
        if not self.log_header_printed:
            self.log_header_printed = True
            print(_log_legend(self.log_sources()))
            print()
            print(_LOG_HEADER)
            print()
        if bound is None:
            bound = self.global_bound()
        best = np.inf if self.incumbent is None else self.incumbent[0]
        gap = self.gap(bound)
        if not np.isfinite(gap):
            gap_text = "inf"
        elif 100 * gap >= 9999:
            gap_text = "Large"
        else:
            gap_text = f"{100 * gap:.2f}%"
        leaves = max(0, self.nodes - self.branched_nodes)
        conflicts = 0 if self.conflicts is None else len(self.conflicts)
        print(
            f" {source:1.1} {_log_count(self.nodes):>7} {_log_count(len(self.open) + len(self.exact_reserved)):>7}"
            f"   {_log_count(leaves):>7} {100 * self.explored():6.2f}%"
            f"   {_log_value(self.model_value(bound)):<15}"
            f" {_log_value(self.model_value(best)):<15} {gap_text:>8}"
            f"   {len(self.cut_pool):6d} {len(self.cut_list):6d} {conflicts:6d}"
            f"   {_log_count(self.lp_epochs):>7} {now - self.start:7.1f}s"
        )

    def log_heartbeat(self, stop):
        """The heartbeat thread (``verbose``): a progress line whenever the
        search has printed none for ``log_every_seconds``, so the global
        bound and incumbent keep showing through long steps that log
        nothing themselves (a root cut round's separation, the root's
        heuristics and exact strong branching). On irp the first exact cut
        round separated for 70 s with no line after the first incumbent's.
        It starts once the table has (its header printed) and stops at the
        solving report. The bound is read while the search runs, so a line
        can catch a node between lists; an error skips the line."""
        interval = max(0.5, float(self.log_every))
        while not stop.wait(interval):
            if not self.log_header_printed or self.log_closed or self.log_paused:
                continue
            if time.time() - self.last_log < self.log_every:
                continue
            try:
                self.log(" ", None)
            except Exception:  # a list changed under the read: next time
                pass

    def report(self, status, bound, objective):
        """The end of the log, as HiGHS's solving report: the last table
        line, then the status, bounds, gap, time and counts."""
        with self.log_lock:
            if self.log_header_printed:
                self.log(" ", bound, force=True)
            self.log_closed = True  # no heartbeat line inside the report
        print()
        print("Solving report")
        rows = [("Status", _STATUS_TEXT.get(status, status))]
        rows.append(
            ("Primal bound", "inf" if objective is None else f"{objective:.10g}")
        )
        rows.append(("Dual bound", f"{self.model_value(bound):.10g}"))
        gap = self.gap(bound)
        gap_text = f"{100 * gap:.4g}%" if np.isfinite(gap) else "inf"
        rows.append(("Gap", f"{gap_text} (tolerance: {100 * self.rel_gap:.4g}%)"))
        elapsed = time.time() - self.start
        rows.append(("Timing", f"{elapsed:.2f} (total)"))
        rows.append(("", f"{self.heuristic_seconds:.2f} (heuristics)"))
        if self.strong_branch_seconds:
            rows.append(("", f"{self.strong_branch_seconds:.2f} (strong branching)"))
        tree = self.tree_elapsed()
        rows.append(
            (
                "Nodes",
                f"{self.nodes}"
                + (f" ({self.nodes / tree:.1f}/s in the tree)" if tree > 0 else ""),
            )
        )
        rows.append(("LP epochs", f"{self.lp_epochs} (total)"))
        if self.cut_list or self.root_bound_before_cuts is not None:
            families = [cut["family"] for cut in self.cut_list]
            names = (
                ("gomory", "Gomory"),
                ("cover", "cover"),
                ("mir", "MIR"),
                ("flowcover", "flow cover"),
                ("clique", "clique"),
                ("zerohalf", "zero-half"),
                ("impliedbound", "implied bound"),
            )
            counts = ", ".join(
                f"{families.count(key)} {name}"
                for key, name in names
                if families.count(key)
            )
            rows.append(
                ("Cuts", f"{len(families)} in the LP" + (f" ({counts})" if counts else ""))
            )
            if self.use_tree_cuts:
                rows.append(("", f"{self.tree_cuts_added} added in the tree"))
            rows.append(("", f"{len(self.cut_pool)} in the pool"))
        if self.conflicts is not None and any(self.conflicts.added.values()):
            added = self.conflicts.added
            rows.append(
                (
                    "Conflicts",
                    f"{added['objective']} objective, {added['nogood']} no-goods",
                )
            )
            rows.append(
                (
                    "",
                    f"{self.conflict_pruned} nodes pruned, "
                    f"{self.conflict_tightenings} bounds tightened",
                )
            )
        if self.strong_branch_nodes:
            rows.append(
                (
                    "Strong branching",
                    f"{self.strong_branch_nodes} nodes, {self.strong_branch_lps} LPs, "
                    f"{self.strong_fixings} fixings",
                )
            )
        if self.background_roundings:
            rows.append(
                (
                    "Roundings",
                    f"{self.background_roundings} in the background "
                    f"({self.background_jumps} with a jump)",
                )
            )
        if self.exact_plunge_stats["nodes"]:
            stats = self.exact_plunge_stats
            rows.append(
                (
                    "Exact plunges",
                    f"{stats['nodes']} nodes in {stats['plunges']} plunges, depth "
                    f"to {stats['max depth']}, {self.exact_plunge_seconds:.1f}s",
                )
            )
        if self.gpu_dive_stats["runs"]:
            stats = self.gpu_dive_stats
            rows.append(
                (
                    "GPU dives",
                    f"{stats['dives']} in {stats['runs']} runs, {stats['steps']} "
                    f"steps, {stats['leaves']} leaves, {stats['finds']} improved, "
                    f"{self.gpu_dive_seconds:.1f}s",
                )
            )
        if self.near_integral_tries:
            rows.append(
                (
                    "Near-integral",
                    f"{self.near_integral_tries} tried, "
                    f"{self.near_integral_finds} improved",
                )
            )
        for label, value in rows:
            print(f"  {label:<18}{value}")

    def result(self, status):
        # Background points that are in already (the rest are dropped).
        self.collect_heuristics(schedule=False)
        if status == "time_limit" and self.interrupt.is_set():
            status = "interrupted"  # it took the time-limit path
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
            self.report(status, bound, objective)
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
            "integral_splits": self.integral_splits,
            "near_integral": {
                "tries": self.near_integral_tries,
                "finds": self.near_integral_finds,
            },
            "background_roundings": self.background_roundings,
            "background_jumps": self.background_jumps,
            "exact_plunges": dict(
                self.exact_plunge_stats, seconds=self.exact_plunge_seconds
            ),
            "gpu_dives": dict(
                self.gpu_dive_stats,
                seconds=self.gpu_dive_seconds,
                crossover_seconds=self.gpu_dive_crossover_seconds,
            ),
            "inout_cuts": self.inout_cuts,
            "inout_rounds": self.inout_rounds,
            "tree_cut_refreshes": self.tree_cut_refreshes,
            "tree_cut_async_jobs": self.tree_cut_async_jobs,
            "exact_bounds_reused": self.exact_bounds_reused,
            "tree_pool_refreshes": self.tree_pool_refreshes,
            "tree_cut_requeued": self.tree_cut_requeued,
            "tree_cut_seconds": self.tree_cut_seconds,
            "tree_cut_pool_hits": self.tree_cut_pool_hits,
            "tree_cut_skipped": self.tree_cut_skipped,
            "tree_cut_families": dict(self.tree_cut_families),
            "cut_pool": len(self.cut_pool),
            "global_redcost_fixed": self.global_redcost_fixed,
            "cut_pool_rows": len(self.cut_archive),
            "lp_epochs": self.lp_epochs,
            "tree_seconds": self.tree_elapsed(),
            "seconds": time.time() - self.start,
            "history": self.history,
            "cuts": {
                family: sum(cut["family"] == family for cut in self.cut_list)
                for family in (
                    "gomory",
                    "cover",
                    "mir",
                    "flowcover",
                    "clique",
                    "zerohalf",
                    "impliedbound",
                )
            },
            "root_bound_before_cuts": (
                None
                if self.root_bound_before_cuts is None
                else self.model_value(self.root_bound_before_cuts)
            ),
            "heuristic_seconds": self.heuristic_seconds,
            "batches": self.batches,
            "objective_step": self.objective_step,
            "reduced_cost_tightenings": self.reduced_cost_tightenings,
            "dual_repairs": dict(self.dual_repair_stats),
            "children_pruned": self.children_pruned,
            "lp_repairs": self.lp_repairs,
            "lp_repairs_skipped": self.lp_repairs_skipped,
            "coefficients_tightened": self.coefficients_tightened,
            "tree_cuts_added": self.tree_cuts_added,
            "tree_cuts_dropped": self.tree_cuts_dropped,
            "symmetry": self.symmetry_stats,
            "restarts": len(self.restart_log),
            "restart_log": list(self.restart_log),
            "rins_calls": self.rins_calls,
            "rens_calls": self.rens_calls,
            "rens_finds": self.rens_finds,
            "first_solution_calls": self.first_solution_calls,
            "first_solution_finds": self.first_solution_finds,
            "rins_finds": self.rins_finds,
            "rins_fixing_rate": self.rins_fixing_rate,
            "batch_timing": {
                "wait_seconds": self.batch_wait_seconds,
                "evaluate_seconds": self.batch_evaluate_seconds,
                "bound_seconds": self.bound_seconds,
                "cut_refresh_seconds": self.cut_refresh_seconds,
                "exact_collect_seconds": self.exact_collect_seconds,
                "exact_service_seconds": self.exact_service_seconds,
                "branch_seconds": self.branch_seconds,
                "heuristic_seconds": self.heuristic_seconds,
            },
            "crossover_calls": self.crossover_calls,
            "crossover_finds": self.crossover_finds,
            "proximity_calls": self.proximity_calls,
            "proximity_finds": self.proximity_finds,
            "proximity_empty": self.proximity_empty,
            "submip_lane_seconds": self.submip_lane_seconds,
            "objective_levels": self.level_stats(),
            "exact_cut_rounds": self.exact_cut_stats,
            "conflicts": (
                dict(self.conflicts.added)
                if self.conflicts is not None
                else {"objective": 0, "farkas": 0, "nogood": 0}
            ),
            "conflict_pruned": self.conflict_pruned,
            "conflict_tightenings": self.conflict_tightenings,
            "bound_stops": self.bound_stops,
            "submip_calls": self.submip_calls,
            "submip_seconds": self.submip_seconds,
            "parallel_heuristics": {
                "jobs": self.worker_jobs,
                "seconds": self.worker_seconds,
                "improvements": self.worker_finds,
                "jump_slices": self.jump_slices,
                "jump_restarts": self.jump_restarts,
                "jump_restart_finds": self.jump_restart_finds,
            },
            "strong_branch_nodes": self.strong_branch_nodes,
            "strong_branch_lps": self.strong_branch_lps,
            "strong_branch_seconds": self.strong_branch_seconds,
            "strong_fixings": self.strong_fixings,
            "root_strong_lps": self.root_strong_lps,
            "root_strong_seconds": self.root_strong_seconds,
            "root_exact_strong": dict(self.root_exact_stats),
            "cutoff_probing": (
                dict(self.probe_stats) if self.settings["cutoff_probing"] else None
            ),
            "restart_presolve": [dict(entry) for entry in self.restart_presolve_log],
            # Block decomposition's progress (``None``: it did not run).
            "block_decomposition": (
                None
                if self.linking_blocks is None
                else dict(
                    blocks=len(self.linking_blocks.blocks),
                    linking_rows=int(self.linking_blocks.linking.size),
                    **self.block_holder.stats,
                )
            ),
            "root_crossover": self.root_crossover_used,
            "root_lp": dict(self.root_lp_stats),
            "round_repair_calls": self.round_repair_calls,
            "round_repair_finds": self.round_repair_finds,
            "round_repair_seconds": self.round_repair_seconds,
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
            "exact_nodes": {
                "solved": self.exact_solved,
                "infeasible": self.exact_infeasible,
                "failed": self.exact_failed,
                "seconds": self.exact_seconds,
                "bound_gain": self.exact_gain,
                "strong_lps": self.exact_strong_lps,
                "strong_seconds": self.exact_strong_seconds,
                "children_reused": self.exact_reused,
                "pivots": self.exact_pivots,
                "strong_iterations": self.exact_strong_iterations,
                "warm_bases": self.exact_warm_bases,
                "overflow_nodes": self.overflow_nodes,
                "overflow_converged": self.overflow_converged,
                "band": self.exact_band,
                "band_min": self.exact_band_range[0],
                "band_max": self.exact_band_range[1],
                "band_scale": self.band_scale,
                "band_scale_min": self.band_scale_range[0],
                "band_scale_max": self.band_scale_range[1],
                "band_tunes": self.band_tunes,
                "band_rates": self.band_rates,
                "band_cap_nodes": self.band_cap_nodes,
                "band_capped_batches": self.band_capped_batches,
            },
            "mip_start_accepted": self.mip_start_accepted,
            "one_opt_finds": self.one_opt_finds,
            "plunges_stopped": self.plunges_stopped,
            "implied_branchings": self.implied_branchings,
        }

    def node_crossover_due(self):
        """Whether this node's LP point is crossed over (``node_crossover``,
        every ``node_crossover_every`` nodes, time permitting)."""
        if self.exact_plunge_active:
            return not self.out_of_time()  # exact plunges cross every node over
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

    def highs_root_lp(self, lp, lower, upper):
        """
        The first root LP solved by HiGHS (``root_lp="highs"``), for timing
        against PDHG: HiGHS's LP solver with its presolve, over the searched
        rows and the box, then PDHG on ``lp`` warm from HiGHS's vertex and
        duals, so the search goes on from a PDHG result as usual (from an
        optimal point it stops at its first convergence check). The vertex
        is kept as the root's crossover (``root_vertex_state``), so the
        root's exact duals need no crossover. Returns the PDHG result,
        ``"infeasible"`` when HiGHS proves the LP infeasible, or ``None``
        when HiGHS does not solve it (PDHG then solves it cold).
        """
        import highspy

        mip = self.mip
        n = mip.num_variables
        highs = _highs_rows(mip)
        highs.changeColsBounds(
            n,
            np.arange(n, dtype=np.int32),
            np.asarray(lower, np.float64),
            np.asarray(upper, np.float64),
        )
        for option, value in (
            ("presolve", "on"),
            ("solver", "choose"),
            ("parallel", "choose"),
            ("time_limit", float(self.seconds_left() or np.inf)),
        ):
            highs.setOptionValue(option, value)
        started = time.time()
        highs.run()
        self.root_lp_stats["highs_seconds"] += time.time() - started
        status = highs.getModelStatus()
        if status == highspy.HighsModelStatus.kInfeasible:
            return "infeasible"
        if status != highspy.HighsModelStatus.kOptimal:
            return None
        x = np.asarray(highs.getSolution().col_value, np.float64)
        y_eq, y_ineq = _jaddle_duals(highs, mip)
        self.root_vertex_state = (
            mip, SaddleState(primal=x, dual_ineq=y_ineq, dual_eq=y_eq)
        )
        # The LP's dual blocks (``jl.solve`` pads an empty one to a row).
        dual_eq = np.zeros(lp.A_eq.shape[0])
        dual_ineq = np.zeros(lp.A_ineq.shape[0])
        dual_eq[: y_eq.size] = y_eq
        dual_ineq[: y_ineq.size] = y_ineq
        warm = (
            SaddleState(
                primal=jnp.asarray(x),
                dual_ineq=jnp.asarray(dual_ineq),
                dual_eq=jnp.asarray(dual_eq),
            ),
            None,
        )
        return self.solve_lp(lp, warm, self.root_max_epochs)

    def root_vertex(self, lower, upper, solution):
        """Cross the root LP's point over to an optimal vertex with HiGHS
        (``jaddle_linear.crossover``). Returns ``(x, basis)``, or
        ``(None, None)`` when HiGHS does not prove the vertex optimal within
        its time limit (5% of ``max_seconds``, or 60 s without one)."""
        limit = (
            60.0
            if self.deadline is None
            else max(1.0, 0.05 * (self.deadline - self.start))
        )
        crossed = jl.crossover(
            self.node_lp(lower, upper), solution, highs_options={"time_limit": limit}
        )
        if not crossed["converged"] or crossed["basis"] is None:
            return None, None
        # Its exact duals, for ``root_exact_terms`` (while the rows stay).
        self.root_vertex_state = (self.mip, crossed["solution"])
        return np.asarray(crossed["solution"].primal, np.float64), crossed["basis"]

    def current_root_vertex(self):
        """The primal point of the last root crossover if it was for the
        current rows (``self.mip``), else ``None``."""
        state = self.root_vertex_state
        if state is None or state[0] is not self.mip:
            return None
        return np.asarray(state[1].primal, np.float64)

    def root_exact_terms(self, lower, upper, result):
        """
        The root's Lagrangian terms for reduced-cost fixing of the root box
        (``tighten_root``). PDHG duals solved to a loose tolerance leave tiny
        wrong-sign reduced costs on variables with an infinite bound; the
        bound then drops them and is not rigorous, and reduced-cost fixing
        and dual proofs need a rigorous one. With ``root_crossover``
        (``"auto"``: only then), the root LP is crossed over to an exact
        vertex (``root_vertex``, reusing RENS's crossover when the rows are
        the same), whose duals give a rigorous bound. On buildingenergy
        (145,000 variables with no upper bound) the crossover takes 2 s and
        gives 33,246.215, HiGHS's LP optimum, where the PDHG bound (which
        dropped 8,209 reduced costs) gave 33,247.9, above it; with an
        incumbent, the exact duals also fix the two dearest sizing variables
        to zero.
        """
        terms = self.dual_terms(result["solution"])
        bound, rigorous = self.bound_over(terms, lower, upper)
        self.root_terms = terms
        self.root_terms_rigorous = rigorous and np.isfinite(bound)
        if self.root_terms_rigorous or not self.root_crossover or self.out_of_time():
            return
        state = self.root_vertex_state
        if state is None or state[0] is not self.mip:
            self.root_vertex(lower, upper, result["solution"])
            state = self.root_vertex_state
        if state is None or state[0] is not self.mip:
            return  # HiGHS did not prove a vertex optimal in time
        exact = self.vertex_terms(state[1], lower, upper)
        exact_bound, exact_rigorous = self.bound_over(exact, lower, upper)
        if exact_rigorous and np.isfinite(exact_bound):
            self.root_terms, self.root_terms_rigorous = exact, True
            self.root_crossover_used = True
            if self.verbose:
                print(
                    f"Root crossover: exact duals, bound "
                    f"{self.model_value(exact_bound):.6e}, "
                    f"{time.time() - self.start:.1f}s"
                )

    def cut_loop(self, lower, upper, result):
        """Rounds of root cuts: separate at the LP point, append the cuts,
        re-solve warm by PDHG. With Gomory cuts, each round crosses over to
        a vertex, whose basis gives them (and from which the other families
        separate too).

        A cut stays in the LP until it has been idle (slack, with a zero
        dual, at the round's re-solve) for ``cut_age`` rounds in a row,
        then it leaves the LP for the global cut pool (``pool_cuts``). On
        degenerate LPs a cut slack at one optimum is often binding again at
        the next, and dropped at once the rounds cycle between them
        (ns1830653 sat at 11,622 for good, as HiGHS's simplex rounds did
        before they kept slack cuts 10 rounds; bab5's root went from
        −107,807 to −107,379 with them kept). ``max_cut_rows`` caps the
        rows (``cap_cut_rows``).

        From the second round on, each round's point is rounded
        (``round_heuristics``), and the background improving jump is queued
        again from each new incumbent.

        Stops when a round finds no cut, or, once three rounds have run, the
        bound gains less than ``cut_min_progress`` over the last three:
        relative to the gap to the incumbent when there is one, else to the
        bound. Also stops when a re-solve's bound reaches the cutoff
        (``root_resolve``). Returns the last root LP result.

        With ``cut_round_lp="highs"`` the rounds run on HiGHS's dual simplex
        instead (``exact_cut_loop``), unless it cannot solve the root LP."""
        if self.cut_round_lp == "highs":
            out = self.exact_cut_loop(lower, upper, result)
            if out is not None:
                return out
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
            # Pick up background jump points and queue the next slice from
            # the incumbent, so the improving jump runs beside the rounds.
            self.collect_heuristics()
            if self.root_closed():
                return result  # the caller closes the root
            round_started = time.time()
            solution = result["solution"]
            x = np.asarray(solution.primal, np.float64)
            candidates = []
            if self.gomory:
                # Cross over to a vertex: its basis gives the Gomory cuts, and
                # the other families separate better from a vertex too.
                vertex, basis = self.root_vertex(lower, upper, solution)
                if vertex is not None:
                    x = vertex
                    self.root_vertex_x = vertex
                    # The rows the basis is for: the crossover's
                    # (``self.mip``, padding rows included).
                    candidates += gomory_cuts(
                        self.mip.A,
                        self.mip.b,
                        self.mip.n_eq,
                        np.where(self.cut_int, lower, base.lower),
                        np.where(self.cut_int, upper, base.upper),
                        self.cut_int,
                        basis,
                        x,
                        max_cuts=self.gomory_candidates or 2 * self.max_cuts_per_round,
                        # Dense cuts are fine here: the ones left idle leave
                        # the LP after ``cut_age`` rounds.
                        max_nnz=None,
                    )
            if round_number > 1:
                # Round the round's point (with ``parallel_heuristics`` on a
                # background thread, beside this round's separation and
                # re-solve). The first round's point is the root LP's, which
                # the root heuristics have just rounded.
                self.round_heuristics(
                    x,
                    lower,
                    upper,
                    (solution, result["opt_state"]),
                    0.1 * (time.time() - round_started),
                )
            for separator in separators:
                candidates += separator(x)
            candidates += self.zero_half(x, lower, upper, 2 * self.max_cuts_per_round)
            inout = self.inout_candidates(
                separators, x, lower, upper, 2 * self.max_cuts_per_round
            )
            candidates += inout or []
            found = self.select_cuts(candidates, x, n, self.max_cuts_per_round)
            self.inout_result(found, inout)
            self.archive_cuts(found)
            # Every violated candidate goes into the pool, as HiGHS's, for
            # the tree to take back when a node point violates it; the ones
            # in the LP go in last (the newest, so the last to be evicted).
            self.pool_cuts(violated_cuts(candidates, x))
            self.pool_cuts(found)
            if not found:
                break
            dual_ineq = np.asarray(solution.dual_ineq)
            binding = [cut["age"] < self.cut_age for cut in self.cut_list]
            if self.max_cut_rows is not None:
                found = self.cap_cut_rows(binding, found)
                if not found:
                    break  # the binding cuts fill the room
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
                capacity = (
                    2 * max(need_rows, capacity[0]),
                    2 * max(need_nnz, capacity[1]),
                )
            A_cuts, b_cuts = _cut_block(self.cut_list, n, *capacity)
            self.set_mip(base.with_rows(A_cuts, b_cuts))
            # Warm start: the old duals, zero for the new rows and padding.
            duals = np.zeros(base_m + A_cuts.shape[0], dtype=dual_ineq.dtype)
            duals[:base_m] = dual_ineq[:base_m]
            duals[base_m : base_m + len(kept)] = [dual for _, dual in kept]
            warm = solution._replace(dual_ineq=jnp.asarray(duals))
            result = self.root_resolve(
                lower, upper, (warm, result["opt_state"]), self.round_epochs()
            )
            if result["stop_reason"] in ("interrupted", "time_limit", "bound_target"):
                return result
            self.age_cuts(result["solution"])
            new_bound, _ = self.lp_bound(result["solution"], lower, upper)
            bound = max(bound, new_bound)
            history.append(bound)
            # The root's bound (the global one until it is evaluated), and
            # the progress line at its usual pace.
            if np.isfinite(bound):
                self.active_bound = max(self.active_bound, self.round_bound(bound))
            self.log(" ", None)
            if self.root_closed():
                return result  # the caller closes the root
            # Progress comes in bursts (and a short re-solve's bound can dip),
            # so judge it over three rounds, once three have run. With an
            # incumbent it is judged against the gap left: on rmine13 the
            # root's gap to the best known value is 0.27% of the objective,
            # so a test against the objective stopped after one round.
            if len(history) < 4:
                continue
            gain = history[-1] - history[-4]
            scale = max(1.0, abs(bound + self.mip.offset))
            if self.incumbent is not None:
                scale = min(scale, max(1e-9 * scale, self.incumbent[0] - bound))
            if not np.isfinite(gain) or gain <= self.cut_min_progress * scale:
                break
        result = self.compact_cuts(lower, upper, result)
        if (
            not result["converged"]
            and result["stop_reason"] != "bound_target"
            and not self.out_of_time()
        ):
            # The rounds stopped early; the tree starts from a converged root.
            result = self.root_resolve(
                lower,
                upper,
                (result["solution"], result["opt_state"]),
                self.root_max_epochs,
            )
        return result

    def exact_cut_loop(self, lower, upper, result):
        """
        ``cut_loop``'s rounds on exact LPs (``cut_round_lp="highs"``): one
        HiGHS dual simplex holds the searched rows and the cuts, added and
        dropped as rows, so each re-solve starts from the last basis. Each
        round separates at the exact vertex (Gomory cuts from its basis)
        with ``cut_loop``'s separators, selection, pool, ageing
        (``cut_age``, at most ``_EXACT_CUT_AGE``), row cap (``max_cut_rows``)
        and stopping rule, for up to ``cut_rounds`` rounds and
        ``_EXACT_CUT_SHARE`` of ``max_seconds``. The kept cuts then go into the PDHG root
        LP, re-solved warm from HiGHS's solution, so the tree starts as
        after ``cut_loop``.

        Each round cuts off only its own point, so the bound builds over
        many rounds, and PDHG re-solves of ``cut_round_epochs`` give
        inexact points to separate from. On beasleyC3, Jaddle's separators
        at exact points reach 718 in 40 rounds (HiGHS's root: 733); ten
        PDHG rounds took 46 s to reach 661. Returns the root LP result, or
        ``None`` when HiGHS cannot solve the root LP (``cut_loop`` then
        runs on PDHG).
        """
        import highspy

        started = time.time()
        base = self.base
        n = base.num_variables
        base_rows = base.A.shape[0]
        base_m = base_rows - base.n_eq
        lower = np.asarray(lower, np.float64)
        upper = np.asarray(upper, np.float64)
        highs = _highs_rows(base)
        highs.changeColsBounds(n, np.arange(n, dtype=np.int32), lower, upper)
        optimal = highspy.HighsModelStatus.kOptimal
        seconds = {"lp": 0.0, "gomory": 0.0, "separation": 0.0}

        def solve():
            """The LP's vertex, duals (Jaddle's convention) and value, or
            ``None`` (not optimal: infeasible, out of time or failed)."""
            limit = 60.0 if self.deadline is None else self.deadline - time.time()
            if limit <= 0:
                return None
            highs.setOptionValue("time_limit", highs.getRunTime() + limit)
            solve_started = time.time()
            highs.run()
            seconds["lp"] += time.time() - solve_started
            if highs.getModelStatus() != optimal:
                return None
            x = np.asarray(highs.getSolution().col_value, np.float64)
            y_eq, y_ineq = _jaddle_duals(highs, base)
            return x, y_eq, y_ineq, float(highs.getInfo().objective_function_value)

        def rows_and_bound(solved):
            """The LP's rows (the searched ones, then the cuts) and the
            Lagrangian bound of the solve's duals over the box: valid at any
            duals, so it holds however exact HiGHS's are."""
            cuts_A, cuts_b = _cut_rows(self.cut_list, n)
            A_rows = sp.vstack([sp.csr_matrix(base.A), cuts_A]).tocsr()
            b_rows = np.concatenate([np.asarray(base.b, np.float64), cuts_b])
            _, y_eq, y_ineq, _ = solved
            terms = lagrangian_terms(
                np.asarray(base.c, np.float64),
                A_rows,
                b_rows,
                base.n_eq,
                np.concatenate([y_eq, y_ineq]),
            )
            return A_rows, b_rows, box_bound(*terms, lower, upper)[0]

        def raise_root_bound(value):
            """The root's bound (the global one until it is evaluated) up to
            ``value``, and the progress line (at its usual pace)."""
            if np.isfinite(value):
                self.active_bound = max(self.active_bound, self.round_bound(value))
            self.log(" ", None)

        solved = solve()
        if solved is None:
            return None
        A_rows, b_rows, value = rows_and_bound(solved)
        raise_root_bound(value)
        separators = [self.separator(base, lower, upper)]
        if self.original is not base:
            separators.append(self.separator(self.original, lower, upper))
        bound, _ = self.lp_bound(result["solution"], lower, upper)
        self.root_bound_before_cuts = bound
        history = [solved[3]]
        warm = (result["solution"], result["opt_state"])  # for the heuristics
        self.cut_list = []
        rounds = 0
        # Exact duals show a cut idle for good sooner than PDHG's.
        age_limit = min(self.cut_age, _EXACT_CUT_AGE)
        stop_at = None
        if self.deadline is not None:
            stop_at = started + _EXACT_CUT_SHARE * (self.deadline - self.start)
        for round_number in range(1, self.cut_rounds + 1):
            if self.out_of_time() or (stop_at is not None and time.time() > stop_at):
                break
            self.collect_heuristics()
            if self.root_closed():
                break
            round_started = time.time()
            x = solved[0]
            candidates = []
            if self.gomory:
                gomory_started = time.time()
                candidates += gomory_cuts(
                    A_rows,
                    b_rows,
                    base.n_eq,
                    lower,
                    upper,
                    self.cut_int,
                    highs.getBasis(),
                    x,
                    max_cuts=self.gomory_candidates or 2 * self.max_cuts_per_round,
                    max_nnz=None,
                )
                seconds["gomory"] += time.time() - gomory_started
            if round_number > 1:
                self.round_heuristics(
                    x, lower, upper, warm, 0.1 * (time.time() - round_started)
                )
            separation_started = time.time()
            for separator in separators:
                candidates += separator(x)
            candidates += self.zero_half(x, lower, upper, 2 * self.max_cuts_per_round)
            inout = self.inout_candidates(
                separators, x, lower, upper, 2 * self.max_cuts_per_round
            )
            candidates += inout or []
            found = self.select_cuts(candidates, x, n, self.max_cuts_per_round)
            self.inout_result(found, inout)
            self.archive_cuts(found)
            self.pool_cuts(violated_cuts(candidates, x))
            self.pool_cuts(found)
            seconds["separation"] += time.time() - separation_started
            if not found:
                break
            binding = [cut["age"] < age_limit for cut in self.cut_list]
            if self.max_cut_rows is not None:
                found = self.cap_cut_rows(binding, found)
                if not found:
                    break  # the binding cuts fill the room
            drop = [base_rows + i for i, keep in enumerate(binding) if not keep]
            if drop:
                highs.deleteRows(len(drop), np.array(drop, np.int32))
            new = [
                dict(family=family, idx=idx, coef=coef, rhs=rhs, age=0)
                for family, idx, coef, rhs in found
            ]
            self.cut_list = [
                cut for cut, keep in zip(self.cut_list, binding) if keep
            ] + new
            add_A, add_b = _cut_rows(new, n)
            highs.addRows(
                len(new),
                np.full(len(new), -np.inf),
                add_b,
                add_A.nnz,
                add_A.indptr[:-1].astype(np.int32),
                add_A.indices.astype(np.int32),
                add_A.data.astype(np.float64),
            )
            rounds += 1
            solved = solve()
            if solved is None:
                # Infeasible with the cuts, or out of time: the PDHG re-solve
                # below bounds it (an infeasible LP's bound passes the cutoff).
                break
            self.age_cuts_at(solved[0], solved[2][base_m:])
            A_rows, b_rows, value = rows_and_bound(solved)
            raise_root_bound(value)
            if self.root_closed():
                break
            history.append(max(history[-1], solved[3]))
            if len(history) < 4:
                continue
            gain = history[-1] - history[-4]
            scale = max(1.0, abs(history[-1] + self.mip.offset))
            if self.incumbent is not None:
                scale = min(scale, max(1e-9 * scale, self.incumbent[0] - history[-1]))
            if not np.isfinite(gain) or gain <= self.cut_min_progress * scale:
                break
        self.exact_cut_stats = {
            "rounds": rounds,
            "cuts": len(self.cut_list),
            "lp_bound": None if solved is None else self.model_value(solved[3]),
            "seconds": time.time() - started,
            **{f"{key}_seconds": value for key, value in seconds.items()},
        }
        if not self.cut_list or self.root_closed():
            return result  # with the root closed, the caller closes it
        # Into the PDHG root LP, warm from HiGHS's vertex and duals (the
        # last solve's; cuts added after a failed one start at zero).
        A_cuts, b_cuts = _cut_block(self.cut_list, n)
        self.set_mip(base.with_rows(A_cuts, b_cuts))
        template = result["solution"]
        dual_ineq = np.zeros(base_m + A_cuts.shape[0])
        dual_eq = np.zeros(np.shape(template.dual_eq)[0])
        x = np.asarray(template.primal, np.float64)
        if solved is not None:
            x, y_eq, y_ineq, _ = solved
            dual_ineq[: y_ineq.size] = y_ineq
            dual_eq[: y_eq.size] = y_eq
            self.root_vertex_x = x
        warm_state = template._replace(
            primal=x, dual_eq=dual_eq, dual_ineq=jnp.asarray(dual_ineq)
        )
        result = self.root_resolve(
            lower, upper, (warm_state, result["opt_state"]), self.root_max_epochs
        )
        if result["stop_reason"] in ("interrupted", "time_limit", "bound_target"):
            return result
        result = self.compact_cuts(lower, upper, result)
        if (
            not result["converged"]
            and result["stop_reason"] != "bound_target"
            and not self.out_of_time()
        ):
            result = self.root_resolve(
                lower,
                upper,
                (result["solution"], result["opt_state"]),
                self.root_max_epochs,
            )
        return result

    def round_heuristics(self, x, lower, upper, warm, seconds):
        """Cheap roundings of a cut round's point ``x`` (its crossover
        vertex when there is one): fix-and-propagate and, while there is no
        incumbent, bulk rounding for up to ``seconds``. HiGHS rounds its LP
        point during root separation too. (1-opt runs on each incumbent in
        ``try_incumbent``.)

        With ``parallel_heuristics``, fix-and-propagate goes to the
        background ``"round"`` lane instead (``background_rounding``,
        with ``parallel_rounding``; skipped while the lane is still busy
        with an earlier point), and bulk rounding, a device
        call, is left to ``root_heuristics`` after the rounds: the rounds do
        not wait for either."""
        if self.workers is not None:
            self.background_rounding(x, lower, upper, warm)
            return
        started = time.time()
        if self.use_fix_and_propagate:
            candidate = fix_and_propagate(
                x, lower, upper, self.is_int, self.heuristic_propagator()
            )
            if candidate is not None:
                self.try_incumbent(candidate, "P", warm)
        if self.use_bulk_rounding and self.incumbent is None:
            deadline = time.time() + max(seconds, 1.0)
            for point in self.bulk_round([x] * self.bulk_starts, lower, upper, deadline):
                if point is not None:
                    self.try_incumbent(point, "B", warm)
        self.heuristic_seconds += time.time() - started

    def background_rounding(self, x, lower, upper, warm, jump_seconds=0.0):
        """With ``parallel_heuristics`` and ``parallel_rounding``: queue
        fix-and-propagate of ``x`` over ``lower``/``upper`` (with the
        objective row at the cutoff, ``heuristic_propagator``) on the
        background ``"round"`` lane, unless it is still busy with an
        earlier point. If the rounding fails and ``jump_seconds`` is
        positive (with the numba jump), the same job runs an improving
        feasibility jump from ``x`` in that box, below the cutoff, for that
        long (``_node_heuristic_job``). Its point comes back through
        ``collect_heuristics`` (``"P"`` or ``"J"``). Returns whether it was
        queued."""
        if (
            self.workers is None
            or not self.use_parallel_rounding
            or not self.workers.idle("round")
            or self.out_of_time()
        ):
            return False
        jump = None
        if jump_seconds > 0 and self.use_jump and self.jump_backend == "numba":
            cutoff = self.cutoff(box=False)
            jump = dict(
                mip=self.base,
                seconds=float(jump_seconds),
                deadline=self.deadline,
                tol=self.feas_tol,
                max_iterations=np.iinfo(np.int64).max,
                seed=self.background_roundings,
                cutoff=float(cutoff) if np.isfinite(cutoff) else None,
                improve_step=self.objective_step,
            )
            self.background_jumps += 1
        self.workers.submit(
            "round",
            "fixprop",
            _node_heuristic_job,
            np.array(x, np.float64),
            np.array(lower, np.float64),
            np.array(upper, np.float64),
            self.is_int,
            self.heuristic_propagator(),
            jump=jump,
            info=warm,
        )
        self.background_roundings += 1
        return True

    def jump_cutoff(self):
        """The objective row (``c·x <= cutoff``) for the jumps that run
        while there is no incumbent: the cutoff when it comes from a
        solution outside this search (``cutoff_from_start``: a ``mip_start``
        under presolve, or a presolved restart, ``restart_presolve``), else
        ``None``. Without it they stop at their first feasible point, which
        is rarely better than that solution and ``try_incumbent`` drops; so
        the search had no incumbent, and the improving jumps (which take
        the cutoff) never started."""
        if not self.cutoff_from_start:
            return None
        cutoff = self.cutoff(box=False)
        return float(cutoff) if np.isfinite(cutoff) else None

    def launch_root_jumps(self, lower, upper):
        """The feasibility jump before the root LP (``root_jump_seconds``),
        as HiGHS runs its jump before its first LP: with background workers,
        the numba jump and no incumbent, each jump lane starts one from the
        bounds nearest zero (its own seed), for up to ``root_jump_seconds``,
        while the device solves the root LP. They need no LP point. They
        stop at the search's first incumbent (``_early_jump_job``). On
        neos-5106984-jizera (299k columns after presolve) the jump needs
        about 30-45 s from zero, where the root's own jumps had about 5 s
        each, after the root LP. Returns whether any started."""
        if (
            self.workers is None
            or not self.use_jump
            or self.jump_backend != "numba"
            or self.incumbent is not None
            or self.root_jump_seconds <= 0
            or self.out_of_time()
        ):
            return False
        lower = np.array(lower, np.float64)
        upper = np.array(upper, np.float64)
        start = np.clip(np.zeros(lower.shape), lower, upper)
        deadline = time.time() + self.root_jump_seconds
        if self.deadline is not None:
            deadline = min(deadline, self.deadline)
        self.early_jump = threading.Event()
        started = False
        for i, lane in enumerate(self.workers.jump_lanes):
            if not self.workers.idle(lane):
                continue
            self.workers.submit(
                lane,
                "jump",
                _early_jump_job,
                lambda: self.incumbent is not None,
                self.early_jump,
                self.base,
                start.copy(),
                lower,
                upper,
                tol=self.feas_tol,
                max_iterations=self.jump_iterations,
                deadline=deadline,
                seed=101 + i,
                cutoff=self.jump_cutoff(),
                backend="numba",
            )
            started = True
        return started

    def try_trivial(self, lower, upper):
        """The trivial heuristic (``trivial_heuristic``), before the root
        LP: ``trivial_points`` in the root box, each a matvec; a feasible
        one goes to ``try_incumbent``, marked ``"Z"``. Points that need
        their continuous variables re-solved are skipped (that costs an
        LP)."""
        if not self.use_trivial:
            return
        for point in trivial_points(self.original, lower, upper):
            if self.original.is_feasible(point, self.feas_tol):
                self.try_incumbent(point, "Z", None)

    def block_decomposition(self, lower, upper):
        """Block decomposition (``block_decomposition``) before the root
        LP, in the foreground, with no incumbent yet: when
        ``linking_blocks`` finds blocks (a few hundredths of a second on
        stp3d), ``negotiate_blocks`` searches for a point for up to
        ``block_decomposition_seconds``, with the CPU to itself (the early
        jumps start after it, and only if it found nothing). A point it
        finds is the incumbent, marked ``"K"``. Runs once a solve."""
        if (
            not self.use_block_decomposition
            or self.block_decomposition_seconds <= 0
            or self.block_holder is not None  # once a solve
            or self.incumbent is not None
            or self.out_of_time()
        ):
            return
        self.block_holder = _HighsHolder()
        blocks = linking_blocks(self.original)
        if blocks is None:
            return
        self.linking_blocks = blocks
        if self.verbose:
            print(
                f"Block decomposition: {len(blocks.blocks)} blocks joined by "
                f"{blocks.linking.size} linking rows"
            )
        deadline = time.time() + self.block_decomposition_seconds
        if self.deadline is not None:
            deadline = min(deadline, self.deadline)
        x, found, _ = negotiate_blocks(
            self.original,
            blocks,
            np.array(lower, np.float64),
            np.array(upper, np.float64),
            deadline=deadline,
            tol=self.feas_tol,
            holder=self.block_holder,
        )
        if found:
            self.try_incumbent(x, "K", None)

    def early_jumps_running(self):
        """Whether jumps started before the root LP are still at work."""
        return (
            self.early_jump is not None
            and not self.early_jump.is_set()
            and self.workers is not None
            and any(not self.workers.idle(lane) for lane in self.workers.jump_lanes)
        )

    def background_root_heuristics(self, lower, upper, result):
        """With ``parallel_heuristics``: the root heuristics on the first
        root LP, before the cut rounds, as background jobs, so the rounds
        start at once. Fix-and-propagate from the LP point goes to the
        ``"round"`` lane (``background_rounding``) and, with the numba jump and no incumbent, the
        jumps from the LP point and from the bounds nearest zero to the
        ``"jump"`` lane (each skipped if an incumbent has come by the time
        it starts; together within the heuristic budget). Bulk rounding and
        the pump run on the device, so they wait for ``root_heuristics``
        after the rounds, if there is still no incumbent then. Their points
        are checked as the rounds collect them (``collect_heuristics``); the
        first incumbent starts the background improving jump. Returns False
        (nothing queued) without background workers."""
        if self.workers is None:
            return False
        x = np.asarray(result["solution"].primal, np.float64)
        lower = np.array(lower, np.float64)
        upper = np.array(upper, np.float64)
        warm = (result["solution"], result["opt_state"])
        # The box and warm start of the background improving jump.
        self.jump_box = (lower.copy(), upper.copy(), warm)
        self.background_rounding(x, lower, upper, warm)
        if (
            self.use_jump
            and self.jump_backend == "numba"
            and self.incumbent is None
            and self.workers.idle("jump")
        ):
            if self.heuristic_budget is not None:
                budget = float(self.heuristic_budget)
            elif self.deadline is not None:
                budget = 0.2 * (self.deadline - self.start)
            else:
                budget = 60.0
            now = time.time()
            starts = ((x, 0), (np.clip(np.zeros_like(x), lower, upper), 1))
            for i, (start, seed) in enumerate(starts):
                deadline = now + budget * (i + 1) / len(starts)
                if self.deadline is not None:
                    deadline = min(deadline, self.deadline)
                self.workers.submit(
                    "jump",
                    "jump",
                    _jump_job,
                    lambda: self.incumbent is not None,
                    self.base,
                    start.copy(),
                    lower,
                    upper,
                    tol=self.feas_tol,
                    max_iterations=self.jump_iterations,
                    deadline=deadline,
                    seed=seed,
                    cutoff=self.jump_cutoff(),
                    backend="numba",
                    info=warm,
                )
        return True

    def root_resolve(self, lower, upper, warm, max_epochs):
        """The root LP over ``lower``/``upper`` re-solved warm (a cut round,
        or after one), at the search's precision. It stops once its
        Lagrangian bound reaches the cutoff (``node_bound_target``): an LP
        made infeasible by valid cuts, whose MIP is then infeasible, has
        duals that diverge along a Farkas ray, so its bound passes the
        cutoff (``box_cutoff`` before an incumbent) and the root closes on
        it. No infeasibility detection: its chunked host checks cost more
        than the re-solve on large LPs."""
        target = self.node_bound_target()
        return self.solve_lp(
            self.node_lp(lower, upper),
            warm,
            max_epochs,
            **({} if target is None else {"bound_target": target}),
        )

    def age_cuts(self, solution):
        """Count, for each cut in the LP (``cut_list``), the re-solves in a
        row it has been idle at: slack by more than ``1e-6·(1 + |rhs|)``
        with a dual of at most ``1e-9`` of the largest. Any other re-solve
        resets it to 0."""
        base_m = self.base.A.shape[0] - self.base.n_eq
        self.age_cuts_at(
            np.asarray(solution.primal, np.float64),
            np.asarray(solution.dual_ineq, np.float64)[base_m:],
        )

    def age_cuts_at(self, x, y):
        """``age_cuts`` at the point ``x`` with ``y`` the cut rows' duals
        (in ``cut_list`` order)."""
        y_scale = 1e-9 * (1.0 + np.abs(y).max(initial=0.0))
        for i, cut in enumerate(self.cut_list):
            slack = cut["rhs"] - cut["coef"] @ x[cut["idx"]]
            idle = slack > 1e-6 * (1.0 + abs(cut["rhs"])) and y[i] <= y_scale
            cut["age"] = cut["age"] + 1 if idle else 0

    def cap_cut_rows(self, binding, found):
        """Keep a round within ``max_cut_rows``: clear ``binding`` (in place)
        for the kept cuts idle longest (``age_cuts``), then take fewer of
        the new cuts ``found`` (most efficacious first). Only idle cuts go.
        Returns the new cuts that fit."""
        kept = [i for i, keep in enumerate(binding) if keep]
        excess = len(kept) + len(found) - self.max_cut_rows
        if excess <= 0:
            return found
        idle = [i for i in kept if self.cut_list[i]["age"] > 0]
        idle.sort(key=lambda i: -self.cut_list[i]["age"])  # stable: oldest first
        for i in idle[:excess]:
            binding[i] = False
        excess -= min(excess, len(idle))
        return found[: max(0, len(found) - excess)]

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
        return self.root_resolve(
            lower, upper, (warm, result["opt_state"]), self.root_max_epochs
        )

    def node_separators(self):
        """The root's separators (``separator``) over the root box, for node
        cut rounds: for the searched rows and, when coefficient tightening
        changed them, the rows as given, as in ``cut_loop``. Rebuilt when
        the rows or the root box change (a restart, reduced-cost fixing);
        the box only shrinks, so their cuts hold at every node. Call under
        ``_separator_lock``."""
        key = (self.base, self.original, self.root_lower, self.root_upper)
        held = self._node_separators
        if held is None or any(a is not b for a, b in zip(held[0], key)):
            separators = [self.separator(self.base, self.root_lower, self.root_upper)]
            if self.original is not self.base:
                separators.append(
                    self.separator(self.original, self.root_lower, self.root_upper)
                )
            held = self._node_separators = (key, separators)
        return held[1]

    def node_separated(self, x):
        """Cuts at a node's LP point ``x`` from the root's separators
        over the root box, valid at every node."""
        with self._separator_lock:
            return [cut for separator in self.node_separators() for cut in separator(x)]

    def select_cuts(self, candidates, x, n, max_cuts):
        """``select_cuts`` with this search's ``cut_score_weights``: the
        objective, the cut integrality and the incumbent."""
        incumbent = self.incumbent
        return select_cuts(
            candidates,
            x,
            n,
            max_cuts=max_cuts,
            weights=self.cut_score_weights,
            objective=self.base.c,
            is_int=self.cut_int,
            incumbent=None if incumbent is None else incumbent[1],
        )

    def zero_half(self, x, lower, upper, max_cuts, mip=None):
        """Zero-half cuts at ``x`` (``zero_half_cuts``) from the LP's rows as
        searched, the cuts in it included,
        over the box ``lower``/``upper`` (the root's, so the cuts hold
        everywhere). The cut rows count too: on bab5, combining them takes
        the root bound (exact LPs, 20 rounds) from −108,341 to −107,410."""
        if not self.use_zero_half:
            return []
        mip = self.mip if mip is None else mip  # a worker passes the rows it took
        return zero_half_cuts(
            mip.A, mip.b, mip.n_eq, x, lower, upper, self.cut_int, max_cuts=max_cuts
        )

    def inout_candidates(self, separators, x, lower, upper, max_cuts):
        """
        In-out separation (``inout_separation``): the cuts ``separators``
        and ``zero_half`` find at ``x̃ = λ·x + (1 − λ)·x̄`` for the incumbent
        ``x̄`` (clipped to the box), or ``None`` when it does not run (off,
        no incumbent, or given up on for this one).

        A valid cut ``a·x <= b`` that ``x̃`` violates is violated at ``x`` by
        more when ``x̄`` satisfies it: ``a·x̃ = λ·a·x + (1 − λ)·a·x̄``. The
        separators pick rows, divisors and covers from the point they are
        given, and from a point nearer the feasible region they find deeper
        cuts (Ben-Ameur and Neto 2007; Fischetti and Salvagnin 2010). The
        cuts still go through ``select_cuts`` at ``x``, so soundness never
        rests on ``x̄`` (it may lie outside a root box reduced-cost fixing
        has shrunk). λ starts at ``inout_separation`` for each incumbent;
        ``inout_result`` moves it towards ``x``.
        """
        if self.inout_separation is None or self.incumbent is None:
            return None
        value, incumbent = self.incumbent
        x = np.asarray(x, np.float64)
        if incumbent.shape != x.shape:
            return None
        if self.inout_at != value:
            self.inout_at = value
            self.inout_lambda = self.inout_separation
        if self.inout_lambda is None:  # given up on for this incumbent
            return None
        lam = self.inout_lambda
        point = np.clip(lam * x + (1.0 - lam) * incumbent, lower, upper)
        self.inout_rounds += 1
        candidates = [cut for separator in separators for cut in separator(point)]
        candidates += self.zero_half(point, lower, upper, max_cuts)
        return candidates

    def inout_result(self, found, inout):
        """Count the cuts ``select_cuts`` kept (``found``) from the in-out
        point's candidates (``inout``). When it kept none, λ moves halfway to
        1 (towards the LP point); past ``_INOUT_MAX_LAMBDA`` the incumbent
        is not used again until it changes."""
        if inout is None:
            return
        ids = {id(cut) for cut in inout}
        kept = sum(id(cut) in ids for cut in found)
        self.inout_cuts += kept
        if kept == 0 and self.inout_lambda is not None:
            lam = 0.5 * (1.0 + self.inout_lambda)
            self.inout_lambda = None if lam > _INOUT_MAX_LAMBDA else lam

    def separator(self, mip, lower, upper):
        """A ``CutSeparator`` with this search's aggregation settings. The
        one for the searched rows (``self.base``) also separates clique
        cuts (``clique_cuts``) from the conflict graph of the searched and
        the given rows over this box, built on the first call; the box only
        tightens after that (the root box), so its conflicts stay valid."""
        graph = None
        if self.use_clique_cuts and mip is self.base:
            if self.conflict_graph is None:
                mips = [self.base] + (
                    [self.original] if self.original is not self.base else []
                )
                self.conflict_graph = ConflictGraph(
                    mips, lower, upper, integrality=self.cut_int
                )
            if (
                self.use_probing_cliques
                and self.probe_cache is not None
                and self._graph_probe_cache is not self.probe_cache
            ):
                self.probing_stats["conflicts"] += self.conflict_graph.add_implications(
                    self.probe_cache, lower, upper, self.cut_int
                )
                self._graph_probe_cache = self.probe_cache
            graph = (
                self.conflict_graph
                if len(self.conflict_graph) or self.conflict_graph.extra
                else None
            )
        implied = None
        if (
            self.use_implied_bound_cuts
            and mip is self.base
            and self.probe_cache is not None
        ):
            # Built once per probing (each restart probes again).
            if (
                self._implied_bounds is None
                or self._implied_bounds[0] is not self.probe_cache
            ):
                built = ImpliedBounds(self.probe_cache, lower, upper, self.cut_int)
                self._implied_bounds = (self.probe_cache, built)
            implied = self._implied_bounds[1] if len(self._implied_bounds[1]) else None
        return CutSeparator(
            mip,
            lower,
            upper,
            aggregate_integers=self.aggregate_integers,
            max_aggregation=self.max_aggregation,
            aggregation_width=self.aggregation_width,
            cmir_per_row=self.cmir_per_row,
            flow_covers=self.use_flow_covers,
            mir_cuts=self.use_mir_cuts,
            backend=self.separation_backend,
            device=self.separation_device,
            conflict_graph=graph,
            integrality=self.cut_int,
            implied_bounds=implied,
            implied_variable_bounds=self.use_implied_variable_bounds,
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
        per_round = self.max_tree_cuts_per_round
        min_rows = max(_TREE_CUT_MIN_ROWS, 2 * per_round)
        # The cuts the pool holds (``tree_cut_rows``; at least the root's
        # and one round more) and their nonzeros (``tree_cut_nnz``; with
        # only the rows given, that many cuts of the root cuts' mean size,
        # or 16 nonzeros, whichever is larger).
        if self.tree_cut_rows is None:
            rows = max(min_rows, 2 * root_rows)
        else:
            rows = max(self.tree_cut_rows, root_rows + per_round)
        if self.tree_cut_nnz is not None:
            nnz = max(self.tree_cut_nnz, root_nnz + per_round * min(n, 16))
        elif self.tree_cut_rows is None:
            nnz = max(256, 2 * root_nnz, min_rows * min(n, 16))
        else:
            per_cut = max(min(n, 16), root_nnz / max(root_rows, 1))
            nnz = max(256, math.ceil(rows * per_cut))
        self.tree_nnz_cap = _next_power_of_two(nnz)
        # Enough rows for the cuts plus the zero rows that pad the nonzeros.
        self.tree_padding_rows = -(-self.tree_nnz_cap // n)
        self.tree_rows_cap = _next_power_of_two(rows + self.tree_padding_rows)
        dual_ineq = np.asarray(result["solution"].dual_ineq)
        self.set_mip(base.with_rows(*self.tree_block()))
        self.slot_version = [self.cut_version] * self.tree_rows_cap
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

    def fit_warm(self, node):
        """``node``'s warm start for the current cut block: the duals of
        the slots a tree cut refresh has given another cut since the warm
        start was made (``cut_version``) are zeroed. Kept cuts keep theirs."""
        warm = node.warm
        if (
            warm is None
            or warm[0] is None
            or node.cut_version is None
            or node.cut_version >= self.cut_version
        ):
            return warm
        return (self.stale_free(warm[0], node.cut_version), warm[1])

    def stale_free(self, solution, version):
        """``solution`` with the duals of the cut block's slots given a new
        cut since ``version`` (``cut_version``) zeroed; kept cuts keep
        theirs."""
        if version >= self.cut_version:
            return solution
        stale = [i for i, v in enumerate(self.slot_version) if v > version]
        if not stale:
            return solution
        base_m = self.base.A.shape[0] - self.base.n_eq
        duals = np.array(solution.dual_ineq)
        if duals.shape[0] < base_m + len(self.slot_version):
            return solution
        duals[[base_m + i for i in stale]] = 0.0
        return solution._replace(dual_ineq=duals)

    def tree_cuts_due(self):
        """The tree cut refresh due now: ``"full"`` (separation too) every
        ``tree_cut_every`` nodes, else ``"pool"`` (the cut pool only) every
        ``tree_pool_every``, else ``None``. Neither while refreshes have
        taken more than ``tree_cut_share`` of the tree's time (after its
        first seconds)."""
        if not (self.use_tree_cuts and self.slot_version):
            return None
        full = (
            self.tree_cut_every
            and self.nodes - self.last_refresh_nodes >= self.tree_cut_every
        )
        pool = (
            not full
            and self.tree_pool_every
            and len(self.cut_pool)
            and self.nodes - self.last_pool_nodes >= self.tree_pool_every
        )
        if not (full or pool):
            return None
        tree = self.tree_elapsed()
        if (
            tree > _TREE_CUT_GRACE_SECONDS
            and self.tree_cut_seconds > self.tree_cut_share * tree
        ):
            if full:
                self.tree_cut_skipped += 1
                self.last_refresh_nodes = self.nodes
            self.last_pool_nodes = self.nodes
            return None
        if self.out_of_time():
            return None
        return "full" if full else "pool"

    def after_node_solves(self, members, boxes, results):
        """
        Between solving node LPs (a batch, or one node) and evaluating them:
        a tree cut refresh when due (``refresh_cuts``; ``tree_cuts_due``: a
        full one, or one from the cut pool alone), from all of their
        points. The results then have the duals of replaced slots zeroed (a
        zero multiplier is a valid one, so their bounds hold for the new
        rows). Members whose own point a new cut cuts off, with re-solves
        left (``tree_cut_rounds``), go back in the queue to be solved again
        against the new cuts before they branch; their pseudocosts keep the
        LP value before the cuts. Returns the results and the set of
        re-queued positions.
        """
        requeued = set()
        kind = self.tree_cuts_due()
        if kind is None:
            return results, requeued
        started = time.time()
        self.last_pool_nodes = self.nodes
        if kind == "full":
            self.last_refresh_nodes = self.nodes
        solved = [
            i
            for i, result in enumerate(results)
            if result.get("solution") is not None
            and np.asarray(result["solution"].dual_ineq).shape[0] == self.m_ineq
        ]
        if not solved:
            return results, requeued
        order = sorted(solved, key=lambda i: members[i].bound)
        refreshing = time.perf_counter()  # not ``started``: that one is time.time()
        replaced, added = self.refresh_cuts(
            [results[i]["solution"] for i in solved],
            [results[i]["solution"] for i in order],
            separate=kind == "full",
        )
        self.cut_refresh_seconds += time.perf_counter() - refreshing
        if replaced:
            base_m = self.base.A.shape[0] - self.base.n_eq
            rows = [base_m + i for i in replaced]
            fixed = []
            for result in results:
                solution = result.get("solution")
                if solution is not None:
                    duals = np.array(solution.dual_ineq)
                    if duals.shape[0] == self.m_ineq:
                        duals[rows] = 0.0
                        result = dict(
                            result,
                            solution=solution._replace(dual_ineq=jnp.asarray(duals)),
                        )
                fixed.append(result)
            results = fixed
        if added and self.tree_cut_rounds:
            for i in solved:
                node, result = members[i], results[i]
                if node.cut_rounds >= self.tree_cut_rounds or self.out_of_time():
                    continue
                x = np.asarray(result["solution"].primal, np.float64)
                if not any(
                    coef @ x[idx] - rhs > 1e-6 * (1.0 + abs(rhs))
                    for idx, coef, rhs in added
                ):
                    continue
                value = node.branch_value
                if value is None and result.get("converged"):
                    value = float(self.mip.c @ x)
                lower, upper = boxes[i]
                requeue = _Node(
                    node.bound,
                    node.depth,
                    self.compress(lower, upper),
                    (result["solution"], result.get("opt_state")),
                    node.branch,
                    node.epochs,
                    node.inferences,
                    cut_rounds=node.cut_rounds + 1,
                    branch_value=value,
                )
                self.push(requeue)
                requeued.add(i)
            self.tree_cut_requeued += len(requeued)
        self.tree_cut_seconds += time.time() - started
        return results, requeued

    def separate_at(self, x, mip, lower, upper):
        """Tree cut candidates at the node point ``x``: the root's
        separators (``node_separated``) and zero-half cuts from ``mip``'s
        rows over the root box ``lower``/``upper``, so each holds at every
        node; cuts denser than ``tree_cut_max_nnz`` are left out. Returns
        ``(selected, violated)``: the ``max_tree_cuts_per_round`` that
        ``select_cuts`` picks at ``x``, and every violated one (for the
        pool). Safe on a worker thread: it reads only its arguments and
        the separators, under ``_separator_lock``."""
        per_point = self.max_tree_cuts_per_round
        candidates = self.node_separated(x)
        candidates += self.zero_half(x, lower, upper, 2 * per_point, mip=mip)
        candidates = [c for c in candidates if c[1].size <= self.tree_cut_max_nnz]
        selected = self.select_cuts(candidates, x, mip.num_variables, per_point)
        return selected, violated_cuts(candidates, x)

    def submit_cut_job(self, points):
        """Separate tree cuts at ``points`` on the worker thread
        (``tree_cut_async``), with the rows and root box as they are now;
        ``collect_cut_job`` takes the result at a later refresh. With a job
        still running, these points are skipped.

        On a GPU the main thread spends 20–40% of the tree blocked on the
        device, which releases the GIL: the worker separates in that time.
        The cuts reach the LP a refresh later."""
        if self._cut_job is not None:
            return
        if self._cut_worker is None:
            from concurrent.futures import ThreadPoolExecutor

            self._cut_worker = ThreadPoolExecutor(1, thread_name_prefix="jaddle-cuts")
        mip = self.mip
        lower, upper = self.root_lower.copy(), self.root_upper.copy()

        def job():
            found = []
            for x in points:
                if self.out_of_time():
                    break
                found.append((x, *self.separate_at(x, mip, lower, upper)))
            return found

        self._cut_job = (self._cut_worker.submit(job), self.base)
        self.tree_cut_async_jobs += 1

    def collect_cut_job(self):
        """The finished worker separation (``submit_cut_job``) as a list of
        ``(x, selected, violated)``, or ``[]`` while it runs, without one,
        after it failed, or when the searched MIP changed since (a restart:
        its cuts may not fit the new rows)."""
        job = self._cut_job
        if job is None or not job[0].done():
            return []
        self._cut_job = None
        future, base = job
        try:
            found = future.result()
        except Exception:
            return []
        return found if base is self.base else []

    def refresh_cuts(self, solutions, best_first, separate=True):
        """
        A tree cut refresh of the LP's cut block (``tree_cuts``) from the
        node LP solutions just solved (``solutions``; ``best_first``: them
        by node bound, lowest first):

        1. Each slot's cut ages by one if it is idle (slack, with a zero
           dual) at every one of them, else back to 0.
        2. Candidates: the pooled cuts (``cut_pool``) any of the points
           violates, not in the block, and cuts separated at the points of
           the ``tree_cut_points`` lowest-bound nodes (the root's separators
           over the root box, ``node_separated``, and zero-half), each cut
           holding at every node. Cuts denser than ``tree_cut_max_nnz`` are
           left out.
        3. The most efficacious (at their best point) go in, at most
           ``max_tree_cuts_per_round``, none a near-duplicate of a cut in
           the block or of another new one.
        4. Each takes a free slot, else the slot idle longest (at least
           ``_TREE_CUT_IDLE_REFRESHES`` refreshes); with no slot it is
           dropped (``tree_cuts_dropped``).
        5. The block is rebuilt at its fixed capacity and the LP's rows
           swapped (``set_mip(rows_only=True)``): the batched node solver
           takes the rows as data, so nothing recompiles.

        Without ``separate`` (a pool refresh, every ``tree_pool_every``
        nodes), only pooled cuts are candidates and no cut ages: as HiGHS
        checks its cut pool at every node, between its rarer separations.

        Returns the slots given new cuts that held others before, and the
        cuts added (as ``(idx, coef, rhs)``).
        """
        if separate:
            self.tree_cut_refreshes += 1
        else:
            self.tree_pool_refreshes += 1
        base = self.base
        n = base.num_variables
        base_m = base.A.shape[0] - base.n_eq
        points = [np.asarray(sol.primal, np.float64) for sol in solutions]
        k = len(self.cut_list)
        if k:
            idle_all = np.ones(k, bool)
            rhs = np.array([cut["rhs"] for cut in self.cut_list])
            R = sp.csr_matrix(
                (
                    np.concatenate([cut["coef"] for cut in self.cut_list]),
                    np.concatenate([cut["idx"] for cut in self.cut_list]),
                    np.concatenate(
                        [[0], np.cumsum([cut["idx"].size for cut in self.cut_list])]
                    ),
                ),
                shape=(k, n),
            )
            for sol, x in zip(solutions, points if separate else ()):
                y = np.asarray(sol.dual_ineq, np.float64)[base_m : base_m + k]
                y_scale = 1e-9 * (1.0 + np.abs(y).max(initial=0.0))
                slack = rhs - R @ x
                idle_all &= (slack > 1e-6 * (1.0 + np.abs(rhs))) & (y <= y_scale)
            if separate:
                for i, cut in enumerate(self.cut_list):
                    cut["age"] = cut["age"] + 1 if idle_all[i] else 0

        held = {cut.get("key") or _cut_key(_as_cut(cut)) for cut in self.cut_list}
        best = {}  # key -> (efficacy, cut)

        def consider(cut, x):
            _, idx, coef, rhs = cut
            if idx.size == 0 or idx.size > self.tree_cut_max_nnz:
                return
            efficacy = (coef @ x[idx] - rhs) / max(np.linalg.norm(coef), 1e-12)
            if efficacy < _CUT_POOL_MIN_EFFICACY:
                return
            key = _cut_key(cut)
            if key in held:
                return
            if key not in best or efficacy > best[key][0]:
                best[key] = (efficacy, cut)

        pooled = set()
        with self._shared_lock:
            for x in points:
                for cut in self.cut_pool.violated(x, held):
                    consider(cut, x)
                    pooled.add(_cut_key(cut))
        separated = []  # every violated candidate, for the pool
        # The worker's cuts from an earlier refresh's points, judged at
        # those points as a separation here would judge them.
        for x, selected, violated in self.collect_cut_job():
            separated += violated
            for cut in selected:
                consider(cut, x)
        points = [
            np.asarray(sol.primal, np.float64)
            for sol in best_first[: self.tree_cut_points if separate else 0]
        ]
        if self.tree_cut_async and points:
            self.submit_cut_job(points)
            points = []
        for x in points:
            if self.out_of_time():
                break
            selected, violated = self.separate_at(x, self.mip, self.root_lower, self.root_upper)
            separated += violated
            for cut in selected:
                consider(cut, x)
        # Into the pool, as at the root: a later node may violate them.
        self.pool_cuts(separated)
        ranked = sorted(best.items(), key=lambda item: -item[1][0])
        found = []
        if ranked and self.max_tree_cuts_per_round > 0:
            # None a near-duplicate of a cut in the block or of another new
            # one: ``select_cuts``'s greedy pass (``_spread_out``) over all
            # the candidates at once. A product per candidate per cut taken
            # was a quarter of the tree's time on ns1830653 (thousands of
            # pooled candidates a refresh).
            held_rows = None
            if self.cut_list:
                held_rows = sp.csr_matrix(R.multiply(1.0 / np.maximum(
                    np.sqrt(np.asarray(R.multiply(R).sum(axis=1)).ravel()), 1e-12
                )[:, None]))
            lengths = np.array([cut[1].size for _, (_, cut) in ranked])
            rows = np.repeat(np.arange(len(ranked)), lengths)
            cols = np.concatenate(
                [np.asarray(cut[1], np.int64) for _, (_, cut) in ranked]
            )
            vals = np.concatenate(
                [np.asarray(cut[2], np.float64) for _, (_, cut) in ranked]
            )
            norm = np.maximum(
                np.sqrt(np.bincount(rows, vals * vals, minlength=len(ranked))), 1e-12
            )
            directions = sp.csr_matrix(
                (vals / norm[rows], (rows, cols)), shape=(len(ranked), n)
            )
            taken = _spread_out(
                directions,
                np.arange(len(ranked)),
                self.max_tree_cuts_per_round,
                held=held_rows,
            )
            found = [(ranked[i][0], ranked[i][1][1]) for i in taken]
        if not found:
            return [], []
        self.pool_cuts([cut for _, cut in found])
        self.archive_cuts([cut for _, cut in found])
        self.tree_cut_pool_hits += sum(key in pooled for key, _ in found)

        max_rows = self.tree_rows_cap - self.tree_padding_rows
        nnz = sum(cut["idx"].size for cut in self.cut_list)
        replaced, added, families, dropped = [], [], [], 0
        version = self.cut_version + 1
        for key, (family, idx, coef, rhs) in found:
            new = dict(family=family, idx=idx, coef=coef, rhs=rhs, age=0, key=key)
            if len(self.cut_list) < max_rows and nnz + idx.size <= self.tree_nnz_cap:
                self.slot_version[len(self.cut_list)] = version
                self.cut_list.append(new)
                nnz += idx.size
            else:
                slots = [
                    i
                    for i, cut in enumerate(self.cut_list)
                    if cut["age"] >= _TREE_CUT_IDLE_REFRESHES
                    and i not in replaced
                    and nnz - cut["idx"].size + idx.size <= self.tree_nnz_cap
                ]
                if not slots:
                    dropped += 1  # no room (``tree_cut_rows``, ``tree_cut_nnz``)
                    continue
                slot = max(slots, key=lambda i: self.cut_list[i]["age"])
                nnz += idx.size - self.cut_list[slot]["idx"].size
                self.cut_list[slot] = new
                self.slot_version[slot] = version
                replaced.append(slot)
            added.append((idx, coef, rhs))
            families.append(family)
        self.tree_cuts_dropped += dropped
        if not added:
            return [], []
        self.cut_version = version
        self.tree_cuts_added += len(added)
        self.tree_cut_families.update(families)
        self.set_mip(base.with_rows(*self.tree_block()), rows_only=True)
        return replaced, added

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
            stalled = len(distances) > 5 and distances[-1] > 0.9 * np.mean(
                distances[-6:-1]
            )
            repeated = previous is not None and np.array_equal(
                rounded[ints], previous[ints]
            )
            key = rounded[ints].tobytes()
            if (repeated or stalled or key in seen) and near_miss is not None:
                # Cycling: continue from the jump's near miss.
                rounded[ints] = np.clip(
                    np.round(near_miss[ints]), lower[ints], upper[ints]
                )
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
                    backend=self.jump_backend,
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
            if (
                self.use_fix_and_propagate or self.use_bulk_rounding
            ) and distance < 0.9 * best_distance:
                best_distance = distance
                if self.use_fix_and_propagate:
                    candidate = fix_and_propagate(
                        x, lower, upper, ints, self.row_propagator
                    )
                    if candidate is not None and self.try_incumbent(
                        candidate, "F", warm
                    ):
                        return True
                if self.use_bulk_rounding:
                    for point in self.bulk_round(
                        [x] * self.bulk_starts, lower, upper, deadline, seed=iteration
                    ):
                        if point is not None and self.try_incumbent(point, "F", warm):
                            return True
        return False

    def rins(self, x, warm):
        """RINS around the incumbent and the node LP point ``x``. With the
        sub-MIP lane (``submip_solver="highs"``) the neighbourhood goes to
        HiGHS on its thread (``submit_rins``). Otherwise a Jaddle sub-search
        solves it (``jaddle_rins``), and its node limit (``rins_nodes``)
        doubles after a call that improves the incumbent and halves after
        one that does not, within [50, 2000], as HiGHS adapts its own
        sub-MIP effort."""
        if self.submip_lane is not None:
            self.submit_rins(x)
            return
        started = time.time()
        seconds = self.rins_seconds
        if self.deadline is not None:
            seconds = min(seconds, self.deadline - started)
        before = self.incumbent[0]
        found = self.jaddle_rins(x, seconds, warm)
        self.rins_calls += 1
        if found is not None:
            self.try_incumbent(found, "N", warm)
        improved = self.incumbent[0] < before
        self.rins_finds += improved
        self.rins_nodes = int(
            min(2000, max(50, self.rins_nodes * (2 if improved else 0.5)))
        )
        elapsed = time.time() - started
        self.heuristic_seconds += elapsed
        self.submip_seconds += elapsed
        self.submip_calls += 1

    def submip_budget_left(self):
        """Whether RINS is within ``heuristic_effort`` of the elapsed time
        (never with ``heuristic_effort=0``). RINS on the sub-MIP lane takes
        no time from the search, so it always has budget."""
        if self.submip_lane is not None:
            return True
        return self.submip_seconds < self.heuristic_effort * (time.time() - self.start)

    def submip_seconds_left(self, seconds):
        """``seconds``, cut to the time left before the deadline."""
        if self.deadline is not None:
            seconds = min(seconds, self.deadline - time.time())
        return seconds

    def rins_box(self, x, seconds, min_fixed=0.3, tolerance=None):
        """RINS's neighbourhood for HiGHS: the integers on which the
        incumbent and ``x`` agree, within ``tolerance`` (``None``: the
        integrality tolerance), are fixed at the incumbent's values. A PDHG
        point stops short of the vertex (a binary at 0.9997), so the lane
        passes ``rins_tolerance``. ``None`` when ``seconds`` is not positive,
        fewer than ``min_fixed`` of the integers agree, or the same
        neighbourhood was searched at the same cutoff. Below a fixing rate
        of 1 (``adapt_rins_fixing``), only that share of the agreeing
        integers, drawn at random, is fixed."""
        ints = self.is_int
        incumbent = self.incumbent[1]
        tol = self.int_tol if tolerance is None else tolerance
        agree = ints & (np.abs(np.asarray(x, np.float64) - incumbent) <= tol)
        if seconds <= 0 or agree.sum() < min_fixed * ints.sum():
            return None
        if self.rins_fixing_rate < 1.0:
            agree &= self.rins_rng.random(agree.size) < self.rins_fixing_rate
        key = (agree.tobytes(), incumbent[agree].tobytes(), self.cutoff())
        if key in self.rins_seen:
            self.rins_skipped += 1
            return None
        self.rins_seen.add(key)
        lower = np.where(agree, incumbent, self.base.lower)
        upper = np.where(agree, incumbent, self.base.upper)
        return lower, upper

    def submit_rins(self, x):
        """RINS on the sub-MIP lane around the incumbent and the point
        ``x`` (``rins_box`` at ``rins_tolerance``). With the lane busy the
        call waits (``rins_pending``). Returns whether a job went in."""
        if self.submip_lane.busy():
            self.rins_pending = True
            return False
        seconds = self.submip_seconds_left(self.rins_seconds)
        box = self.rins_box(x, seconds, tolerance=self.rins_tolerance)
        if box is None:
            return False
        self.submit_submip("rins", box, seconds)
        self.rins_calls += 1
        return True

    def adapt_rins_fixing(self, found, seconds):
        """Move RINS's fixing rate after a lane job that took ``seconds``
        and ``found`` a point or not, as SCIP's adaptive LNS moves its
        fixing rates. A job that found nothing and ended early (within
        ``_RINS_EARLY_SHARE`` of its seconds: HiGHS proved the box empty
        below the cutoff) had too small a neighbourhood, so the rate drops
        by ``_RINS_FIXING_STEP``, to at least ``_RINS_MIN_FIXING``; one
        that ran out of time had too large a one, and the rate rises back
        towards 1. On 30n20b8 the incumbent and the node points agree on
        96% of the integers, nearly all zeros of the time-indexed binaries;
        fixing all of them left HiGHS an empty box in 0.02 s on every call,
        while fixing 80% of them found 453 → 353 in 0.8 s."""
        if found:
            return
        if seconds < _RINS_EARLY_SHARE * self.submip_budget:
            rate = self.rins_fixing_rate * _RINS_FIXING_STEP
            self.rins_fixing_rate = max(_RINS_MIN_FIXING, rate)
        else:
            self.rins_fixing_rate = min(1.0, self.rins_fixing_rate / _RINS_FIXING_STEP)

    def submit_submip(self, kind, box, seconds):
        """Hand ``box`` of ``self.base`` to the sub-MIP lane, pruned at the
        cutoff."""
        lower, upper = box
        self.submip_lane.submit(
            kind, self.base, self.is_int, lower, upper, self.cutoff(box=False), seconds
        )
        self.submip_last = kind
        self.submip_budget = float(seconds)
        self.submip_calls += 1

    def remember_solution(self, value, x):
        """Keep the feasible point ``x`` in the solution pool (crossover's):
        the best ``_SOLUTION_POOL_SIZE`` points with distinct integers, best
        first."""
        key = x[self.is_int].tobytes()
        pool = self.solution_pool
        if any(entry[1] == key for entry in pool):
            return
        pool.append((value, key, np.array(x, np.float64)))
        pool.sort(key=lambda entry: entry[0])
        del pool[_SOLUTION_POOL_SIZE:]

    def crossover_box(self, rng, tries=5):
        """Crossover's neighbourhood: the incumbent and
        ``crossover_solutions − 1`` other pool points (drawn at random), with
        the integers on which they all agree fixed there. Pool points outside
        the base box (after a restart) are left out. ``None`` when no draw
        of ``tries`` fixes at least ``crossover_min_fixed`` of the integers in
        a neighbourhood not yet searched at this cutoff."""
        base, ints = self.base, self.is_int
        points = [
            x
            for _, _, x in self.solution_pool
            if np.all(x[ints] >= base.lower[ints]) and np.all(x[ints] <= base.upper[ints])
        ]
        if self.incumbent is None or len(points) < self.crossover_solutions:
            return None
        first = self.incumbent[1]
        others = [x for x in points if not np.array_equal(x[ints], first[ints])]
        need = self.crossover_solutions - 1
        if len(others) < need:
            return None
        for _ in range(tries):
            picks = rng.choice(len(others), size=need, replace=False)
            agree = ints.copy()
            for i in picks:
                agree &= others[i] == first
            if agree.sum() < self.crossover_min_fixed * ints.sum():
                continue
            key = ("crossover", agree.tobytes(), first[agree].tobytes(), self.cutoff())
            if key in self.rins_seen:
                continue
            self.rins_seen.add(key)
            lower = np.where(agree, first, base.lower)
            upper = np.where(agree, first, base.upper)
            return lower, upper
        return None

    def crossover(self):
        """Crossover on the sub-MIP lane (``crossover_box``), as SCIP's
        crossover and cuOpt's recombiners: where several of the search's
        solutions agree, the rest is searched by HiGHS. Needs no LP point,
        so it suits PDHG's interior ones. Returns whether a job went in."""
        seconds = self.submip_seconds_left(self.crossover_seconds)
        if seconds <= 0:
            return False
        box = self.crossover_box(np.random.default_rng(self.crossover_calls))
        if box is None:
            return False
        self.submit_submip("crossover", box, seconds)
        self.crossover_calls += 1
        return True

    def proximity(self):
        """Proximity search on the sub-MIP lane (``_proximity_job``,
        Fischetti & Monaci): the whole base box under the cutoff, searched
        for the point nearest the incumbent. The objective is only a row,
        so HiGHS looks for a feasible point, and needs no LP point, as
        RINS does. Once an incumbent at a cutoff. Returns whether a job
        went in."""
        seconds = self.submip_seconds_left(self.proximity_seconds)
        if seconds <= 0:
            return False
        base, ints = self.base, self.is_int
        center, cutoff = self.incumbent[1], self.cutoff(box=False)
        key = ("proximity", center[ints].tobytes(), cutoff)
        if key in self.rins_seen:
            return False
        self.rins_seen.add(key)
        if not _proximity_distance(center, base.lower, base.upper, ints).any():
            return False
        self.submip_lane.submit_proximity(
            base, ints, center, base.lower, base.upper, cutoff, seconds
        )
        self.submip_last = "proximity"
        self.submip_budget = float(seconds)
        self.submip_calls += 1
        self.proximity_calls += 1
        return True

    def feed_submip(self):
        """Keep the sub-MIP lane busy between node solves: the root's RENS
        when it is waiting; while there is no incumbent, HiGHS on the whole
        root box (``first_solution``) and RENS around the node points
        (``tree_rens``) in turn; then a pending RINS call (around the
        latest node point), proximity search and crossover take turns, and
        each fills the lane when the others have nothing to search."""
        lane = self.submip_lane
        if lane is None or lane.busy() or self.out_of_time():
            return
        if self.rens_seconds:
            # The root's RENS first, when it is waiting (``rens``).
            box, self.rens_pending = self.rens_pending, None
            if box is not None and self.submit_rens(box):
                return
        if self.incumbent is None:
            if getattr(self, "root_lower", None) is not None:
                self.feed_first_solution()
            return
        order = ("rins", "proximity", "crossover")
        start = order.index(self.submip_last) + 1 if self.submip_last in order else 0
        for kind in order[start:] + order[:start]:
            if kind == "rins":
                if self.rins_seconds and self.rins_pending and self.recent_points:
                    self.rins_pending = False
                    if self.submit_rins(self.recent_points[-1]):
                        return
            elif kind == "proximity":
                if self.proximity_seconds and self.proximity():
                    return
            elif self.crossover_seconds and self.crossover():
                return

    def collect_submip(self):
        """Check the point of the sub-MIP lane's finished job, if any, with
        ``try_incumbent`` (RINS ``"N"``, crossover ``"C"``, proximity
        ``"X"``, RENS ``"E"``)."""
        lane = self.submip_lane
        if lane is None:
            return
        done = lane.finished()
        if done is None:
            return
        kind, x, empty, seconds = done
        self.submip_lane_seconds += seconds
        if x is None:
            if kind == "rins":
                self.adapt_rins_fixing(False, seconds)
            self.proximity_empty += empty
            return
        before = None if self.incumbent is None else self.incumbent[0]
        source = {"rins": "N", "proximity": "X", "rens": "E", "first": "W"}.get(kind, "C")
        self.try_incumbent(x, source, None)
        improved = self.incumbent is not None and (
            before is None or self.incumbent[0] < before
        )
        if kind == "rins":
            self.adapt_rins_fixing(improved, seconds)
            self.rins_finds += improved
        elif kind == "proximity":
            self.proximity_finds += improved
        elif kind == "rens":
            self.rens_finds += improved
        elif kind == "first":
            self.first_solution_finds += improved
        else:
            self.crossover_finds += improved

    def service_probing(self):
        """Collect a finished cutoff probing round and start the next
        (``cutoff_probing``). A round that shows the root box holds nothing
        better than the incumbent raises the bound to the cutoff
        (``bound_floor``), which ends the search (``root_closed`` at the
        root) with the incumbent optimal."""
        if self.probe_lane is None or getattr(self, "root_lower", None) is None:
            return
        if not self.collect_probing():
            self.bound_floor = max(self.bound_floor, self.cutoff_bound())
            self.close_all()
            return
        self.feed_probing()

    def feed_probing(self):
        """Start a cutoff probing round (``_ProbeLane``) when the lane is
        idle and there is an incumbent: once per proof cutoff, and again at
        the same cutoff while the last round fixed enough to tighten the
        next one's LP (``_PROBE_AGAIN_SHARE``). A new cutoff waits until
        the room (the cutoff less the global bound) is at most
        ``_PROBE_ROOM_SHRINK`` of the last round's, unless that round fixed
        enough; without a finite bound there is no round. The candidates are the root
        box's free integers, the largest root reduced costs first (the
        sides likeliest to close); a round takes at most
        ``cutoff_probing_seconds`` and a fifth of the time left."""
        lane = self.probe_lane
        if lane is None or lane.busy() or self.incumbent is None or self.out_of_time():
            return
        cutoff = self.proof_cutoff()
        if not np.isfinite(cutoff):
            return
        if cutoff == self.probe_cutoff and not self.probe_again:
            return
        # ``global_bound`` walks the open nodes: once a second at most.
        now = time.time()
        if now - self.probe_checked < 1.0:
            return
        self.probe_checked = now
        bound = self.global_bound()
        if not np.isfinite(bound):
            return
        room = cutoff - bound
        if (
            not self.probe_again
            and self.probe_room is not None
            and room > _PROBE_ROOM_SHRINK * self.probe_room
        ):
            return
        lower, upper = self.root_lower, self.root_upper
        if np.any(lower > upper):
            return
        cols = np.flatnonzero(self.is_int & (lower < upper))
        if cols.size == 0:
            return
        terms = self.root_terms
        if terms is not None and np.size(terms[1]) == lower.size:
            score = np.abs(np.asarray(terms[1], np.float64)[cols])
            cols = cols[np.argsort(-score, kind="stable")]
        budget = self.probe_seconds
        left = self.seconds_left()
        if left is not None:
            budget = min(budget, 0.2 * left)
        if budget <= 0:
            return
        self.probe_cutoff, self.probe_again, self.probe_room = cutoff, False, room
        lane.submit(
            self.mip,
            (self.lp_lower(lower), self.lp_upper(upper)),
            cols,
            cutoff,
            (self.A_T, self.is_int, self.reduced_cost_tol),
            self.probe_iterations,
            time.time() + budget,
        )

    def collect_probing(self):
        """Fix the root box by a finished cutoff probing round, then
        propagate it. Every fix holds for each solution better than the
        cutoff the round probed at, which only falls, over the root box it
        probed, which only shrinks, so the fixes hold for the search from
        here on, restarts included. Returns False when the round showed the
        root box holds nothing better than the incumbent."""
        lane = self.probe_lane
        if lane is None:
            return True
        done = lane.finished()
        if done is None:
            return True
        results, seconds = done
        stats = self.probe_stats
        stats["rounds"] += 1
        stats["seconds"] += seconds
        lower, upper = self.root_lower.copy(), self.root_upper.copy()
        empty, complete, probed = False, True, 0
        for out in results:
            stats["lps"] += out["lps"]
            stats["pivots"] += out["pivots"]
            probed += out["probed"]
            empty |= out["status"] == "empty"
            complete &= out["status"] == "done"
            cols, values = out["upper"]
            if cols:
                cols = np.asarray(cols, np.int64)
                upper[cols] = np.minimum(upper[cols], values)
            cols, values = out["lower"]
            if cols:
                cols = np.asarray(cols, np.int64)
                lower[cols] = np.maximum(lower[cols], values)
        stats["probed"] += probed
        fixed = int(
            np.count_nonzero(
                self.is_int
                & (self.root_lower < self.root_upper)
                & (lower == upper)
            )
        )
        stats["fixed"] += fixed
        if self.verbose:
            print(
                f"Cutoff probing: {fixed} of {probed} integers fixed "
                f"({sum(out['lps'] for out in results)} LPs, {seconds:.1f}s"
                + (", box empty" if empty else "")
                + ")"
            )
        if empty or np.any(lower > upper):
            stats["empty"] = True
            return False
        if fixed:
            if self.propagator is not None:
                lower, upper, infeasible = self.propagator(lower, upper)
                if infeasible:
                    stats["empty"] = True
                    return False
            self.root_lower, self.root_upper = lower, upper
        self.probe_again = complete and fixed >= _PROBE_AGAIN_SHARE * max(probed, 1)
        return True

    def level_box(self):
        """The box the objective's levels are searched in: the root box,
        or the base MIP's before there is one."""
        lower = getattr(self, "root_lower", None)
        if lower is None:
            return self.base.lower, self.base.upper
        return lower, self.root_upper

    def service_levels(self):
        """Check the points the level lane found (``"L"``), then hand it
        the base MIP, root box, cutoff and deadline (``objective_levels``)."""
        lane = self.level_lane
        if lane is None:
            return
        while lane.found:
            _, x = lane.found.popleft()
            before = None if self.incumbent is None else self.incumbent[0]
            self.try_incumbent(x, "L", None)
            self.level_finds += self.incumbent is not None and (
                before is None or self.incumbent[0] < before
            )
        # The tree's bound, once there is a tree: the thread starts there.
        known = self.global_bound() if self.tree_started else -np.inf
        lane.update(
            self.base, self.is_int, *self.level_box(), self.cutoff(), self.deadline, known
        )

    def level_floor(self):
        """A bound on every solution better than the cutoff: the lowest
        objective level not shown empty (``objective_levels``; ``inf``
        when none is left, ``-inf`` without levels)."""
        lane = self.level_lane
        if lane is None:
            return -np.inf
        lower, upper = self.level_box()
        cols = lane.columns
        cutoff = self.cutoff()
        key = (cutoff, lane.proved, lower[cols].tobytes(), upper[cols].tobytes())
        if self.level_floor_cache[0] != key:
            self.level_floor_cache = (key, lane.floor(cutoff, lower, upper))
        return self.level_floor_cache[1]

    def level_stats(self):
        """``result["objective_levels"]``: ``None`` without levels."""
        lane = self.level_lane
        if lane is None:
            return None
        floor = self.level_floor()
        return {
            "columns": [int(j) for j in lane.columns],
            "levels": int(lane.levels.size),
            "empty": int(lane.proved),
            "jobs": int(lane.jobs),
            "finds": int(self.level_finds),
            "seconds": float(lane.busy_seconds),
            "floor": self.model_value(floor) if np.isfinite(floor) else None,
        }

    def levels_settled(self):
        """Whether every objective level below the cutoff is shown empty,
        which ends the search (the incumbent, if any, is optimal)."""
        return self.level_lane is not None and self.level_floor() >= self.cutoff()

    def round_repair(self, x, lower, upper, warm, snap=None):
        """
        Round and repair from the LP point ``x``: its roundings up, to
        nearest and down within ``[lower, upper]`` (``integer_roundings``),
        less those already tried and those whose fixed box propagation shows
        infeasible, each tried as an incumbent (marked ``"U"``):
        ``try_incumbent`` solves for the continuous variables with a Jaddle
        LP when the rounding alone is not feasible. ``snap`` (``None``: the
        PDHG snapping, ``round_snap``) for exact vertices is ``int_tol``.
        Stops after ``round_repair_seconds``.
        """
        if not self.use_round_repair or self.out_of_time():
            return
        if snap is None:
            snap = self.round_snap
        lower = np.asarray(lower, np.float64)
        upper = np.asarray(upper, np.float64)
        started = time.time()
        before = None if self.incumbent is None else self.incumbent[0]
        tried = False
        for values in integer_roundings(x, self.is_int, lower, upper, snap):
            if self.out_of_time() or time.time() - started > self.round_repair_limit:
                break
            key = values.tobytes()
            if key in self.round_repair_seen:
                continue
            # Every rounding gets tried once: a failed or worse one stays so,
            # as the cutoff only falls.
            self.round_repair_seen.add(key)
            fixed_lower = np.where(self.is_int, 0.0, self.mip.lower)
            fixed_upper = np.where(self.is_int, 0.0, self.mip.upper)
            fixed_lower[self.is_int] = values
            fixed_upper[self.is_int] = values
            if self.row_propagator(fixed_lower, fixed_upper)[2]:
                continue
            point = np.clip(np.asarray(x, np.float64), lower, upper)
            point[self.is_int] = values
            tried = True
            self.try_incumbent(point, "U", warm)
        if not tried:
            return
        self.round_repair_calls += 1
        self.round_repair_finds += self.incumbent is not None and (
            before is None or self.incumbent[0] < before
        )
        elapsed = time.time() - started
        self.round_repair_seconds += elapsed
        self.heuristic_seconds += elapsed

    def round_repair_due(self):
        """Whether a tree node gets round and repair: every
        ``round_repair_every`` nodes, while it has taken at most
        ``heuristic_effort`` of the elapsed time."""
        if not (self.use_round_repair and self.round_repair_every):
            return False
        if not self.heuristic_due("round_repair", self.round_repair_every):
            return False
        elapsed = time.time() - self.start
        return self.round_repair_seconds <= self.heuristic_effort * elapsed

    def schedule_improve_jump(self):
        """With ``parallel_heuristics`` and the numba jump: improving jumps
        (``feasibility_jump(improve=True)`` below the cutoff) on every idle
        jump lane (``jump_threads``). A new incumbent gets one slice of
        ``_JUMP_SLICE_SECONDS`` from it; every other idle lane restarts
        (``jump_restarts``) for ``_JUMP_RESTART_SECONDS`` from a perturbed
        incumbent or a recent node's LP point (``jump_restart_start``).
        Before, a slice that found nothing was not repeated until the
        incumbent changed, and the lane sat idle: on ns1830653 the jump
        found nearly every incumbent, and was idle most of the time."""
        if (
            self.workers is None
            or not self.use_jump
            or self.jump_backend != "numba"
            or self.incumbent is None
            or self.out_of_time()
        ):
            return
        value, x = self.incumbent
        fresh = self.jump_slice_from is None or value < self.jump_slice_from
        lower, upper, warm = self.jump_box or (self.base.lower, self.base.upper, None)
        limit = self.jump_iterations
        if limit is None:
            limit = np.iinfo(np.int64).max
        for lane in self.workers.jump_lanes:
            if not self.workers.idle(lane):
                continue
            if fresh:
                start, seconds, kind = x.copy(), _JUMP_SLICE_SECONDS, "jump"
                self.jump_slice_from = value
                fresh = False
            elif self.use_jump_restarts:
                start = self.jump_restart_start(x, lower, upper)
                seconds, kind = _JUMP_RESTART_SECONDS, "jumprestart"
                self.jump_restarts += 1
            else:
                continue
            if self.deadline is not None:
                seconds = min(seconds, self.deadline - time.time())
            if seconds <= 0:
                return
            self.jump_slices += 1
            self.workers.submit(
                lane,
                kind,
                feasibility_jump,
                self.base,
                start,
                lower,
                upper,
                tol=self.feas_tol,
                max_iterations=limit,
                deadline=time.time() + seconds,
                seed=2 + self.jump_slices,
                cutoff=self.cutoff(box=False),
                backend="numba",
                improve=True,
                improve_step=self.objective_step,
                info=warm,
            )

    def jump_restart_start(self, x, lower, upper):
        """A restart point for a background jump (``jump_restarts``),
        alternating between the incumbent ``x`` with a random 5-20% of its
        integers moved (a binary flipped, another integer one step, within
        ``lower``/``upper``) and a recent node's LP point (the tree's view
        near the bound), when there is one."""
        rng = np.random.default_rng(self.jump_slices)
        if self.recent_points and self.jump_restarts % 2:
            return np.array(self.recent_points[rng.integers(len(self.recent_points))])
        start = x.copy()
        ints = np.flatnonzero(self.is_int & (lower < upper))
        if ints.size:
            count = max(1, int(rng.uniform(0.05, 0.2) * ints.size))
            cols = rng.choice(ints, size=count, replace=False)
            step = np.where(rng.random(count) < 0.5, -1.0, 1.0)
            moved = np.clip(start[cols] + step, lower[cols], upper[cols])
            # A step that hit a bound goes the other way instead.
            stuck = moved == start[cols]
            moved[stuck] = np.clip(start[cols][stuck] - step[stuck], lower[cols][stuck], upper[cols][stuck])
            start[cols] = moved
        return start

    def collect_heuristics(self, schedule=True):
        """Check the points of finished background jobs
        (``parallel_heuristics``) with ``try_incumbent`` (jumps marked
        ``J``, fix-and-propagate ``P``), then queue the next improving-jump
        slice (unless ``schedule`` is false). The level lane's points too
        (``service_levels``), cutoff probing's (``service_probing``) and
        the sub-MIP lane's (``collect_submip``, then ``feed_submip``): the
        lane needs no tree, so proximity search and crossover start from
        the first incumbent, during the root's cut rounds. On eil33-2 the
        rounds ran past 100 s with only the jump's points (1,773 against
        an optimum of 934, which HiGHS finds within 60 s)."""
        self.service_levels()
        self.service_probing()
        self.collect_submip()
        self.feed_submip()
        if self.workers is None:
            return
        for kind, found, seconds, warm in self.workers.finished():
            before = None if self.incumbent is None else self.incumbent[0]
            self.worker_jobs += 1
            self.worker_seconds += seconds
            x, feasible, mark = found
            if feasible:
                source = mark if mark in ("P", "J") else _JOB_SOURCES[kind]
                self.try_incumbent(x, source, warm)
            improved = self.incumbent is not None and (
                before is None or self.incumbent[0] < before
            )
            self.worker_finds += improved
            if kind == "jumprestart":
                self.jump_restart_finds += improved
        if schedule:
            self.schedule_improve_jump()

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
        sub_mip = MIP(lp, self.is_int, base.offset, base.maximise, base.implied)
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
            rins_seconds=0.0,
            rens_seconds=0.0,
            crossover_seconds=0.0,
            proximity_seconds=0.0,
            first_solution_seconds=0.0,
            submip_solver="jaddle",
            cutoff_probing=False,
            dive_every=0,
            tree_heuristics=False,
            node_crossover=False,
            batch_size=1,
            verbose=False,
            parallel_heuristics=False,
            block_decomposition=False,
            trivial_heuristic=False,
            interrupt=self.interrupt,
        )
        if settings["branching"] in ("auto", "reliability"):
            # Strong branching would spend the sub-MIP's short budget.
            settings["branching"] = "pseudocost"
        settings["probing"] = False
        # A sub-MIP's box breaks most symmetry anyway.
        settings["symmetry"] = False
        settings["restarts"] = 0
        # Nor the root sweep or crossover (the sub-MIP has no incumbent to fix with).
        settings["root_strong_candidates"] = 0
        settings["root_crossover"] = False
        settings["round_repair"] = False
        settings["exact_nodes"] = 0
        settings["mip_start"] = None
        # A sub-MIP's cutoff is its parent's, not a solution of its own:
        # it dives for one (and needs no node cuts in its short budget).
        settings["cutoff_from_start"] = False
        settings["objective_levels"] = False  # the parent's lane has them
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
        searched for ``rens_seconds`` and ``rens_nodes`` nodes by a Jaddle
        sub-search (``sub_search``). Needs no incumbent; a cutoff, when there is one, keeps only better points. Skipped when
        fewer than ``min_fixed`` of the integers are integral. A point found
        is marked ``"E"``.
        """
        box = self.rens_box(x, lower, upper, min_fixed, self.int_tol)
        if box is None:
            return
        if self.submip_lane is not None:
            # HiGHS searches it on the lane (``submit_rens``), now or once
            # the lane is free, while the root goes on.
            self.rens_pending = box
            self.feed_submip()
            return
        sub_lower, sub_upper = box
        started = time.time()
        seconds = self.rens_seconds
        if self.deadline is not None:
            seconds = min(seconds, self.deadline - started)
        if seconds <= 0:
            return
        found = self.sub_search(sub_lower, sub_upper, seconds, self.rens_nodes, warm)
        self.submip_calls += 1
        if found is not None:
            self.try_incumbent(found, "E", warm)
        self.heuristic_seconds += time.time() - started

    def rens_box(self, x, lower, upper, min_fixed, tolerance):
        """RENS's neighbourhood of the point ``x`` within ``[lower,
        upper]``: the integers within ``tolerance`` of a whole number (in
        the box) fixed there, the others restricted to ``[floor, ceil]``;
        the continuous variables keep the base MIP's bounds. ``None`` when
        fewer than ``min_fixed`` of the integers are fixed, or the box is
        empty."""
        ints = self.is_int
        x = np.asarray(x, np.float64)
        near = np.round(x)
        integral = (
            ints & (np.abs(x - near) <= tolerance) & (near >= lower) & (near <= upper)
        )
        if integral.sum() < min_fixed * ints.sum():
            return None
        sub_lower = np.where(
            ints, np.maximum(lower, np.floor(x + tolerance)), self.base.lower
        )
        sub_upper = np.where(
            ints, np.minimum(upper, np.ceil(x - tolerance)), self.base.upper
        )
        sub_lower = np.where(integral, near, sub_lower)
        sub_upper = np.where(integral, near, sub_upper)
        if np.any(sub_lower > sub_upper):
            return None
        return sub_lower, sub_upper

    def submit_rens(self, box):
        """RENS's neighbourhood ``box`` to the sub-MIP lane for HiGHS, for
        ``rens_seconds``, unless it was searched at this cutoff already.
        Returns whether a job went in."""
        seconds = self.submip_seconds_left(self.rens_seconds)
        if seconds <= 0:
            return False
        # Hashed: a call per node point would keep two copies of the box
        # each otherwise.
        key = ("rens", hash((box[0].tobytes(), box[1].tobytes())), self.cutoff())
        if key in self.rins_seen:
            return False
        self.rins_seen.add(key)
        self.submit_submip("rens", box, seconds)
        self.rens_calls += 1
        return True

    def tree_rens(self):
        """RENS around the latest node LP point while there is no
        incumbent (``rens_seconds``, with the sub-MIP lane): RINS,
        proximity search and crossover all start from an incumbent, so
        before the first one RENS is the lane's only work. A node point
        comes from a deeper box than the root's, and PDHG's points stop
        short of their vertex, so an integer within ``rins_tolerance`` of a
        whole number counts as integral; at least
        ``_TREE_RENS_MIN_FIXED`` of the integers must be. On piperout-08
        the root vertex's neighbourhood is infeasible (HiGHS shows it in
        no time) and Jaddle's search found no incumbent, while HiGHS's
        first one, the optimum, came from a root sub-MIP after its probing
        and cuts. Returns whether a job went in."""
        if self.incumbent is not None or not self.recent_points:
            return False
        box = self.rens_box(
            self.recent_points[-1],
            self.root_lower,
            self.root_upper,
            _TREE_RENS_MIN_FIXED,
            self.rins_tolerance,
        )
        return box is not None and self.submit_rens(box)

    def feed_first_solution(self):
        """The lane's work before the first incumbent: HiGHS on the whole
        root box (``submit_first_solution``) and RENS around a node point
        (``tree_rens``) take turns, each filling the lane when the other
        has nothing to search."""
        first = bool(self.first_solution_seconds)
        if first and self.submip_last != "first" and self.submit_first_solution():
            return
        if self.rens_seconds and self.tree_rens():
            return
        if first:
            self.submit_first_solution()

    def submit_first_solution(self):
        """HiGHS's MIP solver on the whole root box (``first_solution_seconds``),
        stopped at its first point (``mip_max_improving_sols=1``), as
        proximity search searches the whole box once there is an incumbent:
        HiGHS's own probing, cuts and heuristics put to finding a first
        point. Each call gets twice the last one's time, since HiGHS starts
        over each time. On piperout-08 every RENS neighbourhood of the root
        vertex and of the node points was infeasible (fixing even a random
        tenth of the integral binaries was), and Jaddle had no incumbent
        after 120 s; HiGHS on the whole MIP finds its first point, the
        optimum, at 20 s. Marked ``"W"``. Returns whether a job went in."""
        seconds = self.submip_seconds_left(self.first_solution_budget)
        if seconds <= 0:
            return False
        self.submip_lane.submit(
            "first",
            self.base,
            self.is_int,
            self.lp_lower(self.root_lower),
            self.lp_upper(self.root_upper),
            self.cutoff(box=False),
            seconds,
            options={"mip_max_improving_sols": 1},
        )
        self.submip_last = "first"
        self.submip_budget = float(seconds)
        self.submip_calls += 1
        self.first_solution_calls += 1
        self.first_solution_budget *= 2.0
        return True

    def dive_seconds(self):
        """The time a dive may take: the heuristic budget
        (``heuristic_seconds``; ``None``: 20% of ``max_seconds``, as the root
        heuristics get). On buildingenergy, two root dives without a limit
        took 4.5 minutes of 50-epoch re-solves of a 145,000-variable LP,
        fixing one of 8,000 fractional integers a step."""
        if self.heuristic_budget is not None:
            return float(self.heuristic_budget)
        if self.deadline is not None:
            return 0.2 * (self.deadline - self.start)
        return np.inf

    def dive(self, x, lower, upper, warm, rule=None, deadline=None):
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
        them. It stops at ``deadline`` (``None``: ``dive_seconds`` from now).
        """
        started = time.time()
        if deadline is None:
            deadline = started + self.dive_seconds()
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
                    ints
                    & (np.abs(x - np.round(x)) > self.int_tol)
                    & (x > lower)
                    & (x < upper)
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
                    mean = np.array(
                        [
                            (
                                self.pc_sum[d][known[d]].sum()
                                / self.pc_count[d][known[d]].sum()
                                if known[d].any()
                                else 1.0
                            )
                            for d in range(2)
                        ]
                    )
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
                if time.time() >= deadline:
                    return
                result = self.solve_lp(self.node_lp(lower, upper), warm, epochs)
                if result["stop_reason"] in ("interrupted", "time_limit"):
                    return
                bound, _ = self.bound_over(
                    self.result_terms(result, lower, upper), lower, upper
                )
                if self.round_bound(bound) >= self.cutoff():
                    return
                x = np.asarray(result["solution"].primal, np.float64)
                warm = (result["solution"], result["opt_state"])
            self.try_incumbent(x, "D", warm)
        finally:
            self.heuristic_seconds += time.time() - started

    def gpu_dive_due(self):
        """Whether the GPU dives run after this batch: on, every
        ``gpu_dive_every`` batches, within ``gpu_dive_share`` of the tree's
        time (after its first seconds)."""
        if not self.gpu_dive_every or self.batch_size < 2 or self.out_of_time():
            return False
        if self.batches % self.gpu_dive_every:
            return False
        tree = self.tree_elapsed()
        return tree <= _TREE_CUT_GRACE_SECONDS or (
            self.gpu_dive_seconds < self.gpu_dive_share * tree
        )

    def gpu_dives(self, starts):
        """
        Batched LP dives on the device, for incumbents: up to ``batch_size``
        dives at once, from the lowest-bound ``starts`` (``(bound, x, lower,
        upper, warm)`` of nodes that branched), each step one batched call
        (``solve_batch``) for every dive still alive.

        A step fixes, in each dive, every free integer within
        ``gpu_dive_snap`` of an integer, or else the least fractional
        ``gpu_dive_fix_share`` of the fractional ones (at least one), to
        its nearest integer, or towards the incumbent in every other dive
        (guided diving); propagates (objective-aware,
        ``heuristic_propagator``), halving a rejected chunk down to one
        variable, whose other rounding is tried last; a chunk whose LP bound
        then reaches the cutoff is undone and the next is a quarter its
        size (growing back twofold a step after); then re-solves the dives' LPs warm for
        ``gpu_dive_epochs``. A dive ends at an integral LP point (tried as
        an incumbent, ``"D"``), an infeasible box, a bound at the cutoff, or
        after ``gpu_dive_steps``. Truncated node LPs leave most integers
        fractional (ns1830653: a median of 263 of 458, never under 122, at
        depths to 26), and a node's children reach a batch two steps later,
        so the tree itself reaches no integral leaf; fixing chunks, the
        dives reach one in a few dozen batched calls.
        """
        started = time.time()
        ints = self.is_int
        starts = sorted(starts, key=lambda start: start[0])[: self.batch_size]
        guided = self.incumbent is not None
        dives = [
            dict(
                x=np.asarray(x, np.float64),
                lower=np.array(lower, np.float64),
                upper=np.array(upper, np.float64),
                warm=warm,
                guide=guided and i % 2 == 1,
                # The lowest-bound few cross each LP over to a vertex.
                crossover=i < self.gpu_dive_crossover,
                limit=None,  # the most integers a step fixes (None: no limit)
                last=None,  # the last step's single fix (column, value)
                fixed=0,  # how many the last step fixed
                previous=None,  # the box before the last step
            )
            for i, (_, x, lower, upper, warm) in enumerate(starts)
        ]
        self.gpu_dive_stats["runs"] += 1
        self.gpu_dive_stats["dives"] += len(dives)
        before = None if self.incumbent is None else self.incumbent[0]
        try:
            for _ in range(self.gpu_dive_steps):
                alive = []
                for dive in dives:
                    if self.fix_dive_chunk(dive, ints):
                        alive.append(dive)
                if not alive or self.out_of_time():
                    break
                self.gpu_dive_stats["steps"] += 1
                results = self.solve_batch(
                    [(dive["lower"], dive["upper"]) for dive in alive],
                    [dive["warm"][0] for dive in alive],
                    [dive["warm"][1] for dive in alive],
                    self.gpu_dive_epochs,
                    self.batch_size,
                )
                dives = []
                cutoff = self.cutoff()
                for dive, result in zip(alive, results):
                    raw, _ = self.bound_over(
                        self.result_terms(result, dive["lower"], dive["upper"]),
                        dive["lower"],
                        dive["upper"],
                    )
                    if self.round_bound(raw) >= cutoff:
                        if dive["fixed"] > 1 and dive["previous"] is not None:
                            # The chunk cost too much: back to the box
                            # before it, a quarter of it next time.
                            self.gpu_dive_stats["backtracks"] += 1
                            dive["lower"], dive["upper"] = dive["previous"]
                            dive["limit"] = max(1, dive["fixed"] // 4)
                            dive["fixed"], dive["previous"] = 0, None
                            dives.append(dive)
                            continue
                        last = dive.get("last")
                        if last is not None and dive["previous"] is not None:
                            # One variable's rounding cut it off: its other
                            # rounding, once, from the box before it.
                            j, value = last
                            lower, upper = dive["previous"]
                            other = value - 1.0 if value > dive["x"][j] else value + 1.0
                            if lower[j] <= other <= upper[j]:
                                self.gpu_dive_stats["flips"] += 1
                                dive["lower"], dive["upper"] = lower, upper
                                dive["flip"] = (j, other)
                                dive["previous"] = dive["last"] = None
                                dives.append(dive)
                                continue
                        self.gpu_dive_stats["cut off"] += 1
                        continue  # no better point below
                    dive["x"] = np.asarray(result["solution"].primal, np.float64)
                    dive["warm"] = (result["solution"], result["opt_state"])
                    if dive["crossover"]:
                        self.cross_over_dive(dive)
                    if dive["limit"] is not None:
                        dive["limit"] *= 2  # it held: grow again
                    dives.append(dive)
                if not dives:
                    break
            for dive in dives:  # out of steps: the last points, rounded
                self.gpu_dive_stats["out of steps"] += 1
                self.try_incumbent(dive["x"], "D", dive["warm"])
        finally:
            elapsed = time.time() - started
            self.gpu_dive_seconds += elapsed
            self.heuristic_seconds += elapsed
            self.gpu_dive_stats["finds"] += self.incumbent is not None and (
                before is None or self.incumbent[0] < before
            )

    def cross_over_dive(self, dive):
        """Cross a GPU dive's LP point over to an optimal vertex with HiGHS
        (``jaddle_linear.crossover``, on the host): the dive then fixes from
        the vertex, where far more integers are exactly integral (on
        ns1830653's dive LPs about half as many fractional as at the PDHG
        point, in 50-90 ms), and warm-starts its next LP there. Kept as is
        when HiGHS does not prove the vertex optimal in time."""
        started = time.time()
        limit = 10.0
        if self.deadline is not None:
            limit = min(limit, max(self.seconds_left(), 0.0))
        try:
            crossed = jl.crossover(
                self.node_lp(dive["lower"], dive["upper"]),
                dive["warm"][0],
                highs_options={"time_limit": limit},
            )
        finally:
            self.gpu_dive_stats["crossovers"] += 1
            self.gpu_dive_crossover_seconds += time.time() - started
        if not crossed["converged"]:
            return
        dive["x"] = np.asarray(crossed["solution"].primal, np.float64)
        # In the shapes of the dive's PDHG warm start: the batch stacks the
        # dives' warm starts, and ``crossover`` leaves out the zero row
        # ``solve()`` gives an empty row block (on pg5_34, presolved at a
        # restart down to no equality rows, ``dual_eq`` came back ``(0,)``
        # beside the others' ``(1,)``, and the batch raised).
        dive["warm"] = (_shaped_like(crossed["solution"], dive["warm"][0]), dive["warm"][1])

    def fix_dive_chunk(self, dive, ints):
        """One step of a GPU dive (``gpu_dives``) on the host: fix a chunk of
        its free integers in its box and propagate. Returns whether the dive
        goes on to a re-solve; a dive at an integral point is tried as an
        incumbent and ends."""
        x, lower, upper = dive["x"], dive["lower"], dive["upper"]
        free = ints & (lower < upper)
        distance = np.abs(x - np.round(x))
        fractional = free & (distance > self.int_tol)
        if not fractional.any():
            self.gpu_dive_stats["leaves"] += 1
            self.try_incumbent(x, "D", dive["warm"])
            return False
        if dive.get("flip") is not None:
            # The other rounding of a single fix whose LP was cut off.
            j, value = dive.pop("flip")
            lo, up = lower.copy(), upper.copy()
            lo[j] = up[j] = value
            lo, up, infeasible = self.heuristic_propagator()(lo, up)
            if infeasible:
                self.gpu_dive_stats["infeasible"] += 1
                return False
            dive["fixed"], dive["previous"], dive["last"] = 1, None, None
            dive["lower"], dive["upper"] = lo, up
            return True
        target = np.clip(np.round(x), lower, upper)
        if dive["guide"] and self.incumbent is not None:
            # Towards the incumbent: its value, or the nearer integer on its
            # side of x.
            incumbent = self.incumbent[1]
            target = np.clip(
                np.where(incumbent > x, np.ceil(x), np.floor(x)), lower, upper
            )
        chunk = free & (distance <= self.gpu_dive_snap)
        if not (chunk & fractional).any():
            cand = np.flatnonzero(fractional)
            count = max(1, math.ceil(self.gpu_dive_fix_share * cand.size))
            chunk = np.zeros_like(free)
            chunk[cand[np.argsort(distance[cand], kind="stable")[:count]]] = True
        if dive["limit"] is not None and chunk.sum() > dive["limit"]:
            # After a backtrack: the most nearly integral few only.
            cand = np.flatnonzero(chunk)
            chunk = np.zeros_like(free)
            chunk[cand[np.argsort(distance[cand], kind="stable")[: dive["limit"]]]] = True
        propagator = self.heuristic_propagator()
        # The chunk, most nearly integral first; one propagation rejects
        # it: its first half, and so on down to one variable, which then
        # tries its other rounding (as ``fix_and_propagate`` splits).
        order = np.flatnonzero(chunk)
        order = order[np.argsort(distance[order], kind="stable")]
        size = order.size
        tries = []
        while size >= 1:
            tries.append((order[:size], target[order[:size]]))
            if size == 1:
                break
            size //= 2
        j = order[0]
        if fractional[j]:
            other = np.floor(x[j]) if target[j] > x[j] else np.ceil(x[j])
            if lower[j] <= other <= upper[j] and other != target[j]:
                tries.append((order[:1], np.array([other])))
        for cols, values in tries:
            lo, up = lower.copy(), upper.copy()
            lo[cols] = values
            up[cols] = values
            lo, up, infeasible = propagator(lo, up)
            if not infeasible:
                dive["fixed"] = int(cols.size)
                self.gpu_dive_stats["fixed"] += dive["fixed"]
                dive["previous"] = (lower, upper)
                dive["last"] = (int(cols[0]), float(values[0])) if cols.size == 1 else None
                dive["lower"], dive["upper"] = lo, up
                return True
        if dive["previous"] is not None and dive["fixed"] > 1:
            # Every try is rejected: the last chunk left no room. Undo it
            # and fix a quarter as many next time.
            self.gpu_dive_stats["backtracks"] += 1
            dive["lower"], dive["upper"] = dive["previous"]
            dive["limit"] = max(1, dive["fixed"] // 4)
            dive["fixed"], dive["previous"] = 0, None
            return True
        self.gpu_dive_stats["infeasible"] += 1
        return False

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
            and self.tree_heuristic_seconds
            < self.tree_heuristic_share * (time.time() - self.start)
            and not self.out_of_time()
            and self.heuristic_due("tree", self.tree_heuristic_every)
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
            self.tree_heuristic_share * (started - self.start)
            - self.tree_heuristic_seconds
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
                data = None  # the numba jump builds its rows each call (cheap)
                if self.jump_backend == "jax":
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
                    backend=self.jump_backend,
                    # With numba, it goes on below each point it finds.
                    improve=self.jump_backend == "numba",
                    improve_step=self.objective_step,
                )
                if feasible:
                    self.try_incumbent(point, "J", warm)
        finally:
            self.tree_heuristic_calls += 1
            self.tree_heuristic_finds += int(self.incumbent[0] < before)
            elapsed = time.time() - started
            self.tree_heuristic_seconds += elapsed
            self.heuristic_seconds += elapsed

    def improve_jump(self, start, lower, upper, warm, deadline, seed):
        """``feasibility_jump`` with ``improve`` (numba) from ``start``, below
        the cutoff, until ``deadline``; its best point goes through
        ``try_incumbent``. With a finite deadline and no ``jump_iterations``,
        only the deadline stops it: it keeps finding better points."""
        limit = self.jump_iterations
        if limit is None and np.isfinite(deadline):
            limit = np.iinfo(np.int64).max
        x, feasible, _ = feasibility_jump(
            self.base,
            start,
            lower,
            upper,
            tol=self.feas_tol,
            max_iterations=limit,
            deadline=None if not np.isfinite(deadline) else deadline,
            seed=seed,
            cutoff=self.cutoff(box=False),
            backend="numba",
            improve=True,
            improve_step=self.objective_step,
        )
        if feasible:
            self.try_incumbent(x, "J", warm)

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
            cutoff=None if self.incumbent is not None else self.jump_cutoff(),
            backend=self.jump_backend,
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

    def root_heuristics(self, lower, upper, result, improve=True):
        """Fix-and-propagate from the root LP point, then the feasibility
        pump while there is no incumbent, within the heuristic budget. It
        runs on the first root LP, so the cut rounds have an incumbent (and
        the background improving jump runs beside them), and again after
        them; the two calls share one budget. Without background workers,
        the improving jump from the incumbent runs only when ``improve``
        (after the cut rounds: before them it would take the whole
        budget)."""
        started = time.time()
        if self.heuristic_budget is not None:
            budget = self.heuristic_budget
        elif self.deadline is not None:
            budget = 0.2 * (self.deadline - self.start)
        else:
            budget = np.inf
        deadline = started + max(0.0, budget - self.root_heuristic_used)
        warm = (result["solution"], result["opt_state"])
        x = np.asarray(result["solution"].primal, np.float64)
        if self.use_fix_and_propagate:
            candidate = fix_and_propagate(
                x, lower, upper, self.is_int, self.heuristic_propagator()
            )
            if candidate is not None:
                self.try_incumbent(candidate, "P", warm)
        # The stages after fix-and-propagate (bulk rounding, two jump starts,
        # the pump) each get an equal share of the time left when they
        # start, so a slow one cannot starve the rest. On
        # academictimetablesmall, bulk rounding and before it the device
        # jump from the LP point took the whole budget, though a jump from
        # zero finds a point in under a second.
        stages = [
            int(self.use_bulk_rounding) + 2 * int(self.use_jump) + int(self.use_pump)
        ]

        def stage_deadline():
            now = time.time()
            own = deadline
            if np.isfinite(deadline) and stages[0] > 1:
                own = now + (deadline - now) / stages[0]
            stages[0] -= 1
            return own

        if self.use_bulk_rounding and self.incumbent is None:
            # Several bulk roundings of the LP point at once: they succeed on
            # other models than the classic one.
            own = stage_deadline()
            for point in self.bulk_round([x] * self.bulk_starts, lower, upper, own):
                if point is not None:
                    self.try_incumbent(point, "B", warm)
        if self.use_jump and self.incumbent is None:
            # From the rounded LP point, then from the bounds nearest zero
            # (unless the early root jumps are still on it).
            starts = ((x, 0), (np.clip(np.zeros_like(x), lower, upper), 1))
            if self.early_jumps_running():
                starts = starts[:1]
            for start, seed in starts:
                if self.incumbent is not None or time.time() >= deadline:
                    break
                self.jump(start, lower, upper, warm, stage_deadline(), seed)
        if self.use_pump and self.incumbent is None:
            self.feasibility_pump(lower, upper, result, deadline)
            if (
                self.use_jump
                and self.incumbent is None
                and self.pump_last is not None
                and time.time() < deadline
            ):
                self.jump(self.pump_last, lower, upper, warm, deadline, seed=2)
        # The box and warm start of the background improving jump.
        self.jump_box = (lower.copy(), upper.copy(), warm)
        if self.workers is not None:
            self.schedule_improve_jump()
        elif (
            improve
            and self.use_jump
            and self.jump_backend == "numba"
            and self.incumbent is not None
            and time.time() < deadline
        ):
            # The time left goes to better points: a jump from the incumbent
            # that goes on below each point it finds. On
            # academictimetablesmall it takes a first point of about 5,600
            # down to the hundreds within seconds.
            self.improve_jump(self.incumbent[1], lower, upper, warm, deadline, seed=3)
        self.collect_heuristics(schedule=False)
        self.heuristic_seconds += time.time() - started
        self.root_heuristic_used += time.time() - started

    def next_node(self):
        """The next node to solve: the pending plunge child, or else the top
        of the heap (which starts a new plunge). With exact plunges
        (``exact_plunges``) the plunge child is theirs: the batches take the
        heap's top."""
        if self.plunge_node is not None and not self.exact_plunges_on():
            node, self.plunge_node = self.plunge_node, None
            return node
        _, _, node = heapq.heappop(self.open)
        self.plunge_left = self.plunge_depth
        return node

    def node_batch(
        self, lp_lower, lp_upper, warm_starts, max_epochs=None, steps=None, lp=None
    ):
        """The node LPs over the boxes ``lp_lower``/``lp_upper`` (``(batch,
        n)``), each warm-started from its row of ``warm_starts`` and of
        ``steps`` (the step sizes ``(k, eta)``, each ``(batch,)``; ``None``:
        ``solve``'s defaults), in one device call
        (``jaddle_linear.solve_many``) at ``node_precision`` with the node
        options, for ``max_epochs`` (``None``: ``node_max_epochs``). With
        ``node_bound_stop`` a member also stops once its Lagrangian bound
        passes ``node_bound_target()``. The LP's rows go in as data, so a new
        cut block of the same capacity reuses the compiled solver. Returns a
        ``jaddle_linear.SolveCoreResult`` and the members' final step sizes
        ``(k, eta)``; their arrays stay on the device."""
        if max_epochs is None:
            max_epochs = self.node_max_epochs
        if self._node_batch_settings is None:
            import inspect

            accepted = set(inspect.signature(jl.solve_many).parameters) - {
                "lp",
                "lower_bounds",
                "upper_bounds",
                "initial_solutions",
                "initial_opt_states",
                "max_epochs",
                "bound_target",
                "device",
                "precision",
            }
            unknown = sorted(set(self.node_lp_options) - accepted)
            if unknown:
                raise ValueError(
                    f"node_lp_options {', '.join(unknown)} cannot be used by batched "
                    "node LPs (jaddle_linear.solve_many)"
                )
            # Node LPs: ``lp_options`` (what solve_many takes), then the
            # node layer (``node_lp_tolerance``, ``node_lp_options``).
            self._node_batch_settings = {
                key: value
                for key, value in {**self.lp_kwargs, **self.node_lp_kwargs}.items()
                if key in accepted
            }
        target = None
        if self.node_bound_stop:
            target = self.node_bound_target()
            target = np.inf if target is None else float(target)
        with self.node_settings():
            out = jl.solve_many(
                self.node_mip_lp if lp is None else lp,
                lp_lower,
                lp_upper,
                initial_solutions=warm_starts,
                initial_opt_states=steps,
                max_epochs=max_epochs,
                bound_target=target,
                **self._node_batch_settings,
            )
        core = jl.SolveCoreResult(
            out["solution"], out["converged"], out["stop_code"], out["epochs"]
        )
        return core, tuple(out["opt_state"])[:2]

    def solve_batch(self, boxes, solutions, opt_states, max_epochs, size):
        """``read_batch`` of ``launch_batch``: the batch's results, waited
        for."""
        return self.read_batch(
            self.launch_batch(boxes, solutions, opt_states, max_epochs, size)
        )

    def launch_batch(self, boxes, solutions, opt_states, max_epochs, size):
        """Dispatch the node LPs of ``boxes`` to the device without waiting
        (``read_batch`` reads them back): one vmapped call, each warm-started
        from its solution, padded to ``size`` members with copies of the
        last. With ``node_bound_stop``, a member also stops once its
        Lagrangian bound passes ``node_bound_target()`` (``stop_reason``
        ``"bound_target"``). Each member also starts from the step sizes
        ``(k, eta)`` of its ``opt_states`` entry (a member without one takes
        another's: they share the LP), so a child does not re-learn its
        parent's. Returns a handle for ``read_batch``."""
        count = len(opt_states)
        pad = size - len(boxes)
        boxes = list(boxes) + [boxes[-1]] * pad
        solutions = list(solutions) + [solutions[-1]] * pad
        steps = [_step_sizes(state) for state in opt_states]
        known = [step for step in steps if step is not None]
        if known:
            steps = [known[0] if step is None else step for step in steps]
            steps += [steps[-1]] * pad
        lp_lower = np.stack([self.lp_lower(lo) for lo, _ in boxes])
        lp_upper = np.stack([self.lp_upper(up) for _, up in boxes])
        with self.node_settings():
            # Root and node points may differ in precision (``node_precision``).
            # The members' points are host arrays (each batch is read back
            # whole), so they are stacked and cast on the host and sent in one
            # transfer per leaf: casting and sending them member by member
            # took half of ``solve_batch`` with batches of 128 (ns1830653).
            node_dtype = np.dtype(self.node_dtype)

            def stack(*leaves):
                block = np.stack([np.asarray(leaf) for leaf in leaves])
                if np.issubdtype(block.dtype, np.floating) and block.dtype != node_dtype:
                    block = block.astype(node_dtype)
                return jnp.asarray(block)

            warm_starts = jax.tree.map(stack, *solutions)
            dtype = _wide(self.node_dtype)
            steps = (
                tuple(
                    jnp.asarray(np.array([step[j] for step in steps]), dtype)
                    for j in (0, 1)
                )
                if known
                else None
            )
        if self.pipeline_batches:
            # XLA drives a data-dependent while loop (the epochs) from the
            # host on a GPU, so the call only returns once the LPs are done:
            # it runs on the device thread, while this one evaluates the
            # previous batch. The LP is taken now, before a tree cut refresh
            # here can replace it.
            return (
                count,
                self.on_device_thread(
                    self.node_batch,
                    lp_lower,
                    lp_upper,
                    warm_starts,
                    max_epochs,
                    steps,
                    self.node_mip_lp,
                ),
            )
        return (
            count,
            self.node_batch(lp_lower, lp_upper, warm_starts, max_epochs, steps),
        )

    def on_device_thread(self, fn, *args):
        """Run ``fn(*args)`` on the search's device thread
        (``pipeline_batches``), in submission order, under the search's
        device. Returns a ``Future``."""
        if self._device_thread is None:
            from concurrent.futures import ThreadPoolExecutor

            self._device_thread = ThreadPoolExecutor(
                1, thread_name_prefix="jaddle-device"
            )
        device = self.device

        def call():
            # Waited for here: on a CPU, JAX returns before the work is
            # done, and the future must only be done once its arrays are
            # (``await_device`` services the exact lane until then).
            if device is None:
                return jax.block_until_ready(fn(*args))
            with jax.default_device(device):
                return jax.block_until_ready(fn(*args))

        return self._device_thread.submit(call)

    def await_device(self, future):
        """Wait for a device thread call (``on_device_thread``), servicing
        the exact lane meanwhile (``service_exact``): the lane's finished
        nodes are evaluated and its threads handed their next as soon as
        one is done, instead of once a step. On ns1830653 (CPU) the main
        thread spent half the tree blocked on the batches' LPs with 4-8 of
        the 16 lane threads at work. Called only where every node is in a
        list the global bound counts and no node is being evaluated."""
        from concurrent.futures import FIRST_COMPLETED, wait

        lane = self.exact_lane
        while lane is not None and not self._lane_busy and not future.done():
            wait(
                [future, *(job[0] for job in lane.jobs)],
                timeout=0.05,
                return_when=FIRST_COMPLETED,
            )
            if future.done() or self.out_of_time():
                break
            self.service_exact()
        return future.result()

    def service_exact(self):
        """Evaluate the lane's finished nodes and refill it
        (``collect_exact``, ``feed_exact``), between the tree loop's own
        calls (``await_device``)."""
        started = time.perf_counter()
        try:
            self.collect_exact()
            self.feed_exact()
        finally:
            spent = time.perf_counter() - started
            self.exact_collect_seconds += spent
            self.exact_service_seconds += spent

    def read_batch(self, handle, service=False):
        """The results of a ``launch_batch``, one ``solve_lp``-style result
        per box (padding dropped), whose ``opt_state`` is the member's
        final ``(k, eta)``. Waits for the device; with ``service``, the
        exact lane is serviced meanwhile (``await_device``)."""
        count, launched = handle
        started = time.perf_counter()
        serviced = self.exact_service_seconds
        if hasattr(launched, "result"):  # dispatched on the device thread
            launched = (
                self.await_device(launched) if service else launched.result()
            )
        # One transfer for the whole batch, then slices on the host.
        out, steps = jax.device_get(launched)
        self.batch_wait_seconds += (
            time.perf_counter() - started - (self.exact_service_seconds - serviced)
        )
        converged = np.asarray(out.converged)
        stop_code = np.asarray(out.stop_code)
        epochs = np.asarray(out.epochs)
        return [
            {
                "solution": jax.tree.map(lambda leaf: leaf[i], out.solution),
                "converged": bool(converged[i]),
                "stop_reason": {
                    1: "certificate",
                    2: "primal_stall",
                    4: "bound_target",
                }.get(int(stop_code[i]), "max_epochs"),
                "epochs": int(epochs[i]),
                "opt_state": (steps[0][i], steps[1][i]),
            }
            for i in range(count)
        ]

    def batch_step(self):
        """Solve up to ``batch_size`` open nodes in one batched device call,
        then evaluate them in order on the host exactly as the serial search
        does. A short batch is padded with copies of its last member, whose
        results are ignored. With ``pipeline_batches``, ``pipelined_step``
        instead."""
        if self.pipeline_batches:
            self.pipelined_step()
            return
        batch = self.launch_next()
        if batch is not None:
            self.process_batch(batch)
            self.flush_branches()
        if self.exact_plunges_on():
            self.exact_plunge(force=batch is None)

    def pipelined_step(self):
        """One step of the pipelined batch loop (``pipeline_batches``):

        1. the children of the batch processed last step, built on the
           device after the batch then in flight, are read and queued;
        2. the next batch is picked (those children among the candidates)
           and dispatched, without waiting;
        3. the batch dispatched last step is read back and evaluated on the
           host while the device solves the next one, and its children are
           dispatched to follow it.

        So the host's evaluation of one batch (bounds, cuts, heuristics,
        branching) overlaps the device's LPs for the next, where the plain
        loop left the device idle meanwhile. A node's children reach a
        batch two steps after it."""
        pending = self.pending_children
        if pending is not None:
            # Waited for while they are still ``pending_children`` (the
            # global bound counts them there), servicing the exact lane.
            started = time.perf_counter()
            serviced = self.exact_service_seconds
            if hasattr(pending["out"], "result"):
                self.await_device(pending["out"])
            self.batch_wait_seconds += (
                time.perf_counter() - started
                - (self.exact_service_seconds - serviced)
            )
            self.pending_children = None
            self.finish_branches(pending)
        current, self.in_flight = self.in_flight, self.launch_next()
        if current is not None:
            self.process_batch(current)
            self.pending_children = self.launch_branches()
        if self.exact_plunges_on():
            # With nothing else to do (only the plunge child left), the
            # plunge runs whatever its share, or the loop would stall.
            idle = (
                current is None
                and self.in_flight is None
                and self.pending_children is None
            )
            self.exact_plunge(force=idle)

    def exact_plunges_on(self):
        """Whether exact plunges (``exact_plunges``) run beside the batches."""
        return self.use_exact_plunges and self.batch_size > 1

    def exact_plunge(self, force=False):
        """
        Exact plunges beside the batched search (``exact_plunges``): the
        plunge child (or, with none, the heap's top, starting a new plunge)
        is solved alone to convergence (``solve_node_lp``,
        ``node_max_epochs``), crossed over to a vertex (``cross_over_node``)
        and evaluated as any node, so an integral vertex is an incumbent
        (``"T"``) and the plunge goes on into its preferred child, the other
        one queued: depth-first search with the tree's own backtracking,
        propagation and conflicts, as HiGHS's dives. The batches' truncated
        LPs (``batch_epochs``) raise the bound but reach no integral leaf
        (ns1830653: at least 122 of 458 integers fractional at every node to
        depth 26), and LP dives with fixed chunks die there (every free
        integer fractional even at the vertex). Runs while the plunges have
        taken at most ``exact_plunge_share`` of the tree's time; with
        ``force`` (nothing else can go on), at least one node whatever.
        """
        started = time.time()
        tree = self.tree_elapsed()
        if (
            not force
            and tree > _TREE_CUT_GRACE_SECONDS
            and self.exact_plunge_seconds > self.exact_plunge_share * tree
        ):
            return
        budget = started + max(
            0.05, self.exact_plunge_share * (tree + 1.0) - self.exact_plunge_seconds
        )
        self.exact_plunge_active = True
        first = force
        try:
            while not self.out_of_time() and (first or time.time() < budget):
                first = False
                if self.plunge_node is not None:
                    node, self.plunge_node = self.plunge_node, None
                elif self.open:
                    _, _, node = heapq.heappop(self.open)
                    self.plunge_left = self.plunge_depth
                    self.exact_plunge_stats["plunges"] += 1
                else:
                    break
                if node.bound >= self.cutoff():
                    self.close(node.bound)
                    continue
                lower, upper = self.expand(node.changes)
                if np.any(lower > upper):
                    self.close(max(node.bound, self.cutoff_bound()))
                    continue
                box = self.prune_by_conflicts(node, lower, upper)
                if box is None:
                    continue
                lower, upper = box
                result = self.solve_node_lp(
                    lower, upper, self.fit_warm(node), self.node_max_epochs
                )
                if result["stop_reason"] in ("interrupted", "time_limit"):
                    self.push(node)
                    break
                self.nodes += 1
                self.exact_plunge_stats["nodes"] += 1
                self.exact_plunge_stats["max depth"] = max(
                    self.exact_plunge_stats["max depth"], node.depth
                )
                self.evaluate(node, lower, upper, None, result)
                self.log(" ", None)
        finally:
            self.exact_plunge_active = False
            self.exact_plunge_seconds += time.time() - started

    def pipeline_nodes(self):
        """The nodes the pipeline holds outside the queue
        (``pipeline_batches``): those dispatched and not yet evaluated, and
        those evaluated whose children are still on the device. The global
        bound counts them."""
        nodes = []
        if self.in_flight is not None:
            nodes += self.in_flight["members"]
        if self.pending_children is not None:
            nodes += [item["node"] for item in self.pending_children["items"]]
        return nodes

    def drain_pipeline(self):
        """Empty the pipeline when the tree loop stops: the children on the
        device are queued, and the dispatched batch's nodes go back to the
        queue unevaluated (their LPs are dropped)."""
        pending, self.pending_children = self.pending_children, None
        batch, self.in_flight = self.in_flight, None
        if pending is not None:
            self.finish_branches(pending)
        if batch is not None:
            for node in batch["members"]:
                self.push(node)

    def exact_eligible(self, node):
        """Whether the exact lane may take ``node``: not taken before, below
        the root, with a warm start (whose shapes its exact solution
        takes), once the heap is in bound order (not the dive before the
        first incumbent)."""
        return (
            self.exact_lane is not None
            and not node.exact_tried
            and node.depth > 0
            and node.has_warm_solution()
            and not self.diving()
            and not self.out_of_time()
        )

    def exact_band_top(self):
        """The top of the band the lane keeps (``exact_band``): the global
        bound plus ``exact_band`` of the gap to the cutoff; ``-inf`` (no
        band) without the lane, a band or a cutoff. With ``"auto"``, no
        higher than the lane's capacity (``band_capacity_top``)."""
        if self.exact_lane is None or self.exact_band <= 0.0:
            return -np.inf
        cutoff = self.cutoff()
        bound = self.band_bound = self.global_bound()
        if not (np.isfinite(cutoff) and np.isfinite(bound)):
            return -np.inf
        top = bound + self.exact_band * max(cutoff - bound, 0.0)
        if self.exact_band_auto:
            cap = self.band_capacity_top()
            self.band_capped = cap < top
            self.band_capped_batches += self.band_capped
            top = min(top, cap)
        return top

    def band_cap_size(self):
        """The nodes the lane's queue keeps (``band_capacity_top``): each
        thread's share is the nodes it clears in
        ``_EXACT_BAND_HORIZON_BATCHES`` batches (the mean seconds of a lane
        job against the mean time between batch launches), within
        ``_EXACT_BAND_CAP_PER_THREAD``; ``_EXACT_OVERFLOW_KEEP`` a thread
        before both are known."""
        threads = self.exact_lane.threads
        if not self.exact_jobs or not self.batch_period or self.exact_seconds <= 0.0:
            return _EXACT_OVERFLOW_KEEP * threads
        per_job = self.exact_seconds / self.exact_jobs
        share = _EXACT_BAND_HORIZON_BATCHES * self.batch_period / per_job
        low, high = _EXACT_BAND_CAP_PER_THREAD
        return threads * int(min(high, max(low, round(share))))

    def band_capacity_top(self):
        """The bound of the ``band_cap_size``-th lowest node the lane could
        take (reserved, or open and not yet tried): a band no higher holds
        only what the threads reach within a couple of batches, and the
        device takes the rest at once instead of leaving it to wait.
        ``inf`` when fewer nodes are open."""
        k = self.band_cap_size()
        self.band_cap_nodes = k
        bounds = [entry[0] for entry in self.exact_reserved]
        bounds += [
            entry[2].bound
            for entry in self.open
            if not entry[2].exact_tried and entry[2].depth > 0
        ]
        if len(bounds) < k:
            return np.inf
        return heapq.nsmallest(k, bounds)[-1]

    def adapt_exact_band(self):
        """With ``exact_band="auto"``, once a batch: the band is
        ``band_scale`` over the relative gap (``gap``), within
        [``_EXACT_BAND_MIN``, ``_EXACT_BAND_MAX``], so its share of the gap
        grows as the gap closes: at the starting scale (0.25) a quarter of
        the gap at a 100% gap, half at 50%, all of it from 25% down. The
        scale itself moves with the lift each side gets near the band's top
        (``tune_band_scale``), and the band's top stops at what the threads
        can reach (``band_capacity_top``)."""
        if not self.exact_band_auto or self.exact_lane is None:
            return
        if self.incumbent is None:
            return  # no band before an incumbent
        self.tune_band_scale()
        gap = self.gap(self.global_bound())
        band = _EXACT_BAND_MAX if gap <= 0.0 else self.band_scale / gap
        self.exact_band = min(_EXACT_BAND_MAX, max(_EXACT_BAND_MIN, band))
        low, high = self.exact_band_range
        self.exact_band_range = [min(low, self.exact_band), max(high, self.exact_band)]

    def record_band_edge(self, node, side, lift, seconds):
        """Count a node that ``side`` (``"exact"`` or ``"device"``) solved
        near the auto band's top for ``tune_band_scale``: the bound its LP
        gained (``lift``, at most to the cutoff) in ``seconds`` of that
        side's time. Only a node tagged for that side when it was taken
        (``node.band_edge``) counts: one the lane gave back, or the band
        overflowed to the device, does not. The tag is cleared."""
        tagged, node.band_edge = node.band_edge, None
        if tagged != side or not np.isfinite(lift) or seconds <= 0.0:
            return
        sums = self.band_edge_sums[side]
        sums[0] += max(lift, 0.0)
        sums[1] += seconds
        sums[2] += 1

    def tune_band_scale(self):
        """Move the auto band's scale (``band_scale``) once each side of the
        band's top has solved ``_EXACT_BAND_TUNE_SAMPLES`` nodes since the
        last move: up by ``_EXACT_BAND_SCALE_STEP`` when the lane's exact
        LPs near the top lifted more bound a thread-second than the
        device's LPs just above it did a device-second, else down, within
        ``_EXACT_BAND_SCALE_LIMITS``. The rates are compared at the edge,
        where the choice is made: at the bottom of the band exact LPs
        always win. While the cap (``band_capacity_top``) holds the band
        down, the scale does not set it, and stays."""
        exact, device = self.band_edge_sums["exact"], self.band_edge_sums["device"]
        if min(exact[2], device[2]) < _EXACT_BAND_TUNE_SAMPLES:
            return
        exact_rate = exact[0] / exact[1]
        device_rate = device[0] / device[1]
        self.band_rates = (exact_rate, device_rate)
        self.band_edge_sums = {"exact": [0.0, 0.0, 0], "device": [0.0, 0.0, 0]}
        if self.band_capped:
            return
        step = _EXACT_BAND_SCALE_STEP if exact_rate > device_rate else 1.0 / _EXACT_BAND_SCALE_STEP
        low, high = _EXACT_BAND_SCALE_LIMITS
        self.band_scale = min(high, max(low, self.band_scale * step))
        self.band_tunes += 1
        lo, hi = self.band_scale_range
        self.band_scale_range = [min(lo, self.band_scale), max(hi, self.band_scale)]

    @staticmethod
    def basis_shape(mip):
        """The columns and rows a HiGHS basis for ``mip``'s LP has."""
        return mip.num_variables, mip.A.shape[0]

    def store_basis(self, basis, mip):
        """Keep ``basis`` (for ``mip``'s rows) and return its key, or
        ``None`` when ``exact_basis_memory`` has no room for one."""
        if basis is None or not basis.valid:
            return None
        shape = self.basis_shape(mip)
        capacity = int(self.exact_basis_memory // max(1, sum(shape)))
        if capacity < 1:
            return None
        key = next(self.basis_keys)
        self.basis_store[key] = [shape, basis, 1]  # held by its node
        while len(self.basis_store) > capacity:
            self.basis_store.popitem(last=False)
        return key

    def hold_basis(self, key):
        """Count one more node starting from basis ``key``."""
        entry = self.basis_store.get(key)
        if entry is not None:
            entry[2] += 1

    def drop_basis(self, key):
        """One node fewer holds basis ``key``; it goes once none does.
        Returns its entry (``None``: evicted)."""
        entry = self.basis_store.get(key)
        if entry is not None:
            entry[2] -= 1
            if entry[2] <= 0:
                del self.basis_store[key]
        return entry

    def take_basis(self, key):
        """The basis ``key`` for a node's exact LP, or ``None`` (evicted, or
        of another shape: a restart). A tree cut refresh keeps the cut
        block's shape, so a basis from before it still fits; HiGHS takes
        any basis of the right shape as a start and solves on from it."""
        entry = self.drop_basis(key)
        if entry is None or entry[0] != self.basis_shape(self.mip):
            return None
        return entry[1]

    def release_bases(self, node):
        """``node`` lets go of its bases (its own, its strong sides', or the
        one it would have started from), once its children hold theirs.
        Nodes closed without branching keep theirs until the store is full."""
        for key in [node.exact_basis, *(node.exact_sides or {}).values()]:
            self.drop_basis(key)
        node.exact_basis = node.exact_sides = None

    def feed_exact(self):
        """Hand the lane the lowest reserved nodes while it has room."""
        lane = self.exact_lane
        busy, self._lane_busy = self._lane_busy, True
        try:
            while lane is not None and self.exact_reserved and lane.room():
                _, _, node = heapq.heappop(self.exact_reserved)
                if node.bound >= self.cutoff():
                    self.close(node.bound)
                    continue
                box = self.expand(node.changes)
                if np.any(box[0] > box[1]):
                    self.close(max(node.bound, self.cutoff_bound()))
                    continue
                box = self.prune_by_conflicts(node, *box)
                if box is None:
                    continue
                if node.exact_known:
                    self.evaluate_known(node, box)
                    continue
                self.submit_exact(node, box)
        finally:
            self._lane_busy = busy

    def wants_exact(self, node):
        """Without a band (``exact_band=0``): whether the lane takes
        ``node``, popped from the top of the heap, now. Once the heap is in
        bound order its top holds the global bound, so the lane takes the
        lowest-bound nodes whenever it has room."""
        return self.exact_eligible(node) and self.exact_lane.room()

    def overflow_band(self, limit):
        """Up to ``limit`` of the highest-bound nodes reserved for the exact
        lane past the ``_EXACT_OVERFLOW_KEEP`` a thread it keeps, taken out
        of the band for the device to solve to convergence (``converge``),
        lowest bound first. The lane keeps the lowest nodes."""
        lane, reserved = self.exact_lane, self.exact_reserved
        if lane is None or limit <= 0:
            return []
        excess = len(reserved) - _EXACT_OVERFLOW_KEEP * lane.threads
        if excess <= 0:
            return []
        take = min(limit, excess)
        reserved.sort()  # a sorted list is still a heap
        nodes = [entry[2] for entry in reserved[-take:]]
        del reserved[-take:]
        for node in nodes:
            node.exact_tried = node.converge = True
            node.band_edge = None
        self.overflow_nodes += len(nodes)
        return nodes

    def strong_iteration_cap(self):
        """The pivots a lane strong-branching side gets
        (``exact_strong_iterations``). ``"auto"``: twice the mean pivots of
        the lane's node LPs so far, within ``_EXACT_STRONG_AUTO`` (as SCIP
        limits its strong branching LPs), its upper end before any. A side
        stopped there keeps its duals' bound, lower than its LP value."""
        if self.exact_strong_auto and self.exact_jobs:
            low, high = _EXACT_STRONG_AUTO
            mean = self.exact_pivots / self.exact_jobs
            self.exact_strong_iterations = int(min(high, max(low, 2.0 * mean)))
        return self.exact_strong_iterations

    def submit_exact(self, node, box):
        node.exact_tried = True
        lower, upper = box
        strong = None
        if self.exact_strong_candidates > 0:
            # Strong branching in the job (``_exact_strong``), against a
            # snapshot of the lane's exact pseudocosts.
            strong = (
                self.is_int,
                self.int_tol,
                self.exact_pc_sum.copy(),
                self.exact_pc_count.copy(),
                self.exact_strong_candidates,
                self.strong_iteration_cap(),
                self.reliability_threshold,
            )
        basis = None
        if node.exact_basis is not None:
            basis = self.take_basis(node.exact_basis)
            node.exact_basis = None
        self.exact_lane.submit(
            node,
            box,
            (self.lp_lower(lower), self.lp_upper(upper)),
            self.mip,
            self.cut_version,
            strong,
            basis=basis,
            keep=self.exact_basis_memory > 0,
            bound_terms=(self.A_T, self.is_int, self.reduced_cost_tol),
            strong_options=dict(
                lookahead=self.exact_strong_lookahead,
                # A side whose dual objective passes the cutoff is closed:
                # it stops there (``node_bound_target``, as node LPs do).
                objective_bound=(
                    self.node_bound_target() if self.exact_strong_cutoff else None
                ),
            ),
        )

    def collect_exact(self):
        """Evaluate the nodes the exact lane has solved, as batch members
        are evaluated: the HiGHS solution stands in for the PDHG one, so
        the node's bound is its exact duals' Lagrangian bound (rigorous at
        any duals), it branches on the vertex, and its children's bounds
        come from those duals over their boxes. With the lane's strong
        branching (``exact_strong_candidates``), the node branches by its
        strong children (``exact_strong_children``). A node HiGHS proves
        infeasible is closed; one it did not finish goes back to the queue
        for the device."""
        lane = self.exact_lane
        if lane is None:
            return
        busy, self._lane_busy = self._lane_busy, True
        try:
            self._collect_exact(lane)
        finally:
            self._lane_busy = busy

    def _collect_exact(self, lane):
        done = lane.finished()
        # Until its turn, each finished node is in no list but this one: the
        # global bound and ``explored`` (logged while the others are
        # evaluated) count it here. Each node's own evaluation holds its
        # bound (``active_bound``) until its children are queued.
        self.exact_unevaluated = [entry[0] for entry in done]
        try:
            for entry in done:
                self.exact_unevaluated.pop(0)
                self.collect_one(*entry)
        finally:
            self.exact_unevaluated = []

    def collect_one(
        self, node, box, version, outcome, children, strong_seconds, seconds, extra
    ):
        """``collect_exact`` for one finished lane job."""
        self.exact_seconds += seconds
        self.exact_strong_seconds += strong_seconds
        self.exact_strong_lps += len(children)
        self.exact_pivots += extra["pivots"]
        self.exact_jobs += 1
        self.exact_warm_bases += bool(extra["warm"])
        status, x, y_eq, y_ineq = outcome
        if node.bound >= self.cutoff():
            self.close(node.bound)
            return
        lp_seconds = seconds - strong_seconds
        if status == "infeasible":
            self.exact_infeasible += 1
            self.record_band_edge(node, "exact", self.cutoff() - node.bound, lp_seconds)
            self.close(max(node.bound, self.cutoff_bound()))
            return
        solution = None
        if status == "optimal" and version <= self.cut_version:
            solution = self.exact_solution(node, x, y_eq, y_ineq, version)
        if solution is None:
            # Unsolved, or for rows the search no longer has (a restart).
            self.exact_failed += 1
            node.band_edge = None
            self.push(node)
            return
        self.exact_solved += 1
        if extra["mip"] is not None:
            # The children's start (``exact_basis_memory``).
            node.exact_basis = self.store_basis(extra["basis"], extra["mip"])
            node.exact_sides = {
                side: key
                for side, basis in extra["sides"].items()
                if (key := self.store_basis(basis, extra["mip"])) is not None
            }
        # The bounds the lane's thread computed (``exact_lane_bounds``) are
        # this thread's own when the job solved the rows searched now: the
        # same function on the same arrays. Otherwise they are recomputed.
        known = None
        if extra.get("mip") is self.mip and version == self.cut_version:
            known = extra.get("bounds")
            self.exact_bounds_reused += known is not None
        strong = self.exact_strong_children(
            node, box, version, solution, children, known
        )
        self.evaluate_exact(node, box, solution, strong, known, lp_seconds)

    def exact_solution(self, node, x, y_eq, y_ineq, version):
        """A HiGHS solution (Jaddle's dual convention) as a ``SaddleState``
        shaped as ``node``'s warm start, with the duals of cut slots
        replaced since ``version`` zeroed (``stale_free``); ``None`` when
        the shapes do not fit the current rows."""
        template = node.warm[0]
        if (
            y_ineq.size != np.shape(template.dual_ineq)[0]
            or y_eq.size > np.shape(template.dual_eq)[0]
        ):
            return None
        dual_eq = np.zeros(np.shape(template.dual_eq)[0])
        dual_eq[: y_eq.size] = y_eq
        solution = template._replace(primal=x, dual_eq=dual_eq, dual_ineq=y_ineq)
        if version < self.cut_version:
            solution = self.stale_free(solution, version)
        return solution

    def known_bound(self, known, key, solution, lower, upper):
        """``exact_bound`` of ``solution`` over the box, from ``known`` (the
        lane's ``exact_lane_bounds``) when it holds ``key``."""
        if known is not None and known.get(key) is not None:
            return known[key]
        return self.exact_bound(solution, lower, upper)

    def exact_strong_children(self, node, box, version, solution, children, known=None):
        """The lane's strong-branching ``children`` (``_exact_strong``'s
        list) as ``choose_branch``'s ``exact_strong``: ``({var: (down,
        up)}, solved)`` of ``_StrongChild``, bounded by each side's duals
        over its box (rigorous at any duals, so an iteration-limited side
        counts too), or ``None`` without children. Each solved side also
        records an exact pseudocost observation."""
        if not children:
            return None
        lower, upper = box
        x = np.asarray(solution.primal, np.float64)
        node_raw, _ = self.known_bound(known, "node", solution, lower, upper)
        sides = {}
        for var, down, status, cx, cy_eq, cy_ineq in children:
            lo, up = lower.copy(), upper.copy()
            if down:
                up[var] = min(up[var], math.floor(x[var]))
            else:
                lo[var] = max(lo[var], math.ceil(x[var]))
            if status == "infeasible" or lo[var] > up[var]:
                sides.setdefault(var, {})[down] = _StrongChild(np.inf, True, None)
                continue
            child = None
            if status in ("optimal", "partial"):
                child = self.exact_solution(node, cx, cy_eq, cy_ineq, version)
            if child is None:  # failed: no information
                sides.setdefault(var, {})[down] = None
                continue
            raw, rigorous = self.known_bound(known, (var, down), child, lo, up)
            if not (rigorous and np.isfinite(raw)):
                raw = node.bound
            bound = max(self.round_bound(raw), node.bound)
            closed = rigorous and bound >= self.cutoff()
            sides.setdefault(var, {})[down] = _StrongChild(
                bound,
                closed,
                None if closed else child,
                raw,
                status == "optimal" and not closed,
            )
            distance = x[var] - math.floor(x[var]) if down else math.ceil(x[var]) - x[var]
            if distance > 0 and np.isfinite(node_raw) and np.isfinite(raw):
                side = 0 if down else 1
                self.exact_pc_sum[side, var] += max(raw - node_raw, 0.0) / distance
                self.exact_pc_count[side, var] += 1
        out, solved = {}, set()
        for var, both in sides.items():
            if both.get(True) is not None and both.get(False) is not None:
                out[var] = (both[True], both[False])
                solved.add(var)
        return (out, solved) if out else None

    def root_exact_strong(self, root, lower, upper, result):
        """
        Exact strong branching at the root (``exact_root_candidates``),
        over all the lane's threads at once: HiGHS solves the root LP over
        ``[lower, upper]`` exactly, the fractional integers at its vertex
        with the largest ``impact · f(1 − f)`` (as ``root_sweep`` ranks
        them) are dealt out to the threads, and each side is re-solved from
        the root's basis (``_ExactLane.split``), within
        ``exact_root_seconds`` in all (and a tenth of the time left). The
        root then branches as a lane node does: its vertex and exact duals
        in place of the PDHG point, its children by ``apply_strong`` (empty
        sides fix their variable, the bound rises to the best lesser
        child, solved sides are their children's LPs), the sides' gains
        seeding the exact pseudocosts, and the bases kept for the children.
        Returns the result for ``evaluate``, or ``None`` (no lane, off, no
        time, or no optimal root vertex) to branch on the PDHG point.
        """
        lane = self.exact_lane
        k = self.exact_root_candidates
        if lane is None or k == 0 or self.out_of_time():
            return None
        started = time.time()
        budget = self.exact_root_seconds
        left = self.seconds_left()
        if left is not None:
            budget = min(budget, 0.1 * left)
        if budget <= 0:
            return None
        deadline = started + budget
        mip = self.mip
        lp_box = (self.lp_lower(lower), self.lp_upper(upper))
        (status, x, y_eq, y_ineq), basis, pivots = lane.root_lp(mip, lp_box, budget)
        stats = self.root_exact_stats
        stats["pivots"] += pivots
        if status != "optimal" or basis is None:
            stats["seconds"] += time.time() - started
            return None
        root.warm = (result["solution"], result["opt_state"])
        solution = self.exact_solution(root, x, y_eq, y_ineq, self.cut_version)
        if solution is None:
            stats["seconds"] += time.time() - started
            return None
        frac = x - np.floor(x)
        cand = np.flatnonzero(
            self.is_int
            & (frac > self.int_tol)
            & (frac < 1.0 - self.int_tol)
            & (lower < upper)
        )
        f = frac[cand]
        cand = cand[np.argsort(-(self.impact[cand] * f * (1.0 - f)), kind="stable")]
        if k is not None:
            cand = cand[:k]
        children, sides = [], {}
        if cand.size:
            chunks = [cand[i :: lane.threads] for i in range(lane.threads)]
            zeros = np.zeros((2, mip.num_variables))
            request = (
                self.is_int, self.int_tol, zeros, zeros, int(cand.size),
                self.exact_root_iterations, 1,
            )
            children, sides, _ = lane.split(
                mip, lp_box, x, basis, chunks, request, deadline
            )
        seen_sum, seen_count = self.exact_pc_sum.copy(), self.exact_pc_count.copy()
        strong = self.exact_strong_children(
            root, (lower, upper), self.cut_version, solution, children
        )
        # The root sweep's job too: seed the device's pseudocosts, which
        # its nodes branch by, with the sides' exact gains.
        self.pc_sum += self.exact_pc_sum - seen_sum
        self.pc_count += self.exact_pc_count - seen_count
        root.exact_basis = self.store_basis(basis, mip)
        root.exact_sides = {
            side: key
            for side, side_basis in sides.items()
            if (key := self.store_basis(side_basis, mip)) is not None
        }
        root.exact_tried = True
        exact_bound, _ = self.exact_bound(solution, lower, upper)
        stats["candidates"] += int(cand.size)
        stats["lps"] += len(children)
        stats["fixed"] += sum(child[2] == "infeasible" for child in children)
        stats["seconds"] += time.time() - started
        stats["bound"] = (
            self.model_value(exact_bound) if np.isfinite(exact_bound) else None
        )
        return dict(
            solution=solution,
            converged=True,
            stop_reason="exact",
            epochs=0,
            opt_state=result["opt_state"],
            exact_strong=strong,
        )

    def evaluate_exact(
        self, node, box, solution, strong=None, known=None, lp_seconds=None
    ):
        """Evaluate ``node`` with an exact LP ``solution`` (and the lane's
        strong children, ``strong``) as a batch member is evaluated;
        ``known``: the lane's bounds (``exact_lane_bounds``), if usable;
        ``lp_seconds``: its LP's time on the lane (``record_band_edge``)."""
        lower, upper = box
        raw, _ = self.known_bound(known, "node", solution, lower, upper)
        if np.isfinite(raw):
            self.exact_gain += max(raw - node.bound, 0.0)
        if lp_seconds is not None:
            self.record_band_edge(
                node, "exact", min(raw, self.cutoff()) - node.bound, lp_seconds
            )
        node.band_edge = None
        self.nodes += 1
        result = dict(
            solution=solution,
            converged=True,
            stop_reason="exact",
            epochs=0,
            opt_state=node.warm[1],
            exact_strong=strong,
        )
        self.evaluate(node, lower, upper, None, result)
        self.log(" ", None)

    def evaluate_known(self, node, box):
        """Evaluate a node whose warm start is already an optimal vertex of
        its LP (``exact_known``: from the lane's strong branching) without
        solving it again. Its box lies inside the strong child's, so the
        duals' Lagrangian bound over it holds and is no lower."""
        node.exact_tried = True
        self.exact_reused += 1
        self.evaluate_exact(node, box, self.fit_warm(node)[0])

    def launch_next(self):
        """Pick up to ``batch_size`` open nodes and dispatch their LPs
        (``launch_batch``). Returns the batch (members, boxes, the device
        handle and the cut block version it was solved with), or ``None``
        with no node to solve."""
        members, boxes = [], []
        in_flight = 0 if self.in_flight is None else len(self.in_flight["members"])
        now = time.perf_counter()
        if self.last_launch is not None:
            period = now - self.last_launch
            self.batch_period = (
                period
                if self.batch_period is None
                else 0.9 * self.batch_period + 0.1 * period
            )
        self.last_launch = now
        self.feed_exact()
        self.adapt_exact_band()
        band_top = self.exact_band_top()
        # The auto band's edges (``tune_band_scale``): its top half for the
        # lane, and as far again above it for the device.
        edge_low = edge_high = band_top
        if self.exact_band_auto and np.isfinite(band_top):
            depth = max(band_top - self.band_bound, 0.0)
            edge_low = band_top - _EXACT_BAND_EDGE * depth
            edge_high = band_top + _EXACT_BAND_EDGE * depth
        if self.exact_overflow_share > 0.0:
            # The top of a band too deep for the lane, solved to convergence
            # on the device (``overflow_band``).
            for node in self.overflow_band(
                int(self.exact_overflow_share * self.batch_size)
            ):
                if node.bound >= self.cutoff():
                    self.close(node.bound)
                    continue
                box = self.expand(node.changes)
                if np.any(box[0] > box[1]):
                    self.close(max(node.bound, self.cutoff_bound()))
                    continue
                box = self.prune_by_conflicts(node, *box)
                if box is None:
                    continue
                members.append(node)
                boxes.append(box)
        plunge_child = self.plunge_node is not None and not self.exact_plunges_on()
        while len(members) < self.batch_size and (self.open or plunge_child):
            plunge_child = False
            if (
                self.max_nodes is not None
                and self.nodes + in_flight + len(members) >= self.max_nodes
            ):
                break
            from_heap = self.plunge_node is None or self.exact_plunges_on()
            node = self.next_node()
            if node.bound >= self.cutoff():
                self.close(node.bound)
                continue
            eligible = from_heap and self.exact_eligible(node)
            if eligible and node.bound <= band_top:
                # In the band the lane keeps (``exact_band``): it waits for
                # an exact LP instead of a truncated one.
                if node.bound >= edge_low and edge_low < band_top:
                    node.band_edge = "exact"
                heapq.heappush(
                    self.exact_reserved, (node.bound, next(self.counter), node)
                )
                continue
            if eligible and node.bound <= edge_high and band_top < edge_high:
                node.band_edge = "device"
            box = self.expand(node.changes)
            if np.any(box[0] > box[1]):
                # Emptied by reduced-cost fixing of the root box.
                self.close(max(node.bound, self.cutoff_bound()))
                continue
            box = self.prune_by_conflicts(node, *box)
            if box is None:
                continue
            if node.exact_known and self.exact_lane is not None:
                # Solved already by the lane's strong branching.
                self.evaluate_known(node, box)
                continue
            if from_heap and self.exact_band <= 0.0 and self.wants_exact(node):
                self.submit_exact(node, box)
                continue
            members.append(node)
            boxes.append(box)
        if not members:
            return None
        warms = [self.fit_warm(node) for node in members]
        handle = self.launch_batch(
            boxes,
            [warm[0] for warm in warms],
            [warm[1] for warm in warms],
            min(self.batch_epochs, self.node_max_epochs),
            self.batch_size,
        )
        self.batches += 1
        return dict(
            members=members,
            boxes=boxes,
            handle=handle,
            version=self.cut_version,
            launched=now,
        )

    def process_batch(self, batch):
        """Read back a dispatched batch (``launch_next``) and evaluate its
        members in order on the host, as the serial search does. Members to
        split on the device are left in ``deferred_branches``. When due, the
        GPU dives (``gpu_dives``) then start from the members that
        branched."""
        self.dive_starts = [] if self.gpu_dive_due() else None
        try:
            self._process_batch(batch)
        finally:
            starts, self.dive_starts = self.dive_starts, None
        if starts:
            self.gpu_dives(starts)

    def _process_batch(self, batch):
        members, boxes = batch["members"], batch["boxes"]
        self.batch_waiting = members
        # The members wait in ``batch_waiting`` and none is being evaluated:
        # the exact lane is serviced while the device finishes their LPs.
        results = self.read_batch(batch["handle"], service=True)
        if batch["version"] < self.cut_version:
            # Solved before a tree cut refresh changed slots of the cut
            # block: those slots' duals are zeroed (a zero multiplier is
            # valid), so the bounds hold for the rows as they are now.
            results = [
                dict(result, solution=self.stale_free(result["solution"], batch["version"]))
                for result in results
            ]
        evaluating = time.perf_counter()
        # A member's share of the device's time, for ``record_band_edge``.
        member_seconds = (evaluating - batch.get("launched", evaluating)) / max(
            len(members), 1
        )
        results, requeued = self.after_node_solves(members, boxes, results)
        self.deferring = self.batch_on_device()
        try:
            for i, (node, result) in enumerate(zip(members, results)):
                self.batch_waiting = members[i + 1 :]
                self.lp_epochs += result["epochs"]
                if i in requeued:
                    # Back in the queue for new tree cuts: not a node yet.
                    continue
                self.nodes += 1
                self.bound_stops += result["stop_reason"] == "bound_target"
                # An incumbent found by an earlier member may prune this one.
                if node.bound >= self.cutoff():
                    self.close(node.bound)
                    continue
                lower, upper = boxes[i]
                requeues = self.batch_requeues
                self.evaluate(node, lower, upper, None, result)
                if node.band_edge is not None:
                    lift = min(self.last_raw_bound, self.cutoff()) - node.bound
                    self.record_band_edge(node, "device", lift, member_seconds)
                if self.batch_requeues > requeues:
                    # Back in the queue unconverged: a slice of its LP, not a
                    # node. It counts when it finishes.
                    self.nodes -= 1
                    continue
                self.log(" ", None)
        finally:
            self.deferring = False
            self.batch_waiting = []
            self.batch_evaluate_seconds += time.perf_counter() - evaluating
        self.flush_roundings()

    def auto_batch_size(self, mip, children_backend):
        """``batch_size="auto"``: on a CPU, ``_AUTO_BATCH_CPU`` (it runs the
        members one after another); on a GPU or TPU, as many nodes as fit
        in its free memory, at most ``_AUTO_BATCH_ACCELERATOR``
        (``auto_batch_size``), or that many when the device reports no
        memory. The node LP counted is ``mip``'s with the tree cut block
        ``reserve_tree_cuts`` will give it (before root cuts), at the node
        LPs' precision; the children kernels count when ``batch_on_device``
        would run them."""
        device = self.device or jax.devices()[0]
        if device.platform == "cpu":
            return _AUTO_BATCH_CPU
        try:
            stats = device.memory_stats() or {}
        except Exception:
            stats = {}
        if "bytes_limit" not in stats:
            return _AUTO_BATCH_ACCELERATOR
        free = stats["bytes_limit"] - stats.get("bytes_in_use", 0)
        n, m, nnz = mip.num_variables, mip.A.shape[0], mip.A.nnz
        cut_rows = cut_nnz = 0
        if self.use_tree_cuts:
            if self.tree_cut_rows is None:
                cut_rows = max(_TREE_CUT_MIN_ROWS, 2 * self.max_tree_cuts_per_round)
            else:
                cut_rows = self.tree_cut_rows
            cut_nnz = _next_power_of_two(
                self.tree_cut_nnz or max(256, cut_rows * min(n, 16))
            )
        kernel_nnz = 1 << max(8, math.ceil(math.log2(max(nnz + cut_nnz, 1))))
        if self.node_precision is None:
            itemsize = np.dtype(mip.lp.c.dtype).itemsize
        else:
            with self.node_settings():
                itemsize = np.dtype(_wide(jo.jaddle_dtype())).itemsize
        width = max(n, nnz)

        def children(batch):
            # ``batch_on_device``'s rule, at the MIP's own nonzeros (the
            # cuts' come later, and more only turn the kernels off).
            if children_backend == "numpy" or batch < 2:
                return False
            if children_backend == "jax":
                return True
            return nnz > _SMALL_MIP_NNZ and 2 * batch * width <= 2**25

        return auto_batch_size(
            n, m + cut_rows, nnz + cut_nnz, free, itemsize, kernel_nnz, children
        )

    def batch_on_device(self):
        """Whether batched search splits and rounds its members on the
        device (``children_backend``)."""
        if self.children_backend == "numpy" or self.batch_size < 2:
            return False
        if self.children_backend == "jax":
            return True
        width = max(self.mip.num_variables, self.mip.A.nnz)
        return self.mip.A.nnz > _SMALL_MIP_NNZ and 2 * self.batch_size * width <= 2**25

    def kernel_nnz_capacity(self):
        """The nonzeros the device kernels (``_ChildKernels``,
        ``_StrongKernels``) are padded to: enough for the searched rows
        with a full tree cut block (``tree_nnz_cap``), to a power of two.
        Tree cut refreshes change the block's nonzeros; unpadded, each one
        recompiled the kernels (on ns1830653, 14 times a minute)."""
        nnz = self.mip.A.nnz
        if self.use_tree_cuts and self.slot_version:
            nnz = max(nnz, self.base.A.nnz + self.tree_nnz_cap)
        return 1 << max(8, math.ceil(math.log2(max(nnz, 1))))

    def child_kernels(self):
        held = self._child_kernels
        if held is not None and held[0] is not self.mip and held[1].set_mip(self.mip):
            self._child_kernels = (self.mip, held[1])  # same shapes: kept
        if self._child_kernels is None or self._child_kernels[0] is not self.mip:
            kernels = _ChildKernels(
                self.mip,
                self.propagator is not None,
                self.reduced_cost_tol,
                self.batch_size,
                nnz_capacity=self.kernel_nnz_capacity(),
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
        pending = self.launch_branches()
        if pending is not None:
            self.finish_branches(pending)

    def launch_branches(self):
        """Dispatch the children of the batch's deferred nodes
        (``deferred_branches``) to the device without waiting. Returns the
        handle for ``finish_branches`` (``None`` with nothing deferred)."""
        pending, self.deferred_branches = self.deferred_branches, []
        if not pending:
            return None
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
                lower[0],
                upper[0],
                var[0],
                xv[0],
                r[0],
                constant[0],
            )
        if self.pipeline_batches:
            # After the next batch's LPs, on the device thread (the
            # propagation is a device while loop too).
            out = self.on_device_thread(kernels.launch, lower, upper, var, xv, r, constant)
        else:
            out = kernels.launch(lower, upper, var, xv, r, constant)
        return {"items": pending, "out": out}

    def finish_branches(self, pending):
        """Read back a ``launch_branches`` and ``branch`` each node on the
        host (conflict pool, pruning, the queue)."""
        out = pending["out"]
        started = time.perf_counter()
        if hasattr(out, "result"):  # dispatched on the device thread
            out = out.result()
        with jax.enable_x64(True):
            out = jax.device_get(out)
        self.batch_wait_seconds += time.perf_counter() - started
        self.batched_children += 1
        started = time.perf_counter()
        for i, item in enumerate(pending["items"]):
            row = {key: value[i] for key, value in out.items()}
            self.branch(**item, precomputed=row)
        self.branch_seconds += time.perf_counter() - started

    def break_symmetry(self, mip, lower, upper):
        """
        Symmetry breaking at the root (``symmetry``): the column symmetries
        of the searched MIP over the box (``symmetry_generators``), their
        orbitopes (``orbitopes``), and rows and fixings that keep one
        optimal solution of each orbit (``symmetry_rows``). The rows join
        the searched MIP (``self.base``) as coefficient tightening's do;
        incumbents are still checked against the MIP as given, and every
        feasible point keeps its objective value, so the search stays
        exact. Returns the box with the fixings, propagated, or ``None``
        when it is empty. With ``symmetry="auto"`` and no igraph, nothing
        happens.
        """
        started = time.time()
        try:
            generators = symmetry_generators(mip, lower, upper)
        except ImportError:
            if self.symmetry is True:
                raise
            return lower, upper
        matrices = orbitopes(generators, mip.num_variables)
        A_rows, b_rows, fixed_upper = symmetry_rows(mip, matrices, lower, upper)
        fixed = int(np.count_nonzero(fixed_upper < upper))
        if A_rows.shape[0]:
            self.base = mip.with_rows(A_rows, b_rows)
            self.set_mip(self.base)
        upper = fixed_upper
        self.symmetry_stats = {
            "generators": len(generators),
            "orbitopes": [tuple(int(v) for v in m.shape) for m in matrices],
            "rows": int(A_rows.shape[0]),
            "fixed": fixed,
            "seconds": time.time() - started,
        }
        if self.verbose and generators:
            shapes = ", ".join(f"{r}×{p}" for r, p in self.symmetry_stats["orbitopes"])
            print(
                f"Symmetry: {len(generators)} generators, {len(matrices)} orbitopes"
                + (f" ({shapes})" if shapes else "")
                + f", {A_rows.shape[0]} rows, {fixed} fixed, "
                f"{self.symmetry_stats['seconds']:.2f}s"
            )
        if self.propagator is not None and (fixed or A_rows.shape[0]):
            lower, upper, infeasible = self.propagator(lower, upper, max_passes=20)
            if infeasible:
                return None
        return lower, upper

    def run(self):
        """The search, on ``self.device`` when one is set. Background
        heuristics (``parallel_heuristics``) stop when it returns.

        Ctrl-C, as in ``jl.solve``, ends the search with the best point so
        far: the first press calls ``request_interrupt`` (the search winds
        down along its time-limit path, with status ``"interrupted"``); a
        second raises ``KeyboardInterrupt`` as usual. The handler is set
        only on the main thread, and only by the outermost search."""
        installed, previous = False, None
        if self.owns_interrupt and threading.current_thread() is threading.main_thread():

            def on_sigint(signum, frame):
                if self.interrupt.is_set():  # the second press: abort
                    signal.signal(signal.SIGINT, previous or signal.default_int_handler)
                    raise KeyboardInterrupt
                print("KeyboardInterrupt received. Returning current solution.")
                self.request_interrupt()

            try:
                previous = signal.signal(signal.SIGINT, on_sigint)
                installed = True
            except ValueError:  # not the main interpreter
                pass
        heartbeat = heartbeat_stop = None
        if self.verbose:
            heartbeat_stop = threading.Event()
            heartbeat = threading.Thread(
                target=self.log_heartbeat,
                args=(heartbeat_stop,),
                name="jaddle-log",
                daemon=True,
            )
            heartbeat.start()
        try:
            if self.device is None:
                return self._run()
            with jax.default_device(self.device):
                return self._run()
        finally:
            if heartbeat is not None:
                heartbeat_stop.set()
                heartbeat.join(timeout=5.0)
            if installed:
                signal.signal(signal.SIGINT, previous or signal.default_int_handler)
            if self.workers is not None:
                self.workers.shutdown()
            if self.exact_lane is not None:
                self.exact_lane.shutdown()
            if self.submip_lane is not None:
                self.submip_lane.shutdown()
            if self.probe_lane is not None:
                self.probe_lane.shutdown()
            if self.level_lane is not None:
                self.level_lane.shutdown()
            if self._cut_worker is not None:
                self._cut_worker.shutdown(wait=False, cancel_futures=True)
            if self._device_thread is not None:
                self._device_thread.shutdown(wait=True)
                self._device_thread = None

    def _run(self):
        """Rounds of the search: prepare the root box, solve the root, search
        the tree; a restart (``restarts``) starts a new round from the root
        box that reduced-cost fixing has left."""
        lower, upper = self.mip.lower.copy(), self.mip.upper.copy()
        if np.any(lower > upper):
            return self.result("infeasible")
        first = True
        while True:
            box = self._prepare(lower, upper, first)
            if isinstance(box, dict):
                return box
            first = False
            out = self._root(*box)
            if out is None:
                out = self._tree()
            if out != "restart":
                return out if isinstance(out, dict) else self.result(out)
            lower, upper = self.restart()
            if self.restart_presolve:
                out = self.presolved_round(lower, upper)
                if out is not None:
                    return out

    def _prepare(self, lower, upper, first):
        """The root box: propagated, with coefficient tightening over it,
        symmetry breaking (the first round only) and probing. Returns it, or
        a result when it is empty."""
        mip = self.mip
        if self.propagator is not None:
            lower, upper, infeasible = self.propagator(lower, upper, max_passes=20)
            if infeasible:
                return self.empty_root()
        if self.use_coefficient_tightening:
            box = (lower, upper)
            if self.propagator is None:
                box = self.row_propagator(lower, upper, max_passes=20)[:2]
            tightened, count = tighten_coefficients(mip, *box)
            if count:
                # Same integer-feasible points; search the tightened rows
                # (incumbents are still checked against the MIP as given).
                self.coefficients_tightened += count
                self.base = tightened
                self.set_mip(tightened)
                mip = tightened
                if self.propagator is not None:
                    lower, upper, infeasible = self.propagator(
                        lower, upper, max_passes=20
                    )
                    if infeasible:
                        return self.empty_root()
        if self.symmetry and first:
            box = self.break_symmetry(mip, lower, upper)
            if box is None:
                return self.empty_root()
            mip, lower, upper = self.mip, *box
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
                max_passes=self.probing_passes,
                max_implications=self.probing_max_implications,
            )
            for key, value in (
                ("probed", probed.probed),
                ("fixed", probed.fixed),
                ("tightened", probed.tightened),
                ("seconds", time.time() - started),
            ):
                self.probing_stats[key] += value
            if probed.infeasible:
                return self.empty_root()
            lower, upper = probed.lower, probed.upper
            self.probe_cache = probed.cache
        self.root_lower, self.root_upper = lower, upper
        with np.errstate(invalid="ignore"):
            top = float(
                np.sum(
                    np.where(
                        mip.c > 0,
                        mip.c * upper,
                        np.where(mip.c < 0, mip.c * lower, 0.0),
                    )
                )
            )
        if np.isfinite(top):
            # Every feasible point has c·x <= top, so a node whose bound
            # passes this holds none: an infeasible node LP, whose bound
            # diverges along its Farkas ray, stops on it before there is an
            # incumbent (one ran all 200 epochs on ns1830653, and held its
            # batch for 1.8 s).
            self.box_cutoff = top + 1e-6 * (1.0 + abs(top)) + 1e-9
        self.restart_free = int(np.count_nonzero(self.is_int & (lower < upper)))
        if self.verbose and first:
            if self.probing_stats["probed"]:
                stats = self.probing_stats
                print(
                    f"Probing: {stats['probed']} variables, {stats['fixed']} fixed, "
                    f"{stats['tightened']} bounds tightened, {stats['seconds']:.2f}s"
                )
            print(
                f"MIP: {mip.num_variables} variables ({mip.num_integers} integer), "
                f"{mip.A.shape[0]} rows, {mip.A.nnz} nonzeros, on "
                f"{(self.device or jax.devices()[0]).platform}, "
                f"{self.batch_size} nodes a batch"
                + (" (auto)" if self.batch_size_auto else "")
            )
            if self.coefficients_tightened:
                print(
                    f"Coefficient tightening: {self.coefficients_tightened} coefficients"
                )
            if self.objective_step is not None:
                print(f"Objective values are multiples of {self.objective_step:g}")
            if self.level_lane is not None:
                lane = self.level_lane
                print(
                    f"Objective levels: {lane.columns.size} integer columns, "
                    f"{lane.levels.size} levels, searched by HiGHS on a thread"
                )
        return lower, upper

    def _root(self, lower, upper):
        """The root: its LP, cut rounds, heuristics, reduced-cost fixing and
        evaluation. Returns ``None`` (on to the tree), ``"restart"``, or the
        search's result when the root ends it."""
        mip = self.mip
        self.service_levels()  # the level lane starts beside the root LP
        root = _Node(-np.inf, 0, None, None, None)
        # The root is in no list until it is evaluated: its bound holds the
        # global bound down meanwhile (for incumbents logged by the root
        # heuristics), from its LP bound once that is known.
        self.active_bound = -np.inf
        if self.mip_start is not None:
            # The user's starting point (``mip_start``), once, before the
            # root LP: its value prunes and fixes from the start.
            start = _fill_start(
                self.mip_start, self.original.lower, self.original.upper
            )
            self.mip_start = None
            before = None if self.incumbent is None else self.incumbent[0]
            self.try_incumbent(start, "S", None)
            self.mip_start_accepted = self.incumbent is not None and (
                before is None or self.incumbent[0] < before
            )
            if self.verbose and not self.mip_start_accepted:
                print("mip_start: not feasible, ignored")
        lp = self.node_lp(lower, upper)
        # The root LP straight to convergence (``root_max_epochs``, ``None``:
        # no limit). The infeasibility check (``detect_infeasibility``) runs
        # PDHG in chunks of doubling length with a check between them; on
        # bab5's and neos-4306827-ravan's roots it took 0.7-0.8 s and 20-33
        # epochs against 0.3 s and 15-17 for one solve. It runs only on a
        # root that stops unconverged with time left.
        warm = None
        if self.root_warm is not None:
            warm = (self.root_warm["solution"], self.root_warm["opt_state"])
        self.try_trivial(lower, upper)
        self.block_decomposition(lower, upper)
        self.launch_root_jumps(lower, upper)
        lp_started = time.time()
        result = None
        if self.root_lp == "highs":
            result = self.highs_root_lp(lp, lower, upper)
            if isinstance(result, str):  # "infeasible"
                return self.empty_root()
        if result is None:
            result = self.solve_lp(lp, warm, self.root_max_epochs)
        self.root_lp_stats["seconds"] += time.time() - lp_started
        if self.verbose:
            stats = self.root_lp_stats
            highs = (
                f" (HiGHS {stats['highs_seconds']:.2f}s)" if self.root_lp == "highs" else ""
            )
            print(
                f"Root LP ({self.root_lp}): {stats['seconds']:.2f}s{highs}, "
                f"{result['epochs']} PDHG epochs"
            )
        if (
            not result["converged"]
            and result["stop_reason"] not in ("interrupted", "time_limit")
            and not self.out_of_time()
        ):
            detection = self.detect(lp, result, self.root_max_epochs)
            if detection["status"] == "primal_infeasible":
                return self.empty_root()
            if detection["status"] == "dual_infeasible":
                return self.result("unbounded_or_infeasible")
            result = detection["result"]
        self.nodes += 1
        # The root LP's bound holds for the root box whatever its duals, so
        # it is the global bound from now (the first incumbent's line, and
        # the heartbeat's through the cut rounds, showed -inf before).
        self.active_bound = max(
            self.active_bound,
            self.round_bound(self.lp_bound(result["solution"], lower, upper)[0]),
        )
        # Points the early root jumps found while the LP was solved.
        self.collect_heuristics(schedule=False)
        if result["stop_reason"] == "interrupted":
            return self.result("interrupted")
        self.root_heuristic_used = 0.0
        use_root_heuristics = (
            self.use_fix_and_propagate
            or self.use_bulk_rounding
            or self.use_jump
            or self.use_pump
        )
        cut_loop_due = (
            self.cuts
            and self.cut_rounds
            and result["stop_reason"] != "time_limit"
            and not self.out_of_time()
        )
        if cut_loop_due and use_root_heuristics:
            # Heuristics on the first root LP, before the cut rounds: the
            # rounds then judge their progress against the gap, and the
            # background improving jump works beside them. Without this,
            # rmine13's first incumbent came only after its cut loop.
            self.active_bound = self.round_bound(
                self.lp_bound(result["solution"], lower, upper)[0]
            )
            # With ``parallel_heuristics`` they run beside the rounds.
            if not self.background_root_heuristics(lower, upper, result):
                self.root_heuristics(lower, upper, result, improve=False)
        if cut_loop_due and not self.out_of_time():
            result = self.cut_loop(lower, upper, result)
            if result["stop_reason"] != "interrupted" and self.root_closed():
                return self.empty_root()
            lp =self.node_lp(lower, upper)
            if self.cut_list and self.propagator is not None:
                # Within the root box, which cutoff probing may have fixed
                # during the rounds.
                lower, upper, infeasible = self.propagator(
                    np.maximum(lower, self.root_lower),
                    np.minimum(upper, self.root_upper),
                    max_passes=20,
                )
                if infeasible:
                    return self.empty_root()
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
        # Every round's bound holds for the root box (the cuts are valid,
        # and propagation only narrows the box), so keep the rounds' best:
        # the last re-solve's can be lower.
        self.active_bound = max(
            self.active_bound,
            self.round_bound(self.lp_bound(result["solution"], lower, upper)[0]),
        )
        rounds_bound = self.active_bound
        # A point HiGHS found beside the cut rounds gives the heuristics
        # below a cutoff (and spares those that run only without one).
        self.collect_heuristics(schedule=False)
        if self.root_closed():
            return self.empty_root()
        if self.use_round_repair and not self.out_of_time():
            # Round and repair first, from an exact vertex when the root is
            # crossed over anyway (RENS, ``root_crossover``): the later
            # heuristics, RENS and RINS then start with a cutoff.
            started = time.time()
            vertex = self.current_root_vertex()
            if vertex is None and self.root_crossover is not False:
                self.root_vertex(lower, upper, result["solution"])
                vertex = self.current_root_vertex()
            if vertex is not None:
                self.round_repair(
                    vertex,
                    lower,
                    upper,
                    (result["solution"], result["opt_state"]),
                    snap=self.int_tol,
                )
            self.round_repair(
                np.asarray(result["solution"].primal, np.float64),
                lower,
                upper,
                (result["solution"], result["opt_state"]),
            )
        if use_root_heuristics:
            self.root_heuristics(lower, upper, result)
        if self.root_closed():
            return self.empty_root()
        root_x = np.asarray(result["solution"].primal, np.float64)
        root_warm = (result["solution"], result["opt_state"])
        if self.rens_seconds and not self.out_of_time():
            # From an exact vertex, as HiGHS's RENS works from its simplex
            # root: far more of it is integral than of a PDHG point. The cut
            # loop keeps its last one; otherwise cross over now
            # (``root_crossover``).
            vertex = self.root_vertex_x
            if vertex is None:
                vertex = self.current_root_vertex()  # round and repair's
            if vertex is None and self.root_crossover is not False:
                vertex, _ = self.root_vertex(lower, upper, result["solution"])
            if vertex is not None:
                self.rens(vertex, lower, upper, root_warm, min_fixed=0.3)
            else:
                self.rens(root_x, lower, upper, root_warm)
        if self.dive_every and not self.out_of_time():
            # The root dives share one heuristic budget.
            started = time.time()
            deadline = started + self.dive_seconds()
            self.dive(
                root_x, lower, upper, root_warm, rule="fractional", deadline=deadline
            )
            if self.incumbent is not None and time.time() < deadline:
                self.dive(
                    root_x, lower, upper, root_warm, rule="guided", deadline=deadline
                )
        if (
            self.rins_seconds
            and self.incumbent is not None
            and not self.out_of_time()
            and self.submip_budget_left()
        ):
            self.rins_pending = False
            self.rins(root_x, root_warm)
        # Root reduced-cost fixing from the root heuristics' incumbent, so
        # the root sweep and the first children work in the fixed box.
        self.root_exact_terms(lower, upper, result)
        if not self.tighten_root():
            return self.empty_root()
        if self.restart_due(_RESTART_ROOT_SHARE):
            return "restart"
        if self.root_lower is not lower:
            lower, upper = self.root_lower, self.root_upper
            lp = None  # rebuilt over the fixed box if ``evaluate`` needs it
        # The root starts from the rounds' best bound, which holds for its
        # box (only narrowed since), and its children from the root's:
        # ``evaluate`` keeps the larger of it and the last re-solve's. On
        # irp the root's line dropped from the rounds' 33,249.82 to the last
        # re-solve's 33,242.52, and the tree's bound stayed there.
        if np.isfinite(rounds_bound):
            root.bound = max(root.bound, rounds_bound)
        # The root branched as an exact lane node (``exact_root_candidates``).
        result = self.root_exact_strong(root, lower, upper, result) or result
        self.evaluate(root, lower, upper, lp, result)
        self.tree_started = True
        self.log(" ", None, force=True)  # the root's line
        return None

    def _tree(self):
        """The node loop. Returns the search's status, or ``"restart"``."""
        status = "optimal"
        # The tree's clock, for strong branching's share (``strong_branching_on``),
        # summed over the rounds between restarts.
        self.tree_clock = (time.time(), self.strong_branch_seconds)
        try:
            while (
                self.open
                or self.plunge_node is not None
                or self.in_flight is not None
                or self.pending_children is not None
                or (
                    self.exact_lane is not None
                    and (self.exact_lane.jobs or self.exact_reserved)
                )
            ):
                self.collect_heuristics()
                collecting = time.perf_counter()
                self.collect_exact()
                self.exact_collect_seconds += time.perf_counter() - collecting
                self.feed_exact()
                self.collect_submip()
                self.feed_submip()
                if self.levels_settled():
                    break  # no objective level below the cutoff is left
                if (
                    self.exact_lane is not None
                    and (self.exact_lane.jobs or self.exact_reserved)
                    and not self.open
                    and self.plunge_node is None
                    and self.in_flight is None
                    and self.pending_children is None
                ):
                    # Only the lane's nodes are left: wait for one.
                    if self.out_of_time():
                        status = "time_limit"
                        break
                    self.exact_lane.wait(0.05)
                    continue
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
                if self.restart_due(_RESTART_TREE_SHARE):
                    status = "restart"
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
                result = self.solve_node_lp(
                    lower, upper, self.fit_warm(node), self.node_max_epochs
                )
                if result["stop_reason"] == "interrupted":
                    self.push(node)
                    status = "interrupted"
                    break
                if result["stop_reason"] == "time_limit":
                    self.push(node)
                    status = "time_limit"
                    break
                (result,), requeued = self.after_node_solves(
                    [node], [(lower, upper)], [result]
                )
                if requeued:
                    continue  # back in the queue for new tree cuts
                self.bound_stops += result["stop_reason"] == "bound_target"
                self.nodes += 1
                # ``evaluate`` builds the LP at the search's precision if needed.
                self.evaluate(node, lower, upper, None, result)
                self.log(" ", None)
        except KeyboardInterrupt:
            status = "interrupted"
        finally:
            # The pipeline's nodes back in the queue (``pipeline_batches``),
            # so the bound and a restart see them; likewise the exact
            # lane's, unsolved.
            self.drain_pipeline()
            if self.exact_lane is not None:
                for node in self.exact_lane.drop():
                    node.exact_tried = False
                    self.push(node)
            started, strong_before = self.tree_clock
            self.tree_seconds += time.time() - started
            self.tree_strong_seconds += self.strong_branch_seconds - strong_before
            self.tree_clock = None
        return status


def solve(
    model,
    mip_rel_gap=1e-4,
    mip_abs_gap=1e-6,
    max_nodes=None,
    max_seconds=None,
    branching="auto",
    node_selection="hybrid",
    integrality_tolerance=1e-5,
    feasibility_tolerance=1e-6,
    lp_tolerance=1e-4,
    node_max_epochs=None,
    root_max_epochs=None,
    infeasibility_tolerance=1e-8,
    heuristic_every=0,
    propagate=True,
    coefficient_tightening=True,
    cuts=False,
    cut_rounds=10,
    max_cuts_per_round=500,
    cut_min_progress=None,
    cut_round_epochs=10,
    gomory_cuts=True,
    aggregate_integers="auto",
    flow_covers=True,
    mir_cuts=True,
    clique_cuts=True,
    zero_half_cuts=True,
    cut_age=100,
    max_cut_rows=10000,
    cut_pool_size=_CUT_POOL_SIZE,
    gomory_candidates=400,
    inout_separation=0.5,
    root_redcost_propagation=True,
    global_redcost_fixing=True,
    cut_pool_propagation=True,
    plunge_child="rounding",
    implied_integer_cuts=True,
    separation_backend="auto",
    separation_device="cpu",
    node_bound_stop=True,
    dual_repair=True,
    node_lp_tolerance=1e-3,
    heuristic_effort=0.3,
    reliability_threshold=4,
    strong_candidates=8,
    strong_epochs=50,
    strong_max_depth=None,
    strong_screen=32,
    strong_backend="jax",
    root_strong_candidates=4,
    root_strong_epochs=5,
    root_crossover=False,
    round_repair=False,
    round_repair_every=None,
    round_repair_seconds=10.0,
    tree_cuts=True,
    tree_cut_every=10,
    max_tree_cuts_per_round=200,
    tree_cut_rows=1000,
    tree_cut_nnz=None,
    tree_cut_points=4,
    tree_cut_async=True,
    tree_cut_rounds=1,
    tree_cut_share=0.25,
    tree_cut_max_nnz=None,
    fix_and_propagate=False,
    feasibility_jump=True,
    jump_iterations=None,
    repair=True,
    repair_max_epochs=20,
    rins_seconds=10.0,
    rins_every=5,
    rins_nodes=200,
    rens_seconds=0,
    rens_nodes=500,
    dive_every=0,
    dive_depth=20,
    plunge_depth=4,
    batch_size="auto",
    feasibility_pump=False,
    pump_iterations=30,
    pump_tolerance=1e-4,
    pump_jump_iterations=3000,
    pump_jump_share=0.2,
    probing=False,
    probing_seconds=None,
    probing_max_variables=None,
    probing_backend="auto",
    probing_max_useless=1000,
    probing_passes=_PROBE_MAX_PASSES,
    probing_max_implications=_PROBE_MAX_IMPLICATIONS,
    bulk_rounding=False,
    bulk_starts=4,
    repair_share=0.2,
    batch_epochs=4,
    children_backend="auto",
    batch_roundings=False,
    tree_heuristics=False,
    tree_heuristic_every=10,
    tree_heuristic_share=0.05,
    tree_jump_iterations=None,
    node_crossover=False,
    node_crossover_every=1,
    node_crossover_seconds=2.0,
    heuristic_seconds=10.0,
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
    precision="float32",
    node_precision="float32",
    jump_backend="auto",
    parallel_heuristics=True,
    symmetry=False,
    restarts=5,
    restart_presolve=False,
    mip_start=None,
    implied_bound_cuts=True,
    probing_cliques=True,
    implied_variable_bounds=True,
    max_aggregation=6,
    aggregation_width=_CUT_AGGREGATION_WIDTH,
    cmir_per_row=_CUT_CMIR_PER_ROW,
    cut_score_weights=_CUT_SCORE_WEIGHTS,
    exact_nodes=16,
    exact_node_seconds=10.0,
    exact_band=0.25,
    exact_strong_candidates=8,
    exact_strong_iterations="auto",
    exact_overflow_share=0.0,
    exact_overflow_epochs=None,
    exact_basis_memory=2e9,
    exact_strong_lookahead=None,
    exact_strong_cutoff=False,
    exact_root_candidates=128,
    exact_root_seconds=60.0,
    exact_root_iterations=None,
    cutoff_probing=True,
    cutoff_probing_threads=4,
    cutoff_probing_seconds=60.0,
    cutoff_probing_iterations=1000,
    one_opt=False,
    objective_propagation=True,
    plunge_gap=0.25,
    branch_implied=True,
    near_integral_tolerance=0,
    tree_pool_every=1,
    parallel_rounding=False,
    node_jump_seconds=1.0,
    pipeline_batches=True,
    gpu_dive_every=0,
    gpu_dive_steps=30,
    gpu_dive_epochs=8,
    gpu_dive_share=0.2,
    gpu_dive_snap=0.1,
    gpu_dive_fix_share=0.1,
    gpu_dive_crossover=4,
    exact_plunges=False,
    exact_plunge_share=0.25,
    jump_threads=1,
    jump_restarts=True,
    root_jump_seconds=60.0,
    block_decomposition=False,
    block_decomposition_seconds=300.0,
    trivial_heuristic=True,
    submip_solver="highs",
    rins_tolerance=0.1,
    crossover_seconds=10.0,
    crossover_solutions=3,
    crossover_min_fixed=0.5,
    proximity_seconds=10.0,
    first_solution_seconds=0.0,
    objective_levels=True,
    objective_level_seconds=2.0,
    cut_round_lp="jaddle",
    root_lp="jaddle",
):
    """
    Solve a mixed-integer LP by branch and bound, with ``jaddle_linear.solve``
    for the LP at every node.

    ``model`` is a ``MIP``, or anything ``from_highs`` reads: a path to a file
    HiGHS reads, a ``highspy.Highs`` with a model loaded or a
    ``highspy.HighsLp``.

    The search:

    * **Root.** Bounds are tightened by activity-based propagation
      (``propagate``), and the root LP is solved to convergence by
      ``jaddle_linear.solve`` (within ``root_max_epochs``; ``None``, the
      default: no limit). Only a root LP that stops unconverged with time
      left goes on to ``jaddle_linear.detect_infeasibility``, which
      certifies infeasibility or an unbounded relaxation; with no epoch
      limit, an infeasible root LP runs until ``max_seconds``. The checks
      between that call's chunks of doubling length cost a feasible root
      about 2.5× (bab5's: 0.8 s and 24 epochs against 0.3 s and 17).
    * **Symmetry** (``symmetry``, off by default; ``"auto"``: on when
      igraph is installed, ``pip install jaddle[symmetry]``; ``True``
      requires it).
      After coefficient tightening, the column symmetries of the MIP over
      the propagated root box are found as graph automorphisms by bliss
      (``symmetry_generators``, each checked exactly), the ones that swap
      blocks of columns are assembled into orbitopes (``orbitopes``), and
      each orbitope is broken at the root (``symmetry_rows``), keeping one
      optimal solution: packing orbitopes (every row at most one 1) by
      fixing ``x[i, j] = 0`` for ``j > i`` and rows ``x[i, j] <=
      Σ_{k<i} x[k, j−1]`` (lexicographic column order), others by ordering
      the columns' sums. The rows join the searched MIP as coefficient
      tightening's do; incumbents are checked against the MIP as given.
      ``result["symmetry"]`` gives the generators, the orbitopes' shapes,
      the rows, the fixings and the time. On bab5 bliss takes 0.02 s and
      finds 5 two-column orbitopes (440 or 446 binaries a column, 4,412
      columns in all, none packing): 5 rows.
    * **Restarts** (``restarts``, default 5: the most restarts; ``0``
      turns them off). Reduced-cost fixing of the root box (with the root's
      exact duals, whenever the incumbent improves) fixes integers for the
      whole search. Once it has fixed 2.5% of the integers free at the start
      of the round (checked at the root, after its heuristics) or 10% (in
      the tree), the search restarts from that box, as HiGHS does: the tree
      goes, coefficient tightening, probing, the root LP, the cut rounds and
      the root's heuristics are redone over the smaller box, and the
      incumbent, pseudocosts, conflict pool and symmetry rows carry over.
      The cuts that were tight or had a dual at their last solve join the
      searched MIP's rows for good (valid for the smaller box too), so the
      restarted root starts from about their bound and its cut rounds build
      on it. The bound reached before a restart stays as a floor.
      ``result["restarts"]`` counts them and ``result["restart_log"]`` gives
      each one's time, nodes, share fixed, integers left free, cuts kept
      and bound.
      HiGHS restarts four times on bab5 (20,508 → 11,243 columns) and
      solves it in about 530 s; Jaddle's root reduced-cost fixing there
      fixed 8 to 2,908 integers (up to 14%), depending on its root bound
      and incumbent.
      With ``restart_presolve=True`` (off by default) a restart also runs
      HiGHS's presolve on the searched MIP over the restarted root box, so
      the fixed columns and the rows they empty leave the LP (a restart
      alone keeps them, at ``l = u``). When it removes at least 10% of the
      columns, the rest of the search is a new search of the reduced MIP:
      its points come back through HiGHS's postsolve (marked ``"Q"``), and
      its bound and status are the result's. HiGHS cannot map the
      incumbent into the reduced columns, so that search starts from the
      cutoff alone (as ``presolve=True`` does from a ``mip_start``), with
      the pseudocosts, conflicts and cut pool left behind; it may restart,
      and presolve, again. ``result["restart_presolve"]`` gives each
      presolve's status, columns and rows before and after, and, when the
      search went on in the reduced MIP, its status, nodes and seconds.
    * **More from the incumbent** (all on by default). Each holds for
      every solution better than the cutoff, which is all the search needs.
      ``root_redcost_propagation`` puts the root's reduced-cost row
      (``constant + r·x <= cutoff`` over the root box) in the conflict pool
      (``ConflictPool.set_global``), redone with each incumbent. At a node
      whose bounds have moved variables off their favoured bounds, it uses
      up that part of the room and can fix others before the node's LP.
      ``global_redcost_fixing`` runs reduced-cost fixing of the root box
      with every node LP's duals, not only the root's: ``c·x >= constant +
      r·x`` holds for every feasible ``x`` whatever the duals. The result
      reports ``global_redcost_fixed``.
    * **In-out separation** (``inout_separation``, default 0.5;
      ``None``: off; a value in (0, 1)). With an incumbent ``x̄``, the root's cut rounds and
      the tree's (``tree_cuts``) also separate at ``λ·x + (1 − λ)·x̄``.
      A valid cut violated there is violated at the LP point ``x`` by
      more, since ``x̄`` satisfies it. From a point nearer the feasible
      region the separators pick other rows, divisors and covers, and
      find deeper cuts. All candidates are still chosen at ``x``
      (``select_cuts``). λ starts at ``inout_separation`` for each
      incumbent. A round that keeps none of the in-out cuts moves it
      halfway to 1, and past 0.9 the incumbent is not used again until it
      changes. At the root an incumbent exists during the rounds after a
      ``mip_start`` and after a restart. Results: ``inout_cuts`` (in-out cuts kept) and
      ``inout_rounds``.
    * **Branching on implied integers** (``branch_implied``, on by
      default). Fractional implied integers (``MIP.implied``: continuous
      columns HiGHS's presolve showed some optimum has integral) join the
      branching candidates at nodes that branch, scored like the integers
      (pseudocosts, strong branching, ``branching_impact``). They are not
      part of the integrality test: a node whose integers are integral
      still closes on its repaired point. Their box bounds go into the node
      LPs rounded outward to whole numbers. Splitting at ``x <= k`` /
      ``x >= k + 1`` keeps an optimum, since every choice of the integers
      has a best completion with them integral. On ns1830653, 134 implied
      columns cost 1,000 each and hold most of the objective, and
      branching on the binaries alone leaves the LP bound where it is.
      ``result["implied_branchings"]`` counts the branchings on them.
    * **Objective propagation** (``objective_propagation``, on by
      default). The search's bound propagation (each node's box before its
      LP, every child after branching, strong branching's children, the
      root box at a restart) also runs the objective row
      ``c·x <= cutoff`` (``CutoffPropagator``), at the cutoff of the moment
      (``proof_cutoff``: the largest value a solution better than the
      incumbent by the gaps can have). Variables whose extreme values
      would use up the room between the box's least objective and the
      cutoff are tightened, and a box with no room left closes before its
      LP. Its no-goods hold for every solution better than the cutoff,
      which only falls. The heuristics' propagation has always had the
      row. Batched device children (``children_backend``) and device
      strong branching (``strong_backend="jax"``) do not run it.
    * **Cut pool propagation** (``cut_pool_propagation``, on by default).
      Every cut the root and tree rounds select is kept, and the search's
      propagation runs them beside the rows at every node, after they have
      left the LP, as HiGHS propagates its cut pool. Cuts with more than
      ``max(32, n / 8)`` nonzeros are not kept (dense ones propagate little
      and cost every node), nor more than the newest 10,000. Each holds
      over the root box it was separated in, which only shrinks, so all
      stay valid, across restarts too. ``result["cut_pool_rows"]`` counts
      them. On ns1830653 HiGHS keeps 6,000 to 10,000 cuts in its pool with
      100 to 170 in its LP.
    * **Starting solution** (``mip_start``, default ``None``). A point
      with one entry per column of ``model`` (NaN allowed for continuous
      columns, which the repair LP fills in; integer entries are rounded)
      is tried before the root LP, as any heuristic's point would be
      (source ``"S"``). ``result["mip_start_accepted"]`` says whether it
      was feasible (``None``: no start). With ``presolve=True`` it is
      completed and checked in the original model (HiGHS cannot map a
      point into the reduced one); the reduced search then looks only for
      points better by more than ``mip_abs_gap``/``mip_rel_gap``, and the
      start is returned when none turns up (``"optimal"`` once the search
      proves there is none).
    * **1-opt** (``one_opt``, off by default). Every new incumbent is
      passed to ``one_opt``: each integer with a nonzero cost moves in its
      improving direction as far as its bounds and the rows' slack allow,
      largest gains first, continuous variables fixed (source ``"1"``,
      counted in ``result["one_opt_finds"]``).
    * **Coefficient tightening** (``coefficient_tightening``). Over the
      propagated root box, each binary's coefficient in an inequality row is
      cut to the row's excess ``maxact − b`` when it is larger
      (``tighten_coefficients``). This shrinks big-M rows such as
      ``y − M z <= 0`` to the bound on ``y`` that the other rows imply,
      without changing the integer-feasible points; the search then runs on
      the tightened rows, and incumbents are still checked against the MIP
      as given.
    * **Root cuts** (``cuts``, off by default). Up to ``cut_rounds`` rounds
      (default 10; ``None``: 40 with Gomory cuts on, else 5) of
      Gomory mixed-integer cuts (``gomory_cuts``, on by default; ``"auto"``
      turns them on only for MIPs with at most 10,000 nonzeros, whose
      crossovers take milliseconds), knapsack
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
      flow set over those variable bounds. ``mir_cuts=False`` (on by
      default) keeps no c-MIR cuts while the rows and aggregates still give
      their cover and flow cover cuts (the tableau MIR cuts of
      ``gomory_cuts`` are separate). MIR cuts come from single rows
      and from aggregates of up to three rows that cancel continuous
      variables; with ``aggregate_integers``
      (``"auto"``, the default: on for MIPs with at most 10,000 nonzeros,
      and for every MIP with the numba separation backend, where a round
      costs milliseconds), aggregates of up to five rows also cancel the
      most fractional integer term, so pure-integer rows combine. On lseu that lifts the root bound
      (exact LPs) from 1,061 to 1,100. ``max_aggregation`` (default 6) sets the
      rows added to an aggregate instead (``None``: 2, or 4 with
      ``aggregate_integers``); on map06, whose rows bound continuous
      variables through chains of other continuous variables, 6 with
      probing's variable bounds (below) takes the root bound (exact LPs)
      from −395.8 to −393.9. The first aggregation step of a row tries the
      partner rows of its ``aggregation_width`` best columns to cancel
      (default 3), each continuing greedily, so a row gives up to
      ``aggregation_width × max_aggregation`` aggregates instead of
      ``max_aggregation``; the ``cmir_per_row`` most efficacious c-MIR cuts
      over a row's aggregates are kept (default 3; 1 keeps the best only,
      as before). Cover and MIR cuts are separated by
      ``CutSeparator``:
      up to three covers per row (``cover_cuts``), each lifted exactly by
      a dynamic program over the knapsack. With coefficient tightening,
      they are separated from the rows as given as well as the tightened
      rows, since the original knapsacks can give stronger covers. A
      round keeps at most
      ``max_cuts_per_round`` cuts (default 500) with efficacy (distance cut
      off) at least 1e-4 that are not near-parallel, retires cuts that are
      slack, and re-solves the LP warm. Candidates are ranked as SCIP's
      hybrid cut selector ranks them (``select_cuts``): efficacy over the
      round's largest, plus ``cut_score_weights = (w_obj, w_int, w_dir)``
      (default ``(0.1, 0.1, 0.5)``) times the cut's |cosine| with the
      objective, its share of integer columns, and its directed cutoff
      distance (how far it lies along the segment to the incumbent) over the
      round's largest; ``None`` ranks by efficacy alone, as before. Tree
      cut rounds pick each node point's candidates the same way, then
      merge the points' picks by efficacy. With 10 a round, the strongest
      covers took every slot on bab5 and no Gomory cut got in: 100 takes
      its root bound (with clique cuts) from −113,254 to −111,358, and
      ns1830653's from 10,621 to 11,622, HiGHS's root bound. A cut leaves
      the LP once it has been idle (slack, with a zero dual, at the
      round's re-solve) for ``cut_age`` rounds in a row (default 100), and
      stays in the global cut pool (below). On degenerate LPs a cut slack
      at one optimum is often binding again at the next, and dropped at
      once the rounds cycle between them (ns1830653 sat at 11,622). GMI
      cuts are dense (hundreds of nonzeros on ns1830653), so they are
      allowed at the root (``gomory_cuts(max_nnz=None)``) but leave once
      idle. Once three rounds have run, the rounds
      stop when the bound gains less than ``cut_min_progress`` over the
      last three (default 1e-3 with the 40-round setting, else 1e-4),
      relative to the gap to the incumbent when there is one, else to the
      bound (rmine13's root is 0.27% of its objective from the best known
      value, so the objective-relative test alone stopped after one
      round). Each round's re-solve stops after ``cut_round_epochs``
      (default 10): separation needs only a reasonable point and the
      Lagrangian bound holds at any duals; the root is solved to convergence
      once the rounds end. On lseu that cuts the root from 17s to 7s (bound
      1,091 → 1,095).
      With ``root_lp="highs"`` (default ``"jaddle"``; for timing the two)
      HiGHS solves the first root LP, with its presolve and its choice of
      solver, and PDHG then re-solves it warm from HiGHS's vertex and duals
      (``highs_root_lp``), so the search goes on from a PDHG result; the
      vertex serves as the root's crossover. ``result["root_lp"]`` gives
      the ``solver``, the root LP's ``seconds`` and, of them,
      ``highs_seconds``.
      With ``cut_round_lp="highs"`` (default ``"jaddle"``) the rounds re-solve
      with HiGHS's dual simplex instead, warm from the last basis, with the
      cuts added and dropped as rows (``exact_cut_loop``): each round
      separates at an exact vertex, Gomory cuts come from its basis without
      a crossover, and a cut leaves once idle for ``min(cut_age, 10)``
      rounds. The rounds stop as above, or after 25% of ``max_seconds``;
      the kept cuts then go into the PDHG root LP, re-solved warm from
      HiGHS's solution. Each round cuts off only its own point, so the
      bound builds over many rounds: on beasleyC3 (with ``cut_rounds=60``)
      21 exact rounds took the root from 238 to 717–728 in 47 s, where ten
      PDHG rounds took 46 s to reach 661 (HiGHS's root: 733). Results:
      ``exact_cut_rounds`` (``None`` unless they ran: ``rounds``, ``cuts``,
      the last ``lp_bound``, ``seconds``, and the ``lp_seconds``,
      ``gomory_seconds`` and ``separation_seconds`` within them).
      Cover and MIR cuts follow from a row and the bounds alone, so they
      are valid whatever the LP point's accuracy; Gomory cuts rest on the
      crossover's basis, which HiGHS proves optimal, and skip tableau rows
      that do not
      reproduce the basis to 1e-7. The cut block is padded to
      power-of-two sizes, so rounds rarely recompile the LP solver.
      Separation is compiled with numba when it is installed
      (``separation_backend="auto"``, the default, or ``"numba"``;
      ``jaddle._separate_numba``): one call per round, without the GIL,
      runs every tight row's covers, aggregation path, c-MIR and flow
      cover searches, with the numpy searches' cuts up to rounding in
      near-ties. At the LP point a round took 19 ms on beasleyC3 against
      127 ms batched in JAX and 390 ms in numpy, 91 ms on air05 against
      1.5 s in both, and 112 ms on dano3_3 against 0.84 s and 3.9 s.
      Without numba, ``"auto"`` is ``"jax"``, batched separation:
      once a round has at least 32 base inequalities (rows and their
      aggregates), the c-MIR and flow cover searches run on all of them in
      jitted float64 JAX calls (``batched_cmir_cuts``,
      ``batched_flow_cover_cuts``), with the same cuts as the per-row numpy
      searches (``"numpy"``) up to rounding in near-ties. Per round that is
      2.4× faster on fixnet6 and 4× on vpm2; the first calls compile a few
      shapes (one or two seconds, then cached on disk). They run on the
      CPU (``separation_device="cpu"``, the default; ``"search"``: on the
      search's device): float64 is slow on consumer GPUs, and the
      transfers cost more than the work on rows this small. On beasleyC3
      (GPU search) a round's separation went from 1.6 s to 1.0 s, and the
      exact cut rounds fit 22 rounds (root 728.5) where they had fit 19
      (718).
      Clique cuts (``clique_cuts``, on by default; family ``"clique"``)
      come from a conflict graph on the binaries (``ConflictGraph``), built
      once from the rows over the root box: two literals (``x_j`` or
      ``1 − x_j``) conflict when a row cannot hold with both at 1. Each
      round grows cliques greedily from the literals the LP point puts
      above zero, and lifts the violated ones with literals at zero; the
      cut is ``Σ literals <= 1``. On bab5 the graph takes 0.2 s, and with
      exact LPs the clique cuts alone take the root bound from −115,370 to
      −113,411 in 56 cuts; with the other families they add about 1,000
      to the bound after 20 rounds.
      Implied-bound cuts (``implied_bound_cuts``, on by default; family
      ``"impliedbound"``) come from probing (``ImpliedBounds``): when
      ``z = 0`` implies ``x_j <= u0`` and ``z = 1`` implies ``x_j <= u1``
      for a binary ``z``, the cut is ``x_j <= u0 + (u1 − u0)·z`` (likewise
      for lower bounds); the most violated go to cut selection with the
      other families. With ``implied_variable_bounds`` (on by default) the
      ones on continuous columns also join the rows' variable bounds for
      c-MIR and flow cover cuts.
      Zero-half cuts (``zero_half_cuts``, on by default; family
      ``"zerohalf"``) are {0, ½}-Chvátal–Gomory cuts: half the sum of a set
      of rows with integer coefficients, rounded down
      (``zero_half_cuts``). Each round finds the sets by Gaussian
      elimination over GF(2) on the tight rows' coefficient parities, as
      HiGHS's mod-k separator does, from the LP's rows and its cuts. On
      bab5, with exact LPs and 20 rounds, they take the root bound with
      the other families from −111,209 to −107,410, past HiGHS's root
      bound (−107,454); their separation takes about 4 s of it.
      ``max_cut_rows`` (default 10,000; ``None``: no limit) caps the cut rows in
      the root LP. A round that would pass it drops the kept cuts idle
      longest (never an active one), then takes fewer new cuts, and the
      rounds stop once the active cuts fill it. ``gomory_candidates``
      (default 400; ``None``: twice ``max_cuts_per_round``) is the number of
      tableau rows a round tries for Gomory cuts, nearest ½ first. Every
      candidate a root or tree round separates that its point violates,
      not only those chosen for the LP, joins the global cut pool
      (``_CutPool``, the newest ``cut_pool_size``, default 10,000), as
      HiGHS's does; the tree's refreshes take cuts back from it.
    * **Tree cuts** (``tree_cuts``, on by default). After the root cuts,
      the LP's cut block gets a fixed capacity: ``tree_cut_rows`` cuts with
      ``tree_cut_nnz`` nonzeros between them (both rounded up to powers of
      two; ``tree_cut_rows`` defaults to 1,000; ``None``: twice the root's
      cuts and nonzeros, and room for at least two rounds). Every ``tree_cut_every`` nodes (default
      10), between solving node LPs (a batch, or one node) and evaluating
      them, the block is refreshed (``refresh_cuts``) from all of their
      points. In between, every ``tree_pool_every`` nodes (default 1,
      ``0``: off), a pool refresh takes back the pooled cuts the points
      violate, without separating or ageing (steps 2-5 with the pool as
      the only candidates), as HiGHS checks its cut pool at every node.
      A full refresh:

      1. each slot's cut ages by one if it is idle (slack, with a zero
         dual) at every point, else back to 0;
      2. candidates are the pooled cuts any point violates, not in the
         block, and the cuts the root's separators (c-MIR, covers, flow
         covers, cliques, implied bounds) and zero-half separation find at
         the points of the ``tree_cut_points`` (default 4) lowest-bound
         nodes, all over the root box, so they hold at every node; cuts
         with more than ``tree_cut_max_nnz`` nonzeros (default
         ``max(64, n / 8)``) are left out;
      3. at most ``max_tree_cuts_per_round`` (default 200) of the most
         efficacious go in, none a near-duplicate of another, each in a
         free slot or in place of the cut idle longest (at least two
         refreshes); ones that find no room are dropped
         (``result["tree_cuts_dropped"]``);
      4. members whose own point a new cut cuts off go back in the queue to
         be solved again against it before they branch, up to
         ``tree_cut_rounds`` times each (default 1); their pseudocosts use
         the LP value before the new cuts.

      The block keeps its shape, and the batched node solver
      (``jaddle_linear.solve_many``) and the device kernels take the rows
      as data, so a refresh recompiles nothing, batched or serial. The
      results just solved have the duals of replaced slots zeroed, so their
      bounds hold for the new rows, and an open node's warm start has them
      zeroed when it is solved (``fit_warm``, by the block version it was
      made under). Refreshes take at most ``tree_cut_share`` (default 0.25)
      of the tree's time after its first seconds, and the tree heuristics
      then run by a clock that counts their time as nodes
      (``heuristic_clock``). Results: ``tree_cut_refreshes``,
      ``tree_cuts_added``, ``tree_cut_pool_hits``, ``tree_cut_requeued``,
      ``tree_cut_families``, ``tree_cut_seconds``, ``tree_cut_skipped`` and
      ``cut_pool``.
    * **Node LPs.** Every LP of the search is Jaddle's PDHG: the root and
      its cut rounds at ``precision``, node LPs at ``node_precision``,
      batched on the device with ``batch_size > 1``. Crossover
      (``jaddle_linear.crossover``, HiGHS from the PDHG point) gives exact
      vertices where asked: each Gomory cut round, ``root_crossover`` and
      ``node_crossover``. A node's bound is the Lagrangian bound of its
      duals, rigorous whatever their accuracy.
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
      tightened with the root's duals whenever the incumbent improves
      (first just before the root branches, with the root heuristics'
      incumbent). That needs a rigorous root bound. PDHG duals at a loose
      tolerance leave tiny wrong-sign reduced costs on variables with an
      infinite bound, which the bound drops. With ``root_crossover``
      (off by default; ``"auto"``: only then; ``True``: always) the root LP is crossed over
      to an exact vertex (reusing RENS's crossover), and its duals are used
      instead. On buildingenergy that takes 2 s and gives a rigorous 33,246.2,
      HiGHS's LP optimum; the PDHG bound gave 33,247.9, above it. Node
      crossovers (``node_crossover``) likewise prefer the vertex's rigorous
      bound.
      Since that bound holds at any duals, a node LP stops as soon as it
      closes the node (``node_bound_stop``, on by default:
      ``jl.solve(bound_target=...)`` checks it every epoch). An infeasible
      node's duals diverge along a Farkas ray and pass the cutoff within a
      few epochs, instead of running ``node_max_epochs``. On ns1830653 (3
      minutes) that took the search from 140 nodes to 385 (21.9 to 10.2
      epochs a node) and the bound from 11,678 to 11,976 (``bound_stops``
      in the result). No node LP or root cut round runs infeasibility
      detection: a node LP that stops unconverged keeps its bound and
      branches on its point.
      The bound charges each reduced cost at the bound it points at, so on
      a continuous column boxed to 1e6 a dual error of 1e-3 costs 1e3 of
      bound. With ``dual_repair`` (on by default; ``DualRepair``) each
      device node's and strong-branching child's duals are repaired before
      the bound is taken: inequality duals are scaled back until no
      reduced cost points at a continuous column's wide side (one more
      than 10·(1 + |xⱼ|) from the node's point), and the higher of the two
      bounds is kept. On CMS750_4 the node bounds were 50 to 43,000 below
      HiGHS's LP over the same box; repaired, 98.7% of that came back, for
      about 7 ms a node, and reduced-cost fixing went from 0 tightenings to
      37,846 in 120 s (``dual_repairs`` in the result: tried, improved,
      seconds).
    * **Conflicts** (``conflicts``, on by default; ``ConflictPool``). Each
      node or child closed by its Lagrangian bound leaves a dual proof,
      ``constant + r·x <= cutoff``, which every better solution satisfies
      whatever the accuracy of the duals, and a child that propagation
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
      then the heap's top) and solves their LPs in one device call
      (``jaddle_linear.solve_many``: ``solve()``'s compiled programs vmapped
      over the nodes' bounds and warm starts, with the LP's rows as data,
      so it compiles once per shape and tree cut refreshes reuse it). The
      nodes are then evaluated in order on
      the host exactly as in the serial search. Members run in lockstep,
      for at most ``batch_epochs`` (default 4) epochs a call. By default
      (``node_max_epochs=None``) a node LP gets that one call and branches
      on its point (truncated node LPs; its Lagrangian bound holds at any
      duals, and its pseudocost gains use that bound when the LP did not
      converge); only a node with every integer fixed gets up to 200
      epochs, so it closes proven. With ``node_max_epochs`` given, an
      unfinished node goes back to the queue, warm-started, until that many
      in all. On ns1830653 (GPU, batches of 8) the truncated LPs took the
      tree from 16 to 147 nodes a second, with a higher bound at 60 s.
      Worth it on a GPU; on a CPU the members run one after another. The
      default, ``"auto"`` (``auto_batch_size``), gives 32 on a CPU and, on
      a GPU or TPU, the largest power of two up to 128 whose batch fits in
      three quarters of the device's free memory, from the node LP's size
      (variables, rows and nonzeros, with the tree cut block) and whether
      the children kernels would run; 128 when the device reports no
      memory. ``1``: the serial search. With ``pipeline_batches`` (default ``True``) the
      batches overlap: while the device solves one batch's LPs, the host
      evaluates the batch before (bounds, tree cuts, heuristics, branching),
      and its children are built on the device after the batch in flight,
      reaching a batch two steps later. The LP and child calls run on a
      device thread: XLA drives their data-dependent loops from the host on
      a GPU, so a call returns only once its work is done. A batch solved
      before a tree cut refresh has the replaced slots' duals zeroed
      (``stale_free``). On ns1830653 (GPU, 128 a batch, repeatable runs)
      the overlap took 440 to 523 nodes a second.
    * **Device** (``device``). ``"auto"`` (default) runs MIPs with at
      most 10,000 nonzeros on the CPU: their node LPs are latency-bound
      on an accelerator (flugpl: 3s on the CPU, 18s on a GPU).
      ``"cpu"``, ``"gpu"``, ``"tpu"`` or a ``jax.Device`` pick one;
      ``None`` keeps JAX's default device.
    * **Precision** (``precision``). ``"float32"`` (default), ``"float64"``
      or ``"float16"`` runs the whole search, node LPs included, at that
      profile (see ``jaddle_optimisers.configure_jax``); ``None`` keeps the
      global one. A ``MIP`` built at another precision is
      converted. Propagation, cuts and feasibility checks stay in float64
      on the host either way. ``node_precision`` (or
      ``node_lp_options={"precision": ...}``) runs the node LPs at another
      profile: serial and batched node solves and strong branching's child
      LPs. The root LP, cut rounds, heuristic LPs and infeasibility checks
      stay at ``precision``. Warm starts are cast between the two, and
      node bounds stay valid at any precision (they are Lagrangian bounds,
      computed in float64 from the node's duals). The defaults,
      ``precision="float32", node_precision="float32"``, keep every LP
      cheap; ``precision="float64"`` gets an accurate root; ``node_precision=None`` keeps the node LPs at
      ``precision``.
    * **Branching** (``branching``): ``"auto"`` (default) is
      ``"pseudocost"``: PDHG's truncated child LPs make strong branching in
      the tree dear for what it learns. On bab5 plain pseudocost branching
      chose a variable with no observation in some direction 93% of the
      time; at sampled nodes the chosen variable ranked a median 17th of
      about 60 by exact strong branching. The PDHG root sweep (8
      candidates, 50 epochs) recorded no gain for 7 of its 8 variables,
      whose exact gains were 5–20. ``"pseudocost"`` scores each
      fractional variable by the product of its estimated down and up bound
      gains, learnt from the nodes solved so far (most fractional until
      there are any); ``"most_fractional"`` picks the variable nearest
      ½. Both pseudocost rules settle near ties by each variable's
      conflict, cutoff and inference history (``branching_scores``, as
      SCIP and HiGHS do), and pseudocosts measure a child's LP before its
      re-solve with new tree cuts. With either pseudocost rule, the root
      first strong-branches its
      ``root_strong_candidates`` (default 4; ``"auto"``: ``strong_candidates``;
      ``0`` turns it off) fractional integers with the largest
      ``branching_impact · f(1 − f)`` (column length times relative cost) for
      ``root_strong_epochs`` (default 5; ``None``: ``strong_epochs``), in one vmapped
      call. This seeds their pseudocosts, so the search sees which variables
      move the bound before it has branched on them. On buildingenergy one
      fractional sizing variable (35,000 rows) holds the whole root gap,
      while the 8,000 fractional hourly integers (3 rows each) gain about 0:
      pseudocost branching without the sweep picks the hourly ones, and the
      bound stalls. The sweep's child bounds that are not rigorous (dropped
      reduced costs) score the variables but neither raise nor close
      anything. ``"reliability"`` also ranks its unreliable candidates by
      score times impact. ``"reliability"`` is pseudocost branching that first
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
      best by pseudocost score, with the conflict pool too. The children's
      bounds feed the
      pseudocosts (gains between unrounded bounds), fix a candidate with a
      closed child to its other side, raise the node's bound, and become
      the children's bounds and warm starts. ``strong_max_depth`` (``None``:
      no limit) stops it below a depth; sub-MIP searches use
      ``"pseudocost"``. Each child's bounds are propagated, so many
      infeasible children are dropped without an LP solve.
    * **Node selection** (``node_selection``): ``"hybrid"`` (default) dives
      depth first until the first incumbent, then takes the best bound with
      plunges: after each best-bound node, up to ``plunge_depth`` (default 4) steps
      continue into its preferred child, so the search keeps reaching leaves
      (and new incumbents) while it raises the bound. A plunge also stops
      at a child whose bound is more than ``plunge_gap`` (default 0.25, as
      SCIP's ``maxplungequot``) of the way from the best open bound to the
      cutoff: such a child is close to being pruned, and the nodes that
      hold the global bound get the next LP instead. The better the
      incumbent, the shorter the plunges; ``1.0`` keeps only the depth
      limit. ``result["plunges_stopped"]`` counts them;
      ``"best_first"`` and ``"depth_first"`` do one throughout.
    * **Incumbents.** A node LP point with integral integers (within
      ``integrality_tolerance``), and the LP point rounded at every
      ``heuristic_every``-th node (default 0: off), are rounded and checked. So is a node
      LP point whose integers are all within ``near_integral_tolerance``
      (default 0: off) of integers, once per rounding, marked
      ``"T"``; the node still branches. A loose PDHG point (float32 nodes
      at 1e-4 without node crossover) rarely passes the integrality test
      even at an integral optimum (a binary at 0.9997)
      (``result["near_integral"]``: its tries and finds). A point is
      feasible when each row's violation relative to ``1 + |b_i|`` and its bound
      violation are at most ``feasibility_tolerance``. With continuous
      variables, a rounded point that is not is repaired by fixing the
      integers and solving the LP in the rest by PDHG, for up to
      ``repair_max_epochs`` epochs. The repair is skipped when the same
      integers already failed, when no violated row has a continuous
      variable, or when bound propagation shows the fixed box infeasible.
    * **Probing** (``probing``, off by default). Before the root LP, every
      integer variable is split in two (a binary into 0 and 1, a general
      integer at its midpoint) and both sides are propagated (``probe``):
      an empty side fixes the variable to the other one (with all it
      implies), and otherwise every bound tightens to the hull of the two
      sides. Conflict-prone variables go first, up to
      ``probing_max_variables`` within ``probing_seconds`` (``None``: 5% of
      ``max_seconds`` up to 30 s, 10 s without a limit), and it stops after
      ``probing_max_useless`` variables in a row that change nothing. Models with more
      than 10,000 nonzeros probe chunks of variables at once in one vmapped
      JAX call (``probing_backend``). Each side propagates for up to
      ``probing_passes`` passes (default 100; it stops sooner at a fixed
      point): on map06 a binary reaches continuous variables down chains
      about 50 rows deep, and 5 passes found under half of what it implies.
      What each side implies is kept as a ``ProbingCache`` for bulk
      rounding and implied-bound cuts; the JAX backend keeps at most
      ``probing_max_implications`` per side (``None``, the default: all).
      With numba installed (``probing_backend="auto"``, or ``"numba"``)
      each side propagates incrementally from the probed variable, visiting
      only the rows whose variables changed (``jaddle._probe_numba``),
      where the numpy and JAX backends sweep every row each pass (on
      presolved mzzv11 20 ms and 80 ms a variable on the CPU). Under numba
      a probe that implies a bound on
      another variable is not counted towards ``probing_max_useless``.
      With ``probing_cliques`` (on by default) the pairs of binaries
      probing shows cannot both take the values found
      (``ConflictGraph.add_implications``) join the conflict graph of the
      clique cuts, so cliques can grow across rows
      (``result["probing"]["conflicts"]``).
    * **Primal heuristics.** On the first root LP, before the cut rounds
      (which then judge progress against the gap, with the background
      improving jump running beside them), and again after them, sharing
      one budget; each round from the second also rounds its point
      (fix-and-propagate, and bulk rounding while there is no incumbent).
      With ``parallel_heuristics`` the ones before and during the rounds
      run in the background instead (below), and bulk rounding and the
      pump wait for the call after the rounds.
      ``fix_and_propagate``
      (off by default; the function of that name) rounds the root LP point in chunks with
      bound propagation after each; if that fails (or is off),
      ``bulk_fix_and_propagate`` (``bulk_rounding``, off by default; independent of
      ``fix_and_propagate``) rounds it from ``bulk_starts`` starts side by side
      (cuOpt's bulk rounding: bulks of √(free) variables, two random
      roundings per bulk propagated together, the probing cache screening
      them first), marked ``"B"``. While there is still no incumbent,
      ``feasibility_jump`` (the function; an LP-free weighted local search
      on the rows) runs for up to ``jump_iterations`` steps from the
      rounded LP point and from the bounds nearest zero. Bulk rounding, the
      two jump starts and the pump each get an equal share of the
      heuristic time left when they start, so a slow one cannot starve the
      rest. With numba, the time still left after an incumbent goes to an
      improving jump from it (``feasibility_jump(improve=True)``: it goes
      on below each point it finds), and the tree heuristics' jumps
      improve the same way. ``jump_backend`` picks the jump: ``"numba"``, the
      incremental one on the host (``pip install jaddle[fast]``), moves
      one variable a step and updates only its rows, as HiGHS does;
      ``"jax"`` makes every step a pass over all the nonzeros on the
      device; ``"auto"`` (default) uses numba when it is installed. On
      academictimetablesmall the numba jump finds a point in 0.1–1 s,
      against 12 s to over 90 s on a laptop GPU. Then the
      objective feasibility pump (``feasibility_pump``, off by default) runs for up to
      ``pump_iterations`` warm-started LP solves at ``pump_tolerance``,
      followed by one more jump from the pump's last rounded point. The
      pump is fused with the jump (cuOpt's extended pump): each rounding
      that is not feasible gets a jump of ``pump_jump_iterations`` steps
      (default 3000; ``None``: 3000 with JAX; with numba, whose steps are
      single moves, ``5·(n + m)`` and at least 10,000),
      within ``pump_jump_share`` of the pump's time, and when the pump
      cycles the jump's near miss becomes its next rounding. They
      share ``heuristic_seconds`` (default 10; ``None``: 20% of
      ``max_seconds``, or no time limit without one). In the tree, fix-and-propagate also runs at
      every ``heuristic_every``-th node where plain rounding fails. Their
      incumbents are marked ``"P"``, ``"J"`` and ``"F"`` in the log and
      history. A jump that ends near-feasible is handed to ``repair`` (a
      backtracking search with propagation over the variables around its
      violated rows), marked ``"R"``, for at most ``repair_share`` of the
      heuristic time left. RINS (``rins_seconds``, default 10 s a call;
      ``0``: off; ``None``: 5 s) runs once at the root and every
      ``rins_every`` nodes once there is an incumbent, and after each new
      incumbent from another source: the integers on which the incumbent
      and the node's LP point agree are fixed, and the rest is searched for
      a better solution, marked ``"N"``. With ``submip_solver="highs"``
      (default) HiGHS searches it on a CPU thread (``_SubMipLane``, one
      job at a time, pruned at the cutoff) while the device keeps solving
      node LPs; an integer agrees when the point is within
      ``rins_tolerance`` (default 0.1) of the incumbent, since a PDHG point
      stops short of the vertex (a binary at 0.9997). A call that finds the
      lane busy waits for it, and between nodes the lane takes the latest
      node point. With ``submip_solver="jaddle"`` this branch and bound
      searches it on the main thread (``_BranchAndBound.jaddle_rins``,
      reusing the compiled LP solver, up to ``rins_nodes`` nodes, at the
      integrality tolerance). ``rins_nodes`` doubles after a call that
      improves the incumbent and halves after one that does not (within
      50–2000), and those calls stop while they have taken more than
      ``heuristic_effort`` (0.3) of the elapsed time
      (``heuristic_effort=0``: no RINS).
    * **Crossover** (``crossover_seconds``, default 10 s a call; ``0``:
      off; needs ``submip_solver="highs"``). Every feasible point the
      search checks, improving or not, joins a pool of the 10 best with
      distinct integers (the background jumps give many). When the sub-MIP
      lane is free, the incumbent and ``crossover_solutions − 1`` (default
      2) other pool points drawn at random have the integers on which they
      all agree fixed, and HiGHS searches the rest, marked ``"C"``, as
      SCIP's crossover and cuOpt's recombiners do. A neighbourhood must fix
      at least ``crossover_min_fixed`` (default 0.5) of the integers, and
      none is searched twice at one cutoff. It needs no LP point, so PDHG's
      interior points cost it nothing. RINS and crossover take turns on the
      lane. Results: ``rins_calls``, ``rins_finds``, ``crossover_calls``,
      ``crossover_finds`` and ``submip_lane_seconds``.
    * **Proximity search** (``proximity_seconds``, default 10 s a call;
      ``0``: off; needs ``submip_solver="highs"``), as Fischetti and
      Monaci's. HiGHS searches the whole base box with the objective as a
      row, ``c·x`` below the cutoff, for the point nearest the incumbent
      (the L1 distance over the integers at a bound, all of a binary's),
      stops at its first point, and re-solves the continuous variables for
      the true objective with the integers fixed; marked ``"X"``. It needs
      no LP point, so it helps where PDHG's points guide RINS poorly. One
      call per incumbent and cutoff; it takes turns on the lane with RINS
      and crossover. Results: ``proximity_calls``, ``proximity_finds`` and
      ``proximity_empty`` (calls HiGHS showed had no better point).
    * **Before the first incumbent** (needs ``submip_solver="highs"``):
      RINS, proximity search and crossover start from an incumbent, so
      until there is one the lane runs, in turn, HiGHS's MIP solver on the
      whole root box stopped at its first point (``first_solution_seconds``,
      default 0: off; each call twice as long as the last, as HiGHS starts
      over; marked ``"W"``) and, with ``rens_seconds``, RENS around the
      latest node LP point (an integer within ``rins_tolerance`` of a whole
      number fixed there, at least 30% of them; marked ``"E"``). With the
      lane, the root's RENS goes to it too, instead of a Jaddle
      sub-search. On piperout-08 every RENS neighbourhood is infeasible,
      while HiGHS on the whole MIP finds its first point, the optimum, at
      20 s. Results: ``first_solution_calls``, ``first_solution_finds``,
      ``rens_calls`` and ``rens_finds``.
    * **Cutoff probing** (``cutoff_probing``, on by default), on
      ``cutoff_probing_threads`` CPU threads (default 4) of its own. Once
      there is an incumbent, every free integer of the root box is tried
      on the side its root LP vertex is not on (a binary at 0: ``x = 1``),
      by HiGHS's dual simplex over the search's rows (cuts included) from
      the LP's basis, for up to ``cutoff_probing_iterations`` pivots
      (default 1,000), stopped once the side's dual objective passes the
      cutoff. A side that cannot get below the cutoff holds no better
      solution, so the integer is fixed in the root box for the rest of
      the search: strong branching on every integer with the cutoff as the
      question. The largest root reduced costs go first. One round per
      cutoff, of at most ``cutoff_probing_seconds`` (default 60 s) and a
      fifth of the time left, and another at the same cutoff while a round
      fixes at least 5% of what it probed. A round that shows the box
      empty proves the incumbent optimal. On eil33-2, at the optimum's
      cutoff, reduced costs fix 1,005 of 4,516 binaries, probing 4,396 in
      two rounds. Results: ``cutoff_probing`` (``rounds``, ``probed``,
      ``lps``, ``pivots``, ``fixed``, ``seconds``, ``empty``).
    * **Objective levels** (``objective_levels``, on by default). When
      every nonzero cost is on at most 3 integer columns with finite bounds
      and the box holds at most 10,000 assignments of them (levels), HiGHS
      searches the levels on a CPU thread of its own, lowest objective
      first from the tree's bound up (the levels below it are empty
      already), each with its columns fixed in the root box: for
      ``objective_level_seconds`` (default 2 s) at first, twice as long at
      each new try. A level shown infeasible drops out, and the bound is
      at least the lowest level left (``objective_level_table``); a point
      found is checked as any heuristic's (``"L"``), and the lowest level
      that holds one is optimal. The search ends once no level below the
      cutoff is left. The LP cannot see such levels empty: on 30n20b8
      (``100·R1 + 51·R2``) it stays at 151 deep in the tree, while HiGHS's
      presolve shows each level below the optimum, 302, empty in at most
      2 s. Results: ``objective_levels`` (``None`` when off or not
      applicable; else the ``columns``, ``levels``, those shown ``empty``,
      ``jobs``, ``finds``, ``seconds`` and the ``floor`` reached).
    * **Round and repair** (``round_repair``, off by default). The root LP
      point (first its exact vertex, when the root is crossed over for RENS
      or ``root_crossover``), the LP points of the root's two strong
      children of the variable it branches on, and every
      ``round_repair_every`` (``None``: ``heuristic_every``) node's LP
      point have their integers rounded up, to nearest and down
      (``integer_roundings``; near-integral PDHG values are snapped first,
      by ``100 · lp_tolerance``). Each rounding that propagation does not
      rule out is tried as an incumbent, marked ``"U"``: ``try_incumbent``
      solves for the continuous variables with a PDHG LP (as above), within
      ``round_repair_seconds`` (10) a call and, in the tree,
      ``heuristic_effort``. Rounding up suits integers that count units
      for continuous variables (on buildingenergy the sizing variable's up
      child is where the good points are).
    * **RENS** (``rens_seconds``, default 0: off; ``None`` gives 5s for
      MIPs with at most 10,000 nonzeros, else off). After the root heuristics,
      the integers that are integral at an exact root vertex (the cut
      loop's last crossover, else one crossed over for RENS when
      ``root_crossover`` allows) are fixed and the fractional ones
      restricted to their floor and ceiling, and that sub-MIP is searched
      by a Jaddle sub-search (``sub_search``, up to ``rens_nodes`` nodes),
      marked ``"E"``. It needs no incumbent.
    * **Early root jumps** (``root_jump_seconds``, default 60; ``0``:
      off). With background workers and the numba jump, each jump lane
      (``jump_threads``) starts the feasibility jump from the bounds nearest
      zero, with its own seed, before the root LP, as HiGHS jumps before its
      first LP: the jump needs no LP point, and it runs on the CPU while the
      device solves the root. They stop at the first incumbent. On
      neos-5106984-jizera (299k columns, 391k rows after presolve) the jump
      needs 30-45 s from zero, where the root's own jumps get about 5 s
      each, and only after the root LP.
    * **Block decomposition** (``block_decomposition``, default off;
      ``block_decomposition_seconds``, default 300). When a few long
      ``<=`` rows are all that join the MIP's blocks (``linking_blocks``),
      negotiated congestion (``negotiate_blocks``, the rip-up-and-reroute
      of VLSI routers) runs before the root LP, in the foreground: each
      block solved alone by HiGHS (its LP relaxation, warm from the last
      sweep, with the MIP when that is fractional), with prices on the
      linking rows that rise where blocks overuse them. It needs no LP of
      the whole MIP. A point it finds is the first incumbent, marked
      ``"K"``; then the root LP and the tree go on from it. On stp3d (24
      nets joined by 2,171 node capacity rows, no presolve), where the jump
      and roundings find nothing, it found a point in 34 s (37-146 s in
      runs beside others); after presolve it has not settled within three
      minutes. ``result["block_decomposition"]`` reports its sweeps,
      the overused rows at the last one, and its seconds.
    * **Trivial heuristic** (``trivial_heuristic``, on). Before the root
      LP, every variable at zero, at its lower bound, at its upper bound,
      and at the bound its locks allow (``trivial_points``), one matvec
      each; a feasible one is the first incumbent, marked ``"Z"``.
    * **Background heuristics** (``parallel_heuristics``, on by default).
      The numba improving jump runs on a background thread, beside the
      node LPs, instead of between them. The search hands a slice over when
      the thread is free and checks the points of finished slices between
      nodes (marked ``"J"``). The root cut rounds do not wait for
      heuristics either: the first root LP's fix-and-propagate and its two
      jumps (LP point, bounds nearest zero), and each later round's
      fix-and-propagate, run on background threads beside the rounds
      (``"P"``, ``"J"``), and the rounds collect their points. With
      ``parallel_rounding`` (off by default; independent of
      ``fix_and_propagate``) every node's LP point is rounded the same way beside the node
      LPs (``"P"``; a point is skipped while the thread is still busy with
      an earlier one; ``result["background_roundings"]``). When the
      rounding fails, the same job runs an improving feasibility jump
      from the node point in its box, below the cutoff, for
      ``node_jump_seconds`` (default 1; ``0``: off; ``"J"``;
      ``result["background_jumps"]``): node points are often far from
      integral (ns1830653: about 200 of 458 integers at 0.5), where
      rounding always failed but a 1 s jump beat the incumbent from 2 of
      30. Each slice runs 5 s from the incumbent of its
      time; a slice that finds nothing better is not repeated until the
      incumbent changes. The device heuristics (bulk rounding, the pump,
      the tree heuristics) and RINS (a Jaddle sub-search) stay on the main
      thread. The jobs use NumPy only, never JAX, and stop within about a
      second when the search ends. A run then depends on timing, so it is
      no longer repeatable; turn it off for that.
    * **Dives** (``dive_every``; default 0: off). LP-based dives
      at the root and every ``dive_every`` nodes: round a fractional integer
      (fractional, guided or pseudocost rule), propagate, re-solve the LP
      warm, up to ``dive_depth`` times; an integral end point is marked
      ``"D"``.
    * **GPU tree heuristics** (``tree_heuristics``, off by default). Once
      there is an incumbent, every ``tree_heuristic_every`` (10) nodes, the
      node's LP point is bulk rounded (``bulk_starts`` starts, in the node
      box, with the objective row at the cutoff), marked ``"B"``; if that
      finds nothing better, ``feasibility_jump`` runs from it for up to
      ``tree_jump_iterations`` steps (``None``: as ``pump_jump_iterations``)
      in the root box with the row
      ``c·x <= cutoff`` added, so it looks only for better points, marked
      ``"J"``. They run while they have taken at most
      ``tree_heuristic_share`` (0.05) of the elapsed time, each call stops
      once it has used the share left (but gets at least 1 s), and they skip a
      rounding of the LP point already tried at the same cutoff. A find
      also triggers RINS at the next node.
    * **Exact nodes** (``exact_nodes``, default 16; ``0``: off). With
      batched search, ``exact_nodes`` CPU threads solve the LPs of the
      nodes that hold the global bound exactly with HiGHS's dual simplex,
      while the device solves the rest in batches (``_ExactLane``). Once
      the heap is in bound order (after the first incumbent), the nodes
      within ``exact_band`` (default 0.25) of the gap above the global
      bound are kept for the threads, lowest first (two queued a thread),
      and the device never takes them: a truncated LP rarely lifts such a
      node's bound, and splitting it only doubles the nodes that hold the
      global bound (on map06 with the lane taking only the heap's top, the
      device split the bottom of the queue faster than the lane solved it,
      and the bound did not move). The device takes the rest of the queue,
      which feeds the heuristics and prunes near the cutoff. With
      ``exact_band=0`` a thread instead takes the heap's top whenever it has
      room, and the device the rest. With ``exact_band="auto"`` the band
      widens as the gap closes: once a batch it is 0.25 over the relative
      gap, within [0.001, 1] (a quarter of the gap at a 100% gap, half at
      50%, all of it from 25% down), so its width stays near 25% of the
      incumbent's value.
      An objective near 0 keeps the relative gap large and the band at its
      floor. A queue wider than the threads can solve waits for them unless
      ``exact_overflow_share`` hands its top to the device. The
      ``exact_nodes`` stats report
      the band at the end (``band``) and its range (``band_min``,
      ``band_max``). A node goes to the lane at most once.
      HiGHS re-solves from the parent's optimal basis (below), else from
      the thread's last basis, within ``exact_node_seconds`` (10) a node;
      the node is then evaluated as a batch member, with the exact solution in place of the PDHG one: its bound is the exact
      duals' Lagrangian bound (rigorous at any duals), it branches on the
      vertex, and its children's bounds come from those duals over their
      boxes. A node HiGHS proves infeasible closes; one it does not finish
      goes back to the queue for the device. Truncated PDHG node LPs rarely
      lift a node's bound above its parent's, so without exact LPs the
      global bound can stall while the tree grows (map06: 7,800 nodes
      moved it 3 units); a map06 node took HiGHS 1.1–1.6 s.
      Each lane node's optimal basis (and each strong side's) is kept for
      its children (``exact_basis_memory``, default 2e9: the bytes the
      kept bases may take, about one a column and row each, oldest dropped
      first; ``0``: off), and a child's LP starts from it instead of from
      whatever node its thread solved last, as branch and bound warm-starts
      dual simplex. On presolved map06 a child took 870–3,100 pivots
      (0.5–1.6 s) from its parent's basis against 3,400–5,200 (1.8–2.9 s)
      from another node's. A device node's children keep the basis it
      would have started from.
      The root is not a lane node, so it gets exact strong branching of its
      own (``exact_root_candidates``, default 128; ``None``: every
      fractional integer; ``0``: off, and the PDHG root sweep,
      ``root_strong_candidates``, runs instead). Just before the root
      branches, a lane thread solves the root LP exactly, and the
      fractional integers at its vertex with the largest
      ``impact · f(1 − f)`` are dealt out to all the threads, each side
      re-solved from the root's basis for up to ``exact_root_iterations``
      pivots (``None``, the default: no limit), all within
      ``exact_root_seconds`` (60, and at most a tenth of the time left).
      The root then branches as a lane node does, on its vertex: empty
      sides fix their variable, the bound rises to the best lesser child,
      solved sides become their children's LPs and bases, and the gains
      seed both the lane's exact pseudocosts and the device's. On map06 a
      side takes about 0.5–1.6 s, so its 69 fractional integers take about
      10 s on 16 threads. A strong-branching side stopped by HiGHS's time
      limit keeps its duals' bound, as one stopped by the pivot limit
      does. ``result["root_exact_strong"]`` gives the candidates, side
      LPs, empty sides (``fixed``), the root LP's pivots, the time and the
      exact root bound.
      Each lane node also chooses its branching variable by strong
      branching in the same job (``exact_strong_candidates``, default 8;
      ``0``: off): of the fractional integers whose exact pseudocosts have
      fewer than ``reliability_threshold`` observations a side, those with
      the largest pseudocost product are split, and each side is re-solved
      from the node's basis for at most ``exact_strong_iterations`` dual
      simplex pivots (``"auto"``, the default: twice the mean pivots of the
      lane's node LPs so far, within 10 to 500, as SCIP limits its strong
      branching LPs; a number: that many;
      ``result["exact_nodes"]["strong_iterations"]`` gives the limit at the
      end). With 5,000, most sides ran to optimality and strong branching
      took 60-85% of the lane's time; on ns1830653 and 30n20b8 (60 s, CPU)
      ``"auto"`` gave 1.1-1.7 times the nodes. With
      ``exact_strong_lookahead`` (default ``None``: off) the candidates
      after that many in a row that did not beat the best gain product so
      far are not split, as SCIP's strong branching looks ahead (with 3,
      30n20b8 went from 14 to 10 sides a node, ns1830653 from 4.2 to 2.4).
      With ``exact_strong_cutoff`` (default ``False``) a side stops once
      its dual objective passes the cutoff (HiGHS's ``objective_bound``):
      its duals' bound closes it either way. A side stopped early still has feasible duals,
      so its Lagrangian bound holds. The children then go through
      ``apply_strong`` as device strong branching's do (a side HiGHS proves
      infeasible fixes the variable, and the node's bound rises to the best
      lesser child bound), the gains feed the lane's own exact pseudocosts
      (truncated node LPs underestimate gains), and a child whose side
      solved to optimal keeps that vertex, so the lane evaluates it without
      a solve. At map06's root, exact strong branching found +14.6 for the
      best variable against +1.5 for the median one.
      When the band holds more than ``_EXACT_OVERFLOW_KEEP`` (4) nodes a
      thread, the device takes its top (``exact_overflow_share``, default
      0, off; else up to that share of each batch): those nodes leave
      the band and their LPs run to convergence over as many batches as it
      takes (continuous batching, within ``exact_overflow_epochs``;
      ``None``, the default: 200), instead of the truncated ``batch_epochs``,
      so the device lifts band nodes' bounds many at a time while the
      threads take the lowest.
      ``result["exact_nodes"]`` holds the nodes solved, proved infeasible
      and not finished, the threads' seconds, the bound they gained over
      the nodes' own bounds, the strong-branching LPs and their seconds,
      the children evaluated from their strong-branching vertex, the node
      LPs' dual simplex ``pivots`` and those started from a kept basis
      (``warm_bases``), and the band's nodes sent to the device
      (``overflow_nodes``) and those whose LP converged there (``overflow_converged``).
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
    and strong branching's child LPs. Node LPs solve to ``node_lp_tolerance``
    (default 1e-3; ``None``: ``lp_tolerance``, by default 1e-4; at 1e-3 with
    float32 node LPs, small MIPs can end ``"unresolved"``) and check
    convergence every 64 iterations (``node_lp_options=None`` means
    ``{"iterations_per_epoch": 64}`` unless ``lp_options`` sets it; ``{}``:
    ``jaddle_linear.solve``'s 256).

    ``presolve=True`` presolves with HiGHS, runs the branch and bound on the
    reduced MIP, and maps the incumbent back with HiGHS's postsolve. The
    result then also has ``"postsolve_feasible"``: whether the postsolved
    point passes the feasibility check on the model as given.
    ``highs_options`` (a dict) is applied to the HiGHS instance that reads
    and presolves.

    Stops when the gap closes, the tree is empty, or at ``max_nodes``,
    ``max_seconds`` (wall clock, from entry) or Ctrl-C. With
    ``presolve=True``, HiGHS's presolve does not count towards
    ``max_seconds`` or ``"seconds"``, as in ``jl.solve_with_presolve``: it is
    timed on its own (``"presolve_seconds"``).

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
      ``status == "optimal"``. ``"interrupted"`` is Ctrl-C: as with
      ``jl.solve``, the first press ends the search with the best point so
      far and a valid bound (it stops as a time limit would, the LP being
      solved at its next chunk); a second press raises
      ``KeyboardInterrupt``.
    * ``"bound_rigorous"``: False when a bound used a dropped reduced cost
      (see ``lagrangian_bound``).
    * ``"nodes"`` (nodes finished: with batched node LPs, a node whose LP
      goes back to the queue unconverged counts once, when it finishes;
      the returns are ``"batch_requeues"``), ``"infeasible_nodes"``
      (dropped by propagation or proven infeasible), ``"integral_splits"``
      (nodes whose LP point was integral but failed the feasibility check
      or came from an LP that did not converge: they are split on a free
      integer, preferably one in a row the point breaks, rather than
      closed on their LP bound, which would hold the global bound there),
      ``"unresolved_nodes"`` (such nodes with every integer fixed, closed;
      their bound still counts), ``"lp_epochs"``, ``"seconds"``.
    * ``"history"``: one dict per new incumbent (``seconds``, ``nodes``,
      ``objective``, ``bound``, ``source``: ``"H"`` rounding, ``"P"``
      fix-and-propagate, ``"J"`` feasibility jump, ``"R"`` repair, ``"F"``
      feasibility pump, ``"U"`` round and repair, ``"N"``
      RINS (``rins_seconds``), ``"T"`` tree, ``"S"`` the user's start
      (``mip_start``) or ``"1"`` 1-opt (``one_opt``));
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
    * ``"parallel_heuristics"``: the background jobs
      (``parallel_heuristics``) collected (``"jobs"``), their run time
      (``"seconds"``, beside the search's own), those whose point improved
      the incumbent (``"improvements"``) and the improving-jump slices
      started (``"jump_slices"``).
    * ``"node_crossovers"``, ``"node_crossover_vertices"``,
      ``"node_crossover_integral"``, ``"node_crossover_seconds"``: node
      crossovers (``node_crossover``), those that proved a vertex optimal,
      the vertices that were integral where the PDHG point was not, and
      their time.
    * ``"coefficients_tightened"``: coefficients cut by coefficient
      tightening; ``"tree_cuts_added"``: cuts added in the tree;
      ``"tree_cuts_dropped"``: violated tree cuts left out for want of room
      in the pool (``tree_cut_rows``, ``tree_cut_nnz``).
    * ``"strong_branch_nodes"``, ``"strong_branch_lps"``,
      ``"strong_branch_seconds"``, ``"strong_fixings"``: nodes that
      strong-branched, the child LPs they solved, the time taken and the
      variables fixed because one child was closed; ``"root_strong_lps"``,
      ``"root_strong_seconds"``: the root sweep's child LPs and time
      (``root_strong_candidates``); ``"root_crossover"``: whether the root
      bound and reduced-cost fixing used exact crossover duals;
      ``"round_repair_calls"``, ``"round_repair_finds"``,
      ``"round_repair_seconds"``: round and repair (``round_repair``) calls,
      those that improved the incumbent, and their time.
    * ``"cuts"``: the number of cuts in the final LP, by family
      (``"gomory"``, ``"cover"``, ``"mir"``, ``"flowcover"``); ``"root_bound_before_cuts"``: the root bound
      before them (``None`` when no cut round ran).
    """
    start = time.time()
    deadline = None if max_seconds is None else start + max_seconds
    if node_lp_options is None:
        # Node LPs check convergence every 64 iterations (jl.solve: 256),
        # unless lp_options sets it for every LP.
        node_lp_options = (
            {}
            if "iterations_per_epoch" in (lp_options or {})
            else {"iterations_per_epoch": 64}
        )
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
        max_tree_cuts_per_round=max_tree_cuts_per_round,
        tree_cut_rows=tree_cut_rows,
        tree_cut_nnz=tree_cut_nnz,
        tree_cut_points=tree_cut_points,
        tree_cut_async=tree_cut_async,
        tree_cut_rounds=tree_cut_rounds,
        tree_cut_share=tree_cut_share,
        tree_cut_max_nnz=tree_cut_max_nnz,
        fix_and_propagate=fix_and_propagate,
        feasibility_jump=feasibility_jump,
        jump_iterations=jump_iterations,
        repair=repair,
        repair_max_epochs=repair_max_epochs,
        rins_seconds=rins_seconds,
        rins_every=rins_every,
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
        probing_passes=probing_passes,
        probing_max_implications=probing_max_implications,
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
        node_precision=node_precision,
        jump_backend=jump_backend,
        parallel_heuristics=parallel_heuristics,
        symmetry=symmetry,
        restarts=restarts,
        restart_presolve=restart_presolve,
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
        mir_cuts=mir_cuts,
        clique_cuts=clique_cuts,
        zero_half_cuts=zero_half_cuts,
        cut_age=cut_age,
        max_cut_rows=max_cut_rows,
        cut_pool_size=cut_pool_size,
        gomory_candidates=gomory_candidates,
        inout_separation=inout_separation,
        root_redcost_propagation=root_redcost_propagation,
        global_redcost_fixing=global_redcost_fixing,
        cut_pool_propagation=cut_pool_propagation,
        plunge_child=plunge_child,
        implied_integer_cuts=implied_integer_cuts,
        separation_backend=separation_backend,
        separation_device=separation_device,
        node_bound_stop=node_bound_stop,
        dual_repair=dual_repair,
        node_lp_tolerance=node_lp_tolerance,
        heuristic_effort=heuristic_effort,
        reliability_threshold=reliability_threshold,
        strong_candidates=strong_candidates,
        strong_epochs=strong_epochs,
        strong_max_depth=strong_max_depth,
        strong_screen=strong_screen,
        strong_backend=strong_backend,
        root_strong_candidates=root_strong_candidates,
        root_strong_epochs=root_strong_epochs,
        root_crossover=root_crossover,
        round_repair=round_repair,
        round_repair_every=round_repair_every,
        round_repair_seconds=round_repair_seconds,
        mip_start=mip_start,
        implied_bound_cuts=implied_bound_cuts,
        probing_cliques=probing_cliques,
        implied_variable_bounds=implied_variable_bounds,
        max_aggregation=max_aggregation,
        aggregation_width=aggregation_width,
        cmir_per_row=cmir_per_row,
        cut_score_weights=cut_score_weights,
        exact_nodes=exact_nodes,
        exact_node_seconds=exact_node_seconds,
        exact_band=exact_band,
        exact_strong_candidates=exact_strong_candidates,
        exact_strong_iterations=exact_strong_iterations,
        exact_overflow_share=exact_overflow_share,
        exact_overflow_epochs=exact_overflow_epochs,
        exact_basis_memory=exact_basis_memory,
        exact_strong_lookahead=exact_strong_lookahead,
        exact_strong_cutoff=exact_strong_cutoff,
        exact_root_candidates=exact_root_candidates,
        exact_root_seconds=exact_root_seconds,
        exact_root_iterations=exact_root_iterations,
        cutoff_probing=cutoff_probing,
        cutoff_probing_threads=cutoff_probing_threads,
        cutoff_probing_seconds=cutoff_probing_seconds,
        cutoff_probing_iterations=cutoff_probing_iterations,
        one_opt=one_opt,
        objective_propagation=objective_propagation,
        plunge_gap=plunge_gap,
        branch_implied=branch_implied,
        near_integral_tolerance=near_integral_tolerance,
        tree_pool_every=tree_pool_every,
        parallel_rounding=parallel_rounding,
        node_jump_seconds=node_jump_seconds,
        pipeline_batches=pipeline_batches,
        gpu_dive_every=gpu_dive_every,
        gpu_dive_steps=gpu_dive_steps,
        gpu_dive_epochs=gpu_dive_epochs,
        gpu_dive_share=gpu_dive_share,
        gpu_dive_snap=gpu_dive_snap,
        gpu_dive_fix_share=gpu_dive_fix_share,
        gpu_dive_crossover=gpu_dive_crossover,
        exact_plunges=exact_plunges,
        exact_plunge_share=exact_plunge_share,
        jump_threads=jump_threads,
        jump_restarts=jump_restarts,
        root_jump_seconds=root_jump_seconds,
        block_decomposition=block_decomposition,
        block_decomposition_seconds=block_decomposition_seconds,
        trivial_heuristic=trivial_heuristic,
        submip_solver=submip_solver,
        rins_tolerance=rins_tolerance,
        crossover_seconds=crossover_seconds,
        crossover_solutions=crossover_solutions,
        crossover_min_fixed=crossover_min_fixed,
        proximity_seconds=proximity_seconds,
        first_solution_seconds=first_solution_seconds,
        objective_levels=objective_levels,
        objective_level_seconds=objective_level_seconds,
        cut_round_lp=cut_round_lp,
        root_lp=root_lp,
    )
    return _solve_model(
        model, presolve, precision, highs_options, feasibility_tolerance, settings
    )


def _solve_model(model, presolve, precision, highs_options, feasibility_tolerance, settings):
    """The search of ``solve``, with or without HiGHS's presolve."""
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
                        mip.implied,
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
    # HiGHS cannot map a point into the presolved MIP, so the user's start
    # is completed and checked here, and its value becomes the reduced
    # search's cutoff.
    mip_start = check_mip_start(settings["mip_start"], original)
    settings = dict(settings, mip_start=None)
    start_x = None
    if mip_start is not None:
        start_x = _complete_start(
            original, mip_start, feasibility_tolerance, settings["round_repair_seconds"]
        )
        if settings["verbose"] and start_x is None:
            print("mip_start: not feasible, ignored")
    presolve_started = time.time()
    status = _presolve(highs)
    PS = highspy.HighsPresolveStatus
    # As in ``jl.solve_with_presolve``: presolve is timed on its own
    # (``result["presolve_seconds"]``) and not counted in ``max_seconds`` or
    # ``result["seconds"]``, so the clock and the deadline move by it.
    presolve_seconds = time.time() - presolve_started
    start += presolve_seconds
    settings = dict(
        settings,
        start=start,
        deadline=(
            None if settings["deadline"] is None else settings["deadline"] + presolve_seconds
        ),
    )
    if settings["verbose"]:
        print(f"HiGHS presolve: {status.name} ({presolve_seconds:.1f}s)")

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
            "integral_splits": 0,
            "near_integral": {"tries": 0, "finds": 0},
            "background_roundings": 0,
            "background_jumps": 0,
            "gpu_dives": {
                "runs": 0, "dives": 0, "steps": 0, "leaves": 0, "finds": 0,
                "seconds": 0.0,
            },
            "exact_plunges": {"plunges": 0, "nodes": 0, "max depth": 0, "seconds": 0.0},
            "block_decomposition": None,
            "exact_nodes": {
                "solved": 0,
                "infeasible": 0,
                "failed": 0,
                "seconds": 0.0,
                "bound_gain": 0.0,
                "strong_lps": 0,
                "strong_seconds": 0.0,
                "children_reused": 0,
                "pivots": 0,
                "strong_iterations": 0,
                "warm_bases": 0,
                "overflow_nodes": 0,
                "overflow_converged": 0,
                "band": 0.0,
                "band_min": 0.0,
                "band_max": 0.0,
                "band_scale": _EXACT_BAND_GAP_SCALE,
                "band_scale_min": _EXACT_BAND_GAP_SCALE,
                "band_scale_max": _EXACT_BAND_GAP_SCALE,
                "band_tunes": 0,
                "band_rates": (None, None),
                "band_cap_nodes": None,
                "band_capped_batches": 0,
            },
            "inout_cuts": 0,
            "inout_rounds": 0,
            "tree_cut_refreshes": 0,
            "tree_cut_async_jobs": 0,
            "exact_bounds_reused": 0,
            "tree_pool_refreshes": 0,
            "tree_cut_requeued": 0,
            "tree_cut_seconds": 0.0,
            "tree_cut_pool_hits": 0,
            "tree_cut_skipped": 0,
            "tree_cut_families": {},
            "cut_pool": 0,
            "global_redcost_fixed": 0,
            "cut_pool_rows": 0,
            "lp_epochs": 0,
            "tree_seconds": 0.0,
            "seconds": time.time() - start,
            "history": [],
            "cuts": {
                "gomory": 0,
                "cover": 0,
                "mir": 0,
                "flowcover": 0,
                "clique": 0,
                "zerohalf": 0,
                "impliedbound": 0,
            },
            "root_bound_before_cuts": None,
            "heuristic_seconds": 0.0,
            "batches": 0,
            "objective_step": None,
            "reduced_cost_tightenings": 0,
            "dual_repairs": {"tried": 0, "improved": 0, "seconds": 0.0},
            "children_pruned": 0,
            "lp_repairs": 0,
            "lp_repairs_skipped": 0,
            "coefficients_tightened": 0,
            "tree_cuts_added": 0,
            "tree_cuts_dropped": 0,
            "symmetry": None,
            "restarts": 0,
            "restart_log": [],
            "rins_calls": 0,
            "rins_finds": 0,
            "rens_calls": 0,
            "rens_finds": 0,
            "first_solution_calls": 0,
            "first_solution_finds": 0,
            "rins_fixing_rate": 1.0,
            "batch_timing": {
                "wait_seconds": 0.0, "evaluate_seconds": 0.0, "bound_seconds": 0.0,
                "cut_refresh_seconds": 0.0, "exact_collect_seconds": 0.0,
                "exact_service_seconds": 0.0,
                "branch_seconds": 0.0, "heuristic_seconds": 0.0,
            },
            "crossover_calls": 0,
            "crossover_finds": 0,
            "proximity_calls": 0,
            "proximity_finds": 0,
            "proximity_empty": 0,
            "submip_lane_seconds": 0.0,
            "cutoff_probing": None,
            "restart_presolve": [],
            "objective_levels": None,
            "exact_cut_rounds": None,
            "mip_start_accepted": None if mip_start is None else start_x is not None,
            "one_opt_finds": 0,
            "plunges_stopped": 0,
            "implied_branchings": 0,
            "presolve": status.name,
            "presolve_seconds": presolve_seconds,
        }

    if status == PS.kInfeasible:
        return settled("infeasible")
    if status == PS.kUnboundedOrInfeasible:
        return settled("unbounded_or_infeasible")
    if status == PS.kNotReduced:
        settings["mip_start"] = mip_start
        result = _BranchAndBound(original, **settings).run()
        result["presolve"] = status.name
        result["presolve_seconds"] = presolve_seconds
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
        start_value = None
        if start_x is not None:
            # Only points better than the start by more than the gap
            # tolerances are searched for (min form, with offsets).
            start_value = float(original.c @ start_x) + original.offset
            allowance = max(
                settings["mip_abs_gap"], settings["mip_rel_gap"] * abs(start_value)
            )
            settings["initial_cutoff"] = start_value - allowance - reduced.offset
            settings["cutoff_from_start"] = True
        result = _BranchAndBound(reduced, **settings).run()
        result["presolve"] = status.name
        result["presolve_seconds"] = presolve_seconds
        result["mip_start_accepted"] = (
            None if mip_start is None else start_x is not None
        )
        reduced_x = result["x"]
        if reduced_x is None and start_x is not None:
            # Nothing better than the start: it is the answer.
            if result["status"] == "infeasible":
                # Proven: no point beats the cutoff.
                result["status"], result["converged"] = "optimal", True
                bound = settings["initial_cutoff"] + reduced.offset
            else:
                bound = original.sign * result["bound"]
                bound = start_value if np.isnan(bound) else min(bound, start_value)
            result["x"] = start_x
            result["objective"] = original.objective(start_x)
            result["bound"] = original.sign * bound
            result["gap"] = _relative_gap(start_value, bound)
            result["postsolve_feasible"] = True
            result["seconds"] = time.time() - start
            return result
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
