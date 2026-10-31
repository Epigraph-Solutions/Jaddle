"""
CVXPY as a modelling front end for Jaddle.

Three routes, all returning results the CVXPY way (``prob.status``,
``prob.value``, ``var.value``, ``constraint.dual_value``):

* ``JaddleSolver`` is a CVXPY QP solver: ``prob.solve(solver=JaddleSolver())``
  solves any problem CVXPY can put in QP form
  ``min ½xᵀPx + qᵀx  s.t.  Ax = b, Fx <= g, lower <= x <= upper``. An LP
  (``P = 0``) goes through ``jaddle_linear.solve_with_presolve`` (HiGHS
  presolve, then Jaddle, then HiGHS postsolve), a QP through
  ``jaddle_convex.quadratic_program``. A mixed-integer LP (``integer=True``
  or ``boolean=True`` variables) goes through ``jaddle_mip.solve``.
* ``JaddleConicSolver`` is a CVXPY conic solver for problems with
  second-order, PSD and exponential cones (so ``log``, ``log_det``,
  ``log_sum_exp``, ...): ``prob.solve(solver=JaddleConicSolver())``.
  They are solved by ``jaddle_linear.solve`` on cone-constrained variables
  (``jaddle.cones``).
* ``solve(prob)`` handles general convex problems. CVXPY's own route would
  rewrite them into cones (exponential, second-order), which the convex
  solver cannot project onto and which turn a smooth loss such as the logistic into many
  extra variables and rows. Instead ``compile_problem`` walks the expression
  tree and builds a ``JaddleCP`` of JAX functions for
  ``jaddle_convex.solve``: smooth atoms stay smooth, and only piecewise-linear
  ones (``abs``, ``maximum``, ``norm1``, ...) become epigraph variables and
  linear rows, and a variable defined by an equality (``y == C @ x + v``)
  is substituted out. CVXPY still checks that the problem is DCP. A problem CVXPY
  can put in QP form goes to ``JaddleSolver`` instead, unless
  ``qp_route=False``.

CVXPY is an optional dependency (``pip install jaddle[cvxpy]``) and
``import jaddle`` does not import it.

Unless given, every tolerance defaults to ``DEFAULT_TOLERANCE`` (1e-6), the
accuracy CVXPY users expect of a solver, rather than the 1e-3 of Jaddle's own
functions. On the QP route the tolerances are relative (as
``jaddle_linear.solve`` and ``jaddle_convex.quadratic_program`` test them), on
the compiled route absolute (as ``jaddle_convex.solve`` tests them).
"""

import time

import cvxpy as cp
import cvxpy.settings as s
import jax
import jax.numpy as jnp
import jax.scipy.special as jss
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import (
    DivExpression,
    MulExpression,
    multiply,
)
from cvxpy.atoms.affine.broadcast_to import broadcast_to
from cvxpy.atoms.affine.concatenate import Concatenate
from cvxpy.atoms.affine.diag import diag_mat, diag_vec
from cvxpy.atoms.affine.hstack import Hstack
from cvxpy.atoms.affine.index import index, special_index
from cvxpy.atoms.affine.promote import Promote
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.sum import Sum
from cvxpy.atoms.affine.trace import Trace
from cvxpy.atoms.affine.transpose import transpose
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.atoms.affine.vstack import Vstack
from cvxpy.atoms.elementwise.abs import abs as cvx_abs
from cvxpy.atoms.elementwise.entr import entr
from cvxpy.atoms.elementwise.exp import exp as cvx_exp
from cvxpy.atoms.elementwise.huber import huber
from cvxpy.atoms.elementwise.logistic import logistic
from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.atoms.elementwise.minimum import minimum
from cvxpy.atoms.elementwise.power import Power
from cvxpy.atoms.log_sum_exp import log_sum_exp
from cvxpy.atoms.max import max as cvx_max
from cvxpy.atoms.min import min as cvx_min
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.norm_inf import norm_inf
from cvxpy.atoms.pnorm import Pnorm
from cvxpy.atoms.quad_form import QuadForm
from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.constraints import SOC, Equality, ExpCone, Inequality, NonNeg, NonPos, Zero
from cvxpy.constraints.psd import SvecPSD
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import SolverStats
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from cvxpy.utilities.psd_utils import TriangleKind

import jaddle.jaddle_convex as jc
import jaddle.jaddle_linear as jl
import jaddle.jaddle_optimisers as jo
from jaddle.cones import Cones
from jaddle.jaddle_basic_types import LP, JaddleCP, SaddleState, scipy_to_bcoo

DEFAULT_TOLERANCE = 1e-6

# The stepping jaddle_convex.solve gets unless overridden: the best general
# configuration of benchmarks/benchmark_convex.py. Its own default (alternating
# steps of fixed size) diverges on the free, curvature-free columns CVXPY's
# QP form introduces.
QP_DEFAULTS = dict(
    update_mode="extragradient",
    step_size=0.5,
    restarts=True,
    epochs_per_restart=5,
)

# The compiled route adds gradient scaling ("eg_ad_rs_gs"), since a compiled
# problem is not equilibrated.
COMPILED_DEFAULTS = dict(QP_DEFAULTS, gradient_scaling=True)

# Below this argument entr(u) = -u log u is compiled as its tangent line, so
# the iterates, which respect only the box, see a finite value and gradient.
ENTR_FLOOR = 1e-8

_LIMIT_STOPS = ("max_epochs", "time_limit", "interrupted", "primal_stall")


def _status(result):
    """The CVXPY status of a Jaddle result: ``OPTIMAL`` when certified,
    ``OPTIMAL_INACCURATE`` when only the presolved LP certified (postsolve
    can enlarge a first-order point's residuals), ``USER_LIMIT`` for a budget
    or stall stop (the point is still returned)."""
    stop = result["stop_reason"]
    if stop == "presolve_infeasible":
        return s.INFEASIBLE
    if stop == "presolve_unbounded_or_infeasible":
        return s.INFEASIBLE_OR_UNBOUNDED
    if result["converged"]:
        return s.OPTIMAL
    if stop == "certificate":
        return s.OPTIMAL_INACCURATE
    if stop in _LIMIT_STOPS:
        return s.USER_LIMIT
    return s.SOLVER_ERROR


def _with_tolerances(options, keys):
    """``options`` with each tolerance in ``keys`` set to ``options["tol"]``
    (default ``DEFAULT_TOLERANCE``) unless given by name."""
    options = dict(options)
    tol = options.pop("tol", None)
    for key in keys:
        options.setdefault(key, DEFAULT_TOLERANCE if tol is None else tol)
    return options


_QP_TOLERANCES = (
    "primal_feasibility_tolerance",
    "dual_feasibility_tolerance",
    "dual_gap_tolerance",
)
_COMPILED_TOLERANCES = (
    "primal_grad_norm_tolerance",
    "primal_feasibility_tolerance",
    "complementarity_slack_tolerance",
)


class _EqualByType:
    """Solver instances of one class compare equal. CVXPY keeps a problem's
    compiled form and its solver cache (which holds the warm start) only while
    the solver it is given equals the last one, so without this each
    ``solver=JaddleSolver()`` would recompile and start cold."""

    def __eq__(self, other):
        return type(other) is type(self)

    def __hash__(self):
        return hash(type(self))


# %%
# Route 1: a CVXPY QP solver


class JaddleSolver(_EqualByType, QpSolver):
    """
    Jaddle as a CVXPY QP solver: ``prob.solve(solver=JaddleSolver(), **opts)``.

    An LP is solved by ``jaddle_linear.solve_with_presolve``; ``presolve=False``
    solves it with ``jaddle_linear.solve`` directly instead. A QP is solved by
    ``jaddle_convex.quadratic_program``, with ``QP_DEFAULTS`` for its
    stepping. The other keyword arguments of ``prob.solve`` go to whichever
    of these runs (``verbose`` included). ``tol`` sets the three relative
    tolerances (``primal_feasibility_tolerance``,
    ``dual_feasibility_tolerance``, ``dual_gap_tolerance``) that are not given
    by name; it defaults to ``DEFAULT_TOLERANCE``.

    ``warm_start=True`` (CVXPY's default) starts a QP, or an LP with
    ``presolve=False``, from the previous solve's point when the problem's
    sizes are unchanged. (The presolved LP's variables are not the problem's,
    so the presolve route always starts cold.) Solving the same problem with
    another solver in between clears CVXPY's cache, and with it the point.

    ``prob.solver_stats.extra_stats`` is Jaddle's result dict (certificate,
    epochs, stop reason, presolve sizes).

    A mixed-integer LP (variables declared ``integer=True`` or
    ``boolean=True``) is solved by ``jaddle_mip.solve``, Jaddle's branch and
    bound, which is highly experimental: check its answers against an
    established MIP solver. The keyword arguments of ``prob.solve`` go to
    ``jaddle_mip.solve`` (``max_seconds``, ``mip_rel_gap``, ...), and ``tol``
    sets its ``lp_tolerance``. The status is ``OPTIMAL`` when the gap closes,
    ``USER_LIMIT`` when a time or node limit stops the search (with the best
    point, if any) and ``OPTIMAL_INACCURATE`` when the search ends with an
    incumbent but the gap open. There are no duals (``dual_value`` is
    ``None``), and ``extra_stats`` holds ``gap``, ``nodes`` and the proven
    ``bound``, the latter for CVXPY's minimisation form (negated for a
    ``Maximize``, objective constant excluded). A mixed-integer QP raises ``SolverError``. Run in float64
    (``jaddle_optimisers.configure_jax("float64")``) for the default
    tolerances.
    """

    MIP_CAPABLE = True
    BOUNDED_VARIABLES = True

    def name(self):
        return "JADDLE"

    def import_solver(self) -> None:
        import jaddle  # noqa: F401

    def solve_via_data(
        self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None
    ):
        options = _with_tolerances(solver_opts, _QP_TOLERANCES)
        options["verbose"] = verbose
        presolve = options.pop("presolve", True)

        P, q = data[s.P], np.asarray(data[s.Q], dtype=np.float64)
        n = q.shape[0]
        lower, upper = data[s.LOWER_BOUNDS], data[s.UPPER_BOUNDS]
        lower = np.full(n, -np.inf) if lower is None else np.asarray(lower, float)
        upper = np.full(n, np.inf) if upper is None else np.asarray(upper, float)
        problem = dict(
            A_eq=sp.csc_matrix(data[s.A]),
            b_eq=np.asarray(data[s.B], dtype=np.float64),
            A_ineq=sp.csc_matrix(data[s.F]),
            b_ineq=np.asarray(data[s.G], dtype=np.float64),
        )
        is_lp = P.nnz == 0 or not np.any(P.data)

        if data.get(s.BOOL_IDX) or data.get(s.INT_IDX):
            if not is_lp:
                raise cp.error.SolverError(
                    "JADDLE solves mixed-integer LPs only; this problem has a"
                    " quadratic objective."
                )
            return _solve_mip(
                q, problem, lower, upper, data[s.BOOL_IDX], data[s.INT_IDX],
                verbose, solver_opts,
            )

        initial = None
        cached = None if solver_cache is None else solver_cache.get(self.name())
        if warm_start and cached is not None and not (is_lp and presolve):
            if (
                cached.primal.shape[0] == n
                and cached.dual_eq.shape[0] == problem["A_eq"].shape[0]
                and cached.dual_ineq.shape[0] == problem["A_ineq"].shape[0]
            ):
                initial = cached
        if initial is not None:
            options.setdefault("initial_solution", initial)

        start = time.time()
        if is_lp:
            lp = LP(
                q,
                problem["A_eq"],
                problem["b_eq"],
                problem["A_ineq"],
                problem["b_ineq"],
                lower,
                upper,
            )
            if presolve:
                result = jl.solve_with_presolve(lp, **options)
            else:
                result = jl.solve(jl.to_jaddle_sparse(lp), **options)
                result["objective"] = float(q @ np.asarray(result["solution"].primal))
        else:
            Q = sp.csr_matrix(P)
            result = jc.quadratic_program(
                Q=0.5 * (Q + Q.T),
                c=q,
                lower_bounds=lower,
                upper_bounds=upper,
                **problem,
                **{**QP_DEFAULTS, **options},
            )
        result["wall_seconds"] = time.time() - start

        solution = result["solution"]
        if solution is not None:
            # solve() may pad an empty block with one zero row.
            solution = SaddleState(
                primal=np.asarray(solution.primal, dtype=np.float64),
                dual_ineq=np.asarray(solution.dual_ineq, dtype=np.float64)[
                    : problem["A_ineq"].shape[0]
                ],
                dual_eq=np.asarray(solution.dual_eq, dtype=np.float64)[
                    : problem["A_eq"].shape[0]
                ],
            )
            result["solution"] = solution
            if solver_cache is not None:
                solver_cache[self.name()] = solution
        return result

    def invert(self, results, inverse_data):
        if results.get("mip"):
            return _invert_mip(self, results, inverse_data)
        status = _status(results)
        attr = {
            s.SOLVE_TIME: results.get("solve_seconds", results["wall_seconds"]),
            s.NUM_ITERS: results.get("epochs"),
            s.EXTRA_STATS: results,
        }
        if status not in s.SOLUTION_PRESENT:
            return failure_solution(status, attr)
        solution = results["solution"]
        # Jaddle's Lagrangian ½xᵀPx + qᵀx + y_eqᵀ(Ax - b) + y_ineqᵀ(Fx - g),
        # y_ineq >= 0, is the convention CVXPY's QP duals use (as OSQP's).
        y = np.concatenate([solution.dual_eq, solution.dual_ineq])
        n_eq = inverse_data[self.DIMS].zero
        dual_vars = utilities.get_dual_values(
            y[:n_eq], utilities.extract_dual_value, inverse_data[self.EQ_CONSTR]
        ) | utilities.get_dual_values(
            y[n_eq:], utilities.extract_dual_value, inverse_data[self.NEQ_CONSTR]
        )
        return Solution(
            status,
            results["objective"] + inverse_data[s.OFFSET],
            {self.VAR_ID: solution.primal},
            dual_vars,
            attr,
        )

    def cite(self, data):
        return ""


_MIP_LIMITS = ("time_limit", "node_limit", "interrupted")


def _solve_mip(q, problem, lower, upper, bool_idx, int_idx, verbose, solver_opts):
    """``JaddleSolver``'s route for a mixed-integer LP: ``jaddle_mip.solve``
    on CVXPY's QP data (a minimisation, offset excluded)."""
    import jaddle.jaddle_mip as jm

    options = dict(solver_opts)
    tol = options.pop("tol", None)
    if tol is not None:
        options.setdefault("lp_tolerance", tol)
    options["verbose"] = verbose

    n = q.shape[0]
    integrality = np.zeros(n, dtype=bool)
    integrality[list(bool_idx)] = True
    integrality[list(int_idx)] = True
    # CVXPY leaves a boolean variable's 0/1 range to the solver.
    bool_idx = np.asarray(bool_idx, dtype=int)
    lower, upper = lower.copy(), upper.copy()
    lower[bool_idx] = np.maximum(lower[bool_idx], 0.0)
    upper[bool_idx] = np.minimum(upper[bool_idx], 1.0)

    lp = LP(
        q,
        problem["A_eq"],
        problem["b_eq"],
        problem["A_ineq"],
        problem["b_ineq"],
        lower,
        upper,
    )
    start = time.time()
    result = jm.solve(jm.MIP(lp, integrality), **options)
    result["wall_seconds"] = time.time() - start
    result["mip"] = True
    if result["x"] is not None:
        result["objective"] = float(q @ result["x"])
    return result


def _mip_status(result):
    """The CVXPY status of a ``jaddle_mip.solve`` result. A budget stop or an
    unresolved search with an incumbent still returns its point."""
    status, found = result["status"], result["x"] is not None
    if status == "optimal":
        return s.OPTIMAL
    if status == "infeasible":
        return s.INFEASIBLE
    if status == "unbounded_or_infeasible":
        return s.INFEASIBLE_OR_UNBOUNDED
    if status in _MIP_LIMITS:
        return s.USER_LIMIT
    if status == "unresolved" and found:
        return s.OPTIMAL_INACCURATE
    return s.SOLVER_ERROR


def _invert_mip(solver, results, inverse_data):
    status = _mip_status(results)
    attr = {
        s.SOLVE_TIME: results.get("seconds", results["wall_seconds"]),
        s.NUM_ITERS: results.get("nodes"),
        s.EXTRA_STATS: results,
    }
    if status not in s.SOLUTION_PRESENT or results["x"] is None:
        return failure_solution(status, attr)
    # Branch and bound gives no duals: constraint.dual_value stays None.
    return Solution(
        status,
        results["objective"] + inverse_data[s.OFFSET],
        {solver.VAR_ID: np.asarray(results["x"], dtype=np.float64)},
        {},
        attr,
    )


# %%
# Route 1b: a CVXPY conic solver (second-order and PSD cones)


class JaddleConicSolver(_EqualByType, ConicSolver):
    """
    Jaddle as a CVXPY conic solver:
    ``prob.solve(solver=JaddleConicSolver(), **opts)``.

    Takes problems CVXPY can put in the cone form
    ``min cᵀx  s.t.  Ax + s = b,  s ∈ {0} x R+ x SOC x ... x PSD x ... x EXP x ...``
    and
    solves them with ``jaddle_linear.solve`` on cone-constrained variables
    (``jaddle.cones``). Zero rows become equality rows and nonnegative rows
    inequality rows ``Ax <= b``. A second-order, PSD or exponential block
    whose rows each pick out one distinct, unbounded variable (``X >> 0`` for
    a variable ``X``) puts the cone on those variables directly; any other
    block gets slack variables ``s`` in the cone with equality rows
    ``Ax + s = b``. ``cone_substitution=False`` uses slacks for every block.
    PSD blocks use SCS's ``svec`` (lower triangle by columns, off-diagonals
    times ``√2``). Power cones are not supported.

    The keyword arguments of ``prob.solve`` go to ``jaddle_linear.solve``
    (``verbose`` included). ``tol`` sets its three relative tolerances unless
    they are given by name; it defaults to ``DEFAULT_TOLERANCE``.
    ``warm_start=True`` (CVXPY's default) starts from the previous solve's
    point when the problem's sizes are unchanged. Solving the same problem with
    another solver in between clears CVXPY's cache, and with it the point.

    ``prob.solver_stats.extra_stats`` is Jaddle's result dict.
    """

    MIP_CAPABLE = False
    BOUNDED_VARIABLES = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC, SvecPSD, ExpCone]
    PSD_TRIANGLE_KIND = TriangleKind.LOWER
    PSD_SQRT2_SCALING = True
    # CVXPY's exponential cone, y exp(x / y) <= z, is jaddle.cones's.
    EXP_CONE_ORDER = [0, 1, 2]

    def name(self):
        return "JADDLE_CONIC"

    def import_solver(self) -> None:
        import jaddle  # noqa: F401

    def solve_via_data(
        self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None
    ):
        options = _with_tolerances(solver_opts, _QP_TOLERANCES)
        options["verbose"] = verbose

        dims = data[ConicSolver.DIMS]
        A = sp.csr_matrix(data[s.A])
        b = np.asarray(data[s.B], dtype=np.float64)
        c = np.asarray(data[s.C], dtype=np.float64)
        n = c.shape[0]
        n_zero, n_nonneg = dims.zero, dims.nonneg
        cones = Cones(soc=dims.soc, psd=dims.psd, exp=dims.exp)

        lower, upper = data[s.LOWER_BOUNDS], data[s.UPPER_BOUNDS]
        lower = np.full(n, -np.inf) if lower is None else np.asarray(lower, float)
        upper = np.full(n, np.inf) if upper is None else np.asarray(upper, float)

        substitute = options.pop("cone_substitution", True)
        red = _conic_reduction(
            A, b, c, lower, upper, n_zero, n_nonneg, cones, substitute
        )
        lp = red["lp"]
        A_eq, A_ineq = lp.A_eq, lp.A_ineq

        cached = None if solver_cache is None else solver_cache.get(self.name())
        if warm_start and cached is not None:
            if (
                cached.primal.shape[0] == lp.c.shape[0]
                and cached.dual_eq.shape[0] == A_eq.shape[0]
                and cached.dual_ineq.shape[0] == A_ineq.shape[0]
            ):
                options.setdefault("initial_solution", cached)

        start = time.time()
        result = jl.solve(lp, **options)
        result["wall_seconds"] = time.time() - start

        solution = result["solution"]
        if solution is not None:
            # solve() may pad an empty block with one zero row.
            solution = SaddleState(
                primal=np.asarray(solution.primal, dtype=np.float64),
                dual_ineq=np.asarray(solution.dual_ineq, dtype=np.float64)[
                    : A_ineq.shape[0]
                ],
                dual_eq=np.asarray(solution.dual_eq, dtype=np.float64)[
                    : A_eq.shape[0]
                ],
            )
            result["solution"] = solution
            x, y = _conic_recover(red, solution)
            result["cvxpy_primal"], result["cvxpy_dual"] = x, y
            result["objective"] = float(c @ x)
            result["cone_substitution"] = red["substituted"]
            if solver_cache is not None:
                solver_cache[self.name()] = solution
        return result

    def invert(self, results, inverse_data):
        status = _status(results)
        attr = {
            s.SOLVE_TIME: results.get("solve_seconds", results["wall_seconds"]),
            s.NUM_ITERS: results.get("epochs"),
            s.EXTRA_STATS: results,
        }
        if status not in s.SOLUTION_PRESENT:
            return failure_solution(status, attr)
        n_zero = inverse_data[self.DIMS].zero
        # x and y are already in CVXPY's variable and row order (see
        # _conic_recover); y follows the sign convention of CVXPY's conic
        # duals (as SCS's), with cone duals in the dual cone.
        y = results["cvxpy_dual"]
        dual_vars = utilities.get_dual_values(
            y[:n_zero], utilities.extract_dual_value, inverse_data[self.EQ_CONSTR]
        ) | utilities.get_dual_values(
            y[n_zero:], utilities.extract_dual_value, inverse_data[self.NEQ_CONSTR]
        )
        return Solution(
            status,
            results["objective"] + inverse_data[s.OFFSET],
            {inverse_data[self.VAR_ID]: results["cvxpy_primal"]},
            dual_vars,
            attr,
        )

    def cite(self, data):
        return ""


def _conic_reduction(A, b, c, lower, upper, n_zero, n_nonneg, cones, substitute):
    """Jaddle's LP for CVXPY's cone form ``A x + s = b``, ``s`` in the cones.

    Zero rows become equality rows and nonnegative rows inequality rows. A
    cone block whose rows each select one distinct, unbounded variable
    (``s_r = b_r - a_r x_j``, as CVXPY writes ``X >> 0`` for a variable ``X``)
    is put on those variables directly: ``x_j = (b_r - z_r) / a_r`` with
    ``z`` in the cone, and the block's rows are dropped. Any other cone
    block keeps slack variables ``s`` in the cone and equality rows
    ``A x + s = b``. Keeping both a variable and a slack copy tied by
    equality rows made PDHG reach the cone only through those rows' duals:
    on a 200-node Max-Cut SDP it took 3x the iterations.

    The LP's variables are ``z = [kept columns of x; cone tail]`` with
    ``x = P z + q``. Returns the LP and what ``_conic_recover`` needs.
    """
    A = sp.csr_matrix(A)
    A.eliminate_zeros()
    A.sum_duplicates()
    n, m = c.shape[0], A.shape[0]
    free = ~np.isfinite(lower) & ~np.isfinite(upper)
    used = np.zeros(n, bool)

    # Which cone blocks select variables, in tail order.
    blocks = []
    row = n_zero + n_nonneg
    for size in cones.block_sizes():
        rows = np.arange(row, row + size)
        block = A[rows]
        cols = None
        if substitute and np.all(np.diff(block.indptr) == 1):
            candidate = block.indices.copy()
            if (
                np.unique(candidate).size == size
                and free[candidate].all()
                and not used[candidate].any()
            ):
                cols = candidate
                used[cols] = True
        blocks.append((rows, cols, None if cols is None else block.data.copy()))
        row += size

    kept = np.flatnonzero(~used)
    n_kept = kept.size
    N = n_kept + cones.size
    # x = P z + q.
    P_rows, P_cols, P_vals = [kept], [np.arange(n_kept)], [np.ones(n_kept)]
    q = np.zeros(n)
    slack_rows, slack_pos, sub_pos = [], [], []
    pos = n_kept
    for rows, cols, a in blocks:
        tail = np.arange(pos, pos + rows.size)
        if cols is None:
            slack_rows.append(rows)
            slack_pos.append(tail)
        else:
            P_rows.append(cols)
            P_cols.append(tail)
            P_vals.append(-1.0 / a)
            q[cols] = b[rows] / a
            sub_pos.append((rows, tail))
        pos += rows.size
    P = sp.csr_matrix(
        (np.concatenate(P_vals), (np.concatenate(P_rows), np.concatenate(P_cols))),
        shape=(n, N),
    )
    slack_rows = np.concatenate(slack_rows) if slack_rows else np.zeros(0, int)
    slack_pos = np.concatenate(slack_pos) if slack_pos else np.zeros(0, int)

    zero_rows = np.arange(n_zero)
    nonneg_rows = np.arange(n_zero, n_zero + n_nonneg)
    eq_rows = np.concatenate([zero_rows, slack_rows])
    A_q = A @ q
    slack_cols = sp.csr_matrix(
        (np.ones(slack_rows.size), (np.arange(n_zero, n_zero + slack_rows.size), slack_pos)),
        shape=(eq_rows.size, N),
    )
    A_eq = (A[eq_rows] @ P + slack_cols).tocsc()
    A_ineq = (A[nonneg_rows] @ P).tocsc()
    lp = LP(
        P.T @ c,
        A_eq,
        b[eq_rows] - A_q[eq_rows],
        A_ineq,
        b[nonneg_rows] - A_q[nonneg_rows],
        np.concatenate([lower[kept], np.full(cones.size, -np.inf)]),
        np.concatenate([upper[kept], np.full(cones.size, np.inf)]),
        cones=cones,
    )
    return dict(
        lp=lp,
        P=P,
        q=q,
        m=m,
        eq_rows=eq_rows,
        nonneg_rows=nonneg_rows,
        sub_pos=sub_pos,
        substituted=int(sum(r.size for r, _ in sub_pos)),
    )


def _conic_recover(red, solution):
    """CVXPY's ``x`` and ``y`` (rows in CVXPY's order) from the solution of
    ``_conic_reduction``'s LP.

    A dropped cone block's dual is the reduced cost of its cone variables
    ``z``: the slack form's stationarity in ``s`` gives ``y = c_s + Aᵀy`` on
    the slack columns, and ``z`` stands in for ``s``. It lies in the dual
    cone."""
    lp = red["lp"]
    x = red["P"] @ solution.primal + red["q"]
    y = np.zeros(red["m"])
    y[red["eq_rows"]] = solution.dual_eq
    y[red["nonneg_rows"]] = solution.dual_ineq
    if red["sub_pos"]:
        reduced_cost = (
            np.asarray(lp.c)
            + lp.A_eq.T @ solution.dual_eq
            + lp.A_ineq.T @ solution.dual_ineq
        )
        for rows, tail in red["sub_pos"]:
            y[rows] = reduced_cost[tail]
    return x, y


# %%
# Route 2: compiling the expression tree to a JaddleCP


def _safe_norm(v, axis=None, keepdims=False):
    """The 2-norm with gradient 0 instead of NaN at the origin."""
    squared = jnp.sum(v * v, axis=axis, keepdims=keepdims)
    positive = squared > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, squared, 1.0)), 0.0)


def _safe_pnorm(v, p, axis=None, keepdims=False):
    total = jnp.sum(jnp.abs(v) ** p, axis=axis, keepdims=keepdims)
    positive = total > 0
    return jnp.where(positive, jnp.where(positive, total, 1.0) ** (1.0 / p), 0.0)


def _huber(v, M):
    a = jnp.abs(v)
    return jnp.where(a <= M, v * v, 2 * M * a - M * M)


def _entr(u):
    """``-u log u`` for ``u >= ENTR_FLOOR`` and its tangent line there below:
    concave and C¹, with slope at most ``-log(ENTR_FLOOR) - 1``."""
    d = ENTR_FLOOR
    safe = jnp.maximum(u, d)
    tangent = -d * np.log(d) - (np.log(d) + 1.0) * (u - d)
    return jnp.where(u >= d, -safe * jnp.log(safe), tangent)


def _expand(t, axis, keepdims, ndim):
    """``t``, the result of a reduction over ``axis``, broadcastable against
    the reduced argument again."""
    if axis is None:
        return jnp.reshape(t, (1,) * ndim)
    if keepdims:
        return t
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    return jnp.expand_dims(t, tuple(a % ndim for a in axes))


def _jax_key(key):
    """A CVXPY index key as JAX accepts it: lists, which NumPy reads as
    integer arrays, as arrays."""
    if isinstance(key, list):
        return np.asarray(key)
    if isinstance(key, tuple):
        return tuple(np.asarray(k) if isinstance(k, list) else k for k in key)
    return key


def _even_integer(p):
    return float(p).is_integer() and int(p) % 2 == 0


class CompiledProblem:
    """A CVXPY problem as a ``JaddleCP`` (``cp``), with what is needed to map
    a solution back. Built by ``compile_problem``.

    * ``variables``: ``{var.id: (offset, shape)}`` within the primal vector,
      which also holds the epigraph variables (``num_epigraph`` entries).
    * ``rows``: ``{constraint.id: ("eq" | "ineq", offset, shape)}`` within
      ``dual_eq`` / ``dual_ineq``.
    * ``bounds``: ``{constraint.id: (var.id, side, mask)}`` for the
      constraints held as box bounds, ``side`` +1 for an upper bound and -1
      for a lower one; ``mask`` marks the entries whose bound is that
      constraint's (the box keeps the tightest of several).
    * ``sign``: -1 for a maximisation, solved as the minimisation of ``-f``.
    * ``eliminated``: the equalities that define a variable, substituted out
      (``num_eliminated`` entries), as ``(constraint.id, var.id, sign,
      offset, shape, row)`` with ``row(z) = sign * var + rest``.

    Offsets in ``variables`` and ``eliminated`` index the full vector ``z``,
    which holds every variable; ``cp`` works on the vector of the kept
    entries ``z[keep]``, and ``expand`` maps it back to ``z``, filling in
    the eliminated variables. ``objective`` and ``eq_rows`` /
    ``ineq_rows`` are the minimised objective and ``cp``'s rows as
    functions of ``z``.
    """

    def __init__(
        self,
        cp,
        variables,
        rows,
        bounds,
        sign,
        num_epigraph,
        eliminated=(),
        keep=None,
        expand=None,
        objective=None,
        eq_rows=None,
        ineq_rows=None,
    ):
        self.cp = cp
        self.variables = variables
        self.rows = rows
        self.bounds = bounds
        self.sign = sign
        self.num_epigraph = num_epigraph
        self.eliminated = list(eliminated)
        self.num_eliminated = sum(
            int(np.prod(shape, dtype=int)) for *_, shape, _ in self.eliminated
        )
        self.keep = keep
        self.expand = expand if expand is not None else (lambda x: x)
        self.objective = objective if objective is not None else cp.objective
        self.eq_rows = eq_rows if eq_rows is not None else cp.constraints_eq
        self.ineq_rows = ineq_rows if ineq_rows is not None else cp.constraints_ineq


class _Compiler:
    def __init__(self, epigraph, dtype):
        self.epigraph = epigraph
        self.dtype = dtype
        self.size = 0
        self.lower, self.upper = [], []
        self.variables = {}
        self.eq_rows, self.ineq_rows = [], []
        self.eq_size, self.ineq_size = 0, 0
        self.num_epigraph = 0
        self.memo = {}

    # -- variables and rows

    def block(self, shape, lower=-np.inf, upper=np.inf):
        """A new slice of the primal vector, as ``x -> array of shape``."""
        offset, size = self.size, int(np.prod(shape, dtype=int))
        self.size += size
        self.lower.append(np.broadcast_to(np.asarray(lower, float), shape).ravel())
        self.upper.append(np.broadcast_to(np.asarray(upper, float), shape).ravel())
        return offset, lambda x: jnp.reshape(x[offset : offset + size], shape)

    def row(self, kind, fn, shape):
        """Add rows ``fn(x)`` (``= 0`` or ``<= 0``); returns their offset."""
        size = int(np.prod(shape, dtype=int))

        def flat(x):
            return jnp.ravel(jnp.broadcast_to(fn(x), shape))

        if kind == "eq":
            offset, self.eq_size = self.eq_size, self.eq_size + size
            self.eq_rows.append(flat)
        else:
            offset, self.ineq_size = self.ineq_size, self.ineq_size + size
            self.ineq_rows.append(flat)
        return offset

    def epigraph_variable(self, expr, concave=False):
        """An epigraph (or, ``concave``, hypograph) variable for ``expr``,
        bounded by the atom's sign."""
        lower = 0.0 if expr.is_nonneg() else -np.inf
        upper = 0.0 if expr.is_nonpos() else np.inf
        _, t = self.block(expr.shape, lower, upper)
        self.num_epigraph += int(np.prod(expr.shape, dtype=int))
        return t

    def add_variable(self, var):
        attributes = _attributes(var)
        lower, upper = -np.inf, np.inf
        bounds = attributes.pop("bounds", None)
        if bounds is not None:
            lb, ub = bounds
            value = lambda b: b.value if isinstance(b, cp.Expression) else b
            lower = -np.inf if lb is None else value(lb)
            upper = np.inf if ub is None else value(ub)
        if attributes.pop("nonneg", False) | attributes.pop("pos", False):
            lower = np.maximum(lower, 0.0)
        if attributes.pop("nonpos", False) | attributes.pop("neg", False):
            upper = np.minimum(upper, 0.0)
        if attributes:
            raise NotImplementedError(
                f"variable {var.name()}: attributes {sorted(attributes)} are not"
                " supported by the compiled route"
            )
        offset, fn = self.block(var.shape, lower, upper)
        self.variables[var.id] = (offset, var.shape)
        self.memo[var.id] = fn

    # -- expressions

    def constant(self, value, dense=True):
        if sp.issparse(value):
            if dense:
                return jnp.asarray(value.toarray(), self.dtype)
            return scipy_to_bcoo(value, self.dtype)
        return jnp.asarray(value, self.dtype)

    def expr(self, e):
        """``x -> value of e`` as a JAX function of the primal vector."""
        # Constants carry no id; a shared subexpression is built once.
        key = getattr(e, "id", None) or id(e)
        if key not in self.memo:
            self.memo[key] = self._build(e)
        return self.memo[key]

    def _args(self, e):
        return [self.expr(a) for a in e.args]

    def _build(self, e):
        if isinstance(e, Variable):
            raise ValueError(f"variable {e.name()} is not one of the problem's")
        if isinstance(e, (Constant, Parameter)):
            if e.value is None:
                raise ValueError(f"parameter {e.name()} has no value")
            value = self.constant(e.value)
            return lambda x: value

        # Affine atoms. multiply (elementwise) subclasses MulExpression
        # (matmul), so it is tested first.
        if isinstance(e, multiply):
            f, g = self._args(e)
            return lambda x: f(x) * g(x)
        if isinstance(e, MulExpression):
            lhs, rhs = e.args
            if lhs.ndim == 0 or rhs.ndim == 0:
                # CVXPY also writes a product with a scalar as a MulExpression.
                f, g = self._args(e)
                return lambda x: f(x) * g(x)
            if isinstance(lhs, (Constant, Parameter)) and sp.issparse(lhs.value):
                A, g = self.constant(lhs.value, dense=False), self.expr(rhs)
                return lambda x: A @ g(x)
            f, g = self._args(e)
            return lambda x: f(x) @ g(x)
        if isinstance(e, AddExpression):
            fs = self._args(e)
            return lambda x: sum((f(x) for f in fs[1:]), fs[0](x))
        if isinstance(e, NegExpression):
            (f,) = self._args(e)
            return lambda x: -f(x)
        if isinstance(e, DivExpression):
            f, g = self._args(e)
            return lambda x: f(x) / g(x)
        if isinstance(e, (index, special_index)):
            # CVXPY writes an integer index as a length-1 slice and drops the
            # axis afterwards, so the result is reshaped to e.shape.
            (f,), key, shape = self._args(e), _jax_key(e.key), e.shape
            return lambda x: jnp.reshape(f(x)[key], shape)
        if isinstance(e, (Promote, broadcast_to)):
            (f,), shape = self._args(e), e.shape
            return lambda x: jnp.broadcast_to(f(x), shape)
        if isinstance(e, reshape):
            (f,), shape, order = self._args(e), e.shape, e.order
            return lambda x: jnp.reshape(f(x), shape, order=order)
        if isinstance(e, transpose):
            (f,), axes = self._args(e), getattr(e, "axes", None)
            return lambda x: jnp.transpose(f(x), axes)
        if isinstance(e, Sum):
            (f,), axis, keepdims = self._args(e), e.axis, e.keepdims
            return lambda x: jnp.sum(f(x), axis=axis, keepdims=keepdims)
        if isinstance(e, Hstack):
            fs = self._args(e)
            return lambda x: jnp.hstack([f(x) for f in fs])
        if isinstance(e, Vstack):
            fs = self._args(e)
            return lambda x: jnp.vstack([f(x) for f in fs])
        if isinstance(e, Concatenate):
            fs, axis = self._args(e), e.axis
            return lambda x: jnp.concatenate([f(x) for f in fs], axis=axis)
        if isinstance(e, Trace):
            (f,) = self._args(e)
            return lambda x: jnp.trace(f(x))
        if isinstance(e, (diag_vec, diag_mat)):
            (f,), k = self._args(e), getattr(e, "k", 0)
            return lambda x: jnp.diag(f(x), k)

        # Smooth atoms.
        if isinstance(e, quad_over_lin):
            numerator, denominator = e.args
            if not isinstance(denominator, (Constant, Parameter)):
                raise NotImplementedError(
                    "quad_over_lin with a variable denominator (domain y > 0)"
                    " is not supported by the compiled route"
                )
            f, g = self._args(e)
            axis, keepdims = getattr(e, "axis", None), getattr(e, "keepdims", False)
            return lambda x: jnp.sum(f(x) ** 2, axis=axis, keepdims=keepdims) / g(x)
        if isinstance(e, QuadForm):
            f, P = self._args(e)
            return lambda x: f(x) @ (P(x) @ f(x))
        if isinstance(e, Power):
            p = float(getattr(e, "_p_orig", e.p.value))
            if p == 1:
                return self._args(e)[0]
            if not _even_integer(p):
                raise NotImplementedError(
                    f"power(x, {p}) has a restricted domain and is not supported"
                    " by the compiled route; only even integer powers are"
                )
            (f,) = self._args(e)
            return lambda x: f(x) ** int(p)
        if isinstance(e, cvx_exp):
            (f,) = self._args(e)
            return lambda x: jnp.exp(f(x))
        if isinstance(e, entr):
            (f,) = self._args(e)
            return lambda x: _entr(f(x))
        if isinstance(e, logistic):
            (f,) = self._args(e)
            return lambda x: jnp.logaddexp(0.0, f(x))
        if isinstance(e, log_sum_exp):
            (f,), axis, keepdims = self._args(e), e.axis, e.keepdims
            return lambda x: jss.logsumexp(f(x), axis=axis, keepdims=keepdims)
        if isinstance(e, huber):
            (f,), M = self._args(e), float(e.M.value)
            return lambda x: _huber(f(x), M)
        if isinstance(e, Pnorm):
            p = float(getattr(e, "original_p", e.p))
            if p < 1:
                raise NotImplementedError(
                    f"pnorm(x, {p}) is concave with a restricted domain and is"
                    " not supported by the compiled route"
                )
            (f,), axis, keepdims = self._args(e), e.axis, e.keepdims
            if p == 2:
                return lambda x: _safe_norm(f(x), axis, keepdims)
            return lambda x: _safe_pnorm(f(x), p, axis, keepdims)

        # Piecewise-linear atoms.
        if isinstance(e, cvx_abs):
            return self._abs(e)
        if isinstance(e, (maximum, minimum)):
            fs, concave = self._args(e), isinstance(e, minimum)
            if not self.epigraph:
                op = jnp.minimum if concave else jnp.maximum

                def direct(x):
                    out = fs[0](x)
                    for f in fs[1:]:
                        out = op(out, f(x))
                    return out

                return direct
            t, shape = self.epigraph_variable(e, concave), e.shape
            sign = -1.0 if concave else 1.0
            for f in fs:
                self.row(
                    "ineq", lambda x, f=f: sign * (f(x) - t(x)), shape
                )
            return t
        if isinstance(e, (cvx_max, cvx_min)):
            (f,), concave = self._args(e), isinstance(e, cvx_min)
            axis, keepdims = e.axis, e.keepdims
            if not self.epigraph:
                op = jnp.min if concave else jnp.max
                return lambda x: op(f(x), axis=axis, keepdims=keepdims)
            t, ndim = self.epigraph_variable(e, concave), e.args[0].ndim
            sign = -1.0 if concave else 1.0
            self.row(
                "ineq",
                lambda x: sign * (f(x) - _expand(t(x), axis, keepdims, ndim)),
                e.args[0].shape,
            )
            return t
        if isinstance(e, norm1):
            s_ = self._abs(e.args[0], absolute_of=True)
            axis, keepdims = e.axis, e.keepdims
            return lambda x: jnp.sum(s_(x), axis=axis, keepdims=keepdims)
        if isinstance(e, norm_inf):
            (f,), axis, keepdims = self._args(e), e.axis, e.keepdims
            if not self.epigraph:
                return lambda x: jnp.max(jnp.abs(f(x)), axis=axis, keepdims=keepdims)
            t, arg = self.epigraph_variable(e), e.args[0]
            for sign in (1.0, -1.0):
                self.row(
                    "ineq",
                    lambda x, sign=sign: sign * f(x)
                    - _expand(t(x), axis, keepdims, arg.ndim),
                    arg.shape,
                )
            return t

        raise NotImplementedError(
            f"{type(e).__name__} is not supported by the compiled route"
        )

    def _abs(self, e, absolute_of=False):
        """``abs`` of ``e``'s argument, or with ``absolute_of`` of ``e``
        itself: an epigraph variable ``t >= ±u``."""
        u = e if absolute_of else e.args[0]
        f = self.expr(u)
        if not self.epigraph:
            return lambda x: jnp.abs(f(x))
        _, t = self.block(u.shape, 0.0, np.inf)
        self.num_epigraph += int(np.prod(u.shape, dtype=int))
        for sign in (1.0, -1.0):
            self.row("ineq", lambda x, sign=sign: sign * f(x) - t(x), u.shape)
        return t


def _attributes(var):
    """The attributes set on ``var`` (``nonneg``, ``bounds``, ...)."""
    return {k: v for k, v in var.attributes.items() if v is not False and v is not None}


def _signed_terms(e, sign=1.0):
    """The additive terms of ``e``, each with its sign."""
    if isinstance(e, AddExpression):
        return [t for a in e.args for t in _signed_terms(a, sign)]
    if isinstance(e, NegExpression):
        return _signed_terms(e.args[0], -sign)
    return [(e, sign)]


def _definition(constraint, eligible):
    """``(var, sign)`` when ``constraint`` is an equality reading
    ``sign * var + rest == 0``, with ``var`` a whole variable of the
    constraint's shape whose id is in ``eligible``, and ``rest`` free of it;
    otherwise ``None``."""
    if not isinstance(constraint, Equality):
        return None
    lhs, rhs = constraint.args
    terms = _signed_terms(lhs) + _signed_terms(rhs, -1.0)
    for i, (term, sign) in enumerate(terms):
        if (
            not isinstance(term, Variable)
            or term.id not in eligible
            or term.shape != constraint.shape
        ):
            continue
        rest = (t for j, (t, _) in enumerate(terms) if j != i)
        if all(term.id not in {v.id for v in t.variables()} for t in rest):
            return term, sign
    return None


def _bound_of(constraint):
    """``(variable, side, value)`` when ``constraint`` bounds a whole
    variable by a constant, ``side`` +1 for ``var <= value`` and -1 for
    ``var >= value``; otherwise ``None``."""

    def constant(e):
        return isinstance(e, (Constant, Parameter)) and e.value is not None

    if isinstance(constraint, Inequality):
        lhs, rhs = constraint.args
        if isinstance(lhs, Variable) and constant(rhs):
            return lhs, 1, rhs.value
        if isinstance(rhs, Variable) and constant(lhs):
            return rhs, -1, lhs.value
    if isinstance(constraint, (NonNeg, NonPos)) and isinstance(
        constraint.args[0], Variable
    ):
        return constraint.args[0], (-1 if isinstance(constraint, NonNeg) else 1), 0.0
    return None


def compile_problem(problem, epigraph=True, detect_bounds=True, eliminate=True):
    """
    Compile a DCP ``cvxpy.Problem`` to a ``CompiledProblem`` whose ``cp`` is a
    ``JaddleCP`` for ``jaddle_convex.solve``.

    Variables become slices of one primal vector, with their ``nonneg`` /
    ``nonpos`` / ``bounds`` attributes as box bounds; with ``detect_bounds``
    so do constraints bounding a whole variable by a constant
    (``x >= 0``, ``x <= u``), which then cost no dual. An equality
    ``lhs == rhs`` becomes the row ``lhs - rhs = 0``, an inequality
    ``lhs <= rhs`` the row ``lhs - rhs <= 0``.

    With ``eliminate`` (default) an equality that defines a variable, such as
    ``y == C @ x + v`` for ``v``, is substituted out instead: the variable
    must appear in it once, whole and with coefficient ±1, and have no
    attributes or bounds. The solver then sees neither the variable nor the
    row, which can save many iterations (a free variable tied to others only
    by rows has no curvature of its own); its value and the row's dual are
    recovered afterwards. Definitions are taken in constraint order and
    never reference a variable defined later.

    With ``epigraph`` (default) the piecewise-linear atoms (``abs``,
    ``maximum``, ``max``, ``norm1``, ``norm_inf``, their concave mirrors, and
    ``pos`` / ``neg``, which CVXPY writes with them) become epigraph
    variables with one linear row per piece, exact for a DCP problem; without
    it they are evaluated as they are and the solver sees their subgradients.

    Parameters are read at compile time. Atoms with a restricted domain
    (``log``, ``sqrt``, ...) and second-order or PSD constraints are not
    supported: Jaddle's iterates respect only the box, so they would leave
    the domain. They, and anything else not covered, raise
    ``NotImplementedError``. The exception is ``entr``, continued below
    ``ENTR_FLOOR`` by its tangent line; this is exact when the solution keeps
    its argument at or above ``ENTR_FLOOR``.
    """
    if not problem.is_dcp():
        raise cp.error.DCPError("problem is not DCP")
    compiler = _Compiler(epigraph, jo.jaddle_dtype())
    for var in problem.variables():
        compiler.add_variable(var)

    # Constraints held as box bounds; the box keeps the tightest of several.
    bound_constraints = []
    general = []
    for constraint in problem.constraints:
        bound = _bound_of(constraint) if detect_bounds else None
        if bound is None:
            general.append(constraint)
            continue
        var, side, value = bound
        offset, shape = compiler.variables[var.id]
        size = int(np.prod(shape, dtype=int))
        value = np.broadcast_to(np.asarray(value, float), shape).ravel()
        box = compiler.lower if side < 0 else compiler.upper
        # Blocks are allocated one per variable before any epigraph block,
        # so the variable's block index is its position.
        block = list(compiler.variables).index(var.id)
        box[block] = np.maximum(box[block], value) if side < 0 else np.minimum(
            box[block], value
        )
        bound_constraints.append((constraint, var.id, side, value, offset, size, block))

    # Equalities defining a variable. A variable in an earlier definition
    # cannot be defined later, so definitions reference only kept variables
    # and earlier definitions.
    eliminated = []
    if eliminate:
        eligible = {var.id for var in problem.variables() if not _attributes(var)}
        eligible -= {var_id for _, var_id, *_ in bound_constraints}
        kept = []
        for constraint in general:
            found = _definition(constraint, eligible)
            if found is None:
                kept.append(constraint)
                continue
            var, var_sign = found
            eligible -= {v.id for v in constraint.variables()}
            lhs, rhs = (compiler.expr(a) for a in constraint.args)
            offset, shape = compiler.variables[var.id]
            eliminated.append(
                (
                    constraint.id,
                    var.id,
                    var_sign,
                    offset,
                    shape,
                    lambda z, lhs=lhs, rhs=rhs, shape=shape: jnp.ravel(
                        jnp.broadcast_to(lhs(z) - rhs(z), shape)
                    ),
                )
            )
        general = kept

    rows = {}
    for constraint in general:
        if isinstance(constraint, (Equality, Zero)):
            kind = "eq"
        elif isinstance(constraint, (Inequality, NonNeg, NonPos)):
            kind = "ineq"
        else:
            raise NotImplementedError(
                f"{type(constraint).__name__} constraints are not supported by"
                " the compiled route"
            )
        if isinstance(constraint, (Equality, Inequality)):
            lhs, rhs = (compiler.expr(a) for a in constraint.args)
            fn = lambda x, lhs=lhs, rhs=rhs: lhs(x) - rhs(x)
        else:
            f = compiler.expr(constraint.args[0])
            fn = (lambda x, f=f: -f(x)) if isinstance(constraint, NonNeg) else f
        rows[constraint.id] = (kind, compiler.row(kind, fn, constraint.shape), constraint.shape)

    sign = -1.0 if isinstance(problem.objective, cp.Maximize) else 1.0
    objective_fn = compiler.expr(problem.objective.expr)

    lower = np.concatenate(compiler.lower) if compiler.lower else np.zeros(0)
    upper = np.concatenate(compiler.upper) if compiler.upper else np.zeros(0)
    bounds = {}
    claimed = {}
    for constraint, var_id, side, value, offset, size, block in bound_constraints:
        box = lower if side < 0 else upper
        tight = box[offset : offset + size] == value
        taken = claimed.setdefault((var_id, side), np.zeros(size, bool))
        mask = tight & ~taken
        taken |= mask
        bounds[constraint.id] = (var_id, side, mask)

    eq_rows, ineq_rows = compiler.eq_rows, compiler.ineq_rows
    dtype = compiler.dtype

    def stack(fns):
        if not fns:
            return lambda x: jnp.zeros(0, dtype)
        return lambda x: jnp.concatenate([f(x) for f in fns])

    objective = lambda z: sign * jnp.sum(objective_fn(z))
    eq_full, ineq_full = stack(eq_rows), stack(ineq_rows)
    keep, expand = None, None
    if eliminated:
        removed = np.zeros(compiler.size, bool)
        for *_, offset, shape, _ in eliminated:
            removed[offset : offset + int(np.prod(shape, dtype=int))] = True
        keep = np.flatnonzero(~removed)
        lower, upper = lower[keep], upper[keep]
        full_size, kept_index = compiler.size, jnp.asarray(keep)

        # jit: the solver's setup calls the problem's functions eagerly.
        @jax.jit
        def expand(x):
            # Each eliminated block is still zero when its row is evaluated,
            # so row = sign * 0 + rest and the variable is -sign * row.
            z = jnp.zeros(full_size, x.dtype).at[kept_index].set(x)
            for _, _, var_sign, offset, shape, row in eliminated:
                size = int(np.prod(shape, dtype=int))
                z = z.at[offset : offset + size].set(-var_sign * row(z))
            return z

    reduced = (lambda f: f) if expand is None else (lambda f: lambda x: f(expand(x)))
    jcp = JaddleCP(
        num_variables=lower.size,
        objective=reduced(objective),
        constraints_eq=reduced(eq_full),
        constraints_ineq=reduced(ineq_full),
        lower_bounds=jnp.asarray(lower, dtype),
        upper_bounds=jnp.asarray(upper, dtype),
    )
    return CompiledProblem(
        jcp,
        compiler.variables,
        rows,
        bounds,
        sign,
        compiler.num_epigraph,
        eliminated=eliminated,
        keep=keep,
        expand=expand,
        objective=objective,
        eq_rows=eq_full,
        ineq_rows=ineq_full,
    )


def _unpack(problem, compiled, result):
    """Write a ``jaddle_convex.solve`` result back into ``problem``."""
    status = _status(result)
    attr = {
        s.SOLVE_TIME: result["solve_seconds"],
        s.NUM_ITERS: result["epochs"],
        s.EXTRA_STATS: result,
    }
    if status not in s.SOLUTION_PRESENT:
        solution = failure_solution(status, attr)
    else:
        state = result["solution"]
        z = compiled.expand(state.primal)
        x = np.asarray(z, dtype=np.float64)
        y_eq = np.asarray(state.dual_eq, dtype=np.float64)
        y_ineq = np.asarray(state.dual_ineq, dtype=np.float64)
        primal_vars = {
            var_id: x[offset : offset + int(np.prod(shape, dtype=int))].reshape(shape)
            for var_id, (offset, shape) in compiled.variables.items()
        }
        dual_vars = {}
        for constraint_id, (kind, offset, shape) in compiled.rows.items():
            y = y_eq if kind == "eq" else y_ineq
            size = int(np.prod(shape, dtype=int))
            dual_vars[constraint_id] = y[offset : offset + size].reshape(shape)

        # The Lagrangian over the full vector, with the duals of the
        # eliminated rows found so far.
        eliminated_duals = []

        def lagrangian(z):
            value = (
                compiled.objective(z)
                + jnp.dot(state.dual_eq, compiled.eq_rows(z))
                + jnp.dot(state.dual_ineq, compiled.ineq_rows(z))
            )
            for row, y in eliminated_duals:
                value = value + jnp.dot(y, row(z))
            return value

        # An eliminated variable is free, so ∇_v L = 0, and it enters its
        # own row as sign * v: the row's dual is -sign * ∇_v of the rest of
        # L. Later definitions may use earlier variables but not the reverse,
        # so the rows are solved last to first.
        for constraint_id, _, var_sign, offset, shape, row in reversed(
            compiled.eliminated
        ):
            size = int(np.prod(shape, dtype=int))
            y = -var_sign * jax.grad(lagrangian)(z)[offset : offset + size]
            eliminated_duals.append((row, y))
            dual_vars[constraint_id] = np.asarray(y, np.float64).reshape(shape)

        if compiled.bounds:
            # A bound's dual is the reduced gradient it absorbs: for x >= l,
            # ∇L - y = 0 with y >= 0, so y = max(∇L, 0); for x <= u, max(-∇L, 0).
            gradient = np.asarray(jax.grad(lagrangian)(z), np.float64)
            for constraint_id, (var_id, side, mask) in compiled.bounds.items():
                offset, shape = compiled.variables[var_id]
                g = gradient[offset : offset + mask.size]
                dual = np.where(mask, np.maximum(-side * g, 0.0), 0.0)
                dual_vars[constraint_id] = dual.reshape(shape)
        objective = float(compiled.sign * compiled.cp.objective(state.primal))
        solution = Solution(status, objective, primal_vars, dual_vars, attr)
    problem.unpack(solution)
    problem._solver_stats = SolverStats.from_dict(attr, "JADDLE")


def solve(
    problem, qp_route=True, epigraph=True, detect_bounds=True, eliminate=True, **options
):
    """
    Solve a CVXPY problem with Jaddle and write the result back into it
    (``problem.status``, ``problem.value``, ``var.value``,
    ``constraint.dual_value``, ``problem.solver_stats``); returns
    ``problem.value``.

    A problem CVXPY can put in QP form (``problem.is_qp()``: an LP or QP,
    piecewise-linear terms included) is solved by
    ``problem.solve(solver=JaddleSolver(), **options)``, which gets HiGHS's
    presolve or ``quadratic_program``'s KKT scaling. Any other DCP problem, or
    every problem when ``qp_route=False``, is compiled by ``compile_problem``
    (with ``epigraph``, ``detect_bounds`` and ``eliminate``) and solved by
    ``jaddle_convex.solve`` with ``COMPILED_DEFAULTS`` updated by ``options``.
    Its tolerances are absolute, in the problem's own units.

    On either route ``tol`` (default ``DEFAULT_TOLERANCE``) sets every
    tolerance not given by its own name, so the same call works whichever
    route the problem takes.

    Status: ``"optimal"`` when the solve certified, ``"user_limit"`` when a
    budget (``max_epochs``, ``max_seconds``) stopped it, with the point
    returned; ``problem.solver_stats.extra_stats`` holds Jaddle's result.

    A mixed-integer LP goes to ``JaddleSolver``'s branch and bound; the
    compiled route has no integer variables, so a mixed-integer problem with
    ``qp_route=False``, or one that is not an LP, raises ``SolverError``.
    """
    if problem.is_mixed_integer() and not (qp_route and problem.is_qp()):
        raise cp.error.SolverError(
            "Jaddle solves mixed-integer problems only as LPs, through"
            " JaddleSolver (qp_route=True)."
        )
    if qp_route and problem.is_qp():
        return problem.solve(solver=JaddleSolver(), **options)
    compiled = compile_problem(
        problem, epigraph=epigraph, detect_bounds=detect_bounds, eliminate=eliminate
    )
    options = _with_tolerances(options, _COMPILED_TOLERANCES)
    result = jc.solve(compiled.cp, **{**COMPILED_DEFAULTS, **options})
    _unpack(problem, compiled, result)
    return problem.value
