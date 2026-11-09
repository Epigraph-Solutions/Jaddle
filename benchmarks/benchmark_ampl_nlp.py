"""Convex problems from Mittelmann's AMPL-NLP benchmark, rebuilt in Python.

Source: https://plato.asu.edu/ftp/ampl-nlp.html (models in ampl-nlp-source/,
IPOPT logs in ampl-nlp_logs/). Most of its 47 problems are nonconvex; these
are the ones whose AMPL source is convex:

* ``bearing_400``  the COPS journal bearing, a convex QP with bounds only;
* ``svanberg``     sums of a/(1 +- x) on a box, a separable convex NLP;
* ``qcqp*-c``      random convex QCQPs (PSD Gram matrices, linear equalities).

Each is solved by every Jaddle route that applies (``qp``: JaddleSolver,
``compiled``: jcv.solve's compiled JaddleCP, ``conic``: JaddleConicSolver,
``native``: a hand-written JaddleCP) and checked against Clarabel and, where
the instance is the published one, IPOPT's objective from Mittelmann's logs.
Objectives and constraint violations are recomputed in numpy from the
returned point, independently of the solver's own report.

The qcqp instances use AMPL's generator recipe with numpy's random numbers,
which AMPL's cannot be matched to, so they have the published sizes and
structure but not its numbers; their reference is Clarabel only.

Usage:
    python benchmarks/benchmark_ampl_nlp.py                    # everything
    python benchmarks/benchmark_ampl_nlp.py --small            # quick smoke run
    python benchmarks/benchmark_ampl_nlp.py svanberg qcqp500-3c --routes native conic
    python benchmarks/benchmark_ampl_nlp.py --no-reference --max-seconds 600
    python benchmarks/benchmark_ampl_nlp.py bearing_400 --device cpu   # Jaddle on the CPU
"""

import argparse
import ast
import contextlib
import csv
import io
import time

import cvxpy as cp
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sparse

import jaddle.jaddle_convex as jc
import jaddle.jaddle_cvxpy as jcv
import jaddle.jaddle_optimisers as jo
from jaddle.jaddle_basic_types import JaddleCP

from benchmark import add_jaddle_verbose_flag

jo.configure_jax("float64")

ROUTES = ("qp", "compiled", "conic", "native")


class Instance:
    """One test problem: its numpy data, a CVXPY model (built fresh per
    route, so no route sees another's cached solve), an optional native
    JaddleCP, and a numpy check of objective and constraint violation."""

    def __init__(
        self,
        name,
        n,
        m,
        build,
        evaluate,
        routes,
        native=None,
        ipopt=None,
        grad_scale=1.0,
    ):
        self.name, self.n, self.m = name, n, m
        self.build = build  # () -> (problem, x variable)
        self.evaluate = evaluate  # x -> (objective, relative max violation)
        self.routes = routes
        self.native = native  # () -> JaddleCP, or None
        self.ipopt = ipopt  # IPOPT's objective from the published log, or None
        # The compiled and native routes test stationarity in absolute terms,
        # in the problem's own units; their tolerance is tol * grad_scale,
        # with grad_scale = ||grad f(0)||_inf so the test is about relative.
        self.grad_scale = grad_scale


# %%
# bearing_400.mod (COPS journal bearing)


def bearing(nx=400, ny=400, b=10.0, e=0.1, ipopt=None):
    """``bearing_400.mod``. v lives on a (nx+2) x (ny+2) grid with the
    boundary fixed at 0 (c1-c4), so only the interior is a variable, and
    v >= 0. Both weighted difference sums are copied as written, including
    the second sum's ``2*wq[i] + 2*wq[i-1]`` weight."""
    hx, hy = 2 * np.pi / (nx + 1), 2 * b / (ny + 1)
    wq = (1 + e * np.cos(np.arange(nx + 2) * hx)) ** 3
    c = 0.5 * hx * hy / 6

    def col(i, j):  # interior column of grid point (i, j), or -1 on the boundary
        inside = (i >= 1) & (i <= nx) & (j >= 1) & (j <= ny)
        return np.where(inside, (i - 1) * ny + (j - 1), -1)

    rows, cols, vals, nrow = [], [], [], 0

    def add_differences(i, j, di, dj, h, w):
        # One row per (i, j): sqrt(c w) * (v[i+di, j+dj] - v[i, j]) / h.
        nonlocal nrow
        r = nrow + np.arange(i.size)
        scale = np.sqrt(c * w) / h
        for ci, sign in ((col(i + di, j + dj), 1.0), (col(i, j), -1.0)):
            keep = ci >= 0
            rows.append(r[keep])
            cols.append(ci[keep])
            vals.append(sign * scale[keep])
        nrow += i.size

    i1, j1 = (
        a.ravel()
        for a in np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
    )
    w1 = wq[i1] + 2 * wq[i1 + 1]
    add_differences(i1, j1, 1, 0, hx, w1)
    add_differences(i1, j1, 0, 1, hy, w1)
    i2, j2 = (
        a.ravel()
        for a in np.meshgrid(np.arange(1, nx + 2), np.arange(1, ny + 2), indexing="ij")
    )
    w2 = 2 * wq[i2] + 2 * wq[i2 - 1]
    add_differences(i2, j2, -1, 0, hx, w2)
    add_differences(i2, j2, 0, -1, hy, w2)

    n = nx * ny
    D = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(nrow, n),
    )
    ii = np.repeat(np.arange(1, nx + 1), ny)
    lin = -hx * hy * e * np.sin(ii * hx)
    H = (D.T @ D).tocsc()

    def build():
        # v >= 0 as a variable bound and the quadratic as quad_form(u, D'D):
        # the QP route then gets P, c and a box. As [u >= 0] and
        # sum_squares(D @ u), CVXPY adds n inequality rows and ~4n equality
        # rows t = D u, which a first-order method enforces only through duals.
        u = cp.Variable(n, bounds=[0, np.inf])
        return cp.Problem(cp.Minimize(cp.quad_form(u, cp.psd_wrap(H)) + lin @ u)), u

    def evaluate(u):
        return float(np.sum((D @ u) ** 2) + lin @ u), float(max(0.0, -u.min()))

    return Instance(
        f"bearing_{nx}" if nx == ny else f"bearing_{nx}x{ny}",
        n,
        0,
        build,
        evaluate,
        ("qp",),
        ipopt=ipopt,
    )


# %%
# svanberg.mod


def _svanberg_rows(n):
    """The 9 terms of each constraint as (sign, 1-based index) pairs, with
    sign +1 for 1/(1+x[j]) and -1 for 1/(1-x[j]), and each row's 1-based
    right-hand-side index. Copied from cons1-cons10 of ``svanberg.mod``,
    including cons10's ``x[i+3]`` appearing twice."""
    P, M = 1, -1
    rows = []
    for i in range(6, n - 4 + 1, 2):  # cons1
        rows.append(
            (
                [
                    (M, i - 4),
                    (P, i - 3),
                    (P, i - 2),
                    (M, i - 1),
                    (P, i),
                    (P, i + 1),
                    (M, i + 2),
                    (P, i + 3),
                    (M, i + 4),
                ],
                i,
            )
        )
    rows += [
        (
            [
                (M, 1),
                (M, 2),
                (P, 3),
                (M, 4),
                (P, 5),
                (P, n - 3),
                (M, n - 2),
                (M, n - 1),
                (P, n),
            ],
            1,
        ),
        (
            [
                (M, 1),
                (P, 2),
                (P, 3),
                (M, 4),
                (P, 5),
                (M, 6),
                (M, n - 2),
                (P, n - 1),
                (P, n),
            ],
            2,
        ),
        (
            [
                (M, 1),
                (P, 2),
                (M, 3),
                (M, 4),
                (P, 5),
                (M, 6),
                (P, 7),
                (P, n - 1),
                (M, n),
            ],
            3,
        ),
        ([(P, 1), (P, 2), (M, 3), (P, 4), (P, 5), (M, 6), (P, 7), (M, 8), (M, n)], 4),
        (
            [
                (P, 1),
                (P, n - 7),
                (M, n - 6),
                (M, n - 5),
                (P, n - 4),
                (M, n - 3),
                (M, n - 2),
                (P, n - 1),
                (M, n),
            ],
            n - 3,
        ),
        (
            [
                (P, 1),
                (M, 2),
                (M, n - 6),
                (P, n - 5),
                (P, n - 4),
                (M, n - 3),
                (P, n - 2),
                (P, n - 1),
                (M, n),
            ],
            n - 2,
        ),
        (
            [
                (P, 1),
                (M, 2),
                (P, 3),
                (P, n - 5),
                (M, n - 4),
                (M, n - 3),
                (P, n - 2),
                (M, n - 1),
                (M, n),
            ],
            n - 1,
        ),
        (
            [
                (P, 1),
                (M, 2),
                (P, 3),
                (M, 4),
                (M, n - 4),
                (P, n - 3),
                (P, n - 2),
                (M, n - 1),
                (P, n),
            ],
            n,
        ),
    ]
    for i in range(5, n - 5 + 1, 2):  # cons10
        rows.append(
            (
                [
                    (P, i - 4),
                    (M, i + 3),
                    (M, i - 2),
                    (P, i - 1),
                    (M, i),
                    (M, i + 1),
                    (P, i + 2),
                    (M, i + 3),
                    (P, i + 4),
                ],
                i,
            )
        )
    return rows


def svanberg(n=50000, ipopt=None):
    """``svanberg.mod``: every term is 1/(1 + s x[j]) with s = +-1, so the
    objective and each row are nonnegative combinations of the two vectors
    1/(1+x) and 1/(1-x), convex on the box |x| <= 0.8."""
    rows = _svanberg_rows(n)
    idx = np.array([[j - 1 for _, j in terms] for terms, _ in rows])
    sgn = np.array([[s for s, _ in terms] for terms, _ in rows], dtype=float)
    k = np.arange(1, n + 1)
    b_all = k * 5 / n + 10
    b = b_all[[r - 1 for _, r in rows]]
    a = np.where(k % 2 == 1, k * 2 / n + 1, 5 - k * 3 / n)
    s_obj = np.where(k % 2 == 1, 1.0, -1.0)  # odd i: a/(1+x), even i: a/(1-x)
    m = len(rows)

    # Rows as counts on 1/(1+x) (Mp) and 1/(1-x) (Mm); duplicates sum.
    r = np.repeat(np.arange(m), idx.shape[1])
    Mp = sparse.csr_matrix(
        ((sgn.ravel() > 0).astype(float), (r, idx.ravel())), shape=(m, n)
    )
    Mm = sparse.csr_matrix(
        ((sgn.ravel() < 0).astype(float), (r, idx.ravel())), shape=(m, n)
    )
    ap, am = np.where(s_obj > 0, a, 0.0), np.where(s_obj < 0, a, 0.0)

    def build():
        x = cp.Variable(n, bounds=[-0.8, 0.8])
        yp, ym = cp.inv_pos(1 + x), cp.inv_pos(1 - x)
        return cp.Problem(cp.Minimize(ap @ yp + am @ ym), [Mp @ yp + Mm @ ym <= b]), x

    def evaluate(x):
        viol = max(0.0, (np.abs(x).max() - 0.8))
        if np.abs(x).max() < 1:
            g = (1 / (1 + sgn * x[idx])).sum(axis=1) - b
            viol = max(viol, float(g.max(initial=0.0)))
            f = float(np.sum(a / (1 + s_obj * x)))
        else:
            f, viol = np.nan, np.inf
        return f, viol / (1 + np.abs(b).max())

    def native():
        jidx, jsgn, jb, ja, js = (jnp.asarray(v) for v in (idx, sgn, b, a, s_obj))
        return JaddleCP(
            n,
            lambda x: jnp.sum(ja / (1 + js * x)),
            lambda x: jnp.zeros(0),
            lambda x: jnp.sum(1 / (1 + jsgn * x[jidx]), axis=1) - jb,
            -0.8 * jnp.ones(n),
            0.8 * jnp.ones(n),
        )

    return Instance(
        "svanberg" if n == 50000 else f"svanberg_{n}",
        n,
        m,
        build,
        evaluate,
        ("native", "conic"),
        native=native,
        ipopt=ipopt,
        grad_scale=float(np.abs(a).max()),
    )  # grad f(0) = +-a


# %%
# qcqp*.mod


# The data blocks of the convex qcqp*.mod files (all have mq = 0, sd = 1).
QCQP_INSTANCES = {
    "qcqp500-1c": dict(n=500, ml=20, pl=100, pq=10, sq=0.01, sp=0.1, plf=0.2, pqf=0.2),
    "qcqp500-2c": dict(n=500, ml=10, pl=100, pq=70, sq=0.01, sp=0.02, plf=0.4, pqf=0.1),
    "qcqp500-3c": dict(n=500, ml=10, pl=100, pq=10, sq=0.01, sp=0.2, plf=0.1, pqf=0.2),
    "qcqp750-1c": dict(
        n=750, ml=100, pl=100, pq=45, sq=0.01, sp=0.01, plf=0.1, pqf=0.05
    ),
    "qcqp750-2c": dict(n=750, ml=25, pl=100, pq=13, sq=0.01, sp=0.2, plf=0.1, pqf=0.2),
    "qcqp1000-1c": dict(
        n=1000, ml=100, pl=50, pq=4, sq=0.01, sp=0.01, plf=0.2, pqf=0.2
    ),
    "qcqp1000-2c": dict(
        n=1000, ml=100, pl=5000, pq=7, sq=0.01, sp=0.01, plf=0.02, pqf=0.2
    ),
    "qcqp1500-1c": dict(
        n=1500, ml=500, pl=10000, pq=8, sq=0.01, sp=0.01, plf=0.01, pqf=0.2
    ),
}


def _cholesky_like(rng, n, density):
    """A sparse lower-triangular L as qcqp.mod draws LQ and LP[l]: diagonal
    U(0,1), each strict-lower entry U(-10,10) with probability ``density``."""
    i, j = np.tril_indices(n, -1)
    keep = rng.random(i.size) < density
    vals = rng.uniform(-10, 10, keep.sum())
    d = rng.random(n)
    return sparse.csr_matrix(
        (
            np.concatenate([d, vals]),
            (
                np.concatenate([np.arange(n), i[keep]]),
                np.concatenate([np.arange(n), j[keep]]),
            ),
        ),
        shape=(n, n),
    )


def qcqp(name, n, ml, pl, pq, sq, sp, plf, pqf, seed=0):
    """``qcqp.mod`` with mq = 0 and sd = 1:

        min  0.5 x'Qx + g'x   s.t.  A_eq x = b_eq,  A_lin x <= b_lin,
             a_l'x + 0.5 x'P_l x <= b_l,   Q = LQ LQ', P_l = LP_l LP_l'.

    The file says x* "should solve" it when Q is PSD, but its g has the
    inequality multipliers' sign flipped, so x* is not the optimum (IPOPT
    finds objectives near -1e7 on these); Clarabel is the reference."""
    rng = np.random.default_rng(seed)
    LQ = _cholesky_like(rng, n, sq)
    LP = [_cholesky_like(rng, n, sp) for _ in range(pq)]
    y = rng.standard_normal(ml)
    zl = np.where(rng.random(pl) < plf, rng.uniform(0, 10, pl), 0.0)
    zq = np.where(rng.random(pq) < pqf, rng.uniform(0, 10, pq), 0.0)
    A0 = rng.standard_normal((ml + pl + pq, n))
    big = np.abs(A0) == np.abs(A0).max(axis=1, keepdims=True)
    A = sparse.csr_matrix(np.where(big | (rng.random(A0.shape) < sq), A0, 0.0))
    del A0, big
    xs = rng.standard_normal(n)
    Aeq, Alin, Aq = A[:ml], A[ml : ml + pl], A[ml + pl :]

    Pxs = [L @ (L.T @ xs) for L in LP]
    g = Aeq.T @ y + Alin.T @ zl + Aq.T @ zq - LQ @ (LQ.T @ xs)
    for z, v in zip(zq, Pxs):
        g = g + z * v
    beq, blin = Aeq @ xs, Alin @ xs
    bq = Aq @ xs + 0.5 * np.array([np.sum((L.T @ xs) ** 2) for L in LP])
    bscale = 1 + max(
        np.abs(beq).max(initial=0),
        np.abs(blin).max(initial=0),
        np.abs(bq).max(initial=0),
    )

    # Each quadratic row is divided by max(1, |b_l|) (b_l is up to ~1e5): the
    # same feasible set, but unscaled, Clarabel stalls at a primal residual of
    # ~5e-4 and reports a numerical error on most of these instances.
    cq = np.maximum(1.0, np.abs(bq))

    def build():
        x = cp.Variable(n)
        cons = [Aeq @ x == beq, Alin @ x <= blin]
        cons += [
            Aq[l] @ x / cq[l] + 0.5 * cp.sum_squares(LP[l].T @ x / np.sqrt(cq[l]))
            <= bq[l] / cq[l]
            for l in range(pq)
        ]
        return cp.Problem(cp.Minimize(0.5 * cp.sum_squares(LQ.T @ x) + g @ x), cons), x

    def evaluate(x):
        f = 0.5 * np.sum((LQ.T @ x) ** 2) + g @ x
        hq = Aq @ x + 0.5 * np.array([np.sum((L.T @ x) ** 2) for L in LP]) - bq
        viol = max(
            np.abs(Aeq @ x - beq).max(initial=0),
            (Alin @ x - blin).max(initial=0),
            hq.max(initial=0),
            0.0,
        )
        return float(f), float(viol) / bscale

    # The objective is ~1e6-1e7 with a gradient to match, so an absolute
    # stationarity test of 1e-6 would ask for ~1e-11 relative accuracy.
    return Instance(
        name,
        n,
        ml + pl + pq,
        build,
        evaluate,
        ("compiled", "conic"),
        grad_scale=float(np.abs(g).max()),
    )


# %%
# Running


def instances(small):
    """Problem name -> builder; each instance's data is made only when it runs
    (qcqp1500-1c alone draws a 10508 x 1500 dense matrix)."""
    if small:
        return {
            "bearing_50": lambda: bearing(50, 50),
            "svanberg_1000": lambda: svanberg(1000),
            "qcqp500-3c": lambda: qcqp("qcqp500-3c", **QCQP_INSTANCES["qcqp500-3c"]),
        }
    # IPOPT's bearing_400 value stops on a per-bound complementarity of 4e-9,
    # which over 160,000 active bounds leaves a gap of up to ~6e-4: Clarabel's
    # optimum (-0.15481838, feasible, projected gradient 4e-7) is 1.4e-4 lower.
    out = {
        "bearing_400": lambda: bearing(400, 400, ipopt=-1.5468031724087153e-01),
        "svanberg": lambda: svanberg(50000, ipopt=8.3623820124176607e04),
    }
    out.update(
        {
            name: (lambda name=name, kw=kw: qcqp(name, **kw))
            for name, kw in QCQP_INSTANCES.items()
        }
    )
    return out


def _cvxpy_status(problem):
    stats = problem.solver_stats
    extra = (stats.extra_stats if stats is not None else None) or {}
    return problem.status, extra.get("epochs"), extra.get("solve_seconds")


def _grad_tol(inst, options):
    """``options`` with the stationarity tolerance set to tol * grad_scale."""
    tol = options.get("tol", jcv.DEFAULT_TOLERANCE)
    return dict(options, primal_grad_norm_tolerance=tol * inst.grad_scale)


def run_route(inst, route, options, verbose):
    """Solve ``inst`` by ``route``; returns (status, epochs, jaddle seconds, x)."""
    quiet = (
        contextlib.nullcontext()
        if verbose
        else contextlib.redirect_stdout(io.StringIO())
    )
    with quiet:
        if route == "native":
            opts = jcv._with_tolerances(
                _grad_tol(inst, options), jcv._COMPILED_TOLERANCES
            )
            out = jc.solve(
                inst.native(), verbose=verbose, **{**jcv.COMPILED_DEFAULTS, **opts}
            )
            status = (
                "optimal"
                if out["converged"]
                else out.get("stop_reason", "not converged")
            )
            return (
                status,
                out["epochs"],
                out["solve_seconds"],
                np.asarray(out["solution"].primal),
            )
        problem, x = inst.build()
        if route == "qp":
            jcv.solve(problem, verbose=verbose, **options)
        elif route == "compiled":
            jcv.solve(
                problem, qp_route=False, verbose=verbose, **_grad_tol(inst, options)
            )
        elif route == "conic":
            problem.solve(solver=jcv.JaddleConicSolver(), verbose=verbose, **options)
        status, epochs, secs = _cvxpy_status(problem)
        return status, epochs, secs, (None if x.value is None else np.asarray(x.value))


def reference(inst, verbose):
    """Clarabel's objective at its own point, recomputed by ``inst.evaluate``."""
    problem, x = inst.build()
    t0 = time.perf_counter()
    try:
        problem.solve(solver=cp.CLARABEL, verbose=verbose)
    except cp.error.SolverError:
        return None, "clarabel failed", time.perf_counter() - t0
    secs = time.perf_counter() - t0
    if x.value is None:
        return None, problem.status, secs
    return inst.evaluate(np.asarray(x.value))[0], problem.status, secs


FIELDS = [
    "problem",
    "n",
    "m",
    "route",
    "status",
    "epochs",
    "jaddle_s",
    "wall_s",
    "objective",
    "reference",
    "ref_source",
    "rel_gap",
    "max_viol",
    "ipopt",
]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("problems", nargs="*", help="Problems to run (default: all).")
    p.add_argument(
        "--routes",
        nargs="+",
        choices=ROUTES,
        default=list(ROUTES),
        help="Jaddle routes to try, where they apply (default: all).",
    )
    p.add_argument(
        "--tol", type=float, default=1e-6, help="Jaddle tolerance (default: 1e-6)."
    )
    p.add_argument(
        "--max-seconds", type=float, default=None, help="Per-solve time limit."
    )
    p.add_argument(
        "--small",
        action="store_true",
        help="Small instances for a smoke run: bearing 50x50, svanberg n=1000, "
        "qcqp500-3c.",
    )
    p.add_argument(
        "--no-reference",
        action="store_true",
        help="Skip the Clarabel reference solves (IPOPT values still used).",
    )
    p.add_argument(
        "--csv",
        default="benchmark_ampl_nlp_results.csv",
        help="Results CSV, rewritten after every row (default: %(default)s).",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Jaddle setting (a Python literal), passed to every route, "
        "e.g. --set update_mode='pdhg'.",
    )
    p.add_argument(
        "--device",
        choices=("cpu", "gpu", "tpu"),
        default=None,
        help="Where Jaddle runs (default: JAX's default device, the GPU when "
        "there is one). Clarabel always runs on the CPU.",
    )
    add_jaddle_verbose_flag(p, default=False)
    args = p.parse_args()

    insts = instances(args.small)
    unknown = sorted(set(args.problems) - set(insts))
    if unknown:
        p.error(
            f"unknown problem(s): {', '.join(unknown)}; choose from {', '.join(insts)}"
        )
    options = dict(tol=args.tol)
    for item in args.set:
        key, _, value = item.partition("=")
        try:
            options[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            options[key] = value  # a bare word such as pdhg
    if args.max_seconds is not None:
        options["max_seconds"] = args.max_seconds
    if args.device is not None:
        options["device"] = args.device  # every route's solver takes device=

    results = []
    print(
        f"{'problem':13s} {'route':9s} {'status':17s} {'epochs':>6s} {'jaddle_s':>9s} "
        f"{'wall_s':>8s} {'objective':>17s} {'rel_gap':>9s} {'max_viol':>9s}",
        flush=True,
    )
    for name, make in insts.items():
        if args.problems and name not in args.problems:
            continue
        t0 = time.perf_counter()
        inst = make()
        print(f"# {name}: built in {time.perf_counter() - t0:.1f}s", flush=True)
        ref, src = None, ""
        if not args.no_reference:
            print(f"# {name}: clarabel reference ...", flush=True)
            ref, status, secs = reference(inst, args.jaddle_verbose)
            src = "clarabel"
            line = (
                f"{name:13s} {'clarabel':9s} {status:17s} {'':>6s} {'':>9s} {secs:8.2f}"
            )
            if ref is not None:
                line += f" {ref:17.10e}"
                if inst.ipopt is not None:
                    gap = abs(ref - inst.ipopt) / max(1.0, abs(inst.ipopt))
                    line += f"   (vs IPOPT {inst.ipopt:.10e}: rel gap {gap:.1e})"
            print(line, flush=True)
        if ref is None and inst.ipopt is not None:
            ref, src = inst.ipopt, "ipopt"
        skipped = [r for r in inst.routes if r not in args.routes]
        if skipped:
            print(
                f"# {name}: skipping route(s) {', '.join(skipped)}, not in --routes"
                f" ({name} runs {', '.join(inst.routes)})",
                flush=True,
            )
        for route in inst.routes:
            if route not in args.routes:
                continue
            print(f"# {name}: route {route} ...", flush=True)
            t0 = time.perf_counter()
            try:
                status, epochs, secs, x = run_route(
                    inst, route, options, args.jaddle_verbose
                )
            except Exception as err:  # report and carry on with the next route
                status, epochs, secs, x = (
                    f"error: {type(err).__name__}",
                    None,
                    None,
                    None,
                )
                print(f"{name:13s} {route:9s} {status}: {err}", flush=True)
            wall = time.perf_counter() - t0
            obj, viol = inst.evaluate(x) if x is not None else (np.nan, np.nan)
            gap = abs(obj - ref) / max(1.0, abs(ref)) if ref is not None else np.nan
            row = dict(
                problem=name,
                n=inst.n,
                m=inst.m,
                route=route,
                status=status,
                epochs=epochs,
                jaddle_s=secs,
                wall_s=round(wall, 3),
                objective=obj,
                reference=ref,
                ref_source=src,
                rel_gap=gap,
                max_viol=viol,
                ipopt=inst.ipopt,
            )
            results.append(row)
            fmt = lambda v, f: format(v, f) if isinstance(v, (int, float)) else "-"
            print(
                f"{name:13s} {route:9s} {str(status)[:17]:17s} {fmt(epochs, '6d'):>6s} "
                f"{fmt(secs, '9.2f'):>9s} {wall:8.2f} {obj:17.10e} {gap:9.1e} {viol:9.1e}",
                flush=True,
            )
            with open(args.csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(results)


if __name__ == "__main__":
    main()
