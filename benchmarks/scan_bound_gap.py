# %% [markdown]
# # Node bound gap scanner
#
# How far the Lagrangian bound of Jaddle's approximate node duals sits below
# the node LP's optimum, and how much of that `DualRepair` and `DualPolish`
# (dual ascent on the bound) win back.
#
# For each instance: the root box, then boxes like a search's nodes (a share
# of the integers fixed at the rounded root LP point, propagated). Each box's
# LP is solved by Jaddle at the node tolerance and by HiGHS exactly, and the
# bound of Jaddle's duals over the box is printed raw, repaired, and polished
# for 1, 3 and 10 passes, as the share of the shortfall (HiGHS's optimum less
# the raw bound) each recovers.
#
# Usage:
#     python benchmarks/scan_bound_gap.py --only CMS750_4 air05
#     python benchmarks/scan_bound_gap.py --data-dir data/popso --boxes 6

# %%
import argparse
import glob
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import jax.numpy as jnp
import numpy as np
from scipy.optimize import linprog

from jaddle import jaddle_linear as jl
from jaddle import jaddle_mip as jm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEPS = (1, 3, 10)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(REPO_ROOT, "data", "popso"))
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--boxes", type=int, default=4, help="node-like boxes an instance")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def exact_lp(mip, lower, upper):
    """HiGHS's optimum of the LP over a box (``None`` when it has none)."""
    n_eq = mip.n_eq
    A, b = mip.A, mip.b
    out = linprog(
        mip.c,
        A_ub=A[n_eq:] if A.shape[0] > n_eq else None,
        b_ub=b[n_eq:] if A.shape[0] > n_eq else None,
        A_eq=A[:n_eq] if n_eq else None,
        b_eq=b[:n_eq] if n_eq else None,
        bounds=np.column_stack(
            [np.where(np.isfinite(lower), lower, -np.inf), np.where(np.isfinite(upper), upper, np.inf)]
        ),
        method="highs",
    )
    return float(out.fun) if out.status == 0 else None


def boxes(mip, x, count, rng):
    """The root box, then ``count`` boxes with a growing share of the
    integers fixed at ``x`` rounded, propagated (infeasible ones skipped)."""
    is_int = mip.integrality
    propagate = jm.BoundPropagator(mip.A, mip.row_lower, mip.row_upper, is_int)
    yield "root", mip.lower.copy(), mip.upper.copy()
    ints = np.flatnonzero(is_int)
    for k in range(count):
        share = 0.6 * (k + 1) / count
        lower, upper = mip.lower.copy(), mip.upper.copy()
        cols = rng.choice(ints, int(share * ints.size), replace=False)
        value = np.clip(np.round(x[cols]), lower[cols], upper[cols])
        lower[cols] = upper[cols] = value
        lower, upper, infeasible = propagate(lower, upper)
        if not infeasible:
            yield f"fix {share:.0%}", lower, upper


def scan(path, args):
    name = os.path.basename(path).split(".")[0]
    mip = jm.load(path, presolve=True)
    is_int = mip.integrality
    A_T = mip.A.T.tocsr()
    values = mip.lp.values()
    dtype = mip.lp.c.dtype
    repair = jm.DualRepair(mip, is_int)
    polish = jm.DualPolish(mip, is_int)
    tol = args.tolerance
    rc_tol = tol * (1.0 + np.linalg.norm(mip.c))
    rng = np.random.default_rng(args.seed)
    print(f"\n{name}: {mip.A.shape[0]} rows, {mip.c.size} columns, {int(is_int.sum())} integer")

    def bound(y, lower, upper):
        constant, r = jm.lagrangian_terms(mip.c, mip.A, mip.b, mip.n_eq, y, A_T=A_T)
        return jm.box_bound(constant, r, lower, upper, reduced_cost_tolerance=rc_tol)

    def solve(lower, upper, warm=None):
        lp = mip.lp.with_values(
            values._replace(
                lower_bounds=jnp.asarray(np.where(is_int, lower, mip.lower), dtype),
                upper_bounds=jnp.asarray(np.where(is_int, upper, mip.upper), dtype),
            )
        )
        kwargs = {} if warm is None else dict(initial_solution=warm[0], initial_opt_state=warm[1])
        return jl.solve(
            lp,
            max_epochs=args.max_epochs,
            primal_feasibility_tolerance=tol,
            dual_feasibility_tolerance=tol,
            dual_gap_tolerance=tol,
            **kwargs,
        )

    root = solve(mip.lower, mip.upper)
    x = np.asarray(root["solution"].primal, np.float64)
    warm = (root["solution"], root["opt_state"])
    header = f"  {'box':>8} {'exact LP':>14} {'raw':>14} {'short':>10} {'repair':>7}"
    header += "".join(f" {'pol' + str(s):>7}" for s in SWEEPS) + f" {'ms':>6} {'rig':>7}"
    print(header)
    rows = []
    for label, lower, upper in boxes(mip, x, args.boxes, rng):
        result = root if label == "root" else solve(lower, upper, warm)
        solution = result["solution"]
        m_ineq = mip.A.shape[0] - mip.n_eq
        y = np.concatenate(
            [
                np.asarray(solution.dual_eq, np.float64)[: mip.n_eq],
                np.asarray(solution.dual_ineq, np.float64)[:m_ineq],
            ]
        )
        exact = exact_lp(mip, lower, upper)
        raw, rigorous = bound(y, lower, upper)
        if exact is None:
            print(f"  {label:>8} {'infeasible':>14} {raw:14.6g}")
            continue
        short = exact - raw
        start, best = y, raw
        if repair:
            repaired = repair(mip.c, A_T, y, solution.primal, lower, upper)
            if repaired is not None and bound(repaired, lower, upper)[0] > raw:
                start, best = repaired, bound(repaired, lower, upper)[0]
        shares = [(best - raw) / short if short > 0 and np.isfinite(short) else np.nan]
        after = rigorous
        for sweeps in SWEEPS:
            started = time.perf_counter()
            polished = polish(start, lower, upper, sweeps=sweeps)
            ms = 1e3 * (time.perf_counter() - started)
            value, after = (best, rigorous) if polished is None else bound(polished, lower, upper)
            value = max(value, best)
            shares.append((value - raw) / short if short > 0 and np.isfinite(short) else np.nan)
        assert value <= exact + 1e-6 * (1.0 + abs(exact)) + rc_tol, (value, exact)
        rows.append(shares)
        print(
            f"  {label:>8} {exact:14.6g} {raw:14.6g} {short:10.3g} "
            + " ".join(f"{100 * s:6.1f}%" for s in shares)
            + f" {ms:6.1f} {str(rigorous)[0]}->{str(after)[0]}"
        )
    return name, rows


def main():
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.mps*")))
    if args.only:
        paths = [p for p in paths if os.path.basename(p).split(".")[0] in args.only]
    for path in paths:
        try:
            scan(path, args)
        except Exception as error:  # one instance does not end the scan
            print(f"  failed: {type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
