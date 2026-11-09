# %% [markdown]
# # Jaddle LP Benchmark Harness
#
# Runs Jaddle's saddle-point LP solver against HiGHS-PDLP (a same-class
# first-order method) on a set of MIPLIB LP relaxations, and emits a
# README-ready markdown table plus a CSV.
#
# The MPS files are **not** shipped with Jaddle (they are large and
# gitignored). Download the instances you want from the MIPLIB website
# (https://miplib.zib.de/) and drop the `.mps` files into the `data/pdlp_set/`
# directory at the repo root. This harness globs whatever is present there.
#
# Usage:
#     python benchmarks/benchmark.py
#     python benchmarks/benchmark.py --max-mb 50          # skip huge instances
#     python benchmarks/benchmark.py --only stp3d boeing  # subset by name
#     python benchmarks/benchmark.py --tol 1e-4 --max-epochs 500

# %%
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=5"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax

jax.config.update("jax_bcoo_cusparse_lowering", True)
jax.config.update("jax_optimization_level", "O3")

import argparse
import csv
import gc
import glob
import time

import highspy as hspy
import numpy as np

import jaddle.jaddle_optimisers as jo
import jaddle.jaddle_linear as jl
import jaddle.highs_helpers as hh
import jaddle.presolve as presolve


from scan_bigm import is_bigm_cost, is_bigm_matrix, is_bigm_column

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "pdlp_set")


def add_jaddle_verbose_flag(p, default, default_text="%(default)s"):
    """Add the --jaddle-verbose / --no-jaddle-verbose flag every benchmark
    script shares, with that script's default (`default_text` describes it in
    --help when the default is not a plain boolean)."""
    p.add_argument(
        "--jaddle-verbose",
        action=argparse.BooleanOptionalAction,
        default=default,
        help="Run Jaddle with verbose=True, printing its per-epoch log and "
        "restart messages. With it off and no --max-epochs / --max-seconds "
        f"budget, each solve runs as a single device call. (default: {default_text})",
    )


def add_cost_col_floor_flag(p):
    """Add the --cost-col-floor flag every LP benchmark script shares."""
    p.add_argument(
        "--cost-col-floor",
        type=float,
        default=0.0,
        help="cost_col_floor passed to jl.solve: a costed column whose largest "
        "scaled matrix entry is below it is rescaled so that entry becomes 1 "
        "(fixes epigraph objectives, e.g. fhnw-binschedule0). 0 disables it, "
        "matching jl.solve's default; try 1e-2 to enable (default: %(default)s).",
    )


def add_norm_flag(p):
    """Add the --norm flag every LP benchmark script shares."""
    p.add_argument(
        "--norm",
        default="l2",
        choices=["l2", "inf"],
        help="Norm for Jaddle's primal / dual feasibility residuals, in both "
        "the termination test and the restart merit (jl.solve's "
        "termination_norm and restart_norm). (default: %(default)s)",
    )


def add_polish_flags(p):
    """Add the --gap-tol, --gap-test and --polish flags (feasibility polishing
    and gap-test A/Bs)."""
    p.add_argument(
        "--gap-tol",
        type=float,
        default=None,
        help="Relative duality-gap tolerance for Jaddle (default: same as --tol). "
        "Set it looser than --tol for a tight-feasibility / loose-gap "
        "certificate, the regime where --polish pays off.",
    )
    p.add_argument(
        "--gap-test",
        default="pdlp",
        choices=["abs", "pdlp"],
        help="Which relative gap must meet --gap-tol (jl.solve's gap_test): "
        "'abs' sums the |gap components| so they cannot cancel; 'pdlp' is "
        "PDLP's signed |primal_obj - dual_obj|. (default: %(default)s)",
    )
    p.add_argument(
        "--polish",
        action="store_true",
        help="Solve with jl.solve_with_polishing (PDLP feasibility polishing) "
        "instead of jl.solve. Polishing fires only once the gap is within "
        "--gap-tol, so with --gap-tol equal to --tol it rarely changes anything.",
    )
    p.add_argument(
        "--refine",
        action="store_true",
        help="Solve with jl.solve_with_refinement (iterative refinement: a loose "
        "solve, then scaled correction LPs) instead of jl.solve. Excludes --polish.",
    )


def parse_args():
    p = argparse.ArgumentParser(description="Jaddle vs HiGHS-PDLP LP benchmark.")
    p.add_argument(
        "--data-dir", default=DATA_DIR, help="Directory of .mps files to glob."
    )
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Problem names (without .mps) to restrict to. Default: all in data-dir.",
    )
    p.add_argument(
        "--max-mb",
        type=float,
        default=None,
        help="Skip .mps files larger than this many MB (default: no limit; huge "
        "instances can OOM or run for very long).",
    )
    p.add_argument(
        "--min-mb",
        type=float,
        default=0.0,
        help="Skip .mps files smaller than this many MB (default 0; raise to "
        "target larger LPs only).",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Relative optimality tolerance for both solvers (default 1e-3).",
    )
    p.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Cap Jaddle epochs (None = run to convergence).",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Per-instance Jaddle wall-clock budget in seconds, incl. scaling/"
        "setup and XLA compile (None = no limit). Checked at epoch boundaries, "
        "so a solve can overrun by up to one epoch.",
    )
    p.add_argument(
        "--csv",
        default=os.path.join(REPO_ROOT, "benchmark_results.csv"),
        help="Path to write CSV results.",
    )
    p.add_argument(
        "--highs-solver",
        default="simplex",
        choices=("simplex", "ipm", "pdlp", "none"),
        help="HiGHS solver used for the reference optimum (default simplex). "
        "Pass 'none' to skip the reference solve entirely and report NaN for the "
        "optimum (and rel_obj_gap) -- useful for very large LPs where the exact "
        "solve is prohibitively slow; jaddle still runs and its objective is "
        "reported.",
    )
    p.add_argument(
        "--skip-bigm",
        action="store_true",
        help="Skip instances with a big-M / penalty COST structure (wide "
        "dynamic range in the objective coefficients, e.g. glass4, binkar10_1). "
        "These stall the saddle-point solver at a feasible-but-suboptimal point "
        "and waste the epoch budget. See benchmarks/scan_bigm.py.",
    )
    p.add_argument(
        "--skip-bigm-matrix",
        action="store_true",
        help="Skip instances with a big-M / penalty MATRIX structure (rows where "
        "one coefficient dwarfs its rowmates, e.g. sp150x300d, neos-3754480-"
        "nidda). Like cost big-M but constraint-side: Ruiz scaling cannot flatten "
        "the within-row skew, so the saddle solver stalls on a primal-feasibility "
        "plateau. See benchmarks/scan_bigm.py.",
    )
    p.add_argument(
        "--highs-verbose",
        action="store_true",
        help="Enable HiGHS's own solve logging (output_flag=true), instead of "
        "the default silent reference solve.",
    )
    p.add_argument(
        "--highs-kkt-tolerance",
        type=float,
        default=None,
        help="KKT tolerance passed to HiGHS's reference solve (default: same "
        "value as --tol).",
    )
    p.add_argument(
        "--skip-bigm-column",
        action="store_true",
        help="Skip instances with a big-M / penalty COLUMN structure (a dense "
        "row with a huge coefficient shared across most columns, e.g. germanrr, "
        "leo1, trento1). The column-side mirror of matrix big-M: Ruiz cannot "
        "flatten within-column anisotropy threaded through one shared row, so the "
        "saddle solver never approaches primal feasibility (germanrr: PFR frozen "
        "~1e7, objective still drifting at 80 epochs). See benchmarks/scan_bigm.py.",
    )
    p.add_argument(
        "--presolve",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Presolve with HiGHS and solve the reduced LP via "
        "jl.solve_with_presolve (converged = certified on the presolved LP). "
        "--no-presolve solves the file's LP relaxation as given with jl.solve, "
        "skipping defined-variable elimination too, so converged is certified "
        "on the original LP (the no-presolve setting of the cuPDLP papers). "
        "--no-presolve also turns presolve off in the HiGHS reference solve. "
        "(default: %(default)s)",
    )
    p.add_argument(
        "--eliminate-defined-vars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After HiGHS presolve, substitute out variables defined by a dense "
        "equality row (z = sum_j a_j x_j, e.g. gmut-*, proteindesign*), which "
        "HiGHS keeps. On by default; --no-eliminate-defined-vars disables it. "
        "Ignored with --no-presolve. See "
        "jaddle.presolve.eliminate_defined_variables.",
    )
    p.add_argument(
        "--update-mode",
        default="alternating",
        choices=["alternating", "pdhg", "halpern"],
        help="Jaddle LP update_mode passed to jl.solve (default: alternating).",
    )
    p.add_argument(
        "--jax-profile",
        default="float64",
        choices=["float64", "float32", "float16"],
        help="JAX precision profile passed to jaddle_optimisers.configure_jax "
        "(default: float64).",
    )
    p.add_argument(
        "--diag-clamp",
        type=float,
        default=10.0,
        help="diag_adapt clamp passed to jl.solve: the per-coordinate diagonal "
        "scalings stay in [1/diag_clamp, diag_clamp] around the Ruiz+PC baseline "
        "(default: %(default)s).",
    )
    add_cost_col_floor_flag(p)
    add_norm_flag(p)
    add_polish_flags(p)
    add_jaddle_verbose_flag(p, default=True)
    return p.parse_args()


def highs_reference(
    path,
    highs_solver="simplex",
    tol=1e-3,
    highs_verbose=False,
    highs_kkt_tolerance=None,
    presolve=True,
):
    """Solve the LP relaxation with HiGHS for a trusted reference optimum.

    HiGHS is used here only as an objective *oracle* (ground-truth optimum), not
    as a speed competitor -- a same-class PDLP-vs-PDHG timing race is misleading
    because the two stop on different relative-gap criteria. An exact solver
    (``simplex`` / ``ipm``) is an unambiguous answer key for Jaddle's objective
    gap; ``pdlp`` is available for a same-class comparison point. ``none``
    skips the solve (it can be prohibitively slow on very large LPs).

    ``highs_verbose`` enables HiGHS's own solve logging. ``highs_kkt_tolerance``
    overrides the KKT tolerance passed to HiGHS (defaults to ``tol``).
    ``presolve=False`` turns HiGHS's presolve off, matching Jaddle's
    ``--no-presolve`` run.

    Returns ``(opt_obj, highs_status, highs_seconds)``; NaN, "skipped", NaN
    when skipped. ``highs_seconds`` times only ``run()``, the like-for-like
    counterpart to Jaddle's solve-only timer.
    """
    if highs_solver == "none":
        return float("nan"), "skipped", float("nan")
    highs = hspy.Highs()
    highs.setOptionValue("output_flag", "true" if highs_verbose else "false")
    hh.read_model(highs, path)
    relax_integrality(highs)
    highs.setOptionValue("presolve", "on" if presolve else "off")
    highs.setOptionValue(
        "kkt_tolerance", tol if highs_kkt_tolerance is None else highs_kkt_tolerance
    )
    highs.setOptionValue("solver", highs_solver)
    t0 = time.perf_counter()
    highs.run()
    highs_seconds = time.perf_counter() - t0
    return (
        highs.getInfo().objective_function_value,
        highs.modelStatusToString(highs.getModelStatus()),
        highs_seconds,
    )


def relax_integrality(highs):
    # One batched call: a per-column changeColIntegrality loop took ~110s on
    # rwth-timetable.
    n = highs.getNumCol()
    highs.changeColsIntegrality(
        n, np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.uint8)
    )


def presolved_lp(path, eliminate_defined_vars=True):
    """HiGHS-presolved LP relaxation in Jaddle's standard form, for harnesses
    that hand the same reduced LP to several solvers (benchmark_mpax.py).
    Returns ``(lp, offset)``: add ``offset`` to ``c^T x`` for the original
    objective. No postsolve: ``jl.solve_with_presolve`` is the full pipeline.
    """
    highs = hspy.Highs()
    highs.setOptionValue("output_flag", False)
    hh.read_model(highs, path)
    relax_integrality(highs)
    highs.presolve()
    highs_lp = highs.getPresolvedLp()
    lp = hh.highs_to_standard_form_sparse(highs_lp)
    offset = float(highs_lp.offset_)
    if eliminate_defined_vars:
        lp, extra_offset, _ = presolve.eliminate_defined_variables(lp, verbose=True)
        offset += extra_offset
    return lp, offset


def original_lp(path):
    """The file's LP relaxation in Jaddle's standard form, as a minimisation,
    with no presolve. Returns ``(lp, sign, offset, n_vars, n_cons)``: the
    original objective is ``sign * (lp.c @ x) + offset`` (``sign`` is -1 for a
    maximisation, whose cost is negated), and the sizes are HiGHS's."""
    highs = hspy.Highs()
    highs.setOptionValue("output_flag", False)
    hh.read_model(highs, path)
    relax_integrality(highs)
    highs_lp = highs.getLp()
    lp = hh.highs_to_standard_form_sparse(highs_lp)
    sign = -1.0 if highs_lp.sense_ == hspy.ObjSense.kMaximize else 1.0
    lp.c = sign * lp.c
    return lp, sign, float(highs_lp.offset_), highs_lp.num_col_, highs_lp.num_row_


def lp_solver(polish=False, refine=False):
    """The wrapper replacing ``jl.solve`` for the --polish / --refine arms, or
    ``None`` for plain ``jl.solve``."""
    if polish and refine:
        raise ValueError("--polish and --refine are alternatives")
    if polish:
        return jl.solve_with_polishing
    return jl.solve_with_refinement if refine else None


def jaddle_solve_kwargs(
    tol,
    max_epochs,
    update_mode,
    max_seconds=None,
    verbose=True,
    cost_col_floor=0.0,
    gap_tol=None,
    norm="l2",
    gap_test="pdlp",
    diag_clamp=10.0,
):
    """The ``jl.solve`` settings every benchmark harness uses, so all of them
    (and benchmark_mpax.py's Jaddle arm) run the same configuration.
    ``gap_tol`` (default: ``tol``) sets the duality-gap tolerance alone.
    ``norm`` (``"l2"`` or ``"inf"``) sets both the termination and the
    restart norm. ``gap_test`` is ``jl.solve``'s. ``diag_clamp`` is the
    ``diag_adapt`` clamp (``jl.solve``'s)."""
    return dict(
        max_epochs=max_epochs,
        max_seconds=max_seconds,
        verbose=verbose,
        log_every=10,
        primal_feasibility_tolerance=tol,
        dual_feasibility_tolerance=tol,
        dual_gap_tolerance=tol if gap_tol is None else gap_tol,
        gap_test=gap_test,
        update_mode=update_mode,
        cost_col_floor=cost_col_floor,
        termination_norm=norm,
        restart_norm=norm,
        diag_clamp=diag_clamp,
    )


def run_jaddle(
    lp,
    tol,
    max_epochs,
    update_mode="pdhg",
    max_seconds=None,
    verbose=True,
    cost_col_floor=0.0,
    gap_tol=None,
    polish=False,
    norm="l2",
    gap_test="pdlp",
    refine=False,
    diag_clamp=10.0,
):
    """Solve with Jaddle's saddle-point solver. Returns a dict of metrics.

    ``polish=True`` solves with ``jl.solve_with_polishing`` instead of
    ``jl.solve``; its solve time is then the whole call's wall time (no
    compile amortisation) and ``jaddle_polished`` records whether the returned
    point came from polishing. ``refine=True`` does the same with
    ``jl.solve_with_refinement``; ``jaddle_refine_rounds`` counts its rounds.

    Three times are reported:
      * ``jaddle_solve_seconds`` -- the solver's internal iterate-loop time
        (incl. first-epoch XLA compile, excl. scaling/sparse setup). This is
        the like-for-like figure against HiGHS-PDLP's own ``run()`` timer,
        which likewise excludes problem setup. Use this in the headline table.
      * ``jaddle_corrected_seconds`` -- the same solve-loop time with the one-off
        first-epoch XLA compile amortised out:
        ``n_epochs * (solve - first_epoch) / (n_epochs - 1)``. Estimates the
        steady-state runtime had every epoch run at the warm per-epoch rate.
      * ``jaddle_wall_seconds`` -- the full ``jl.solve()`` call including
        scaling and setup, for transparency.
    """

    jl.lp_summary_statistics(lp)

    t0 = time.perf_counter()
    solve = lp_solver(polish, refine) or jl.solve
    result = solve(
        lp,
        **jaddle_solve_kwargs(
            tol,
            max_epochs,
            update_mode,
            max_seconds,
            verbose,
            cost_col_floor,
            gap_tol,
            norm,
            gap_test,
            diag_clamp,
        ),
    )
    wall_seconds = time.perf_counter() - t0

    solution = result["solution"]
    converged = result["converged"]
    stop_reason = result["stop_reason"]
    solve_seconds = result["solve_seconds"]
    corrected_seconds = result["corrected_seconds"]

    obj = float(lp.objective(solution.primal))
    eq_res = float(lp.eq_slack(solution.primal))
    ineq_res = float(lp.ineq_slack(solution.primal))
    return {
        "jaddle_obj": obj,
        "jaddle_converged": bool(converged),
        # "certificate" = full LP optimality cert met; "primal_stall" = the
        # primal_stop heuristic fired (feasible but not certified optimal, so the
        # objective may be suboptimal even though converged=True); "max_epochs" /
        # "time_limit" = epoch / time budget exhausted. Disambiguates the two
        # ways converged can be True.
        "jaddle_stop_reason": stop_reason,
        "jaddle_epochs": result["epochs"],
        "jaddle_solve_seconds": solve_seconds,
        # Steady-state solve time with the first-epoch XLA compile amortised out:
        # n_epochs * (solve - first_epoch) / (n_epochs - 1). Falls back to
        # solve_seconds when there are fewer than two epochs.
        "jaddle_corrected_seconds": corrected_seconds,
        "jaddle_wall_seconds": wall_seconds,
        "jaddle_eq_res": eq_res,
        "jaddle_ineq_res": ineq_res,
        # Blank unless polish=True.
        "jaddle_polished": result["polish"]["polished"] if polish else "",
        "jaddle_polish_attempts": result["polish"]["attempts"] if polish else "",
        "jaddle_refine_rounds": result["refinement"]["rounds"] if refine else "",
    }


def run_jaddle_presolved(
    path,
    tol,
    max_epochs,
    update_mode="pdhg",
    max_seconds=None,
    verbose=True,
    cost_col_floor=0.0,
    gap_tol=None,
    polish=False,
    eliminate_defined_vars=True,
    norm="l2",
    gap_test="pdlp",
    refine=False,
    diag_clamp=10.0,
):
    """Solve an instance file with ``jl.solve_with_presolve`` (HiGHS presolve,
    optional defined-variable elimination, Jaddle, postsolve). Returns a dict
    of metrics.

    ``jaddle_converged`` is the reduced solve's certificate, as in earlier
    sweeps (the 374/383 headline). ``jaddle_original_certified`` and the
    ``jaddle_orig_*`` residuals judge the postsolved point on the LP as
    given; the two differ when presolve rescales the problem (leo1,
    proteindesign*: relative tolerances are not comparable across the
    rescaling). ``jaddle_obj`` is the original objective, offset included.

    Times: ``jaddle_solve_seconds`` / ``jaddle_corrected_seconds`` are the
    reduced solve's iterate loop (as in ``run_jaddle``),
    ``jaddle_presolve_seconds`` is HiGHS presolve, and
    ``jaddle_wall_seconds`` the whole call, postsolve included.
    """
    t0 = time.perf_counter()
    result = jl.solve_with_presolve(
        path,
        eliminate_defined_vars=eliminate_defined_vars,
        reduced_solver=lp_solver(polish, refine),
        **jaddle_solve_kwargs(
            tol,
            max_epochs,
            update_mode,
            max_seconds,
            verbose,
            cost_col_floor,
            gap_tol,
            norm,
            gap_test,
            diag_clamp,
        ),
    )
    wall_seconds = time.perf_counter() - t0
    presolve_info = result["presolve"]
    cert = result["certificate"] or {}
    return {
        "n_vars": presolve_info["cols"][0],
        "n_cons": presolve_info["rows"][0],
        "n_vars_presolved": presolve_info["cols"][1],
        "n_cons_presolved": presolve_info["rows"][1],
        "jaddle_obj": result["objective"],
        "jaddle_converged": bool(result.get("reduced_converged", result["converged"])),
        "jaddle_original_certified": bool(result["converged"]),
        # "certificate" = full LP optimality cert met; "primal_stall" = the
        # primal_stop heuristic fired (feasible but not certified optimal);
        # "max_epochs" / "time_limit" = budget exhausted; "presolve_*" = HiGHS
        # presolve settled the problem itself.
        "jaddle_stop_reason": result["stop_reason"],
        "jaddle_epochs": result["epochs"],
        "jaddle_solve_seconds": result.get("solve_seconds", 0.0),
        "jaddle_corrected_seconds": result.get("corrected_seconds", 0.0),
        "jaddle_presolve_seconds": presolve_info["seconds"],
        "jaddle_wall_seconds": wall_seconds,
        "jaddle_orig_pfr": cert.get("relative_primal_feasibility_residual", ""),
        "jaddle_orig_dfr": cert.get("relative_dual_feasibility_residual", ""),
        "jaddle_orig_gap": cert.get("relative_gap_abs", ""),
        # Blank unless polish=True.
        "jaddle_polished": (
            result["polish"]["polished"] if polish and "polish" in result else ""
        ),
        "jaddle_polish_attempts": (
            result["polish"]["attempts"] if polish and "polish" in result else ""
        ),
        "jaddle_refine_rounds": (
            result["refinement"]["rounds"] if refine and "refinement" in result else ""
        ),
    }


def run_jaddle_unpresolved(
    path,
    tol,
    max_epochs,
    update_mode="pdhg",
    max_seconds=None,
    verbose=True,
    cost_col_floor=0.0,
    gap_tol=None,
    polish=False,
    norm="l2",
    gap_test="pdlp",
    refine=False,
    diag_clamp=10.0,
):
    """Solve an instance file's LP relaxation as given (no presolve) with
    ``run_jaddle``. The certificate is then on the original LP, so
    ``jaddle_original_certified`` equals ``jaddle_converged``, and
    ``jaddle_obj`` is the original objective, offset included."""
    lp, sign, offset, n_vars, n_cons = original_lp(path)
    res = run_jaddle(
        lp,
        tol,
        max_epochs,
        update_mode,
        max_seconds=max_seconds,
        verbose=verbose,
        cost_col_floor=cost_col_floor,
        gap_tol=gap_tol,
        polish=polish,
        norm=norm,
        gap_test=gap_test,
        refine=refine,
        diag_clamp=diag_clamp,
    )
    res["jaddle_obj"] = sign * res["jaddle_obj"] + offset
    res["jaddle_original_certified"] = res["jaddle_converged"]
    res.update(n_vars=n_vars, n_cons=n_cons)
    return res


def rel_obj_gap(jaddle_obj, opt_obj):
    """Relative gap to the exact optimum |jaddle - opt| / (1 + |opt|),
    PDLP-convention normalisation."""
    return abs(jaddle_obj - opt_obj) / (1.0 + abs(opt_obj))


def discover_instances(args):
    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.mps")))
    instances = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        if args.only and name not in args.only:
            continue
        size_mb = os.path.getsize(path) / 1e6
        if args.max_mb and size_mb > args.max_mb:
            print(f"  skip {name} ({size_mb:.0f} MB > --max-mb {args.max_mb:.0f})")
            continue
        if args.min_mb and size_mb < args.min_mb:
            print(f"  skip {name} ({size_mb:.1f} MB < --min-mb {args.min_mb:.1f})")
            continue
        if args.skip_bigm and is_bigm_cost(path):
            print(f"  skip {name} (big-M cost structure; --skip-bigm)")
            continue
        if args.skip_bigm_matrix and is_bigm_matrix(path):
            print(f"  skip {name} (big-M matrix structure; --skip-bigm-matrix)")
            continue
        if args.skip_bigm_column and is_bigm_column(path):
            print(f"  skip {name} (big-M column structure; --skip-bigm-column)")
            continue
        instances.append((name, path, size_mb))
    return instances


def main():
    args = parse_args()
    jo.configure_jax(args.jax_profile)

    instances = discover_instances(args)
    if not instances:
        print(
            f"No .mps files to run in {args.data_dir}. Download MIPLIB instances "
            "(https://miplib.zib.de/) into data/pdlp_set/ and retry."
        )
        return

    gap_text = "" if args.gap_tol is None else f", gap_tol={args.gap_tol:g}"
    if args.gap_test != "pdlp":
        gap_text += f", gap_test={args.gap_test}"
    polish_text = ", feasibility polishing" if args.polish else ""
    if args.refine:
        if args.polish:
            raise SystemExit("--polish and --refine are alternatives")
        polish_text = ", iterative refinement"
    presolve_text = "" if args.presolve else ", no presolve"
    print(
        f"Running {len(instances)} instance(s) at tol={args.tol:g}"
        f"{gap_text}, norm={args.norm}{polish_text}{presolve_text}\n"
    )
    rows = []
    for name, path, size_mb in instances:
        print(f"=== {name} ({size_mb:.1f} MB) ===")
        row = {"problem": name, "size_mb": round(size_mb, 1)}
        try:
            opt_obj, highs_status, highs_seconds = highs_reference(
                path,
                highs_solver=args.highs_solver,
                tol=args.tol,
                highs_verbose=args.highs_verbose,
                highs_kkt_tolerance=args.highs_kkt_tolerance,
                presolve=args.presolve,
            )
            row.update(
                {
                    "opt_obj": opt_obj,
                    "highs_status": highs_status,
                    "highs_solve_seconds": highs_seconds,
                }
            )
            jaddle_kwargs = dict(
                max_seconds=args.max_seconds,
                verbose=args.jaddle_verbose,
                cost_col_floor=args.cost_col_floor,
                gap_tol=args.gap_tol,
                polish=args.polish,
                norm=args.norm,
                gap_test=args.gap_test,
                refine=args.refine,
                diag_clamp=args.diag_clamp,
            )
            if args.presolve:
                jres = run_jaddle_presolved(
                    path,
                    args.tol,
                    args.max_epochs,
                    args.update_mode,
                    eliminate_defined_vars=args.eliminate_defined_vars,
                    **jaddle_kwargs,
                )
            else:
                jres = run_jaddle_unpresolved(
                    path, args.tol, args.max_epochs, args.update_mode, **jaddle_kwargs
                )
            row.update(jres)
            row["rel_obj_gap"] = (
                rel_obj_gap(jres["jaddle_obj"], opt_obj)
                if jres["jaddle_obj"] is not None
                else float("nan")
            )
            row["error"] = ""
            highs_time_str = (
                "skipped"
                if args.highs_solver == "none"
                else f"solve={highs_seconds:.2f}s"
            )
            obj_str = "—" if jres["jaddle_obj"] is None else f"{jres['jaddle_obj']:.6g}"
            print(
                f"  optimum (HiGHS {args.highs_solver}): {opt_obj:.6g} "
                f"({highs_time_str})  |  "
                f"Jaddle: obj={obj_str} "
                f"(solve={jres['jaddle_solve_seconds']:.2f}s, "
                f"corrected={jres['jaddle_corrected_seconds']:.2f}s, "
                f"wall={jres['jaddle_wall_seconds']:.2f}s, "
                f"converged={jres['jaddle_converged']}, "
                f"original_certified={jres['jaddle_original_certified']}, "
                + (f"polished={jres['jaddle_polished']}, " if args.polish else "")
                + f"rel_gap={row['rel_obj_gap']:.2e})"
            )
        except Exception as exc:  # keep the run going if one instance blows up
            row["error"] = repr(exc)
            print(f"  ERROR: {exc!r}")
        rows.append(row)
        # Rewrite the CSV after every instance so a crash mid-sweep keeps results.
        write_csv(args.csv, rows)
        # Drop JAX's in-memory trace/compile caches: every instance has new
        # shapes and closures, so nothing carries over, and keeping them grows
        # host RSS by ~100 MB per instance over a long sweep.
        jax.clear_caches()
        gc.collect()
        print()

    write_csv(args.csv, rows)
    print_markdown(rows, highs_solver=args.highs_solver, presolve=args.presolve)
    print(f"\nCSV written to {args.csv}")


CSV_FIELDS = [
    "problem",
    "size_mb",
    "n_vars",
    "n_cons",
    "n_vars_presolved",
    "n_cons_presolved",
    "opt_obj",
    "highs_status",
    "highs_solve_seconds",
    "jaddle_obj",
    "jaddle_converged",
    "jaddle_original_certified",
    "jaddle_stop_reason",
    "jaddle_epochs",
    "jaddle_solve_seconds",
    "jaddle_corrected_seconds",
    "jaddle_presolve_seconds",
    "jaddle_wall_seconds",
    "jaddle_orig_pfr",
    "jaddle_orig_dfr",
    "jaddle_orig_gap",
    "jaddle_polished",
    "jaddle_polish_attempts",
    "jaddle_refine_rounds",
    "rel_obj_gap",
    "error",
]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _fmt(x, spec="{:.4g}"):
    if x == "" or x is None:
        return "—"
    try:
        return spec.format(x)
    except (ValueError, TypeError):
        return str(x)


def print_markdown(rows, highs_solver="simplex", presolve=True):
    print("\n## Benchmark Results\n")
    if highs_solver == "none":
        oracle = (
            "Optimum is not computed (--highs-solver none), so rel_obj_gap is "
            "unavailable."
        )
    else:
        oracle = (
            f"Optimum is HiGHS's {highs_solver} solver, used as a ground-truth "
            "objective oracle."
        )
    if presolve:
        certified = (
            "Converged = certified on the presolved LP; Original certified = "
            "the postsolved point certified on the LP as given."
        )
    else:
        certified = (
            "No presolve: Converged = Original certified = certified on the LP "
            "as given."
        )
    print(
        f"_{oracle} Jaddle solve time is solve-only (iterate loop incl. "
        "first-epoch XLA compile, excl. setup/scaling); corrected time amortises "
        "the one-off first-epoch compile out "
        "(`n·(solve−first)/(n−1)`); see `jaddle_wall_seconds` in the CSV for full "
        f"call time. {certified}_\n"
    )
    print(
        "| Problem | Vars | Cons | Optimum | HiGHS solve (s) | "
        "Jaddle obj | Jaddle solve (s) | Jaddle corrected (s) | "
        "Converged | Original certified | Stop | Rel. gap to opt |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|---:|")
    for r in rows:
        if r.get("error"):
            print(
                f"| {r['problem']} | — | — | — | — | — | — | — | ⚠️ error | — | — | — |"
            )
            continue
        conv = "✅" if r.get("jaddle_converged") else "❌"
        orig = "✅" if r.get("jaddle_original_certified") else "❌"
        print(
            f"| {r['problem']} | {_fmt(r.get('n_vars'), '{:d}')} | "
            f"{_fmt(r.get('n_cons'), '{:d}')} | {_fmt(r.get('opt_obj'))} | "
            f"{_fmt(r.get('highs_solve_seconds'), '{:.2f}')} | "
            f"{_fmt(r.get('jaddle_obj'))} | "
            f"{_fmt(r.get('jaddle_solve_seconds'), '{:.2f}')} | "
            f"{_fmt(r.get('jaddle_corrected_seconds'), '{:.2f}')} | {conv} | {orig} | "
            f"{r.get('jaddle_stop_reason') or '—'} | "
            f"{_fmt(r.get('rel_obj_gap'), '{:.2e}')} |"
        )


if __name__ == "__main__":
    main()

# %%
