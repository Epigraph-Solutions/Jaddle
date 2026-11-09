# %% [markdown]
# # Jaddle Full-Pipeline LP Benchmark
#
# Runs the whole of ``jl.solve_with_presolve`` on a set of MIPLIB LP
# relaxations: HiGHS presolve, defined-variable elimination, Jaddle with PDLP
# feasibility polishing on the reduced LP, crossover to an optimal vertex of
# the reduced LP, and HiGHS's postsolve of that vertex and its basis back to
# the original problem. The answer is a vertex, so the reference is an exact
# HiGHS solve, timed end to end against Jaddle's whole call.
#
# Emits a README-ready markdown table plus a CSV. The MPS files are not shipped
# with Jaddle: put them in ``data/pdlp_set/`` at the repo root (see benchmarks/README.md).
#
# Usage:
#     python benchmarks/benchmark_pipeline.py
#     python benchmarks/benchmark_pipeline.py --only stp3d mzzv11
#     python benchmarks/benchmark_pipeline.py --tol 1e-4 --max-seconds 600
#     python benchmarks/benchmark_pipeline.py --no-crossover   # polish only
#     python benchmarks/benchmark_pipeline.py --highs-solver none

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
import time

import jaddle.jaddle_optimisers as jo
import jaddle.jaddle_linear as jl

from benchmark import (
    DATA_DIR,
    REPO_ROOT,
    _fmt,
    add_cost_col_floor_flag,
    add_jaddle_verbose_flag,
    add_norm_flag,
    discover_instances,
    highs_reference,
    jaddle_solve_kwargs,
    rel_obj_gap,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Jaddle full pipeline (presolve, polish, crossover, "
        "postsolve) vs an exact HiGHS solve."
    )
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
        help="Skip .mps files larger than this many MB (default: no limit).",
    )
    p.add_argument(
        "--min-mb",
        type=float,
        default=0.0,
        help="Skip .mps files smaller than this many MB (default 0).",
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Relative tolerance of Jaddle's first-order solve (primal, dual and, "
        "unless --gap-tol is given, gap). Crossover makes the answer exact "
        "whatever it is; a tighter one costs Jaddle time and usually saves "
        "simplex repair pivots. (default: %(default)s)",
    )
    p.add_argument(
        "--gap-tol",
        type=float,
        default=None,
        help="Relative duality-gap tolerance for Jaddle (default: same as --tol). "
        "Polishing fires only once the gap is within it, so a looser gap than "
        "--tol is the regime where polishing pays off.",
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
        help="Per-instance Jaddle wall-clock budget in seconds for the reduced "
        "solve (None = no limit). Presolve, crossover and postsolve are not "
        "counted against it.",
    )
    p.add_argument(
        "--polish",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Solve the reduced LP with feasibility polishing "
        "(solve_with_presolve's polish=True). (default: %(default)s)",
    )
    p.add_argument(
        "--first-polish-epoch",
        type=int,
        default=16,
        help="Earliest epoch a polishing attempt may fire (default: %(default)s).",
    )
    p.add_argument(
        "--polish-fraction",
        type=float,
        default=0.125,
        help="Length of each polishing sub-solve as a fraction of the main "
        "solve's epochs so far (default: %(default)s).",
    )
    p.add_argument(
        "--polish-trigger",
        default="doubling",
        choices=["doubling", "auto"],
        help="When polishing is checked: PDLP's doubling schedule or every "
        "epoch (default: %(default)s).",
    )
    p.add_argument(
        "--crossover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cross the reduced solve's point over to a vertex before postsolve "
        "(solve_with_presolve's run_crossover=True). (default: %(default)s)",
    )
    p.add_argument(
        "--crossover-time-limit",
        type=float,
        default=None,
        help="HiGHS time_limit in seconds for the crossover and its simplex "
        "repair, passed as highs_options (it then applies to the presolving "
        "HiGHS instance too). Default: no limit.",
    )
    p.add_argument(
        "--eliminate-defined-vars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After HiGHS presolve, substitute out variables defined by a dense "
        "equality row (see jaddle.presolve.eliminate_defined_variables). "
        "(default: %(default)s)",
    )
    p.add_argument(
        "--update-mode",
        default="alternating",
        choices=["alternating", "pdhg", "halpern"],
        help="Jaddle LP update_mode passed to jl.solve (default: %(default)s).",
    )
    p.add_argument(
        "--jax-profile",
        default="float64",
        choices=["float64", "float32", "float16"],
        help="JAX precision profile passed to jaddle_optimisers.configure_jax "
        "(default: %(default)s).",
    )
    p.add_argument(
        "--highs-solver",
        default="simplex",
        choices=("simplex", "ipm", "pdlp", "none"),
        help="HiGHS solver for the reference optimum and timing (default "
        "simplex). 'none' skips it and reports NaN for the optimum.",
    )
    p.add_argument(
        "--highs-kkt-tolerance",
        type=float,
        default=None,
        help="KKT tolerance passed to HiGHS's reference solve (default: same "
        "value as --tol).",
    )
    p.add_argument(
        "--highs-verbose",
        action="store_true",
        help="Enable HiGHS's own logging in the reference solve.",
    )
    p.add_argument(
        "--skip-bigm",
        action="store_true",
        help="Skip instances with a big-M cost structure (see scan_bigm.py).",
    )
    p.add_argument(
        "--skip-bigm-matrix",
        action="store_true",
        help="Skip instances with a big-M matrix (row) structure.",
    )
    p.add_argument(
        "--skip-bigm-column",
        action="store_true",
        help="Skip instances with a big-M column structure.",
    )
    p.add_argument(
        "--csv",
        default=os.path.join(REPO_ROOT, "benchmark_pipeline_results.csv"),
        help="Path to write CSV results.",
    )
    add_cost_col_floor_flag(p)
    add_norm_flag(p)
    add_jaddle_verbose_flag(p, default=True)
    return p.parse_args()


def run_pipeline(path, args):
    """Run ``jl.solve_with_presolve`` on one instance file with the flags in
    ``args``. Returns a dict of metrics.

    ``jaddle_original_certified`` is the certificate of the returned point on
    the LP as given; with crossover that point is the postsolved vertex, so
    it should pass whenever the crossover proved optimality
    (``crossover_converged``). ``jaddle_converged`` is the reduced first-order
    solve's own verdict.

    Times: ``jaddle_presolve_seconds`` is HiGHS presolve,
    ``jaddle_solve_seconds`` the reduced solve (with polishing, that whole
    call's wall time), ``crossover_seconds`` the crossover and its simplex
    repair, ``jaddle_other_seconds`` everything else (defined-variable
    elimination, postsolve and its own simplex repair, certificate check) and
    ``jaddle_wall_seconds`` the whole call.
    """
    kwargs = jaddle_solve_kwargs(
        args.tol,
        args.max_epochs,
        args.update_mode,
        args.max_seconds,
        args.jaddle_verbose,
        args.cost_col_floor,
        args.gap_tol,
        args.norm,
    )
    if args.polish:
        kwargs.update(
            first_polish_epoch=args.first_polish_epoch,
            polish_fraction=args.polish_fraction,
            polish_trigger=args.polish_trigger,
        )
    highs_options = (
        None
        if args.crossover_time_limit is None
        else {"time_limit": args.crossover_time_limit}
    )

    t0 = time.perf_counter()
    result = jl.solve_with_presolve(
        path,
        highs_options=highs_options,
        eliminate_defined_vars=args.eliminate_defined_vars,
        polish=args.polish,
        run_crossover=args.crossover,
        diag_clamp=1e10,
        iterations_per_epoch=64,
        **kwargs,
    )
    wall_seconds = time.perf_counter() - t0

    presolve_info = result["presolve"]
    cert = result["certificate"] or {}
    polish = result.get("polish")
    crossed = result.get("crossover")
    solve_seconds = result.get("solve_seconds", 0.0)
    crossover_seconds = crossed["seconds"] if crossed else 0.0
    row = {
        "n_vars": presolve_info["cols"][0],
        "n_cons": presolve_info["rows"][0],
        "n_vars_presolved": presolve_info["cols"][1],
        "n_cons_presolved": presolve_info["rows"][1],
        "jaddle_obj": result["objective"],
        "jaddle_converged": bool(result.get("reduced_converged", result["converged"])),
        "jaddle_original_certified": bool(result["converged"]),
        "jaddle_stop_reason": result["stop_reason"],
        "jaddle_epochs": result["epochs"],
        "jaddle_presolve_seconds": presolve_info["seconds"],
        "jaddle_solve_seconds": solve_seconds,
        "crossover_seconds": crossover_seconds,
        "jaddle_other_seconds": (
            wall_seconds - presolve_info["seconds"] - solve_seconds - crossover_seconds
        ),
        "jaddle_wall_seconds": wall_seconds,
        "jaddle_orig_pfr": cert.get("relative_primal_feasibility_residual", ""),
        "jaddle_orig_dfr": cert.get("relative_dual_feasibility_residual", ""),
        "jaddle_orig_gap": cert.get("relative_gap_abs", ""),
    }
    # Blank when the step was off or presolve left nothing to solve.
    if polish:
        row.update(
            jaddle_polished=polish["polished"],
            jaddle_polish_attempts=polish["attempts"],
            jaddle_polish_epochs=polish["epochs"],
        )
    if crossed:
        row.update(
            crossover_status=crossed["status"],
            crossover_converged=crossed["converged"],
            crossover_highs_status=crossed["crossover_status"],
            crossover_pushes=crossed["pushes"],
            crossover_simplex_iterations=crossed["simplex_iterations"],
            postsolve_simplex_iterations=crossed["postsolve_simplex_iterations"],
        )
    if args.crossover:
        row["basis_valid"] = result.get("basis") is not None
    return row


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

    steps = ["presolve"]
    steps.append("polished solve" if args.polish else "solve")
    if args.crossover:
        steps.append("crossover")
    steps.append("postsolve")
    gap_text = "" if args.gap_tol is None else f", gap_tol={args.gap_tol:g}"
    print(
        f"Running {len(instances)} instance(s) at tol={args.tol:g}{gap_text}, "
        f"norm={args.norm}: {' -> '.join(steps)}\n"
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
            )
            row.update(
                opt_obj=opt_obj,
                highs_status=highs_status,
                highs_solve_seconds=highs_seconds,
            )
            jres = run_pipeline(path, args)
            row.update(jres)
            row["rel_obj_gap"] = (
                rel_obj_gap(jres["jaddle_obj"], opt_obj)
                if jres["jaddle_obj"] is not None
                else float("nan")
            )
            row["error"] = ""
            highs_time_str = (
                "skipped" if args.highs_solver == "none" else f"{highs_seconds:.2f}s"
            )
            obj_str = "—" if jres["jaddle_obj"] is None else f"{jres['jaddle_obj']:.6g}"
            print(
                f"  optimum (HiGHS {args.highs_solver}): {opt_obj:.6g} "
                f"({highs_time_str})  |  Jaddle: obj={obj_str} "
                f"(presolve={jres['jaddle_presolve_seconds']:.2f}s, "
                f"solve={jres['jaddle_solve_seconds']:.2f}s, "
                f"crossover={jres['crossover_seconds']:.2f}s, "
                f"other={jres['jaddle_other_seconds']:.2f}s, "
                f"wall={jres['jaddle_wall_seconds']:.2f}s, "
                f"stop={jres['jaddle_stop_reason']}, "
                + (
                    f"polished={jres['jaddle_polished']}, "
                    if "jaddle_polished" in jres
                    else ""
                )
                + (
                    f"crossover={jres['crossover_status']} "
                    f"({jres['crossover_simplex_iterations']}+"
                    f"{jres['postsolve_simplex_iterations']} pivots), "
                    if "crossover_status" in jres
                    else ""
                )
                + f"original_certified={jres['jaddle_original_certified']}, "
                f"rel_gap={row['rel_obj_gap']:.2e})"
            )
        except Exception as exc:  # keep the run going if one instance blows up
            row["error"] = repr(exc)
            print(f"  ERROR: {exc!r}")
        rows.append(row)
        # Rewrite the CSV after every instance so a crash mid-sweep keeps results.
        write_csv(args.csv, rows)
        # Every instance has new shapes; drop JAX's caches to keep host RSS flat.
        jax.clear_caches()
        gc.collect()
        print()

    write_csv(args.csv, rows)
    print_markdown(rows, args)
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
    "jaddle_polished",
    "jaddle_polish_attempts",
    "jaddle_polish_epochs",
    "crossover_status",
    "crossover_converged",
    "crossover_highs_status",
    "crossover_pushes",
    "crossover_simplex_iterations",
    "postsolve_simplex_iterations",
    "basis_valid",
    "jaddle_presolve_seconds",
    "jaddle_solve_seconds",
    "crossover_seconds",
    "jaddle_other_seconds",
    "jaddle_wall_seconds",
    "jaddle_orig_pfr",
    "jaddle_orig_dfr",
    "jaddle_orig_gap",
    "rel_obj_gap",
    "error",
]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _tick(value):
    if value == "" or value is None:
        return "—"
    return "✅" if value else "❌"


def print_markdown(rows, args):
    print("\n## Full-Pipeline Benchmark Results\n")
    if args.highs_solver == "none":
        oracle = "No HiGHS reference (--highs-solver none), so rel. gap is unavailable."
    else:
        oracle = (
            f"Optimum and HiGHS time are HiGHS's {args.highs_solver} solver on "
            "the original file (its own presolve included)."
        )
    print(
        f"_{oracle} Jaddle wall is the whole `jl.solve_with_presolve` call: "
        "presolve, reduced solve"
        + (" with feasibility polishing" if args.polish else "")
        + (", crossover" if args.crossover else "")
        + ", postsolve and the certificate check, incl. XLA compile. "
        "Pivots = crossover repair + postsolve repair simplex iterations. "
        "Certified = the returned point certified on the LP as given._\n"
    )
    print(
        "| Problem | Vars | Cons | Optimum | HiGHS (s) | Jaddle obj | "
        "Solve (s) | Crossover (s) | Wall (s) | Polished | Pivots | "
        "Certified | Rel. gap to opt |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|:---:|---:|")
    for r in rows:
        if r.get("error"):
            print(
                f"| {r['problem']} | — | — | — | — | — | — | — | — | — | — | "
                "⚠️ error | — |"
            )
            continue
        pivots = (
            r["crossover_simplex_iterations"] + r["postsolve_simplex_iterations"]
            if "crossover_simplex_iterations" in r
            else ""
        )
        print(
            f"| {r['problem']} | {_fmt(r.get('n_vars'), '{:d}')} | "
            f"{_fmt(r.get('n_cons'), '{:d}')} | {_fmt(r.get('opt_obj'))} | "
            f"{_fmt(r.get('highs_solve_seconds'), '{:.2f}')} | "
            f"{_fmt(r.get('jaddle_obj'))} | "
            f"{_fmt(r.get('jaddle_solve_seconds'), '{:.2f}')} | "
            f"{_fmt(r.get('crossover_seconds'), '{:.2f}')} | "
            f"{_fmt(r.get('jaddle_wall_seconds'), '{:.2f}')} | "
            f"{_tick(r.get('jaddle_polished'))} | {_fmt(pivots, '{:d}')} | "
            f"{_tick(r.get('jaddle_original_certified'))} | "
            f"{_fmt(r.get('rel_obj_gap'), '{:.2e}')} |"
        )


if __name__ == "__main__":
    main()

# %%
