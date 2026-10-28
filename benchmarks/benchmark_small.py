"""
Benchmark Jaddle's branch and bound against HiGHS on small MIPLIB 3.0
instances.

MIPLIB 3.0 (1996) has instances of a few hundred to a few thousand
variables, most of which HiGHS solves in seconds, so a run over the whole
list takes minutes rather than the hours MIPLIB 2017 needs. Missing
instances are downloaded into ``data/small/`` from the MIPLIB 3.0 page:

    https://miplib2010.zib.de/miplib3/miplib3.html

Each instance is solved by ``jm.solve`` and by HiGHS with the same time
limit. One row per instance goes to ``results/small_mips.csv`` (written
after every instance, so an interrupted run keeps its results), and a
summary table is printed at the end.

Usage (from the repo root):

    python benchmarks/benchmark_small.py                  # all, 120 s each
    python benchmarks/benchmark_small.py --seconds 10 p0033 lseu bell5
    python benchmarks/benchmark_small.py --presolve --batch-size 16
"""

import argparse
import csv
import os
import time
import urllib.request

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax

jax.config.update("jax_platform_name", "cpu")

import jaddle.jaddle_optimisers as jo

jo.configure_jax("float64")

import highspy

import jaddle.jaddle_mip as jm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(ROOT, "data", "small")
RESULTS = os.path.join(ROOT, "results", "small_mips.csv")
URL = "https://miplib2010.zib.de/miplib3/miplib3/{name}.mps.gz"

# Small MIPLIB 3.0 instances with known optima, roughly smallest first.
INSTANCES = [
    "p0033",
    "flugpl",
    "egout",
    "enigma",
    "stein27",
    "lseu",
    "mod008",
    "pk1",
    "gt2",
    "bell5",
    "bell3a",
    "p0201",
    "p0282",
    "vpm1",
    "vpm2",
    "rgn",
    "pp08a",
    "misc03",
    "stein45",
    "misc07",
    "khb05250",
    "fixnet6",
    "blend2",
    "dcmulti",
    "p0548",
    "gesa2",
    "qnet1",
    "p2756",
]

FIELDS = [
    "name",
    "variables",
    "integers",
    "rows",
    "status",
    "objective",
    "bound",
    "gap",
    "nodes",
    "seconds",
    "reduced_cost_tightenings",
    "children_pruned",
    "objective_step",
    "lp_repairs",
    "lp_repairs_skipped",
    "tree_cuts_added",
    "highs_status",
    "highs_objective",
    "highs_seconds",
    "matches_highs",
    "error",
]


def instance_path(name):
    """The instance's file, downloaded first if missing (``None`` when the
    download fails)."""
    for extension in (".mps", ".mps.gz"):
        path = os.path.join(DATA_DIR, name + extension)
        if os.path.exists(path):
            return path
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name + ".mps.gz")
    url = URL.format(name=name)
    try:
        urllib.request.urlretrieve(url, path)
    except OSError as error:
        print(f"{name}: download failed ({error}); get it from {url}")
        if os.path.exists(path):
            os.remove(path)
        return None
    return path


def run_highs(path, seconds):
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", True)
    highs.setOptionValue("time_limit", float(seconds))
    highs.readModel(path)
    started = time.time()
    highs.run()
    elapsed = time.time() - started
    status = highs.modelStatusToString(highs.getModelStatus())
    info = highs.getInfo()
    objective = info.objective_function_value if info.primal_solution_status else None
    return status, objective, elapsed


def run_instance(name, path, args):
    row = {"name": name}
    try:
        mip = jm.load(path)
        row.update(
            variables=mip.num_variables,
            integers=mip.num_integers,
            rows=mip.A.shape[0],
        )
        result = jm.solve(
            path if args.presolve else mip,
            presolve=args.presolve,
            max_seconds=args.seconds,
            batch_size=args.batch_size,
            tree_cuts=args.tree_cuts,
            max_cuts_per_round=1000,
            feasibility_jump=False,
            feasibility_pump=False,
            fix_and_propagate=False,
            lp_tolerance=1e-4,
            node_lp_tolerance=1e-4,
            device=None if args.device == "default" else args.device,
            verbose=args.verbose,
            gomory_cuts=True,
        )
        row.update(
            {
                key: result[key]
                for key in (
                    "status",
                    "objective",
                    "bound",
                    "gap",
                    "nodes",
                    "seconds",
                    "reduced_cost_tightenings",
                    "children_pruned",
                    "objective_step",
                    "lp_repairs",
                    "lp_repairs_skipped",
                    "tree_cuts_added",
                )
            }
        )
    except Exception as error:  # keep going: one bad instance shouldn't stop the run
        row["error"] = f"{type(error).__name__}: {error}"

    status, objective, elapsed = run_highs(path, args.seconds)
    row.update(highs_status=status, highs_objective=objective, highs_seconds=elapsed)
    if row.get("objective") is not None and objective is not None:
        tolerance = 1e-4 * max(1.0, abs(objective))
        row["matches_highs"] = abs(row["objective"] - objective) <= tolerance
    return row


def fmt(value, spec):
    return "-" if value is None or value == "" else format(value, spec)


def print_summary(rows):
    print()
    print(
        f"{'name':<10} {'vars':>6} {'ints':>6} {'status':<12} {'objective':>14} "
        f"{'gap':>8} {'nodes':>7} {'time':>7}   {'HiGHS obj':>14} {'time':>7} match"
    )
    for row in rows:
        gap = row.get("gap")
        print(
            f"{row['name']:<10} {fmt(row.get('variables'), 'd'):>6} "
            f"{fmt(row.get('integers'), 'd'):>6} "
            f"{(row.get('status') or 'error'):<12} "
            f"{fmt(row.get('objective'), '.6g'):>14} "
            f"{(fmt(100 * gap, '.2f') + '%') if gap is not None and gap < 1e9 else '-':>8} "
            f"{fmt(row.get('nodes'), 'd'):>7} {fmt(row.get('seconds'), '.1f'):>7}   "
            f"{fmt(row.get('highs_objective'), '.6g'):>14} "
            f"{fmt(row.get('highs_seconds'), '.1f'):>7} "
            f"{'' if row.get('matches_highs') is None else ('yes' if row['matches_highs'] else 'NO')}"
        )
    solved = sum(row.get("status") == "optimal" for row in rows)
    matched = sum(bool(row.get("matches_highs")) for row in rows)
    jaddle_time = sum(row.get("seconds") or 0.0 for row in rows)
    highs_time = sum(row.get("highs_seconds") or 0.0 for row in rows)
    print(
        f"\nJaddle optimal on {solved}/{len(rows)}, objective matches HiGHS on "
        f"{matched}/{len(rows)}. Total time: Jaddle {jaddle_time:.0f}s, "
        f"HiGHS {highs_time:.0f}s."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("instances", nargs="*", help="names to run (default: all)")
    parser.add_argument("--seconds", type=float, default=120.0, help="time limit each")
    parser.add_argument("--presolve", action="store_true", help="HiGHS presolve first")
    parser.add_argument("--batch-size", type=int, default=1, help="jm.solve batch_size")
    parser.add_argument(
        "--device",
        default="auto",
        help="jm.solve device: auto (CPU when small), cpu, gpu, or default (JAX's)",
    )
    parser.add_argument("--tree-cuts", action="store_true", help="branch and cut")
    parser.add_argument("--verbose", action="store_true", help="jm.solve progress log")
    parser.add_argument("--out", default=RESULTS, help="CSV path")
    args = parser.parse_args()

    names = args.instances or INSTANCES
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for name in names:
            path = instance_path(name)
            if path is None:
                continue
            print(f"{name} ...", flush=True)
            row = run_instance(name, path, args)
            rows.append(row)
            writer.writerow(row)
            handle.flush()
    print_summary(rows)
    print(f"Results: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
