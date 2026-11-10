# %% [markdown]
# # Branch and Bound on MIPLIB Instances
# This harness loads each MIPLIB mixed-integer program in a directory with
# its integrality, presolves it with HiGHS, and runs Jaddle's branch and bound
# on the reduced problem with an optional time limit. The incumbent is mapped
# back to the original variables with HiGHS's postsolve. HiGHS's own branch and
# cut runs on the same instance for comparison, and a CSV collects the results.
#
# Usage:
#
#     python benchmarks/benchmark_mip.py                          # every .mps in data/pdlp_set/
#     python benchmarks/benchmark_mip.py --data-dir data/popso    # another directory
#     python benchmarks/benchmark_mip.py --only air05 mas76       # subset by name
#     python benchmarks/benchmark_mip.py --max-seconds 120 --no-highs
#     python benchmarks/benchmark_mip.py --device cpu|gpu|tpu|auto
#
# `--device` picks where Jaddle's search runs (default `cpu`; `auto` puts
# MIPs with at most 10,000 nonzeros on the CPU and the rest on JAX's default
# device). `--only` also accepts a path to a model file (.mps, .lp, or another
# format HiGHS reads), which then runs whatever directory it is in. The MIPLIB
# files are not shipped with Jaddle; download them from the
# [MIPLIB website](https://miplib.zib.de/).
#
# Jaddle's branch and bound is deliberately simple, so on most MIPLIB
# instances it will not close the gap. Expect a valid bound and, usually, an
# incumbent from Jaddle's own primal heuristics.

# %%
import argparse
import csv
import gc
import glob
import os
import time

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import highspy
import jax
import jaddle.jaddle_mip as jm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "pdlp_set")
HIGHS_DEFAULT_SECONDS = 36000

# %% [markdown]
# ## Options and instances


# %%
def parse_args():
    p = argparse.ArgumentParser(
        description="Jaddle branch and bound vs HiGHS on MIPLIB instances"
    )
    p.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help="directory of .mps files to glob (default: data/pdlp_set/)",
    )
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="problem names (without .mps), or model file paths, to restrict to "
        "(default: all in --data-dir)",
    )
    p.add_argument(
        "--device",
        choices=("cpu", "gpu", "tpu", "auto"),
        default="cpu",
        help="device for Jaddle's search (default: cpu)",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="per-instance time limit for Jaddle's solve, also used for HiGHS "
        f"(default: no limit for Jaddle, {HIGHS_DEFAULT_SECONDS} s for HiGHS)",
    )
    p.add_argument(
        "--no-highs", action="store_true", help="skip the HiGHS comparison solve"
    )
    p.add_argument(
        "--csv",
        default=os.path.join(REPO_ROOT, "benchmark_mip_results.csv"),
        help="path to write CSV results (default: benchmark_mip_results.csv)",
    )
    return p.parse_args()


def discover_instances(args):
    """(name, path) pairs: every .mps in --data-dir, or the --only subset.
    An --only entry that is an existing file is taken as is."""
    files = [o for o in (args.only or []) if os.path.isfile(o)]
    names = {o for o in (args.only or []) if not os.path.isfile(o)}
    instances = [(os.path.basename(f).split(".")[0], f) for f in files]
    missing = set(names)
    if not args.only or names:
        for path in sorted(glob.glob(os.path.join(args.data_dir, "*.mps"))):
            name = os.path.splitext(os.path.basename(path))[0]
            if names and name not in names:
                continue
            missing.discard(name)
            instances.append((name, path))
    for name in sorted(missing):
        print(f"  not found in {args.data_dir}: {name}")
    return instances


# %% [markdown]
# ## Branch and bound
# `jm.solve`'s defaults are the setup tuned on map06:
# - float32 PDHG LPs, with node LPs batched on the device (`batch_size="auto"`:
#   as many as fit in a GPU's free memory, up to 128; 32 on a CPU), each cut
#   short after 4 epochs, and short plunges (4 steps), so batches stay full;
# - no root cuts, probing or symmetry detection;
# - the feasibility jump (at the root, then on a background thread), with
#   repair of its near misses; integers an equality row defines (such as
#   start times `S_j = Σ_t t·x_jt`) are substituted out of its rows;
# - RINS and crossover as HiGHS sub-MIPs on a CPU thread: the integers on
#   which the incumbent and a node point (RINS) or several solutions
#   (crossover) agree are fixed, and HiGHS searches the rest; when RINS's
#   boxes come back empty at once, it fixes only a share of them (its
#   fixing rate, down to 40%);
# - 16 CPU threads that solve the nodes within a quarter of the gap of the
#   global bound exactly with HiGHS's dual simplex and strong-branch them
#   (8 candidates, each side for up to twice the mean pivots of a node LP,
#   at most 500), while the device batches the rest.
#
# The log marks incumbents from the jump `J`, repair `R`, RINS `N` and
# crossover `C`. The background threads make runs unrepeatable.


# %%
def run_jaddle(path, args):
    mip = jm.load(path)
    row = {
        "n_vars": mip.num_variables,
        "n_ints": mip.num_integers,
        "n_rows": mip.A.shape[0],
    }
    print(f"{row['n_vars']} variables ({row['n_ints']} integer), {row['n_rows']} rows")
    started = time.time()
    result = jm.solve(
        path,
        verbose=True,
        log_every_seconds=10,
        device=args.device,
        presolve=True,
        exact_nodes=8,
        exact_band="auto",
        exact_overflow_share=0.0,
        exact_strong_lookahead=8,
        batch_size=16,
        # cuts=True,
        max_seconds=args.max_seconds,
        tree_cut_every=100,
        cut_pool_size=1e5,
        tree_cut_rows=1e4,
        cut_rounds=100,
        root_lp="jaddle",
        cut_round_lp="highs",
        probing=True,
        restarts=0,
        gomory_cuts=False,
        cuts=True,
        jump_threads=4,
        proximity_seconds=10,
        plunge_depth=2,
    )
    row.update(
        {
            "jaddle_status": result["status"],
            "jaddle_objective": result["objective"],
            "jaddle_bound": result["bound"],
            "jaddle_gap": result["gap"],
            "jaddle_nodes": result.get("nodes"),
            # As in benchmark.py: the solve's own time, HiGHS's presolve
            # apart, and the whole call (file read and postsolve included).
            "jaddle_seconds": result["seconds"],
            "jaddle_presolve_seconds": result.get("presolve_seconds", 0.0),
            "jaddle_wall_seconds": time.time() - started,
            "jaddle_root_lp": result.get("root_lp", {}).get("solver"),
            "jaddle_root_lp_seconds": result.get("root_lp", {}).get("seconds"),
            "jaddle_postsolve_feasible": result.get("postsolve_feasible"),
        }
    )
    try:
        print_details(result)
    except (KeyError, TypeError) as exc:  # an early exit can lack some stats
        print(f"  (details unavailable: {exc!r})")
    return row


def print_details(result):
    print(f"Status: {result['status']}")
    exact = result["exact_nodes"]
    print(
        f"Exact nodes: {exact['solved']} solved, {exact['infeasible']} infeasible, "
        f"{exact['failed']} unfinished, {exact['seconds']:.1f}s in the threads, "
        f"bound gained {exact['bound_gain']:.3g} in all; strong branching "
        f"{exact['strong_lps']} LPs in {exact['strong_seconds']:.1f}s, "
        f"{exact['children_reused']} children reused; "
        f"{exact['pivots'] / max(1, exact['solved']):.0f} pivots a node, "
        f"{exact['warm_bases']} from the parent's basis"
    )
    root = result["root_exact_strong"]
    print(
        f"Root exact strong branching: {root['candidates']} candidates, "
        f"{root['lps']} LPs ({root['fixed']} sides empty) in {root['seconds']:.1f}s, "
        f"exact root bound {root['bound']}"
    )
    print(
        f"Sub-MIPs (HiGHS): RINS {result['rins_finds']} finds in {result['rins_calls']} "
        f"calls (fixing rate {result['rins_fixing_rate']:.2f}), crossover "
        f"{result['crossover_finds']} finds in "
        f"{result['crossover_calls']} calls, {result['submip_lane_seconds']:.1f}s"
    )
    print(f"Bound: {result['bound']:.6g}")
    print(
        f"Global reduced-cost fixing: {result['global_redcost_fixed']} bounds; "
        f"restarts: {result['restarts']}; in-out separation: "
        f"{result['inout_cuts']} cuts kept in {result['inout_rounds']} rounds"
    )
    print(
        f"Tree cuts: {result['tree_cuts_added']} added in "
        f"{result['tree_cut_refreshes']} refreshes ({result['tree_cut_pool_hits']} "
        f"from the pool, {result['tree_cuts_dropped']} dropped for room), "
        f"{result['tree_cut_requeued']} nodes re-solved for them, "
        f"{result['tree_cut_seconds']:.1f}s; families {result['tree_cut_families']}; "
        f"pool {result['cut_pool']}"
    )
    if result["x"] is not None:
        print(f"Incumbent: {result['objective']:.6g} (gap {100 * result['gap']:.2f}%)")
        print(f"Feasible after postsolve: {result['postsolve_feasible']}")


# %% [markdown]
# ## HiGHS for comparison
# HiGHS's branch and cut with the same time limit (36000 s without
# `--max-seconds`); `--no-highs` skips it. As in benchmark.py, HiGHS's time
# is its `run()`, its presolve included, while `jaddle_seconds` leaves
# Jaddle's (HiGHS) presolve out: it is `jaddle_presolve_seconds`, and
# `jaddle_wall_seconds` is the whole call.


# %%
def run_highs(path, args):
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", True)
    highs.setOptionValue("parallel", "on")
    highs.setOptionValue("threads", 12)
    limit = args.max_seconds if args.max_seconds is not None else HIGHS_DEFAULT_SECONDS
    highs.setOptionValue("time_limit", float(limit))
    highs.readModel(path)
    started = time.time()
    highs.run()
    seconds = time.time() - started
    info = highs.getInfo()
    status = highs.modelStatusToString(highs.getModelStatus())
    print(
        f"HiGHS: {status}, incumbent {info.objective_function_value:.6g}, "
        f"bound {info.mip_dual_bound:.6g}"
    )
    return {
        "highs_status": status,
        "highs_objective": info.objective_function_value,
        "highs_bound": info.mip_dual_bound,
        "highs_gap": info.mip_gap,
        "highs_nodes": info.mip_node_count,
        "highs_seconds": seconds,
    }


# %% [markdown]
# ## The sweep

# %%
CSV_FIELDS = [
    "problem",
    "size_mb",
    "n_vars",
    "n_ints",
    "n_rows",
    "jaddle_status",
    "jaddle_objective",
    "jaddle_bound",
    "jaddle_gap",
    "jaddle_nodes",
    "jaddle_seconds",
    "jaddle_presolve_seconds",
    "jaddle_wall_seconds",
    "jaddle_root_lp",
    "jaddle_root_lp_seconds",
    "jaddle_postsolve_feasible",
    "highs_status",
    "highs_objective",
    "highs_bound",
    "highs_gap",
    "highs_nodes",
    "highs_seconds",
    "error",
]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def main():
    args = parse_args()
    instances = discover_instances(args)
    if not instances:
        print(
            f"No .mps files to run in {args.data_dir}. Download MIPLIB instances "
            "(https://miplib.zib.de/) and retry."
        )
        return
    limit = "no limit" if args.max_seconds is None else f"{args.max_seconds:g} s"
    print(
        f"Running {len(instances)} instance(s) on {args.device}, Jaddle {limit}"
        f"{', no HiGHS' if args.no_highs else ''}\n"
    )
    rows = []
    for name, path in instances:
        size_mb = os.path.getsize(path) / 1e6
        print(f"=== {name} ({size_mb:.1f} MB) ===")
        row = {"problem": name, "size_mb": round(size_mb, 1), "error": ""}
        try:
            row.update(run_jaddle(path, args))
            if not args.no_highs:
                row.update(run_highs(path, args))
        except Exception as exc:  # keep the sweep going if one instance blows up
            row["error"] = repr(exc)
            print(f"  ERROR: {exc!r}")
        rows.append(row)
        # Rewrite the CSV after every instance so a crash mid-sweep keeps results.
        write_csv(args.csv, rows)
        # Each instance has new shapes, so JAX's caches never carry over.
        jax.clear_caches()
        gc.collect()
        print()
    print(f"CSV written to {args.csv}")


if __name__ == "__main__":
    main()

# %%
