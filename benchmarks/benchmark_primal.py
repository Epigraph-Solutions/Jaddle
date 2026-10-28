"""
Measure Jaddle's own primal heuristics on MIPLIB 2017 instances: how often
the root heuristics find a feasible point, how fast, and how good it is.

Only the root runs (``max_nodes=1``), with no help from HiGHS: no RENS, no
RINS and no HiGHS incumbent. Presolve is HiGHS's (``presolve=True``), as in
the cuOpt primal-heuristics paper (Çördük et al., 2025), and the heuristics
get the whole time limit. The instances are the smallest ``--count`` files
in ``data/`` (by size) with a known objective in MIPLIB's solution list,
which is downloaded to ``data/miplib2017.solu`` on first use.

The primal gap is the paper's: 1 without a point or when the signs differ,
else ``|z − z*| / max(|z|, |z*|)``. One row per instance goes to
``results/primal.csv`` (written after every instance), and a summary is
printed at the end.

Usage (from the repo root):

    python benchmarks/benchmark_primal.py                    # 30 instances, 60 s
    python benchmarks/benchmark_primal.py --count 40 --seconds 120
    python benchmarks/benchmark_primal.py --label fused-pump --out results/primal_fused.csv
    python benchmarks/benchmark_primal.py --label old --set probing=False \
        --set bulk_rounding=False --set pump_jump_iterations=0
"""

import argparse
import ast
import csv
import glob
import os
import time
import urllib.request

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jaddle.jaddle_optimisers as jo

jo.configure_jax("float64")

import jaddle.jaddle_mip as jm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(ROOT, "data")
SOLU = os.path.join(DATA_DIR, "miplib2017.solu")
SOLU_URL = "https://miplib.zib.de/downloads/miplib2017-v31.solu"
RESULTS = os.path.join(ROOT, "results", "primal.csv")

FIELDS = [
    "label",
    "name",
    "optimum",
    "found",
    "objective",
    "primal_gap",
    "first_seconds",
    "first_source",
    "sources",
    "seconds",
    "status",
    "error",
]


def known_objectives(path):
    """``{name: objective}`` from a MIPLIB ``.solu`` file (``=opt=`` and
    ``=best=`` lines; infeasible and unknown instances are left out)."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(SOLU_URL, path)
    values = {}
    with open(path) as handle:
        for line in handle:
            parts = line.split()
            if len(parts) == 3 and parts[0] in ("=opt=", "=best="):
                try:
                    values[parts[1]] = float(parts[2])
                except ValueError:
                    pass
    return values


def primal_gap(value, optimum):
    if value is None or value * optimum < 0:
        return 1.0
    scale = max(abs(value), abs(optimum))
    return 0.0 if scale == 0 else abs(value - optimum) / scale


def instances(data_dir, objectives, count):
    files = sorted(glob.glob(os.path.join(data_dir, "*.mps")), key=os.path.getsize)
    chosen = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        if name in objectives:
            chosen.append((name, path))
        if len(chosen) == count:
            break
    return chosen


def run_instance(name, path, optimum, args):
    row = {"label": args.label, "name": name, "optimum": optimum}
    started = time.time()
    try:
        result = jm.solve(
            path,
            presolve=True,
            max_nodes=1,
            max_seconds=args.seconds,
            heuristic_seconds=args.seconds,
            cuts=False,
            rens_seconds=0.0,
            rins_seconds=0.0,
            highs_incumbent_seconds=0.0,
            precision=args.precision,
            device=None if args.device == "default" else args.device,
            verbose=args.verbose,
            **args.settings,
        )
    except Exception as error:  # an instance that fails is a row, not a stop
        row.update(found=False, primal_gap=1.0, error=repr(error)[:200])
        row["seconds"] = round(time.time() - started, 2)
        return row
    history = result["history"]
    value = result["objective"]
    row.update(
        found=value is not None,
        objective=value,
        primal_gap=round(primal_gap(value, optimum), 6),
        first_seconds=round(history[0]["seconds"], 2) if history else None,
        first_source=history[0]["source"] if history else None,
        sources="".join(entry["source"] for entry in history),
        seconds=round(result["seconds"], 2),
        status=result["status"],
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("instances", nargs="*", help="names to run (default: the smallest --count)")
    parser.add_argument("--count", type=int, default=30, help="instances, smallest first")
    parser.add_argument("--seconds", type=float, default=60.0, help="time limit each")
    parser.add_argument("--precision", default="float32", help="jm.solve precision")
    parser.add_argument("--device", default="default", help="jm.solve device ('default': auto)")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--solu", default=SOLU, help="MIPLIB .solu file")
    parser.add_argument("--label", default="baseline", help="a name for this configuration")
    parser.add_argument("--out", default=RESULTS, help="CSV path")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra jm.solve setting (a Python literal), e.g. --set probing=False",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.settings = {}
    for item in args.set:
        key, _, value = item.partition("=")
        args.settings[key.strip()] = ast.literal_eval(value.strip())

    objectives = known_objectives(args.solu)
    if args.instances:
        chosen = [
            (name, os.path.join(args.data_dir, f"{name}.mps")) for name in args.instances
        ]
    else:
        chosen = instances(args.data_dir, objectives, args.count)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for name, path in chosen:
            row = run_instance(name, path, objectives.get(name), args)
            rows.append(row)
            writer.writerow(row)
            handle.flush()
            print(
                f"{name:32s} found={str(row['found']):5s} gap={row['primal_gap']:<8} "
                f"first={row.get('first_seconds')}s via {row.get('first_source')} "
                f"({row.get('sources') or ''}) {row.get('error') or ''}",
                flush=True,
            )

    found = [row for row in rows if row["found"]]
    gaps = [row["primal_gap"] for row in rows]
    firsts = sorted(row["first_seconds"] for row in found if row["first_seconds"] is not None)
    print("----------------------------------------------")
    print(f"{args.label}: {len(found)}/{len(rows)} feasible, mean primal gap {sum(gaps) / max(len(gaps), 1):.3f}")
    if firsts:
        print(f"median time to first incumbent: {firsts[len(firsts) // 2]:.1f}s")
    by_source = {}
    for row in found:
        by_source[row["first_source"]] = by_source.get(row["first_source"], 0) + 1
    print("first incumbent by source:", by_source)


if __name__ == "__main__":
    main()
