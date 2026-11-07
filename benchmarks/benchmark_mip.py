# %% [markdown]
# # Branch and Bound on a MIPLIB Instance
# This example loads a MIPLIB mixed-integer program with its integrality,
# presolves it with HiGHS, and runs Jaddle's branch and bound on the reduced
# problem with a time limit. The incumbent is mapped back to the original
# variables with HiGHS's postsolve.
#
# Pass the model file on the command line: any MPS file (or another format
# HiGHS reads, such as `.lp`), e.g.
#
#     python benchmarks/benchmark_mip.py path/to/model.mps [--device cpu|gpu|tpu|auto]
#
# `--device` picks where Jaddle's search runs (default `cpu`; `auto` puts
# MIPs with at most 10,000 nonzeros on the CPU and the rest on JAX's default
# device). A bare name with no such file, such as `academictimetablesmall`, is looked
# up as `data/<name>.mps` at the repo root. The MIPLIB files are not shipped
# with Jaddle; download them from the [MIPLIB website](https://miplib.zib.de/).
#
# Jaddle's branch and bound is deliberately simple, so on most MIPLIB
# instances it will not close the gap. Expect a valid bound and, usually, an
# incumbent from Jaddle's own primal heuristics.

# %%
import argparse
import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import highspy
import jaddle.jaddle_mip as jm

# %% [markdown]
# ## The instance

# %%
parser = argparse.ArgumentParser(
    description="Jaddle branch and bound on a MIPLIB instance"
)
parser.add_argument("model", help="model file (.mps, .lp, ...) or MIPLIB name in data/")
parser.add_argument(
    "--device",
    choices=("cpu", "gpu", "tpu", "auto"),
    default="cpu",
    help="device for Jaddle's search (default: cpu)",
)
args = parser.parse_args()
DEVICE = args.device
PATH_TO_MPS = args.model
if not os.path.exists(PATH_TO_MPS):
    PATH_TO_MPS = os.path.join(
        os.path.dirname(__file__), "..", "data", f"{PATH_TO_MPS}.mps"
    )
    if not os.path.exists(PATH_TO_MPS):
        sys.exit(f"no such file: {args.model} (nor {os.path.normpath(PATH_TO_MPS)})")
PROBLEM_NAME = os.path.basename(PATH_TO_MPS).split(".")[0]
MAX_SECONDS = 36000

mip = jm.load(PATH_TO_MPS)
print(
    f"{PROBLEM_NAME}: {mip.num_variables} variables ({mip.num_integers} integer), "
    f"{mip.A.shape[0]} rows"
)

# %% [markdown]
# ## Branch and bound
# `jm.solve`'s defaults are the setup tuned on map06:
# - float32 PDHG LPs, with node LPs 32 a batch on the device, each cut
#   short after 4 epochs, and short plunges (4 steps), so batches stay full;
# - no root cuts, probing or symmetry detection;
# - the feasibility jump (at the root, then on 4 background threads), with
#   repair of its near misses;
# - RINS and crossover as HiGHS sub-MIPs on a CPU thread: the integers on
#   which the incumbent and a node point (RINS) or several solutions
#   (crossover) agree are fixed, and HiGHS searches the rest;
# - 8 CPU threads that solve the nodes holding the global bound exactly
#   with HiGHS's dual simplex and strong-branch them, while the device
#   batches the rest.
#
# The log marks incumbents from the jump `J`, repair `R`, RINS `N` and
# crossover `C`. The background threads make runs unrepeatable.

# %%
result = jm.solve(
    PATH_TO_MPS,
    verbose=True,
    log_every_seconds=10,
    device=DEVICE,
    batch_size=32,
    presolve=True,
    # HiGHS sub-MIPs on a CPU thread (the defaults, spelt out; 0 turns one
    # off): RINS around the incumbent and the latest node point, and
    # crossover of the pool's solutions, up to 10 s a call.
    rins_seconds=10,
    crossover_seconds=10,
    jump_threads=1,
    exact_nodes=16,
    exact_strong_candidates=8,
    exact_band=0.25,
    exact_overflow_share=0.5,
    exact_strong_iterations=5000,
)
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
    f"calls, crossover {result['crossover_finds']} finds in "
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
# HiGHS's branch and cut with the same time limit.

# %%
highs = highspy.Highs()
highs.setOptionValue("output_flag", True)
highs.setOptionValue("parallel", "on")
highs.setOptionValue("threads", 12)
highs.setOptionValue("time_limit", float(MAX_SECONDS))
highs.readModel(PATH_TO_MPS)
highs.run()
info = highs.getInfo()
print(
    f"HiGHS: {highs.getModelStatus()}, incumbent {info.objective_function_value:.6g}, "
    f"bound {info.mip_dual_bound:.6g}"
)

# %%
