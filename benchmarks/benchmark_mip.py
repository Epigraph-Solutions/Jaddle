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
# Every LP of the search is Jaddle's (PDHG): the root and its cut rounds in
# float64, node LPs `batch_size` at a time on the device (laptop GPUs run
# float64 at a small fraction of float32 speed). HiGHS only reads the model,
# presolves it (`presolve=True` works on HiGHS's reduced MIP and postsolves
# the incumbent) and crosses PDHG points over to vertices (Gomory cut
# rounds, `root_crossover`, `node_crossover`).
#
# Every primal heuristic is on by default:
# - root probing;
# - round and repair: LP points rounded up, to nearest and down, the
#   continuous rest solved by a PDHG LP (at the root, the root's strong
#   children and every few nodes);
# - bulk fix-and-propagate (several starts at once; the classic one is off);
# - the feasibility jump, then an improving jump from each incumbent;
# - the feasibility pump, with a short jump from each rounding it rejects;
# - RENS at the root (small MIPs) and RINS in the tree, as Jaddle
#   sub-searches;
# - in the tree: LP dives every 50 nodes, and GPU bulk rounding and jumps
#   from node points every 10.
#
# The log marks their incumbents `U` (round and repair), `P`
# (fix-and-propagate), `B` (bulk), `J` (jump), `F` (pump), `R` (repair), `E`
# (RENS), `N` (RINS) and `D` (dive). The numba improving jump runs on a
# background thread beside the node LPs (`parallel_heuristics`), so runs are
# not repeatable.
#
# Cuts: the root cut rounds keep a cut until it has been idle for
# `cut_age` rounds, and every cut joins a global pool. Tree cuts
# (`tree_cuts`) refresh the LP's cut block between node solves: pooled cuts
# the batch's points violate come back, new ones are separated at the
# lowest-bound points, and members a new cut cuts off are solved again
# before they branch. The block keeps its shape, so the batched node solver
# never recompiles.
#
# Also on here, though off by default: the root cut pool's limits
# (`max_cut_rows`, `gomory_candidates`) and three uses of the incumbent for
# the bound. `root_redcost_propagation` propagates the root's reduced-cost
# row at every node. `global_redcost_fixing` fixes the root box with every
# LP's duals, not only the root's. `inout_separation` also separates cuts at
# a point between the LP point and the incumbent.

# %%
result = jm.solve(
    PATH_TO_MPS,
    presolve=True,
    max_seconds=MAX_SECONDS,
    verbose=True,
    log_every_seconds=10,
    device=DEVICE,
    lp_tolerance=1e-6,
    root_crossover=False,
    cut_rounds=30,
    # 128 nodes a batch on a GPU, 32 on a CPU (ns1830653, GPU: 147 nodes a
    # second with 8, 394 with 128).
    batch_size=32,
    # Truncated node LPs (the defaults, spelt out): each node LP runs at most
    # 2 epochs and then branches on its point, with no requeue. On
    # ns1830653 this took the tree from 16 to 147 nodes a second.
    batch_epochs=2,
    node_max_epochs=None,
    node_lp_tolerance=1e-3,
    node_crossover=False,
    node_precision="float32",
    cuts=True,
    tree_cuts=True,
    cut_pool_propagation=True,
    plunge_depth=1000,
    plunge_child="rounding",
    max_cuts_per_round=50,
    max_tree_cuts_per_round=1e3,
    max_cut_rows=2000,
    tree_cut_rows=2000,
    tree_cut_every=10,
    cut_age=100,
    gomory_candidates=400,
    inout_separation=0.5,
    root_redcost_propagation=True,
    global_redcost_fixing=True,
    implied_integer_cuts=True,
    one_opt=True,
    objective_propagation=True,
    branch_implied=True,
    parallel_heuristics=True,
    # Off for throughput: the sub-search and dive heuristics run on the main
    # thread and stall the node loop (RINS, RENS and dives took over half
    # the tree's time on ns1830653).
    round_repair=True,
    rins_seconds=0,
    rens_seconds=0,
    dive_every=0,
)
print(f"Status: {result['status']}")
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
