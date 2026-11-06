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
    symmetry=False,
    presolve=True,
    max_seconds=MAX_SECONDS,
    verbose=True,
    log_every_seconds=10,
    device=DEVICE,
    lp_tolerance=1e-4,
    precision="float32",
    root_crossover=False,
    probing=False,
    cut_rounds=10,
    # 128 nodes a batch on a GPU, 32 on a CPU (ns1830653, GPU: 147 nodes a
    # second with 8, 394 with 128). A short batch is padded to batch_size, so
    # when plunging leaves one or two nodes a batch (map06: 19 batches for 19
    # nodes, ~10 s each at 32 on a CPU), a small batch wastes far less.
    batch_size=150,
    # Truncated node LPs (the defaults, spelt out): each node LP runs at most
    # 2 epochs and then branches on its point, with no requeue. On
    # ns1830653 this took the tree from 16 to 147 nodes a second.
    batch_epochs=4,
    node_max_epochs=None,
    node_lp_tolerance=1e-2,
    node_crossover=False,
    node_precision="float16",
    cuts=False,
    tree_cuts=True,
    cut_pool_propagation=True,
    # The default plunge depth: deep plunges leave the node holding the global
    # bound unvisited (map06 stayed at -402.6 while plunged nodes reached -384).
    plunge_depth=4,
    plunge_child="rounding",
    max_cuts_per_round=500,
    max_tree_cuts_per_round=200,
    max_cut_rows=1e4,
    tree_cut_rows=1000,
    tree_cut_every=10,
    cut_age=100,
    gomory_candidates=400,
    # c-MIR aggregates of up to 7 rows. With probing's variable bounds (on by
    # default) this took map06's root bound (exact LPs) from −395.8 to −393.9.
    max_aggregation=6,
    inout_separation=0.5,
    root_redcost_propagation=True,
    global_redcost_fixing=True,
    implied_integer_cuts=True,
    objective_propagation=True,
    branch_implied=True,
    # Longer, wider root strong branching: with the default 50 epochs no
    # map06 child got past the parent's bound, so the pseudocosts learned
    # nothing (an exact LP gives +14.6 for the best variable).
    root_strong_candidates=4,
    root_strong_epochs=5,
    # Batched search (the defaults, spelt out): the host evaluates one batch
    # while the device solves the next.
    pipeline_batches=True,
    # The only primal heuristic: the feasibility jump. At the root (from the
    # LP point and from the bounds nearest zero), then on background threads:
    # 4 lanes, never idle, one slice from each new incumbent and restarts
    # from a perturbed incumbent or a node's LP point.
    parallel_heuristics=True,
    feasibility_jump=True,
    jump_threads=4,
    jump_restarts=True,
    # Every other primal heuristic off.
    fix_and_propagate=False,  # root rounding
    bulk_rounding=False,  # root and tree bulk rounding
    feasibility_pump=False,  # root pump
    repair=True,  # sub-MIP repair of a jump's near miss
    round_repair=False,  # rounding plus a PDHG repair LP
    heuristic_every=0,  # rounding the node LP point every few nodes ("H")
    near_integral_tolerance=0,  # rounding near-integral node points ("T")
    parallel_rounding=False,  # node points: fix-and-propagate, then a 1 s jump
    tree_heuristics=False,  # GPU bulk rounding and jump from node points
    one_opt=False,  # 1-opt on each new incumbent
    rins_seconds=0,  # RINS sub-searches
    rens_seconds=0,  # RENS at the root
    dive_every=0,  # serial LP dives
    gpu_dive_every=0,  # batched GPU dives
    exact_plunges=False,  # converged plunges with crossover
    # Exact nodes: CPU threads solve the nodes that hold the global bound
    # with HiGHS's dual simplex while the GPU batches the rest. Nodes within
    # exact_band of the gap above the global bound are kept for the threads
    # (0: the threads take the heap's top when they have room, the GPU the
    # rest). On map06 a node takes HiGHS 1.1-1.6 s, and the exact vertices
    # sped up the incumbent (-254 at 109 s, against -201 without).
    exact_nodes=8,
    exact_node_seconds=10,
    exact_band=0.005,
)
print(f"Status: {result['status']}")
exact = result["exact_nodes"]
print(
    f"Exact nodes: {exact['solved']} solved, {exact['infeasible']} infeasible, "
    f"{exact['failed']} unfinished, {exact['seconds']:.1f}s in the threads, "
    f"bound gained {exact['bound_gain']:.3g} in all"
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
