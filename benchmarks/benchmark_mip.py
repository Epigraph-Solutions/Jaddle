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
# `jm.solve`'s defaults are tuned for this kind of run: large MIPs on a GPU.
# `presolve=True` works on HiGHS's reduced MIP and postsolves the incumbent.
# Every primal heuristic is on by default, Jaddle's and HiGHS's:
# - root probing;
# - round and repair: LP points rounded up, to nearest and down, with
#   HiGHS solving the continuous rest exactly (at the root, the root's
#   strong children and every few nodes);
# - bulk fix-and-propagate (several starts at once; the classic one is off);
# - the feasibility jump, then an improving jump from each incumbent;
# - the feasibility pump, with a short jump from each rounding it rejects;
# - HiGHS: RENS at the root, RINS in the tree (every 5 nodes, up to 30% of
#   the run), and its own MIP search for an incumbent if all of the above
#   fail;
# - in the tree: LP dives every 50 nodes, and GPU bulk rounding and jumps
#   from node points every 10.
#
# The log marks their incumbents `U` (round and repair), `P`
# (fix-and-propagate), `B` (bulk), `J` (jump), `F` (pump), `R` (repair), `E`
# (RENS), `N` (RINS), `X` (HiGHS) and `D` (dive). The CPU ones (HiGHS, round
# and repair in the tree, and the numba improving jump) run on background
# threads beside the node LPs (`parallel_heuristics`), so runs are not
# repeatable. With HiGHS searching the RINS sub-MIPs, ns1830653 went from
# 131,808 to 36,808 within 10 s of the first point in a 5-minute test (best
# known: 20,622).
#
# The other defaults: pseudocost branching (reliability branching spent over
# half the run strong-branching here; the root still strong-branches its
# highest-impact candidates once, which on buildingenergy finds the sizing
# variable holding the root gap); the root LP and cut rounds in float64;
# node LPs solved exactly by HiGHS's dual simplex, one node at a time
# (`node_lp_solver="highs"`; on bab5 PDHG node bounds were too loose for the
# global bound to move). With `node_lp_solver="jaddle"` node LPs run on the
# device in float32 (laptop GPUs run float64 at a small fraction of float32
# speed), `batch_size` at a time (on ns1830653, 32 halves the device time
# per node against 16; tree cuts are off when batching).
#
# Also on here, though off by default: the root cut pool's limits
# (`max_cut_rows`, `simplex_cut_age`, `gomory_candidates`), and three uses of
# the incumbent for the bound. `cutoff_probing` gives exact HiGHS LPs to the
# integers that root reduced-cost fixing only just fails to fix (it needs
# HiGHS node LPs). `root_redcost_propagation` propagates the root's
# reduced-cost row at every node. `global_redcost_fixing` fixes the root box
# with every LP's duals, not only the root's.

# %%
result = jm.solve(
    PATH_TO_MPS,
    presolve=True,
    max_seconds=MAX_SECONDS,
    verbose=True,
    log_every_seconds=10,
    device=DEVICE,
    lp_tolerance=1e-4,
    root_crossover=False,
    cut_rounds=30,
    batch_size=64,
    tree_cuts=False,
    cut_pool_propagation=True,
    plunge_depth=1000,
    plunge_child="rounding",
    max_cuts_per_round=50,
    max_tree_cuts_per_round=50,
    node_lp_solver="jaddle",
    rins_solver="highs",
    max_cut_rows=2000,
    tree_cut_rows=2000,
    simplex_cut_age=10,
    gomory_candidates=400,
    cutoff_probing=200,
    root_redcost_propagation=True,
    global_redcost_fixing=True,
    implied_integer_cuts=True,
)
print(f"Status: {result['status']}")
print(f"Bound: {result['bound']:.6g}")
print(
    f"Cutoff probing: {result['cutoff_probing_fixed']} fixed in "
    f"{result['cutoff_probing_lps']} LPs, {result['cutoff_probing_seconds']:.1f}s; "
    f"global reduced-cost fixing: {result['global_redcost_fixed']} bounds; "
    f"restarts: {result['restarts']}"
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
