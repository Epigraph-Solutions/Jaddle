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
#     python examples/mip/miplib.py path/to/model.mps
#
# A bare name with no such file, such as `academictimetablesmall`, is looked
# up as `data/<name>.mps` at the repo root. The MIPLIB files are not shipped
# with Jaddle; download them from the [MIPLIB website](https://miplib.zib.de/).
#
# Jaddle's branch and bound is deliberately simple, so on most MIPLIB
# instances it will not close the gap. Expect a valid bound and, usually, an
# incumbent from Jaddle's own primal heuristics.

# %%
import os
import sys

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import highspy
import jaddle.jaddle_mip as jm

# %% [markdown]
# ## The instance

# %%
if len(sys.argv) < 2:
    sys.exit(f"usage: python {sys.argv[0]} <model.mps | MIPLIB name in data/>")
PATH_TO_MPS = sys.argv[1]
if not os.path.exists(PATH_TO_MPS):
    PATH_TO_MPS = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", f"{PATH_TO_MPS}.mps"
    )
    if not os.path.exists(PATH_TO_MPS):
        sys.exit(f"no such file: {sys.argv[1]} (nor {os.path.normpath(PATH_TO_MPS)})")
PROBLEM_NAME = os.path.basename(PATH_TO_MPS).split(".")[0]
MAX_SECONDS = 36000

mip = jm.load(PATH_TO_MPS)
print(
    f"{PROBLEM_NAME}: {mip.num_variables} variables ({mip.num_integers} integer), "
    f"{mip.A.shape[0]} rows"
)

# %% [markdown]
# ## Branch and bound
# `presolve=True` works on HiGHS's reduced MIP and postsolves the incumbent.
# Every primal heuristic is on, Jaddle's and HiGHS's:
# - root probing;
# - fix-and-propagate, classic and bulk (several starts at once);
# - the feasibility jump, then an improving jump from each incumbent;
# - the feasibility pump, with a short jump from each rounding it rejects;
# - HiGHS: RENS at the root, RINS in the tree, and its own MIP search for
#   an incumbent if all of the above fail;
# - in the tree: LP dives, and GPU bulk rounding and jumps from node points.
#
# The log marks their incumbents `P` (fix-and-propagate), `B` (bulk), `J`
# (jump), `F` (pump), `R` (repair), `E` (RENS), `N` (RINS), `X` (HiGHS) and
# `D` (dive). With `parallel_heuristics=True` the CPU ones (HiGHS and the
# numba improving jump) run on background threads beside the node LPs. With
# HiGHS searching the RINS sub-MIPs, ns1830653 went from 131,808 to 36,808
# within 10 s of the first point in a 5-minute test (best known: 20,622).

# %%
result = jm.solve(
    PATH_TO_MPS,
    # Reliability branching spent over half the run strong-branching here.
    branching="pseudocost",
    # node_crossover=True,
    presolve=True,
    max_seconds=MAX_SECONDS,
    gomory_cuts=True,
    # Laptop GPUs run float64 at a small fraction of float32 speed.
    # The root LP and cut rounds run in float64; node LPs at node_precision.
    precision="float32",
    node_precision="float32",
    node_crossover=True,
    cut_round_epochs=10,
    lp_tolerance=1e-4,
    node_lp_tolerance=1e-3,
    # Any other jl.solve option for node LPs only (root and heuristic LPs
    # keep lp_options), e.g. check convergence every 64 iterations:
    node_lp_options={"iterations_per_epoch": 64},
    max_cuts_per_round=1000,
    heuristic_every=5,
    # Primal heuristics, after cuOpt's GPU heuristics. On ns1830653 the
    # pump finds the first point at about 70 s, so 20 s was too short.
    heuristic_seconds=10,
    probing=True,  # root probing, batched in JAX on large models
    fix_and_propagate=False,  # classic fix-and-propagate, then...
    bulk_rounding=True,  # ...bulk fix-and-propagate from 4 starts at once
    # Feasibility jump: incremental on the host with numba (jaddle[fast]),
    # which then also spends leftover heuristic time improving the incumbent.
    feasibility_jump=True,
    feasibility_pump=True,  # the pump, fused with short jumps
    pump_jump_iterations=3000,
    repair_share=0.2,  # cap repair so later jumps and the pump still run
    # HiGHS searches the RENS and RINS sub-MIPs ("jaddle": Jaddle's own branch
    # and bound, slower on large MIPs). RENS gets 10% of max_seconds, up to
    # 30 s. HiGHS's own MIP search runs for up to 10 s only if no other
    # heuristic finds an incumbent.
    rins_solver="highs",
    highs_incumbent_seconds=10,
    rens_seconds=None,
    # RINS improves the incumbent in the tree: run it often, and let it use up
    # to 30% of the run.
    rins_every=5,
    heuristic_effort=0.3,
    # HiGHS's RENS/RINS and the numba improving jump on background threads,
    # beside the node LPs (runs are then not repeatable).
    parallel_heuristics=True,
    # LP dives every 50 nodes (and at the root): round, propagate, re-solve.
    dive_every=50,
    # GPU heuristics from node LP points once there is an incumbent: bulk
    # rounding, then a feasibility jump that looks only for better points.
    tree_heuristics=True,
    tree_heuristic_every=10,
    tree_heuristic_share=0.05,
    # On a GPU: 32 node LPs per device call, at most 10 epochs each a call
    # (unfinished ones go back to the queue). On ns1830653 that halves the
    # device time per node against 16. Tree cuts are off when batching.
    batch_size=32,
    batch_epochs=10,
    cuts=False,
    tree_cuts=False,
    verbose=True,
    log_every_seconds=10,
    device="gpu",
)
print(f"Status: {result['status']}")
print(f"Bound: {result['bound']:.6g}")
if result["x"] is not None:
    print(f"Incumbent: {result['objective']:.6g} (gap {100 * result['gap']:.2f}%)")
    print(f"Feasible after postsolve: {result['postsolve_feasible']}")

# %% [markdown]
# ## HiGHS for comparison
# HiGHS's branch and cut with the same time limit.

# %%
highs = highspy.Highs()
highs.setOptionValue("output_flag", True)
highs.setOptionValue("time_limit", float(MAX_SECONDS))
highs.readModel(PATH_TO_MPS)
highs.run()
info = highs.getInfo()
print(
    f"HiGHS: {highs.getModelStatus()}, incumbent {info.objective_function_value:.6g}, "
    f"bound {info.mip_dual_bound:.6g}"
)

# %%
