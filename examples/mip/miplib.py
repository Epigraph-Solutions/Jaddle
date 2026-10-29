# %% [markdown]
# # Branch and Bound on a MIPLIB Instance
# This example loads a MIPLIB mixed-integer program with its integrality,
# presolves it with HiGHS, and runs Jaddle's branch and bound on the reduced
# problem with a time limit. The incumbent is mapped back to the original
# variables with HiGHS's postsolve.
#
# The MPS files are not shipped with Jaddle. Download them from the
# [MIPLIB website](https://miplib.zib.de/) into `data/` at the repo root.
#
# Jaddle's branch and bound is deliberately simple, so on most MIPLIB
# instances it will not close the gap. Expect a valid bound and, usually, an
# incumbent from Jaddle's own primal heuristics.

# %%
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import highspy
import jaddle.jaddle_mip as jm

# %% [markdown]
# ## The instance

# %%
PROBLEM_NAME = "rmine11"  # name of the MIPLIB problem (without .mps)
PATH_TO_MPS = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", f"{PROBLEM_NAME}.mps"
)
MAX_SECONDS = 3600

mip = jm.load(PATH_TO_MPS)
print(
    f"{PROBLEM_NAME}: {mip.num_variables} variables ({mip.num_integers} integer), "
    f"{mip.A.shape[0]} rows"
)

# %% [markdown]
# ## Branch and bound
# `presolve=True` works on HiGHS's reduced MIP and postsolves the incumbent.
# Jaddle's own primal heuristics look for the first incumbent:
# - root probing;
# - fix-and-propagate, classic and bulk (several starts at once);
# - the feasibility jump;
# - the feasibility pump, with a short jump from each rounding it rejects.
#
# The log marks their incumbents `P`, `B`, `J`, `F` and `R` (repair), and
# RINS's `N`. Only Jaddle's heuristics run here: RENS and RINS search their
# sub-MIPs with Jaddle's own branch and bound (`rins_solver="jaddle"`), and
# HiGHS supplies no incumbent. HiGHS still presolves the model, but finds no
# solutions. With HiGHS searching the RINS sub-MIPs instead (the default),
# ns1830653 went from 131,808 to 36,808 within 10 s of the first point in a
# 5-minute test (best known: 20,622). Jaddle's sub-search is slower there.

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
    precision="float32",
    cut_round_epochs=10,
    lp_tolerance=1e-3,
    node_lp_tolerance=1e-3,
    # Any other jl.solve option for node LPs only (root and heuristic LPs
    # keep lp_options), e.g. check convergence every 64 iterations:
    node_lp_options={"iterations_per_epoch": 64},
    max_cuts_per_round=1000,
    heuristic_every=5,
    # Primal heuristics, after cuOpt's GPU heuristics. On ns1830653 the
    # pump finds the first point at about 70 s, so 20 s was too short.
    heuristic_seconds=50,
    probing=True,  # root probing, batched in JAX on large models
    fix_and_propagate=False,  # classic fix-and-propagate, then...
    bulk_rounding=True,  # ...bulk fix-and-propagate from 4 starts at once
    feasibility_jump=True,  # feasibility jump on the device
    feasibility_pump=True,  # the pump, fused with short jumps
    pump_jump_iterations=3000,
    repair_share=0.2,  # cap repair so later jumps and the pump still run
    # Jaddle's heuristics only, no HiGHS: RENS and RINS search their sub-MIPs
    # with Jaddle's own branch and bound, and HiGHS never supplies an
    # incumbent. (With rins_solver="jaddle", rens_seconds=None runs RENS only
    # on MIPs with at most 10,000 nonzeros.)
    rins_solver="highs",
    highs_incumbent_seconds=0.0,
    rens_seconds=None,
    # RINS improves the incumbent in the tree: run it often, and let it use up
    # to 30% of the run.
    rins_every=5,
    heuristic_effort=0.3,
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
    tree_cuts=True,
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
