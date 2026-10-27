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
# Jaddle's branch and bound is deliberately simple (root cuts only, and here
# a first incumbent from HiGHS), so on most MIPLIB instances it will not
# close the gap. Expect a valid bound and an incumbent.

# %%
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jaddle.jaddle_optimisers as jo

jo.configure_jax("float64")

import highspy

import jaddle.jaddle_mip as jm

# %% [markdown]
# ## The instance

# %%
PROBLEM_NAME = "ns1830653"  # name of the MIPLIB problem (without .mps)
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
# Jaddle's own primal heuristics rarely find a first incumbent on MIPLIB
# instances, so here they are switched off and HiGHS supplies one instead
# (`highs_incumbent_seconds`, marked `X` in the log). Jaddle's branch and
# bound then takes over from it.

# %%
result = jm.solve(
    PATH_TO_MPS,
    presolve=True,
    max_seconds=MAX_SECONDS,
    gomory_cuts=True,
    lp_tolerance=1e-6,
    fix_and_propagate=False,
    feasibility_jump=False,
    feasibility_pump=False,
    highs_incumbent_seconds=30,
    rins_seconds=10,  # RINS every 50 nodes, HiGHS on the sub-MIP (log: N)
    batch_size=1,  # on a GPU: solve 16 open nodes per device call
    tree_cuts=True,
    verbose=True,
    log_every_seconds=10,
    device="cpu",
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
