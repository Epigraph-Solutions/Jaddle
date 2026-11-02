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
#     python examples/mip/miplib.py path/to/model.mps [--device cpu|gpu|tpu|auto]
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
        os.path.dirname(__file__), "..", "..", "data", f"{PATH_TO_MPS}.mps"
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
# variable holding the root gap); the root LP and cut rounds in float64 and
# node LPs in float32 (laptop GPUs run float64 at a small fraction of
# float32 speed); 32 node LPs per device call (on ns1830653 that halves the
# device time per node against 16; tree cuts are off when batching).

# %%
result = jm.solve(
    PATH_TO_MPS,
    branching="reliability",
    presolve=True,
    max_seconds=MAX_SECONDS,
    verbose=True,
    log_every_seconds=10,
    device=DEVICE,
    batch_size=1,
    tree_cuts=True,
    node_lp_tolerance=1e-6,
    node_crossover=True,
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
highs.setOptionValue("solver", "pdlp")
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
