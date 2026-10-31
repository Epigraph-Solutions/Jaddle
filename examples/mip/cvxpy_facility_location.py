# %% [markdown]
# # Capacitated Facility Location in CVXPY
# The model from `facility_location.py`, written in CVXPY instead of built
# from arrays. A company can open warehouses at 8 candidate sites to serve 20
# customers. Opening site f costs a fixed amount and gives it a capacity;
# shipping one unit from site f to customer k costs `ship[f, k]`. Each
# customer's demand must be met in full.
#
# The openings are declared `boolean=True`, which makes this a mixed-integer
# LP. `JaddleSolver` hands such problems to `jaddle_mip.solve`, Jaddle's
# branch and bound. The answer is checked against HiGHS.

# %%
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # small problem: CPU is faster

import jaddle.jaddle_optimisers as jo

jo.configure_jax("float64")

import cvxpy as cp
import numpy as np

from jaddle.jaddle_cvxpy import JaddleSolver

# %% [markdown]
# ## The data
# The same random instance as `facility_location.py`.

# %%
rng = np.random.default_rng(7)
n_sites, n_customers = 8, 20
sites = rng.uniform(0, 1, (n_sites, 2))
customers = rng.uniform(0, 1, (n_customers, 2))
demand = rng.uniform(5, 15, n_customers)
capacity = rng.uniform(40, 80, n_sites)
fixed = rng.uniform(60, 120, n_sites)
ship = 20 * np.linalg.norm(sites[:, None, :] - customers[None, :, :], axis=2)

# %% [markdown]
# ## The model

# %%
open_site = cp.Variable(n_sites, boolean=True)
flow = cp.Variable((n_sites, n_customers), nonneg=True)
problem = cp.Problem(
    cp.Minimize(fixed @ open_site + cp.sum(cp.multiply(ship, flow))),
    [
        cp.sum(flow, axis=0) == demand,
        cp.sum(flow, axis=1) <= cp.multiply(capacity, open_site),
    ],
)

# %% [markdown]
# ## Solving
# Keyword arguments of `problem.solve` go to `jaddle_mip.solve`, here a
# tighter relative gap than its default of 1e-4. Branch and bound gives no
# duals; `solver_stats.extra_stats` holds the proven bound, gap and node
# count.

# %%
problem.solve(solver=JaddleSolver(), mip_rel_gap=1e-6, verbose=True)
stats = problem.solver_stats.extra_stats

opened = np.flatnonzero(open_site.value > 0.5)
print(f"\nStatus: {problem.status}, {stats['nodes']} nodes, gap {stats['gap']:.1e}")
print(f"Total cost: {problem.value:.4f}")
print(f"Open sites: {opened.tolist()}")
for f in opened:
    shipped = flow.value[f].sum()
    print(f"  site {f}: ships {shipped:6.2f} of capacity {capacity[f]:6.2f}")

# %% [markdown]
# ## Checking against HiGHS

# %%
jaddle_cost = problem.value
problem.solve(solver=cp.HIGHS)
print(f"HiGHS optimum: {problem.value:.4f}")
relative_error = abs(jaddle_cost - problem.value) / abs(problem.value)
print(f"Relative difference: {relative_error:.2e}")
assert relative_error <= 1e-6

# %%
