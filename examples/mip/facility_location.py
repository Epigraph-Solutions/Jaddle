# %% [markdown]
# # Capacitated Facility Location with Jaddle's MIP Solver
# A company can open warehouses at 8 candidate sites to serve 20 customers.
# Opening site f costs a fixed amount and gives it a capacity; shipping one
# unit from site f to customer k costs `ship[f, k]`. Each customer's demand
# must be met in full. Which sites should open?
#
# ```
# minimise    Σ_f fixed_f y_f + Σ_fk ship_fk x_fk
# subject to  Σ_f x_fk = demand_k               for every customer k
#             Σ_k x_fk <= capacity_f y_f        for every site f
#             y_f ∈ {0, 1},  x_fk >= 0
# ```
#
# The openings `y` are binary and the flows `x` continuous, so this is a
# mixed-integer LP. `jaddle_mip.solve` runs branch and bound on it, solving
# the LP at each node with Jaddle's first-order LP solver. The answer is
# checked against HiGHS.

# %%
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # small problem: CPU is faster

import jaddle.jaddle_optimisers as jo

jo.configure_jax("float64")

import highspy
import numpy as np
import scipy.sparse as sp

import jaddle.jaddle_linear as jl
import jaddle.jaddle_mip as jm

# %% [markdown]
# ## The data
# Sites and customers are random points in the unit square; shipping cost is
# proportional to distance.

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
# ## Building the MIP
# The variables are the 8 openings followed by the 8 × 20 flows. Jaddle's
# standard form has equality rows `A_eq x = b_eq` and `<=` rows
# `A_ineq x <= b_ineq`, so the capacity rows become
# `Σ_k x_fk − capacity_f y_f <= 0`.

# %%
n = n_sites + n_sites * n_customers


def flow(f, k):
    return n_sites + f * n_customers + k


A_eq = sp.lil_matrix((n_customers, n))
A_ineq = sp.lil_matrix((n_sites, n))
for f in range(n_sites):
    A_ineq[f, f] = -capacity[f]
    for k in range(n_customers):
        A_eq[k, flow(f, k)] = 1.0
        A_ineq[f, flow(f, k)] = 1.0

lp = jl.LP(
    c=np.concatenate([fixed, ship.ravel()]),
    A_eq=A_eq.tocsc(),
    b_eq=demand,
    A_ineq=A_ineq.tocsc(),
    b_ineq=np.zeros(n_sites),
    lower_bounds=np.zeros(n),
    upper_bounds=np.concatenate([np.ones(n_sites), np.full(n - n_sites, np.inf)]),
)
integer = np.arange(n) < n_sites  # the openings are integer (binary)
mip = jm.MIP(lp, integer)

# %% [markdown]
# ## Solving
# The log shows the node count, the open nodes, the best bound, the incumbent
# and the gap. Lines marked `H` found an incumbent by rounding an LP point,
# lines marked `T` at a node whose LP point was already integral.

# %%
result = jm.solve(mip, verbose=True)

opened = np.flatnonzero(result["x"][:n_sites] > 0.5)
print(f"\nStatus: {result['status']}, {result['nodes']} nodes")
print(f"Total cost: {result['objective']:.4f} (bound {result['bound']:.4f})")
print(f"Open sites: {opened.tolist()}")
flows = result["x"][n_sites:].reshape(n_sites, n_customers)
for f in opened:
    print(
        f"  site {f}: ships {flows[f].sum():6.2f} of capacity {capacity[f]:6.2f}"
    )

# %% [markdown]
# ## Checking against HiGHS
# `jm.to_highs` turns the MIP into a HiGHS model, which HiGHS's own
# branch and cut solves exactly.

# %%
highs = highspy.Highs()
highs.setOptionValue("output_flag", False)
highs.passModel(jm.to_highs(mip))
highs.run()
reference = highs.getInfo().objective_function_value
print(f"HiGHS optimum: {reference:.4f}")
relative_error = abs(result["objective"] - reference) / abs(reference)
print(f"Relative difference: {relative_error:.2e}")
assert result["status"] == "optimal"
assert relative_error <= 1e-4  # the default mip_rel_gap
