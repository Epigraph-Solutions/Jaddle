# %% [markdown]
# # Risk-Constrained Kelly Betting over Two Million Scenarios
# A Kelly investor picks portfolio weights `w` to maximise the expected log of
# wealth after one period, which maximises the long-run growth rate. Full
# Kelly is famously aggressive. Busseti, Ryu and Boyd (2016,
# "Risk-Constrained Kelly Gambling") tame it with one extra convex
# constraint, which bounds the chance that wealth ever falls below a fraction
# `α` of its starting value by `β`:
#
#     maximise    mean_i log(1 + xᵢ·w)
#     subject to  mean_i (1 + xᵢ·w)^(−λ) <= 1,   λ = log β / log α
#                 1ᵀw <= 1,   w >= 0
#
# Each `xᵢ` is a scenario of the risky assets' returns over cash, and the
# rest of the wealth, `1 − 1ᵀw`, sits in cash.
#
# The constraint is about the tail: crashes dominate it. A model fitted to a
# thousand scenarios sees only a handful of them and overfits, so it breaks
# its own risk limit on new data. The fix is many more scenarios. But a conic
# solver needs one cone per scenario (see `examples/conic/cvxpy_kelly.py`),
# so millions of scenarios are out of reach for it. Both terms are means
# over scenarios with only 100 shared weights, which is exactly the shape
# `jaddle_stochastic` is built for: it estimates the gradients from small
# random batches and goes through all the scenarios only a few times per
# epoch.

# %%
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time

import jax
import jax.numpy as jnp
import numpy as np
import jaddle.jaddle_convex as jc
import jaddle.jaddle_stochastic as js
import jaddle.jaddle_optimisers as jo

jo.configure_jax("float64")

# %% [markdown]
# ## Scenarios
# 100 assets with different means, volatilities and market betas, as in the
# conic Kelly example, plus a rare crash: in 0.5% of periods the market
# falls by about 40%. That is about 10,000 crashes in the 2,000,000 training
# scenarios, but only about 5 in a sample of 1,000. The test set is a further
# million fresh scenarios.

# %%
rng = np.random.default_rng(0)
num_assets = 100
mean = rng.uniform(0.00, 0.08, num_assets)
vol = rng.uniform(0.05, 0.40, num_assets)
beta = rng.uniform(0.3, 1.2, num_assets)


def sample(m):
    market = rng.normal(0.0, 0.15, (m, 1))
    crash = rng.random((m, 1)) < 0.005
    market = np.where(crash, -0.4 + 0.1 * rng.standard_normal((m, 1)), market)
    gross = 1.0 + mean + beta * market + vol * rng.standard_normal((m, num_assets))
    return (
        np.maximum(gross, 0.05) - 1.0
    )  # returns over cash; an asset can't fall below 0


X = sample(2_000_000)
X_test = sample(1_000_000)

# %% [markdown]
# ## The problem, one scenario at a time
# The limit here is "never lose 10% of starting wealth, with probability at
# least 90%": `α = 0.9` and `β = 0.1`, so `λ ≈ 21.9`.
#
# The solver keeps `w` in its box `[0, 1]`, but `1ᵀw <= 1` only holds at the
# solution, so along the way wealth `1 + x·w` could reach zero, where `log`
# and the power are undefined. Below `t₀ = 0.5` we swap each for its
# second-order Taylor expansion at `t₀`. The patched functions are smooth,
# still concave and convex, and equal to the originals wherever wealth stays
# above `t₀`. We check at the end that it does in every scenario, so the
# patch never changes the answer.
#
# The start point is `w = 0`, all cash, which satisfies every constraint.

# %%
alpha, beta_limit = 0.9, 0.1
lam = np.log(beta_limit) / np.log(alpha)
t0 = 0.5


def patched(f, df, d2f):
    """``f`` above ``t0`` and its quadratic expansion at ``t0`` below."""

    def g(t):
        d = t - t0
        quadratic = f(t0) + df(t0) * d + 0.5 * d2f(t0) * d**2
        return jnp.where(t >= t0, f(jnp.maximum(t, t0)), quadratic)

    return g


safe_log = patched(jnp.log, lambda t: 1 / t, lambda t: -1 / t**2)
safe_risk = patched(
    lambda t: t**-lam,
    lambda t: -lam * t ** (-lam - 1),
    lambda t: lam * (lam + 1) * t ** (-lam - 2),
)
data = jnp.asarray(X)  # to the device once; every problem below shares it


def build_problem(data, risk_limit=True):
    budget = lambda w: w.sum() - 1.0  # noqa: E731
    if risk_limit:
        constraints = dict(
            constraints_ineq=lambda w, x: jnp.stack([safe_risk(1 + x @ w), 0.0]),
            constraints_ineq_data=data,
            constraints_ineq_det=lambda w: jnp.stack([-1.0, budget(w)]),
        )
    else:
        constraints = dict(constraints_ineq_det=lambda w: jnp.stack([budget(w)]))
    return js.JaddleCPStochastic(
        num_variables=num_assets,
        lower_bounds=jnp.zeros(num_assets),
        upper_bounds=jnp.ones(num_assets),
        objective=lambda w, x: -safe_log(1 + x @ w),
        objective_data=data,
        **constraints,
    )


tolerances = dict(
    primal_grad_norm_tolerance=1e-5,
    primal_feasibility_tolerance=1e-5,
    complementarity_slack_tolerance=1e-5,
)


def drawdown_probability(w, periods=100):
    """The fraction of wealth paths, each ``periods`` test scenarios long,
    that ever fall below ``alpha``."""
    wealth = np.cumprod(1 + (X_test @ w).reshape(-1, periods), axis=1)
    return np.mean(wealth.min(axis=1) < alpha)


def report(name, w):
    wealth = 1 + X_test @ w
    print(
        f"{name:24s} growth {np.mean(np.log(wealth)):.4f}  "
        f"risk {np.mean(wealth**-lam):.3f}  "
        f"P(drawdown) {drawdown_probability(w):.3f}  "
        f"cash {1 - w.sum():.2f}  assets {np.sum(w > 1e-4)}"
    )


# %% [markdown]
# ## Solving with minibatches
# Full Kelly first, then with the risk limit. "Growth" is the expected log
# return per period, "risk" the constraint's mean `(1 + x·w)^(−λ)` (the limit
# is 1), and "P(drawdown)" the fraction of 10,000 simulated 100-period wealth
# paths that ever fall below 90%. All three are measured on the test set.
results = {}
for name, risk_limit in [("Full Kelly", False), ("Risk-constrained", True)]:
    start = time.time()
    results[name] = js.solve(
        build_problem(data, risk_limit), batch_size=1280, **tolerances
    )
    print(
        f"{name}: {results[name]['stop_reason']}, "
        f"{results[name]['epochs']} epochs, {time.time() - start:.1f} s"
    )
    assert results[name]["stop_reason"] == "certificate"
for name, result in results.items():
    report(name, np.asarray(result["solution"].primal))
w_risk = np.asarray(results["Risk-constrained"]["solution"].primal)

# %% [markdown]
# Full Kelly grows fastest but breaks the limit. The risk-constrained
# portfolio gives up some growth, holds cash, and keeps its risk at the limit
# on the test set. The Busseti bound is conservative, so its simulated
# drawdown probability comes in well under `β`. Wealth stays above `t₀` in
# every training scenario, so the patched functions match the true ones at
# both solutions.
for result in results.values():
    assert np.min(1 + X @ np.asarray(result["solution"].primal)) > t0

# %% [markdown]
# ## A thousand scenarios is not enough
# The same problem fitted to 1,000 scenarios is small, so `jc.solve` handles
# it directly. Each fit below uses a different slice of the training set and
# meets the limit on its own scenarios. On the test set they break it, because
# a thousand scenarios hold too few crashes to pin down the tail: they take
# more risk than they think, and their simulated drawdown probabilities come
# out well above the full-data portfolio's. (These stay under `β`, thanks to
# the bound's slack, but the guarantee itself is gone.)
print("1,000-scenario fits, measured on the test set:")
broken = 0
for k in range(5):
    small = build_problem(jnp.asarray(X[1000 * k : 1000 * (k + 1)])).full_cp()
    w = np.asarray(jc.solve(small, **tolerances)["solution"].primal)
    report(f"  slice {k}", w)
    broken += np.mean((1 + X_test @ w) ** -lam) > 1
print(f"{broken} of 5 break the limit out of sample")

# %% [markdown]
# ## The price of safety
# The risk constraint's dual is the growth given up per unit of risk budget:
# loosening the limit from 1 to 1 + ε would raise the growth rate by about
# dual × ε.
print(
    "Dual of the risk limit:",
    f"{results['Risk-constrained']['solution'].dual_ineq[0]:.4f}",
)

# %% [markdown]
# ## Checking against the full-gradient solver
# `full_cp()` turns the same problem into an ordinary `JaddleCP` that averages
# over all two million scenarios at every step, for `jc.solve`. Both solvers
# should find the same growth rate and dual. The objective is very flat along
# some directions (assets that are almost interchangeable), so the weights
# themselves agree less tightly than the growth rate does.
#
# On a CPU the two stochastic solves take about 2.5 s and 6.5 s, compilation
# included, and `jc.solve` about 40 s. On an RTX 5070 Ti laptop GPU they take
# about 2.5 s and 6 s, and `jc.solve` about 6.5 s.
problem = build_problem(data).full_cp()
start = time.time()
reference = jc.solve(problem, **tolerances)
print(f"jc.solve: {reference['stop_reason']}, {time.time() - start:.1f} s")
w_ref = np.asarray(reference["solution"].primal)
print(
    "Growth, stochastic vs full:",
    f"{-float(problem.objective(jnp.asarray(w_risk))):.6f}",
    f"{-float(problem.objective(jnp.asarray(w_ref))):.6f}",
)
print("Largest weight difference:", f"{np.abs(w_risk - w_ref).max():.1e}")
print(
    "Risk duals, stochastic vs full:",
    f"{results['Risk-constrained']['solution'].dual_ineq[0]:.4f}",
    f"{reference['solution'].dual_ineq[0]:.4f}",
)

# %%
