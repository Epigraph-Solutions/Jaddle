# %% [markdown]
# # Fair Logistic Regression on a Large Dataset, with Minibatches
# This is `examples/cp/fair_logistic_regression.py` scaled up to 200,000
# applicants with 1,000 features each: a 1.6 GB training set in float64.
# There, every iteration of `jc.solve` evaluated the log-loss and the fairness
# constraint on all the data. Both are means over applicants, so
# `jaddle_stochastic` can estimate their gradients from a small random batch
# instead, and only go through all the data once per epoch, to anchor its
# variance-reduced (SVRG) gradients and to check convergence.
#
# Convergence is tested on the full problem, so a `"certificate"` stop means
# the same thing as for `jc.solve`. The example is written for a GPU. On a
# CPU the stochastic solves take about 2 s each, but the `jc.solve` check
# takes over 20 seconds.

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
# ## A biased world
# As in the convex example, group B (30% of applicants) has the same skills
# as group A, but its historical labels were marked down. Each applicant has
# 990 skill features, and merit is a random mix of all of them. Ten proxy
# features (think postcode) are each only weakly tied to the group, but
# together they reveal it, even though the group attribute is not a feature.
rng = np.random.default_rng(0)
num_skills, num_proxies = 990, 10
skill_weights = rng.standard_normal(num_skills) / np.sqrt(num_skills)


def sample(m):
    group_b = rng.random(m) < 0.3
    skills = rng.standard_normal((m, num_skills))
    proxies = group_b[:, None] + 1.5 * rng.standard_normal((m, num_proxies))
    merit = skills @ skill_weights + 0.5 * rng.standard_normal(m)
    X = np.hstack([skills, proxies])
    return X, merit - group_b > 0, merit > 0, group_b


X, y, y_fair, group_b = sample(200_000)
X_test, y_test, y_fair_test, group_b_test = sample(50_000)
num_features = X.shape[1]

# %% [markdown]
# ## The problem, one applicant at a time
# A `JaddleCPStochastic` takes each term as a function of the parameters and
# **one example**, plus the data the examples come from (any pytree of
# arrays whose leading axis indexes the examples). The solver averages it
# over the data, or over a sampled batch of it. The ridge penalty and the
# bound `c` do not depend on the data, so they go in the deterministic
# `*_det` parts.
#
#     minimise    mean_i softplus(−ỹᵢ (xᵢ·θ + b)) + λ ||θ||²
#     subject to  |mean_i (sᵢ − s̄)(xᵢ·θ + b)| ≤ c
#
# The constraint is the covariance between group membership and the score,
# written as the two inequalities `cov − c ≤ 0` and `−cov − c ≤ 0`.
#
# Here the data is a dict of arrays with one row per applicant: `X` is
# (N, 1000), and the signed label `y` and centred group `s` are (N,). Each
# iteration draws `batch_size` row numbers and takes those rows from every
# array in the dict, so an applicant's features, label and group stay
# together. The functions below then see one applicant `a`: `a["X"]` has
# shape (1000,), and `a["y"]` and `a["s"]` are scalars. The objective and
# the constraints each draw their own rows.
#
# The data goes to the device once, here. Every problem built from it below
# shares that copy, and the solvers pass it to their compiled programs as an
# argument.
data = jax.tree.map(
    jnp.asarray,
    dict(X=X, y=np.where(y, 1.0, -1.0), s=group_b - group_b.mean()),
)
ridge = 1e-3


def score(params, applicant):
    return applicant["X"] @ params[:-1] + params[-1]


def build_problem(c):
    return js.JaddleCPStochastic(
        num_variables=num_features + 1,
        lower_bounds=jnp.full(num_features + 1, -jnp.inf),
        upper_bounds=jnp.full(num_features + 1, jnp.inf),
        objective=lambda p, a: jax.nn.softplus(-a["y"] * score(p, a)),
        objective_data=data,
        objective_det=lambda p: ridge * jnp.sum(p[:-1] ** 2),
        constraints_ineq=lambda p, a: jnp.stack(
            [a["s"] * score(p, a), -a["s"] * score(p, a)]
        ),
        constraints_ineq_data=data,
        constraints_ineq_det=lambda p: jnp.array([-c, -c]),
    )


tolerances = dict(
    primal_grad_norm_tolerance=1e-5,
    primal_feasibility_tolerance=1e-5,
    complementarity_slack_tolerance=1e-5,
)


def report(name, params):
    params = np.asarray(params)
    approve = X_test @ params[:-1] + params[-1] > 0
    print(
        f"{name:22s} accuracy {np.mean(approve == y_test):.3f}  "
        f"fair accuracy {np.mean(approve == y_fair_test):.3f}  "
        f"gap {approve[~group_b_test].mean() - approve[group_b_test].mean():+.3f}"
    )


# %% [markdown]
# ## Solving with minibatches
# `batch_size` examples are drawn per iteration, and an epoch is about one
# pass over the data in batches (here 200,000 / 1,280 ≈ 160 iterations).
# We solve without the constraint (`c` huge) and with a tight one. The
# unconstrained model learns the bias from the proxies: it approves group A
# far more often. "Fair accuracy" scores the models against the unbiased
# labels.
results = {}
for name, c in [("Unconstrained", 1e3), ("Fair (c = 0.01)", 0.01)]:
    start = time.time()
    results[name] = js.solve(build_problem(c), batch_size=1280, **tolerances)
    print(
        f"{name}: {results[name]['stop_reason']}, "
        f"{results[name]['epochs']} epochs, {time.time() - start:.1f} s"
    )
for name, result in results.items():
    report(name, result["solution"].primal)
fair = results["Fair (c = 0.01)"]

# %% [markdown]
# ## Checking against the full-gradient solver
# `full_cp()` turns the same problem into an ordinary `JaddleCP`, whose
# functions average over all the data, for `jc.solve`. Both solvers should
# find the same model and the same dual (the price of fairness), and the
# stochastic one should get there faster.
problem = build_problem(0.01).full_cp()
start = time.time()
reference = jc.solve(problem, **tolerances)
full_seconds = time.time() - start
print(f"jc.solve: {reference['stop_reason']}, {full_seconds:.1f} s")
print(
    "Largest parameter difference:",
    f"{np.abs(fair['solution'].primal - reference['solution'].primal).max():.1e}",
)
print("Duals, stochastic:", np.round(np.asarray(fair["solution"].dual_ineq), 4))
print("Duals, full:      ", np.round(np.asarray(reference["solution"].dual_ineq), 4))

# %%
