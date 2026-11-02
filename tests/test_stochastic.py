"""Tests for the stochastic solver, jaddle.jaddle_stochastic."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaddle.jaddle_convex as jc
import jaddle.jaddle_stochastic as js

TOLS = dict(
    primal_grad_norm_tolerance=1e-6,
    primal_feasibility_tolerance=1e-6,
    complementarity_slack_tolerance=1e-6,
)
LOOSE = dict(
    primal_grad_norm_tolerance=1e-3,
    primal_feasibility_tolerance=1e-3,
    complementarity_slack_tolerance=1e-3,
)


def solve(scp, **kwargs):
    kwargs.setdefault("max_epochs", 300)
    kwargs.setdefault("batch_size", 64)
    return js.solve(scp, **kwargs)


def fair_logistic(n=4000, c=0.02, seed=0):
    """Ridge logistic regression with a finite-sum covariance (demographic
    parity) constraint, as in examples/cp/fair_logistic_regression.py. The
    group shifts two features, so the constraint binds at this ``c``."""
    rng = np.random.default_rng(seed)
    group = rng.random(n) < 0.3
    skills = rng.standard_normal((n, 3))
    proxies = group[:, None] + 0.7 * rng.standard_normal((n, 2))
    merit = skills @ np.array([1.0, 0.7, 0.5]) + 0.5 * rng.standard_normal(n)
    data = dict(
        X=np.hstack([skills, proxies]),
        y=np.where(merit - group > 0, 1.0, -1.0),
        s=group - group.mean(),
    )
    m = data["X"].shape[1] + 1

    def score(p, ex):
        return ex["X"] @ p[:-1] + p[-1]

    def covariance(p, ex):
        return jnp.stack([ex["s"] * score(p, ex), -ex["s"] * score(p, ex)])

    return js.JaddleCPStochastic(
        num_variables=m,
        lower_bounds=jnp.full(m, -jnp.inf),
        upper_bounds=jnp.full(m, jnp.inf),
        objective=lambda p, ex: jax.nn.softplus(-ex["y"] * score(p, ex)),
        objective_data=data,
        objective_det=lambda p: 1e-3 * jnp.sum(p[:-1] ** 2),
        constraints_ineq=covariance,
        constraints_ineq_data=data,
        constraints_ineq_det=lambda p: jnp.array([-c, -c]),
    )


def simplex_least_squares(n=2000, m=10, seed=0):
    """Least squares over the probability simplex: a sampled objective with
    deterministic constraints (an equality row and the bounds)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, m))
    b = A @ rng.dirichlet(np.ones(m)) + 0.1 * rng.standard_normal(n)
    return js.JaddleCPStochastic(
        num_variables=m,
        lower_bounds=jnp.zeros(m),
        upper_bounds=jnp.full(m, jnp.inf),
        objective=lambda x, ex: 0.5 * (ex["a"] @ x - ex["b"]) ** 2,
        objective_data=dict(a=A, b=b),
        constraints_eq_det=lambda x: jnp.array([jnp.sum(x) - 1.0]),
    )


def test_full_cp_matches_handwritten():
    rng = np.random.default_rng(1)
    n, m = 53, 4
    A, b, w = rng.standard_normal((n, m)), rng.standard_normal(n), rng.random(n)
    scp = js.JaddleCPStochastic(
        num_variables=m,
        lower_bounds=-jnp.ones(m),
        upper_bounds=jnp.ones(m),
        objective=lambda x, ex: jnp.log1p(jnp.exp(ex["a"] @ x - ex["b"])),
        objective_data=dict(a=A, b=b),
        objective_det=lambda x: jnp.sum(x**2),
        constraints_eq_det=lambda x: jnp.array([x[0] - x[1]]),
        constraints_ineq=lambda x, wi: jnp.stack([wi * x[2], wi * x[3] ** 2]),
        constraints_ineq_data=w,
    )
    assert scp.num_examples == {"objective": n, "constraints_ineq": n}
    assert (scp.num_eq_constraints(), scp.num_ineq_constraints()) == (1, 2)

    def objective(x):
        return jnp.mean(jnp.log1p(jnp.exp(A @ x - b))) + jnp.sum(x**2)

    def ineq(x):
        return jnp.stack([jnp.mean(w) * x[2], jnp.mean(w) * x[3] ** 2])

    x = jnp.asarray(rng.standard_normal(m))
    for chunk in [None, 7, 53, 100]:
        full = scp.full_cp(eval_chunk_size=chunk)
        np.testing.assert_allclose(full.objective(x), objective(x), rtol=1e-12)
        np.testing.assert_allclose(full.constraints_ineq(x), ineq(x), rtol=1e-12)
        np.testing.assert_allclose(full.constraints_eq(x), [x[0] - x[1]])
        np.testing.assert_allclose(
            jax.grad(full.objective)(x), jax.grad(objective)(x), rtol=1e-12
        )
        # Forward mode too: gradient scaling takes Jacobian-vector products.
        np.testing.assert_allclose(
            jax.jacfwd(full.constraints_ineq)(x), jax.jacfwd(ineq)(x), rtol=1e-12
        )


def test_oracle_is_unbiased():
    scp = fair_logistic(n=300)
    full = scp.full_cp()
    oracle = scp.oracle(batch_size=16)
    x = jnp.linspace(-0.5, 0.5, scp.num_variables)
    duals = (jnp.array([0.3, 0.7]), jnp.zeros(0))

    def primal_gradient(lagrangian_map):
        one = jnp.ones(())
        _, vjp = jax.vjp(lagrangian_map, x)
        return vjp((one,) + duals)[0]

    @jax.jit
    def sampled(key):
        sample = oracle.draw(key)
        values = oracle.lagrangian_map(x, sample)
        return values, primal_gradient(lambda z: oracle.lagrangian_map(z, sample))

    keys = jax.random.split(jax.random.PRNGKey(0), 20000)
    values, gradients = jax.vmap(sampled)(keys)
    exact = (full.objective(x), full.constraints_ineq(x), full.constraints_eq(x))
    exact_gradient = primal_gradient(
        lambda z: (full.objective(z), full.constraints_ineq(z), full.constraints_eq(z))
    )
    for estimate, target in zip(values, exact):
        np.testing.assert_allclose(jnp.mean(estimate, axis=0), target, atol=5e-3)
    np.testing.assert_allclose(jnp.mean(gradients, axis=0), exact_gradient, atol=5e-3)


@pytest.mark.parametrize("variance_reduction", ["svrg", None])
@pytest.mark.parametrize("update_mode", ["extragradient", "pdhg", "alternating"])
def test_exact_oracle_reproduces_convex_solve(
    monkeypatch, update_mode, variance_reduction
):
    """With an 'oracle' that returns the exact map, the stochastic iterations
    are the deterministic ones (the SVRG correction is zero up to rounding,
    which the adaptive step can amplify to about the tolerance)."""
    scp = fair_logistic(n=500)
    full = scp.full_cp()

    def exact_map(x, sample):
        return full.objective(x), full.constraints_ineq(x), full.constraints_eq(x)

    exact = jc.GradientOracle(
        draw=lambda key: jnp.zeros(()),
        lagrangian_map=exact_map,
        constraints=lambda x, sample: exact_map(x, sample)[1:],
    )
    monkeypatch.setattr(scp, "oracle", lambda batch_size: exact)
    settings = dict(update_mode=update_mode, iterations_per_epoch=50, max_epochs=200)
    expected = jc.solve(full, **settings, **TOLS)
    result = solve(
        scp, variance_reduction=variance_reduction, step_decay=None, **settings, **TOLS
    )
    assert result["stop_reason"] == expected["stop_reason"] == "certificate"
    assert result["epochs"] == expected["epochs"]
    for a, b in zip(result["solution"], expected["solution"]):
        np.testing.assert_allclose(a, b, atol=1e-5)


def test_sampled_constraint_matches_convex_solve():
    scp = fair_logistic()
    expected = jc.solve(scp.full_cp(), **TOLS)
    result = solve(scp, **TOLS)
    assert result["converged"]
    assert result["stop_reason"] == "certificate"
    # The covariance constraint binds, with a nonzero dual.
    assert float(jnp.max(expected["solution"].dual_ineq)) > 1e-2
    np.testing.assert_allclose(
        result["solution"].primal, expected["solution"].primal, atol=1e-4
    )
    np.testing.assert_allclose(
        result["solution"].dual_ineq, expected["solution"].dual_ineq, atol=1e-3
    )


@pytest.mark.parametrize("update_mode", ["extragradient", "pdhg", "alternating"])
def test_sampled_objective_every_update_mode(update_mode):
    scp = simplex_least_squares()
    expected = jc.solve(scp.full_cp(), **TOLS)
    result = solve(scp, update_mode=update_mode, **TOLS)
    assert result["stop_reason"] == "certificate"
    np.testing.assert_allclose(
        result["solution"].primal, expected["solution"].primal, atol=1e-5
    )


def test_plain_sgda_with_step_decay():
    scp = simplex_least_squares()
    expected = jc.solve(scp.full_cp(), **TOLS)
    result = solve(scp, variance_reduction=None, **LOOSE)
    assert result["stop_reason"] == "certificate"
    np.testing.assert_allclose(
        result["solution"].primal, expected["solution"].primal, atol=1e-2
    )


def test_eval_chunk_size_and_precision():
    scp = simplex_least_squares(n=500)
    result = solve(scp, eval_chunk_size=128, precision="float32", **LOOSE)
    assert result["stop_reason"] == "certificate"
    assert result["solution"].primal.dtype == jnp.float32


def test_seed_reproducibility():
    scp = simplex_least_squares(n=500)
    settings = dict(max_epochs=3, iterations_per_epoch=20)
    first = solve(scp, seed=1, **settings, **TOLS)
    again = solve(scp, seed=1, **settings, **TOLS)
    other = solve(scp, seed=2, **settings, **TOLS)
    np.testing.assert_array_equal(first["solution"].primal, again["solution"].primal)
    assert not np.array_equal(first["solution"].primal, other["solution"].primal)


@pytest.mark.parametrize(
    "options",
    [
        dict(batch_size=0),
        dict(batch_size=1.5),
        dict(variance_reduction="saga"),
        dict(step_decay=0.0),
        dict(step_decay="fast"),
        dict(iterations_per_epoch=0),
        dict(eval_chunk_size=0),
    ],
)
def test_invalid_options(options):
    with pytest.raises(ValueError):
        solve(simplex_least_squares(n=50), **options)


def test_invalid_problems():
    m = 3
    box = dict(num_variables=m, lower_bounds=jnp.zeros(m), upper_bounds=jnp.ones(m))
    with pytest.raises(ValueError, match="together"):
        js.JaddleCPStochastic(**box, objective=lambda x, ex: x @ ex)
    with pytest.raises(ValueError, match="no term has data"):
        js.JaddleCPStochastic(**box, objective_det=lambda x: x @ x)
    with pytest.raises(ValueError, match="leading"):
        js.JaddleCPStochastic(
            **box,
            objective=lambda x, ex: x @ ex["a"],
            objective_data=dict(a=np.ones((4, m)), b=np.ones(5)),
        )
    with pytest.raises(TypeError):
        js.solve(jc.JaddleCP(m, jnp.sum, jnp.sum, jnp.sum, box["lower_bounds"], box["upper_bounds"]))
