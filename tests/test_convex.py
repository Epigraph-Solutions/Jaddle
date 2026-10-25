"""Tests for the convex solver, jaddle.jaddle_convex."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

import jaddle.jaddle_convex as jc
import jaddle.jaddle_optimisers as jo

TOLS = dict(
    primal_grad_norm_tolerance=1e-6,
    primal_feasibility_tolerance=1e-6,
    complementarity_slack_tolerance=1e-6,
)


def project_onto_simplex(v):
    """Exact Euclidean projection onto {x >= 0, sum x = 1}."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > css - 1)[0][-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def isotonic_fit(y, lo, hi):
    """Exact box-constrained isotonic regression: pool-adjacent-violators, then
    clip to the box (clipping preserves optimality for this problem)."""
    blocks = []  # [mean, size]
    for value in y:
        blocks.append([value, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, s2 = blocks.pop()
            m1, s1 = blocks.pop()
            blocks.append([(m1 * s1 + m2 * s2) / (s1 + s2), s1 + s2])
    fit = np.concatenate([np.full(s, m) for m, s in blocks])
    return np.clip(fit, lo, hi)


def simplex_problem(n=50, seed=0):
    a = np.random.default_rng(seed).standard_normal(n)
    cp = jc.JaddleCP(
        num_variables=n,
        objective=lambda x: jnp.sum((x - a) ** 2),
        constraints_eq=lambda x: jnp.array([jnp.sum(x) - 1.0]),
        constraints_ineq=lambda x: jnp.zeros(0),
        lower_bounds=jnp.zeros(n),
        upper_bounds=jnp.full(n, jnp.inf),
    )
    return cp, project_onto_simplex(a)


def solve(cp, **kwargs):
    kwargs.setdefault("max_epochs", 300)
    return jc.solve(cp, iterations_per_epoch=200, **TOLS, **kwargs)


@pytest.mark.parametrize(
    "update_mode", ["extragradient", "pdhg", "alternating"]
)
def test_simplex_projection(update_mode):
    cp, expected = simplex_problem()
    result = solve(cp, update_mode=update_mode)
    assert result["converged"]
    assert result["stop_reason"] == "certificate"
    np.testing.assert_allclose(result["solution"].primal, expected, atol=1e-5)


def test_isotonic_regression():
    n = 60
    x = np.linspace(-1, 1, n)
    y = x**3 + 0.15 * np.random.default_rng(0).standard_normal(n)
    cp = jc.JaddleCP(
        num_variables=n,
        objective=lambda p: jnp.sum((p - y) ** 2),
        constraints_eq=lambda p: jnp.zeros(0),
        constraints_ineq=lambda p: p[:-1] - p[1:],
        lower_bounds=-jnp.ones(n),
        upper_bounds=jnp.ones(n),
    )
    result = solve(cp)
    assert result["converged"]
    np.testing.assert_allclose(
        result["solution"].primal, isotonic_fit(y, -1.0, 1.0), atol=1e-4
    )


DIAG_MODES = pytest.mark.parametrize(
    "update_mode", ["alternating", "extragradient", "pdhg"]
)


@DIAG_MODES
def test_diag_adapt_simplex(capsys, update_mode):
    cp, expected = simplex_problem()
    result = solve(
        cp, verbose=True, update_mode=update_mode, restarts=True, diag_adapt=True
    )
    assert result["stop_reason"] == "certificate"
    np.testing.assert_allclose(result["solution"].primal, expected, atol=1e-5)
    out = capsys.readouterr().out
    assert "Restart" not in out or "t-spread" in out


@DIAG_MODES
def test_diag_adapt_isotonic_regression(update_mode):
    n = 60
    x = np.linspace(-1, 1, n)
    y = x**3 + 0.15 * np.random.default_rng(0).standard_normal(n)
    cp = jc.JaddleCP(
        num_variables=n,
        objective=lambda p: jnp.sum((p - y) ** 2),
        constraints_eq=lambda p: jnp.zeros(0),
        constraints_ineq=lambda p: p[:-1] - p[1:],
        lower_bounds=-jnp.ones(n),
        upper_bounds=jnp.ones(n),
    )
    result = solve(cp, update_mode=update_mode, restarts=True, diag_adapt=True)
    assert result["converged"]
    np.testing.assert_allclose(
        result["solution"].primal, isotonic_fit(y, -1.0, 1.0), atol=1e-4
    )


@pytest.mark.parametrize(
    "options",
    [
        dict(restarts=False),
        dict(adaptive_eta=False, optimiser=jo.gd(0.1)),
    ],
)
def test_diag_adapt_without_its_requirements_has_no_effect(options):
    # The diagonal changes only at restarts and only the adaptive line search
    # measures in it, so elsewhere diag_adapt=True runs as diag_adapt=False.
    cp, _ = simplex_problem()
    kwargs = dict(max_epochs=5, **options)
    on = solve(cp, diag_adapt=True, **kwargs)
    off = solve(cp, diag_adapt=False, **kwargs)
    np.testing.assert_array_equal(on["solution"].primal, off["solution"].primal)


def misscaled_isotonic(objective_scale, row_scale, n=60):
    # test_isotonic_regression's problem with the objective and each
    # inequality row multiplied by a scale.
    x = np.linspace(-1, 1, n)
    y = x**3 + 0.15 * np.random.default_rng(0).standard_normal(n)
    cp = jc.JaddleCP(
        num_variables=n,
        objective=lambda p: objective_scale * jnp.sum((p - y) ** 2),
        constraints_eq=lambda p: jnp.zeros(0),
        constraints_ineq=lambda p: row_scale * (p[:-1] - p[1:]),
        lower_bounds=-jnp.ones(n),
        upper_bounds=jnp.ones(n),
    )
    return cp, isotonic_fit(y, -1.0, 1.0)


def test_gradient_scaling_fixes_misscaled_rows_and_objective():
    # Objective x1e3 and rows x 10^U(-3, 3). The tolerances are absolute in the
    # problem's own (scaled-up) units, hence 1e-3.
    tols = dict(
        primal_grad_norm_tolerance=1e-3,
        primal_feasibility_tolerance=1e-3,
        complementarity_slack_tolerance=1e-3,
    )
    rows = 10.0 ** np.random.default_rng(1).uniform(-3, 3, 59)
    reference_cp, fit = misscaled_isotonic(1.0, 1.0)
    cp, _ = misscaled_isotonic(1e3, rows)

    unscaled = jc.solve(cp, iterations_per_epoch=200, max_epochs=50, **tols,
                        gradient_scaling=False)
    assert not unscaled["converged"]

    result = jc.solve(cp, iterations_per_epoch=200, max_epochs=300, **tols,
                      gradient_scaling=True)
    assert result["stop_reason"] == "certificate"
    np.testing.assert_allclose(result["solution"].primal, fit, atol=1e-5)
    # The returned duals are the given problem's: the well-scaled problem's
    # multipliers times objective scale / row scale.
    reference = solve(reference_cp)
    expected = np.asarray(reference["solution"].dual_ineq) * 1e3 / rows
    np.testing.assert_allclose(
        result["solution"].dual_ineq, expected, atol=1e-4 * (1 + np.abs(expected).max())
    )


@pytest.mark.parametrize("m, probes, exact", [(40, 100, True), (10, 16, True), (40, 16, False)])
def test_gradient_scales_row_norms(m, probes, exact):
    # n=60: exact by forward mode when n <= probes, by reverse mode when
    # m <= probes, otherwise estimated from `probes` random JVPs.
    rng = np.random.default_rng(0)
    A = rng.standard_normal((m, 60)) * 10.0 ** rng.uniform(-2, 2, (m, 1))
    cp = jc.JaddleCP(
        num_variables=60,
        objective=lambda x: 5.0 * jnp.sum(x),
        constraints_eq=lambda x: jnp.zeros(0),
        constraints_ineq=lambda x: jnp.asarray(A) @ x - 1.0,
        lower_bounds=-jnp.ones(60),
        upper_bounds=jnp.ones(60),
    )
    sigma, e = jc._gradient_scales(cp, jnp.zeros(60), probes)
    assert float(sigma) == pytest.approx(1.0 / (5.0 * np.sqrt(60)))
    true = 1.0 / np.linalg.norm(A, axis=1)
    if exact:
        np.testing.assert_allclose(e, true, rtol=1e-10)
    else:
        ratio = np.asarray(e) / true
        assert ratio.min() > 0.5 and ratio.max() < 2.0


def test_gradient_scaling_zero_gradient_row():
    # ‖x‖² <= 1 has a zero gradient at the start point x0 = 0: its scale
    # stays 1, and the solve still certifies.
    a = np.random.default_rng(2).standard_normal(20)
    cp = jc.JaddleCP(
        num_variables=20,
        objective=lambda x: jnp.sum((x - a) ** 2),
        constraints_eq=lambda x: jnp.zeros(0),
        constraints_ineq=lambda x: jnp.array([x @ x - 1.0]),
        lower_bounds=jnp.full(20, -jnp.inf),
        upper_bounds=jnp.full(20, jnp.inf),
    )
    _, e = jc._gradient_scales(cp, jnp.zeros(20), 16)
    np.testing.assert_array_equal(e, [1.0])
    result = solve(cp, update_mode="extragradient", gradient_scaling=True)
    assert result["stop_reason"] == "certificate"
    np.testing.assert_allclose(result["solution"].primal, a / np.linalg.norm(a), atol=1e-5)


def test_epoch_budget():
    cp, _ = simplex_problem()
    result = jc.solve(cp, max_epochs=1, iterations_per_epoch=5, **TOLS)
    assert not result["converged"]
    assert result["stop_reason"] == "max_epochs"


def test_invalid_update_mode():
    cp, _ = simplex_problem()
    with pytest.raises(ValueError):
        jc.solve(cp, update_mode="bogus")


@pytest.mark.parametrize("adaptive_eta", [False, None])
@pytest.mark.parametrize("step_size", [1.0, 0.5])
def test_alternating_fixed_step(adaptive_eta, step_size):
    # adaptive_eta off runs alternating on jo.gd(step_size)'s fixed step.
    cp, expected = simplex_problem()
    result = solve(
        cp, update_mode="alternating", adaptive_eta=adaptive_eta, step_size=step_size
    )
    assert result["converged"]
    np.testing.assert_allclose(result["solution"].primal, expected, atol=1e-5)


@pytest.mark.parametrize("update_mode", ["alternating", "extragradient", "pdhg"])
@pytest.mark.parametrize("step_size", [0.1, 3.0])
def test_step_size_seeds(update_mode, step_size):
    cp, expected = simplex_problem()
    result = solve(cp, update_mode=update_mode, step_size=step_size)
    assert result["converged"]
    np.testing.assert_allclose(result["solution"].primal, expected, atol=1e-5)


@pytest.mark.parametrize(
    "options",
    [
        {"adaptive_eta": 0.5},
        {"adaptive_eta": "auto"},
        {"step_size": -1.0},
        {"step_size": 0.0},
        {"step_size": "auto"},
    ],
)
def test_invalid_step_settings(options):
    cp, _ = simplex_problem()
    with pytest.raises(ValueError):
        jc.solve(cp, **options)


def test_optimiser_requires_fixed_step():
    # The adaptive step (the default) would bypass an explicit optimiser.
    cp, _ = simplex_problem()
    with pytest.raises(ValueError, match="adaptive_eta=False"):
        jc.solve(cp, optimiser=jo.gd(0.1))
    result = solve(cp, optimiser=jo.gd(0.1), adaptive_eta=False)
    assert result["converged"]


def test_pdhg_requires_adaptive_eta():
    cp, _ = simplex_problem()
    with pytest.raises(ValueError):
        jc.solve(cp, update_mode="pdhg", adaptive_eta=None)


# --- Traceable solves, batching and derivatives ------------------------------

import jax  # noqa: E402

TIGHT = dict(
    primal_grad_norm_tolerance=1e-11,
    primal_feasibility_tolerance=1e-11,
    complementarity_slack_tolerance=1e-11,
    iterations_per_epoch=500,
    max_epochs=3000,
)


def build_scaled_simplex(params):
    # Projection of a onto {x >= 0, sum x = s}: x* = s * P(a / s).
    a, s = params
    n = a.shape[0]
    return jc.JaddleCP(
        num_variables=n,
        objective=lambda x: jnp.sum((x - a) ** 2),
        constraints_eq=lambda x: jnp.array([jnp.sum(x) - s]),
        constraints_ineq=lambda x: jnp.zeros(0),
        lower_bounds=jnp.zeros(n),
        upper_bounds=jnp.full(n, jnp.inf),
    )


def exact_scaled_projection(a, s):
    return s * project_onto_simplex(np.asarray(a) / s)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"update_mode": "pdhg"},
        {"update_mode": "pdhg", "restarts": True},
        {"restarts": True, "average": True},
        {"update_mode": "extragradient", "restarts": True, "diag_adapt": True},
        {"gradient_scaling": True},
    ],
)
def test_make_solver_matches_solve(options):
    cp, _ = simplex_problem(n=30)
    kwargs = dict(TOLS, iterations_per_epoch=200, max_epochs=300, **options)
    reference = jc.solve(cp, **kwargs)
    # Called without an outer jit (which may round differently), the traced
    # solve is solve() exactly.
    result = jc.make_solver(lambda _: cp, **kwargs)(jnp.zeros(()))
    assert int(result.epochs) == reference["epochs"]
    for a, b in zip(result.solution, reference["solution"]):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_make_solver_vmap_batch():
    a = np.random.default_rng(1).standard_normal((6, 30))
    solve_fn = jc.make_solver(build_scaled_simplex, **TIGHT)
    result = jax.jit(jax.vmap(solve_fn))((jnp.asarray(a), jnp.ones(6)))
    assert np.asarray(result.converged).all()
    for i in range(6):
        np.testing.assert_allclose(
            result.solution.primal[i], exact_scaled_projection(a[i], 1.0), atol=1e-7
        )


def test_optimal_value_gradient():
    # z*(a, s) = ||x* - a||^2: dz/da = 2 (a - x*) and dz/ds = mu*, checked
    # against central differences of the exact projection.
    rng = np.random.default_rng(2)
    a, s = jnp.asarray(rng.standard_normal(10)), jnp.asarray(1.5)
    value_fn = jc.make_optimal_value(build_scaled_simplex, return_solution=True, **TIGHT)
    (z, result), (grad_a, grad_s) = jax.value_and_grad(value_fn, has_aux=True)((a, s))
    x = exact_scaled_projection(a, float(s))
    assert abs(float(z) - np.sum((x - np.asarray(a)) ** 2)) < 1e-8
    np.testing.assert_allclose(grad_a, 2 * (np.asarray(a) - x), atol=1e-7)

    def z_exact(s_):
        x_ = exact_scaled_projection(a, s_)
        return np.sum((x_ - np.asarray(a)) ** 2)

    h = 1e-6
    fd = (z_exact(float(s) + h) - z_exact(float(s) - h)) / (2 * h)
    assert abs(float(grad_s) - fd) < 1e-6


def test_optimal_value_gradient_through_bounds():
    # Upper bounds as a parameter: the active ones carry -nu_upper.
    rng = np.random.default_rng(3)
    n = 8

    def build(params):
        a, u = params
        return jc.JaddleCP(
            num_variables=n,
            objective=lambda x: jnp.sum((x - a) ** 2),
            constraints_eq=lambda x: jnp.array([jnp.sum(x) - 2.0]),
            constraints_ineq=lambda x: jnp.zeros(0),
            lower_bounds=jnp.zeros(n),
            upper_bounds=u,
        )

    a = jnp.asarray(rng.standard_normal(n) + 0.5)
    u = jnp.full(n, 0.45)
    value_fn = jax.jit(jc.make_optimal_value(build, **TIGHT))
    grad_u = jax.grad(value_fn)((a, u))[1]
    k = int(np.argmin(np.asarray(grad_u)))  # most active upper bound
    assert float(grad_u[k]) < 0
    h = 1e-5
    fd = (float(value_fn((a, u.at[k].add(h)))) - float(value_fn((a, u.at[k].add(-h))))) / (2 * h)
    assert abs(float(grad_u[k]) - fd) < 1e-6


def test_solution_vjp_matches_finite_differences():
    rng = np.random.default_rng(4)
    a, s = jnp.asarray(rng.standard_normal(10)), jnp.asarray(1.5)
    x_fn = jc.make_solution(build_scaled_simplex, **TIGHT)
    cot = rng.standard_normal(10)
    grad_a, grad_s = jax.grad(lambda p: jnp.dot(x_fn(p), cot))((a, s))
    h = 1e-6

    def fd(bump):
        return cot @ (bump(h) - bump(-h)) / (2 * h)

    for j in range(10):
        e = np.zeros(10)
        e[j] = 1.0
        expected = fd(lambda d: exact_scaled_projection(np.asarray(a) + d * e, float(s)))
        assert abs(float(grad_a[j]) - expected) < 1e-6
    expected = fd(lambda d: exact_scaled_projection(a, float(s) + d))
    assert abs(float(grad_s) - expected) < 1e-6


def test_solution_vjp_is_nan_when_undefined():
    # A linear objective over a face of optima: the KKT system is singular,
    # so dx*/dθ is undefined and the VJP says so with NaN.
    def build(c):
        return jc.JaddleCP(
            num_variables=2,
            objective=lambda x: c @ x,
            constraints_eq=lambda x: jnp.array([jnp.sum(x) - 1.0]),
            constraints_ineq=lambda x: jnp.zeros(0),
            lower_bounds=jnp.zeros(2),
            upper_bounds=jnp.full(2, jnp.inf),
        )

    x_fn = jc.make_solution(build, **dict(TIGHT, max_epochs=50))
    # sum(x) = 1 everywhere, so its derivative is genuinely 0 and the singular
    # system is consistent for that cotangent; x[0] depends on where in the
    # face the solution lies, which is undefined.
    grad_sum = jax.grad(lambda c: x_fn(c).sum())(jnp.array([1.0, 1.0]))
    np.testing.assert_allclose(grad_sum, 0.0, atol=1e-9)
    grad = jax.grad(lambda c: x_fn(c)[0])(jnp.array([1.0, 1.0]))
    assert np.all(np.isnan(np.asarray(grad)))


QP_TOLS = dict(
    primal_feasibility_tolerance=1e-7,
    dual_feasibility_tolerance=1e-7,
    dual_gap_tolerance=1e-7,
)


def simplex_qp(n=50, seed=0):
    # simplex_problem as matrices: sum (x - a)^2 = ½ xᵀ(2I)x - 2aᵀx + const.
    a = np.random.default_rng(seed).standard_normal(n)
    qp = dict(
        Q=2.0 * np.eye(n),
        c=-2.0 * a,
        A_eq=np.ones((1, n)),
        b_eq=np.ones(1),
        lower_bounds=np.zeros(n),
    )
    return qp, project_onto_simplex(a)


def solve_qp(qp, **kwargs):
    kwargs.setdefault("max_epochs", 300)
    return jc.quadratic_program(**qp, iterations_per_epoch=200, **QP_TOLS, **kwargs)


def assert_qp_certified(result, tol=2e-7):
    assert result["converged"]
    # The certificate is recomputed from the unscaled matrices.
    cert = result["certificate"]
    assert cert["relative_primal_feasibility_residual"] <= tol
    assert cert["relative_dual_feasibility_residual"] <= tol
    assert cert["relative_gap"] <= tol


@pytest.mark.parametrize("scaled", [True, False])
def test_quadratic_program(scaled):
    qp, expected = simplex_qp()
    options = {} if scaled else dict(ruiz_iterations=0, pc_iterations=0)
    result = solve_qp(qp, **options)
    assert_qp_certified(result)
    s = result["solution"]
    np.testing.assert_allclose(s.primal, expected, atol=1e-5)
    assert s.dual_eq.shape == (1,)
    assert s.dual_ineq.shape == (0,)
    objective = float(expected @ expected + qp["c"] @ expected)
    assert abs(result["objective"] - objective) <= 1e-5 * (1 + abs(objective))


def test_quadratic_program_pdhg():
    # The objective's curvature (Q) is only caught by the full-operator
    # Lipschitz test, not by the constraint coupling alone.
    qp, expected = simplex_qp()
    result = solve_qp(qp, update_mode="pdhg")
    assert_qp_certified(result)
    np.testing.assert_allclose(result["solution"].primal, expected, atol=1e-5)


@DIAG_MODES
def test_quadratic_program_diag_adapt(update_mode):
    qp, expected = simplex_qp()
    result = solve_qp(qp, update_mode=update_mode, restarts=True, diag_adapt=True)
    assert_qp_certified(result)
    np.testing.assert_allclose(result["solution"].primal, expected, atol=1e-5)


def test_quadratic_program_inequality_rows():
    # Box-constrained isotonic regression, with sparse inequality rows.
    import scipy.sparse as sp

    n = 60
    y = np.linspace(-1, 1, n) ** 3 + 0.15 * np.random.default_rng(0).standard_normal(n)
    difference = sp.diags([np.ones(n - 1), -np.ones(n - 1)], [0, 1], shape=(n - 1, n))
    result = solve_qp(
        dict(
            Q=2.0 * sp.identity(n),
            c=-2.0 * y,
            A_ineq=difference,
            b_ineq=np.zeros(n - 1),
            lower_bounds=-np.ones(n),
            upper_bounds=np.ones(n),
        )
    )
    assert_qp_certified(result)
    np.testing.assert_allclose(
        result["solution"].primal, isotonic_fit(y, -1.0, 1.0), atol=1e-4
    )


def random_qp(seed=0, n=12, m_eq=2, m_ineq=8):
    """A feasible dense QP with eq rows, ineq rows and bounds, minus its c and
    b (returned separately so they can be varied)."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    matrices = dict(
        Q=M @ M.T / n + 0.1 * np.eye(n),
        A_eq=rng.standard_normal((m_eq, n)),
        A_ineq=rng.standard_normal((m_ineq, n)),
        lower_bounds=-np.ones(n),
        upper_bounds=np.ones(n),
    )
    return matrices, rng


def random_qp_vectors(matrices, rng):
    # b is built around a point inside the box, so every draw is feasible.
    x0 = rng.uniform(-0.5, 0.5, matrices["Q"].shape[0])
    c = rng.standard_normal(x0.shape[0])
    b_eq = matrices["A_eq"] @ x0
    b_ineq = matrices["A_ineq"] @ x0 + rng.uniform(0.0, 0.5, matrices["A_ineq"].shape[0])
    return c, b_eq, b_ineq


QP_SOLVER_SETTINGS = dict(QP_TOLS, iterations_per_epoch=200, max_epochs=300)


def test_make_qp_solver_matches_quadratic_program():
    matrices, rng = random_qp()
    c, b_eq, b_ineq = random_qp_vectors(matrices, rng)
    reference = jc.quadratic_program(
        **matrices, c=c, b_eq=b_eq, b_ineq=b_ineq, **QP_SOLVER_SETTINGS
    )
    assert_qp_certified(reference)
    solve_fn = jc.make_qp_solver(**matrices, **QP_SOLVER_SETTINGS)
    result = solve_fn(c, b_eq, b_ineq)
    assert bool(result.converged)
    for a, b in zip(result.solution, reference["solution"]):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5)


def test_make_qp_solver_vmap_batch():
    matrices, rng = random_qp(seed=1)
    batch = [random_qp_vectors(matrices, rng) for _ in range(5)]
    c, b_eq, b_ineq = (jnp.asarray(np.stack(v)) for v in zip(*batch))
    solve_fn = jc.make_qp_solver(**matrices, **QP_SOLVER_SETTINGS)
    result = jax.jit(jax.vmap(solve_fn))(c, b_eq, b_ineq)
    assert np.asarray(result.converged).all()
    for i, (c_i, b_eq_i, b_ineq_i) in enumerate(batch):
        reference = jc.quadratic_program(
            **matrices, c=c_i, b_eq=b_eq_i, b_ineq=b_ineq_i, **QP_SOLVER_SETTINGS
        )
        np.testing.assert_allclose(
            result.solution.primal[i], reference["solution"].primal, atol=1e-5
        )


def test_make_qp_solver_warm_start():
    # Started at the solution of a nearby problem, the solve takes fewer epochs.
    matrices, rng = random_qp(seed=2)
    c, b_eq, b_ineq = random_qp_vectors(matrices, rng)
    solve_fn = jax.jit(jc.make_qp_solver(**matrices, **QP_SOLVER_SETTINGS))
    nearby = solve_fn(c + 1e-3, b_eq, b_ineq)
    cold, warm = solve_fn(c, b_eq, b_ineq), solve_fn(c, b_eq, b_ineq, nearby.solution)
    assert bool(cold.converged) and bool(warm.converged)
    assert int(warm.epochs) < int(cold.epochs)
    np.testing.assert_allclose(warm.solution.primal, cold.solution.primal, atol=1e-5)


def test_make_qp_solver_requires_rhs():
    matrices, _ = random_qp()
    solve_fn = jc.make_qp_solver(**matrices)
    with pytest.raises(ValueError, match="b_eq"):
        solve_fn(np.zeros(12), b_ineq=np.zeros(8))


def test_quadratic_program_is_scale_invariant():
    # The same problem in variables x = S z, with its row rescaled: unusable
    # as given, and the scaling recovers it, duals included.
    qp, expected = simplex_qp()
    reference = solve_qp(qp)
    rng = np.random.default_rng(1)
    s = 10.0 ** rng.uniform(-2, 2, expected.size)
    row = 1e3
    rescaled = dict(
        Q=qp["Q"] * np.outer(s, s),
        c=qp["c"] * s,
        A_eq=row * qp["A_eq"] * s,
        b_eq=row * qp["b_eq"],
        lower_bounds=qp["lower_bounds"] / s,
    )
    result = solve_qp(rescaled)
    assert_qp_certified(result)
    np.testing.assert_allclose(s * result["solution"].primal, expected, atol=1e-5)
    np.testing.assert_allclose(
        row * result["solution"].dual_eq, reference["solution"].dual_eq, atol=1e-4
    )
    unscaled = solve_qp(rescaled, ruiz_iterations=0, pc_iterations=0, max_epochs=20)
    assert not unscaled["converged"]


def test_quadratic_program_warm_start():
    qp, _ = simplex_qp()
    cold = solve_qp(qp)
    warm = solve_qp(
        qp, initial_solution=cold["solution"], initial_opt_state=cold["opt_state"]
    )
    assert_qp_certified(warm)
    assert warm["epochs"] == 1


def test_quadratic_program_dense_and_sparse_storage_agree(monkeypatch):
    # A full Q is held as a dense array, a sparse one as a BCOO; forcing the
    # full one through the BCOO path gives the same solution.
    import scipy.sparse as sp

    n = 40
    rng = np.random.default_rng(0)
    factor = rng.standard_normal((n, n))
    qp = dict(
        Q=factor @ factor.T / n + np.eye(n),
        c=rng.standard_normal(n),
        A_eq=np.ones((1, n)),
        b_eq=np.ones(1),
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
    )
    dtype = jo.jaddle_dtype()
    assert isinstance(jc._device_matrix(sp.csr_matrix(qp["Q"]), dtype), jax.Array)
    # A small matrix is dense whatever its density; a large sparse one is not.
    assert isinstance(jc._device_matrix(sp.identity(n, format="csr"), dtype), jax.Array)
    assert isinstance(jc._device_matrix(sp.identity(300, format="csr"), dtype), BCOO)
    assert isinstance(jc._device_matrix(sp.csr_matrix((0, n)), dtype), BCOO)

    dense = solve_qp(qp)
    monkeypatch.setattr(jc, "_DENSE_DENSITY", np.inf)
    monkeypatch.setattr(jc, "_DENSE_SIZE", 0)
    sparse = solve_qp(qp)
    assert_qp_certified(dense)
    assert_qp_certified(sparse)
    np.testing.assert_allclose(
        dense["solution"].primal, sparse["solution"].primal, atol=1e-5
    )
    np.testing.assert_allclose(
        dense["solution"].dual_eq, sparse["solution"].dual_eq, atol=1e-5
    )


def test_quadratic_program_rejects_asymmetric_Q():
    with pytest.raises(ValueError, match="symmetric"):
        jc.quadratic_program(Q=np.array([[1.0, 1.0], [0.0, 1.0]]), c=np.zeros(2))


def test_diverged_solve_does_not_certify():
    # Free columns without curvature (the epigraph form of sum((x - a)^2)):
    # the fixed-step alternating scheme diverges to NaN from k = 0.1 without
    # restarts, and a NaN residual must not pass the convergence test.
    a = np.array([0.3, -0.2])
    result = jc.quadratic_program(
        Q=np.diag([0.0, 0.0, 2.0, 2.0]),
        c=np.zeros(4),
        A_eq=np.array([[-1.0, 0, 1, 0], [0, -1, 0, 1], [1, 1, 0, 0]]),
        b_eq=np.array([-a[0], -a[1], 1.0]),
        ruiz_iterations=0,
        pc_iterations=0,
        max_epochs=300,
        iterations_per_epoch=5,
        update_mode="alternating",
        adaptive_eta=None,
        restarts=False,
        k_init=0.1,
    )
    assert not np.all(np.isfinite(result["solution"].primal))
    assert not result["converged"]
    assert result["stop_reason"] == "max_epochs"


@pytest.mark.parametrize("update_mode", ["extragradient", "pdhg", "alternating"])
@pytest.mark.parametrize("restarts", [True, False])
def test_no_averaging(update_mode, restarts):
    # average=False carries no average buffer; the solve still converges.
    cp, expected = simplex_problem()
    result = solve(
        cp, update_mode=update_mode, average=False, restarts=restarts
    )
    assert result["converged"]
    np.testing.assert_allclose(result["solution"].primal, expected, atol=1e-5)


def test_save_load_cp_solves_like_original(tmp_path):
    cp, expected = simplex_problem()
    path = tmp_path / "simplex.jcp"
    jc.save_cp(cp, path)
    loaded = jc.load_cp(path)
    result = solve(loaded)  # gradient_scaling on (the default)
    assert result["converged"]
    np.testing.assert_allclose(result["solution"].primal, expected, atol=1e-5)
    original = solve(cp)
    np.testing.assert_allclose(
        result["solution"].primal, original["solution"].primal, atol=1e-6
    )


def test_save_load_cp_values_and_gradients(tmp_path):
    # n=60, m=40 > probes=16: the scales come from random JVPs, which the
    # loaded CP cannot take itself.
    rng = np.random.default_rng(0)
    A = rng.standard_normal((40, 60))
    a = rng.standard_normal(60)
    cp = jc.JaddleCP(
        num_variables=60,
        objective=lambda x: jnp.sum((x - a) ** 2) + jnp.sum(jnp.sin(x)),
        constraints_eq=lambda x: jnp.array([jnp.sum(x) - 1.0]),
        constraints_ineq=lambda x: jnp.asarray(A) @ x - 1.0,
        lower_bounds=-jnp.ones(60),
        upper_bounds=jnp.ones(60),
        dual_bound=lambda d_ineq, d_eq: -jnp.sum(d_ineq**2) + jnp.sum(d_eq),
        residuals=lambda s: (
            jnp.max(jnp.abs(s.primal)),
            jnp.sum(s.dual_ineq),
            jnp.sum(s.dual_eq),
        ),
    )
    path = tmp_path / "cp.jcp"
    jc.save_cp(cp, path)
    loaded = jc.load_cp(path)

    x = jnp.asarray(rng.uniform(-1, 1, 60))
    state = jc.SaddleState(
        primal=x,
        dual_ineq=jnp.asarray(rng.uniform(0, 1, 40)),
        dual_eq=jnp.asarray(rng.standard_normal(1)),
    )
    for name in ("objective", "constraints_eq", "constraints_ineq"):
        np.testing.assert_allclose(
            getattr(loaded, name)(x), getattr(cp, name)(x), rtol=1e-6
        )

    def lagrangian(p, s):
        return (
            p.objective(s.primal)
            + s.dual_ineq @ p.constraints_ineq(s.primal)
            + s.dual_eq @ p.constraints_eq(s.primal)
        )

    for got, want in zip(
        jax.tree.leaves(jax.grad(lambda s: lagrangian(loaded, s))(state)),
        jax.tree.leaves(jax.grad(lambda s: lagrangian(cp, s))(state)),
    ):
        np.testing.assert_allclose(got, want, rtol=1e-6)
    np.testing.assert_allclose(
        loaded.dual_bound(state.dual_ineq, state.dual_eq),
        cp.dual_bound(state.dual_ineq, state.dual_eq),
        rtol=1e-6,
    )
    np.testing.assert_allclose(loaded.residuals(state), cp.residuals(state), rtol=1e-6)
    for got, want in zip(
        loaded.gradient_scales(x), jc._gradient_scales(cp, x, 16)
    ):
        np.testing.assert_allclose(got, want, rtol=1e-6)
    np.testing.assert_array_equal(loaded.lower_bounds, cp.lower_bounds)
    assert loaded.num_eq_constraints() == 1
    assert loaded.num_ineq_constraints() == 40


def test_load_cp_without_optional_parts(tmp_path):
    cp, _ = simplex_problem()
    jc.save_cp(cp, tmp_path / "cp.jcp")
    loaded = jc.load_cp(tmp_path / "cp.jcp")
    assert loaded.dual_bound is None
    assert loaded.residuals is None


def test_save_load_solver(tmp_path):
    a = np.random.default_rng(3).standard_normal(30)
    solve_fn = jc.make_solver(build_scaled_simplex, max_epochs=300)
    params = (jnp.asarray(a), jnp.asarray(1.5))
    jc.save_solver(solve_fn, tmp_path / "solver.jcp", params)
    loaded = jc.load_solver(tmp_path / "solver.jcp")
    result = loaded(params)
    expected = solve_fn(params)
    assert bool(result.converged)
    np.testing.assert_allclose(
        result.solution.primal, expected.solution.primal, atol=1e-10
    )
    np.testing.assert_allclose(
        jax.jit(loaded)(params).solution.primal, expected.solution.primal, atol=1e-6
    )


def test_save_load_solver_batched(tmp_path):
    rng = np.random.default_rng(4)
    solve_fn = jc.make_solver(build_scaled_simplex, max_epochs=300)
    one = (jnp.zeros(30), jnp.asarray(1.0))
    jc.save_solver(solve_fn, tmp_path / "batched.jcp", one, batch=True)
    loaded = jc.load_solver(tmp_path / "batched.jcp")
    for size in (1, 4):  # any batch size from the one file
        batch = (jnp.asarray(rng.standard_normal((size, 30))), jnp.linspace(1.0, 2.0, size))
        result = loaded(batch)
        expected = jax.vmap(solve_fn)(batch)
        np.testing.assert_allclose(
            result.solution.primal, expected.solution.primal, atol=1e-10
        )
        np.testing.assert_array_equal(result.epochs, expected.epochs)


def test_save_load_differentiable_solver(tmp_path):
    a = jnp.asarray(np.random.default_rng(5).standard_normal(30))
    params = (a, jnp.asarray(1.5))
    value_fn = jc.make_optimal_value(build_scaled_simplex, **TIGHT)
    jc.save_solver(value_fn, tmp_path / "value.jcp", params, vjp_order=1)
    loaded = jc.load_solver(tmp_path / "value.jcp")
    np.testing.assert_allclose(loaded(params), value_fn(params), rtol=1e-8)
    for got, want in zip(jax.grad(loaded)(params), jax.grad(value_fn)(params)):
        np.testing.assert_allclose(got, want, atol=1e-8)

    x_fn = jc.make_solution(build_scaled_simplex, **TIGHT)
    jc.save_solver(x_fn, tmp_path / "x.jcp", params, batch=True, vjp_order=1)
    loaded_x = jc.load_solver(tmp_path / "x.jcp")
    batch = jax.tree.map(lambda p: jnp.stack([p, 1.1 * p]), params)
    loss = lambda f: lambda b: jnp.sum(f(b) ** 2)
    for got, want in zip(
        jax.grad(loss(loaded_x))(batch), jax.grad(loss(jax.vmap(x_fn)))(batch)
    ):
        np.testing.assert_allclose(got, want, atol=1e-8)


def test_load_checks_file_kind(tmp_path):
    cp, _ = simplex_problem()
    jc.save_cp(cp, tmp_path / "cp.jcp")
    with pytest.raises(ValueError, match="load_cp"):
        jc.load_solver(tmp_path / "cp.jcp")
