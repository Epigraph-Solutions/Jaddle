"""Second-order, PSD and exponential cones: the projections, the LP solver on
cone-constrained variables, and the CVXPY conic solver against Clarabel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

import jaddle.jaddle_linear as jl
from jaddle.cones import Cones, project_exp, smat, svec

TIGHT = dict(
    primal_feasibility_tolerance=1e-8,
    dual_feasibility_tolerance=1e-8,
    dual_gap_tolerance=1e-8,
)

CONES = Cones(soc=(3, 1, 5), psd=(3, 2, 3))


# %%
# Projections


def test_moreau_decomposition():
    # v = Π_K(v) - Π_K(-v), with the two parts orthogonal (K self-dual).
    v = np.random.default_rng(0).standard_normal(CONES.size)
    p = np.asarray(CONES.project(jnp.asarray(v)))
    q = np.asarray(CONES.project(jnp.asarray(-v)))
    np.testing.assert_allclose(p - q, v, atol=1e-12)
    assert abs(p @ q) < 1e-12
    np.testing.assert_allclose(CONES.project(jnp.asarray(p)), p, atol=1e-12)


def test_soc_projection_cases():
    K = Cones(soc=(3, 3, 3))
    inside, polar, between = [2.0, 1.0, 1.0], [-2.0, 1.0, 1.0], [0.0, 3.0, 4.0]
    p = np.asarray(K.project(jnp.asarray(inside + polar + between)))
    np.testing.assert_allclose(p[:3], inside)
    np.testing.assert_allclose(p[3:6], 0.0)
    np.testing.assert_allclose(p[6:], [2.5, 1.5, 2.0])


def test_psd_projection_matches_eigh():
    v = np.random.default_rng(1).standard_normal(CONES.size)
    p = np.asarray(CONES.project(jnp.asarray(v)))
    offset = CONES.soc_size
    for d in CONES.psd:
        length = d * (d + 1) // 2
        lam, U = np.linalg.eigh(smat(v[offset : offset + length]))
        expected = svec(U @ np.diag(np.maximum(lam, 0.0)) @ U.T)
        np.testing.assert_allclose(p[offset : offset + length], expected, atol=1e-12)
        offset += length


def test_svec_round_trip_and_inner_product():
    rng = np.random.default_rng(2)
    A, B = (rng.standard_normal((4, 4)) for _ in range(2))
    A, B = A + A.T, B + B.T
    np.testing.assert_allclose(smat(svec(A)), A)
    assert svec(A) @ svec(B) == pytest.approx(np.trace(A @ B))


def test_tie_is_uniform_per_block():
    d = jnp.exp(jnp.asarray(np.random.default_rng(3).standard_normal(CONES.size + 2)))
    tied = np.asarray(CONES.tie(d))
    np.testing.assert_array_equal(tied[:2], np.asarray(d)[:2])
    offset = 2
    for size in CONES.block_sizes():
        block = tied[offset : offset + size]
        assert np.ptp(block) < 1e-12
        offset += size


def _exp_reference(v):
    """Brute-force projection onto the exponential cone: the best of the
    closure's two pieces and the boundary point (ρy, y, y e^ρ), with ρ found
    by a grid search refined by a bounded scalar minimisation."""
    from scipy.optimize import minimize_scalar

    def boundary(rho):
        a = np.array([rho, 1.0, np.exp(rho)])
        y = max(a @ v, 0.0) / (a @ a)
        return y * a

    def dist(rho):
        return np.sum((boundary(rho) - v) ** 2)

    grid = np.linspace(-40.0, 40.0, 8001)
    k = int(np.argmin([dist(r) for r in grid]))
    rho = minimize_scalar(
        dist,
        bounds=(grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]),
        method="bounded",
        options={"xatol": 1e-14},
    ).x
    candidates = [np.array([min(v[0], 0.0), 0.0, max(v[2], 0.0)]), boundary(rho)]
    if v[1] > 0:
        z = max(v[2], v[1] * np.exp(min(v[0] / v[1], 700.0)))
        candidates.append(np.array([v[0], v[1], z]))
    return min(candidates, key=lambda c: np.sum((c - v) ** 2))


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e2])
def test_exp_projection_matches_brute_force(scale):
    V = np.random.default_rng(7).standard_normal((40, 3)) * scale
    P = np.asarray(project_exp(jnp.asarray(V)))
    for v, p in zip(V, P):
        ref = _exp_reference(v)
        # Never measurably farther from v than the reference, and close to it.
        excess = np.sum((p - v) ** 2) - np.sum((ref - v) ** 2)
        assert excess <= 1e-9 * np.sum(v**2)
        assert np.linalg.norm(p - ref) <= 1e-5 * np.linalg.norm(v)


def test_exp_projection_cases():
    inside = [0.0, 1.0, 2.0]  # 1·e^0 <= 2
    polar = [-1.0, -1.0, -1.0]  # x, y <= 0: projects to (x, 0, max(z, 0))
    corner = [-2.0, -1.0, 3.0]
    dual_side = [1.0, 0.0, -3.0]  # -v is in K* = {(u, v, w): -u e^(v/u) <= e w}
    V = jnp.asarray([inside, polar, corner, dual_side, [0.0, 0.0, 0.0]])
    P = np.asarray(project_exp(V))
    np.testing.assert_allclose(P[0], inside)
    np.testing.assert_allclose(P[1], [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(P[2], [-2.0, 0.0, 3.0])
    np.testing.assert_allclose(P[3], 0.0, atol=1e-12)
    np.testing.assert_allclose(P[4], 0.0)
    # Idempotent, and finite on extreme inputs.
    np.testing.assert_allclose(project_exp(jnp.asarray(P)), P, atol=1e-12)
    extreme = jnp.asarray([[1e8, 1.0, 1.0], [-1e8, 1e-8, 1.0], [1e-300, 1e-300, 0.0]])
    assert np.all(np.isfinite(np.asarray(project_exp(extreme))))


def test_exp_projection_warm_start():
    # A guessed root only changes where the search starts, never the answer:
    # good guesses (the roots of a nearby point), none (NaN) and bad ones.
    rng = np.random.default_rng(9)
    V = jnp.asarray(rng.standard_normal((500, 3)))
    cold = np.asarray(project_exp(V))
    _, roots = project_exp(V - 1e-4, jnp.full(500, jnp.nan))
    for guess in (roots, jnp.full(500, jnp.nan), jnp.asarray(rng.uniform(-50, 50, 500))):
        warm, new_roots = project_exp(V, guess)
        np.testing.assert_allclose(warm, cold, atol=1e-10)
        assert new_roots.shape == (500,)


def test_dual_residual():
    # Self-dual cones: r - Π_{K*}(r) is r - Π_K(r).
    v = jnp.asarray(np.random.default_rng(8).standard_normal(CONES.size))
    np.testing.assert_allclose(
        CONES.dual_residual(v), v - CONES.project(v), atol=1e-12
    )
    # Exponential cone: zero on K* and nonzero off it.
    K = Cones(exp=2)
    in_dual = jnp.asarray([-1.0, 0.0, 1.0, -1.0, 1.0, 1.0])  # e·w >= -u e^(v/u)
    np.testing.assert_allclose(K.dual_residual(in_dual), 0.0, atol=1e-12)
    out = jnp.asarray([1.0, 0.0, 0.0, -1.0, 0.0, -1.0])
    assert np.all(np.abs(np.asarray(K.dual_residual(out))).reshape(2, 3).sum(1) > 0.1)


def test_cone_variables_must_be_unbounded():
    n = 4
    with pytest.raises(ValueError, match="infinite bounds"):
        jl.LP(
            np.zeros(n),
            sp.csc_matrix((0, n)),
            np.zeros(0),
            sp.csc_matrix((0, n)),
            np.zeros(0),
            np.zeros(n),
            np.full(n, np.inf),
            cones=Cones(soc=(3,)),
        )


# %%
# The LP solver with cones


def unit_ball_lp(c, radius=1.0):
    """min cᵀx s.t. ‖x‖ <= radius, as [t, x] ∈ SOC with t = radius."""
    n = c.shape[0] + 1
    return jl.LP(
        jnp.concatenate([jnp.zeros(1), c]),
        sp.csc_matrix(np.eye(1, n)),
        jnp.reshape(radius, (1,)),
        sp.csc_matrix((0, n)),
        jnp.zeros(0),
        jnp.full(n, -jnp.inf),
        jnp.full(n, jnp.inf),
        cones=Cones(soc=(n,)),
    )


def test_soc_closed_form():
    c = np.random.default_rng(4).standard_normal(6)
    result = jl.solve(unit_ball_lp(c), **TIGHT)
    assert result["converged"]
    x = np.asarray(result["solution"].primal)[1:]
    np.testing.assert_allclose(x, -c / np.linalg.norm(c), atol=1e-6)


def test_min_eigenvalue_sdp_with_linear_variable():
    # min z + <C, X>  s.t.  tr X = 1,  X PSD,  0 <= z <= 2   ->  λ_min(C).
    rng = np.random.default_rng(5)
    d = 6
    B = rng.standard_normal((d, d))
    C = B + B.T
    L = d * (d + 1) // 2
    lp = jl.LP(
        np.r_[1.0, svec(C)],
        sp.csc_matrix(np.r_[0.0, svec(np.eye(d))][None, :]),
        np.ones(1),
        sp.csc_matrix((0, 1 + L)),
        np.zeros(0),
        np.r_[0.0, np.full(L, -np.inf)],
        np.r_[2.0, np.full(L, np.inf)],
        cones=Cones(psd=(d,)),
    )
    result = jl.solve(lp, **TIGHT)
    assert result["converged"]
    x = np.asarray(result["solution"].primal)
    assert x[0] == pytest.approx(0.0, abs=1e-8)
    assert svec(C) @ x[1:] == pytest.approx(np.linalg.eigvalsh(C)[0], abs=1e-6)


def test_exp_cone_closed_forms():
    # min z s.t. (1, 1, z) ∈ K_exp  ->  e.
    lp = jl.LP(
        np.array([0.0, 0.0, 1.0]),
        sp.csc_matrix(np.eye(2, 3)),
        np.ones(2),
        sp.csc_matrix((0, 3)),
        np.zeros(0),
        np.full(3, -np.inf),
        np.full(3, np.inf),
        cones=Cones(exp=1),
    )
    result = jl.solve(lp, **TIGHT)
    assert result["converged"]
    assert float(result["solution"].primal[2]) == pytest.approx(np.e, rel=1e-6)

    # max Σ -pᵢ log pᵢ s.t. Σ pᵢ = 1  ->  log n, with p = 1/n. With
    # (tᵢ, pᵢ, 1) ∈ K_exp, i.e. tᵢ <= -pᵢ log pᵢ, maximise Σ tᵢ.
    n = 6
    # Variables: p (n, boxed >= 0), then n exp blocks (tᵢ, pᵢ', 1ᵢ).
    blocks = 3 * n
    A_eq = np.zeros((1 + 2 * n, n + blocks))
    b_eq = np.zeros(1 + 2 * n)
    A_eq[0, :n], b_eq[0] = 1.0, 1.0
    for i in range(n):
        A_eq[1 + i, n + 3 * i + 1], A_eq[1 + i, i] = 1.0, -1.0  # pᵢ' = pᵢ
        A_eq[1 + n + i, n + 3 * i + 2], b_eq[1 + n + i] = 1.0, 1.0
    c = np.zeros(n + blocks)
    c[n::3] = -1.0
    lp = jl.LP(
        c,
        sp.csc_matrix(A_eq),
        b_eq,
        sp.csc_matrix((0, n + blocks)),
        np.zeros(0),
        np.r_[np.zeros(n), np.full(blocks, -np.inf)],
        np.full(n + blocks, np.inf),
        cones=Cones(exp=n),
    )
    result = jl.solve(lp, **TIGHT)
    assert result["converged"]
    x = np.asarray(result["solution"].primal)
    assert -c @ x == pytest.approx(np.log(n), rel=1e-6)
    np.testing.assert_allclose(x[:n], 1.0 / n, atol=1e-6)


def test_exp_cone_optimal_value_gradient():
    # z*(a) = min z s.t. (a, 1, z) ∈ K_exp = e^a, so dz*/da = e^a.
    def build(a):
        return jl.LP(
            jnp.array([0.0, 0.0, 1.0]),
            sp.csc_matrix(np.eye(2, 3)),
            jnp.stack([a, 1.0]),
            sp.csc_matrix((0, 3)),
            jnp.zeros(0),
            jnp.full(3, -jnp.inf),
            jnp.full(3, jnp.inf),
            cones=Cones(exp=1),
        )

    a = 0.7
    z, grad = jax.value_and_grad(jl.make_optimal_value(build, **TIGHT))(a)
    assert z == pytest.approx(np.exp(a), rel=1e-6)
    assert grad == pytest.approx(np.exp(a), rel=1e-5)


def test_scaling_keeps_cone_blocks_uniform():
    rng = np.random.default_rng(6)
    n_lin = 3
    n = n_lin + CONES.size
    A = sp.random(5, n, density=0.5, random_state=7, format="csc")
    lp = jl.LP(
        rng.standard_normal(n),
        A,
        np.ones(5),
        sp.csc_matrix((0, n)),
        np.zeros(0),
        np.r_[np.zeros(n_lin), np.full(CONES.size, -np.inf)],
        np.full(n, np.inf),
        cones=CONES,
    )
    _, _, col_scale, _ = jl.scale_problem(jl.to_jaddle_sparse(lp))
    col_scale = np.asarray(col_scale)
    offset = n_lin
    for size in CONES.block_sizes():
        assert np.ptp(col_scale[offset : offset + size]) == 0.0
        offset += size


def test_optimal_value_gradient_and_batch():
    # z*(c, r) = -r ‖c‖, so dz/dc = -r c / ‖c‖ and dz/dr = -‖c‖.
    def build(params):
        return unit_ball_lp(*params)

    c, r = jnp.array([1.0, -2.0, 0.5, 3.0]), 2.0
    z, (grad_c, grad_r) = jax.value_and_grad(jl.make_optimal_value(build, **TIGHT))(
        (c, r)
    )
    # A float, not a JAX array: pytest>=9.2's approx reads ``expected.flat``,
    # which JAX arrays raise on.
    norm = float(jnp.linalg.norm(c))
    assert z == pytest.approx(-r * norm, rel=1e-6)
    np.testing.assert_allclose(grad_c, -r * c / norm, atol=1e-6)
    assert grad_r == pytest.approx(-norm, rel=1e-6)

    cs, rs = jnp.stack([c, -c, 2 * c]), jnp.array([1.0, 2.0, 3.0])
    batch = jl.solve_batch(build, (cs, rs), **TIGHT)
    values = jnp.einsum("bi,bi->b", batch.solution.primal[:, 1:], cs)
    np.testing.assert_allclose(values, -rs * jnp.linalg.norm(cs, axis=1), rtol=1e-6)


def test_unsupported_routes_raise():
    lp = unit_ball_lp(jnp.ones(3))
    with pytest.raises(NotImplementedError):
        jl.solve_with_presolve(lp)
    with pytest.raises(NotImplementedError):
        jl.detect_infeasibility(lp)


# %%
# CVXPY conic solver, against Clarabel


def _cvxpy_problems():
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(0)

    n = 12
    mu = rng.normal(0.05, 0.05, n)
    F = rng.standard_normal((n, n)) / np.sqrt(n)
    w = cp.Variable(n)
    # The minimum achievable risk is 0.216, so the 0.3 limit binds.
    portfolio = [cp.sum(w) == 1, w >= 0, cp.norm(F.T @ w) <= 0.3]
    yield "portfolio", cp.Problem(cp.Maximize(mu @ w), portfolio), portfolio, [w]

    A, b = rng.standard_normal((20, 6)), rng.standard_normal(20)
    x = cp.Variable(6)
    ball = [cp.norm(x, 2) <= 0.5]
    yield "robust_ls", cp.Problem(cp.Minimize(cp.norm(A @ x - b)), ball), ball, [x]

    d = 5
    B = rng.standard_normal((d, d))
    X = cp.Variable((d, d), symmetric=True)
    trace_one = [X >> 0, cp.trace(X) == 1]
    yield (
        "min_eigenvalue",
        cp.Problem(cp.Minimize(cp.trace((B + B.T) @ X)), trace_one),
        trace_one,
        [X],
    )

    # Max-Cut's relaxation has many optimal X, so only the value and the duals
    # are compared (no variables).
    d = 6
    W = np.triu(rng.random((d, d)) < 0.5, 1).astype(float)
    W = W + W.T
    L = np.diag(W.sum(1)) - W
    X = cp.Variable((d, d), symmetric=True)
    unit_diag = [X >> 0, cp.diag(X) == 1]
    yield (
        "maxcut",
        cp.Problem(cp.Maximize(cp.trace(L @ X) / 4), unit_diag),
        unit_diag,
        [],
    )

    # Exponential cones.
    R = 1 + rng.normal(0.01, 0.1, (200, 8))
    w = cp.Variable(8, nonneg=True)
    budget = [cp.sum(w) == 1]
    yield "kelly", cp.Problem(cp.Maximize(cp.sum(cp.log(R @ w)) / 200), budget), budget, [w]

    X = rng.standard_normal((100, 5))
    y = np.sign(X @ rng.standard_normal(5) + 0.5 * rng.standard_normal(100))
    beta = cp.Variable(5)
    loss = cp.sum(cp.logistic(-cp.multiply(y, X @ beta))) / 100
    yield (
        "logistic",
        cp.Problem(cp.Minimize(loss + 0.1 * cp.sum_squares(beta))),
        [],
        [beta],
    )

    A = rng.standard_normal((30, 6))
    x = cp.Variable(6)
    ball = [cp.norm(x, 2) <= 1]  # second-order and exponential cones together
    yield "log_sum_exp", cp.Problem(cp.Minimize(cp.log_sum_exp(A @ x)), ball), ball, [x]

    V = rng.standard_normal((4, 12))
    lam = cp.Variable(12, nonneg=True)
    simplex = [cp.sum(lam) == 1]  # PSD and exponential cones together
    yield (
        "d_optimal",
        cp.Problem(cp.Maximize(cp.log_det(V @ cp.diag(lam) @ V.T)), simplex),
        simplex,
        [lam],
    )


@pytest.mark.parametrize(
    "name",
    [
        "portfolio",
        "robust_ls",
        "min_eigenvalue",
        "maxcut",
        "kelly",
        "logistic",
        "log_sum_exp",
        "d_optimal",
    ],
)
def test_cvxpy_conic_matches_clarabel(name):
    cp = pytest.importorskip("cvxpy")
    import jaddle.jaddle_cvxpy as jcv

    problem, constraints, variables = next(
        p[1:] for p in _cvxpy_problems() if p[0] == name
    )
    problem.solve(solver="CLARABEL")
    value = problem.value
    primals = [np.copy(v.value) for v in variables]
    duals = [np.copy(c.dual_value) for c in constraints]

    # The epoch budget turns a non-converging case into a failure, not a hang.
    problem.solve(solver=jcv.JaddleConicSolver(), tol=1e-8, max_epochs=500)
    assert problem.status == cp.OPTIMAL
    assert problem.value == pytest.approx(value, abs=1e-6 * (1 + abs(value)))
    for v, x in zip(variables, primals):
        np.testing.assert_allclose(v.value, x, atol=1e-4)
    for c, y in zip(constraints, duals):
        np.testing.assert_allclose(c.dual_value, y, atol=1e-4)


def _substitution_cases():
    """(name, problem, constraints, variables, variables put on the cone)."""
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(3)
    d = 4
    B = rng.standard_normal((d, d))
    C = B + B.T
    L = d * (d + 1) // 2

    # PSD rows with b != 0: X - I in PSD.
    X = cp.Variable((d, d), symmetric=True)
    cons = [X - np.eye(d) >> 0, cp.trace(X) == d + 1]
    yield "shifted_psd", cp.Problem(cp.Minimize(cp.trace(C @ X)), cons), cons, [X], L

    # A PSD block on a variable next to a second-order cone on an expression.
    X = cp.Variable((d, d), symmetric=True)
    v = rng.standard_normal(d)
    cons = [X >> 0, cp.trace(X) == 1, cp.norm(X @ v) <= 0.5]
    yield "psd_and_soc", cp.Problem(cp.Minimize(cp.trace(C @ X)), cons), cons, [X], L

    # The same variable in two PSD blocks: only the first is substituted.
    X = cp.Variable((d, d), symmetric=True)
    cons = [X >> 0, X - 0.1 * np.eye(d) >> 0, cp.trace(X) == 1]
    yield "two_psd", cp.Problem(cp.Minimize(cp.trace(C @ X)), cons), cons, [X], L

    # A cone on bounded variables keeps its slack form.
    t = cp.Variable(nonneg=True)
    x = cp.Variable(3)
    cons = [cp.SOC(t, x), cp.sum(x) == 1]
    yield "bounded_soc", cp.Problem(cp.Minimize(t), cons), cons, [t, x], 0


def _flat(dual):
    # An SOC constraint's dual is a list (t part, x part).
    if isinstance(dual, (list, tuple)):
        return np.concatenate([np.ravel(d) for d in dual])
    return np.ravel(np.copy(dual))


@pytest.mark.parametrize("name", ["shifted_psd", "psd_and_soc", "two_psd", "bounded_soc"])
def test_cvxpy_cone_substitution(name):
    cp = pytest.importorskip("cvxpy")
    import jaddle.jaddle_cvxpy as jcv

    problem, constraints, variables, substituted = next(
        p[1:] for p in _substitution_cases() if p[0] == name
    )
    problem.solve(solver="CLARABEL")
    value = problem.value
    primals = [np.copy(v.value) for v in variables]
    duals = [_flat(c.dual_value) for c in constraints]

    problem.solve(solver=jcv.JaddleConicSolver(), tol=1e-8, max_epochs=500)
    assert problem.status == cp.OPTIMAL
    assert problem.solver_stats.extra_stats["cone_substitution"] == substituted
    assert problem.value == pytest.approx(value, abs=1e-6 * (1 + abs(value)))
    for v, x in zip(variables, primals):
        np.testing.assert_allclose(v.value, x, atol=1e-4)
    for c, y in zip(constraints, duals):
        np.testing.assert_allclose(_flat(c.dual_value), y, atol=1e-4)


def test_cvxpy_conic_warm_start():
    # A fresh solver instance per solve must still warm-start: CVXPY clears
    # its cache whenever the solver it is given changes.
    cp = pytest.importorskip("cvxpy")
    import jaddle.jaddle_cvxpy as jcv

    problem = next(p[1] for p in _cvxpy_problems() if p[0] == "portfolio")
    problem.solve(solver=jcv.JaddleConicSolver(), tol=1e-8, max_epochs=500)
    cold = problem.solver_stats.num_iters
    problem.solve(solver=jcv.JaddleConicSolver(), tol=1e-8, max_epochs=500)
    assert problem.status == cp.OPTIMAL
    assert problem.solver_stats.num_iters < cold
