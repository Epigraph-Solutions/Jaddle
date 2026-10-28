"""Tests for the LP solver, jaddle.jaddle_linear."""

import threading

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import linprog

import jaddle.jaddle_linear as jl

TOL = 1e-6


def toy_lp():
    # minimise 3 x1 + 2 x2  s.t.  x1 + x2 >= 4,  x >= 0.  Optimum (0, 4), obj 8.
    return jl.LP(
        c=np.array([3.0, 2.0]),
        A_eq=sp.csc_matrix((0, 2)),
        b_eq=np.zeros(0),
        A_ineq=sp.csc_matrix([[-1.0, -1.0]]),
        b_ineq=np.array([-4.0]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.full(2, np.inf),
    )


def random_lp(seed=0, m_eq=5, m_ineq=15, n=30):
    """A feasible, bounded LP with equality and inequality rows and a mix of
    boxed and half-bounded variables."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.1, 0.9, n)
    A_eq = rng.standard_normal((m_eq, n))
    A_ineq = rng.standard_normal((m_ineq, n))
    upper = np.ones(n)
    upper[: n // 3] = np.inf
    return jl.LP(
        c=rng.standard_normal(n),
        A_eq=sp.csc_matrix(A_eq),
        b_eq=A_eq @ x0,
        A_ineq=sp.csc_matrix(A_ineq),
        b_ineq=A_ineq @ x0 + rng.uniform(0.1, 1.0, m_ineq),
        lower_bounds=np.zeros(n),
        upper_bounds=upper,
    )


def reference_objective(lp):
    bounds = [
        (lo, None if np.isinf(hi) else hi)
        for lo, hi in zip(lp.lower_bounds, lp.upper_bounds)
    ]
    res = linprog(
        lp.c,
        A_ub=lp.A_ineq,
        b_ub=lp.b_ineq,
        A_eq=lp.A_eq if lp.A_eq.shape[0] else None,
        b_eq=lp.b_eq if lp.A_eq.shape[0] else None,
        bounds=bounds,
        method="highs",
    )
    assert res.status == 0
    return res.fun


def solve(lp, tol=TOL, **kwargs):
    return jl.solve(
        lp,
        primal_feasibility_tolerance=tol,
        dual_feasibility_tolerance=tol,
        dual_gap_tolerance=tol,
        max_epochs=kwargs.pop("max_epochs", 500),
        **kwargs,
    )


def assert_certified(lp, result, ref_obj, tol=TOL):
    assert result["converged"]
    assert result["stop_reason"] == "certificate"
    assert_certifies(lp, result["solution"], ref_obj, tol)


def assert_certifies(lp, sol, ref_obj, tol=TOL):
    obj = float(lp.objective(sol.primal))
    assert abs(obj - ref_obj) / (1 + abs(ref_obj)) < 10 * tol
    # Independently re-check the certificate in the original problem's units.
    cert = jl.evaluate_lp_certificate(
        jl.to_jaddle_sparse(lp), sol.primal, sol.dual_eq, sol.dual_ineq
    )
    assert float(cert["relative_gap"]) < 10 * tol
    assert float(cert["relative_primal_feasibility_residual"]) < 10 * tol
    assert float(cert["relative_dual_feasibility_residual"]) < 10 * tol


def test_toy_lp():
    lp = toy_lp()
    result = solve(lp, tol=1e-4)
    assert result["stop_reason"] == "certificate"
    np.testing.assert_allclose(result["solution"].primal, [0.0, 4.0], atol=1e-3)


@pytest.mark.parametrize("update_mode", ["alternating", "pdhg", "halpern"])
def test_random_lp_matches_reference(update_mode):
    lp = random_lp()
    assert_certified(lp, solve(lp, update_mode=update_mode), reference_objective(lp))


@pytest.mark.parametrize("seed", [1, 2])
def test_other_random_lps(seed):
    lp = random_lp(seed=seed, m_eq=8, m_ineq=25, n=50)
    assert_certified(lp, solve(lp), reference_objective(lp))


@pytest.mark.parametrize("update_mode", ["alternating", "pdhg", "halpern"])
def test_diag_adapt_certifies(update_mode, capsys):
    lp = random_lp()
    result = solve(lp, update_mode=update_mode, diag_adapt=True, verbose=True)
    assert_certified(lp, result, reference_objective(lp))
    if "Restart" in capsys.readouterr().out:
        assert result["epochs"] > 0


def test_diag_adapt_frozen_matches_off():
    # diag_theta=0 keeps t = s = 1, so the run is the plain solve.
    lp = random_lp(seed=1, m_eq=8, m_ineq=25, n=50)
    off = solve(lp, diag_adapt=False)
    frozen = solve(lp, diag_adapt=True, diag_theta=0.0)
    assert frozen["stop_reason"] == off["stop_reason"] == "certificate"
    assert abs(frozen["epochs"] - off["epochs"]) <= 1
    assert_certifies(lp, frozen["solution"], reference_objective(lp))


@pytest.mark.parametrize("step_size", [0.3, 1.0, "auto"])
def test_step_size_seeds(step_size):
    lp = random_lp()
    result = solve(lp, step_size=step_size)
    assert_certified(lp, result, reference_objective(lp))


@pytest.mark.parametrize("update_mode", ["alternating", "pdhg", "halpern"])
@pytest.mark.parametrize("step_size", [0.3, "auto"])
def test_fixed_step(update_mode, step_size):
    # adaptive_eta=False holds eta at step_size: every step is accepted and
    # the returned eta is the starting one.
    lp = random_lp()
    result = solve(
        lp,
        update_mode=update_mode,
        adaptive_eta=False,
        step_size=step_size,
        diag_adapt=False,
        max_epochs=2000,
    )
    assert_certified(lp, result, reference_objective(lp))
    if step_size != "auto":
        assert float(result["opt_state"][1]) == step_size


def test_fixed_step_none():
    result = solve(toy_lp(), tol=1e-4, adaptive_eta=None, step_size="auto")
    assert result["stop_reason"] == "certificate"


@pytest.mark.parametrize(
    "options",
    [
        {"adaptive_eta": 0.3},
        {"adaptive_eta": "auto"},
        {"step_size": -1.0},
        {"step_size": 0.0},
        {"step_size": "bogus"},
    ],
)
def test_invalid_step_settings(options):
    with pytest.raises(ValueError):
        jl.solve(toy_lp(), **options)


def test_invalid_update_mode():
    with pytest.raises(ValueError):
        jl.solve(toy_lp(), update_mode="synchronous")


def test_invalid_gap_test():
    with pytest.raises(ValueError):
        jl.solve(toy_lp(), gap_test="signed")


def test_abs_gap_test():
    # The no-cancellation gap is never smaller than the signed PDLP one, so
    # the point passes both gaps and still matches the reference.
    lp = random_lp()
    result = solve(lp, gap_test="abs")
    assert result["converged"] and result["stop_reason"] == "certificate"
    cert = solve_certificate(lp, result["solution"])
    assert cert["relative_primal_feasibility_residual"] <= TOL
    assert cert["relative_dual_feasibility_residual"] <= TOL
    assert cert["relative_gap_abs"] <= TOL
    assert cert["relative_gap"] <= cert["relative_gap_abs"]
    ref = reference_objective(lp)
    assert abs(float(lp.objective(result["solution"].primal)) - ref) <= 1e-4 * (
        1 + abs(ref)
    )


def test_unscaled_solve():
    lp = random_lp()
    assert_certified(lp, solve(lp, scale=False), reference_objective(lp))


def test_accepts_jaddle_lp():
    lp = random_lp()
    result = solve(jl.to_jaddle_sparse(lp))
    assert_certified(lp, result, reference_objective(lp))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
def test_spmv(dtype):
    from jaddle.jaddle_basic_types import scipy_to_bcoo, spmv

    rng = np.random.default_rng(0)
    S = sp.random(60, 40, density=0.2, random_state=1, format="csr")
    A = scipy_to_bcoo(S, dtype)
    x = jnp.asarray(rng.standard_normal(40), dtype)
    y = spmv(A, x)
    assert y.dtype == dtype
    expected = S @ np.asarray(x, np.float64)
    tol = 1e-5 if dtype == jnp.float32 else 2e-2
    np.testing.assert_allclose(np.asarray(y, np.float64), expected, rtol=tol, atol=tol)
    # Batched and vmapped inputs give the same result as one at a time.
    X = jnp.stack([x, 2 * x], axis=1)
    np.testing.assert_array_equal(spmv(A, X)[:, 0], y)
    np.testing.assert_array_equal(jax.vmap(lambda v: spmv(A, v))(X.T)[0], y)
    if dtype == jnp.float32:
        np.testing.assert_array_equal(y, A @ x)


def test_epoch_budget():
    result = solve(random_lp(), tol=1e-12, max_epochs=1)
    assert not result["converged"]
    assert result["stop_reason"] == "max_epochs"
    assert result["epochs"] == 1


def test_time_budget():
    # The budget is spent before the first epoch boundary, where it is checked.
    result = solve(random_lp(), tol=1e-12, max_epochs=None, max_seconds=1e-3)
    assert not result["converged"]
    assert result["stop_reason"] == "time_limit"


def test_same_shape_solve_reuses_compiled_programs(monkeypatch):
    # _device_solver only runs while solve() traces its device programs. A
    # second LP of the same shapes, with different numbers, must reuse them.
    traces = []
    device_solver = jl._device_solver

    def counting(*args, **kwargs):
        traces.append(1)
        return device_solver(*args, **kwargs)

    monkeypatch.setattr(jl, "_device_solver", counting)
    jl._solve_programs.cache_clear()
    first, second = random_lp(seed=0), random_lp(seed=1)
    assert_certified(first, solve(first), reference_objective(first))
    n_traces = len(traces)
    assert n_traces > 0
    assert_certified(second, solve(second), reference_objective(second))
    assert len(traces) == n_traces


def test_warm_start():
    lp = random_lp()
    loose = solve(lp, tol=1e-3)
    assert loose["stop_reason"] == "certificate"
    tight = solve(
        lp,
        initial_solution=loose["solution"],
        initial_opt_state=loose["opt_state"],
    )
    assert_certified(lp, tight, reference_objective(lp))


def solve_certificate(lp, sol):
    """The certificate solve() terminates on (l2 norm, PDLP dual residual)."""
    # solve() gives an empty constraint class one zero row, and its duals.
    padded = getattr(jl, "__pad_empty_blocks")(jl.to_jaddle_sparse(lp))
    cert = jl.evaluate_lp_certificate(
        padded,
        sol.primal,
        sol.dual_eq,
        sol.dual_ineq,
        norm="l2",
        dual_residual="pdlp",
    )
    return {k: float(v) for k, v in cert.items()}


def test_certificate_matches_solve():
    lp = random_lp()
    cert = solve_certificate(lp, solve(lp)["solution"])
    assert cert["relative_primal_feasibility_residual"] <= TOL
    assert cert["relative_dual_feasibility_residual"] <= TOL
    assert cert["relative_gap"] <= TOL


def test_warm_start_from_optimum_stays_put():
    # A certified point must certify again after one epoch from it. The duals
    # were once warm-started off by the objective scale c_max.
    lp = random_lp()
    sol = solve(lp)["solution"]
    again = solve(lp, initial_solution=sol, max_epochs=1, iterations_per_epoch=8)
    assert again["stop_reason"] == "certificate"
    y = np.concatenate([sol.dual_eq, sol.dual_ineq])
    y_again = np.concatenate([again["solution"].dual_eq, again["solution"].dual_ineq])
    assert np.linalg.norm(y_again - y) <= 1e-3 * (1 + np.linalg.norm(y))


def perturbed_optimum(seed=0, size=1e-3):
    lp = random_lp()
    sol = solve(lp)["solution"]
    rng = np.random.default_rng(seed)

    def noise(v):
        return v + size * rng.standard_normal(v.shape)

    return lp, sol, noise


def test_primal_polish_restores_feasibility():
    lp, sol, noise = perturbed_optimum()
    bad = jl.SaddleState(
        primal=noise(sol.primal), dual_ineq=sol.dual_ineq, dual_eq=sol.dual_eq
    )
    before = solve_certificate(lp, bad)["relative_primal_feasibility_residual"]
    x = jl.primal_polish(jl.to_jaddle_sparse(lp), bad)
    polished = jl.SaddleState(primal=x, dual_ineq=sol.dual_ineq, dual_eq=sol.dual_eq)
    after = solve_certificate(lp, polished)["relative_primal_feasibility_residual"]
    assert after < 1e-2 * before


def test_primal_polish_objective_bound():
    lp, sol, noise = perturbed_optimum()
    bad = jl.SaddleState(
        primal=noise(sol.primal), dual_ineq=sol.dual_ineq, dual_eq=sol.dual_eq
    )
    before = solve_certificate(lp, bad)["relative_primal_feasibility_residual"]
    objective = float(np.asarray(lp.c) @ np.asarray(sol.primal))
    bound = objective + 1e-3 * (1 + abs(objective))
    x = jl.primal_polish(
        jl.to_jaddle_sparse(lp), bad, objective_bound=bound, max_epochs=4
    )
    polished = jl.SaddleState(primal=x, dual_ineq=sol.dual_ineq, dual_eq=sol.dual_eq)
    after = solve_certificate(lp, polished)["relative_primal_feasibility_residual"]
    assert after < 1e-2 * before
    assert float(np.asarray(lp.c) @ np.asarray(x)) <= bound + 1e-6 * (1 + abs(bound))


def test_dual_polish_restores_feasibility():
    lp, sol, noise = perturbed_optimum()
    bad = jl.SaddleState(
        primal=sol.primal, dual_ineq=noise(sol.dual_ineq), dual_eq=noise(sol.dual_eq)
    )
    before = solve_certificate(lp, bad)["relative_dual_feasibility_residual"]
    dual_eq, dual_ineq = jl.dual_polish(jl.to_jaddle_sparse(lp), bad)
    polished = jl.SaddleState(primal=sol.primal, dual_ineq=dual_ineq, dual_eq=dual_eq)
    after = solve_certificate(lp, polished)["relative_dual_feasibility_residual"]
    assert after < 1e-2 * before
    # A feasibility projection, not a jump across the dual feasible set.
    y = np.concatenate([sol.dual_eq, sol.dual_ineq])
    moved = np.linalg.norm(np.concatenate([dual_eq, dual_ineq]) - y)
    perturbation = np.linalg.norm(np.concatenate([bad.dual_eq, bad.dual_ineq]) - y)
    assert moved <= perturbation


@pytest.mark.parametrize("polish_objective_slack", [None, 0.5])
@pytest.mark.parametrize("polish_trigger", ["auto", "doubling"])
@pytest.mark.parametrize("make_lp", [toy_lp, random_lp])
def test_solve_with_polishing(make_lp, polish_trigger, polish_objective_slack):
    # Tight feasibility, loose gap; polish from the first epoch on.
    lp = make_lp()
    result = jl.solve_with_polishing(
        lp,
        tol=1e-9,
        dual_gap_tolerance=1e-4,
        first_polish_epoch=1,
        polish_trigger=polish_trigger,
        polish_objective_slack=polish_objective_slack,
        iterations_per_epoch=16,
        max_epochs=2000,
    )
    assert result["stop_reason"] == "certificate"
    assert set(result["polish"]) == {"attempts", "epochs", "polished"}
    cert = solve_certificate(lp, result["solution"])
    assert cert["relative_primal_feasibility_residual"] <= 1e-9
    assert cert["relative_dual_feasibility_residual"] <= 1e-9
    assert cert["relative_gap"] <= 1e-4
    ref = reference_objective(lp)
    assert abs(cert["objective"] - ref) / (1 + abs(ref)) < 1e-3


@pytest.mark.parametrize("fix_threshold", [None, 100.0])
@pytest.mark.parametrize("make_lp", [toy_lp, random_lp])
def test_solve_with_refinement(make_lp, fix_threshold):
    # A loose first solve, then correction rounds to a tight certificate.
    lp = make_lp()
    result = jl.solve_with_refinement(
        lp,
        tol=1e-10,
        inner_tol=1e-3,
        fix_threshold=fix_threshold,
        iterations_per_epoch=16,
        max_epochs=4000,
    )
    assert result["stop_reason"] == "certificate"
    assert result["converged"]
    refinement = result["refinement"]
    assert refinement["rounds"] >= 1
    assert len(refinement["history"]) == refinement["rounds"] + 1
    assert result["epochs"] >= sum(refinement["epochs_per_round"])
    cert = solve_certificate(lp, result["solution"])
    assert cert["relative_primal_feasibility_residual"] <= 1e-10
    assert cert["relative_dual_feasibility_residual"] <= 1e-10
    assert cert["relative_gap"] <= 1e-10
    ref = reference_objective(lp)
    assert abs(cert["objective"] - ref) / (1 + abs(ref)) < 1e-8


def test_refinement_correction_round_trip():
    # A correction of scale * (target - current) maps back onto the target.
    lp, sol, noise = perturbed_optimum()
    jlp = getattr(jl, "__pad_empty_blocks")(jl.to_jaddle_sparse(lp))
    x = jnp.clip(noise(sol.primal), jlp.lower_bounds, jlp.upper_bounds)
    y_eq, y_ineq = noise(sol.dual_eq), jnp.abs(noise(sol.dual_ineq))
    scale_p, scale_d = 2.0**10, 2.0**20
    n_ineq = y_ineq.shape[0]
    correction = jl.SaddleState(
        primal=jnp.concatenate([scale_p * (sol.primal - x), jnp.zeros(n_ineq)]),
        dual_ineq=jnp.zeros(1),
        dual_eq=scale_d
        * jnp.concatenate([sol.dual_eq - y_eq, sol.dual_ineq - y_ineq]),
    )
    back = jl._apply_correction(jlp, x, y_eq, y_ineq, correction, scale_p, scale_d)
    np.testing.assert_allclose(back.primal, sol.primal, atol=1e-12)
    np.testing.assert_allclose(back.dual_eq, sol.dual_eq, atol=1e-12)
    np.testing.assert_allclose(back.dual_ineq, sol.dual_ineq, atol=1e-12)


def test_refinement_correction_lp_recovers_optimum():
    # Solving one correction LP tightly from a perturbed point lands on the
    # optimum: the shifted, scaled problem is exact.
    lp, sol, noise = perturbed_optimum()
    jlp = getattr(jl, "__pad_empty_blocks")(jl.to_jaddle_sparse(lp))
    x = jnp.clip(noise(sol.primal), jlp.lower_bounds, jlp.upper_bounds)
    y_eq, y_ineq = noise(sol.dual_eq), jnp.maximum(noise(sol.dual_ineq), 0.0)
    bad = jl.SaddleState(primal=x, dual_ineq=y_ineq, dual_eq=y_eq)
    before = solve_certificate(lp, bad)
    scale_p = scale_d = 2.0**8
    correction_lp, x = jl._correction_lp(
        jl._slack_form_base(jlp), jlp, x, y_eq, y_ineq, scale_p, scale_d
    )
    correction = solve(correction_lp, tol=1e-9, max_epochs=2000)
    assert correction["stop_reason"] == "certificate"
    fixed = jl._apply_correction(
        jlp, x, y_eq, y_ineq, correction["solution"], scale_p, scale_d
    )
    after = solve_certificate(lp, fixed)
    for key in (
        "relative_primal_feasibility_residual",
        "relative_dual_feasibility_residual",
    ):
        assert after[key] < 1e-2 * before[key]
    ref = reference_objective(lp)
    assert abs(after["objective"] - ref) / (1 + abs(ref)) < 1e-6


def test_solve_with_refinement_epoch_budget():
    result = jl.solve_with_refinement(random_lp(), tol=1e-12, max_epochs=3)
    assert result["stop_reason"] == "max_epochs"
    assert result["epochs"] == 3
    assert not result["converged"]


def small_lp(c, A_ineq=None, b_ineq=None, A_eq=None, b_eq=None, ub=None):
    n = len(c)

    def block(A):
        return sp.csc_matrix(np.array(A, float)) if A is not None else sp.csc_matrix((0, n))

    def rhs(b):
        return np.array(b, float) if b is not None else np.zeros(0)

    return jl.LP(
        c=np.array(c, float),
        A_eq=block(A_eq),
        b_eq=rhs(b_eq),
        A_ineq=block(A_ineq),
        b_ineq=rhs(b_ineq),
        lower_bounds=np.zeros(n),
        upper_bounds=np.full(n, np.inf) if ub is None else np.array(ub, float),
    )


def contradictory_random_lp():
    # A random feasible LP plus the pair a.x <= 0 and a.x >= 1.
    lp = random_lp(seed=4)
    a = sp.csc_matrix(np.random.default_rng(3).standard_normal(lp.c.shape[0]))
    return jl.LP(
        c=lp.c,
        A_eq=lp.A_eq,
        b_eq=lp.b_eq,
        A_ineq=sp.vstack([lp.A_ineq, a, -a]).tocsc(),
        b_ineq=np.concatenate([lp.b_ineq, [0.0, -1.0]]),
        lower_bounds=lp.lower_bounds,
        upper_bounds=lp.upper_bounds,
    )


@pytest.mark.parametrize(
    "lp, status",
    [
        (small_lp([1, 1], [[1, 1], [-1, -1]], [1, -3]), "primal_infeasible"),
        (small_lp([1], [[-1]], [-2], ub=[1]), "primal_infeasible"),
        (small_lp([1, 1], A_eq=[[1, 1], [1, 1]], b_eq=[1, 2]), "primal_infeasible"),
        (contradictory_random_lp(), "primal_infeasible"),
        (small_lp([-1, 0], [[1, -1]], [1]), "dual_infeasible"),
        (random_lp(), "optimal"),
        (toy_lp(), "optimal"),
    ],
)
def test_detect_infeasibility(lp, status):
    result = jl.detect_infeasibility(lp, max_epochs=2000)
    assert result["status"] == status
    if status == "optimal":
        return
    # The returned rays certify in their own right.
    cert = jl.evaluate_infeasibility_certificate(
        getattr(jl, "__pad_empty_blocks")(jl.to_jaddle_sparse(lp)),
        result["primal_ray"],
        result["dual_ray_eq"],
        result["dual_ray_ineq"],
    )
    key = (
        "primal_infeasibility_ratio"
        if status == "primal_infeasible"
        else "dual_infeasibility_ratio"
    )
    assert float(cert[key]) <= 1e-8


@pytest.mark.parametrize("seed", range(6))
def test_detect_infeasibility_no_false_positive(seed):
    lp = random_lp(seed=seed, m_eq=seed % 4, m_ineq=10 + seed, n=20 + 2 * seed)
    result = jl.detect_infeasibility(lp, max_epochs=300, first_check_epoch=1)
    assert result["status"] in ("optimal", "undetermined")


def test_infeasibility_certificate_ignores_rounding():
    # A zero dual ray must not certify, whatever the rounding in its objective.
    lp = jl.to_jaddle_sparse(random_lp())
    y_eq = np.zeros(lp.n_eq)
    y_ineq = np.full(lp.A_ineq.shape[0], 1e-300)
    cert = jl.evaluate_infeasibility_certificate(
        lp, np.zeros(lp.c.shape[0]), y_eq, y_ineq
    )
    assert float(cert["primal_infeasibility_ratio"]) == np.inf
    assert float(cert["dual_infeasibility_ratio"]) == np.inf


@pytest.mark.parametrize("polish_trigger", ["auto", "doubling"])
def test_solve_with_polishing_epoch_budget(polish_trigger):
    result = jl.solve_with_polishing(
        random_lp(),
        tol=1e-12,
        first_polish_epoch=1,
        polish_trigger=polish_trigger,
        max_epochs=3,
    )
    assert result["stop_reason"] == "max_epochs"
    assert result["epochs"] == 3


def test_solve_polish_trigger_stops_once_gap_closes():
    # Loose gap, unreachable feasibility: the trigger stops the solve at the
    # first allowed epoch whose gap test passes, unconverged.
    lp = random_lp()
    kwargs = dict(
        primal_feasibility_tolerance=1e-14,
        dual_feasibility_tolerance=1e-14,
        dual_gap_tolerance=1e-2,
        iterations_per_epoch=16,
        max_epochs=500,
    )
    result = jl.solve(lp, _polish_trigger=(3, None), **kwargs)
    assert result["stop_reason"] == "polish_ready"
    assert not result["converged"]
    assert 3 <= result["epochs"] < 500
    cert = solve_certificate(lp, result["solution"])
    assert cert["relative_gap"] <= 1e-2
    # A residual gate it cannot meet never fires.
    gated = jl.solve(lp, _polish_trigger=(0, 1.0), **{**kwargs, "max_epochs": 20})
    assert gated["stop_reason"] == "max_epochs"


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"update_mode": "pdhg"},
        {"update_mode": "halpern"},
        {"scale": False},
        {"diag_adapt": False},
        {"adaptive_eta": False, "step_size": "auto", "diag_adapt": False},
    ],
)
def test_make_solver_matches_solve(options):
    # One traced solve runs the same algorithm as solve(), but not to the bit:
    # the two paths compile to different XLA programs, and the adaptive steps
    # amplify the rounding differences by an amount that depends on the jax
    # version and the CPU (a CI runner saw ~1e-6 relative under halpern). So
    # check that both stop together and the traced solution certifies.
    lp = random_lp()
    tolerances = dict(
        primal_feasibility_tolerance=TOL,
        dual_feasibility_tolerance=TOL,
        dual_gap_tolerance=TOL,
    )
    reference = jl.solve(lp, **tolerances, **options)
    result = jax.jit(jl.make_solver(lambda _: lp, **tolerances, **options))(
        jnp.zeros(())
    )
    assert reference["stop_reason"] == "certificate"
    assert bool(result.converged) and int(result.stop_code) == 1
    assert abs(int(result.epochs) - reference["epochs"]) <= 1
    assert_certifies(lp, result.solution, reference_objective(lp))


def test_make_solver_vmap_batch():
    # Batch over the cost, matrix values and right-hand sides through
    # `with_values`: every member certifies and matches HiGHS. (Members are
    # not bit-identical to separate solves: XLA rounds batched reductions
    # differently, and the adaptive steps amplify it.)
    lp = jl.to_jaddle_sparse(random_lp())
    base = lp.values()
    rng = np.random.default_rng(0)
    size = 4

    def jitter(v):
        return v * (1 + 0.1 * rng.uniform(-1, 1, (size,) + v.shape))

    batch = jl.LPValues(
        c=jitter(base.c),
        A_eq_data=jitter(base.A_eq_data),
        b_eq=jitter(base.b_eq),
        A_ineq_data=jitter(base.A_ineq_data),
        b_ineq=jitter(base.b_ineq),
        lower_bounds=np.broadcast_to(base.lower_bounds, (size,) + base.lower_bounds.shape),
        upper_bounds=np.broadcast_to(base.upper_bounds, (size,) + base.upper_bounds.shape),
    )
    solve_fn = jl.make_solver(
        lp.with_values,
        primal_feasibility_tolerance=TOL,
        dual_feasibility_tolerance=TOL,
        dual_gap_tolerance=TOL,
    )
    result = jax.jit(jax.vmap(solve_fn))(batch)
    assert np.asarray(result.converged).all()
    for i in range(size):
        member = lp.with_values(jax.tree.map(lambda a: a[i], batch))
        member_lp = jl.LP(
            c=np.asarray(member.c),
            A_eq=getattr(jl, "__convert_to_scipy")(member.A_eq),
            b_eq=np.asarray(member.b_eq),
            A_ineq=getattr(jl, "__convert_to_scipy")(member.A_ineq),
            b_ineq=np.asarray(member.b_ineq),
            lower_bounds=np.asarray(member.lower_bounds),
            upper_bounds=np.asarray(member.upper_bounds),
        )
        ref = reference_objective(member_lp)
        obj = float(member.c @ result.solution.primal[i])
        assert abs(obj - ref) / (1 + abs(ref)) < 10 * TOL


def test_make_solver_builds_lp_from_params():
    # build_lp written with jnp, as in jc: the equality block is a jnp
    # constant (made concrete by ensure_compile_time_eval, so its sparsity is
    # kept), the inequality block a dense matrix that depends on params (so
    # every entry is stored). Batched under jit, each member matches HiGHS.
    base = random_lp()
    A_eq = np.asarray(base.A_eq.todense())
    A_eq[np.abs(A_eq) < 0.5] = 0.0
    A_ineq = np.asarray(base.A_ineq.todense())
    x0 = np.random.default_rng(3).uniform(0.1, 0.9, base.c.shape[0])

    def build_lp(params):
        cost_scale, ineq_scale = params
        A = jnp.asarray(A_ineq) * ineq_scale
        return jl.LP(
            c=cost_scale * jnp.asarray(base.c),
            A_eq=jnp.asarray(A_eq),
            b_eq=jnp.asarray(A_eq @ x0),
            A_ineq=A,
            b_ineq=A @ x0 + 0.5,
            lower_bounds=jnp.asarray(base.lower_bounds),
            upper_bounds=jnp.asarray(base.upper_bounds),
        )

    params = (jnp.array([1.0, 1.5, 0.7]), jnp.array([1.0, 0.8, 1.2]))
    result = jax.jit(
        jax.vmap(
            jl.make_solver(
                build_lp,
                primal_feasibility_tolerance=TOL,
                dual_feasibility_tolerance=TOL,
                dual_gap_tolerance=TOL,
            )
        )
    )(params)
    assert np.asarray(result.converged).all()
    for i in range(3):
        member = build_lp(jax.tree.map(lambda a: a[i], params))
        member_lp = jl.LP(
            c=np.asarray(member.c),
            A_eq=sp.csc_matrix(np.asarray(member.A_eq)),
            b_eq=np.asarray(member.b_eq),
            A_ineq=sp.csc_matrix(np.asarray(member.A_ineq)),
            b_ineq=np.asarray(member.b_ineq),
            lower_bounds=np.asarray(member.lower_bounds),
            upper_bounds=np.asarray(member.upper_bounds),
        )
        assert_certifies(
            member_lp,
            jax.tree.map(lambda a: a[i], result.solution),
            reference_objective(member_lp),
        )


def test_make_solver_keeps_constant_sparsity_under_jit():
    # The pattern comes from what does not depend on params: a jnp constant
    # keeps its nonzeros under jit, and a traced dense matrix stores them all.
    lp = random_lp()
    dense_eq = np.asarray(lp.A_eq.todense())
    dense_eq[:, ::2] = 0.0

    def build_lp(t):
        return jl.LP(
            c=t * jnp.asarray(lp.c),
            A_eq=jnp.asarray(dense_eq),
            b_eq=jnp.asarray(lp.b_eq),
            A_ineq=t * jnp.asarray(lp.A_ineq.todense()),
            b_ineq=jnp.asarray(lp.b_ineq),
            lower_bounds=jnp.asarray(lp.lower_bounds),
            upper_bounds=jnp.asarray(lp.upper_bounds),
        )

    pad = getattr(jl, "_build_padded")
    shapes = {}

    def nnz(t):
        pattern, _ = pad(build_lp, t)
        shapes["eq"] = pattern.A_eq.nse
        shapes["ineq"] = pattern.A_ineq.nse
        return t

    jax.jit(nnz)(1.0)
    assert shapes["eq"] == np.count_nonzero(dense_eq)
    assert shapes["ineq"] == lp.A_ineq.shape[0] * lp.A_ineq.shape[1]


def test_solve_batch_broadcasts_shared_fields():
    # Only c and b_ineq batched; matrices, b_eq and bounds shared.
    lp = random_lp()
    costs = np.stack([np.asarray(lp.c) * s for s in (1.0, 1.1, 0.9)])
    rhs = np.stack([np.asarray(lp.b_ineq) + d for d in (0.0, 0.05, 0.1)])

    def build_lp(params):
        c, b_ineq = params
        return jl.LP(
            c=c,
            A_eq=lp.A_eq,
            b_eq=lp.b_eq,
            A_ineq=lp.A_ineq,
            b_ineq=b_ineq,
            lower_bounds=lp.lower_bounds,
            upper_bounds=lp.upper_bounds,
        )

    result = jl.solve_batch(
        build_lp,
        (costs, rhs),
        primal_feasibility_tolerance=TOL,
        dual_feasibility_tolerance=TOL,
        dual_gap_tolerance=TOL,
    )
    assert result.solution.primal.shape == (3, lp.c.shape[0])
    assert np.asarray(result.converged).all()
    for i in range(3):
        member = build_lp((costs[i], rhs[i]))
        ref = reference_objective(member)
        obj = float(costs[i] @ result.solution.primal[i])
        assert abs(obj - ref) / (1 + abs(ref)) < 10 * TOL


def test_solve_batch_pads_empty_blocks():
    # toy_lp has no equality rows; the builder's LP is padded like solve()'s.
    lp = toy_lp()

    def build_lp(c):
        return jl.LP(
            c, lp.A_eq, lp.b_eq, lp.A_ineq, lp.b_ineq, lp.lower_bounds, lp.upper_bounds
        )

    result = jl.solve_batch(build_lp, np.stack([lp.c, 2 * lp.c]))
    assert np.asarray(result.converged).all()


def test_optimal_value_gradient_matches_finite_differences():
    tight = dict(
        primal_feasibility_tolerance=1e-10,
        dual_feasibility_tolerance=1e-10,
        dual_gap_tolerance=1e-10,
    )
    lp = random_lp()
    jlp = jl.to_jaddle_sparse(lp)
    values = jlp.values()
    value_fn = jl.make_optimal_value(jlp.with_values, **tight)
    assert abs(float(value_fn(values)) - reference_objective(lp)) < 1e-8
    grad = jax.grad(value_fn)(values)
    to_scipy = getattr(jl, "__convert_to_scipy")

    def highs_value(v):
        m = jlp.with_values(v)
        return reference_objective(
            jl.LP(
                np.asarray(m.c),
                to_scipy(m.A_eq),
                np.asarray(m.b_eq),
                to_scipy(m.A_ineq),
                np.asarray(m.b_ineq),
                np.asarray(m.lower_bounds),
                np.asarray(m.upper_bounds),
            )
        )

    # The value is piecewise linear, so a small central difference that keeps
    # the optimal basis is exact. Check each field's largest entry.
    h = 1e-5
    for field in jl.LPValues._fields:
        g = np.asarray(getattr(grad, field))
        k = int(np.argmax(np.abs(g)))
        arr = np.asarray(getattr(values, field))
        if not np.isfinite(arr[k]):
            continue

        def bumped(d):
            a = arr.copy()
            a[k] += d
            return values._replace(**{field: a})

        fd = (highs_value(bumped(h)) - highs_value(bumped(-h))) / (2 * h)
        assert abs(fd - g[k]) <= 1e-6 * (1 + abs(fd)), field


def test_optimal_value_gradient_through_build_lp():
    # The envelope gradient is carried back through build_lp: with
    # c = t c0 and b_ineq = b0 + s, dz*/dt and dz*/ds match finite
    # differences of HiGHS solves.
    tight = dict(
        primal_feasibility_tolerance=1e-10,
        dual_feasibility_tolerance=1e-10,
        dual_gap_tolerance=1e-10,
    )
    lp = random_lp()

    def build_lp(params):
        t, s = params
        return jl.LP(
            c=t * lp.c,
            A_eq=lp.A_eq,
            b_eq=lp.b_eq,
            A_ineq=lp.A_ineq,
            b_ineq=lp.b_ineq + s,
            lower_bounds=lp.lower_bounds,
            upper_bounds=lp.upper_bounds,
        )

    value_fn = jl.make_optimal_value(build_lp, **tight)
    params = (jnp.asarray(1.2), jnp.asarray(0.1))
    grad_t, grad_s = jax.jit(jax.grad(value_fn))(params)
    h = 1e-5
    fd_t = (
        reference_objective(build_lp((1.2 + h, 0.1)))
        - reference_objective(build_lp((1.2 - h, 0.1)))
    ) / (2 * h)
    fd_s = (
        reference_objective(build_lp((1.2, 0.1 + h)))
        - reference_objective(build_lp((1.2, 0.1 - h)))
    ) / (2 * h)
    assert abs(fd_t - float(grad_t)) <= 1e-6 * (1 + abs(fd_t))
    assert abs(fd_s - float(grad_s)) <= 1e-6 * (1 + abs(fd_s))


def test_optimal_value_gradient_vmaps():
    lp = jl.to_jaddle_sparse(random_lp())
    base = lp.values()
    tolerances = dict(
        primal_feasibility_tolerance=TOL,
        dual_feasibility_tolerance=TOL,
        dual_gap_tolerance=TOL,
    )

    def build_lp(c):
        return lp.with_values(base._replace(c=c))

    value_fn = jl.make_optimal_value(build_lp, **tolerances)
    costs = np.stack([np.asarray(base.c), 1.2 * np.asarray(base.c)])
    grads = jax.vmap(jax.grad(value_fn))(costs)
    # dz*/dc = x*: each row is that member's optimal primal.
    for i in range(2):
        x = jl.make_solver(build_lp, **tolerances)(costs[i]).solution.primal
        np.testing.assert_allclose(grads[i], x, atol=1e-4)


def test_perturbed_solution_matches_closed_form():
    # min c.x over {x1 + x2 = 4, x >= 0}: x1* = 4 [c1 < c2], so the perturbed
    # solution is x1 = 4 Φ((c2 - c1) / (σ√2)), with a closed-form derivative.
    from scipy.stats import norm

    lp = small_lp([3, 2], A_eq=[[1, 1]], b_eq=[4])
    sigma = 1.0

    def build_lp(c):
        return jl.LP(
            c, lp.A_eq, lp.b_eq, lp.A_ineq, lp.b_ineq, lp.lower_bounds, lp.upper_bounds
        )

    x_fn = jl.make_perturbed_solution(
        build_lp,
        sigma=sigma,
        num_samples=2048,
        primal_feasibility_tolerance=1e-6,
        dual_feasibility_tolerance=1e-6,
        dual_gap_tolerance=1e-6,
        max_epochs=200,
    )
    c = np.array([3.0, 2.0])
    key = jax.random.PRNGKey(0)
    d = (c[1] - c[0]) / (sigma * np.sqrt(2))
    x1 = 4 * norm.cdf(d)
    dx1 = 4 * norm.pdf(d) / (sigma * np.sqrt(2))
    x = jax.jit(x_fn)(c, key)
    jac = jax.jit(jax.jacrev(x_fn))(c, key)
    np.testing.assert_allclose(x, [x1, 4 - x1], atol=0.05)
    np.testing.assert_allclose(jac[0], [-dx1, dx1], atol=0.1)


def test_solution_vjp_matches_finite_differences():
    tight = dict(
        primal_feasibility_tolerance=1e-11,
        dual_feasibility_tolerance=1e-11,
        dual_gap_tolerance=1e-11,
    )
    lp = jl.to_jaddle_sparse(random_lp())
    values = lp.values()
    to_scipy = getattr(jl, "__convert_to_scipy")

    def highs_x(v):
        m = lp.with_values(v)
        res = linprog(
            np.asarray(m.c),
            A_ub=to_scipy(m.A_ineq),
            b_ub=np.asarray(m.b_ineq),
            A_eq=to_scipy(m.A_eq),
            b_eq=np.asarray(m.b_eq),
            bounds=[
                (lo, None if np.isinf(hi) else hi)
                for lo, hi in zip(np.asarray(m.lower_bounds), np.asarray(m.upper_bounds))
            ],
            method="highs-ds",
        )
        return res.x

    x_fn = jl.make_solution(lp.with_values, **tight)
    g = np.random.default_rng(1).standard_normal(values.c.shape[0])
    vjp = jax.grad(lambda v: jnp.dot(x_fn(v), g))(values)
    assert not np.any(np.asarray(vjp.c))  # x* is piecewise constant in c
    h = 1e-6
    for field in jl.LPValues._fields[1:]:
        gv = np.asarray(getattr(vjp, field))
        k = int(np.argmax(np.abs(gv)))
        arr = np.asarray(getattr(values, field))

        def bumped(d):
            a = arr.copy()
            a[k] += d
            return values._replace(**{field: a})

        fd = g @ (highs_x(bumped(h)) - highs_x(bumped(-h))) / (2 * h)
        assert abs(fd - gv[k]) <= 1e-6 * (1 + abs(fd)), field


def test_solution_vjp_rejects_optimal_face():
    # Every point of x1 + x2 = 1, x >= 0 is optimal; PDHG returns its middle,
    # where dx*/dθ is undefined.
    lp = jl.to_jaddle_sparse(small_lp([1, 1], [[-1, -1]], [-1]))
    x_fn = jl.make_solution(
        lp.with_values,
        primal_feasibility_tolerance=1e-10,
        dual_feasibility_tolerance=1e-10,
        dual_gap_tolerance=1e-10,
    )
    np.testing.assert_allclose(x_fn(lp.values()), [0.5, 0.5], atol=1e-6)
    with pytest.raises(Exception, match="not a nondegenerate vertex"):
        jax.grad(lambda v: x_fn(v).sum())(lp.values())


PRESOLVE_TOL = dict(
    primal_feasibility_tolerance=1e-8,
    dual_feasibility_tolerance=1e-8,
    dual_gap_tolerance=1e-8,
)


def reducible_lp():
    # random_lp plus a fixed column (lower = upper = 0.5) that appears in the
    # rows, and a singleton row x0 <= 0.7: HiGHS presolve removes both.
    lp = random_lp()
    rng = np.random.default_rng(5)
    m_eq, m_ineq = lp.A_eq.shape[0], lp.A_ineq.shape[0]
    fixed_eq = sp.csc_matrix(rng.standard_normal((m_eq, 1)))
    fixed_ineq = sp.csc_matrix(rng.standard_normal((m_ineq, 1)))
    singleton = sp.csc_matrix(([1.0], ([0], [0])), shape=(1, lp.c.shape[0] + 1))
    return jl.LP(
        c=np.append(lp.c, 1.0),
        A_eq=sp.hstack([lp.A_eq, fixed_eq]).tocsc(),
        b_eq=lp.b_eq + 0.5 * fixed_eq.toarray().ravel(),
        A_ineq=sp.vstack([sp.hstack([lp.A_ineq, fixed_ineq]), singleton]).tocsc(),
        b_ineq=np.append(lp.b_ineq + 0.5 * fixed_ineq.toarray().ravel(), 0.7),
        lower_bounds=np.append(lp.lower_bounds, 0.5),
        upper_bounds=np.append(lp.upper_bounds, 0.5),
    )


@pytest.mark.parametrize("make_lp", [random_lp, reducible_lp])
def test_solve_with_presolve(make_lp):
    lp = make_lp()
    result = jl.solve_with_presolve(lp, **PRESOLVE_TOL)
    if make_lp is reducible_lp:
        assert result["presolve"]["status"] == "kReduced"
    assert result["converged"]
    ref = reference_objective(lp)
    assert abs(result["objective"] - ref) / (1 + abs(ref)) < 1e-6
    # The solution and certificate are for the LP as given, in its own rows.
    s = result["solution"]
    assert s.primal.shape == lp.c.shape
    assert s.dual_eq.shape == lp.b_eq.shape
    assert s.dual_ineq.shape == lp.b_ineq.shape
    cert = solve_certificate(lp, s)
    assert cert["relative_primal_feasibility_residual"] <= 1e-6
    assert cert["relative_dual_feasibility_residual"] <= 1e-6


def test_solve_with_presolve_polish():
    # Tight feasibility, loose gap; polish from the first epoch on.
    lp = reducible_lp()
    result = jl.solve_with_presolve(
        lp,
        polish=True,
        tol=1e-8,
        dual_gap_tolerance=1e-4,
        first_polish_epoch=1,
        iterations_per_epoch=16,
        max_epochs=2000,
    )
    assert result["converged"]
    assert set(result["polish"]) == {"attempts", "epochs", "polished"}
    cert = solve_certificate(lp, result["solution"])
    assert cert["relative_primal_feasibility_residual"] <= 1e-8
    assert cert["relative_dual_feasibility_residual"] <= 1e-8
    assert cert["relative_gap"] <= 1e-4
    with pytest.raises(ValueError, match="alternatives"):
        jl.solve_with_presolve(lp, polish=True, reduced_solver=jl.solve)


def test_solve_with_presolve_reduced_to_empty():
    result = jl.solve_with_presolve(toy_lp())
    assert result["stop_reason"] == "presolve_solved"
    assert result["converged"]
    np.testing.assert_allclose(result["solution"].primal, [0.0, 4.0], atol=1e-9)
    # With nothing left to cross over, postsolve still supplies the basis.
    result = jl.solve_with_presolve(toy_lp(), run_crossover=True)
    assert result["crossover"] is None
    assert basic_count(result["basis"]) == 1


def test_solve_with_presolve_detects_infeasibility():
    lp = small_lp([1, 1], [[1, 1], [-1, -1]], [1, -3])
    result = jl.solve_with_presolve(lp)
    assert result["stop_reason"] == "presolve_infeasible"
    assert not result["converged"]
    assert result["solution"] is None


def test_solve_with_presolve_reads_files(tmp_path):
    import highspy

    from jaddle.highs_helpers import jaddle_lp_to_highs

    lp = reducible_lp()
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.passModel(jaddle_lp_to_highs(lp))
    path = str(tmp_path / "reducible.mps")
    highs.writeModel(path)
    result = jl.solve_with_presolve(path, **PRESOLVE_TOL)
    assert result["converged"]
    ref = reference_objective(lp)
    assert abs(result["objective"] - ref) / (1 + abs(ref)) < 1e-6


def write_mps(lp, path, maximise=False, offset=0.0):
    import highspy

    from jaddle.highs_helpers import jaddle_lp_to_highs

    model = jaddle_lp_to_highs(lp)
    model.offset_ = offset
    if maximise:
        model.sense_ = highspy.ObjSense.kMaximize
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.passModel(model)
    highs.writeModel(str(path))


def test_load(tmp_path):
    lp = reducible_lp()
    path = tmp_path / "reducible.mps"
    write_mps(lp, path, offset=2.5)
    ref = reference_objective(lp) + 2.5

    loaded = jl.load(path)
    assert isinstance(loaded, jl.JaddleLP)
    assert loaded.c.shape == lp.c.shape
    assert loaded.b_eq.shape == lp.b_eq.shape
    assert loaded.b_ineq.shape == lp.b_ineq.shape
    np.testing.assert_allclose(loaded.c, lp.c)

    for presolve in (False, True):
        loaded, offset = jl.load(str(path), presolve=presolve, return_offset=True)
        if presolve:
            assert loaded.c.shape[0] < lp.c.shape[0]
        result = jl.solve(loaded, **PRESOLVE_TOL)
        assert result["converged"]
        objective = float(loaded.c @ result["solution"].primal) + offset
        assert abs(objective - ref) / (1 + abs(ref)) < 1e-6


@pytest.mark.parametrize("presolve", [False, True])
def test_load_maximisation(tmp_path, presolve):
    # maximise -c.x is the same problem; load returns it as a minimisation.
    lp = random_lp()
    flipped = jl.LP(
        -lp.c, lp.A_eq, lp.b_eq, lp.A_ineq, lp.b_ineq, lp.lower_bounds, lp.upper_bounds
    )
    path = tmp_path / "max.mps"
    write_mps(flipped, path, maximise=True, offset=1.5)
    loaded, offset = jl.load(path, presolve=presolve, return_offset=True)
    result = jl.solve(loaded, **PRESOLVE_TOL)
    objective = float(loaded.c @ result["solution"].primal) + offset
    ref = reference_objective(lp) - 1.5
    assert abs(objective - ref) / (1 + abs(ref)) < 1e-6


def test_load_rejects_bad_models(tmp_path):
    with pytest.raises(ValueError, match="could not read"):
        jl.load(tmp_path / "missing.mps")
    path = tmp_path / "infeasible.mps"
    write_mps(small_lp([1, 1], [[1, 1], [-1, -1]], [1, -3]), path)
    with pytest.raises(ValueError, match="infeasible"):
        jl.load(path, presolve=True)


@pytest.mark.parametrize("options", [{}, {"run_crossover": True}], ids=str)
def test_solve_with_presolve_maximisation(options):
    import highspy

    from jaddle.highs_helpers import jaddle_lp_to_highs

    # maximise -c.x + 1.5 over reducible_lp's feasible set.
    lp = reducible_lp()
    model = jaddle_lp_to_highs(lp)
    model.col_cost_ = -np.asarray(model.col_cost_)
    model.sense_ = highspy.ObjSense.kMaximize
    model.offset_ = 1.5
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.passModel(model)
    highs.run()
    reference = highs.getSolution()
    ref = 1.5 - reference_objective(lp)
    assert abs(highs.getInfo().objective_function_value - ref) < 1e-6

    highs.clearSolver()
    result = jl.solve_with_presolve(highs, **PRESOLVE_TOL, **options)
    assert result["presolve"]["status"] == "kReduced"
    assert result["converged"]
    assert abs(result["objective"] - ref) / (1 + abs(ref)) < 1e-6
    # The duals are HiGHS's for the maximisation, and the model is untouched.
    for name in ("col_value", "col_dual", "row_dual"):
        np.testing.assert_allclose(
            result["highs_solution"][name], getattr(reference, name), atol=1e-5
        )
    assert highs.getLp().sense_ == highspy.ObjSense.kMaximize
    np.testing.assert_allclose(highs.getLp().col_cost_, -lp.c)
    # The Jaddle solution certifies the equivalent minimisation, the LP itself.
    cert = solve_certificate(lp, result["solution"])
    assert cert["relative_dual_feasibility_residual"] <= 1e-6


def basic_count(basis):
    import highspy

    statuses = list(basis.col_status) + list(basis.row_status)
    return sum(status == highspy.HighsBasisStatus.kBasic for status in statuses)


def test_crossover_reaches_a_vertex():
    # The vertex is exact even from a loose first-order solution.
    lp = random_lp()
    loose = solve(lp, tol=1e-3)
    crossed = jl.crossover(lp, loose["solution"])
    assert crossed["converged"]
    ref = reference_objective(lp)
    assert abs(crossed["objective"] - ref) <= 1e-9 * (1 + abs(ref))
    # A basis has one basic variable per row.
    assert basic_count(crossed["basis"]) == lp.b_eq.size + lp.b_ineq.size
    s = crossed["solution"]
    assert s.dual_eq.shape == lp.b_eq.shape
    assert s.dual_ineq.shape == lp.b_ineq.shape
    cert = solve_certificate(lp, s)
    assert cert["relative_primal_feasibility_residual"] <= 1e-9
    assert cert["relative_dual_feasibility_residual"] <= 1e-9
    assert cert["relative_gap_abs"] <= 1e-9


def test_crossover_without_equality_rows():
    # solve() pads toy_lp's empty equality block; crossover() must not.
    lp = toy_lp()
    crossed = jl.crossover(lp, solve(lp)["solution"])
    assert crossed["converged"]
    np.testing.assert_allclose(crossed["solution"].primal, [0.0, 4.0], atol=1e-12)
    assert crossed["solution"].dual_eq.shape == (0,)


def test_crossover_repairs_a_doubly_infeasible_basis(monkeypatch):
    import highspy

    from jaddle import highs_helpers as hh

    # A noisy optimum crosses over to a basis that is neither primal nor dual
    # feasible, which the repair handles in two stages: the primal simplex on
    # relaxed bounds, then the dual simplex.
    lp = random_lp(seed=1, m_eq=20, m_ineq=60, n=120)
    model = hh.jaddle_lp_to_highs(lp)
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.passModel(model)
    highs.run()
    optimum = highs.getSolution()
    dual_eq, dual_ineq = hh.highs_duals_to_jaddle(
        np.asarray(optimum.row_dual), model.row_lower_, model.row_upper_
    )
    rng = np.random.default_rng(0)
    noisy = jl.SaddleState(
        primal=np.asarray(optimum.col_value) + 0.03 * rng.standard_normal(lp.c.size),
        dual_ineq=np.maximum(dual_ineq + 0.03 * rng.standard_normal(dual_ineq.size), 0),
        dual_eq=dual_eq + 0.03 * rng.standard_normal(dual_eq.size),
    )

    first_stage = getattr(jl, "__relaxed_primal_repair")
    stages = []

    def recording(highs, *bounds):
        pivots = first_stage(highs, *bounds)
        # The first stage leaves the dual simplex selected when it has run.
        stages.append((highs.getOptionValue("simplex_strategy")[1], pivots))
        return pivots

    monkeypatch.setattr(jl, "__relaxed_primal_repair", recording)
    crossed = jl.crossover(lp, noisy)
    assert [strategy for strategy, _ in stages] == [1]
    assert crossed["converged"]
    assert crossed["simplex_iterations"] >= stages[0][1] > 0
    ref = reference_objective(lp)
    assert abs(crossed["objective"] - ref) <= 1e-6 * (1 + abs(ref))
    assert basic_count(crossed["basis"]) == lp.b_eq.size + lp.b_ineq.size

    # A simplex strategy of the caller's is left to do the whole repair.
    stages.clear()
    crossed = jl.crossover(lp, noisy, highs_options={"simplex_strategy": 1})
    assert stages == []
    assert crossed["converged"]
    assert abs(crossed["objective"] - ref) <= 1e-6 * (1 + abs(ref))


def test_solve_with_presolve_crossover():
    import highspy

    from jaddle.highs_helpers import jaddle_lp_to_highs

    # Two-sided rows are two rows in Jaddle's standard form, so the basis has
    # to be mapped back onto HiGHS's rows before postsolve.
    lp = reducible_lp()
    model = jaddle_lp_to_highs(lp)
    row_lower = np.array(model.row_lower_)
    row_lower[lp.b_eq.size :] = lp.b_ineq - 1.2
    model.row_lower_ = row_lower
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.passModel(model)
    highs.run()
    ref = highs.getInfo().objective_function_value

    result = jl.solve_with_presolve(model, run_crossover=True)
    assert result["converged"]
    assert result["crossover"]["converged"]
    assert abs(result["objective"] - ref) <= 1e-9 * (1 + abs(ref))
    assert basic_count(result["basis"]) == model.num_row_


def test_warm_start_without_padding_rows():
    # toy_lp has no equality rows; a warm start in its own row layout (no
    # dual for the padding row solve() adds) must be accepted.
    lp = toy_lp()
    cold = solve(lp)
    s = cold["solution"]
    warm = solve(
        lp,
        initial_solution=jl.SaddleState(
            primal=s.primal, dual_ineq=s.dual_ineq, dual_eq=np.zeros(0)
        ),
    )
    assert warm["stop_reason"] == "certificate"


def test_optimal_value_returns_its_own_solution():
    # With return_solution=True the gradient and the solution come from one
    # solve, so they agree exactly: dz*/dc = x*, dz*/db = -y*.
    lp = jl.to_jaddle_sparse(random_lp())
    value_fn = jl.make_optimal_value(lp.with_values, return_solution=True, **PRESOLVE_TOL)
    (z, result), grads = jax.value_and_grad(value_fn, has_aux=True)(lp.values())
    s = result.solution
    assert bool(result.converged)
    np.testing.assert_array_equal(grads.c, s.primal)
    np.testing.assert_array_equal(grads.b_eq, -s.dual_eq)
    np.testing.assert_array_equal(grads.b_ineq, -s.dual_ineq)
    assert float(z) == float(lp.c @ s.primal)


def test_cancel_event_stops_solve():
    cancel = threading.Event()
    cancel.set()
    result = solve(random_lp(), tol=1e-12, cancel=cancel)
    assert not result["converged"]
    assert result["stop_reason"] == "cancelled"


def test_solve_concurrent_matches_reference():
    lp = random_lp()
    result = jl.solve_concurrent(lp, **PRESOLVE_TOL)
    assert result["converged"]
    assert result["winner"] in ("jaddle", "highs")
    assert set(result["legs"]) == {"jaddle", "highs"}
    ref = reference_objective(lp)
    assert abs(result["objective"] - ref) / (1 + abs(ref)) < 1e-6


def test_solve_concurrent_ipm_wins_when_pdlp_is_starved():
    # One PDLP epoch cannot converge, so the IPM leg must be the winner.
    lp = random_lp()
    result = jl.solve_concurrent(lp, max_epochs=1, **PRESOLVE_TOL)
    assert result["winner"] == "highs"
    assert result["converged"]
    assert not result["legs"]["jaddle"]["converged"]
    ref = reference_objective(lp)
    assert abs(result["objective"] - ref) / (1 + abs(ref)) < 1e-6


@pytest.mark.parametrize("update_mode", ["pdhg", "halpern"])
def test_no_averaging(update_mode):
    # average=False carries no average buffer; the solve still certifies.
    # ("alternating" needs the average to converge, so it is left out.)
    lp = random_lp()
    result = solve(lp, update_mode=update_mode, average=False, report_best=False)
    assert_certified(lp, result, reference_objective(lp))


SOLVER_TOLS = dict(
    primal_feasibility_tolerance=TOL,
    dual_feasibility_tolerance=TOL,
    dual_gap_tolerance=TOL,
)


def jittered_values(lp, size, seed=0):
    base = lp.values()
    rng = np.random.default_rng(seed)
    return jax.tree.map(
        lambda v: jnp.asarray(v * (1 + 0.05 * rng.uniform(-1, 1, (size,) + v.shape))),
        base,
    )


def test_save_load_solver(tmp_path):
    lp = jl.to_jaddle_sparse(random_lp())
    solve_fn = jl.make_solver(lp.with_values, **SOLVER_TOLS)
    jl.save_solver(solve_fn, tmp_path / "lp.jls", lp.values())
    loaded = jl.load_solver(tmp_path / "lp.jls")
    result, expected = loaded(lp.values()), solve_fn(lp.values())
    assert bool(result.converged)
    np.testing.assert_allclose(result.solution.primal, expected.solution.primal, atol=1e-10)
    assert int(result.epochs) == int(expected.epochs)


def test_save_load_solver_batched(tmp_path):
    lp = jl.to_jaddle_sparse(random_lp())
    solve_fn = jl.make_solver(lp.with_values, **SOLVER_TOLS)
    jl.save_solver(solve_fn, tmp_path / "batched.jls", lp.values(), batch=True)
    loaded = jl.load_solver(tmp_path / "batched.jls")
    for size in (1, 3):  # 1 is padded to the file's minimum batch of 2
        batch = jittered_values(lp, size)
        result, expected = loaded(batch), jax.vmap(solve_fn)(batch)
        assert result.solution.primal.shape == (size, lp.c.shape[0])
        np.testing.assert_allclose(
            result.solution.primal, expected.solution.primal, atol=1e-10
        )


def test_save_load_optimal_value_gradient(tmp_path):
    lp = jl.to_jaddle_sparse(random_lp())
    value_fn = jl.make_optimal_value(lp.with_values, **SOLVER_TOLS)
    jl.save_solver(value_fn, tmp_path / "value.jls", lp.values(), batch=True, vjp_order=1)
    loaded = jl.load_solver(tmp_path / "value.jls")
    batch = jittered_values(lp, 2)
    np.testing.assert_allclose(loaded(batch), jax.vmap(value_fn)(batch), rtol=1e-10)
    got = jax.grad(lambda b: loaded(b).sum())(batch)
    want = jax.grad(lambda b: jax.vmap(value_fn)(b).sum())(batch)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, atol=1e-10)


def test_save_load_perturbed_solution(tmp_path):
    lp = jl.to_jaddle_sparse(small_lp([3, 2], A_eq=[[1, 1]], b_eq=[4]))
    x_fn = jl.make_perturbed_solution(
        lp.with_values, sigma=1.0, num_samples=8, max_epochs=200, **SOLVER_TOLS
    )
    key = jax.random.PRNGKey(0)
    jl.save_solver(x_fn, tmp_path / "perturbed.jls", lp.values(), key, vjp_order=1)
    loaded = jl.load_solver(tmp_path / "perturbed.jls")
    np.testing.assert_allclose(loaded(lp.values(), key), x_fn(lp.values(), key), atol=1e-10)
    # jax.grad of a scalar, not jacrev: a loaded function cannot be vmap-ed.
    g = jnp.array([1.0, -2.0])
    np.testing.assert_allclose(
        jax.grad(lambda c: g @ loaded(lp.values()._replace(c=c), key))(lp.values().c),
        jax.grad(lambda c: g @ x_fn(lp.values()._replace(c=c), key))(lp.values().c),
        atol=1e-8,
    )


def test_save_solution_without_gradient(tmp_path):
    lp = jl.to_jaddle_sparse(random_lp())
    x_fn = jl.make_solution(lp.with_values, **SOLVER_TOLS)
    jl.save_solver(x_fn, tmp_path / "x.jls", lp.values())
    loaded = jl.load_solver(tmp_path / "x.jls")
    np.testing.assert_allclose(loaded(lp.values()), x_fn(lp.values()), atol=1e-10)
    # Its gradient runs scipy on the host, which cannot be exported.
    with pytest.raises(Exception, match="vjp_order=0"):
        jl.save_solver(x_fn, tmp_path / "x1.jls", lp.values(), vjp_order=1)


def test_load_solver_checks_file_kind(tmp_path):
    import jaddle.jaddle_convex as jc

    cp = jc.JaddleCP(
        2, lambda x: x @ x, lambda x: jnp.zeros(0), lambda x: jnp.zeros(0),
        jnp.zeros(2), jnp.ones(2),
    )
    jc.save_cp(cp, tmp_path / "cp.jcp")
    with pytest.raises(ValueError, match="jaddle_convex.load_cp"):
        jl.load_solver(tmp_path / "cp.jcp")


def assert_same_lp(got, want):
    for name in ("c", "b_eq", "b_ineq", "lower_bounds", "upper_bounds", "b"):
        np.testing.assert_array_equal(getattr(got, name), getattr(want, name))
        assert getattr(got, name).dtype == getattr(want, name).dtype
    for name in ("A_eq", "A_ineq", "A", "A_T"):
        g, w = getattr(got, name), getattr(want, name)
        assert g.shape == w.shape
        np.testing.assert_array_equal(g.data, w.data)
        np.testing.assert_array_equal(g.indices, w.indices)
    assert got.cones == want.cones


def test_save_load_lp(tmp_path):
    lp = jl.to_jaddle_sparse(random_lp())
    jl.save_lp(lp, tmp_path / "lp.jlp")
    loaded = jl.load_lp(tmp_path / "lp.jlp")
    assert_same_lp(loaded, lp)
    got, want = jl.solve(loaded, **SOLVER_TOLS), jl.solve(lp, **SOLVER_TOLS)
    assert got["converged"]
    np.testing.assert_array_equal(got["solution"].primal, want["solution"].primal)


def test_save_load_lp_with_cones(tmp_path):
    from jaddle.cones import Cones

    lp = jl.to_jaddle_sparse(random_lp())
    n = lp.c.shape[0]
    cones = Cones(soc=(3, 2), psd=(2,), exp=1)
    lower = np.asarray(lp.lower_bounds).copy()
    upper = np.asarray(lp.upper_bounds).copy()
    lower[n - cones.size :], upper[n - cones.size :] = -np.inf, np.inf
    conic = jl.JaddleLP(
        lp.c, lp.A_eq, lp.b_eq, lp.A_ineq, lp.b_ineq,
        jnp.asarray(lower), jnp.asarray(upper), cones=cones,
    )
    jl.save_lp(conic, tmp_path / "conic.jlp")
    assert_same_lp(jl.load_lp(tmp_path / "conic.jlp"), conic)


def test_load_lp_dtype(tmp_path):
    lp = jl.to_jaddle_sparse(random_lp())
    jl.save_lp(lp, tmp_path / "lp.jlp")
    loaded = jl.load_lp(tmp_path / "lp.jlp", dtype=jnp.float32)
    assert loaded.c.dtype == jnp.float32 and loaded.A.data.dtype == jnp.float32
    np.testing.assert_allclose(loaded.A_eq.data, lp.A_eq.data, rtol=1e-6)


def test_save_lp_rejects_scipy_lp(tmp_path):
    with pytest.raises(TypeError, match="to_jaddle_sparse"):
        jl.save_lp(random_lp(), tmp_path / "lp.jlp")
    jl.save_lp(jl.to_jaddle_sparse(random_lp()), tmp_path / "lp.jlp")
    with pytest.raises(ValueError, match="load_lp"):
        jl.load_solver(tmp_path / "lp.jlp")


@pytest.mark.parametrize("precision", ["float32", "float16"])
@pytest.mark.parametrize("scale", [True, False])
def test_per_call_precision(precision, scale):
    import jaddle.jaddle_optimisers as jo

    lp = random_lp()
    J = jl.to_jaddle_sparse(lp)
    result = solve(J, tol=1e-3, precision=precision, scale=scale, max_epochs=3000)
    assert result["solution"].primal.dtype == jnp.float32
    assert_certified(lp, result, reference_objective(lp), tol=1e-3)
    # The global float64 profile is back.
    assert jax.config.jax_enable_x64
    assert jo.jaddle_dtype() == jnp.float64
    assert jl.solve(J, max_epochs=1)["solution"].primal.dtype == jnp.float64


def test_per_call_precision_warm_start_and_scipy_lp():
    lp = random_lp()
    ref = reference_objective(lp)
    warm = solve(lp)["solution"]  # float64
    result = solve(lp, tol=1e-4, precision="float32", initial_solution=warm)
    assert result["solution"].primal.dtype == jnp.float32
    assert_certified(lp, result, ref, tol=1e-4)


@pytest.mark.parametrize("device", ["cpu", jax.devices("cpu")[0]])
def test_per_call_device(device):
    lp = random_lp()
    result = solve(lp, device=device)
    assert_certified(lp, result, reference_objective(lp))
    assert result["solution"].primal.devices() == {jax.devices("cpu")[0]}


@pytest.mark.parametrize(
    "options, match",
    [(dict(device="nope"), "unknown device"), (dict(precision="float8"), "profile")],
)
def test_per_call_settings_rejected(options, match):
    with pytest.raises(ValueError, match=match):
        jl.solve(toy_lp(), **options)


# %% bound_target: stop once the Lagrangian bound reaches a target
def boxed_lp(seed=0, m_eq=4, m_ineq=12, n=25, infeasible=False):
    """A feasible LP with every variable in [0, 2] (so the Lagrangian bound
    is finite at any duals); ``infeasible`` adds contradictory rows."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0.2, 1.8, n)
    A_eq = rng.standard_normal((m_eq, n))
    A_ineq = rng.standard_normal((m_ineq, n))
    b_ineq = A_ineq @ x0 + rng.uniform(0.1, 1.0, m_ineq)
    if infeasible:
        row = rng.standard_normal(n)
        A_ineq = np.vstack([A_ineq, row, -row])
        b_ineq = np.concatenate([b_ineq, [-1.0, -1.0]])  # row·x <= −1 and >= 1
    return jl.LP(
        c=rng.standard_normal(n),
        A_eq=sp.csc_matrix(A_eq),
        b_eq=A_eq @ x0,
        A_ineq=sp.csc_matrix(A_ineq),
        b_ineq=b_ineq,
        lower_bounds=np.zeros(n),
        upper_bounds=np.full(n, 2.0),
    )


def lagrangian_of(lp, solution):
    """The Lagrangian bound of a solution's duals, computed on the host."""
    y_eq = np.asarray(solution.dual_eq, np.float64)[: lp.A_eq.shape[0]]
    y_ineq = np.maximum(np.asarray(solution.dual_ineq, np.float64)[: lp.A_ineq.shape[0]], 0.0)
    r = lp.c + lp.A_eq.T @ y_eq + lp.A_ineq.T @ y_ineq
    box = np.where(r > 0, r * lp.lower_bounds, r * lp.upper_bounds)
    return box.sum() - lp.b_eq @ y_eq - lp.b_ineq @ y_ineq


TIGHT = dict(
    primal_feasibility_tolerance=1e-8, dual_feasibility_tolerance=1e-8, dual_gap_tolerance=1e-8
)


def test_bound_target_stops_with_a_valid_bound():
    lp = boxed_lp()
    optimum = jl.solve(lp, **TIGHT)
    z = float(lp.c @ np.asarray(optimum["solution"].primal))
    target = z - 0.05 * (1 + abs(z))
    result = jl.solve(lp, bound_target=target, max_epochs=2000, **TIGHT)
    assert result["stop_reason"] == "bound_target" and not result["converged"]
    assert result["epochs"] <= optimum["epochs"]
    assert result["lagrangian_bound"] >= target
    host = lagrangian_of(lp, result["solution"])
    assert host == pytest.approx(result["lagrangian_bound"], rel=1e-8, abs=1e-8)
    assert target <= host <= z + 1e-6 * (1 + abs(z))


def test_bound_target_above_the_optimum_never_fires():
    lp = boxed_lp(1)
    result = jl.solve(lp, bound_target=1e6, **TIGHT)
    assert result["stop_reason"] == "certificate" and result["converged"]
    assert result["lagrangian_bound"] < 1e6


def test_bound_target_catches_an_infeasible_lp():
    lp = boxed_lp(2, infeasible=True)
    result = jl.solve(lp, bound_target=1e3, max_epochs=500, **TIGHT)
    assert result["stop_reason"] == "bound_target"
    assert result["epochs"] < 50
    assert lagrangian_of(lp, result["solution"]) >= 1e3


def test_bound_target_needs_no_vertex_bias():
    with pytest.raises(ValueError, match="bound_target"):
        jl.solve(boxed_lp(), bound_target=0.0, vertex_bias=1e-6)
