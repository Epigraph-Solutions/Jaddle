"""The float16 profile, end to end.

The precision profile is global and conftest.py sets float64, so these checks
run in a child process that configures float16 itself. Under float16 only the
scaled LP the solver iterates on is stored in half precision; iterates,
norms, step sizes and the final certificate are at least float32.
"""

import os
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent(
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.optimize import linprog

    import jaddle.jaddle_optimisers as jo

    jo.configure_jax("float16")
    import jax.numpy as jnp
    import jaddle.jaddle_linear as jl


    def random_lp(seed, m_eq=5, m_ineq=15, n=30, scale=1.0, c_zero=False):
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(0.1, 0.9, n)
        A_eq = rng.standard_normal((m_eq, n)) * scale
        A_ineq = rng.standard_normal((m_ineq, n)) * scale
        upper = np.ones(n)
        upper[: n // 3] = np.inf
        c = np.zeros(n) if c_zero else rng.standard_normal(n) * scale
        return jl.LP(
            c=c,
            A_eq=sp.csc_matrix(A_eq),
            b_eq=A_eq @ x0,
            A_ineq=sp.csc_matrix(A_ineq),
            b_ineq=A_ineq @ x0 + rng.uniform(0.1, 1.0, m_ineq) * scale,
            lower_bounds=np.zeros(n),
            upper_bounds=upper,
        )


    def reference(lp):
        bounds = [(lo, None if np.isinf(hi) else hi)
                  for lo, hi in zip(lp.lower_bounds, lp.upper_bounds)]
        res = linprog(lp.c, A_ub=lp.A_ineq, b_ub=lp.b_ineq, A_eq=lp.A_eq,
                      b_eq=lp.b_eq, bounds=bounds, method="highs")
        assert res.status == 0
        return res.fun


    def check(lp, tol, refine=False):
        solver = jl.solve_with_refinement if refine else jl.solve
        r = solver(lp, primal_feasibility_tolerance=tol,
                   dual_feasibility_tolerance=tol, dual_gap_tolerance=tol,
                   max_epochs=2000)
        sol = r["solution"]
        assert sol.primal.dtype == jnp.float32, sol.primal.dtype
        cert = jl.evaluate_lp_certificate(
            jl.to_jaddle_sparse(lp), sol.primal, sol.dual_eq, sol.dual_ineq)
        cert = {k: float(v) for k, v in cert.items()}
        assert all(np.isfinite(v) for v in cert.values()), cert
        ref = reference(lp)
        rel = abs(cert["objective"] - ref) / (1 + abs(ref))
        assert rel < 10 * tol, (rel, cert, ref)
        assert cert["relative_primal_feasibility_residual"] < 10 * tol, cert
        assert cert["relative_dual_feasibility_residual"] < 10 * tol, cert
        return r


    # Plain solve to a tolerance float16 data can reach.
    for seed in range(3):
        check(random_lp(seed), 1e-3)
    # Coefficients and an objective far beyond float16's range (65504):
    # the unscaled LP and the certificate must not overflow.
    check(random_lp(0, scale=1e5), 1e-3)
    # A feasibility problem (c = 0): k starts at its floor and the first
    # epoch barely moves; the primal-weight update must not divide 0 by 0.
    r = jl.solve(random_lp(1, c_zero=True), max_epochs=50)
    assert np.all(np.isfinite(np.asarray(r["solution"].primal)))
    # Iterative refinement measures residuals on the float32 LP, so it gets
    # past the float16 rounding of the matrix.
    check(random_lp(0), 1e-6, refine=True)

    # A float64 solve in this float16 session: x64 on for the call only.
    import jax
    lp = random_lp(0)
    r = jl.solve(jl.to_jaddle_sparse(lp), precision="float64",
                 primal_feasibility_tolerance=1e-8,
                 dual_feasibility_tolerance=1e-8, dual_gap_tolerance=1e-8)
    assert r["converged"] and r["solution"].primal.dtype == jnp.float64
    ref = reference(lp)
    obj = float(np.asarray(lp.c) @ np.asarray(r["solution"].primal))
    assert abs(obj - ref) / (1 + abs(ref)) < 1e-7, (obj, ref)
    assert not jax.config.jax_enable_x64 and jo.jaddle_dtype() == jnp.float16

    # Quadratic programs: float16 matrices, float32 iterates, and a
    # `converged` that is judged on the QP as given.
    import jaddle.jaddle_convex as jc


    def random_qp(n, dense):
        rng = np.random.default_rng(n)
        if dense:
            F = rng.standard_normal((n, n // 2))
            Q = F @ F.T / n + 0.1 * np.eye(n)
            A = rng.standard_normal((n // 2, n))
        else:
            F = sp.random(n, n, density=5 / n, random_state=1)
            Q = (F @ F.T + 0.1 * sp.eye(n)).tocsr()
            A = sp.random(n // 2, n, density=8 / n, random_state=2).tocsr()
        x0 = rng.uniform(0, 1, n)
        return dict(Q=Q, c=rng.standard_normal(n), A_ineq=A,
                    b_ineq=A @ x0 + 0.1, A_eq=np.ones((1, n)),
                    b_eq=np.array([x0.sum()]), lower_bounds=np.zeros(n),
                    upper_bounds=np.full(n, 2.0))


    def qp_tols(tol):
        return dict(primal_feasibility_tolerance=tol,
                    dual_feasibility_tolerance=tol, dual_gap_tolerance=tol)


    for n, dense in ((200, True), (3000, False)):
        qp = random_qp(n, dense)
        r = jc.quadratic_program(**qp, max_epochs=500, **qp_tols(1e-3))
        cert = r["certificate"]
        assert np.all(np.isfinite(np.asarray(r["solution"].primal)))
        assert r["converged"], cert
        assert cert["relative_primal_feasibility_residual"] <= 1e-3, cert
        assert cert["relative_dual_feasibility_residual"] <= 1e-3, cert
        assert cert["relative_gap"] <= 1e-3, cert
        # A tolerance below float16's rounding of the matrices: whatever the
        # solver concludes on the rounded QP, `converged` must agree with the
        # certificate of the QP as given.
        r = jc.quadratic_program(**qp, max_epochs=300, **qp_tols(1e-4))
        cert = r["certificate"]
        met = (cert["relative_primal_feasibility_residual"] <= 1e-4
               and cert["relative_dual_feasibility_residual"] <= 1e-4
               and cert["relative_gap"] <= 1e-4)
        assert r["converged"] == met, (r["converged"], cert)
    # Only the matrices are stored in float16; the vectors (and so the
    # iterates) are float32.
    scaled = jc._scale_qp(qp["Q"], 3000, qp["A_eq"], qp["A_ineq"],
                          qp["lower_bounds"], qp["upper_bounds"], 10, 1)[-1]
    assert all(m.dtype == jnp.float16 for m in scaled[:5])
    assert all(v.dtype == jnp.float32 for v in scaled[5:])
    print("OK")
    """
)


def test_float16_profile():
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.pop("JADDLE_JAX_PROFILE", None)
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, (
        proc.stdout[-3000:] + proc.stderr[-3000:]
    )
