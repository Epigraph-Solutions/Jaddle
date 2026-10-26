# %%
import jax
import jax.numpy as jnp
from optax.projections import projection_non_negative, projection_box
import optax
from typing import NamedTuple
import time
from jaddle.jaddle_basic_types import JaddleCP, SaddleState, _dot, _wide
import jaddle.jaddle_optimisers as jo
from jaddle import jaddle_export as _export
import numpy as np

# Epoch budget for a chunk that should run until convergence: large enough never
# to bind, small enough that `count + n` cannot overflow an int32 counter.
_UNBOUNDED_EPOCHS = 2**30
# Target wall time of one chunk of epochs between host check-ins.
_CHUNK_TARGET_SECONDS = 1.0
# Buffered verbose-log rows / restart events per chunk.
_LOG_CAP = 64
_EVT_CAP = 64

_ADAPTIVE_MODES = ("alternating", "extragradient", "pdhg")
# Adaptive modes whose line search validates each step one iteration late.
_LAGGED_MODES = ("alternating", "pdhg")


def _dual_stack(state):
    return jnp.concatenate([state.dual_eq, state.dual_ineq])


def _kkt_residuals(cp, state):
    """The generic convergence measures of ``cp`` at ``state``, in ``cp``'s
    own units: ``(objective, primal_grad_norm, constraint_bound,
    complementarity_slack)`` -- the ∞-norm projected-gradient (stationarity)
    residual, the largest constraint violation, and the largest |λ g| relative
    to ``1 + |objective|``."""

    def lagrangian_with_obj(s):
        # (lagrangian, objective) as (value, aux), so value_and_grad gets the
        # objective without a second forward pass.
        obj = cp.objective(s.primal)
        lagrangian = (
            obj
            + s.dual_ineq @ cp.constraints_ineq(s.primal)
            + s.dual_eq @ cp.constraints_eq(s.primal)
        )
        return lagrangian, obj

    (_, objective_value), gradient = jax.value_and_grad(
        lagrangian_with_obj, has_aux=True
    )(state)

    projected_primal = projection_box(
        state.primal - gradient.primal, cp.lower_bounds, cp.upper_bounds
    )
    primal_grad_norm = jnp.max(jnp.abs(state.primal - projected_primal))

    ineq_violations = jnp.maximum(gradient.dual_ineq, 0.0)
    max_ineq_violation = (
        jnp.max(ineq_violations) if ineq_violations.size > 0 else jnp.zeros(())
    )
    eq_violations = jnp.abs(gradient.dual_eq)
    max_eq_violation = (
        jnp.max(eq_violations) if eq_violations.size > 0 else jnp.zeros(())
    )
    complementarity_slack = (
        jnp.max(jnp.abs(state.dual_ineq * gradient.dual_ineq))
        if state.dual_ineq.size > 0
        else jnp.zeros(())
    ) / (1.0 + jnp.abs(objective_value))
    constraint_bound = jnp.maximum(max_ineq_violation, max_eq_violation)
    return objective_value, primal_grad_norm, constraint_bound, complementarity_slack


# gradient_scaling clamps every scale factor to [1/clip, clip].
_GRADIENT_SCALING_CLIP = 1e4


def _gradient_scales(cp, x0, probes):
    """IPOPT-style gradient-based scales at ``x0``: ``sigma = 1/‖∇f(x0)‖₂`` for
    the objective and ``e_i = 1/‖∇g_i(x0)‖₂`` for each constraint row (stacked
    ``[eq; ineq]``), clamped to ``[1/clip, clip]``; a zero gradient keeps 1.

    The row norms are exact when the Jacobian is cheap (``n <= probes``:
    forward mode, or ``m <= probes``: reverse mode) and otherwise estimated
    from ``probes`` Rademacher JVPs: ``E[(J v)_i²] = ‖J_i‖²``, so the cost is
    ``probes`` JVPs whatever ``m`` and ``n``. The key is fixed, so the scales
    are deterministic and traceable."""

    def stacked(x):
        return jnp.concatenate([cp.constraints_eq(x), cp.constraints_ineq(x)])

    n = x0.shape[0]
    m = jax.eval_shape(stacked, x0).shape[0]
    grad_f = jax.grad(cp.objective)(x0)
    if m == 0:
        row_norm2 = jnp.zeros(0, x0.dtype)
    elif n <= probes or m <= probes:
        jac = jax.jacfwd(stacked) if n <= probes else jax.jacrev(stacked)
        row_norm2 = jnp.sum(jac(x0) ** 2, axis=1)
    else:
        V = jax.random.rademacher(jax.random.PRNGKey(0), (probes, n), dtype=x0.dtype)
        JV = jax.vmap(lambda v: jax.jvp(stacked, (x0,), (v,))[1])(V)
        row_norm2 = jnp.mean(JV**2, axis=0)

    def scale(norm2):
        safe = jnp.where(norm2 > 0.0, norm2, 1.0)
        return jnp.where(
            norm2 > 0.0,
            jnp.clip(
                1.0 / jnp.sqrt(safe),
                1.0 / _GRADIENT_SCALING_CLIP,
                _GRADIENT_SCALING_CLIP,
            ),
            1.0,
        )

    return scale(_dot(grad_f, grad_f)), scale(row_norm2)


def _gradient_scaled(cp, x0, probes):
    """``cp`` with its objective and constraint rows rescaled by
    ``_gradient_scales`` at ``x0`` (the variables are untouched). Returns
    ``(scaled_cp, to_scaled, from_scaled, sigma)``: the two maps convert a
    ``SaddleState``'s duals between the problems, since
    ``σf + Σ λ̃_i e_i g_i = σ (f + Σ (e_i λ̃_i / σ) g_i)`` gives
    ``λ_i = e_i λ̃_i / σ``. The scaled problem's ``residuals`` evaluate ``cp``'s
    own (or the generic ``_kkt_residuals``) at the unscaled point, so the
    convergence test keeps the original problem's units."""
    x0 = projection_box(x0, cp.lower_bounds, cp.upper_bounds)
    m_eq = jax.eval_shape(cp.constraints_eq, x0).shape[0]
    # A CP from ``load_cp`` carries its scales as an exported function: its
    # own callables have no forward mode or batching, which the scales need.
    scales_fn = getattr(cp, "gradient_scales", None)
    if scales_fn is None:
        scales_fn = lambda x: _gradient_scales(cp, x, probes)
    sigma, e = jax.jit(scales_fn)(x0)
    e_eq, e_ineq = e[:m_eq], e[m_eq:]

    def from_scaled(state):
        return SaddleState(
            primal=state.primal,
            dual_ineq=e_ineq * state.dual_ineq / sigma,
            dual_eq=e_eq * state.dual_eq / sigma,
        )

    def to_scaled(state):
        return SaddleState(
            primal=state.primal,
            dual_ineq=sigma * state.dual_ineq / e_ineq,
            dual_eq=sigma * state.dual_eq / e_eq,
        )

    def residuals(state):
        original = from_scaled(state)
        if getattr(cp, "residuals", None) is not None:
            return cp.residuals(original)
        return _kkt_residuals(cp, original)[1:]

    dual_bound = None
    if cp.dual_bound is not None:

        def dual_bound(dual_ineq, dual_eq):
            return sigma * cp.dual_bound(
                e_ineq * dual_ineq / sigma, e_eq * dual_eq / sigma
            )

    scaled_cp = JaddleCP(
        num_variables=cp.num_variables,
        objective=lambda x: sigma * cp.objective(x),
        constraints_eq=lambda x: e_eq * cp.constraints_eq(x),
        constraints_ineq=lambda x: e_ineq * cp.constraints_ineq(x),
        lower_bounds=cp.lower_bounds,
        upper_bounds=cp.upper_bounds,
        dual_bound=dual_bound,
        residuals=residuals,
    )
    return scaled_cp, to_scaled, from_scaled, sigma


def _reported_point(carry, average):
    # The point the last epoch's metrics describe: the average or the last
    # iterate (with report_best, whichever had the lower merit).
    if not average:
        return carry["state"]
    return jax.tree.map(
        lambda a, s: jnp.where(carry["used_avg"], a, s), carry["avg"], carry["state"]
    )


def _init_k_slot(k, eta, state, update_mode):
    # The k-slot rides in opt_state next to the optimiser state:
    #   - k-scaling only (eta None):        k
    #   - adaptive extragradient:           (k, eta)
    #   - adaptive pdhg / alternating:      (k, eta, z_prev, F_prev, valid)
    # pdhg and alternating validate each step one iteration late: z_prev is
    # the iterate the current one was stepped from (at step eta), F_prev the
    # saddle operator there. valid=False (fresh start / restart) skips the check once.
    dtype = _wide(state.primal.dtype)  # scalars at least float32
    k = jnp.asarray(k, dtype)
    if eta is None:
        return k
    eta = jnp.asarray(eta, dtype)
    if update_mode not in _LAGGED_MODES:
        return (k, eta)
    zeros = jax.tree.map(jnp.zeros_like, state)
    zeros2 = jax.tree.map(jnp.zeros_like, state)
    return (k, eta, zeros, zeros2, jnp.asarray(False))


def _make_epoch_fn(
    cp: JaddleCP,
    optimiser,
    weight_function=lambda _: 1.0,
    average=True,
    update_mode="alternating",
    k_scaling=False,
    adaptive_step=False,
    diag=False,
):
    # Returns `run_epoch(max_iter, start_iter, state, average_state, opt_state,
    # total_weight)`, which runs one epoch of `max_iter` iterations and returns
    # (i, state, average_state, opt_state, total_weight). With `diag`
    # (diag_adapt, adaptive step only) it also takes `diag=(t, s)` and
    # `accum=(H_x, H_y)` keywords and returns the updated `accum` last: the
    # steps become tau = eta/k * t, sigma = eta*k * s, the line search measures
    # in that metric, and each accepted step adds its eta-normalised squared
    # movement (Δ/eta)² to the accumulators `solve` turns into the next
    # diagonal at a restart. It is a plain
    # traceable function, not jitted: `solve` calls it inside its device-side
    # epoch loop (`run_chunk`), which is jitted as a whole. `max_iter` must be a
    # Python int (it is the scan length).
    #
    # Per-iteration adaptive step size (Malitsky-Tam local-Lipschitz line
    # search). With `adaptive_step` the extragradient scheme replaces
    # the optimiser's fixed learning rate with a single scalar base step eta,
    # line-searched every iteration: the look-ahead and corrector already
    # evaluate grad twice, so the local Lipschitz estimate
    #     L_hat = ‖g_half - g‖_w / ‖z_half - z‖_w
    # (w = the k-weighted norm) comes free, and the step is admissible while
    # eta · L_hat <= 1/sqrt(2). The primal/dual steps are tau=eta/k, sigma=eta*k.
    # eta is packed alongside k in the k-slot of opt_state. Requires k_scaling
    # (it needs k). pdhg and alternating share the one-iteration-late line
    # search below; alternating (theta = 0) is not contractive in the last
    # iterate, so its guarantees are for the average, and the line search only
    # keeps its step within the local Lipschitz bound.
    if adaptive_step and not k_scaling:
        raise ValueError("adaptive_eta requires k_scaling (primal weight k)")
    diag_on = bool(diag)
    # k-scaling is an orthogonal option (any update_mode): a primal weight k
    # rescales the primal/dual gradients by (1/k, k) before opt_update, so the
    # dual/primal step ratio is k**2. When on, k is packed into opt_state and
    # rebalanced at each restart in `solve` (PDLP-style); constant within an
    # epoch.

    def projection_primal(primal_state):
        return projection_box(primal_state, cp.lower_bounds, cp.upper_bounds)

    # Hand-written saddle gradient of the Lagrangian
    #   L = obj(x) + d_ineq·g_ineq(x) + d_eq·g_eq(x).
    # Its structure lets each half be computed independently:
    #   - dual partials are just the constraint residuals g(x) — a forward eval,
    #     no autodiff;
    #   - the primal partial is grad(obj)(x) + J_ineqᵀ·d_ineq + J_eqᵀ·d_eq,
    #     a single VJP of the constraint map against the duals plus the objective
    #     gradient.
    # The negation on the dual partials (descent on x / ascent on the duals) is
    # folded into the returned dual fields. Splitting `grad` into one-sided
    # variants lets the alternating/extragradient schemes pay for only the half
    # they actually consume.

    def constraints(primal):
        return (cp.constraints_ineq(primal), cp.constraints_eq(primal))

    def lagrangian_map(primal):
        # Stacks the objective and both constraint maps into one function so a
        # single reverse pass covers all of them. The primal partial of the
        # Lagrangian is the VJP of this map seeded with cotangents
        # (1.0, dual_ineq, dual_eq): the 1.0 on the objective output reproduces
        # grad(obj), and the dual cotangents reproduce Jᵀ·dual — in one
        # traversal of the user's graph instead of grad(obj) + a separate
        # constraints VJP.
        return (
            cp.objective(primal),
            cp.constraints_ineq(primal),
            cp.constraints_eq(primal),
        )

    def grad_primal_only(state):
        # Primal partial only: grad(obj) + Jᵀ·dual, via one fused VJP. The
        # objective/residual outputs are produced by the VJP's forward pass and
        # discarded here.
        one = jnp.ones((), state.primal.dtype)
        _, vjp_fn = jax.vjp(lagrangian_map, state.primal)
        (primal_grad,) = vjp_fn((one, state.dual_ineq, state.dual_eq))
        return primal_grad

    def grad_dual_only(state):
        # Dual partials only: the (negated) constraint residuals. A plain
        # forward eval — no objective gradient, no VJP.
        res_ineq, res_eq = constraints(state.primal)
        return -res_ineq, -res_eq

    def grad(state):
        # Full saddle gradient in a single reverse pass: one fused VJP over
        # (objective, constraints_ineq, constraints_eq) seeded with
        # (1.0, dual_ineq, dual_eq). The forward pass yields the objective value
        # (discarded) and the residuals (reused for the dual partials), so the
        # whole saddle gradient costs one VJP instead of grad(obj) + a separate
        # constraints VJP.
        one = jnp.ones((), state.primal.dtype)
        (_, res_ineq, res_eq), vjp_fn = jax.vjp(lagrangian_map, state.primal)
        (primal_grad,) = vjp_fn((one, state.dual_ineq, state.dual_eq))
        return SaddleState(
            primal=primal_grad,
            dual_ineq=-res_ineq,
            dual_eq=-res_eq,
        )

    def opt_update(gradient, opt_state, state):
        return optimiser.update(gradient, opt_state, state)

    def keep_only_primal(updates):
        return SaddleState(
            primal=updates.primal,
            dual_ineq=jnp.zeros_like(updates.dual_ineq),
            dual_eq=jnp.zeros_like(updates.dual_eq),
        )

    def keep_only_dual(updates):
        return SaddleState(
            primal=jnp.zeros_like(updates.primal),
            dual_ineq=updates.dual_ineq,
            dual_eq=updates.dual_eq,
        )

    def scale_by_k(gradient, k):
        # Primal weight split: (1/k, k) on (primal, dual) gradients.
        return SaddleState(
            primal=gradient.primal / k,
            dual_ineq=gradient.dual_ineq * k,
            dual_eq=gradient.dual_eq * k,
        )

    # When k-scaling is on, k is carried as the last element of opt_state. These
    # helpers keep the step bodies agnostic to whether k is packed or not. In
    # adaptive_step mode the packed k-slot is a (k, eta) pair so the
    # per-iteration step size eta rides alongside the primal weight k; otherwise
    # it is just k (or absent when k_scaling is off).
    def unpack_k(opt_state):
        if k_scaling:
            if adaptive_step:
                inner, (k, eta) = opt_state
                return inner, k, eta
            return opt_state
        return opt_state, None

    def pack_k(opt_state, k, eta=None):
        if k_scaling:
            if adaptive_step:
                return (opt_state, (k, eta))
            return (opt_state, k)
        return opt_state

    def run_epoch(
        max_iter,
        start_iter,
        state,
        average_state,
        opt_state,
        total_weight,
        diag=None,
        accum=None,
    ):
        if diag_on:
            t, s = diag
            m_eq = state.dual_eq.shape[0]
            s_eq, s_ineq = s[:m_eq], s[m_eq:]

        # --- Adaptive extragradient step (Malitsky-Tam line search) ---
        # Mirrors the linear solver's adaptive extragradient path. Each
        # iteration takes a raw look-ahead + corrector at base step eta,
        # estimates the local Lipschitz constant from the two gradient
        # evaluations, and shrinks eta below eta_bar = (1/sqrt2)/L_hat if the
        # trial overshot. No optimiser learning rate is consumed here — eta
        # is the only step control.
        if adaptive_step:
            _MT = 1.0 / jnp.sqrt(2.0)

            # Local Lipschitz estimates are taken in the k-weighted geometry
            # the steps live in (tau = eta/k, sigma = eta*k): displacements
            # in the M-norm k‖dx‖² + ‖dy‖²/k, operator differences in the
            # dual M⁻¹-norm ‖dg_x‖²/k + k‖dg_y‖².
            # Under diag_adapt the weights become per-coordinate:
            # T = (eta/k) diag(t), Σ = eta k diag(s).
            def _znorm2(p, de, di, k):
                if diag_on:
                    return k * _dot(p, p / t) + (1.0 / k) * (
                        _dot(de, de / s_eq) + _dot(di, di / s_ineq)
                    )
                return k * _dot(p, p) + (1.0 / k) * (
                    _dot(de, de) + _dot(di, di)
                )

            def _gnorm2(p, de, di, k):
                if diag_on:
                    return (1.0 / k) * _dot(p, p * t) + k * (
                        _dot(de, de * s_eq) + _dot(di, di * s_ineq)
                    )
                return (1.0 / k) * _dot(p, p) + k * (
                    _dot(de, de) + _dot(di, di)
                )

            def _advance_eta(eta_bar, eta0, ip1):
                # Advance eta for the next iterate (growth allowed once the
                # step is accepted). When eta_bar is +inf the step did not
                # move: hold eta rather than letting growth run to NaN.
                eta_next = jnp.minimum(
                    (1.0 - ip1 ** (-0.3)) * eta_bar,
                    (1.0 + ip1 ** (-0.6)) * eta0,
                )
                eta_next = jnp.where(jnp.isfinite(eta_bar), eta_next, eta0)
                eta_next = jnp.where(jnp.isfinite(eta_next), eta_next, eta0)
                return jnp.maximum(eta_next, 1e-12)

            def _line_search(trial, eta, ip1, keep):
                # Branch-free line search: one trial per iteration. If it
                # overshot its admissible bound (eta > eta_bar), the trial
                # is discarded — the outputs fall back to `keep` (the
                # unchanged iterate) — and eta shrinks to just under
                # eta_bar, so the next iteration retries from the same point.
                # A retry loop (lax.while_loop) would force a device->host
                # sync on its predicate every iteration, which dominates the
                # per-iteration cost on GPU for cheap problems (~8x/epoch on
                # isotonic n=10k); a rejection here only costs one iteration.
                out, eta_bar = trial(eta)
                accept = eta <= eta_bar
                out = jax.tree.map(lambda a, b: jnp.where(accept, a, b), out, keep)
                eta_next = jnp.where(
                    accept,
                    _advance_eta(eta_bar, eta, ip1),
                    jnp.maximum(
                        jnp.minimum((1.0 - ip1 ** (-0.3)) * eta_bar, eta), 1e-12
                    ),
                )
                return out, eta_next, accept

            def _accumulate(i, new_state, average_state, total_weight, accept):
                # Rejected trials leave the iterate unchanged and carry zero
                # averaging weight.
                if average:
                    w = jnp.where(accept, weight_function(i), 0.0)
                    total_weight = total_weight + w
                    frac = jnp.where(total_weight > 0, w / total_weight, 0.0)
                    average_state = optax.incremental_update(
                        new_state, average_state, frac
                    )
                return average_state, total_weight

            if update_mode in _LAGGED_MODES:
                # --- Adaptive nonlinear PDHG (cuPDLP-style, as in jl.solve) ---
                #   x+ = P(x - tau ∇ₓL(x, y))
                #   y+ = P(y + sigma (c(x+) + theta (c(x+) - c(x))))
                # With theta = 1 ("pdhg") the dual reads the constraint values
                # extrapolated in c-space, which is the LP solver's A @ x_bar
                # when c is affine. With theta = 0 ("alternating") it reads
                # c(x+): alternating GDA, as the LP solver's "alternating". One VJP
                # (whose forward pass yields c(x)) plus one constraint-only
                # forward per iteration, versus two VJPs for extragradient.
                #
                # The local Lipschitz estimate of the whole saddle operator
                # F = (∇ₓL, -c) needs F(z+), which only the next iteration's VJP
                # produces, so each step is validated one iteration late: if
                # eta_prev · L_hat > _PDHG_RHO the previous step is rolled back
                # (to the carried z_prev, whose F is carried too) and retried
                # with a shrunk eta. Averaging lags by one step accordingly: an
                # iterate joins the average once it has been validated.
                _PDHG_RHO = 1.0

                inner0, (k0, eta0, z_prev0, F_prev0, valid0) = opt_state

                def step(carry, _):
                    (
                        i,
                        state,
                        average_state,
                        eta,
                        z_prev,
                        F_prev,
                        valid,
                        total_weight,
                    ) = carry[:8]
                    ip1 = jnp.asarray(i + 1, eta.dtype)
                    k = k0
                    F = grad(state)

                    # Validate the step z_prev -> state, taken at step eta.
                    dg2 = _gnorm2(
                        F.primal - F_prev.primal,
                        F.dual_eq - F_prev.dual_eq,
                        F.dual_ineq - F_prev.dual_ineq,
                        k,
                    )
                    dz2 = _znorm2(
                        state.primal - z_prev.primal,
                        state.dual_eq - z_prev.dual_eq,
                        state.dual_ineq - z_prev.dual_ineq,
                        k,
                    )
                    eta_bar = jnp.where(
                        dg2 > 0.0, _PDHG_RHO * jnp.sqrt(dz2 / dg2), jnp.inf
                    )
                    accept = (eta <= eta_bar) | ~valid
                    eta_use = jnp.where(
                        valid,
                        jnp.where(
                            accept,
                            _advance_eta(eta_bar, eta, ip1),
                            jnp.maximum(
                                jnp.minimum((1.0 - ip1 ** (-0.3)) * eta_bar, eta),
                                1e-12,
                            ),
                        ),
                        eta,
                    )
                    # A validated step joins the average (the very first
                    # iterate of an epoch after a reset has no step behind it,
                    # and was averaged by the caller as the restart point).
                    average_state, total_weight = _accumulate(
                        i, state, average_state, total_weight, accept & valid
                    )
                    base, F_base = jax.tree.map(
                        lambda a, b: jnp.where(accept, a, b),
                        (state, F),
                        (z_prev, F_prev),
                    )

                    tau = eta_use / k
                    sigma_eq = sigma_ineq = eta_use * k
                    if diag_on:
                        tau = tau * t
                        sigma_eq, sigma_ineq = sigma_eq * s_eq, sigma_ineq * s_ineq
                    x_new = projection_primal(base.primal - tau * F_base.primal)
                    gd_ineq, gd_eq = grad_dual_only(
                        SaddleState(
                            primal=x_new,
                            dual_ineq=base.dual_ineq,
                            dual_eq=base.dual_eq,
                        )
                    )
                    if update_mode == "pdhg":
                        gd_ineq = 2.0 * gd_ineq - F_base.dual_ineq
                        gd_eq = 2.0 * gd_eq - F_base.dual_eq
                    dual_ineq = projection_non_negative(
                        base.dual_ineq - sigma_ineq * gd_ineq
                    )
                    dual_eq = base.dual_eq - sigma_eq * gd_eq
                    cand = SaddleState(
                        primal=x_new, dual_ineq=dual_ineq, dual_eq=dual_eq
                    )
                    out = (
                        i + 1,
                        cand,
                        average_state,
                        eta_use,
                        base,
                        F_base,
                        jnp.asarray(True),
                        total_weight,
                    )
                    if diag_on:
                        # Squared movement of the step validated here,
                        # z_prev -> state, normalised by the eta it ran at
                        # (zero when it was rolled back).
                        H_x, H_y = carry[8]
                        w = jnp.where(accept & valid, 1.0 / eta, 0.0)
                        dx = (state.primal - z_prev.primal) * w
                        dy = (
                            jnp.concatenate(
                                [
                                    state.dual_eq - z_prev.dual_eq,
                                    state.dual_ineq - z_prev.dual_ineq,
                                ]
                            )
                            * w
                        )
                        out = out + ((H_x + dx * dx, H_y + dy * dy),)
                    return out, None

                init = (
                    start_iter,
                    state,
                    average_state,
                    eta0,
                    z_prev0,
                    F_prev0,
                    valid0,
                    total_weight,
                )
                if diag_on:
                    init = init + (accum,)
                out, _ = jax.lax.scan(step, init, None, length=max_iter)
                (
                    i,
                    state,
                    average_state,
                    eta,
                    z_prev,
                    F_prev,
                    valid,
                    total_weight,
                ) = out[:8]
                opt_state = (inner0, (k0, eta, z_prev, F_prev, valid))
                return (i, state, average_state, opt_state, total_weight) + out[8:]

            def _trial(eta, state, k, g):
                tau = eta / k
                sigma_eq = sigma_ineq = eta * k
                if diag_on:
                    tau = tau * t
                    sigma_eq, sigma_ineq = sigma_eq * s_eq, sigma_ineq * s_ineq
                xh = projection_primal(state.primal - tau * g.primal)
                yh_ineq = projection_non_negative(
                    state.dual_ineq - sigma_ineq * g.dual_ineq
                )
                yh_eq = state.dual_eq - sigma_eq * g.dual_eq
                state_half = SaddleState(primal=xh, dual_ineq=yh_ineq, dual_eq=yh_eq)
                g_half = grad(state_half)
                # Corrector from the ORIGINAL state using the look-ahead grad.
                x_new = projection_primal(state.primal - tau * g_half.primal)
                dual_ineq = projection_non_negative(
                    state.dual_ineq - sigma_ineq * g_half.dual_ineq
                )
                dual_eq = state.dual_eq - sigma_eq * g_half.dual_eq
                cand = SaddleState(primal=x_new, dual_ineq=dual_ineq, dual_eq=dual_eq)

                # Local Lipschitz estimate, measured on the look-ahead
                # displacement.
                dg2 = _gnorm2(
                    g_half.primal - g.primal,
                    g_half.dual_eq - g.dual_eq,
                    g_half.dual_ineq - g.dual_ineq,
                    k,
                )
                dz2 = _znorm2(
                    xh - state.primal,
                    yh_eq - state.dual_eq,
                    yh_ineq - state.dual_ineq,
                    k,
                )
                eta_bar = jnp.where(dg2 > 0.0, _MT * jnp.sqrt(dz2 / dg2), jnp.inf)
                return cand, eta_bar

            def step(carry, _):
                i, state, average_state, opt_state, total_weight = carry[:5]
                opt_state, k, eta = unpack_k(opt_state)
                ip1 = jnp.asarray(i + 1, eta.dtype)

                g = grad(state)
                new_state, eta_next, accept = _line_search(
                    lambda eta_t: _trial(eta_t, state, k, g), eta, ip1, state
                )
                opt_state = pack_k(opt_state, k, eta_next)
                average_state, total_weight = _accumulate(
                    i, new_state, average_state, total_weight, accept
                )
                out = (i + 1, new_state, average_state, opt_state, total_weight)

                if diag_on:
                    # Squared movement of the accepted step (zero when the
                    # line search rejected it), normalised by the eta it ran at.
                    H_x, H_y = carry[5]
                    inv_eta = 1.0 / eta
                    dx = (new_state.primal - state.primal) * inv_eta
                    dy = (
                        jnp.concatenate(
                            [
                                new_state.dual_eq - state.dual_eq,
                                new_state.dual_ineq - state.dual_ineq,
                            ]
                        )
                        * inv_eta
                    )
                    out = out + ((H_x + dx * dx, H_y + dy * dy),)
                return out, None

            init = (start_iter, state, average_state, opt_state, total_weight)
            if diag_on:
                init = init + (accum,)
            out, _ = jax.lax.scan(step, init, None, length=max_iter)
            return out

        def step(carry, _):
            (
                i,
                state,
                average_state,
                opt_state,
                total_weight,
            ) = carry

            opt_state, k = unpack_k(opt_state)

            if update_mode == "alternating":
                # Only the primal half of the start gradient and the dual
                # half of the post-primal gradient are ever consumed, so
                # compute just those: a VJP+obj-grad for the primal, and a
                # plain residual eval for the dual (no VJP, no obj grad).
                grad_primal_start = grad_primal_only(state)
                if k_scaling:
                    grad_primal_start = grad_primal_start / k

                primal_gradient = SaddleState(
                    primal=grad_primal_start,
                    dual_ineq=jnp.zeros_like(state.dual_ineq),
                    dual_eq=jnp.zeros_like(state.dual_eq),
                )
                primal_updates, _ = opt_update(
                    primal_gradient,
                    opt_state,
                    state,
                )
                primal_updates = keep_only_primal(primal_updates)
                state = optax.apply_updates(state, primal_updates)
                state = SaddleState(
                    primal=projection_primal(state.primal),
                    dual_ineq=state.dual_ineq,
                    dual_eq=state.dual_eq,
                )

                dual_ineq_g, dual_eq_g = grad_dual_only(state)
                if k_scaling:
                    dual_ineq_g = dual_ineq_g * k
                    dual_eq_g = dual_eq_g * k
                combined_gradient = SaddleState(
                    primal=grad_primal_start,
                    dual_ineq=dual_ineq_g,
                    dual_eq=dual_eq_g,
                )
                combined_updates, opt_state = opt_update(
                    combined_gradient,
                    opt_state,
                    state,
                )
                dual_updates = keep_only_dual(combined_updates)
                state = optax.apply_updates(state, dual_updates)
                state = SaddleState(
                    primal=state.primal,
                    dual_ineq=projection_non_negative(state.dual_ineq),
                    dual_eq=state.dual_eq,
                )
            else:
                # extragradient
                # --- Look-ahead gradient ---
                g = grad(state)

                scaled_g = scale_by_k(g, k) if k_scaling else g
                la_updates, _ = opt_update(scaled_g, opt_state, state)
                state_half = optax.apply_updates(state, la_updates)
                state_half = SaddleState(
                    primal=projection_primal(state_half.primal),
                    dual_ineq=projection_non_negative(state_half.dual_ineq),
                    dual_eq=state_half.dual_eq,
                )

                # Corrector: gradient at look-ahead point, applied from original state
                g_half = grad(state_half)
                scaled_g_half = scale_by_k(g_half, k) if k_scaling else g_half
                corr_updates, opt_state = opt_update(scaled_g_half, opt_state, state)
                state = optax.apply_updates(state, corr_updates)
                state = SaddleState(
                    primal=projection_primal(state.primal),
                    dual_ineq=projection_non_negative(state.dual_ineq),
                    dual_eq=state.dual_eq,
                )

            opt_state = pack_k(opt_state, k)

            # `average` is a Python-static bool, so branch on it at trace
            # time rather than threading a per-step lax.cond (which would
            # force XLA to evaluate the predicate and the running-mean AXPY
            # every iteration even when averaging is off — the same class of
            # issue fixed at the epoch level for the metrics path).
            if average:
                w = weight_function(i)
                total_weight = total_weight + w
                average_state = optax.incremental_update(
                    state, average_state, w / total_weight
                )

            return (
                i + 1,
                state,
                average_state,
                opt_state,
                total_weight,
            ), None

        (i, state, average_state, opt_state, total_weight), _ = jax.lax.scan(
            step,
            (start_iter, state, average_state, opt_state, total_weight),
            None,
            length=max_iter,
        )

        return i, state, average_state, opt_state, total_weight

    return run_epoch


def _check_settings(optimiser, adaptive_eta, step_size, update_mode, k_scale):
    """Resolve and validate the settings ``solve()`` and ``make_solver`` share.
    Returns ``(optimiser, step_size, adaptive_step, k_scaling, k_lo, k_hi)``."""
    # adaptive_eta switches the line search; step_size is the starting eta
    # either way (the fixed step's learning rate when it is off).
    if adaptive_eta is not None and not isinstance(adaptive_eta, (bool, np.bool_)):
        raise ValueError(
            f"adaptive_eta must be True, False or None, got {adaptive_eta!r}; "
            "pass the starting step as step_size=..."
        )
    if isinstance(step_size, str) or not step_size > 0.0:
        raise ValueError(f"step_size must be a float > 0, got {step_size!r}")
    adaptive_step = bool(adaptive_eta)

    # The adaptive step replaces the optimiser's update, so an explicit
    # optimiser would be silently ignored.
    if optimiser is not None and adaptive_step:
        raise ValueError(
            "optimiser is used only by the fixed-step scheme: pass adaptive_eta=False"
        )
    if optimiser is None:
        optimiser = jo.gd(step_size)

    valid_update_modes = ["alternating", "extragradient", "pdhg"]
    if update_mode not in valid_update_modes:
        raise ValueError(f"update_mode must be one of {valid_update_modes}")

    # Per-iteration adaptive step size (Malitsky-Tam line search). Every mode
    # supports it; it needs the primal weight k, hence k_scale. pdhg has no
    # fixed-step (optimiser) form here, so it requires the line search.
    if adaptive_step:
        if update_mode not in _ADAPTIVE_MODES:
            raise ValueError(
                f"adaptive_eta is only supported with update_mode in {_ADAPTIVE_MODES}"
            )
        if k_scale is None:
            raise ValueError("adaptive_eta requires k_scale (primal weight k)")
    elif update_mode == "pdhg":
        raise ValueError("update_mode='pdhg' requires adaptive_eta=True")

    # ``k_scale`` is the public knob for primal-weight scaling: ``None`` disables
    # it, otherwise it sets a symmetric clamp band ``[1/k_scale, k_scale]``.
    k_scaling = k_scale is not None
    if k_scaling:
        k_lo, k_hi = 1.0 / k_scale, k_scale
    else:
        k_lo, k_hi = None, None
    return optimiser, step_size, adaptive_step, k_scaling, k_lo, k_hi


def _device_solver(
    cp,
    *,
    optimiser,
    initial_solution,
    initial_opt_state,
    iterations_per_epoch,
    primal_grad_norm_tolerance,
    primal_feasibility_tolerance,
    complementarity_slack_tolerance,
    weight_function,
    verbose,
    log_every,
    average,
    report_best,
    update_mode,
    k_scaling,
    k_lo,
    k_hi,
    k_theta,
    k_init,
    eta_init,
    adaptive_step,
    restarts,
    epochs_per_restart,
    restart_multiplier,
    restart_decay,
    iterations_per_epoch_decay,
    iterations_per_epoch_min,
    diag_adapt=False,
    diag_theta=0.5,
    diag_clamp=10.0,
):
    """The device side of ``solve()`` for one problem: the jitted epoch loop
    (iterations, metrics, convergence test, restarts and the primal weight)
    and its initial state. ``cp``'s functions may close over traced values, so
    this also runs inside ``jit`` / ``vmap`` / ``grad`` (``make_solver``).

    Returns ``(build_chunk, carry)``: ``build_chunk(ipe)`` gives the jitted
    ``run_chunk(carry, n_epochs)`` for an epoch length, and ``carry`` is the
    initial loop state.
    """
    # The diagonal is learned over a restart cycle and changed only at a
    # restart, and only the adaptive line search measures in its metric, so
    # without restarts or with the fixed-step optimiser it does nothing (as in
    # jaddle_linear.solve): skip its accumulators.
    diag_adapt = bool(diag_adapt and adaptive_step and restarts)

    def langrangian(state):
        return (
            cp.objective(state.primal)
            + state.dual_ineq @ cp.constraints_ineq(state.primal)
            + state.dual_eq @ cp.constraints_eq(state.primal)
        )

    def grad(state):
        gradient = jax.grad(langrangian)(state)
        return SaddleState(
            primal=gradient.primal,
            dual_ineq=gradient.dual_ineq,
            dual_eq=gradient.dual_eq,
        )

    has_dual_bound = cp.dual_bound is not None

    @jax.jit
    def compute_epoch_metrics(average_state):
        (
            objective_value,
            primal_grad_norm,
            constraint_bound,
            complementarity_slack,
        ) = _kkt_residuals(cp, average_state)

        if getattr(cp, "residuals", None) is not None:
            primal_grad_norm, constraint_bound, complementarity_slack = cp.residuals(
                average_state
            )

        if has_dual_bound:
            dual_bound = cp.dual_bound(
                average_state.dual_ineq,
                average_state.dual_eq,
            )
            duality_gap = objective_value - dual_bound
        else:
            duality_gap = jnp.nan
        dual_gap_is_finite = has_dual_bound & jnp.isfinite(duality_gap)

        return (
            objective_value,
            primal_grad_norm,
            complementarity_slack,
            constraint_bound,
            duality_gap,
            dual_gap_is_finite,
        )

    def check_convergence(
        primal_grad_norm,
        complementarity_slack,
        constraint_bound,
    ):
        # True while NOT yet converged. Written as the negation of "all within
        # tolerance" so that a NaN residual (a diverged solve) never certifies.
        return ~(
            (primal_grad_norm <= primal_grad_norm_tolerance)
            & (complementarity_slack <= complementarity_slack_tolerance)
            & (constraint_bound <= primal_feasibility_tolerance)
        )

    if initial_solution is None:
        initial_solution = cp.initial_solution()

    # PDLP-style primal-weight initialisation. When k-scaling is on and k_init is
    # left as None we derive it from the objective/RHS norms ||c|| / ||b||, which
    # puts the primal/dual step ratio in the right order of magnitude before
    # iteration 1 instead of starting symmetric. For a linear objective
    # c = grad(obj)(0), and for constraints of the form Ax - b, evaluating at
    # x = 0 gives -b, so b = -[c_eq(0); c_ineq(0)]. This is generic over
    # JaddleCP/LP/JaddleLP since it only uses the objective/constraint callables.
    if k_scaling and k_init is None:
        zero = jnp.zeros_like(initial_solution.primal)

        # Fuse into one JIT-compiled call: objective forward+grad and both
        # constraint evaluations in a single function to reduce dispatch overhead.
        def _k_init_fn(x):
            obj, c = jax.value_and_grad(cp.objective)(x)
            b = jnp.concatenate([cp.constraints_eq(x), cp.constraints_ineq(x)])
            return c, b

        c, b = jax.jit(_k_init_fn)(zero)
        norm_c2 = _dot(c, c)
        norm_b2 = _dot(b, b)
        # Kept on the device (no float()) so this runs under jit / vmap. As in
        # PDLP, a zero norm (e.g. constraints with no constant term, b = 0)
        # gives k = 1: the ratio would sit at the clamp, which under the wide
        # default k_scale stalled isotonic regression.
        ratio = jnp.sqrt(norm_c2 / jnp.where(norm_b2 > 0, norm_b2, 1.0))
        k_init = jnp.where(
            (norm_c2 > 0) & (norm_b2 > 0), jnp.clip(ratio, k_lo, k_hi), 1.0
        )
    elif k_init is None:
        k_init = 1.0

    if initial_opt_state is not None:
        opt_state = initial_opt_state
    elif k_scaling:
        k_slot = _init_k_slot(
            k_init,
            eta_init if adaptive_step else None,
            initial_solution,
            update_mode,
        )
        opt_state = (optimiser.init(initial_solution), k_slot)
    else:
        opt_state = optimiser.init(initial_solution)

    def kkt_merit(primal_grad_norm, complementarity_slack, constraint_bound):
        # Normalised KKT merit for the restart trigger: the maximum of the three
        # residuals. It is only compared with itself at successive restarts, so
        # a fixed normaliser of 1 is enough.
        return jnp.maximum(
            jnp.maximum(primal_grad_norm, complementarity_slack),
            constraint_bound,
        )

    def _k_of(opt_state):
        # The k-slot is a tuple led by k in adaptive_step mode, plain k otherwise.
        return opt_state[1][0] if adaptive_step else opt_state[1]

    def _rebalance_k(new_state, ref_state, k_prev, diag=None):
        # PDLP-style primal-weight rebalance: drive k from the primal-vs-dual
        # *movement* over the just-finished cycle (distance between iterates),
        # not per-step gradient norms. omega = ||dy|| / ||dx|| under tau = eta/k,
        # sigma = eta*k (matches the linear solver), blended with the current
        # weight in log space (k_theta), then clamped. Squared norms avoid two
        # sqrts; the ratio is preserved.
        dp = new_state.primal - ref_state.primal
        dd = jnp.concatenate(
            [
                new_state.dual_eq - ref_state.dual_eq,
                new_state.dual_ineq - ref_state.dual_ineq,
            ]
        )
        if diag is None:
            move_p2 = _dot(dp, dp)
            move_d2 = _dot(dd, dd)
        else:
            # diag_adapt: movement in the metric the cycle ran under.
            t, s = diag
            move_p2 = _dot(dp, dp / t)
            move_d2 = _dot(dd, dd / s)
        # If either side did not move, the ratio carries no information
        # (0/0 = NaN, or 0 / inf): keep k. (A `+ 1e-60` guard only works in
        # float64; it is 0 in float32 and float16.)
        moved = (move_p2 > 0) & (move_d2 > 0) & jnp.isfinite(move_p2 + move_d2)
        k_target = jnp.where(
            moved, jnp.sqrt(move_d2 / jnp.where(moved, move_p2, 1.0)), k_prev
        )
        log_k = k_theta * jnp.log(k_target) + (1.0 - k_theta) * jnp.log(k_prev)
        return jnp.clip(jnp.exp(log_k), k_lo, k_hi)

    def _select(pred, a, b):
        return jax.tree.map(lambda x, y: jnp.where(pred, x, y), a, b)

    def _strong(tree):
        # Strip JAX's weak typing (from Python-scalar inputs like jnp.asarray(0))
        # so the carry handed to run_chunk always has exactly the types run_chunk
        # returns; otherwise the second call retraces and recompiles the chunk.
        return jax.tree.map(
            lambda x: jax.lax.convert_element_type(x, jnp.asarray(x).dtype), tree
        )

    # --- Device-resident epoch loop -------------------------------------------
    # Epochs run in chunks: one jitted lax.while_loop (`run_chunk`) executes many
    # epochs back to back on the device -- the iterations, the end-of-epoch
    # metrics, the convergence test and the restart / primal-weight logic -- so
    # the host never waits on the device between epochs. Python regains control
    # only between chunks, to enforce max_epochs / max_seconds, print the
    # buffered epoch log and restart messages, and catch Ctrl-C. A chunk also
    # ends early when a restart decays `iterations_per_epoch` (the epoch length
    # is a static scan length, so the next chunk is compiled for the new length)
    # or when the verbose log buffers fill.
    _metric_shapes = jax.eval_shape(compute_epoch_metrics, initial_solution)
    _merit_dtype = _metric_shapes[1].dtype

    def _initial_metrics():
        # +inf everywhere (and gap finiteness False) so the convergence test
        # can't pass before the first epoch has computed real metrics.
        return tuple(
            (
                jnp.zeros(m.shape, m.dtype)
                if m.dtype == jnp.bool_
                else jnp.full(m.shape, jnp.inf, m.dtype)
            )
            for m in _metric_shapes
        )

    def _build_chunk(ipe):
        run_epoch = _make_epoch_fn(
            cp,
            optimiser,
            weight_function,
            average,
            update_mode,
            k_scaling,
            adaptive_step,
            diag=diag_adapt,
        )
        ipe_after_restart = max(
            iterations_per_epoch_min, int(ipe * iterations_per_epoch_decay)
        )

        def epoch(c):
            # The epoch runs on the restart-shifted index, so a restart re-zeros
            # the iteration counter the step schedules and weights see.
            start = c["i"] - c["restart_i_offset"]
            diag, accum = c["diag"], c["accum"]
            epoch_out = run_epoch(
                ipe,
                start,
                c["state"],
                c["avg"],
                c["opt"],
                c["total_weight"],
                **(dict(diag=diag, accum=accum) if diag_adapt else {}),
            )
            if diag_adapt:
                *epoch_out, accum = epoch_out
            shifted_i, state, avg, opt, total_weight = epoch_out
            i = shifted_i + c["restart_i_offset"]
            count = c["count"] + 1

            metrics = compute_epoch_metrics(avg if average else state)
            # `used_avg` tracks which point the metrics describe.
            used_avg = jnp.asarray(bool(average))
            # report_best: when averaging is on, the average and the last iterate
            # are different points and either can be the better solution
            # (averaging stabilises rotational problems but lags the last iterate
            # when it is already contracting). Report/converge on whichever has
            # the lower KKT merit, at the cost of a second metrics pass per epoch.
            if report_best and average:
                state_metrics = compute_epoch_metrics(state)
                iterate_better = kkt_merit(*state_metrics[1:4]) < kkt_merit(
                    *metrics[1:4]
                )
                metrics = _select(iterate_better, state_metrics, metrics)
                used_avg = ~iterate_better
            (
                objective_value,
                primal_grad_norm,
                complementarity_slack,
                constraint_bound,
            ) = metrics[:4]

            log, n_log, evt, n_evt = c["log"], c["n_log"], c["evt"], c["n_evt"]
            if verbose:
                write = (count == 1) | (count % log_every == 0)
                row = jnp.stack(
                    [
                        count,
                        objective_value,
                        primal_grad_norm,
                        complementarity_slack,
                        constraint_bound,
                    ]
                ).astype(log.dtype)
                log = jnp.where(write, log.at[n_log].set(row), log)
                n_log = n_log + write

            # --- Adaptive restart decision ---
            restart_i_offset = c["restart_i_offset"]
            at_restart = c["state_at_last_restart"]
            mal = c["merit_at_last_restart"]
            epochs_since = c["epochs_since_restart"]
            cycle_cap = c["cycle_cap"]
            ipe_next = c["ipe"]
            if restarts:
                epochs_since = epochs_since + 1
                merit = kkt_merit(
                    primal_grad_norm, complementarity_slack, constraint_bound
                )

                # Two-point restart: pick the better of average and iterate.
                # With report_best the metrics already describe the better one.
                restart_used_avg = used_avg
                restart_point = _select(used_avg, avg, state) if average else state
                restart_merit = merit
                if average and not report_best:
                    st = compute_epoch_metrics(state)
                    state_merit = kkt_merit(st[1], st[2], st[3])
                    iterate_better = state_merit < merit
                    restart_point = _select(iterate_better, state, restart_point)
                    restart_merit = jnp.where(iterate_better, state_merit, merit)
                    restart_used_avg = ~iterate_better

                mal = jnp.where(jnp.isfinite(mal), mal, restart_merit)
                sufficient_progress = restart_merit <= restart_decay * mal
                cycle_exhausted = epochs_since >= cycle_cap
                restarted = sufficient_progress | cycle_exhausted

                if k_scaling:
                    k_new = _rebalance_k(restart_point, at_restart, _k_of(opt), diag)
                    # Carry the learned step size across the restart (PDLP
                    # convention, as in the linear solver): re-seeding forced
                    # the adaptive rule to re-climb from eta_init after
                    # every restart.
                    k_slot = _init_k_slot(
                        k_new,
                        opt[1][1] if adaptive_step else None,
                        restart_point,
                        update_mode,
                    )
                    opt_restart = (optimiser.init(restart_point), k_slot)
                else:
                    k_new = jnp.asarray(jnp.nan, _merit_dtype)
                    opt_restart = optimiser.init(restart_point)

                # Warm-start from the restart point; reset the optimiser state,
                # averaging, the weight accumulator and the iteration offset.
                opt = _select(restarted, opt_restart, opt)
                if diag_adapt:
                    # After k, which is judged in the metric the cycle ran under.
                    diag_new = jo.rebalance_diag(
                        restart_point, at_restart, diag, accum, diag_theta, diag_clamp
                    )
                    diag = _select(restarted, diag_new, diag)
                    accum = _select(
                        restarted, jax.tree.map(jnp.zeros_like, accum), accum
                    )
                state = _select(restarted, restart_point, state)
                if average:
                    avg = _select(restarted, restart_point, avg)
                at_restart = _select(restarted, restart_point, at_restart)
                total_weight = jnp.where(restarted, 0.0, total_weight)
                restart_i_offset = jnp.where(restarted, i - 1, restart_i_offset)
                mal = jnp.where(restarted, restart_merit, mal)
                epochs_since = jnp.where(restarted, 0, epochs_since)
                cycle_cap = jnp.where(
                    restarted, cycle_cap * restart_multiplier, cycle_cap
                )
                ipe_next = jnp.where(restarted, ipe_after_restart, ipe_next)
                if verbose:
                    row = jnp.stack(
                        [
                            count,
                            jnp.where(sufficient_progress, 0, 1),
                            restart_merit,
                            restart_used_avg,
                            k_new,
                            cycle_cap,
                            ipe_next,
                        ]
                        + (
                            [jo.diag_log_spread(diag[0]), jo.diag_log_spread(diag[1])]
                            if diag_adapt
                            else []
                        )
                    ).astype(evt.dtype)
                    evt = jnp.where(restarted, evt.at[n_evt].set(row), evt)
                    n_evt = n_evt + restarted

            new = dict(
                state=state,
                avg=avg,
                used_avg=used_avg,
                opt=opt,
                diag=diag,
                accum=accum,
                state_at_last_restart=at_restart,
                metrics=metrics,
                count=count,
                i=i,
                restart_i_offset=restart_i_offset,
                total_weight=total_weight,
                merit_at_last_restart=mal,
                epochs_since_restart=epochs_since,
                cycle_cap=cycle_cap,
                ipe=ipe_next,
                done=~check_convergence(
                    primal_grad_norm, complementarity_slack, constraint_bound
                ),
                log=log,
                n_log=n_log,
                evt=evt,
                n_evt=n_evt,
            )
            # Keep the loop carry type-stable.
            return jax.tree.map(
                lambda n, o: jnp.asarray(n).astype(jnp.asarray(o).dtype), new, c
            )

        @jax.jit
        def run_chunk(c, n_epochs):
            end = c["count"] + n_epochs

            def cond(c):
                go = (~c["done"]) & (c["count"] < end) & (c["ipe"] == ipe)
                if verbose:
                    go = go & (c["n_log"] < _LOG_CAP) & (c["n_evt"] < _EVT_CAP)
                return go

            return jax.lax.while_loop(cond, epoch, c)

        return run_chunk

    carry = _strong(
        dict(
            state=initial_solution,
            # No average buffer when averaging is off (saves a state-sized copy).
            avg=initial_solution if average else None,
            used_avg=jnp.asarray(bool(average)),
            opt=opt_state,
            # diag_adapt: the diagonal metric (t, s), dual stacked [eq; ineq],
            # and its per-cycle accumulators (H_x, H_y).
            diag=(
                (
                    jnp.ones_like(initial_solution.primal),
                    jnp.ones_like(_dual_stack(initial_solution)),
                )
                if diag_adapt
                else None
            ),
            accum=(
                (
                    jnp.zeros_like(initial_solution.primal),
                    jnp.zeros_like(_dual_stack(initial_solution)),
                )
                if diag_adapt
                else None
            ),
            state_at_last_restart=initial_solution,
            metrics=_initial_metrics(),
            count=jnp.asarray(0),
            i=jnp.asarray(1),
            restart_i_offset=jnp.asarray(0),
            total_weight=jnp.asarray(0.0),
            merit_at_last_restart=jnp.asarray(jnp.inf, _merit_dtype),
            epochs_since_restart=jnp.asarray(0),
            cycle_cap=jnp.asarray(
                jnp.inf if epochs_per_restart is None else float(epochs_per_restart)
            ),
            ipe=jnp.asarray(iterations_per_epoch),
            done=jnp.asarray(False),
            log=jnp.zeros((_LOG_CAP, 5), _merit_dtype) if verbose else None,
            n_log=jnp.asarray(0),
            evt=(
                jnp.zeros((_EVT_CAP, 9 if diag_adapt else 7), _merit_dtype)
                if verbose
                else None
            ),
            n_evt=jnp.asarray(0),
        )
    )
    return _build_chunk, carry


def solve(
    cp: JaddleCP,
    optimiser=None,
    max_epochs=None,
    max_seconds=None,
    initial_solution=None,
    initial_opt_state=None,
    iterations_per_epoch=256,
    primal_grad_norm_tolerance=1e-2,
    primal_feasibility_tolerance=1e-3,
    complementarity_slack_tolerance=1e-3,
    weight_function=lambda _: 1.0,
    verbose=False,
    log_every=1,
    average=True,
    report_best=True,
    update_mode="alternating",
    k_scale=100.0,
    k_theta=0.5,
    k_init=None,
    adaptive_eta=True,
    step_size=1.0,
    restarts=True,
    epochs_per_restart=None,
    restart_multiplier=1.0,
    restart_decay=0.2,
    iterations_per_epoch_decay=1.0,
    iterations_per_epoch_min=100,
    diag_adapt=True,
    diag_theta=0.5,
    diag_clamp=10.0,
    gradient_scaling=True,
    gradient_scaling_probes=16,
    precompile=True,
):
    """
    Solve a convex saddle-point problem via saddle-point optimisation.

    Runs primal descent / dual ascent on the Lagrangian
    ``f(x) + λᵀg(x) + μᵀh(x)`` (``λ >= 0``), with the primal projected onto the
    box ``[lower_bounds, upper_bounds]``; all gradients come from ``jax.grad``.
    Iterations run in compiled epochs of ``iterations_per_epoch`` steps. After
    each epoch the convergence test is evaluated, and the solve stops once all
    three of the following are within tolerance:

    * stationarity, ``‖x - proj(x - ∇ₓL)‖∞ <= primal_grad_norm_tolerance``;
    * primal feasibility, the largest violation
      ``max(g(x)⁺, |h(x)|) <= primal_feasibility_tolerance``;
    * complementary slackness,
      ``max|λ ⊙ g(x)| / (1 + |f(x)|) <= complementarity_slack_tolerance``.

    The epoch loop runs on the device: each call into JAX executes a chunk of
    many epochs, including the metrics, convergence test and restart logic, with
    no host synchronisation between them. Python regains control between chunks
    (sized to take about a second) to enforce ``max_epochs`` / ``max_seconds``,
    print the verbose log and handle Ctrl-C; with ``verbose=True`` the per-epoch
    ``Time`` is the average over the epoch's chunk. When ``verbose``,
    ``max_epochs`` and ``max_seconds`` are all unset, the whole solve runs as a
    single device call (only an ``iterations_per_epoch_decay`` restart, which
    changes the compiled epoch length, returns to Python); Ctrl-C then takes
    effect only once the call returns.

    Args:
        optimiser: Optax ``GradientTransformation`` for the primal and dual
            players (build one with
            ``jaddle_optimisers.create_saddle_optimiser``, or use a ready-made
            one such as ``jo.gd`` or ``jo.optimistic_gd``). It drives the steps
            of ``"alternating"`` and ``"extragradient"`` when the adaptive step
            is off, so passing one requires ``adaptive_eta=False``. ``None``
            (default) uses plain gradient descent, ``jo.gd(step_size)``, for
            the fixed step.
        max_epochs: Epoch budget (default ``None`` = no limit). The solve stops
            with ``stop_reason="max_epochs"`` once it is reached.
        max_seconds: Wall-clock budget in seconds (default ``None`` = no limit).
            Measured from entry into ``solve()``, so setup and ``precompile``
            count against it. Checked between chunks of epochs, and each chunk is
            sized from the measured epoch time to fit the remaining budget; once
            the budget is spent the current point is returned with
            ``stop_reason="time_limit"``. The solve can overrun by about one
            epoch (shrink ``iterations_per_epoch`` for a tighter cutoff).
        initial_solution: Starting ``SaddleState`` (default ``None`` = zeros
            for the primal and both duals), e.g. the ``"solution"`` of a
            previous solve.
        initial_opt_state: Starting step-size / optimiser state (default
            ``None`` = fresh), e.g. the ``"opt_state"`` of a previous solve.
            Pass it together with ``initial_solution`` to resume a solve.
        iterations_per_epoch: Iterations per compiled epoch (default 256, as
            in ``jaddle_linear.solve``).
            Convergence, budgets, logging and restarts are checked between
            epochs, so smaller values react faster at some per-epoch cost.
        primal_grad_norm_tolerance: Stationarity tolerance (default 1e-2); see
            the convergence test above.
        primal_feasibility_tolerance: Constraint-violation tolerance (default
            1e-3).
        complementarity_slack_tolerance: Relative complementary-slackness
            tolerance (default 1e-3).
        weight_function: Weight ``w(i)`` given to iterate ``i`` in the running
            average (default uniform, ``lambda _: 1.0``; see also
            ``jaddle_optimisers.tail_average``). Only used when
            ``average=True``.
        verbose: Print per-epoch progress (default ``False``).
        log_every: With ``verbose``, print every this many epochs (default 1).
        average: Keep a weighted running average of the iterates (default
            ``True``) and report / test convergence on it (see
            ``report_best``); restarts then resume from whichever of the average
            and the last iterate has the better merit. Averaging helps
            non-contractive schemes such as ``"alternating"``; the adaptive
            schemes converge in the last iterate.
        report_best: With ``average``, test convergence on and return
            whichever of the average and the last iterate has the lower KKT
            merit, re-chosen every epoch (default ``True``), so averaging never
            holds back a last iterate that is already converging. Costs a second
            metrics evaluation per epoch. ``False`` always uses the average.
        update_mode: Selects the stepping scheme (single source of truth):
            ``"alternating"`` (default; primal step, then dual step at the new
            primal: with ``adaptive_eta`` it is ``"pdhg"`` without the
            extrapolation, the dual reading ``c(x+)``, and with
            ``adaptive_eta=False`` the steps come from ``optimiser``),
            ``"extragradient"`` (Korpelevich
            two-call) or ``"pdhg"`` (cuPDLP-style PDHG with the dual reading extrapolated
            constraint values ``2 c(x+) - c(x)``: one VJP plus one
            constraint-only evaluation per iteration; requires
            ``adaptive_eta=True``, and its line search validates each step one
            iteration late, rolling it back if it overshot).
        k_scale: Primal-weight (k) scaling control. ``None`` disables it;
            otherwise a float sets a symmetric clamp band
            ``[1/k_scale, k_scale]`` for ``k`` (default ``100`` →
            ``[0.01, 100]``; ``jaddle_linear.solve`` uses ``1e8``, but here
            ``k`` swinging to 1e-7 drove the adaptive step to its floor, where
            it stayed across restarts). When enabled — orthogonal to ``update_mode``, so
            it composes with every scheme — a primal weight ``k`` rescales the
            primal/dual gradients by ``(1/k, k)`` before each ``opt_update``,
            making the dual/primal step ratio ``k**2``. ``k`` is initialised
            from ``k_init`` and rebalanced at each restart (PDLP-style) from
            primal-vs-dual iterate movement; it is constant within an epoch
            (not adapted per iteration). Tuned by ``k_theta``/``k_scale`` and
            ``k_init``.
        k_theta: Smoothing coefficient for the log-space primal-weight update at
            each restart (default 0.5 = geometric mean of the movement-based
            target and the current weight, matching PDLP). Smaller = slower
            adaptation. Only used when ``k_scale`` is set.
        k_init: Initial primal weight ``k``. ``None`` (default) initialises it to
            the PDLP heuristic ``||c|| / ||b||`` (objective vs RHS norms), where
            ``c = grad(objective)(0)`` and ``b = -[c_eq(0); c_ineq(0)]``, or
            ``1.0`` when either is zero (as in PDLP). Pass a
            float to override (``1.0`` = symmetric steps). Only used when
            ``k_scale`` is set.
        adaptive_eta: Switches on a per-iteration adaptive step size with a
            Malitsky-Tam local-Lipschitz line search (default ``True``, as in
            ``jaddle_linear.solve``), started from ``step_size``. ``False`` (or
            ``None``) uses the fixed step of ``optimiser`` instead (``pdhg``
            requires the line search). The adaptive step is a single scalar
            base step ``eta`` driving the primal step ``tau = eta / k`` and dual
            step ``sigma = eta * k``. The extragradient look-ahead and corrector
            already evaluate the gradient twice, so the local Lipschitz estimate
            ``L_hat = ‖g_half - g‖_w / ‖z_half - z‖_w`` (the k-weighted norm)
            comes free; the step is admissible while ``eta · L_hat <= 1/sqrt2``,
            and ``eta`` is rejected + shrunk if the trial overshot, then advanced
            with a two-sided guard. ``"pdhg"`` and ``"alternating"`` estimate
            ``L_hat`` from consecutive iterations instead, so each step is
            validated one iteration late and rolled back if it overshot.
            Requires ``k_scale`` (the primal weight k). The learned ``eta`` is
            carried across restarts.
        step_size: The starting step ``eta`` (default ``1.0``, as in
            ``jaddle_linear.solve``). With ``adaptive_eta`` it seeds the line
            search, which corrects a poor value within a few iterations;
            without it, it is the learning rate of the default optimiser
            ``jo.gd(step_size)`` (with ``k_scale`` set, again ``tau = eta / k``
            and ``sigma = eta * k``). An explicit ``optimiser`` carries its own
            learning rate and ignores it.
        restarts: Enable adaptive warm restarts (default ``True``, as in
            ``jaddle_linear.solve``). There is
            no cap on how many fire; the triggers alone decide. Each restart
            resets the optimiser momentum and averaging while keeping the
            current iterate as a warm start. A restart fires when the normalised
            KKT merit drops below ``restart_decay`` × the merit at the last
            restart (sufficient-progress restart) or the cycle-length cap is
            exhausted (no-progress restart).
        epochs_per_restart: Length cap (epochs) of the first restart cycle, or
            ``None`` (default) for no cap — restarts then fire only on the
            sufficient-progress trigger. Subsequent caps grow by
            ``restart_multiplier``.
        restart_multiplier: Geometric growth factor for cycle-length caps
            (default 1.0 = fixed length, 2.0 = doubling).
        restart_decay: Sufficient-progress threshold (default 0.2). A restart
            fires early when the merit drops below this fraction of the merit at
            the last restart.
        iterations_per_epoch_decay: Multiplicative decay applied to
            ``iterations_per_epoch`` after each restart (default 1.0 = no
            decay). Values < 1 shrink the epoch length at each restart to spend
            more time checking convergence.
        iterations_per_epoch_min: Floor for the decayed epoch length (default
            100). Only used when ``iterations_per_epoch_decay < 1``.
        diag_adapt: When ``True``, adapt a per-coordinate diagonal metric
            ``(t, s)`` in the spirit of AdaGrad/DoG: the steps become
            ``tau = eta/k * t`` and ``sigma = eta*k * s``. Over each restart
            cycle every coordinate accumulates its squared eta-normalised
            movement; at a restart its entry is multiplied by
            ``(rho / geomean rho)**diag_theta``, where
            ``rho = |net displacement| / sqrt(accumulated)`` (large for a
            steadily drifting coordinate, small for an oscillating one). The
            metric only changes at restarts and the line search measures in it,
            so it has no effect with ``restarts=False`` or the fixed-step
            optimiser (``adaptive_eta=False``). Default ``True``, as in
            ``jaddle_linear.solve``.
        diag_theta: Log-space step of the ``diag_adapt`` update (``0`` freezes
            ``t = s = 1``).
        diag_clamp: ``diag_adapt`` clamp: ``t, s`` stay in
            ``[1/diag_clamp, diag_clamp]``.
        gradient_scaling: When ``True``, rescale the objective and each
            constraint row by the size of its gradient at the start point
            (IPOPT's gradient-based scaling): ``σ f`` and ``e_i g_i`` with
            ``σ = 1/‖∇f(x₀)‖₂``, ``e_i = 1/‖∇g_i(x₀)‖₂``, clamped to
            ``[1e-4, 1e4]`` (a zero gradient keeps 1). The variables are not
            touched, so this fixes rows and objectives in mismatched units but
            not variables in mismatched units. The convergence test and the
            returned duals stay in the problem's own units. Default ``True``
            (``quadratic_program``, which scales its matrices itself, turns it
            off unless asked).
        gradient_scaling_probes: Row norms are exact when the constraint
            Jacobian has at most this many rows or columns, and otherwise
            estimated from this many random Jacobian-vector products
            (default 16).
        precompile: Compile the device loop before the timed loop starts
            (default ``True``), so ``"solve_seconds"`` measures iteration time
            rather than XLA compilation. The compile still counts against
            ``max_seconds``. A restart that changes ``iterations_per_epoch``
            compiles a new loop inside the timed region.

    Returns:
        dict: The solution together with diagnostics. Keys:
            * ``"solution"``: the ``SaddleState`` (primal/dual iterate).
            * ``"converged"``: ``bool``, whether the solve met the convergence
              criteria (``False`` if the epoch / time budget was exhausted or the
              solve was interrupted).
            * ``"stop_reason"``: ``str``, why the solve terminated:
              ``"certificate"`` (the convergence criteria were met, as in
              ``jaddle_linear.solve``), ``"max_epochs"``, ``"time_limit"`` (``max_seconds``
              exhausted) or ``"interrupted"`` (KeyboardInterrupt).
            * ``"opt_state"``: the final optimiser state, for warm-starting a
              subsequent solve via ``initial_opt_state``.
            * ``"solve_seconds"``: ``float`` wall time of the epoch loop,
              excluding setup. With ``precompile=True`` (default) it also
              excludes XLA compilation; with ``precompile=False`` it includes
              the first-epoch compile.
            * ``"epochs"``: ``int``, number of epochs run.
    """

    # max_seconds is a wall-clock budget for the whole call, setup included.
    solve_entry_time = time.time()
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be > 0 (or None for no limit)")

    if log_every < 1:
        raise ValueError("log_every must be >= 1")

    if verbose:
        print("----------------------------------------------")

    optimiser, step_size, adaptive_step, k_scaling, k_lo, k_hi = _check_settings(
        optimiser, adaptive_eta, step_size, update_mode, k_scale
    )

    if verbose:
        print("====Starting Solve====")
        print("----------------------------------------------")

    solve_cp, from_scaled, obj_scale = cp, None, 1.0
    if gradient_scaling:
        if initial_solution is None:
            initial_solution = cp.initial_solution()
        solve_cp, to_scaled, from_scaled, sigma = _gradient_scaled(
            cp, initial_solution.primal, gradient_scaling_probes
        )
        initial_solution = to_scaled(initial_solution)
        obj_scale = float(sigma)

    _build_chunk, carry = _device_solver(
        solve_cp,
        optimiser=optimiser,
        initial_solution=initial_solution,
        initial_opt_state=initial_opt_state,
        iterations_per_epoch=iterations_per_epoch,
        primal_grad_norm_tolerance=primal_grad_norm_tolerance,
        primal_feasibility_tolerance=primal_feasibility_tolerance,
        complementarity_slack_tolerance=complementarity_slack_tolerance,
        weight_function=weight_function,
        verbose=verbose,
        log_every=log_every,
        average=average,
        report_best=report_best,
        update_mode=update_mode,
        k_scaling=k_scaling,
        k_lo=k_lo,
        k_hi=k_hi,
        k_theta=k_theta,
        k_init=k_init,
        eta_init=step_size,
        adaptive_step=adaptive_step,
        restarts=restarts,
        epochs_per_restart=epochs_per_restart,
        restart_multiplier=restart_multiplier,
        restart_decay=restart_decay,
        iterations_per_epoch_decay=iterations_per_epoch_decay,
        iterations_per_epoch_min=iterations_per_epoch_min,
        diag_adapt=diag_adapt,
        diag_theta=diag_theta,
        diag_clamp=diag_clamp,
    )

    def _epochs_arg(n):
        # Same type on every call, so the (possibly AOT-compiled) chunk accepts it.
        return jnp.asarray(n, carry["count"].dtype)

    chunk_fns = {}
    current_ipe = int(iterations_per_epoch)
    if precompile:
        # Compile the first chunk function ahead of time, so the first epoch
        # doesn't pay its compile inside the timed measurement.
        chunk_fns[current_ipe] = (
            _build_chunk(current_ipe).lower(carry, _epochs_arg(1)).compile()
        )

    # With no logging and no budgets there is nothing for the host to do between
    # epochs, so the whole solve is one call: the loop only returns once the
    # convergence test passes (or a restart changes iterations_per_epoch, whose
    # new scan length needs a new compile).
    single_call = not verbose and not max_epochs and max_seconds is None
    chunk_epochs = _UNBOUNDED_EPOCHS if single_call else 1
    seconds_per_epoch = None
    count = 0
    done = False
    restarts_printed = 0
    is_converged = True
    stop_reason = "certificate"

    def print_chunk_log(host, epoch_time):
        nonlocal restarts_printed
        entries = [(int(r[0]), 0, r) for r in host["log"][: int(host["n_log"])]]
        entries += [(int(r[0]), 1, r) for r in host["evt"][: int(host["n_evt"])]]
        for epoch_no, kind, r in sorted(entries, key=lambda e: e[:2]):
            if kind == 0:
                print(
                    f"|Epoch {epoch_no}|"
                    f"|Obj{float(r[1]) / obj_scale:.2e}|"
                    f"|PGN {float(r[2]):.2e}|"
                    f"|CS {float(r[3]):.2e}|"
                    f"|PFR {float(r[4]):.2e}|"
                    f"|Time {epoch_time:.2f}s|"
                )
                print("----------------------------------------------")
                continue
            restarts_printed += 1
            reason = "sufficient-progress" if r[1] == 0 else "cycle-cap"
            which = "avg" if r[3] else "iterate"
            k_msg = f", k={float(r[4]):.3e}" if k_scaling else ""
            if diag_adapt:
                k_msg += f", t-spread=1e{float(r[7]):.2f}, s-spread=1e{float(r[8]):.2f}"
            print(
                f"Restart {restarts_printed} at epoch {epoch_no} "
                f"({reason}, merit={float(r[2]):.2e} "
                f"[{which}], next cap={float(r[5]):.0f} epochs, "
                f"iters/epoch={int(r[6])}{k_msg})"
            )
            print("----------------------------------------------")

    start_time = time.time()

    try:
        while True:
            if done:
                break
            if max_epochs and count >= max_epochs:
                is_converged = False
                stop_reason = "max_epochs"
                if verbose:
                    print(f"Reached maximum epochs: {max_epochs}. Stopping.")
                    print("----------------------------------------------")
                break
            if max_seconds is not None:
                if time.time() - solve_entry_time >= max_seconds:
                    is_converged = False
                    stop_reason = "time_limit"
                    if verbose:
                        print(f"Reached time limit: {max_seconds}s. Stopping.")
                        print("----------------------------------------------")
                    break

            n_epochs = chunk_epochs
            if max_epochs:
                n_epochs = min(n_epochs, max_epochs - count)
            if max_seconds is not None and seconds_per_epoch:
                remaining = max_seconds - (time.time() - solve_entry_time)
                n_epochs = min(n_epochs, max(1, int(remaining / seconds_per_epoch)))

            run_chunk = chunk_fns.get(current_ipe)
            if run_chunk is None:
                run_chunk = chunk_fns[current_ipe] = _build_chunk(current_ipe)

            chunk_start = time.time()
            new_carry = run_chunk(carry, _epochs_arg(n_epochs))
            keys = ["count", "done", "ipe"]
            if verbose:
                keys += ["log", "n_log", "evt", "n_evt"]
            # The one host/device synchronisation per chunk.
            host = jax.device_get({k: new_carry[k] for k in keys})
            carry = new_carry
            chunk_seconds = time.time() - chunk_start

            epochs_run = max(int(host["count"]) - count, 1)
            count = int(host["count"])
            done = bool(host["done"])
            current_ipe = int(host["ipe"])
            if single_call:
                continue
            seconds_per_epoch = chunk_seconds / epochs_run

            if verbose:
                print_chunk_log(host, seconds_per_epoch)
                zero = jnp.zeros_like(carry["n_log"])
                carry = {**carry, "n_log": zero, "n_evt": zero}

            # Size the next chunk to take about _CHUNK_TARGET_SECONDS, growing
            # by at most 8x per chunk.
            chunk_epochs = int(
                min(
                    max(1.0, _CHUNK_TARGET_SECONDS / seconds_per_epoch),
                    8 * epochs_run,
                )
            )
    except KeyboardInterrupt:
        # `carry` still holds the last completed chunk (it is not donated).
        is_converged = False
        stop_reason = "interrupted"
        count = int(carry["count"])
        print("KeyboardInterrupt received. Returning current solution.")
        print("----------------------------------------------")

    opt_state = carry["opt"]
    output = _reported_point(carry, average)
    if from_scaled is not None:
        output = from_scaled(output)

    output = jax.block_until_ready(output)
    end_time = time.time()
    if verbose:
        print(f"Time to solution: {end_time - start_time:.2f} seconds")
        print("----------------------------------------------")
        print(f"Epochs to solution: {count}")
        print("----------------------------------------------")
        print(f"Objective: {cp.objective(output.primal):.5e}")
        print("----------------------------------------------")

    return {
        "solution": output,
        "converged": is_converged,
        "stop_reason": stop_reason,
        "opt_state": opt_state,
        "solve_seconds": end_time - start_time,
        "epochs": count,
    }


# %%


class SolveCoreResult(NamedTuple):
    """What a ``make_solver`` function returns; every field is a JAX array, so
    results batch under ``vmap``. ``solution`` is the reported point (the
    average when ``average=True``, else the last iterate), ``converged``
    whether the convergence test passed, and ``epochs`` the epochs run."""

    solution: SaddleState
    converged: jnp.ndarray
    epochs: jnp.ndarray


def make_solver(
    build_cp,
    max_epochs=None,
    optimiser=None,
    iterations_per_epoch=256,
    primal_grad_norm_tolerance=1e-2,
    primal_feasibility_tolerance=1e-3,
    complementarity_slack_tolerance=1e-3,
    weight_function=lambda _: 1.0,
    average=True,
    report_best=True,
    update_mode="alternating",
    k_scale=100.0,
    k_theta=0.5,
    k_init=None,
    adaptive_eta=True,
    step_size=1.0,
    restarts=True,
    epochs_per_restart=None,
    restart_multiplier=1.0,
    restart_decay=0.2,
    iterations_per_epoch_min=100,
    diag_adapt=True,
    diag_theta=0.5,
    diag_clamp=10.0,
    gradient_scaling=True,
    gradient_scaling_probes=16,
):
    """
    Build a pure-JAX solve function for a family of convex problems.

    ``build_cp(params)`` returns an ordinary ``JaddleCP`` whose functions
    close over ``params`` (any pytree: data, weights, right-hand sides,
    bounds). ``make_solver`` returns
    ``solve_fn(params, initial_solution=None) -> SolveCoreResult``. The whole
    solve is traceable, so ``solve_fn`` can be ``jax.jit``-ed and
    ``jax.vmap``-ed over a batch of ``params`` whose problems share their
    shapes::

        def build_cp(a):
            return JaddleCP(n, objective=lambda x: jnp.sum((x - a) ** 2), ...)

        solve_fn = jax.jit(make_solver(build_cp, max_epochs=300))
        result = solve_fn(a)
        batch = jax.vmap(solve_fn)(a_batch)

    The algorithm is ``solve()``'s, with the same arguments and defaults.
    Called directly, a solve reproduces ``solve()`` exactly; under an outer
    ``jax.jit`` XLA fuses the whole solve into one program and may round
    differently (measured: 1e-16 to 1e-14, at most 1e-6 after the adaptive
    step amplified it, same epoch counts). What needs the host is not
    available: no ``verbose`` log, no ``max_seconds``, and the epoch length
    never changes during a solve (no ``iterations_per_epoch_decay``;
    ``iterations_per_epoch_min`` is capped at ``iterations_per_epoch``).
    ``max_epochs=None`` runs until the convergence test passes, so set a budget
    when that may not happen. Batched members iterate in lockstep until the
    last one stops (finished members are frozen).
    """
    optimiser, step_size, adaptive_step, k_scaling, k_lo, k_hi = _check_settings(
        optimiser, adaptive_eta, step_size, update_mode, k_scale
    )
    n_epochs = _UNBOUNDED_EPOCHS if not max_epochs else int(max_epochs)

    def solve_fn(params, initial_solution=None):
        cp = build_cp(params)
        from_scaled = None
        if gradient_scaling:
            if initial_solution is None:
                initial_solution = cp.initial_solution()
            cp, to_scaled, from_scaled, _ = _gradient_scaled(
                cp, initial_solution.primal, gradient_scaling_probes
            )
            initial_solution = to_scaled(initial_solution)
        build_chunk, carry = _device_solver(
            cp,
            optimiser=optimiser,
            initial_solution=initial_solution,
            initial_opt_state=None,
            iterations_per_epoch=iterations_per_epoch,
            primal_grad_norm_tolerance=primal_grad_norm_tolerance,
            primal_feasibility_tolerance=primal_feasibility_tolerance,
            complementarity_slack_tolerance=complementarity_slack_tolerance,
            weight_function=weight_function,
            verbose=False,
            log_every=1,
            average=average,
            report_best=report_best,
            update_mode=update_mode,
            k_scaling=k_scaling,
            k_lo=k_lo,
            k_hi=k_hi,
            k_theta=k_theta,
            k_init=k_init,
            eta_init=step_size,
            adaptive_step=adaptive_step,
            restarts=restarts,
            epochs_per_restart=epochs_per_restart,
            restart_multiplier=restart_multiplier,
            restart_decay=restart_decay,
            iterations_per_epoch_decay=1.0,
            iterations_per_epoch_min=min(
                iterations_per_epoch_min, iterations_per_epoch
            ),
            diag_adapt=diag_adapt,
            diag_theta=diag_theta,
            diag_clamp=diag_clamp,
        )
        carry = build_chunk(iterations_per_epoch)(
            carry, jnp.asarray(n_epochs, carry["count"].dtype)
        )
        solution = _reported_point(carry, average)
        if from_scaled is not None:
            solution = from_scaled(solution)
        return SolveCoreResult(
            solution=solution,
            converged=carry["done"],
            epochs=carry["count"],
        )

    return solve_fn


_export.register(SolveCoreResult, "jaddle.convex.SolveCoreResult")


def save_cp(cp, path, *, platforms=None, vjp_order=1, gradient_scaling_probes=16):
    """
    Write ``cp`` to ``path`` as a self-contained file that ``load_cp`` reads
    back without the code that built it.

    Each callable (objective, constraints, and ``dual_bound`` / ``residuals``
    when set) is traced with ``jax.export`` and stored as StableHLO, with
    reverse-mode derivatives up to ``vjp_order`` (the solver needs 1). Arrays
    the callables close over are stored inside the file as constants. The
    file is an ``.npz`` (written to ``path`` as given) that loads without
    pickle.

    Shapes and dtype are fixed when saving: the number of variables and
    constraints, and the dtype of ``cp.lower_bounds``. The file runs on
    ``platforms`` (default: the current backend), e.g. ``("cpu", "cuda")``
    for both.

    Exported functions have no forward-mode derivatives and no batching, so a
    loaded CP cannot be used under ``vmap``. ``solve``'s
    ``gradient_scaling`` needs both, so the scales are exported here too,
    from the original callables, with ``gradient_scaling_probes`` fixed at
    save time (the loaded CP ignores ``solve``'s value). To save a family of
    problems ``build_cp(params)``, save a solver for it with ``save_solver``.
    """
    n = int(cp.num_variables)
    dtype = jnp.result_type(float, jnp.asarray(cp.lower_bounds).dtype)
    x = jax.ShapeDtypeStruct((n,), dtype)
    m_eq = jax.eval_shape(cp.constraints_eq, x).shape[0]
    m_ineq = jax.eval_shape(cp.constraints_ineq, x).shape[0]
    d_eq = jax.ShapeDtypeStruct((m_eq,), dtype)
    d_ineq = jax.ShapeDtypeStruct((m_ineq,), dtype)

    def exported(fn, *args, order=vjp_order):
        return _export.serialized(fn, args, platforms, order)

    arrays = {
        "fn_objective": exported(cp.objective, x),
        "fn_constraints_eq": exported(cp.constraints_eq, x),
        "fn_constraints_ineq": exported(cp.constraints_ineq, x),
        # Only evaluated, never differentiated.
        "fn_gradient_scales": exported(
            lambda x0: _gradient_scales(cp, x0, gradient_scaling_probes), x, order=0
        ),
    }
    if cp.dual_bound is not None:
        arrays["fn_dual_bound"] = exported(cp.dual_bound, d_ineq, d_eq)
    if getattr(cp, "residuals", None) is not None:
        state = SaddleState(primal=x, dual_ineq=d_ineq, dual_eq=d_eq)
        arrays["fn_residuals"] = exported(cp.residuals, state)

    meta = dict(
        num_variables=n,
        num_eq_constraints=m_eq,
        num_ineq_constraints=m_ineq,
        dtype=jnp.dtype(dtype).name,
        gradient_scaling_probes=gradient_scaling_probes,
    )
    arrays["lower_bounds"] = np.asarray(cp.lower_bounds, dtype=dtype)
    arrays["upper_bounds"] = np.asarray(cp.upper_bounds, dtype=dtype)
    _export.write(path, "cp", meta, arrays)


def load_cp(path):
    """
    Read a ``JaddleCP`` written by ``save_cp``. It solves with ``solve`` like
    the original, but its shapes are fixed, it cannot be used under ``vmap``,
    and it runs only on the platforms it was saved for (see ``save_cp``).
    Inputs are cast to the saved dtype, so a float32 file also runs under
    x64; a float64 file needs ``jax_enable_x64``.
    """
    meta, arrays, fns = _export.read(path, "cp")
    dtype = jnp.dtype(meta["dtype"])

    def wrap(name):
        if name not in fns:
            return None
        return lambda *args: _export.call_in_saved_dtypes(fns[name], args)

    cp = JaddleCP(
        num_variables=meta["num_variables"],
        objective=wrap("objective"),
        constraints_eq=wrap("constraints_eq"),
        constraints_ineq=wrap("constraints_ineq"),
        lower_bounds=jnp.asarray(arrays["lower_bounds"], dtype),
        upper_bounds=jnp.asarray(arrays["upper_bounds"], dtype),
        dual_bound=wrap("dual_bound"),
        residuals=wrap("residuals"),
    )
    cp.gradient_scales = wrap("gradient_scales")
    return cp


def save_solver(fn, path, *args, batch=False, platforms=None, vjp_order=0):
    """
    Write a solver for a family of problems to ``path``, for ``load_solver``
    to read back without the code that built it.

    ``fn`` is a pure JAX function, typically from ``make_solver``,
    ``make_optimal_value`` or ``make_solution``. ``args`` are example
    arguments for it, usually one problem's params (arrays or
    ``ShapeDtypeStruct``s): only their structure, shapes and dtypes are used,
    and they are fixed in the file. So are the solver's settings
    (tolerances, ``max_epochs``, ...) and anything ``build_cp`` closes over.
    The loaded function takes the same arguments; optional ones not passed
    here (``make_solver``'s ``initial_solution``) keep their defaults.

    ``batch=True`` saves ``jax.vmap(fn)`` with a symbolic batch size: the
    loaded function takes every argument with an extra leading axis, of any
    length. A loaded function cannot be ``vmap``-ed itself, so save it
    batched if you need batches.

    ``vjp_order=1`` saves reverse-mode derivatives with respect to the
    arguments, for ``make_optimal_value`` / ``make_solution`` (``jax.grad``
    then works on the loaded function). ``make_solver``'s solve loop has no
    derivative, so leave it at 0 there. The file runs on ``platforms``
    (default: the current backend), e.g. ``("cpu", "cuda")`` for both.
    """
    _export.save_function(
        fn, path, args, "cp_solver", batch=batch, platforms=platforms, vjp_order=vjp_order
    )


def load_solver(path):
    """
    Read a solver written by ``save_solver``, as a function of the arguments
    it was saved with (each with a leading batch axis if saved with
    ``batch=True``). It works under ``jit``, and under ``jax.grad`` if saved
    with ``vjp_order=1``, but not under ``vmap`` or forward-mode autodiff.
    Inputs are cast to the saved dtypes, so a float32 file also runs under
    x64; a float64 file needs ``jax_enable_x64``.
    """
    return _export.load_function(path, "cp_solver")


def make_optimal_value(build_cp, return_solution=False, **settings):
    """
    The optimal value ``z*(params) = min f(x; params)`` of a family of convex
    problems, as a differentiable function of ``params``.

    ``build_cp`` is as in ``make_solver`` (``settings`` are its options).
    Returns ``value_fn(params) -> z*``. Its gradient comes from the envelope
    theorem, with no differentiation through the iterations: with the
    solution ``(x*, λ*, μ*)`` held fixed,

        ∂z*/∂params = ∂/∂params [ f(x*) + λ*ᵀ g(x*) + μ*ᵀ h(x*)
                                  + ν_lᵀ lower − ν_uᵀ upper ],

    differentiated through ``build_cp``, so data, weights, right-hand sides
    and bounds can all depend on ``params``. The bound multipliers come from
    the stationarity residual ``r = ∇ₓL`` at the solution: ``ν_l = max(r, 0)``
    on finite lower bounds and ``ν_u = max(−r, 0)`` on finite upper bounds.

    Exact where the optimal multipliers are unique; elsewhere ``z*`` has a
    kink and this is the subgradient at the solution found. Only as accurate
    as the solve: use tight tolerances. Works under ``jit`` and ``vmap``.

    ``return_solution=True`` makes ``value_fn`` return ``(z*, SolveCoreResult)``
    from the same solve, for ``jax.grad(value_fn, has_aux=True)``: each call
    is a fresh solve, and separate solves differ slightly.
    """
    solve_fn = make_solver(build_cp, **settings)

    def lagrangian(params, solution, nu_lower, nu_upper):
        cp = build_cp(params)
        x = solution.primal
        lower, upper = jnp.asarray(cp.lower_bounds), jnp.asarray(cp.upper_bounds)
        return (
            cp.objective(x)
            + solution.dual_ineq @ cp.constraints_ineq(x)
            + solution.dual_eq @ cp.constraints_eq(x)
            + nu_lower @ jnp.where(jnp.isfinite(lower), lower, 0.0)
            - nu_upper @ jnp.where(jnp.isfinite(upper), upper, 0.0)
        )

    def value_at_forward(params, solution):
        return build_cp(params).objective(solution.primal), (params, solution)

    def value_at_backward(residuals, g):
        params, solution = residuals
        cp = build_cp(params)
        # Bound multipliers from the stationarity residual r = ∇ₓL.
        r = jax.grad(
            lambda x: cp.objective(x)
            + solution.dual_ineq @ cp.constraints_ineq(x)
            + solution.dual_eq @ cp.constraints_eq(x)
        )(solution.primal)
        lower, upper = jnp.asarray(cp.lower_bounds), jnp.asarray(cp.upper_bounds)
        nu_lower = jnp.where(jnp.isfinite(lower), jnp.maximum(r, 0.0), 0.0)
        nu_upper = jnp.where(jnp.isfinite(upper), jnp.maximum(-r, 0.0), 0.0)
        _, vjp = jax.vjp(lambda p: lagrangian(p, solution, nu_lower, nu_upper), params)
        (grad_params,) = vjp(g)
        # The solution is treated as fixed: no gradient flows into the solve.
        return grad_params, jax.tree.map(jnp.zeros_like, solution)

    @jax.custom_vjp
    def value_at(params, solution):
        return value_at_forward(params, solution)[0]

    value_at.defvjp(value_at_forward, value_at_backward)

    def value_fn(params):
        result = solve_fn(jax.tree.map(jax.lax.stop_gradient, params))
        solution = jax.tree.map(jax.lax.stop_gradient, result.solution)
        z = value_at(params, solution)
        return (z, result) if return_solution else z

    return value_fn


def make_solution(build_cp, active_tol=1e-6, gmres_tol=1e-10, **settings):
    """
    The solution ``x*(params)`` of a family of convex problems, differentiable
    by implicit differentiation of the KKT conditions on the active set.

    ``build_cp`` and ``settings`` are as in ``make_solver``. Returns
    ``x_fn(params) -> x*``. At the solution the active set is frozen:
    variables within ``active_tol`` (relative) of a finite bound, and
    inequality rows with ``g(x*) >= -active_tol``. The KKT conditions then form
    a square system ``F(x, μ, λ; params) = 0``: stationarity ``∇ₓL = 0`` on
    the free variables, ``x = bound(params)`` on the bound ones, ``h(x) = 0``,
    ``g(x) = 0`` on the active rows and ``λ = 0`` on the inactive ones. A VJP
    solves ``(∂F/∂z)ᵀ w = (g, 0, 0)`` matrix-free with GMRES (using
    ``jax.vjp``, so all on the device and fine under ``vmap``) and returns
    ``−(∂F/∂params)ᵀ w``.

    This needs a nondegenerate solution: strict complementarity, independent
    active constraints, and a Hessian of the Lagrangian positive definite on
    their null space (e.g. a strictly convex objective). Where GMRES cannot
    solve the system to ``gmres_tol`` (relative residual), the gradient is
    returned as NaN rather than silently wrong: a jitted function cannot raise.
    Solve to tolerances well below ``active_tol``.
    """
    from jax.scipy.sparse.linalg import gmres

    solve_fn = make_solver(build_cp, **settings)

    def kkt(z, params, masks):
        x, mu, lam = z
        free, at_lower, active = masks
        cp = build_cp(params)
        lower, upper = jnp.asarray(cp.lower_bounds), jnp.asarray(cp.upper_bounds)

        def lag(x):
            return (
                cp.objective(x)
                + lam @ cp.constraints_ineq(x)
                + mu @ cp.constraints_eq(x)
            )

        bound = jnp.where(at_lower, lower, upper)
        bound = jnp.where(jnp.isfinite(bound), bound, 0.0)
        return (
            jnp.where(free, jax.grad(lag)(x), x - bound),
            cp.constraints_eq(x),
            jnp.where(active, cp.constraints_ineq(x), lam),
        )

    def forward(params):
        solution = solve_fn(jax.tree.map(jax.lax.stop_gradient, params)).solution
        return solution.primal, (params, solution)

    def backward(residuals, g):
        params, solution = residuals
        cp = build_cp(params)
        x = solution.primal
        lower, upper = jnp.asarray(cp.lower_bounds), jnp.asarray(cp.upper_bounds)
        at_lower = jnp.isfinite(lower) & (
            jnp.abs(x - lower) <= active_tol * (1 + jnp.abs(lower))
        )
        at_upper = (
            jnp.isfinite(upper)
            & (jnp.abs(x - upper) <= active_tol * (1 + jnp.abs(upper)))
            & ~at_lower
        )
        g_ineq = cp.constraints_ineq(x)
        active = g_ineq >= -active_tol * (1 + jnp.abs(g_ineq))
        masks = (~(at_lower | at_upper), at_lower, active)
        z = (x, solution.dual_eq, solution.dual_ineq)

        _, vjp_z = jax.vjp(lambda zz: kkt(zz, params, masks), z)
        rhs = (g, jnp.zeros_like(solution.dual_eq), jnp.zeros_like(solution.dual_ineq))

        def transpose_jacobian(w):
            return vjp_z(w)[0]

        w, _ = gmres(
            transpose_jacobian, rhs, tol=gmres_tol, atol=0.0, restart=50, maxiter=50
        )
        residual = jax.tree.map(lambda a, b: a - b, transpose_jacobian(w), rhs)
        flat_norm = lambda t: jnp.sqrt(sum(_dot(a, a) for a in jax.tree.leaves(t)))
        ok = flat_norm(residual) <= 1e3 * gmres_tol * (1 + flat_norm(rhs))

        _, vjp_params = jax.vjp(lambda p: kkt(z, p, masks), params)
        (grad_params,) = vjp_params(w)
        return (jax.tree.map(lambda t: jnp.where(ok, -t, jnp.nan), grad_params),)

    @jax.custom_vjp
    def solution_map(params):
        return forward(params)[0]

    solution_map.defvjp(forward, backward)
    return solution_map


# %%
def _equilibrate_kkt(Q, A, ruiz_iterations, pc_iterations):
    """Variable scales ``d`` and constraint scales ``e`` that equilibrate the
    KKT matrix ``[[Q, Aᵀ], [A, 0]]`` symmetrically: Ruiz sweeps (each row's
    largest entry to 1), then Pock-Chambolle sweeps (each row's absolute sum
    to 1). One scale per row of the KKT matrix keeps ``D Q D`` symmetric, so
    the scaling is a change of variables ``x = D x̃``. An empty row keeps
    scale 1."""
    import scipy.sparse as sp

    n = Q.shape[0]
    kkt = abs(sp.bmat([[Q, A.T], [A, None]], format="csr"))
    scale = np.ones(kkt.shape[0])

    def sweep(reduce):
        norms = scale * np.asarray(reduce(kkt @ sp.diags(scale))).ravel()
        return scale / np.sqrt(np.where(norms > 0, norms, 1.0))

    for _ in range(ruiz_iterations):
        scale = sweep(lambda m: m.max(axis=1).toarray())
    for _ in range(pc_iterations):
        scale = sweep(lambda m: m.sum(axis=1))
    return scale[:n], scale[n:]


# A product with a BCOO matrix gathers and scatters every entry, where a dense
# one is a single BLAS call: on a GPU the dense product wins from about a
# quarter of the entries being nonzero, by 9x (n = 1000) to 40x (n = 4000) on a
# full matrix. A small matrix is dense whatever its density: its dense product
# is one cheap call, and under vmap a batched one, where a batched BCOO product
# was 9x slower at 105 x 120 with 20% nonzeros (examples/cp/mpc_fleet.py).
_DENSE_DENSITY = 0.25
_DENSE_SIZE = 1 << 16


def _device_matrix(A, dtype):
    """The scipy matrix ``A`` on the device with ``dtype`` data: a dense array
    if it has at most ``_DENSE_SIZE`` entries or at least ``_DENSE_DENSITY``
    of them are nonzero, else a BCOO."""
    from jaddle.jaddle_basic_types import scipy_to_bcoo

    size = A.shape[0] * A.shape[1]
    if size > 0 and (size <= _DENSE_SIZE or A.nnz >= _DENSE_DENSITY * size):
        return jax.device_put(A.toarray().astype(jnp.dtype(dtype)))
    return scipy_to_bcoo(A, dtype)


def _qp_certificate(Q, c, A_eq, b_eq, A_ineq, b_ineq, lower, upper, solution):
    """Relative KKT residuals and duality gap of a primal-dual pair for
    ``min ½xᵀQx + cᵀx``, in the problem's own units (l2 norms)."""
    x = np.asarray(solution.primal, dtype=np.float64)
    y_eq = np.asarray(solution.dual_eq, dtype=np.float64)
    y_ineq = np.asarray(solution.dual_ineq, dtype=np.float64)
    Qx = Q @ x
    reduced_cost = Qx + c + A_eq.T @ y_eq + A_ineq.T @ y_ineq
    # A reduced cost is paid for by the bound it pushes against, if finite.
    at_lower = (reduced_cost > 0) & np.isfinite(lower)
    at_upper = (reduced_cost < 0) & np.isfinite(upper)
    bound_term = np.sum(
        reduced_cost * np.where(at_lower, lower, np.where(at_upper, upper, 0.0)),
    )
    primal_residual = np.linalg.norm(
        np.concatenate(
            [
                A_eq @ x - b_eq,
                np.maximum(A_ineq @ x - b_ineq, 0.0),
                np.maximum(lower - x, 0.0),
                np.maximum(x - upper, 0.0),
            ]
        )
    )
    dual_residual = np.linalg.norm(
        np.concatenate(
            [np.where(at_lower | at_upper, 0.0, reduced_cost), np.minimum(y_ineq, 0.0)]
        )
    )
    objective = 0.5 * x @ Qx + c @ x
    # Lagrangian lower bound, with the quadratic linearised at x.
    dual_objective = -0.5 * x @ Qx - b_eq @ y_eq - b_ineq @ y_ineq + bound_term
    gap = objective - dual_objective
    b_norm = np.linalg.norm(np.concatenate([b_eq, b_ineq]))
    c_norm = max(np.linalg.norm(c), np.linalg.norm(Qx))
    return {
        "objective": float(objective),
        "dual_objective": float(dual_objective),
        "duality_gap": float(gap),
        "relative_gap": float(abs(gap) / (1 + abs(objective) + abs(dual_objective))),
        "primal_feasibility_residual": float(primal_residual),
        "dual_feasibility_residual": float(dual_residual),
        "relative_primal_feasibility_residual": float(primal_residual / (1 + b_norm)),
        "relative_dual_feasibility_residual": float(dual_residual / (1 + c_norm)),
    }


def _matvec(M, v):
    """``M @ v`` at full precision for a dense ``M``. Under ``vmap`` a product
    with a shared matrix becomes a matrix-matrix product, which the float32
    profile would run in TF32 (about 3 significant digits)."""
    if isinstance(M, jax.Array):
        return jnp.matmul(M, v, precision="highest")
    return M @ v


def _norm(v):
    """‖v‖₂ computed in at least float32 (see ``_wide``)."""
    return jnp.sqrt(_dot(v, v))


def _scaled_qp_cp(matrices, c_s, b_eq_s, b_ineq_s, c_norm, b_norm):
    """The ``JaddleCP`` of a KKT-scaled QP, with the QP certificate (in the
    problem's own units) as its ``residuals``. ``matrices`` is
    ``(Q_s, A_eq_s, A_eq_T, A_ineq_s, A_ineq_T, lower_s, upper_s, d, e_eq,
    e_ineq)``: the scaled data and the variable and row scales. ``c_norm``
    and ``b_norm`` are the l2 norms of the unscaled ``c`` and ``b``."""
    Q_s, A_eq_s, A_eq_T, A_ineq_s, A_ineq_T, lower_s, upper_s, d, e_eq, e_ineq = (
        matrices
    )

    def objective(x):
        # Q is symmetric, so the gradient is Qx + c: one product with Q,
        # where differentiating xᵀQx would take a second with Qᵀ.
        Qx = jax.lax.stop_gradient(_matvec(Q_s, x))
        return x @ Qx - 0.5 * jax.lax.stop_gradient(x) @ Qx + c_s @ x

    def residuals(state):
        # _qp_certificate on the scaled problem: a residual of the original
        # is the scaled one over its row's or column's scale, and the
        # objectives and bound terms are the same numbers in either.
        x = state.primal
        Qx = _matvec(Q_s, x)
        reduced_cost = (
            Qx
            + c_s
            + _matvec(A_eq_T, state.dual_eq)
            + _matvec(A_ineq_T, state.dual_ineq)
        )
        at_lower = (reduced_cost > 0) & jnp.isfinite(lower_s)
        at_upper = (reduced_cost < 0) & jnp.isfinite(upper_s)
        bound_term = jnp.sum(
            reduced_cost
            * jnp.where(at_lower, lower_s, jnp.where(at_upper, upper_s, 0.0))
        )
        dual_residual = _norm(
            jnp.where(at_lower | at_upper, 0.0, reduced_cost) / d
        )
        primal_residual = _norm(
            jnp.concatenate(
                [
                    (_matvec(A_eq_s, x) - b_eq_s) / e_eq,
                    jnp.maximum(_matvec(A_ineq_s, x) - b_ineq_s, 0.0) / e_ineq,
                ]
            )
        )
        primal_objective = 0.5 * x @ Qx + c_s @ x
        dual_objective = (
            -0.5 * x @ Qx
            - b_eq_s @ state.dual_eq
            - b_ineq_s @ state.dual_ineq
            + bound_term
        )
        return (
            dual_residual / (1 + jnp.maximum(c_norm, _norm(Qx / d))),
            primal_residual / (1 + b_norm),
            jnp.abs(primal_objective - dual_objective)
            / (1 + jnp.abs(primal_objective) + jnp.abs(dual_objective)),
        )

    return JaddleCP(
        num_variables=c_s.shape[0],
        objective=objective,
        constraints_eq=lambda x: _matvec(A_eq_s, x) - b_eq_s,
        constraints_ineq=lambda x: _matvec(A_ineq_s, x) - b_ineq_s,
        lower_bounds=lower_s,
        upper_bounds=upper_s,
        residuals=residuals,
    )


def _scale_qp(
    Q, n, A_eq, A_ineq, lower_bounds, upper_bounds, ruiz_iterations, pc_iterations
):
    """Check a QP's matrices and KKT-scale them. Returns the matrices as
    scipy CSR (``None`` blocks as empty ones), the bounds as float arrays,
    the variable scales ``d`` and row scales ``e`` (eq rows first), and the
    scaled device data ``_scaled_qp_cp`` takes as ``matrices``."""
    import scipy.sparse as sp

    Q = sp.csr_matrix(Q, dtype=np.float64)
    if Q.shape != (n, n):
        raise ValueError(f"Q has shape {Q.shape}, expected {(n, n)}")
    if abs(Q - Q.T).max() > 1e-12 * max(abs(Q).max(), 1.0):
        raise ValueError("Q must be symmetric")

    def block(A):
        if A is None:
            return sp.csr_matrix((0, n))
        return sp.csr_matrix(A, dtype=np.float64)

    A_eq, A_ineq = block(A_eq), block(A_ineq)
    m_eq = A_eq.shape[0]
    lower = (
        np.full(n, -np.inf) if lower_bounds is None else np.asarray(lower_bounds, float)
    )
    upper = (
        np.full(n, np.inf) if upper_bounds is None else np.asarray(upper_bounds, float)
    )

    d, e = _equilibrate_kkt(
        Q, sp.vstack([A_eq, A_ineq]), ruiz_iterations, pc_iterations
    )
    D = sp.diags(d)
    # Under the float16 profile only the matrices are stored in half
    # precision; the vectors, and so the iterates, are at least float32.
    matrix_dtype = jo.jaddle_dtype()
    dtype = _wide(matrix_dtype)
    A_eq_scipy = sp.diags(e[:m_eq]) @ A_eq @ D
    A_ineq_scipy = sp.diags(e[m_eq:]) @ A_ineq @ D
    matrices = (
        _device_matrix(D @ Q @ D, matrix_dtype),
        _device_matrix(A_eq_scipy, matrix_dtype),
        _device_matrix(A_eq_scipy.T, matrix_dtype),
        _device_matrix(A_ineq_scipy, matrix_dtype),
        _device_matrix(A_ineq_scipy.T, matrix_dtype),
        jnp.asarray(lower / d, dtype=dtype),
        jnp.asarray(upper / d, dtype=dtype),
        jnp.asarray(d, dtype=dtype),
        jnp.asarray(e[:m_eq], dtype=dtype),
        jnp.asarray(e[m_eq:], dtype=dtype),
    )
    return Q, A_eq, A_ineq, lower, upper, d, e, matrices


def quadratic_program(
    Q,
    c,
    A_eq=None,
    b_eq=None,
    A_ineq=None,
    b_ineq=None,
    lower_bounds=None,
    upper_bounds=None,
    primal_feasibility_tolerance=1e-3,
    dual_feasibility_tolerance=1e-3,
    dual_gap_tolerance=1e-3,
    ruiz_iterations=10,
    pc_iterations=1,
    initial_solution=None,
    **solve_options,
):
    """
    Solve the convex quadratic program

        minimise    ½ xᵀQx + cᵀx
        subject to  A_eq   x  = b_eq
                    A_ineq x <= b_ineq
                    lower_bounds <= x <= upper_bounds

    with ``solve()``, after scaling it from its matrices.

    ``Q`` is symmetric positive semidefinite; it and the constraint matrices
    are NumPy arrays or SciPy sparse matrices. Each is held on the device as a
    dense array if at least a quarter of its entries are nonzero and as a
    sparse one otherwise, whichever form it was passed in. A constraint block
    or a bound left as ``None`` is absent. Other keyword arguments go to
    ``solve()``.

    ``solve()`` sees a problem only through its functions, so it cannot scale
    one; here the entries are known. The KKT matrix ``[[Q, Aᵀ], [A, 0]]`` is
    equilibrated symmetrically with ``ruiz_iterations`` Ruiz sweeps and
    ``pc_iterations`` Pock-Chambolle sweeps (0 and 0 solve the problem as
    given), the scaled problem is solved, and the solution is mapped back.

    Termination is the linear solver's, tested in the problem's own units
    whatever the scaling, with l2 norms: primal feasibility relative to
    ``1 + ‖b‖``, dual feasibility (reduced costs ``Qx + c + Aᵀy`` that no
    finite bound absorbs) relative to ``1 + max(‖c‖, ‖Qx‖)``, and the duality
    gap relative to ``1 + |primal| + |dual|``, where the dual objective is the
    Lagrangian bound with the quadratic linearised at ``x``.

    Under the float16 profile the scaled matrices are stored in half precision
    (the vectors and iterates in float32), so the solver meets its tolerances
    on matrices rounded to about 3 significant digits. ``"converged"`` then
    also requires the ``"certificate"``, computed from the matrices as given,
    to meet them.

    Returns ``solve()``'s dict, with these keys replaced or added:

    * ``"solution"``: a ``SaddleState`` in the problem's own units. The
      Lagrangian is
      ``½xᵀQx + cᵀx + y_eqᵀ(A_eq x − b_eq) + y_ineqᵀ(A_ineq x − b_ineq)``
      with ``y_ineq >= 0``.
    * ``"objective"``: the objective there.
    * ``"certificate"``: the three residuals above, recomputed from the
      unscaled matrices (the keys of
      ``jaddle_linear.evaluate_lp_certificate``).

    ``initial_solution`` is a ``SaddleState`` in the problem's own units;
    ``initial_opt_state`` is passed to ``solve()`` unchanged. The KKT scaling
    already equilibrates the rows, so ``gradient_scaling`` defaults to
    ``False`` here.
    """
    c = np.asarray(c, dtype=np.float64)
    Q, A_eq, A_ineq, lower, upper, d, e, matrices = _scale_qp(
        Q,
        c.shape[0],
        A_eq,
        A_ineq,
        lower_bounds,
        upper_bounds,
        ruiz_iterations,
        pc_iterations,
    )
    m_eq = A_eq.shape[0]
    b_eq = np.asarray(b_eq, dtype=np.float64) if m_eq else np.zeros(0)
    b_ineq = np.asarray(b_ineq, dtype=np.float64) if A_ineq.shape[0] else np.zeros(0)
    dtype = _wide(jo.jaddle_dtype())  # iterates at least float32
    cp = _scaled_qp_cp(
        matrices,
        jnp.asarray(d * c, dtype=dtype),
        jnp.asarray(e[:m_eq] * b_eq, dtype=dtype),
        jnp.asarray(e[m_eq:] * b_ineq, dtype=dtype),
        float(np.linalg.norm(c)),
        float(np.linalg.norm(np.concatenate([b_eq, b_ineq]))),
    )
    if initial_solution is not None:
        initial_solution = SaddleState(
            primal=jnp.asarray(np.asarray(initial_solution.primal) / d, dtype=dtype),
            dual_ineq=jnp.asarray(
                np.asarray(initial_solution.dual_ineq) / e[m_eq:], dtype=dtype
            ),
            dual_eq=jnp.asarray(
                np.asarray(initial_solution.dual_eq) / e[:m_eq], dtype=dtype
            ),
        )
    # The KKT scaling has already equilibrated the rows.
    solve_options.setdefault("gradient_scaling", False)
    result = solve(
        cp,
        initial_solution=initial_solution,
        primal_grad_norm_tolerance=dual_feasibility_tolerance,
        primal_feasibility_tolerance=primal_feasibility_tolerance,
        complementarity_slack_tolerance=dual_gap_tolerance,
        **solve_options,
    )
    scaled = result["solution"]
    solution = SaddleState(
        primal=jnp.asarray(d * np.asarray(scaled.primal, dtype=np.float64)),
        dual_ineq=jnp.asarray(
            e[m_eq:] * np.asarray(scaled.dual_ineq, dtype=np.float64)
        ),
        dual_eq=jnp.asarray(e[:m_eq] * np.asarray(scaled.dual_eq, dtype=np.float64)),
    )
    certificate = _qp_certificate(
        Q, c, A_eq, b_eq, A_ineq, b_ineq, lower, upper, solution
    )
    result.update(
        solution=solution, objective=certificate["objective"], certificate=certificate
    )
    if jnp.dtype(jo.jaddle_dtype()).itemsize < 4:
        # Under the float16 profile the solver met its tolerances on matrices
        # rounded to ~3 digits; judge the answer on the QP as given.
        result["converged"] = bool(
            result["converged"]
            and certificate["relative_primal_feasibility_residual"]
            <= primal_feasibility_tolerance
            and certificate["relative_dual_feasibility_residual"]
            <= dual_feasibility_tolerance
            and certificate["relative_gap"] <= dual_gap_tolerance
        )
    return result


def make_qp_solver(
    Q,
    A_eq=None,
    A_ineq=None,
    lower_bounds=None,
    upper_bounds=None,
    primal_feasibility_tolerance=1e-3,
    dual_feasibility_tolerance=1e-3,
    dual_gap_tolerance=1e-3,
    ruiz_iterations=10,
    pc_iterations=1,
    **settings,
):
    """
    Build a pure-JAX solve function for a family of quadratic programs that
    share their matrices and bounds and differ only in ``c`` and ``b``:

        minimise    ½ xᵀQx + cᵀx
        subject to  A_eq   x  = b_eq
                    A_ineq x <= b_ineq
                    lower_bounds <= x <= upper_bounds

    This is ``quadratic_program`` split in two. The matrices are checked and
    KKT-scaled once, here, on the host; the scaling depends only on them, so
    it serves every ``c`` and ``b``. The returned

        solve_fn(c, b_eq=None, b_ineq=None, initial_solution=None)
            -> SolveCoreResult

    is ``make_solver``'s, so it can be ``jax.jit``-ed and ``jax.vmap``-ed over
    a batch of vectors, as in model predictive control where the measured
    state enters only the right-hand side::

        solve_fn = jax.jit(jax.vmap(make_qp_solver(Q, A_ineq=G, max_epochs=50)))
        batch = solve_fn(c_batch, b_ineq=h_batch)

    ``b_eq`` (``b_ineq``) is required when ``A_eq`` (``A_ineq``) is given.
    ``initial_solution`` and the returned ``solution`` are ``SaddleState``-s
    in the problem's own units, with the multiplier signs of
    ``quadratic_program``. The termination test is ``quadratic_program``'s
    (relative, in the problem's own units). Other keyword arguments go to
    ``make_solver``; ``gradient_scaling`` defaults to ``False``, since the
    KKT scaling already equilibrates the rows.
    """
    n = Q.shape[0]
    _, A_eq, A_ineq, _, _, d, e, matrices = _scale_qp(
        Q,
        n,
        A_eq,
        A_ineq,
        lower_bounds,
        upper_bounds,
        ruiz_iterations,
        pc_iterations,
    )
    m_eq, m_ineq = A_eq.shape[0], A_ineq.shape[0]
    dtype = _wide(jo.jaddle_dtype())  # iterates at least float32
    d = jnp.asarray(d, dtype=dtype)
    e_eq, e_ineq = jnp.asarray(e[:m_eq], dtype=dtype), jnp.asarray(e[m_eq:], dtype=dtype)

    def build_cp(params):
        return _scaled_qp_cp(matrices, *params)

    settings.setdefault("gradient_scaling", False)
    scaled_solve = make_solver(
        build_cp,
        primal_grad_norm_tolerance=dual_feasibility_tolerance,
        primal_feasibility_tolerance=primal_feasibility_tolerance,
        complementarity_slack_tolerance=dual_gap_tolerance,
        **settings,
    )

    def rhs(b, m, name):
        if m == 0:
            return jnp.zeros(0, dtype)
        if b is None:
            raise ValueError(f"{name} is required when its matrix is given")
        b = jnp.asarray(b, dtype=dtype)
        if b.shape != (m,):
            raise ValueError(f"{name} has shape {b.shape}, expected {(m,)}")
        return b

    def solve_fn(c, b_eq=None, b_ineq=None, initial_solution=None):
        c = jnp.asarray(c, dtype=dtype)
        if c.shape != (n,):
            raise ValueError(f"c has shape {c.shape}, expected {(n,)}")
        b_eq = rhs(b_eq, m_eq, "b_eq")
        b_ineq = rhs(b_ineq, m_ineq, "b_ineq")
        params = (
            d * c,
            e_eq * b_eq,
            e_ineq * b_ineq,
            jnp.linalg.norm(c),
            jnp.linalg.norm(jnp.concatenate([b_eq, b_ineq])),
        )
        if initial_solution is not None:
            initial_solution = SaddleState(
                primal=jnp.asarray(initial_solution.primal, dtype) / d,
                dual_ineq=jnp.asarray(initial_solution.dual_ineq, dtype) / e_ineq,
                dual_eq=jnp.asarray(initial_solution.dual_eq, dtype) / e_eq,
            )
        result = scaled_solve(params, initial_solution)
        scaled = result.solution
        return result._replace(
            solution=SaddleState(
                primal=d * scaled.primal,
                dual_ineq=e_ineq * scaled.dual_ineq,
                dual_eq=e_eq * scaled.dual_eq,
            )
        )

    return solve_fn
