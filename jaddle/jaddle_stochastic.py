"""Stochastic saddle-point solver for convex problems whose objective and
constraints are means over data (finite sums).

``JaddleCPStochastic`` describes the problem term by term; ``solve`` runs the
algorithms of ``jaddle_convex.solve`` on minibatch estimates of the
Lagrangian's gradient, while the convergence test, the restart merits and the
reported residuals are computed on the full problem.
"""

import copy
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
from optax.projections import projection_box

import jaddle.jaddle_convex as jc
import jaddle.jaddle_optimisers as jo
from jaddle.jaddle_basic_types import JaddleCP, SaddleState, _wide

_TERMS = ("objective", "constraints_eq", "constraints_ineq")
_VARIANCE_REDUCTIONS = ("svrg", None)


def _num_examples(data, name):
    """The common leading-axis length of the leaves of ``data``."""
    leaves = jax.tree.leaves(data)
    if not leaves:
        raise ValueError(f"{name}_data has no arrays")
    sizes = {np.shape(leaf)[0] if np.ndim(leaf) else None for leaf in leaves}
    if None in sizes or len(sizes) != 1:
        raise ValueError(
            f"every array in {name}_data needs the same leading (example) axis, "
            f"got leading sizes {sorted(s for s in sizes if s is not None)}"
        )
    (n,) = sizes
    if n == 0:
        raise ValueError(f"{name}_data has no examples")
    return n


def _gather(data, idx):
    return jax.tree.map(lambda a: a[idx], data)


def _per_example(fn, x, data):
    """``fn(x, example)`` for every example of ``data``, stacked."""
    return jax.vmap(lambda example: fn(x, example))(data)


class JaddleCPStochastic:
    """A convex problem whose objective and constraints are means over data:

        minimise    f₀(x) + mean_i f(x, a_i)
        subject to  h₀(x) + mean_j h(x, b_j)  = 0
                    g₀(x) + mean_k g(x, c_k) <= 0
                    lower_bounds <= x <= upper_bounds

    Each of ``objective``, ``constraints_eq`` and ``constraints_ineq`` is a
    per-example function ``fn(x, example)`` (a scalar for the objective, a
    vector of the block's rows for the constraints) with its data, a pytree
    of arrays whose leading axis indexes the examples (``example`` is one
    slice of it). The ``*_det`` functions of ``x`` alone are the
    deterministic parts (``f₀, h₀, g₀``, e.g. a ridge penalty). Every part is
    optional; a constraint block with neither part is empty. Each term has
    its own data, of any length.

    ``dual_bound`` and ``residuals`` are as on ``JaddleCP`` and describe the
    full problem.
    """

    def __init__(
        self,
        num_variables,
        lower_bounds,
        upper_bounds,
        objective=None,
        objective_data=None,
        objective_det=None,
        constraints_eq=None,
        constraints_eq_data=None,
        constraints_eq_det=None,
        constraints_ineq=None,
        constraints_ineq_data=None,
        constraints_ineq_det=None,
        dual_bound=None,
        residuals=None,
    ):
        self.num_variables = num_variables
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.dual_bound = dual_bound
        self.residuals = residuals
        given = dict(
            objective=(objective, objective_data, objective_det),
            constraints_eq=(constraints_eq, constraints_eq_data, constraints_eq_det),
            constraints_ineq=(
                constraints_ineq,
                constraints_ineq_data,
                constraints_ineq_det,
            ),
        )
        # name -> (per-example fn or None, data or None, det fn or None)
        self._terms = {}
        self._num_examples = {}
        for name, (fn, data, det) in given.items():
            if (fn is None) != (data is None):
                raise ValueError(
                    f"{name} and {name}_data must be given together "
                    "(use {name}_det for a part without data)"
                )
            if data is not None:
                data = jax.tree.map(jnp.asarray, data)
                self._num_examples[name] = _num_examples(data, name)
            self._terms[name] = (fn, data, det)
        if not self._num_examples:
            raise ValueError(
                "no term has data: use jaddle_convex.JaddleCP for a deterministic "
                "problem"
            )

    @property
    def num_examples(self):
        """Examples per term with data, e.g. ``{"objective": 10000}``."""
        return dict(self._num_examples)

    def _with_data(self, fn):
        """A copy with ``fn`` applied to every data leaf."""
        out = copy.copy(self)
        out._terms = {
            name: (f, None if data is None else jax.tree.map(fn, data), det)
            for name, (f, data, det) in self._terms.items()
        }
        return out

    # --- evaluation ---------------------------------------------------------

    def _value(self, name, x, mean_of):
        """Term ``name`` at ``x``: its deterministic part plus ``mean_of``
        (a function of the per-example fn and its data) of the rest."""
        fn, data, det = self._terms[name]
        parts = []
        if det is not None:
            parts.append(det(x))
        if fn is not None:
            parts.append(mean_of(fn, data))
        if not parts:
            if name == "objective":
                return jnp.zeros((), x.dtype)
            return jnp.zeros(0, x.dtype)
        return parts[0] if len(parts) == 1 else parts[0] + parts[1]

    def _full_mean(self, chunk):
        """``mean_of`` for the exact mean over all the data, in chunks of
        ``chunk`` examples (``None``: all at once)."""

        def mean_of(fn, data, x):
            n = _num_examples(data, "")
            if chunk is None or chunk >= n:
                return jnp.mean(_per_example(fn, x, data), axis=0)
            # Pad to whole chunks with copies of example 0 at weight 0.
            n_chunks = -(-n // chunk)
            positions = np.arange(n_chunks * chunk)
            idx = np.where(positions < n, positions, 0).reshape(n_chunks, chunk)
            weights = (positions < n).reshape(n_chunks, chunk)

            def chunk_sum(args):
                i, w = args
                values = _per_example(fn, x, _gather(data, i))
                return jnp.tensordot(w.astype(values.dtype), values, axes=1)

            sums = jax.lax.map(chunk_sum, (jnp.asarray(idx), jnp.asarray(weights)))
            return jnp.sum(sums, axis=0) / n

        return mean_of

    def full_cp(self, eval_chunk_size=None):
        """The full problem as a ``JaddleCP``: every mean is taken over all
        the data, ``eval_chunk_size`` examples at a time (``None``: all at
        once, which is fastest but holds every example's intermediates in
        memory). Its functions are differentiable in both modes and batch
        under ``vmap``."""
        if eval_chunk_size is not None and (
            isinstance(eval_chunk_size, bool)
            or not isinstance(eval_chunk_size, (int, np.integer))
            or eval_chunk_size <= 0
        ):
            raise ValueError(
                f"eval_chunk_size must be a positive int or None, got {eval_chunk_size!r}"
            )
        mean = self._full_mean(eval_chunk_size)

        def term(name):
            return lambda x: self._value(
                name, x, lambda fn, data: mean(fn, data, x)
            )

        return JaddleCP(
            num_variables=self.num_variables,
            objective=term("objective"),
            constraints_eq=term("constraints_eq"),
            constraints_ineq=term("constraints_ineq"),
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            dual_bound=self.dual_bound,
            residuals=self.residuals,
        )

    def oracle(self, batch_size):
        """A ``jaddle_convex.GradientOracle`` of minibatch estimates: each
        sample draws ``min(batch_size, N)`` examples per term, uniformly with
        replacement, so every estimate is unbiased."""
        sizes = {
            name: min(int(batch_size), n) for name, n in self._num_examples.items()
        }

        def draw(key):
            keys = jax.random.split(key, len(_TERMS))
            return {
                name: jax.random.randint(
                    k, (sizes[name],), 0, self._num_examples[name]
                )
                for name, k in zip(_TERMS, keys)
                if name in sizes
            }

        def value(name, x, sample):
            return self._value(
                name,
                x,
                lambda fn, data: jnp.mean(
                    _per_example(fn, x, _gather(data, sample[name])), axis=0
                ),
            )

        def constraints(x, sample):
            return (
                value("constraints_ineq", x, sample),
                value("constraints_eq", x, sample),
            )

        def lagrangian_map(x, sample):
            return (value("objective", x, sample),) + constraints(x, sample)

        return jc.GradientOracle(
            draw=draw, lagrangian_map=lagrangian_map, constraints=constraints
        )

    # --- shapes and start point ------------------------------------------------

    def _x_struct(self):
        dtype = jnp.result_type(float, jnp.asarray(self.lower_bounds).dtype)
        return jax.ShapeDtypeStruct((self.num_variables,), dtype)

    def num_eq_constraints(self):
        return jax.eval_shape(self.full_cp().constraints_eq, self._x_struct()).shape[0]

    def num_ineq_constraints(self):
        return jax.eval_shape(
            self.full_cp().constraints_ineq, self._x_struct()
        ).shape[0]

    def num_constraints(self):
        return self.num_eq_constraints() + self.num_ineq_constraints()

    def initial_solution(self):
        return self.full_cp().initial_solution()


def _scaled_oracle(oracle, sigma, e_eq, e_ineq):
    """``oracle`` for the gradient-scaled problem ``σf, e g`` (see
    ``jaddle_convex._gradient_scaled``)."""

    def constraints(x, sample):
        ineq, eq = oracle.constraints(x, sample)
        return e_ineq * ineq, e_eq * eq

    def lagrangian_map(x, sample):
        obj, ineq, eq = oracle.lagrangian_map(x, sample)
        return sigma * obj, e_ineq * ineq, e_eq * eq

    return jc.GradientOracle(
        draw=oracle.draw, lagrangian_map=lagrangian_map, constraints=constraints
    )


def _is_positive_int(value):
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, bool)
        and value > 0
    )


@jo._with_local_settings
def solve(
    scp: JaddleCPStochastic,
    optimiser=None,
    max_epochs=None,
    max_seconds=None,
    initial_solution=None,
    initial_opt_state=None,
    batch_size=256,
    variance_reduction="svrg",
    step_decay="auto",
    seed=0,
    eval_chunk_size=None,
    iterations_per_epoch=None,
    primal_grad_norm_tolerance=1e-2,
    primal_feasibility_tolerance=1e-3,
    complementarity_slack_tolerance=1e-3,
    weight_function=lambda _: 1.0,
    verbose=False,
    log_every=1,
    average=True,
    report_best=True,
    update_mode="extragradient",
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
    device=None,
    precision=None,
):
    """
    Solve a ``JaddleCPStochastic`` with stochastic saddle-point optimisation.

    The algorithms are those of ``jaddle_convex.solve`` (same update modes,
    adaptive step, primal weight, restarts, averaging, ``diag_adapt`` and
    gradient scaling), but each iteration estimates the Lagrangian's gradient
    from a minibatch of ``batch_size`` examples per term instead of all the
    data. All gradient evaluations within one iteration share its minibatch,
    so the adaptive step's line search measures the Lipschitz constant of a
    single sampled operator. Everything evaluated between epochs uses the
    full problem: the convergence test (the same three residuals as
    ``jaddle_convex.solve``), the restart merits, the initial primal weight,
    the gradient scales and the verbose log. A ``"certificate"`` stop is
    therefore a certificate for the full problem.

    Arguments not listed here are as in ``jaddle_convex.solve``.

    Args:
        scp: The problem.
        batch_size: Examples drawn per term and iteration, uniformly with
            replacement (default 256; a term with fewer examples draws that
            many).
        variance_reduction: ``"svrg"`` (default) or ``None``. With SVRG each
            epoch evaluates the full Lagrangian map at its start point z̃ (one
            pass over the data), and each iteration uses
            ``F_s(z) - F_s(z̃) + F(z̃)``: unbiased, with a variance that
            vanishes as the iterates approach z̃, so the iterates can converge
            to the tolerances at a fixed step. It costs one extra minibatch
            VJP per iteration. ``None`` uses the plain minibatch gradient
            ``F_s(z)``, whose noise only a decaying step overcomes (see
            ``step_decay``).
        step_decay: Shrinks each update by ``1 / sqrt(1 + i / step_decay)``
            after ``i`` iterations in all (not reset by restarts), leaving its
            exploration alone ("explore aggressively, update conservatively",
            Hsieh et al. 2020): the adaptive schemes move that fraction of the
            way to the line-searched point, whose step and Lipschitz test are
            unchanged; the fixed-step extragradient scales its corrector
            gradient and alternating both of its gradients. This averages out
            the minibatch noise that ``variance_reduction=None`` leaves.
            ``None`` disables it. ``"auto"`` (default) is ``None`` with SVRG
            and ``iterations_per_epoch`` without.
        update_mode: As in ``jaddle_convex.solve``, but the default is
            ``"extragradient"``. It is contractive, so minibatch noise does not
            build up; ``"alternating"`` and ``"pdhg"`` are not, and can
            diverge when the constraints are sampled (constraints given only
            by ``*_det`` parts are fine).
        seed: Seed of the minibatch sampling; a solve is reproducible for a
            given seed.
        eval_chunk_size: Examples evaluated at a time in the full-problem
            passes (``None``, the default: all at once). Set it when one pass
            over all the data does not fit in device memory.
        iterations_per_epoch: Iterations per epoch. ``None`` (default) is
            about one pass over the largest term's data,
            ``ceil(max N / batch_size)``, and at least 100. Each epoch also
            costs one or two full-problem passes for its metrics (and one for
            the SVRG anchor), so an epoch much shorter than a data pass
            spends most of its time on them.

    Returns:
        dict: As ``jaddle_convex.solve``: ``"solution"`` (a ``SaddleState`` in
        the problem's own units), ``"converged"``, ``"stop_reason"``,
        ``"opt_state"``, ``"solve_seconds"`` and ``"epochs"``.
    """
    solve_entry_time = time.time()
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be > 0 (or None for no limit)")
    if log_every < 1:
        raise ValueError("log_every must be >= 1")
    if not isinstance(scp, JaddleCPStochastic):
        raise TypeError(
            f"scp must be a JaddleCPStochastic, got {type(scp).__name__}: "
            "use jaddle_convex.solve for a JaddleCP"
        )
    if not _is_positive_int(batch_size):
        raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
    if variance_reduction not in _VARIANCE_REDUCTIONS:
        raise ValueError(
            f"variance_reduction must be one of {_VARIANCE_REDUCTIONS}, "
            f"got {variance_reduction!r}"
        )
    if iterations_per_epoch is None:
        largest = max(scp.num_examples.values())
        iterations_per_epoch = max(100, math.ceil(largest / batch_size))
    elif not _is_positive_int(iterations_per_epoch):
        raise ValueError(
            "iterations_per_epoch must be a positive int or None, "
            f"got {iterations_per_epoch!r}"
        )
    svrg = variance_reduction == "svrg"
    if isinstance(step_decay, str):
        if step_decay != "auto":
            raise ValueError(
                f"step_decay must be 'auto', None or a number > 0, got {step_decay!r}"
            )
        step_decay = None if svrg else iterations_per_epoch
    if step_decay is not None and not step_decay > 0:
        raise ValueError(
            f"step_decay must be 'auto', None or a number > 0, got {step_decay!r}"
        )

    if precision is not None:
        # The decorator set the profile; bring the caller's data to it.
        dtype = _wide(jo.jaddle_dtype())
        scp = scp._with_data(
            lambda a: a.astype(dtype) if jnp.issubdtype(a.dtype, jnp.floating) else a
        )
        scp.lower_bounds = jnp.asarray(scp.lower_bounds, dtype=dtype)
        scp.upper_bounds = jnp.asarray(scp.upper_bounds, dtype=dtype)
        if initial_solution is not None:
            initial_solution = SaddleState(
                *(jnp.asarray(v, dtype=dtype) for v in initial_solution)
            )

    if verbose:
        print("----------------------------------------------")

    optimiser, step_size, adaptive_step, k_scaling, k_lo, k_hi = jc._check_settings(
        optimiser, adaptive_eta, step_size, update_mode, k_scale
    )

    if verbose:
        print("====Starting Stochastic Solve====")
        print("----------------------------------------------")

    full = scp.full_cp(eval_chunk_size)
    oracle = scp.oracle(batch_size)
    if initial_solution is None:
        initial_solution = full.initial_solution()

    solve_cp, from_scaled, obj_scale = full, None, 1.0
    if gradient_scaling:
        x0 = projection_box(
            initial_solution.primal, full.lower_bounds, full.upper_bounds
        )
        sigma, e = jax.jit(
            lambda x: jc._gradient_scales(full, x, gradient_scaling_probes)
        )(x0)
        solve_cp, to_scaled, from_scaled, _ = jc._gradient_scaled_from(full, sigma, e)
        m_eq = scp.num_eq_constraints()
        oracle = _scaled_oracle(oracle, sigma, e[:m_eq], e[m_eq:])
        initial_solution = to_scaled(initial_solution)
        obj_scale = float(sigma)

    def decay_fn(global_i):
        return 1.0 / jnp.sqrt(1.0 + global_i / float(step_decay))

    build_chunk, carry = jc._device_solver(
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
        oracle=oracle,
        key=jax.random.PRNGKey(seed),
        svrg=svrg,
        step_decay=None if step_decay is None else decay_fn,
    )
    return jc._run_host_loop(
        build_chunk,
        carry,
        iterations_per_epoch=iterations_per_epoch,
        precompile=precompile,
        verbose=verbose,
        max_epochs=max_epochs,
        max_seconds=max_seconds,
        solve_entry_time=solve_entry_time,
        obj_scale=obj_scale,
        k_scaling=k_scaling,
        diag_adapt=bool(diag_adapt and adaptive_step and restarts),
        average=average,
        from_scaled=from_scaled,
        objective=full.objective,
    )
