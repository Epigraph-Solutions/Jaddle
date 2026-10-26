import math
import os
import warnings

import optax
import jax
import jax.numpy as jnp
from typing import Any, Optional
from jaddle.jaddle_basic_types import (
    ScheduleLike,
    LP,
    SaddleState,
)

# The three supported precision profiles. Each is named after the JAX array
# dtype it implies, which solver code reads via jaddle_dtype() to allocate
# state in the active precision.
_PROFILES = ("float64", "float32", "float16")


def _active_profile(profile: Optional[str] = None) -> str:
    if profile is None:
        profile = os.environ.get("JADDLE_JAX_PROFILE", "float64")
    profile = profile.lower()
    if profile not in _PROFILES:
        raise ValueError(
            f"Unknown JAX profile {profile!r}; expected one of {_PROFILES}."
        )
    return profile


def jaddle_dtype():
    """Return the JAX dtype for the currently-configured precision profile."""
    return getattr(jnp, _active_profile())


def configure_jax(jax_profile: Optional[str] = None):
    """
    Configure JAX for one of three precision/performance profiles.

    Args:
        jax_profile: One of:
            - "float64": double precision, maximum safety and accuracy. The
              analogue of PDLP's float64 default. On wide-dynamic-range MIPLIB
              rows, float32 residuals plateau on rounding and "stalled" becomes
              indistinguishable from "out of precision". Full-precision matmuls
              (no TF32) so the extra mantissa actually counts.
            - "float32": single precision, maximum speed. TF32 matmuls.
            - "float16": the scaled LP or QP matrices the solvers iterate on
              are stored in half precision; iterates, norms, step sizes and
              certificates stay float32. Solves reach relative tolerances of
              about 1e-3; for LPs, use solve_with_refinement for tighter ones.
              General convex problems (jaddle_convex.solve) take the dtype of
              their own data.
            If None, reads JADDLE_JAX_PROFILE, defaulting to "float64". Any
            other name raises ValueError.

    Call it after importing jax/jaddle but before creating any arrays or running
    any JAX code. Options JAX reads at import (x64, matmul precision, compile
    cache) are set on the live jax.config; XLA_FLAGS and the allocator env vars
    are read when the backend initialises, so they only apply if no device has
    been touched yet (a warning is raised otherwise). Prints the active
    configuration for verification.
    """
    try:
        from jax._src import xla_bridge

        if xla_bridge.backends_are_initialized():
            warnings.warn(
                "configure_jax called after the JAX backend initialised; "
                "XLA_FLAGS and XLA_PYTHON_CLIENT_* settings will not take "
                "effect. Call it before creating any arrays.",
                stacklevel=2,
            )
    except (ImportError, AttributeError):
        pass

    PROFILE = _active_profile(jax_profile)
    os.environ["JADDLE_JAX_PROFILE"] = PROFILE

    def _append_xla_flag(flag: str):
        # Respect a value the user already set for this flag (match on name).
        name = flag.split("=", 1)[0]
        current = os.environ.get("XLA_FLAGS", "")
        if not any(f.split("=", 1)[0] == name for f in current.split()):
            os.environ["XLA_FLAGS"] = f"{current} {flag}".strip()

    def _set_matmul_precision(default: str):
        # The env var is read only at jax import, so apply it via the live
        # config. A user-set JAX_DEFAULT_MATMUL_PRECISION still wins.
        precision = os.environ.get("JAX_DEFAULT_MATMUL_PRECISION", default)
        jax.config.update("jax_default_matmul_precision", precision)

    # Suppress INFO and WARNING logs from XLA/JAX
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    cache_dir = os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR",
        os.path.expanduser("~/.cache/jaddle_jax"),
    )
    if cache_dir:
        # Persistent XLA compilation cache. The env var alone does nothing once
        # jax is imported (it is read at import time), so set the live config.
        # Cache every compile, not just the >=1 s default: a solve's ~30 small
        # setup kernels each cost 10-60 ms to compile, and a re-run of the same
        # instance (A/B sweeps) then skips all of them plus the ~2 s epoch-loop
        # compile. Set JAX_COMPILATION_CACHE_DIR="" to disable.
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        # LRU cap at 10 GB (one instance's epoch loop is a few MB). JAX needs
        # `filelock` for the cap and otherwise silently skips EVERY cache
        # read/write, so only set it when available (else the cache is unbounded).
        try:
            import filelock  # noqa: F401

            jax.config.update("jax_compilation_cache_max_size", 10 * 2**30)
        except ImportError:
            pass

    if PROFILE == "float64":
        # Double precision: enable x64 and use full-precision (highest) matmuls.
        os.environ["JAX_ENABLE_X64"] = "1"
        # The env var only takes effect if read before JAX initialises; since
        # this module already imported jax, set the live config too so x64
        # works even when configure_jax is called after import.
        jax.config.update("jax_enable_x64", True)
        _set_matmul_precision("highest")
        dtype = "float64"
    else:
        # float32 / float16: maximum speed. No x64; aggressive GPU autotuning
        # and memory preallocation. (JAX has no global "default to 16-bit" mode
        # — JAX_ENABLE_X64 only switches float32<->float64 — so half precision
        # is carried by the dtype that jaddle_dtype() hands to state allocation,
        # not by a config flag.)
        os.environ["JAX_ENABLE_X64"] = "0"
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
        _append_xla_flag("--xla_gpu_autotune_level=5")
        jax.config.update("jax_enable_x64", False)
        dtype = PROFILE

        if PROFILE == "float16":
            # Half precision: accumulate dense matmuls in float32 so f16 inputs
            # don't lose the running sum to rounding. This setting does not
            # reach JAX's sparse matvec (jaddle_basic_types.spmv sums in float32
            # itself), and the LP solver keeps its iterates and scalars in
            # float32 regardless (jaddle_linear._wide).
            _set_matmul_precision("float32")
        else:
            # float32: TF32 matmuls trade a few mantissa bits for throughput.
            _set_matmul_precision("tensorfloat32")

    print(
        "[JAX Profile] "
        f"mode={PROFILE}, "
        f"dtype={dtype}, "
        f"x64={jax.config.jax_enable_x64}, "
        f"matmul_precision={jax.config.jax_default_matmul_precision or 'default'}, "
        f"cache={jax.config.jax_compilation_cache_dir or 'disabled'}, "
        f"xla_flags='{os.environ.get('XLA_FLAGS', '')}'"
    )


def _saddle_param_labels(params):
    if hasattr(params, "_fields"):
        return type(params)(
            primal="primal_opt",
            dual_ineq="dual_opt",
            dual_eq="dual_opt",
        )
    return {
        "primal": "primal_opt",
        "dual_ineq": "dual_opt",
        "dual_eq": "dual_opt",
    }


def create_saddle_optimiser(
    primal_optimizer: optax.GradientTransformation,
    dual_optimizer: optax.GradientTransformation,
):
    """Combine a primal and a dual Optax optimiser into one for a ``SaddleState``.

    The primal optimiser updates ``primal`` and the dual optimiser updates
    ``dual_ineq`` and ``dual_eq``. Pass the result as ``optimiser`` to
    ``jaddle_convex.solve`` in ``"alternating"`` mode.

    Args:
        primal_optimizer: Update rule for the primal variables.
        dual_optimizer: Update rule for the dual variables. If ``None``, the
            primal optimiser is used for both.

    Returns:
        optax.GradientTransformation: The combined optimiser.
    """
    if dual_optimizer is None:
        dual_optimizer = primal_optimizer
    optimiser = optax.partition(
        {
            "primal_opt": primal_optimizer,
            "dual_opt": dual_optimizer,
        },
        param_labels=_saddle_param_labels,
    )

    return optimiser


def gd(lr):
    """Plain gradient descent for both players, with learning rate ``lr``."""
    primal =optax.inject_hyperparams(optax.sgd)(
        learning_rate=lr,
    )
    dual = optax.inject_hyperparams(optax.sgd)(
        learning_rate=lr,
    )
    return create_saddle_optimiser(
        primal,
        dual,
    )


def optimistic_gd(lr):
    """Optimistic gradient descent for both players, with learning rate ``lr``.

    Each step adds a correction from the previous gradient, which damps the
    rotation that plain gradient descent shows on saddle-point problems.
    """
    primal =optax.inject_hyperparams(optax.optimistic_gradient_descent)(
        learning_rate=lr,
    )
    dual = optax.inject_hyperparams(optax.optimistic_gradient_descent)(
        learning_rate=lr,
    )
    return create_saddle_optimiser(
        primal,
        dual,
    )


def optimistic_adadelta(lr):
    """Adadelta-scaled optimistic gradient descent for both players.

    Gradients are rescaled per coordinate by Adadelta, then applied with an
    optimistic step of learning rate ``lr``.
    """
    primal =optax.chain(
        optax.scale_by_adadelta(),
        optax.optimistic_gradient_descent(lr),
    )
    dual = optax.chain(
        optax.scale_by_adadelta(),
        optax.optimistic_gradient_descent(lr),
    )
    return create_saddle_optimiser(
        primal,
        dual,
    )


def gd_dual_momentum(lr, momentum=0.3, nesterov=True):
    """Gradient descent for the primal player and momentum for the dual player.

    Args:
        lr: Learning rate for both players.
        momentum: Momentum coefficient of the dual player.
        nesterov: Use Nesterov momentum for the dual player.
    """
    primal =optax.inject_hyperparams(optax.sgd)(
        learning_rate=lr,
    )
    dual = optax.inject_hyperparams(optax.sgd)(
        learning_rate=lr,
        momentum=momentum,
        nesterov=nesterov,
    )
    return create_saddle_optimiser(
        primal,
        dual,
    )


def tail_average(i_max):
    """Returns a function that computes the tail average of a sequence."""

    def weight_fn(i):
        return jax.lax.cond(
            i < i_max,
            lambda: 1 / i_max,
            lambda: 1.0,
        )

    return weight_fn


def _rebalance_diag_part(r, H, d, theta, log_clamp):
    active = H > 1e-30
    rho = r / jnp.sqrt(jnp.where(active, H, 1.0))
    log_rho = jnp.log(jnp.maximum(rho, 1e-6))
    n_active = jnp.maximum(jnp.sum(active), 1)
    mean = jnp.sum(jnp.where(active, log_rho, 0.0)) / n_active
    u = jnp.where(active, log_rho - mean, 0.0)
    return jnp.exp(jnp.clip(jnp.log(d) + theta * u, -log_clamp, log_clamp))


def rebalance_diag(new_state, ref_state, diag, accum, theta, clamp):
    """``diag_adapt``'s restart update of a per-coordinate diagonal metric
    ``diag = (t, s)`` (primal; dual stacked ``[eq; ineq]``), shared by the
    linear and convex solvers.

    ``accum = (H_x, H_y)`` holds each coordinate's ``Σ (Δ/eta)²`` over the
    cycle from ``ref_state`` to ``new_state``. Per coordinate,
    ``rho = |net displacement| / sqrt(H)`` is ~√N for one drifting steadily and
    ≪ √N for one oscillating (the DoG distance-over-gradients ratio). Its log,
    centred over the coordinates that moved, is a multiplicative step
    (``theta``) on the diagonal, clamped to ``[1/clamp, clamp]`` around 1.
    Centring keeps the geometric mean of the moving coordinates fixed, so the
    solvers' scalar step ``eta`` and primal weight ``k`` stay the only gauges;
    coordinates that never moved (pinned on a bound, slack rows) carry no
    signal and keep their value."""
    t, s = diag
    H_x, H_y = accum
    log_clamp = math.log(clamp)
    r_x = jnp.abs(new_state.primal - ref_state.primal)
    r_y = jnp.abs(
        jnp.concatenate(
            [
                new_state.dual_eq - ref_state.dual_eq,
                new_state.dual_ineq - ref_state.dual_ineq,
            ]
        )
    )
    return (
        _rebalance_diag_part(r_x, H_x, t, theta, log_clamp),
        _rebalance_diag_part(r_y, H_y, s, theta, log_clamp),
    )


def diag_log_spread(d):
    """log10(max / min) of a ``diag_adapt`` diagonal; 0 when empty."""
    return jnp.log10(jnp.max(d, initial=1.0) / jnp.min(d, initial=1.0))
