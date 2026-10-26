import jax.numpy as jnp
from typing import NamedTuple
import optax
from typing import Any, Callable, NamedTuple, Sequence, Union
import jax
import numpy as np

ScheduleLike = Union[float, Callable[[jnp.ndarray], jnp.ndarray]]


# %%
# Basic Types
class SaddleState(NamedTuple):
    primal: jnp.ndarray
    dual_ineq: jnp.ndarray
    dual_eq: jnp.ndarray


class LPValues(NamedTuple):
    """The numbers of an LP
    ``min cᵀx s.t. A_eq x = b_eq, A_ineq x <= b_ineq, lower <= x <= upper``
    on a fixed sparsity pattern: the constraint
    matrices' stored nonzeros (in their BCOO order) plus the vectors. A pytree,
    so it can be batched with ``vmap`` and differentiated."""

    c: jnp.ndarray
    A_eq_data: jnp.ndarray
    b_eq: jnp.ndarray
    A_ineq_data: jnp.ndarray
    b_ineq: jnp.ndarray
    lower_bounds: jnp.ndarray
    upper_bounds: jnp.ndarray


class JaddleCP:
    def __init__(
        self,
        num_variables,
        objective,
        constraints_eq,
        constraints_ineq,
        lower_bounds,
        upper_bounds,
        dual_bound=None,
        residuals=None,
    ):
        self.num_variables = num_variables
        self.objective = objective
        self.constraints_eq = constraints_eq
        self.constraints_ineq = constraints_ineq
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.dual_bound = dual_bound
        # Optional ``residuals(state) -> (stationarity, feasibility,
        # complementarity)``: replaces the three quantities the solver tests
        # for convergence, for a problem with a better certificate of its own.
        self.residuals = residuals

    def initial_primal_solution(self):
        return jnp.zeros(self.num_variables)

    def num_eq_constraints(self):
        return len(self.constraints_eq(self.initial_primal_solution()))

    def num_ineq_constraints(self):
        return len(self.constraints_ineq(self.initial_primal_solution()))

    def num_constraints(self):
        return self.num_eq_constraints() + self.num_ineq_constraints()

    def ineq_slack(self, x):
        return jnp.max(jnp.maximum(self.constraints_ineq(x), 0.0))

    def eq_slack(self, x):
        return jnp.max(jnp.abs(self.constraints_eq(x)))

    def complementarity_slack(self, x, dual_ineq):
        return dual_ineq * (self.constraints_ineq(x))

    def initial_solution(self):
        return SaddleState(
            primal=jnp.zeros(self.num_variables),
            dual_ineq=jnp.zeros(self.num_ineq_constraints()),
            dual_eq=jnp.zeros(self.num_eq_constraints()),
        )


def _check_cones(cones, lower_bounds, upper_bounds):
    """Normalise an empty ``Cones`` to ``None`` and check that cone variables
    (the tail of x) are unbounded, when the bounds are concrete."""
    if cones is None or not cones:
        return None
    n = np.shape(lower_bounds)[0]
    if cones.size > n:
        raise ValueError(f"{n} variables cannot hold cones of total size {cones.size}")
    for bound in (lower_bounds, upper_bounds):
        if isinstance(bound, jax.core.Tracer):
            continue
        if np.any(np.isfinite(np.asarray(bound)[n - cones.size :])):
            raise ValueError("variables in cone blocks must have infinite bounds")
    return cones


class LP:
    def __init__(
        self,
        c,
        A_eq,
        b_eq,
        A_ineq,
        b_ineq,
        lower_bounds,
        upper_bounds,
        cones=None,
    ):
        self.c = c
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.A_ineq = A_ineq
        self.b_ineq = b_ineq
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.cones = _check_cones(cones, lower_bounds, upper_bounds)

    def objective(self, x):
        return self.c @ x

    def num_variables(self):
        return len(self.c)

    def num_eq_constraints(self):
        return self.A_eq.shape[0]

    def num_ineq_constraints(self):
        return self.A_ineq.shape[0]

    def num_constraints(self):
        return self.A_eq.shape[0] + self.A_ineq.shape[0]

    def ineq_slack(self, x):
        return jnp.max(jnp.maximum(self.A_ineq @ x - self.b_ineq, 0.0), initial=0.0)

    def eq_slack(self, x):
        return jnp.max(jnp.abs(self.A_eq @ x - self.b_eq), initial=0.0)

    def diff_eq_slack(self, x):
        return self.A_eq @ x - self.b_eq

    def complementarity_slack(self, x, dual_ineq):
        return (dual_ineq * (self.A_ineq @ x - self.b_ineq)).sum()


def _row_major_order(rows, cols, shape):
    """Permutation putting the unique COO entries ``(rows, cols)`` in row-major
    order, via scipy's O(nnz) CSR conversion (a counting sort) on the host."""
    import scipy.sparse as sp

    perm = sp.csr_matrix(
        (np.arange(rows.size, dtype=np.int64), (rows, cols)), shape=shape
    )
    perm.sort_indices()
    return perm.data


def _concrete_indices(indices):
    """A sparse matrix's indices as a NumPy array. The sparsity pattern is
    worked out on the host, so it cannot depend on traced values."""
    if isinstance(indices, jax.core.Tracer):
        raise ValueError(
            "the sparsity pattern of an LP's constraint matrices must not depend "
            "on traced values: build it from NumPy / SciPy arrays or from values "
            "that do not depend on the solver's params (the numbers may)"
        )
    return np.asarray(indices)


def _wide(dtype):
    """``dtype`` widened to at least float32. Under the float16 profile only
    the scaled problem data is stored in half precision. The iterates, norms,
    dot products and scalar state (k, eta, merits) use this dtype: float16
    squares overflow above 256 and flush to zero below ~2.4e-4, which turns
    step-size tests into inf/inf = NaN, and float16 iterates lose any step
    smaller than ~1/2000 of their value to rounding."""
    return jnp.promote_types(dtype, jnp.float32)


def _dot(a, b):
    """``vdot(a, b)`` computed and returned in at least float32."""
    w = _wide(jnp.result_type(a, b))
    return jnp.vdot(jnp.asarray(a, w), jnp.asarray(b, w))


def spmv(A, x):
    """``A @ x`` for a BCOO ``A``.

    Below 32 bits (the float16 profile) the sum runs in float32 and the result
    is cast back: JAX's float16 sparse scatter-add is 7-24x slower than float32
    on GPU, and the global matmul-precision setting does not reach it. Float32
    and float64 take the plain ``A @ x`` path.
    """
    import jax.experimental.sparse as _jsp

    dtype = jnp.result_type(A.data, x)
    if jnp.dtype(dtype).itemsize >= 4:
        return A @ x
    out = _jsp.bcoo_dot_general(
        A,
        x,
        dimension_numbers=(([1], [0]), ([], [])),
        preferred_element_type=jnp.float32,
    )
    return out.astype(dtype)


def scipy_to_bcoo(A, dtype):
    """Sorted, deduplicated BCOO of the scipy matrix ``A`` with ``dtype`` data.

    Accepts any scipy sparse format. Sorting, deduplication and the dtype cast
    all happen on the host, so building the device array compiles nothing; the
    cast goes straight from the scipy width to ``dtype`` (float16 included).
    """
    import scipy.sparse as sp
    import jax.experimental.sparse as _jsp

    np_float = np.float64 if jnp.dtype(dtype).itemsize == 8 else np.float32
    # sorted_indices() copies, so sum_duplicates() never touches the caller's A.
    A = sp.csr_matrix(A, dtype=np_float).sorted_indices()
    A.sum_duplicates()
    rows = np.repeat(np.arange(A.shape[0], dtype=np.int32), np.diff(A.indptr))
    indices = np.column_stack([rows, A.indices.astype(np.int32)])
    # jax.device_put, not jnp.asarray: the latter compiles a per-shape
    # jit(stage) for every numpy upload.
    return _jsp.BCOO(
        (jax.device_put(A.data.astype(jnp.dtype(dtype))), jax.device_put(indices)),
        shape=A.shape,
        indices_sorted=True,
        unique_indices=True,
    )


class JaddleLP:
    def __init__(
        self,
        c,
        A_eq,
        b_eq,
        A_ineq,
        b_ineq,
        lower_bounds,
        upper_bounds,
        cones=None,
    ):
        self.c = c
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.A_ineq = A_ineq
        self.b_ineq = b_ineq
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.cones = _check_cones(cones, lower_bounds, upper_bounds)
        # Fused [A_eq; A_ineq] for 2-matvec gradient computation.
        #
        # Both A and Aᵀ are kept as BCOO. BCSR was benchmarked here (Aᵀ
        # materialised as its own row-major CSR, not a lazy `.T`) and was MUCH
        # slower than BCOO on this workload — do not switch back to BCSR without
        # re-benchmarking; BCOO's matvec wins for these matrices/precision/device.
        #
        # Aᵀ is stored as an explicit transposed BCOO (column-swapped indices)
        # rather than the lazy `self.A.T`, so `Aᵀ @ y` runs its own matvec
        # instead of the transposed-operand code path.
        #
        # JAX's BCOO matvec has a faster kernel for sorted+unique indices (and
        # the unsorted path can't skip duplicate accumulation), so both are
        # stored row-major sorted. The LP has no duplicate (row, col) entries,
        # so unique_indices is safe to set. The sort is done on the HOST:
        # BCOO.sort_indices() is a device lax.sort compiled per shape, which was
        # ~2-3 s of every solve's setup (stp3d) for a permutation scipy's O(nnz)
        # counting sort finds in milliseconds.
        import jax.experimental.sparse as _jsp

        self.n_eq = A_eq.shape[0]
        m, n = A_eq.shape[0] + A_ineq.shape[0], A_eq.shape[1]
        eq_idx = _concrete_indices(A_eq.indices)
        ineq_idx = _concrete_indices(A_ineq.indices)
        idx = np.concatenate([eq_idx, ineq_idx + np.array([self.n_eq, 0], eq_idx.dtype)])
        # The numbers may be traced (an LP built inside `jit` / `vmap` / `grad`,
        # as in `jaddle_linear.make_solver`); then they stay JAX arrays and only
        # the sparsity pattern is handled on the host.
        traced = any(
            isinstance(x, jax.core.Tracer) for x in (A_eq.data, A_ineq.data, b_eq, b_ineq)
        )
        xp = jnp if traced else np
        put = (lambda x: x) if traced else jax.device_put
        data = xp.concatenate([xp.asarray(A_eq.data), xp.asarray(A_ineq.data)])

        # Stacking two row-sorted blocks (ineq rows offset below eq) is sorted.
        order = None
        if not (A_eq.indices_sorted and A_ineq.indices_sorted):
            order = _row_major_order(idx[:, 0], idx[:, 1], (m, n))
            idx, data = idx[order], data[order]
        order_T = _row_major_order(idx[:, 1], idx[:, 0], (n, m))
        # The permutations depend only on the sparsity pattern; `with_values`
        # reuses them to rebuild A and Aᵀ from new values without host work.
        self._order, self._order_T = order, order_T

        def bcoo(d, i, shape):
            return _jsp.BCOO(
                (put(d), jax.device_put(i)),
                shape=shape,
                indices_sorted=True,
                unique_indices=True,
            )

        self.A = bcoo(data, idx, (m, n))
        self.A_T = bcoo(
            data[order_T], np.ascontiguousarray(idx[order_T][:, ::-1]), (n, m)
        )
        self.b = put(xp.concatenate([xp.asarray(b_eq), xp.asarray(b_ineq)]))

    @classmethod
    def from_scipy(
        cls, c, A_eq, b_eq, A_ineq, b_ineq, lower_bounds, upper_bounds, cones=None
    ):
        """Build a ``JaddleLP`` directly from scipy sparse blocks + numpy vectors.

        Mirrors ``jaddle_linear.to_jaddle_sparse``: resolves the active precision
        profile (float64 under x64, else float32; the float16 profile also
        builds float32 here, since only the scaled copy the solver iterates on
        is stored in float16), converts each constraint block to a sorted,
        deduplicated BCOO, and casts the data array to that dtype. This is
        the JAX-native entry point — callers (e.g. ``highs_to_standard_form_sparse``)
        no longer route through a scipy ``LP``.
        """
        from jaddle.jaddle_optimisers import jaddle_dtype

        # At least float32: float16 is only used for the scaled copy the
        # solver iterates on (raw coefficients overflow its range).
        float_dtype = jnp.promote_types(jaddle_dtype(), jnp.float32)
        if float_dtype == jnp.float64 and not jax.config.jax_enable_x64:
            float_dtype = jnp.float32

        return cls(
            jnp.asarray(c, dtype=float_dtype),
            scipy_to_bcoo(A_eq, float_dtype),
            jnp.asarray(b_eq, dtype=float_dtype),
            scipy_to_bcoo(A_ineq, float_dtype),
            jnp.asarray(b_ineq, dtype=float_dtype),
            jnp.asarray(lower_bounds, dtype=float_dtype),
            jnp.asarray(upper_bounds, dtype=float_dtype),
            cones=cones,
        )

    def values(self):
        """The LP's numbers as an ``LPValues`` pytree (the sparsity pattern
        stays with the ``JaddleLP``)."""
        return LPValues(
            c=self.c,
            A_eq_data=self.A_eq.data,
            b_eq=self.b_eq,
            A_ineq_data=self.A_ineq.data,
            b_ineq=self.b_ineq,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )

    def with_values(self, values):
        """A ``JaddleLP`` with this one's sparsity pattern and new numbers.

        Pure JAX (gathers with the permutations saved at construction, no
        host-side sorting), so it works on traced values under ``jit``,
        ``vmap`` and ``grad``.
        """
        import jax.experimental.sparse as _jsp

        def bcoo(data, like):
            return _jsp.BCOO(
                (data, like.indices),
                shape=like.shape,
                indices_sorted=like.indices_sorted,
                unique_indices=like.unique_indices,
            )

        out = object.__new__(type(self))
        out.c = values.c
        out.b_eq, out.b_ineq = values.b_eq, values.b_ineq
        out.lower_bounds, out.upper_bounds = values.lower_bounds, values.upper_bounds
        out.A_eq = bcoo(values.A_eq_data, self.A_eq)
        out.A_ineq = bcoo(values.A_ineq_data, self.A_ineq)
        out.n_eq = self.n_eq
        out.cones = self.cones
        out._order, out._order_T = self._order, self._order_T
        data = jnp.concatenate([values.A_eq_data, values.A_ineq_data])
        if self._order is not None:
            data = data[self._order]
        out.A = bcoo(data, self.A)
        out.A_T = bcoo(data[self._order_T], self.A_T)
        out.b = jnp.concatenate([values.b_eq, values.b_ineq])
        return out

    def objective(self, x):
        return self.c @ x

    def num_variables(self):
        return len(self.c)

    def num_eq_constraints(self):
        return self.A_eq.shape[0]

    def num_ineq_constraints(self):
        return self.A_ineq.shape[0]

    def num_constraints(self):
        return self.A_eq.shape[0] + self.A_ineq.shape[0]

    def ineq_slack(self, x):
        return jnp.max(jnp.maximum(spmv(self.A_ineq, x) - self.b_ineq, 0.0), initial=0.0)

    def eq_slack(self, x):
        return jnp.max(jnp.abs(spmv(self.A_eq, x) - self.b_eq), initial=0.0)

    def diff_eq_slack(self, x):
        return spmv(self.A_eq, x) - self.b_eq

    def complementarity_slack(self, x, dual_ineq):
        return (dual_ineq * (spmv(self.A_ineq, x) - self.b_ineq)).sum()

    def initial_solution(self):
        """Default start point: the box-projected zero primal with zero duals
        (the PDLP default). Built on the host and uploaded, so it compiles
        nothing."""
        lb, ub = np.asarray(self.lower_bounds), np.asarray(self.upper_bounds)
        dtype = jnp.result_type(float, lb.dtype)
        return SaddleState(
            primal=jax.device_put(np.clip(np.zeros(lb.shape, dtype), lb, ub)),
            dual_ineq=jax.device_put(np.zeros(self.num_ineq_constraints(), dtype)),
            dual_eq=jax.device_put(np.zeros(self.num_eq_constraints(), dtype)),
        )

    def to_scipy(self) -> "LP":
        """Materialise a scipy-backed ``LP`` (CSC matrices, numpy vectors).

        ``JaddleLP`` is the device-side representation the saddle solver iterates
        on; the scaling, primal/dual polish, crossover and eq-projection LU stages
        run host-side on scipy sparse (sparse direct solve, ``lsmr``, boolean
        row-slicing — none of which JAX provides). Those stages call this to get
        the scipy view they need; the solver never does.
        """
        import scipy.sparse as sp

        def _to_csc(bcoo):
            data = np.asarray(bcoo.data)
            idx = np.asarray(bcoo.indices)
            return sp.csc_matrix(
                (data, (idx[:, 0], idx[:, 1])),
                shape=bcoo.shape,
                dtype=np.float64,
            )

        return LP(
            np.asarray(self.c, dtype=np.float64),
            _to_csc(self.A_eq),
            np.asarray(self.b_eq, dtype=np.float64),
            _to_csc(self.A_ineq),
            np.asarray(self.b_ineq, dtype=np.float64),
            np.asarray(self.lower_bounds, dtype=np.float64),
            np.asarray(self.upper_bounds, dtype=np.float64),
            cones=self.cones,
        )


# %%
