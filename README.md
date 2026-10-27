<img width="1536" height="1024" alt="jaddle_logo_full" src="https://github.com/user-attachments/assets/d4e61518-3557-4273-b4d0-0d89510e0f3f" />

[![PyPI](https://img.shields.io/pypi/v/jaddle)](https://pypi.org/project/jaddle/)
[![Python](https://img.shields.io/pypi/pyversions/jaddle)](https://pypi.org/project/jaddle/)
[![Tests](https://github.com/BrendanVR/Jaddle/actions/workflows/tests.yml/badge.svg)](https://github.com/BrendanVR/Jaddle/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/BrendanVR/Jaddle/blob/main/LICENSE.txt)

# Jaddle: The JAX Saddle Solver

*First-order primal–dual solvers for large-scale linear and convex programs, written entirely in JAX.*

Jaddle solves constrained optimisation problems by casting them as saddle-point
problems on the Lagrangian and running primal–dual iterations to convergence.
It provides two solvers:

- **`jaddle.jaddle_linear`**: a restarted, adaptively step-sized PDHG-family
  solver for sparse linear programs, in the tradition of PDLP and cuPDLP.
- **`jaddle.jaddle_convex`**: a general solver for smooth convex programs. The
  user supplies the objective and constraints as ordinary JAX functions, and
  Jaddle derives everything else by automatic differentiation.

On top of the LP solver, **`jaddle.jaddle_mip`** is a simple branch-and-bound
solver for mixed-integer linear programs. It is **highly experimental**: its
interface, defaults and behaviour may change without notice, and its results
should be checked against an established MIP solver.

Both solvers run unchanged on CPU, GPU and TPU. Jaddle is not intended to
replace a commercial or production-grade solver. It aims to be a compact,
readable and hackable platform for first-order optimisation that still holds
up on demanding benchmarks.

## Contents

- [Why JAX](#why-jax)
- [Installation](#installation)
- [Quick start](#quick-start)
- [The linear programming solver](#the-linear-programming-solver)
- [The mixed-integer solver](#the-mixed-integer-solver)
- [The convex programming solver](#the-convex-programming-solver)
- [Modelling with CVXPY](#modelling-with-cvxpy)
- [Numerical precision](#numerical-precision)
- [Benchmarks](#benchmarks)
- [Repository layout](#repository-layout)

## Why JAX

Jaddle has no hand-written kernels. Every solver in the library is expressed
in high-level JAX, and JAX's compiler and transformations supply the
performance and portability. This brings several concrete benefits.

**Compiled performance from Python.** Each solver's whole iteration loop is
traced once and compiled by XLA into a single program. Primal and dual updates,
step size line searches, projections, convergence checks and restart decisions
all run on the accelerator, many epochs at a time, without returning control to
Python. Python steps in only about once a second, to enforce time limits and
print progress, and not at all when there is no logging or budget to enforce.
The LP solver is therefore limited by the speed of sparse matrix–vector
products, which is the limit any first-order LP method faces, rather than by
interpreter overhead or host–device synchronisation.

**Hardware portability.** The same source runs on a laptop CPU, a consumer GPU
or a TPU pod. There are no device-specific code paths to maintain, and moving
between hardware targets requires no code changes. Switching from CPU to GPU is
a matter of installing the appropriate `jax` wheel.

**Automatic differentiation.** The convex solver needs only the objective and
the constraint functions. Gradients of the Lagrangian with respect to the
primal and dual variables come from `jax.grad`, so users never derive,
implement or debug a gradient by hand. Any function JAX can trace can be used
as an objective or constraint, from a least-squares loss to a neural network.
LP solves are differentiable in turn: the optimal value and solution can be
differentiated with respect to the LP's data, and solves batch with `vmap`
(see [Batched and differentiable solves](#batched-and-differentiable-solves)).

**Composable optimisation primitives.** The convex solver builds on
[Optax](https://github.com/google-deepmind/optax). It uses Optax's projection
operators, and with `adaptive_eta=False` it accepts any Optax gradient
transformation as the update rule. A new primal–dual variant is often only a
few lines of code: for example, giving the primal player plain gradient descent
and the dual player Nesterov momentum.

**Precision as a configuration choice.** A single call switches the whole
library between float64, float32 and float16 arithmetic. Nothing is
duplicated per precision, so the trade-off between accuracy and throughput can
be explored without code changes.

**Functional state.** Solver state is an immutable pytree (`SaddleState`, plus
the step size state). Warm starting, checkpointing and resuming a solve
therefore amount to passing a value back in. The solver keeps no hidden mutable
state.

**A small, readable codebase.** Because the compiler handles fusion, memory
layout and device placement, the full library is roughly 8,600 lines of
Python, docstrings included. Algorithms remain close to their mathematical description, which makes
Jaddle well suited to research and experimentation.

## Installation

Jaddle requires Python 3.11 or later and is available on
[PyPI](https://pypi.org/project/jaddle/):

```bash
pip install jaddle
```

This installs the core dependencies: `jax`, `optax`, `numpy`, `scipy` and
`highspy`, and runs on CPU. Optional features are available as extras, which
can be combined, for example `pip install "jaddle[cuda12,examples]"`:

| Extra | Installs | Used for |
|---|---|---|
| `cuda12`, `cuda13` | CUDA-enabled JAX | GPU execution. Choose the extra that matches your CUDA driver; see the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for TPU and other platforms. |
| `examples` | `matplotlib`, `scikit-learn` | The convex examples |
| `rag_examples` | `fastembed`, `snowballstemmer`, and `examples` | The retrieval (RAG) examples |
| `benchmarks` | `mpax` | The benchmarks |
| `cvxpy` | `cvxpy` | [Modelling with CVXPY](#modelling-with-cvxpy) |
| `test` | `pytest`, `cvxpy` | Running the test suite |
| `dev` | `examples`, `test`, `build`, `twine` | Development and packaging |

Jaddle is developed on Linux, including WSL 2. Continuous integration tests
it on Python 3.11, 3.12 and 3.13.

To work on Jaddle itself, or to run the examples, benchmarks and tests, install
from a clone of the repository instead:

```bash
git clone https://github.com/BrendanVR/Jaddle.git
cd Jaddle
pip install -e ".[dev]"
```

### Running the tests

From the repository root:

```bash
pytest
```

The suite runs on CPU in double precision and takes well under a minute. It
checks both solvers against exact or reference solutions (SciPy's HiGHS
interface for LPs, closed-form solutions for the convex problems, Clarabel
for the CVXPY front end), the defined-variable presolve, and the
self-contained examples. GitHub Actions runs
it on every push and pull request.

## Quick start

### A linear program

```python
import numpy as np
import scipy.sparse as sp
import jaddle.jaddle_linear as jl

# minimise 3 x1 + 2 x2  subject to  x1 + x2 >= 4,  x >= 0
lp = jl.LP(
    c=np.array([3.0, 2.0]),
    A_eq=sp.csc_matrix((0, 2)), b_eq=np.zeros(0),
    A_ineq=sp.csc_matrix([[-1.0, -1.0]]), b_ineq=np.array([-4.0]),
    lower_bounds=np.zeros(2), upper_bounds=np.full(2, np.inf),
)

result = jl.solve(lp, verbose=True)
x = result["solution"].primal
print(result["stop_reason"], lp.objective(x))
```

### A convex program

```python
import jax.numpy as jnp
import jaddle.jaddle_convex as jc

y = jnp.linspace(-1.0, 1.0, 100) ** 3   # data to fit

cp = jc.JaddleCP(
    num_variables=100,
    objective=lambda p: jnp.sum((p - y) ** 2),
    constraints_eq=lambda p: jnp.zeros(0),
    constraints_ineq=lambda p: p[:-1] - p[1:],  # p must be non-decreasing
    lower_bounds=-jnp.ones(100),
    upper_bounds=jnp.ones(100),
)

result = jc.solve(cp)
p = result["solution"].primal
```

The [`examples/`](https://github.com/BrendanVR/Jaddle/blob/main/examples/README.md) directory contains complete, annotated
versions of both, together with examples that load and solve MIPLIB instances.

## The linear programming solver

### Problem form

`jaddle_linear.solve` accepts linear programs in the standard form

```
minimise    cᵀx
subject to  A_eq   x  = b_eq
            A_ineq x <= b_ineq
            l <= x <= u
```

where bounds may be infinite. Problems can be built directly from NumPy and
SciPy sparse arrays with `jl.LP`. They can also be converted from MPS files
through HiGHS (`highs_helpers.highs_to_standard_form_sparse`). Internally, constraint matrices are stored as
JAX `BCOO` sparse arrays.

The solver seeks a saddle point of the Lagrangian

```
L(x, y) = cᵀx + y_eqᵀ(A_eq x − b_eq) + y_ineqᵀ(A_ineq x − b_ineq),   y_ineq >= 0,
```

minimising over `x` within its bounds and maximising over the duals `y`.

### Algorithm

Each iteration takes a projected primal descent step and a projected dual
ascent step. The key components follow the design of PDLP and cuPDLP, with
some refinements of our own.

**Update schemes** (`update_mode`)

| Mode | Description |
|---|---|
| `"alternating"` (default) | Gauss–Seidel primal–dual iteration: a primal step, then a dual step evaluated at the new primal iterate. |
| `"pdhg"` | Chambolle–Pock PDHG: the dual step is taken at the extrapolated point `2x⁽ᵏ⁺¹⁾ − x⁽ᵏ⁾`. |
| `"halpern"` | Restarted Halpern-anchored PDHG. Each step blends the PDHG update back towards an anchor point, which accelerates convergence of the last iterate. Best used with restarts. |

**Adaptive step size.** All modes use a cuPDLP-style adaptive step. At every
iteration the step `η` is checked against a local estimate of the interaction
term `(Δy)ᵀ A (Δx)`. A step that overshoots is rejected and shrunk. An accepted
step is advanced conservatively. The step size learned during a solve is
carried across restarts. `adaptive_eta=False` turns the line search off and
holds `η` at `step_size` (plain PDHG; `step_size="auto"` picks a step that is
stable for the scaled problem).

**Primal weight.** The primal and dual step sizes are `η / k` and `η · k`. The
primal weight `k` is initialised from the ratio `‖c‖ / ‖b‖`. At each restart it
is rebalanced from the relative movement of the primal and dual iterates, using
a log-space blend controlled by `k_theta` and clamped to
`[1/k_scale, k_scale]`.

**Adaptive restarts.** A restart resets the averaging (and, for `halpern`, the
anchor) and keeps the current iterate as a warm start. This stops the
iteration from settling into slow rotational orbits. Restarts fire on any of
three conditions, all judged by a normalised KKT merit:

- *sufficient progress*: the merit falls below `restart_decay` times its value
  at the last restart;
- *stalling*: the merit falls below `necessary_decay` times its value at the
  start of the cycle and then starts to rise;
- *no progress*: the cycle reaches its length cap.

The merit can be checked within an epoch (`restart_check_every`), so a restart
does not have to wait for the epoch boundary.

**Diagonal preconditioning.** Before solving, the problem is equilibrated by
Ruiz (L∞) scaling followed by Pock–Chambolle (L1) scaling. By default the
scaling is applied to the augmented matrix `[[A, b], [cᵀ, 0]]`, so cost and
right-hand-side magnitudes also inform the scaling. The objective and
right-hand side are then normalised. All reported quantities are mapped back
to the original problem's units.

**Termination.** The solver stops when the standard LP optimality certificate
holds to a relative tolerance, following the PDLP and HiGHS conventions:

| Test | Normalised by | Tolerance argument |
|---|---|---|
| Primal feasibility | `1 + ‖b‖` | `primal_feasibility_tolerance` |
| Dual feasibility | `1 + ‖c‖` | `dual_feasibility_tolerance` |
| Duality gap | `1 + \|primal obj\| + \|dual obj\|` | `dual_gap_tolerance` |

All three tolerances default to `1e-3`. The norms are L2 by default
(`termination_norm`). Reduced costs are split between the dual residual and
the dual objective as in PDLP, so far-away finite bounds do not swamp the gap.

### Principal options

| Argument | Default | Purpose |
|---|---|---|
| `update_mode` | `"alternating"` | Iteration scheme (see above). |
| `iterations_per_epoch` | `256` | Iterations per compiled epoch. Metrics, logging and termination checks run between epochs. |
| `max_epochs`, `max_seconds` | `None` | Iteration and wall-clock budgets. `max_seconds` includes setup and compilation. |
| `restarts`, `epochs_per_restart` | `True`, `None` | Adaptive restarts and the length cap of a restart cycle (`None` = no cap). |
| `restart_decay`, `necessary_decay` | `0.2`, `0.8` | Thresholds for the sufficient-progress and stalling restarts. |
| `k_scale`, `k_theta`, `k_init` | `1e8`, `0.5`, `None` | Clamp band, smoothing and initial value for the primal weight. |
| `adaptive_eta`, `step_size` | `True`, `1.0` | Line search on the step `η` (off: `η` stays fixed), and the starting `η` (`"auto"` = `0.99/‖[[A, b], [cᵀ, 0]]‖₂`). |
| `diag_adapt`, `diag_theta`, `diag_clamp` | `True`, `0.5`, `10.0` | Per-coordinate diagonal step metric on top of the scaling, adapted at each restart (larger steps for steadily drifting coordinates, smaller for oscillating ones). |
| `ruiz_iterations`, `pc_iterations` | `10`, `1` | Number of preconditioning sweeps. |
| `initial_solution`, `initial_opt_state` | `None` | Warm start from a previous solve's `solution` and `opt_state`. |
| `vertex_bias` | `0.0` | Small cost perturbation that steers the solver to a vertex of the optimal face rather than its interior. Useful before `jl.crossover`. |
| `verbose`, `log_every` | `False`, `1` | Per-epoch progress logging. |

### Results

`solve` returns a dictionary with the following keys:

| Key | Contents |
|---|---|
| `solution` | A `SaddleState(primal, dual_ineq, dual_eq)` in the original problem's units. |
| `converged` | Whether a termination criterion was met. |
| `stop_reason` | `"certificate"`, `"primal_stall"` (only when the opt-in `primal_stop` heuristic is enabled), `"max_epochs"`, `"time_limit"` or `"interrupted"`. |
| `opt_state` | Final step size and primal weight, for warm starting. |
| `solve_seconds` | Time spent in the iteration loop, including the first-epoch compile. |
| `corrected_seconds` | Iteration-loop time with the one-off compile amortised out. |
| `epochs` | Number of epochs run. |

### Batched and differentiable solves

As in the convex front end, write a function `build_lp(params)` that returns
an ordinary `jl.LP` (or `JaddleLP`) whose numbers depend on `params`, any
pytree. `jl.make_solver(build_lp, **options)` turns `solve()` into a pure JAX
function of `params`. It takes the same options and reproduces `solve()`
exactly, but it has no logging, time limit or `vertex_bias`. The function
can be jitted, batched with `jax.vmap` and differentiated:

```python
def build_lp(price):
    return jl.LP(c=price, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq,
                 lower_bounds=lb, upper_bounds=ub)

solve_fn = jax.jit(jl.make_solver(build_lp, max_epochs=1000))
result = solve_fn(price)                 # SolveCoreResult: solution, converged, ...
batch = jax.vmap(solve_fn)(prices)       # many LPs at once
```

The LPs of one function share a sparsity pattern, which is read from whatever
does not depend on `params` (`build_lp` runs under
`jax.ensure_compile_time_eval`, so this holds under `jit` too). SciPy matrices
and dense matrices that don't depend on `params` keep their nonzeros. A dense
matrix that does depend on them stores every entry. To reuse a `JaddleLP`'s
pattern with new numbers, `jl.make_solver(jaddle_lp.with_values)` takes a
`jl.LPValues` pytree as `params`: the cost, the stored nonzeros of `A_eq` and
`A_ineq`, the right-hand sides and the bounds (`jaddle_lp.values()` extracts
them).

In a batch the cost, right-hand sides, bounds and matrix values can all vary.
`jl.solve_batch(build_lp, params, in_axes=0)` is the jitted `vmap`. Each
member is scaled as `solve()` would scale it, and finished members wait for
the slowest. Members are not bit-identical to separate solves, because XLA
rounds batched reductions slightly differently, but each certifies to the
requested tolerance.

Three functions differentiate through a solve. Each takes `build_lp` and
carries its gradient back through it to `params`:

| Function | Differentiates | How |
|---|---|---|
| `jl.make_optimal_value` | the optimal value `z*` w.r.t. every number | envelope theorem: `∂z*/∂c = x*`, `∂z*/∂b = −y*`, `∂z*/∂A_ij = y*_i x*_j`, and the reduced costs for the bounds |
| `jl.make_solution` | `x*` w.r.t. `b`, `A` and the bounds | implicit differentiation of the active constraints at a nondegenerate vertex (raises an error elsewhere) |
| `jl.make_perturbed_solution` | a smoothed `x*` w.r.t. `c` | perturbed optimizer: averages solves at `c + σZ`, batched; `x_fn(params, key)` |

An LP's solution is piecewise constant in its cost, so `dx*/dc` is zero
almost everywhere. That's why the smoothed, perturbed version exists for
learning costs. On degenerate LPs a first-order method can return a point
inside an optimal face, where `make_solution`'s derivative is undefined.
All three are checked against finite differences of HiGHS solutions in the
tests.

### Supporting tools

- **Loading models.** `jl.load(path)` reads an MPS or LP file into a
  `JaddleLP` for `jl.solve` (integer variables are relaxed).
  `jl.load(path, presolve=True)` returns HiGHS's reduced LP instead, and
  `return_offset=True` also returns the objective constant to add to `c @ x`.
  The reduced LP's solution is not mapped back to the file's variables; use
  `jl.solve_with_presolve` for that.
- **Presolve and postsolve.** `jl.solve_with_presolve(model, **options)`
  presolves with HiGHS, solves the reduced LP, and maps the primal *and* dual
  solution back with HiGHS's postsolve. `model` can be an MPS/LP file path, a
  HiGHS model or a Jaddle `LP`. The result is certified on the problem as
  given: `converged` reports the original LP's certificate, and
  `reduced_converged` the reduced solve's. Postsolve can enlarge a first-order
  solution's small dual errors; a postsolved point that misses the
  certificate is returned with `converged=False` and its residuals in
  `certificate`, without a further solve. The two verdicts can disagree when presolve rescales the problem drastically. On leo1
  it folds a row with 1e7 coefficients into the cost, so a reduced solve
  certified at 1e-6 has a correct objective but fails the original's dual
  test. Presolve can also settle a problem outright:
  `stop_reason="presolve_infeasible"` or `"presolve_solved"`.
- **Crossover.** `jl.crossover(lp, solution)` takes a first-order solution to
  a vertex: an optimal basic solution with its basis, computed by HiGHS.
  The point is first snapped to an exactly complementary primal–dual pair,
  which HiGHS's crossover requires, and the simplex method then repairs the
  basis, so the answer is exact however loose the starting solution. On
  mzzv11 (presolved, solved to `1e-4`) the points of two solves needed 125
  and 6,077 repair pivots, 0.1 s and 1.2 s for the whole call, where HiGHS's
  simplex needs 47,774 pivots and 20 s from cold. `jl.solve_with_presolve(model, run_crossover=True)` runs it on the
  reduced LP and lets HiGHS's postsolve carry the basis back to the original
  model.
- **Defined-variable elimination.** `jaddle.presolve.eliminate_defined_variables`
  substitutes out variables defined by dense equality rows (`z = Σ aⱼxⱼ`).
  Rows of this kind hide objective structure from the scaling and can stall
  first-order methods. It maps primal values back, but not duals.
- **Certification.** `jl.evaluate_lp_certificate` reports the relative primal,
  dual and gap residuals of any primal–dual pair independently of a solve.
- **Feasibility polishing.** `jl.solve_with_polishing` is `jl.solve` with
  PDLP-style feasibility polishing: once the gap has closed, short zero-objective
  (`jl.primal_polish`) and zero-RHS (`jl.dual_polish`) sub-solves push the
  residuals down, and the polished pair is returned only if it certifies. It
  pays off when feasibility is tighter than the gap, e.g. `tol=1e-8,
  dual_gap_tolerance=1e-4`: on stp3d (float64) that certifies in 24 s, where a
  plain solve is still at PFR 5e-6 after 150 s.
  `jl.solve_with_presolve(model, polish=True)` uses it for the reduced LP.
- **Iterative refinement.** `jl.solve_with_refinement` is `jl.solve` with
  iterative refinement (Gleixner, Steffy & Wolter, 2016). A loose first solve
  is followed by rounds of correction LPs: the same problem shifted to the
  current point, with the residuals scaled up to O(1). Each correction
  restarts the solver on a well-scaled problem in place of a long tail. Unlike
  polishing, it tightens the gap along with feasibility: on stp3d (float64,
  `tol=1e-8, dual_gap_tolerance=1e-4`) one round certified in about 94 s,
  with the gap at 9.7e-9. `result["refinement"]` records the rounds. For
  presolved models use
  `jl.solve_with_presolve(model, reduced_solver=jl.solve_with_refinement)`.
- **Infeasibility detection.** `jl.detect_infeasibility` classifies an LP as
  `"primal_infeasible"`, `"dual_infeasible"` (unbounded if feasible),
  `"optimal"` or `"undetermined"` the PDLP way: diverging iterates give a
  candidate Farkas or unbounded ray, refined by a small least-squares fix and
  checked with `jl.evaluate_infeasibility_certificate`.
- **Diagnostics.** `jl.lp_summary_statistics` prints the problem dimensions and
  coefficient ranges, which is the first thing to check when an instance
  misbehaves.

### Conic programs

The same solver handles second-order, positive semidefinite and exponential
cones. Pass `cones=jl.Cones(soc=..., psd=..., exp=...)` to `jl.LP` and the
last variables of `x` are constrained to cones, in the style of SCS's cone
dimensions:

```
minimise    cᵀx
subject to  A_eq x = b_eq,  A_ineq x <= b_ineq,  l <= x <= u   (linear variables)
            x[K₁] ∈ K₁, x[K₂] ∈ K₂, ...                         (the tail of x)
```

With `n` variables and `jl.Cones(soc=(3, 5), psd=(4,), exp=2)`, the first
`n − 24` are ordinary linear variables, the next 3 lie in a second-order cone
`{(t, z) : ‖z‖ ≤ t}`, the next 5 in another, the next 10 hold a 4 × 4 PSD
matrix in `svec` form (its upper triangle row by row, off-diagonals times √2;
`jaddle.cones.svec` and `smat` convert), and the last 6 are two exponential
cones `closure{(x, y, z) : y·exp(x/y) ≤ z, y > 0}`. Cone variables must be
unbounded. A conic constraint `b − Ax ∈ K` becomes `Ax + s = b` with slack
variables `s` in the cone.

```python
import numpy as np
import scipy.sparse as sp
import jaddle.jaddle_linear as jl

# minimise cᵀz subject to ‖z‖ <= 1, as (t, z) in a second-order cone, t = 1
c = np.array([1.0, -2.0, 2.0])
n = 1 + c.size
lp = jl.LP(
    c=np.r_[0.0, c],
    A_eq=sp.csc_matrix(np.eye(1, n)), b_eq=np.ones(1),
    A_ineq=sp.csc_matrix((0, n)), b_ineq=np.zeros(0),
    lower_bounds=np.full(n, -np.inf), upper_bounds=np.full(n, np.inf),
    cones=jl.Cones(soc=(n,)),
)
z = jl.solve(lp)["solution"].primal[1:]   # -c / ‖c‖
```

Each step projects the cone variables onto their cones instead of onto a box:
a closed form for second-order cones, one batched eigendecomposition per PSD
size, and for exponential cones Friberg's univariate root search (as in SCS
and Clarabel). The dual condition on a cone block is that its reduced cost
lies in the dual cone `K*`, and the certificate measures dual infeasibility as
the reduced cost's distance to `K*`, computed from the projection onto `K` as
`−Π_K(−r)`. (The second-order and PSD cones are self-dual; the exponential
cone is not.) The exponential cone's projection is built for float64. It is
also the expensive part of an exponential-cone solve: on GPU it runs as a
sequence of small kernels each iteration, so such problems are slower per
iteration than SOC or PSD ones. Scaling, the
diagonal metric and restarts work as for an LP, with column scales tied to one
value per cone block. `jl.solve`, `make_solver`, `solve_batch` and
`make_optimal_value` (differentiable optimal values) all take conic problems;
presolve, crossover, polishing, iterative refinement, infeasibility
detection and `make_solution` do not yet, and raise `NotImplementedError`.
CVXPY problems with
second-order, PSD and exponential constraints go through
`jcv.JaddleConicSolver` (see
[Modelling with CVXPY](#modelling-with-cvxpy)).

## The mixed-integer solver

> **Highly experimental.** The MIP solver is under active development. Its
> interface, defaults and behaviour may change between releases without
> notice, and it has been tested on far fewer problems than the LP solver.
> Check its solutions and bounds against an established MIP solver (such as
> HiGHS, through `jm.to_highs`) before relying on them.

`jaddle.jaddle_mip` is a simple branch-and-bound solver for mixed-integer
linear programs, with `jaddle_linear.solve` solving the LP at every node.
It is meant as a readable starting point, not a competitor to branch-and-cut
codes: cuts are added at the root only and come from two families, and the
primal heuristics are rounding, fix-and-propagate, feasibility jump and a
feasibility pump.

### Problem form

A `MIP` is a Jaddle LP in the standard form above plus a boolean mask of
integer columns:

```python
import jaddle.jaddle_mip as jm

mip = jm.MIP(lp, integrality)          # lp: a jl.LP or JaddleLP
mip = jm.load("model.mps")             # any file HiGHS reads, integrality kept
mip = jm.from_highs(highs)             # a highspy.Highs or HighsLp
result = jm.solve(mip, verbose=True)   # or jm.solve("model.mps", presolve=True)
```

`MIP` keeps the objective constant (`offset`) and whether the model was a
maximisation, so `mip.objective(x)` and the solver's reported values are in
the model's own sense. The bounds of integer variables are rounded inward.
`jm.to_highs(mip)` turns a `MIP` back into a HiGHS model, for example to
compare against HiGHS's own MIP solver. Semi-continuous and semi-integer
columns are not supported.

### Algorithm

- **Bound propagation.** Activity-based bound tightening runs at the root
  and on every child after branching, so many infeasible children are
  dropped without an LP solve. Proving infeasibility is slow for a
  first-order method, so this matters more here than in a simplex-based
  code.
- **Root.** The root LP is solved by `jl.detect_infeasibility`, which
  certifies optimality, infeasibility or an unbounded relaxation.
- **Root cuts.** Up to `cut_rounds` rounds of knapsack cover cuts (with
  complemented variables and extended-cover lifting) and complemented MIR
  cuts, from single rows and from aggregations of up to three rows through
  shared continuous variables. Both families are derived from a row and the
  bounds alone, so they are valid however inaccurate the LP point is; the
  point only picks which cuts to keep. A round keeps the most efficacious
  cuts that are not near-parallel, retires cuts that sat slack with a zero
  dual for two rounds, and re-solves the LP warm. Cuts are not added in the
  tree: every new row count recompiles the LP solver, so node cuts would
  recompile at nearly every node. During the rounds the cut block is
  padded to power-of-two sizes with room to grow, so the rounds share one
  compiled shape; afterwards idle cuts are dropped and the block is rebuilt
  at a tight size, so the node LPs pay only for the cuts in use.
- **Gomory cuts (opt-in).** `gomory_cuts=True` adds Gomory mixed-integer
  cuts. Each round then crosses the root point over to an optimal vertex
  with `jl.crossover`, reads tableau rows from a sparse LU of its basis,
  and separates all three families at the vertex. On small test problems
  they raised the root bound but made the search slower: the dense cuts
  cost more per node LP than the smaller trees saved. On ns1830653
  (presolved) they lift the root bound from 7,297 to 7,631, where cover
  and MIR cuts stop. So they are off by default.
- **Batched nodes.** With `batch_size > 1`, up to that many open nodes are
  solved in one device call: `jax.vmap` of `jl.make_solver` over the
  nodes' bounds and warm starts, compiled once, since the node LPs differ
  only in their bounds. The nodes are then evaluated in order on the host
  as in the serial search. Members run in lockstep, so one slow node holds
  its batch to `node_max_epochs`.
- **Warm-started nodes.** Each child starts from its parent's primal–dual
  point and step-size state. The node LPs share their shapes and solver
  settings, so the compiled solver is reused and a small node takes a few
  milliseconds.
- **Sound bounds from approximate LPs.** A node is pruned on the
  Lagrangian dual bound of its duals,
  `g(y) = −bᵀy + Σⱼ min(rⱼlⱼ, rⱼuⱼ)` with `r = c + Aᵀy`, which is a valid
  lower bound for any duals whether or not the LP converged. Duals that
  diverge on an infeasible node therefore prune it once there is an
  incumbent. A node LP that does not converge within `node_max_epochs` gets
  an infeasibility check of the same length.
- **Bounds from duals.** The Lagrangian terms `(−bᵀy, r)` bound any box,
  so each child is bounded by its parent's duals over its own propagated
  box before its LP is solved, and closed at once when that reaches the
  cutoff. The same terms give reduced-cost fixing: once there is an
  incumbent, an integer variable with `rⱼ > 0` must satisfy
  `xⱼ ≤ lⱼ + (cutoff − g(y))/rⱼ` in any better solution (and likewise from
  the upper bound when `rⱼ < 0`). Each node tightens the bounds its children
  inherit, and the root box is tightened with the root's duals whenever the
  incumbent improves. Both hold for any duals, so approximate node LPs are
  fine.
- **Integral objective.** When every objective value is a multiple of a
  step `g` (integer costs that are multiples of `g` after scaling by up to
  10⁴, and no cost on continuous variables), node bounds are rounded up to
  multiples of `g` and the cutoff becomes at most `incumbent − g/2`.
- **Branching.** `"pseudocost"` (default) uses the product of the
  estimated down and up gains learnt from solved nodes. `"most_fractional"`
  picks the variable nearest ½.
- **Node selection.** `"hybrid"` (default) dives depth first until the
  first incumbent, then takes the best bound with plunges: after each
  best-bound node, up to `plunge_depth` steps follow its preferred child, so
  the search keeps reaching leaves and finding better incumbents while it
  raises the bound. `"best_first"` and
  `"depth_first"` use one rule throughout.
- **Incumbents.** LP points whose integers are integral, and the LP point
  at every `heuristic_every`-th node, are rounded and checked against the
  rows and bounds. With continuous variables, a rounded point that fails
  the check is repaired by fixing the integers and solving the LP for the
  rest, within `repair_max_epochs`. A first-order method can't prove that
  LP infeasible quickly, so the repair is skipped when those integers
  already failed, when no violated row has a continuous variable, or when
  bound propagation shows the fixed box infeasible.
- **Primal heuristics.** Rounding an LP point rarely satisfies assignment
  or partitioning rows, so two heuristics run at the root after the cuts.
  *Fix-and-propagate* fixes the integers to their rounded LP values in
  chunks, those rounding above their lower bound first, with bound
  propagation after each chunk; a rejected chunk is split and retried. It
  needs no LP solve, so it also runs in the tree wherever rounding fails.
  *Feasibility jump* (Luteberget and Sartor) is an LP-free weighted local
  search on the rows: each step considers, for every variable in a violated
  row, the value that satisfies its heaviest violated row, scores each such
  move by its exact effect on the weighted violation, and makes the best
  one; at a local minimum the violated rows get heavier. Every step is a
  few sparse passes over the nonzeros, run on the device in compiled
  chunks. It starts from the rounded root LP point and from the bounds
  nearest zero, and once more from the pump's last point if the pump
  fails.
  The *objective feasibility pump* (Achterberg and Berthold) alternates
  rounding with an LP that minimises the distance to the rounded point,
  blended with a decaying share of the objective, and breaks cycles by
  flipping variables. Only the cost changes between its LPs, so they reuse
  the root's compiled solver, warm-started, at a loose tolerance. On
  generalised assignment problems where rounding finds nothing,
  one or the other finds an incumbent at the root in well under a second.
- **Presolve.** `presolve=True` runs HiGHS's MIP presolve, searches the
  reduced problem, and maps the incumbent back with HiGHS's postsolve.

The one place soundness rests on a tolerance is a reduced cost of the wrong
sign on a variable with an infinite bound, which makes the Lagrangian bound
`−inf`. When it is within the LP tolerance it is dropped, and the result
reports `bound_rigorous=False`.

### Principal options

| Argument | Default | Purpose |
|---|---|---|
| `mip_rel_gap`, `mip_abs_gap` | `1e-4`, `1e-6` | Stop once the incumbent is within these of the bound (HiGHS's defaults). |
| `max_nodes`, `max_seconds` | `None` | Node and wall-clock budgets. |
| `branching`, `node_selection` | `"pseudocost"`, `"hybrid"` | Search rules (see above). |
| `integrality_tolerance` | `1e-5` | How close to an integer an LP value must be to count as integral. |
| `feasibility_tolerance` | `1e-6` | Largest row violation (relative to `1 + ‖b‖∞`) and bound violation an incumbent may have. |
| `lp_tolerance`, `lp_options` | `1e-6`, `None` | The node LPs' relative tolerances, and any other `jl.solve` option. |
| `node_max_epochs`, `root_max_epochs` | `200`, `None` | Epoch budgets for a node LP and the root LP. |
| `heuristic_every` | `10` | Run rounding (then fix-and-propagate if it fails) every this many nodes (`0` = only on integral LP points). |
| `fix_and_propagate`, `feasibility_jump`, `feasibility_pump` | `True`, `True`, `True` | The primal heuristics. |
| `repair_max_epochs` | `20` | Epoch budget of the LP that repairs a rounded point's continuous variables (skipped for integers that already failed, or when propagation shows the fixed box infeasible). |
| `jump_iterations` | `None` | Feasibility jump steps per start (`None`: `max(10000, 20·(n + m))`). |
| `plunge_depth` | `8` | Steps a plunge follows a best-bound node's children (`0`: pure best-bound). |
| `batch_size` | `1` | Open nodes solved together in one `vmap`-ed device call (try 8–32 on a GPU). |
| `highs_incumbent_seconds` | `0` | If set, HiGHS supplies a first incumbent when Jaddle's heuristics find none (off by default). |
| `rins_seconds`, `rins_every` | `0`, `50` | If set, RINS with HiGHS on the sub-MIP every `rins_every` nodes, to improve the incumbent (off by default). |
| `pump_iterations`, `pump_tolerance` | `30`, `1e-4` | The pump's LP solves and their tolerance. |
| `heuristic_seconds` | `None` | Time for the root heuristics (`None`: 20% of `max_seconds`, or no time limit without one). |
| `propagate` | `True` | Bound propagation at the root and at each child. |
| `cuts`, `cut_rounds` | `True`, `5` | Root cover and MIR cuts, and the most rounds of them. |
| `max_cuts_per_round`, `cut_min_progress` | `100`, `1e-4` | Cuts added per round, and the relative bound gain below which the rounds stop. |
| `gomory_cuts` | `False` | Gomory mixed-integer cuts from a crossover basis at each root round. |
| `presolve`, `highs_options` | `False`, `None` | HiGHS presolve and postsolve, and options for the HiGHS instance. |
| `verbose`, `log_every_seconds` | `False`, `1.0` | Progress table. |

Use the `float64` profile for the default tolerances.

### Results

`solve` returns a dictionary with the keys `x` (the incumbent, or `None`),
`objective`, `bound` (a lower bound for a minimisation, an upper bound for a
maximisation), `gap`, `status` (`"optimal"`, `"infeasible"`,
`"unbounded_or_infeasible"`, `"unresolved"`, `"time_limit"`, `"node_limit"`
or `"interrupted"`), `converged`, `bound_rigorous`, the counts `nodes`,
`infeasible_nodes`, `unresolved_nodes` and `lp_epochs`, `seconds`, and
`history`, one entry per new incumbent, `cuts` (the cuts in the final LP by
family: `gomory`, `cover`, `mir`), `root_bound_before_cuts`, `heuristic_seconds`,
`objective_step` (or `None`), `reduced_cost_tightenings`, `children_pruned`
(children closed on their parent's duals), and `lp_repairs` / `lp_repairs_skipped`
(repair LPs solved, and rounded points rejected without one). `"unresolved"` means the search
ended but some nodes had nothing left to branch on and could not be settled
(their LP did not converge, or their integral point failed the feasibility
check); their bounds are still in `bound`. With `presolve=True` it also has
`presolve` (HiGHS's presolve status) and `postsolve_feasible`.

On small problems the search is fast once compiled. In the facility-location
example (8 binaries, 160 continuous flows), five rounds of MIR cuts raise the
root bound from 986.8 to 1204.6 (the optimum is 1209.0, so 98% of the root
gap), and the tree closes in 5 nodes against 61 without cuts. On MIPLIB instances the two cut families close
little of a weak relaxation, and a gap usually remains. On ns1830653
(presolved) a single cover cut raises the root bound from 6,797 to 7,297
(the optimum is 20,622), and 240 s on a CPU explored 98 nodes without an
incumbent.

## The convex programming solver

### Problem form

`jaddle_convex.solve` accepts a `JaddleCP`:

```python
jc.JaddleCP(
    num_variables,      # n
    objective,          # f(x) -> scalar, convex and differentiable
    constraints_eq,     # h(x) -> vector, constrained to h(x) = 0
    constraints_ineq,   # g(x) -> vector, constrained to g(x) <= 0
    lower_bounds,       # l, may contain -inf
    upper_bounds,       # u, may contain +inf
)
```

All functions must be traceable by JAX. The solver forms the Lagrangian
`f(x) + λᵀg(x) + μᵀh(x)` with `λ >= 0`, and obtains every gradient it needs by
automatic differentiation. No derivatives are required from the user.

### Algorithm

**Update schemes** (`update_mode`)

| Mode | Gradient evaluations per iteration | Description |
|---|---|---|
| `"alternating"` (default) | 1 | Primal step, then a dual step at the new primal point: PDHG without the extrapolation. With `adaptive_eta=False` the update rule is a user-supplied Optax optimiser. Best used with restarts. |
| `"extragradient"` | 2 | Korpelevich extragradient: a look-ahead step followed by a corrector step. Contractive for monotone problems. |
| `"pdhg"` | 1 | PDHG as in the LP solver: a primal step, then a dual step that reads the extrapolated constraint values `2g(x⁺) − g(x)`. One gradient plus one constraint evaluation per iteration. Requires the adaptive step. |

**Adaptive step size.** With `adaptive_eta=True` (the default, as in the LP
solver), every scheme uses a Malitsky–Tam line search, started from
`step_size` and based on a local Lipschitz estimate. The two gradient evaluations that
extragradient already performs give an estimate of the local Lipschitz
constant at no extra cost. PDHG and alternating get their estimate from
consecutive iterations, so they check each step one iteration late and undo
it if it overshot. The
step is accepted while it remains within the admissible bound, and is rejected
and shrunk otherwise. The learned step is carried across restarts.

**Primal weight.** As in the LP solver, a primal weight `k` balances the primal
and dual step sizes. It is initialised from the gradient of the objective and
the constraint values at the origin, rebalanced at each restart, and clamped
to `[1/k_scale, k_scale]` (default `k_scale = 100`; the LP solver's `1e8`
let `k` swing far enough to collapse the adaptive step on some problems). It
starts at 1 when the objective gradient or the constraint values at the origin
are zero.

**Diagonal metric.** As in the LP solver, `diag_adapt=True` (the default)
adapts a per-coordinate step metric at each restart: larger steps for
steadily drifting coordinates, smaller for oscillating ones. It applies to all
three schemes with the adaptive step, and has no effect without restarts or
with a custom optimiser.

**Gradient scaling.** The solver sees a problem only through its functions,
so it cannot equilibrate a matrix. Instead, `gradient_scaling=True` (the
default) rescales the objective and each constraint row by the size of its
gradient at the start point, as IPOPT does. This fixes objectives and rows in
mismatched units, but not variables in mismatched units. The convergence test
and the returned duals stay in the problem's own units.
`jc.quadratic_program` scales its matrices itself, so it turns gradient
scaling off unless asked.

**Custom optimisers.** With `adaptive_eta=False`, `optimiser` accepts any Optax
`GradientTransformation`. It is applied separately to the primal and dual
players through `jaddle_optimisers.create_saddle_optimiser`. Ready-made
choices include `jo.gd`, `jo.optimistic_gd`, `jo.optimistic_adadelta` and
`jo.gd_dual_momentum`.

**Restarts and averaging.** Iterate averaging (`average=True`, with an
optional `weight_function`) is on by default. As in the LP solver,
`report_best=True` (the default) tests convergence on, and returns, whichever
of the average and the last iterate has the lower KKT merit each epoch, so
averaging never holds back a last iterate that is already converging. Adaptive
restarts (`restarts=True`) are on by default, as in the LP solver.

**Termination.** The solver stops when all three of the following fall below
their tolerances:

| Test | Tolerance argument | Default |
|---|---|---|
| Stationarity (projected primal gradient norm of the Lagrangian) | `primal_grad_norm_tolerance` | `1e-2` |
| Primal feasibility | `primal_feasibility_tolerance` | `1e-3` |
| Complementary slackness | `complementarity_slack_tolerance` | `1e-3` |

### Principal options

| Argument | Default | Purpose |
|---|---|---|
| `update_mode` | `"alternating"` | Iteration scheme (see above). |
| `adaptive_eta`, `step_size` | `True`, `1.0` | Line search on the step `η`, and the starting `η`. With `adaptive_eta=False` the optimiser's fixed step is used instead. |
| `optimiser` | `None` | Optax optimiser for the fixed step (`adaptive_eta=False`). When `None`, gradient descent with learning rate `step_size` is used. |
| `iterations_per_epoch` | `256` | Iterations per compiled epoch. |
| `max_epochs`, `max_seconds` | `None` | Iteration and wall-clock budgets. |
| `k_scale`, `k_theta`, `k_init` | `100.0`, `0.5`, `None` | Primal weight controls. `k_scale=None` disables the primal weight. |
| `restarts`, `epochs_per_restart` | `True`, `None` | Adaptive restarts (`epochs_per_restart=None` = no cycle cap). |
| `diag_adapt`, `diag_theta`, `diag_clamp` | `True`, `0.5`, `10.0` | Per-coordinate diagonal step metric, adapted at each restart. |
| `gradient_scaling` | `True` | Rescale the objective and constraint rows by their gradients at the start point. |
| `initial_solution`, `initial_opt_state` | `None` | Warm start. |

`solve` returns a dictionary containing `solution` (a `SaddleState`),
`converged`, `stop_reason`, `opt_state` and `solve_seconds`.

### Batched and differentiable solves

For a family of problems, write a function `build_cp(params)` that returns
an ordinary `JaddleCP` whose functions close over `params`. `params` can be
any pytree: data, weights, right-hand sides or bounds. Then:

```python
def build_cp(params):
    a, s = params
    return jc.JaddleCP(
        num_variables=n,
        objective=lambda x: jnp.sum((x - a) ** 2),
        constraints_eq=lambda x: jnp.array([jnp.sum(x) - s]),
        constraints_ineq=lambda x: jnp.zeros(0),
        lower_bounds=jnp.zeros(n), upper_bounds=jnp.full(n, jnp.inf),
    )

solve_fn = jax.jit(jc.make_solver(build_cp, max_epochs=500))
result = solve_fn((a, 1.0))                       # SolveCoreResult
batch = jax.vmap(solve_fn)((a_batch, s_batch))    # many problems at once

value_fn = jc.make_optimal_value(build_cp)
grad_a, grad_s = jax.grad(value_fn)((a, 1.0))     # dz*/dparams
x_fn = jc.make_solution(build_cp)                 # differentiable x*(params)
```

`make_solver` runs `solve()`'s algorithm with the same options, but without
logging or a time limit. `make_optimal_value` differentiates the Lagrangian at
the solution with respect to `params` (the envelope theorem), so it needs no
differentiation through the iterations. `make_solution` implicitly
differentiates the KKT conditions on the active set, matrix-free with GMRES.
That needs a nondegenerate solution and a strictly convex objective (or at
least a nonsingular KKT system); where the system can't be solved, the
gradient is NaN. Both are checked against closed forms and finite differences
in the tests.

### Quadratic programs

`solve` sees a problem only through its functions, so it cannot scale one. For
a quadratic program the matrices are known, and `jc.quadratic_program` uses
them:

```python
result = jc.quadratic_program(
    Q, c,                       # minimise ½ xᵀQx + cᵀx
    A_eq=A_eq, b_eq=b_eq,       # A_eq x = b_eq
    A_ineq=A_ineq, b_ineq=b_ineq,  # A_ineq x <= b_ineq
    lower_bounds=l, upper_bounds=u,
)
x = result["solution"].primal
```

`Q` is symmetric positive semidefinite, the matrices are NumPy arrays or SciPy
sparse matrices, and any constraint block or bound can be left out. Each
matrix is held on the device as a dense array if at least a quarter of its
entries are nonzero (or it is small) and as a sparse one otherwise, so a full `Q` (a kernel
matrix, say) costs one dense product per gradient. The
function equilibrates the KKT matrix `[[Q, Aᵀ], [A, 0]]` symmetrically with
Ruiz and Pock–Chambolle sweeps (`ruiz_iterations`, `pc_iterations`, as in the
LP solver), solves the scaled problem with `solve`, and maps the solution and
its duals back. Other keyword arguments go to `solve`.

It stops on the LP solver's certificate, in the problem's own units whatever
the scaling: `primal_feasibility_tolerance`, `dual_feasibility_tolerance` and
`dual_gap_tolerance` (default `1e-3`) bound the relative primal residual, dual
residual and duality gap. `result["certificate"]` reports the three,
recomputed from the unscaled matrices.

Scaling is what makes badly scaled problems solvable. On synthetic QPs with
200 variables and 80 constraint rows, at a tolerance of `1e-6` with 200
iterations per epoch:

| Problem | As given | Scaled |
|---|---|---|
| `Q` with condition number 10 | 1 epoch | 1 epoch |
| `Q` with condition number 1e4 | 29% from the optimum after 200 epochs | 16 epochs |
| Condition number 10, variables and rows rescaled by up to 1e±2 | 30% from the optimum after 200 epochs | 1 epoch |

In the second row the variable scales come out nearly uniform, so the gain is
from balancing `Q`'s entries (up to 1.7e3) against `A`'s (up to 3.7), which
the primal weight alone does not do.

The stopping test is supplied through `JaddleCP`'s optional `residuals`
argument, a function of the solver state that returns the three quantities
`solve` tests. Any problem with a better certificate of its own can use it.

Many QPs that share their matrices and differ only in `c` and `b` arise in
model predictive control, parametric studies and scenario sweeps.
`jc.make_qp_solver` is `quadratic_program` split in two for that case. It
checks and scales the matrices once, on the host, and returns a pure-JAX
`solve_fn` to `jit` and `vmap` over the vectors:

```python
solve_fn = jax.jit(jax.vmap(jc.make_qp_solver(
    Q, A_ineq=A_ineq, lower_bounds=l, upper_bounds=u, max_epochs=100,
)))
result = solve_fn(c_batch, b_ineq=b_batch, initial_solution=warm_batch)
x = result.solution.primal            # one row per problem
```

It has the same stopping test as `quadratic_program`, and `make_solver`'s
result type (`solution`, `converged`, `epochs`). A matrix with at most 65,536
entries is always held dense, since a batched dense product is far cheaper
than a sparse one at that size.
[`examples/cp/mpc_fleet.py`](examples/cp/mpc_fleet.py) controls a fleet of
1000 systems this way. On a laptop GPU, each MPC solve took about 1/40 of the
time of a warm-started SCS loop on the CPU.
[`examples/cp/rag_reranker.py`](examples/cp/rag_reranker.py) uses it to solve
a ranking SVM's whole regularisation path at once. It learns a reranker over
keyword and dense-embedding features for retrieval on the SciFact benchmark.

## Modelling with CVXPY

[CVXPY](https://www.cvxpy.org/) can serve as the modelling language
(`pip install "jaddle[cvxpy]"`). `jaddle.jaddle_cvxpy` offers three routes, and
all return results the CVXPY way: `problem.status`, `problem.value`,
`var.value`, each constraint's `dual_value`, and Jaddle's full result in
`problem.solver_stats.extra_stats`.

```python
import cvxpy as cp
import jaddle.jaddle_cvxpy as jcv

w = cp.Variable(p)

# An LP or QP: Jaddle as a CVXPY solver.
lasso = cp.Problem(cp.Minimize(cp.sum_squares(X @ w - y) + cp.norm1(w)))
lasso.solve(solver=jcv.JaddleSolver(), tol=1e-6)

# Second-order and PSD cones: Jaddle as a CVXPY conic solver.
risk_limited = cp.Problem(cp.Maximize(mu @ w), [cp.sum(w) == 1, cp.norm(F.T @ w) <= 0.2])
risk_limited.solve(solver=jcv.JaddleConicSolver(), tol=1e-6)

# Any other DCP problem: compiled to a JaddleCP.
logistic = cp.Problem(cp.Minimize(cp.sum(cp.logistic(-cp.multiply(labels, X @ w)))))
jcv.solve(logistic, tol=1e-6)
```

**`JaddleSolver`** is a CVXPY QP solver. CVXPY reduces the problem to
`min ½xᵀPx + qᵀx  s.t.  Ax = b, Fx ≤ g, l ≤ x ≤ u`, piecewise-linear terms
included. An LP then goes through `jl.solve_with_presolve` (HiGHS presolve,
Jaddle, HiGHS postsolve; `presolve=False` skips it), and a QP through
`jc.quadratic_program`. A presolved LP whose postsolved point misses the
tolerance on the original problem is reported as `"optimal_inaccurate"`.
[`examples/lp/cvxpy_transport.py`](https://github.com/BrendanVR/Jaddle/blob/main/examples/lp/cvxpy_transport.py)
is a small worked LP.

**`JaddleConicSolver`** is a CVXPY conic solver for problems CVXPY can put in
the form `min cᵀx  s.t.  Ax + s = b`, with `s` in a product of zero,
nonnegative, second-order, PSD and exponential cones. It solves them with
`jl.solve` on cone-constrained variables (see [Conic programs](#conic-programs));
there is no presolve. When a cone constraint is on a variable itself, such
as `X >> 0` for a variable `X`, the cone goes on that variable directly
rather than on a copy tied to it by equality rows: on a 200-node Max-Cut SDP
that cut the iterations threefold (`cone_substitution=False` turns it off).
Power cones are not supported. Exponential cones bring
in `log`, `log_det`, `log1p`, `rel_entr` and `kl_div`, which the compiled
route below cannot take, alongside `exp`, `log_sum_exp`, `logistic` and
`entr`.
[`examples/conic/cvxpy_portfolio.py`](https://github.com/BrendanVR/Jaddle/blob/main/examples/conic/cvxpy_portfolio.py)
traces a risk-limited portfolio's efficient frontier, and
[`examples/conic/cvxpy_maxcut.py`](https://github.com/BrendanVR/Jaddle/blob/main/examples/conic/cvxpy_maxcut.py)
solves the Goemans–Williamson Max-Cut SDP and rounds it to a cut, and
[`examples/conic/cvxpy_kelly.py`](https://github.com/BrendanVR/Jaddle/blob/main/examples/conic/cvxpy_kelly.py)
finds log-optimal (Kelly) portfolios, one exponential cone per scenario.

**`jcv.solve`** handles the rest. For a general convex problem CVXPY's own
reductions produce cones, and the cone form is often a poor fit for a
first-order method:
on the L1-logistic regression in
[`examples/cp/cvxpy_lasso.py`](https://github.com/BrendanVR/Jaddle/blob/main/examples/cp/cvxpy_lasso.py) it has 701
variables, 1500 rows and 400 exponential cones. `jcv.compile_problem` instead
walks the expression tree and builds a `JaddleCP` of JAX functions, with 101
variables and 100 linear rows for that problem.
[`examples/cp/cvxpy_robust_kalman.py`](https://github.com/BrendanVR/Jaddle/blob/main/examples/cp/cvxpy_robust_kalman.py)
compiles another such problem: a Kalman smoother with `huber(norm(v[t]))` on the measurement
noise, which CVXPY would turn into second-order cones. The compiled route
handles these atoms as follows:

- smooth atoms stay smooth: `logistic`, `log_sum_exp`, `exp`, `huber`,
  `sum_squares`, `quad_form`, `quad_over_lin` with a constant denominator,
  even powers, and p-norms with p ≥ 1;
- `entr` is continued below `jcv.ENTR_FLOOR` (1e-8) by its tangent line, so
  iterates outside its domain see a finite value and gradient. This is exact
  when the solution keeps the argument above the floor.
  [`examples/cp/cvxpy_channel_capacity.py`](https://github.com/BrendanVR/Jaddle/blob/main/examples/cp/cvxpy_channel_capacity.py)
  uses it to compute a channel's capacity;
- piecewise-linear atoms (`abs`, `maximum`, `max`, `norm1`, `norm_inf`, and
  their concave mirrors `minimum` and `min`; `pos` and `neg` are written with
  these) become epigraph variables with one linear row per piece. This is
  exact for a DCP problem. With `epigraph=False` they are left as they are;
- variable attributes (`nonneg`, `nonpos`, `bounds=`) and constraints that
  bound a whole variable by a constant (`x >= 0`) become Jaddle's box, which
  costs no dual. The bound constraints' duals are recovered from the reduced
  gradient.

Other atoms with a restricted domain (`log`, `sqrt`, ...) and second-order
or PSD constraints raise `NotImplementedError` on this route: the convex
solver keeps its iterates inside the box only, so they would leave the domain.
Second-order and PSD constraints go to `JaddleConicSolver` instead. A problem that CVXPY can
put in QP form goes to `JaddleSolver` unless `qp_route=False`, so it gets
presolve or KKT scaling. The compiled problem is solved by `jc.solve` with
adaptive extragradient, restarts and gradient scaling (`jcv.COMPILED_DEFAULTS`).
Any other `jc.solve` options can be passed as keyword arguments.

`tol` sets every tolerance, whichever route the problem takes. It defaults to
`1e-6`, not the `1e-3` of Jaddle's own functions, since CVXPY users expect a
solver to reach that. The QP route's tolerances are relative and the compiled
route's absolute, as in the functions they call.

## Numerical precision

Call `jaddle.jaddle_optimisers.configure_jax(profile)` before creating any
arrays:

| Profile | Arithmetic | Recommended for |
|---|---|---|
| `"float64"` (default) | Double precision, full-precision matrix products | Real-world LPs and any problem needing tight tolerances. Wide-dynamic-range instances can stall on rounding error in single precision. |
| `"float32"` | Single precision, TF32 matrix products | Throughput on well-conditioned problems. |
| `"float16"` | The scaled LP or QP matrices in half precision; iterates, norms, step sizes and the final certificate in float32 | Memory-limited problems, and dense QPs (half the memory, about 20% faster at n = 4000). Matrices are rounded to about 3 significant digits, so solves reach relative tolerances of about 1e-3; for LPs, `solve_with_refinement` gets past that. Not suited to an ill-conditioned `Q`, which the rounding can make indefinite. |

The profile can also be set through the `JADDLE_JAX_PROFILE` environment
variable. `configure_jax` also enables a persistent XLA compilation cache in
`~/.cache/jaddle_jax`, so repeated solves of problems with the same shape skip
recompilation.

## Benchmarks

Jaddle's LP solver is evaluated on the LP relaxations of the 383 MIPLIB 2017
instances used in the PDLP papers (100k–10M nonzeros). The run below used
default solver settings in float64, HiGHS presolve, a relative tolerance of
`1e-4` and a 3600-second limit per instance, on one NVIDIA GeForce RTX 5070 Ti
Laptop GPU.

| Outcome | Instances |
|---|---|
| Certified optimal on the HiGHS-presolved LP | **382 / 383** (one settled by presolve itself) |
| Certified optimal on the original LP, after HiGHS postsolve | **297 / 383** |
| Time limit reached | 1 (hgms62) |

The two certificates differ because postsolve can enlarge a first-order
solution's small dual errors, and because presolve can rescale a problem
drastically, which changes what a relative tolerance means (see
*Presolve and postsolve* above). `jl.solve_with_polishing`,
`jl.solve_with_refinement` and `jl.crossover` can tighten a postsolved point when the original certificate
matters.

Convergence on hgms62 can be recovered by altering the solvers default settings, in particular greatly increasing `diag_clamp` which alters the maximum amount of adaptive diagonal scaling used by the solver.

Shifted geometric means of the solve time (shift 10 s, unsolved instances
counted at the 3600-second limit, as in the PDLP papers):

| Time measured | SGM10 |
|---|---|
| Iteration loop, including the first-epoch compile (`solve_seconds`) | 6.86 s |
| Iteration loop with the compile amortised out (`corrected_seconds`) | 4.72 s |
| Wall clock, including presolve, scaling and setup | 9.56 s |

The benchmark harnesses, instance diagnostics and instructions for reproducing
these results are described in [`benchmarks/README.md`](https://github.com/BrendanVR/Jaddle/blob/main/benchmarks/README.md).

## Repository layout

| Path | Contents |
|---|---|
| [`jaddle/jaddle_linear.py`](https://github.com/BrendanVR/Jaddle/blob/main/jaddle/jaddle_linear.py) | LP solver, scaling, certification and polishing |
| [`jaddle/jaddle_mip.py`](https://github.com/BrendanVR/Jaddle/blob/main/jaddle/jaddle_mip.py) | Branch-and-bound MIP solver and MIP readers |
| [`jaddle/jaddle_convex.py`](https://github.com/BrendanVR/Jaddle/blob/main/jaddle/jaddle_convex.py) | Convex solver |
| [`jaddle/cones.py`](https://github.com/BrendanVR/Jaddle/blob/main/jaddle/cones.py) | Second-order and PSD cones for the LP solver |
| [`jaddle/jaddle_cvxpy.py`](https://github.com/BrendanVR/Jaddle/blob/main/jaddle/jaddle_cvxpy.py) | CVXPY front end |
| [`jaddle/jaddle_basic_types.py`](https://github.com/BrendanVR/Jaddle/blob/main/jaddle/jaddle_basic_types.py) | `SaddleState`, `JaddleCP`, `LP` and `JaddleLP` |
| [`jaddle/jaddle_optimisers.py`](https://github.com/BrendanVR/Jaddle/blob/main/jaddle/jaddle_optimisers.py) | Precision profiles and Optax saddle-point optimisers |
| [`jaddle/presolve.py`](https://github.com/BrendanVR/Jaddle/blob/main/jaddle/presolve.py) | Defined-variable elimination and postsolve |
| [`jaddle/highs_helpers.py`](https://github.com/BrendanVR/Jaddle/blob/main/jaddle/highs_helpers.py) | HiGHS converters and presolve interface |
| [`examples/`](https://github.com/BrendanVR/Jaddle/blob/main/examples/README.md) | Worked LP, MIP, conic and convex examples |
| [`benchmarks/`](https://github.com/BrendanVR/Jaddle/blob/main/benchmarks/README.md) | MIPLIB and convex benchmark harnesses |
| [`tests/`](https://github.com/BrendanVR/Jaddle/tree/main/tests) | Test suite (`pytest`) |
| [`CHANGELOG.md`](https://github.com/BrendanVR/Jaddle/blob/main/CHANGELOG.md) | Changes in each release |

## Citing Jaddle

If you use Jaddle in your research, please cite it. The citation details are
in [`CITATION.cff`](https://github.com/BrendanVR/Jaddle/blob/main/CITATION.cff),
and GitHub's "Cite this repository" button gives them in APA and BibTeX form.

## License

Jaddle is released under the MIT License. See [LICENSE.txt](https://github.com/BrendanVR/Jaddle/blob/main/LICENSE.txt).
