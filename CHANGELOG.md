# Changelog

## Unreleased

### Changed

- The `float16` profile now stores only the scaled LP the solver iterates on
  in half precision. Iterates, norms, dot products, the step size and primal
  weight, and the final certificate use float32, and `to_jaddle_sparse` /
  `JaddleLP.from_scipy` build the unscaled LP in float32. Before, the profile
  returned NaN or a stalled start point on most MIPLIB instances: float16
  squares overflow above 256, which turned the step-size test into
  `inf / inf`, and residuals below ~1e-3 rounded to exactly 0. Plain solves
  reach relative tolerances of about 1e-3 (the matrix's rounding);
  `solve_with_refinement` reaches tighter ones.
- Likewise for `jaddle_convex.quadratic_program` and `make_qp_solver`: under
  `float16` the scaled Q and constraint matrices are stored in half precision
  and the vectors and iterates in float32 (they were all float16). A dense QP
  then takes half the memory and runs about 20% faster than under float32
  (n = 4000). `quadratic_program`'s `converged` also requires the float64
  certificate to meet the tolerances, since float16 used to report
  convergence with residuals above them.
- `jaddle_convex.solve` computes its step-size test, primal-weight
  initialisation and primal-weight update in at least float32, and keeps k
  and eta in at least float32, whatever the dtype of the problem's data.

### Fixed

- The `float16` profile's sparse matrix-vector products sum in float32.
  JAX's float16 sparse sum ran 7-24x slower than float32 on GPU (dc1l: 271 s
  to 1.2 s for 20 epochs).
- The primal-weight update (in both `jaddle_linear.solve` and
  `jaddle_convex.solve`) no longer turns `k` into NaN after an epoch in which
  the primal or the dual did not move. Its `+ 1e-60` guard was 0 outside
  float64; such an epoch now keeps `k`.

## 0.2.6

### Fixed

- `flatbuffers` is now a dependency. `jax.export` needs it to save and load
  functions (`jl.save_solver` / `jl.load_solver`, `jc.save_cp` /
  `jc.load_cp`, `jc.save_solver` / `jc.load_solver`), but `jax` does not
  install it, so these failed with an `ImportError` in a clean install.
- A failed save with `vjp_order=1` no longer blames the function's gradient
  when the cause is a missing package.

## 0.2.5

### Added

- `examples/cp/rag_reranker.py`: hybrid retrieval and a linear ranking-SVM
  reranker on SciFact. BM25 and bge-small dense embeddings (via `fastembed`,
  cached in `data/`) are merged by reciprocal rank fusion. The reranker
  reorders the top 100 using 17 features per pair, including a cross-fitted
  learned term-weight score. SciFact test nDCG@10 is 0.770, against 0.731 for
  the fusion alone and 0.681 for BM25.
- `rag_examples` extra (`fastembed`, `snowballstemmer`, and `examples`) for
  the retrieval examples.
- `jc.save_cp` / `jc.load_cp`: write a `JaddleCP` to a single file with
  `jax.export` and load it back without the code that built it. The file
  holds the objective, the constraints and any `dual_bound` / `residuals`,
  plus the bounds. A loaded CP solves with `jc.solve` as before, including
  `gradient_scaling`, since the scales are exported too. Shapes, dtype and
  target platforms are fixed when saving, and a loaded CP cannot be used
  under `vmap`.
- `jl.save_lp` / `jl.load_lp`: write a `JaddleLP` to a single `.npz` file
  and read it back. The file keeps the numbers exactly (dtype, sparsity
  pattern and entry order) and the cones. `load_lp(path, dtype=...)`
  converts the numbers on loading, e.g. a float64 file to float32.
- `jc.save_solver` / `jc.load_solver` and `jl.save_solver` /
  `jl.load_solver`: the same for a family of problems. They save a function
  built with `make_solver`, `make_optimal_value`, `make_solution` or (LPs)
  `make_perturbed_solution`, with the solver settings fixed:
  `save_solver(fn, path, *example_args)`. `batch=True` saves a batched
  solver that takes any batch size. `vjp_order=1` saves gradients with
  respect to the arguments. `jl.make_solution`'s gradient runs on the host,
  so it saves without one.
- `jl.solve_concurrent`: races Jaddle against HiGHS's interior-point method on
  the same LP and returns the first to converge. The loser is stopped through
  `cancelSolve()` (HiGHS) or a new `cancel` event on `jl.solve` (PDLP, with
  `stop_reason="cancelled"`). The result is the winner's dict plus `"winner"`
  and per-leg `"legs"`. It takes the same inputs as
  `jl.solve_with_presolve`.

### Changed

- `jl.solve` and `jc.solve` no longer keep a running-average copy of the
  solution when `average=False`. This saves one solution-sized buffer
  (primal and duals) in memory. Results are unchanged.

## 0.2.4

### Added

- `jc.make_qp_solver`: batched quadratic programs that share their matrices.
  The matrices are checked and KKT-scaled once, and the returned
  `solve_fn(c, b_eq, b_ineq, initial_solution)` is pure JAX, so it can be
  `jit`-ed and `vmap`-ed over the vectors, with warm starts. It uses the same
  scaling and stopping test as `jc.quadratic_program`.
- `examples/cp/mpc_fleet.py`: closed-loop MPC of 1000 oscillating-mass systems,
  with each step's QPs solved in one batched call. On an RTX 5070 Ti laptop
  GPU, 10.5 µs per QP against 420 µs for warm-started SCS at a tolerance of
  1e-3. Jaddle's plans are also closer to a 1e-9 reference (median relative
  cost error 5e-6, against 2e-4 for SCS). SCS is faster for a single system.

### Changed

- `jc.quadratic_program` and `jc.make_qp_solver` hold matrices with at most
  65,536 entries dense, whatever their density. At 105 × 120 with 20%
  nonzeros, a batched sparse product made the MPC example 9× slower. Dense
  products of these matrices use full precision, so a `vmap`-ed solve is not
  limited to TF32 accuracy.

## 0.2.3

### Added

- `jc.solve(update_mode="pdhg")`: adaptive PDHG for the convex solver, in the
  cuPDLP style used by `jl.solve`. Each iteration costs one VJP and one
  constraint-only evaluation, where extragradient costs two VJPs. The line
  search checks each step one iteration late and rolls it back if it
  overshot. On `benchmarks/benchmark_convex.py` it took fewer or equal epochs
  than adaptive extragradient on every problem, and 25–45% less wall time. It
  requires `adaptive_eta=True`. The default is unchanged.
- `jl.solve(adaptive_eta=False)` (or `None`): plain PDHG on a fixed step, with
  the line search off. `diag_adapt` can make a fixed step unstable, so pass
  `diag_adapt=False` with it.
- `jl.solve_with_refinement`: iterative refinement for LPs (Gleixner, Steffy &
  Wolter, INFORMS JOC 2016). A first solve stops at a loose tolerance, then
  each round solves a correction LP at the current point and adds its
  scaled-down solution. Each round restarts on a well-scaled problem. On stp3d
  (float64, presolved, `tol=1e-8`) it certified in one round, about 94 s, where
  a plain solve was still uncertified after 150 s. On small random LPs it is
  no better than a plain solve. Use it on presolved models with
  `jl.solve_with_presolve(model, reduced_solver=jl.solve_with_refinement)`.
  Conic problems raise `NotImplementedError`.
- `benchmarks/benchmark.py --refine`: solves with `jl.solve_with_refinement`
  (not combinable with `--polish`) and records `jaddle_refine_rounds`.
- Exponential cones: `jl.Cones(exp=k)` adds `k` exponential cones, projected
  with Friberg's root search (ported from SCS). `jcv.JaddleConicSolver` accepts
  CVXPY's `ExpCone`, so `log`, `log_det`, `log_sum_exp`, `logistic`, `entr`,
  `rel_entr` and similar atoms go through it. See
  `examples/conic/cvxpy_kelly.py`. The projection is 1.3–2.2× faster, and an
  exponential-cone epoch about 20% faster on CPU and 10–20% on GPU.
- Second-order and PSD cones in the LP solver: `jl.LP(..., cones=jl.Cones(
  soc=..., psd=...))`. `jl.solve`, `make_solver`, `solve_batch` and
  `make_optimal_value` take conic problems. Presolve, crossover, polishing,
  infeasibility detection and `make_solution` raise `NotImplementedError` for
  them. Pure LPs are unaffected. New module `jaddle.cones`.
- `jcv.JaddleConicSolver`: a CVXPY conic solver for zero, nonnegative,
  second-order and PSD cones. A cone constraint on a variable itself
  (`X >> 0`) is applied to that variable, not to slack copies tied to it by
  equality rows. On the 200-node Max-Cut SDP this cut iterations from about
  6,100 to 2,000 and time from 9.2 s to 4.0 s. `cone_substitution=False`
  restores the slack form. See `examples/conic/`.
- `jcv.solve` compiles CVXPY's `entr` atom. Below `jcv.ENTR_FLOOR` (1e-8) it
  continues along the tangent line, so iterates outside `entr`'s domain get a
  finite value and gradient. See `examples/cp/cvxpy_channel_capacity.py`.
- `examples/cp/cvxpy_robust_kalman.py`: CVXPY's robust Kalman filtering
  example, on the QP route and the compiled route.
- `jcv.compile_problem(eliminate=True)` (also `jcv.solve`): an equality that
  defines a variable, such as `y == C @ x + v` for `v`, is substituted out
  instead of becoming a row. On the robust Kalman example this halves the
  robust solve (29→14 epochs, 1.45→0.77 s).

### Changed

- **Breaking:** `jl.solve` and `jl.make_solver` take `step_size` (default
  `1.0`), the starting step, and `adaptive_eta` (default `True`), a switch for
  the line search. A float or `"auto"` passed as `adaptive_eta` raises and
  names `step_size`. The `"auto"` step is now `0.99/‖[[A, b], [cᵀ, 0]]‖₂`
  (was `1/‖·‖₂`).
- **Breaking:** `jc.solve` and `jc.make_solver` take the same two arguments.
  The fixed-step optimiser is `jo.gd(step_size)`, which was `jo.gd(0.5)`, so
  pass `step_size=0.5` to keep the old fixed step. `adaptive_eta="auto"` is
  gone. An explicit `optimiser` needs `adaptive_eta=False`. `jcv`'s QP
  defaults set `step_size=0.5`.
- **Breaking:** the LP functions `make_solver`, `solve_batch`,
  `make_optimal_value`, `make_solution` and `make_perturbed_solution` take a
  builder. `build_lp(params)` returns an `LP` or `JaddleLP`, and the returned
  function takes `params` (any pytree), so costs, bounds and matrix entries
  can depend on `params`. Migration:
  - the old `LPValues` interface is `make_solver(jaddle_lp.with_values)`;
  - `solve_batch(build_lp, params, in_axes=0)` replaces `solve_batch(lp, values)`;
  - the perturbed solution is `x_fn(params, key)`;
  - `solve_fn()` no longer defaults to the LP's own values, and the `.pattern`
    attribute is gone.
- `jc.solve` and `jc.make_solver` take `jl.solve`'s defaults where the two
  share a setting: `iterations_per_epoch=256` (was 1000), `restarts=True`
  (was `False`), `diag_adapt=True` (was `False`) and `gradient_scaling=True`.
  `k_scale` is `100` (was 10). On `benchmarks/benchmark_convex.py` the new
  defaults certify 9 of 10 runs, against 5 of 10 for the old ones.
- `jc.solve`: `update_mode="alternating"` supports the adaptive step, and
  `adaptive_eta=True` is its default. Adaptive alternating is the `"pdhg"`
  step with the dual reading `c(x⁺)` instead of `2c(x⁺) − c(x)`. Passing an
  `optimiser` requires `adaptive_eta=False`.
- `jc.solve`: `diag_adapt` works with the adaptive `"alternating"` and
  `"pdhg"` modes, not only extragradient. Where it cannot apply, it does
  nothing instead of raising, as in `jl.solve`.
- `jc.solve`: when `‖∇f(0)‖` or the constraint values at 0 are zero, the
  primal weight starts at 1 (PDLP's rule), not at the edge of the `k_scale`
  clamp.
- The adaptive step's retry is no longer a loop inside each PDHG iteration. A
  rejected step leaves the iteration where it was, and the iteration count
  advances only on accepted steps. This costs about 30 µs per iteration on GPU
  less than before. A rejection now uses up one of the epoch's iterations, so
  results can differ slightly.
- `jl.solve` no longer recompiles on every call. Its device programs are
  cached per settings and problem shape, and the problem's numbers are passed
  as arguments. A repeat solve of the same shape skips tracing: a small LP
  re-solve takes 0.05 s instead of 0.8 s. Results are unchanged, and CPU
  results are bit-identical.
- `to_jaddle_sparse` (and `JaddleLP`) accept traced numbers.

### Removed

- `jc.solve(update_mode="forward_reflected")`. It was a net loss against
  adaptive extragradient, taking 2.3–2.9× more epochs. Use `"pdhg"` instead.

## 0.2.2

### Added

- `jaddle.jaddle_cvxpy`, a CVXPY front end (`pip install "jaddle[cvxpy]"`).
  `JaddleSolver` is a CVXPY QP solver, used as
  `prob.solve(solver=JaddleSolver())`. It solves LPs with
  `solve_with_presolve` and QPs with `quadratic_program`. `jcv.solve(prob)`
  compiles any other DCP problem into a `JaddleCP` for the convex solver.
  Smooth atoms (logistic, log-sum-exp, Huber, ...) stay smooth there, and
  piecewise-linear ones become epigraph rows, avoiding CVXPY's
  exponential-cone reformulation. Results are written back as CVXPY's
  (status, values, duals). See `examples/lp/cvxpy_transport.py` and
  `examples/cp/cvxpy_lasso.py`.

- LP and convex `solve`/`make_solver`: `diag_adapt` (with `diag_theta` and
  `diag_clamp`), a per-coordinate diagonal step metric on top of the scaling,
  adapted at each restart. The convex solver supports it only for adaptive
  extragradient with restarts, and leaves it off by default.

- LP `solve`/`make_solver`: `gap_test` selects the relative gap tested
  against `dual_gap_tolerance`. `"pdlp"` is PDLP's signed
  |primal_obj − dual_obj|; `"abs"` sums the absolute complementarity terms so
  they cannot cancel.

- Convex `solve`/`make_solver`: `gradient_scaling` (with `gradient_scaling_probes`)
  rescales the objective and each constraint row by its gradient norm at the
  start point, as IPOPT does. Off by default; the CVXPY compiler turns it on.

### Fixed

- Convex `solve`: a solve whose residuals had become NaN (a diverged run)
  passed the convergence test and was reported as certified. The test now
  requires every residual to be within tolerance, so NaN fails it.

### Changed

- LP `solve`/`make_solver`: `diag_adapt=True` is the default. On the MIPLIB
  2017 LP set it certifies 380/383 presolved instances (376 before) and cuts
  the shifted geometric-mean solve time from 8.62 s to 7.04 s.
- LP certificate: the default gap test is now `gap_test="pdlp"` (the signed
  gap, as in PDLP and cuPDLP). Pass `gap_test="abs"` for the previous,
  stricter no-cancellation test.

## 0.2.1

### Changed

- Convex `solve`: a solve that meets its convergence criteria now reports
  `stop_reason="certificate"`, as the LP solver does, instead of
  `"converged"`.
- Convex `solve` no longer prints its end-of-solve summary unless
  `verbose=True`. In both solvers the max-epochs and time-limit messages are
  printed only with `verbose=True`; the Ctrl-C notice is still always printed.

### Added

- `CITATION.cff`, for GitHub's "Cite this repository".
- Docstrings for `jaddle_linear.to_jaddle_sparse`,
  `jaddle_linear.lp_summary_statistics` and the optimisers in
  `jaddle_optimisers`.

## 0.2.0

### Added

**LP solver (`jaddle.jaddle_linear`)**

- `load`: read an MPS or LP file into a `JaddleLP`, optionally presolved by
  HiGHS.
- `solve_with_presolve`: HiGHS presolve, a Jaddle solve of the reduced LP and
  HiGHS postsolve of the primal and dual solution. The result is certified on
  the LP as given (`converged`), with the reduced solve's verdict in
  `reduced_converged`. Options `polish=True` and `run_crossover=True`.
- `crossover`: take a first-order solution to an optimal basic solution with
  HiGHS.
- `detect_infeasibility` and `evaluate_infeasibility_certificate`: PDLP-style
  primal and dual infeasibility detection.
- Batched and differentiable solves: `make_solver`, `solve_batch`,
  `make_optimal_value`, `make_solution`, `make_perturbed_solution`, the
  `LPValues` pytree and `SolveCoreResult`.
- `solve_with_polishing` now takes the three certificate tolerances, a time
  limit and an objective row for the primal polish
  (`polish_objective_slack`), with a configurable trigger (`polish_trigger`,
  `first_polish_epoch`, `polish_ready_factor`, `polish_fraction`).
- `evaluate_lp_certificate` takes `norm` and `dual_residual`.

**Convex solver (`jaddle.jaddle_convex`)**

- `quadratic_program`: QPs from `Q`, `c` and constraint matrices, with
  symmetric KKT equilibration and an LP-style certificate in the problem's
  own units.
- Batched and differentiable solves: `make_solver`, `make_optimal_value`,
  `make_solution` and `SolveCoreResult`.
- `report_best` (default on): test and return whichever of the average and the
  last iterate has the lower KKT merit.
- `JaddleCP` takes an optional `residuals` function to supply a problem's own
  stopping test.

**HiGHS helpers (`jaddle.highs_helpers`)**

- `read_model`, `jaddle_lp_to_highs`, `highs_duals_to_jaddle`,
  `jaddle_duals_to_highs`, `standard_form_row_map` and
  `standard_form_basis_to_highs`.

**Examples**

- `lp/battery_sizing.py`, `cp/lasso_regression.py`,
  `cp/fair_logistic_regression.py` and `cp/kernel_svm.py`.
- `lp/miplib.py` replaces `lp/miplib_standard.py`.

### Changed

- LP and convex `solve`: `epochs_per_restart` now defaults to `None` (no cap
  on the restart cycle length) instead of `10`.
- Convex `solve`: `update_mode` defaults to `"alternating"` instead of
  `"extragradient"`, and `average` defaults to `True`.
- The LP iteration loop runs on the device in chunks of about a second, which
  makes solves 2–3× faster. Runs are no longer bit-identical to 0.1.0's after
  a restart.

### Removed

- LP `solve`: `eq_projection_threshold`.
- `solve_with_polishing`: `max_rounds`.

### Fixed

- LP `solve`: the relative dual feasibility residual left out the cost
  normalisation factor. Under the default augmented scaling the factor is
  close to 1, but with `scaled_augmented=False` the solver could report false
  certificates.
- LP `solve`: a warm start's duals were mapped into the scaled problem without
  the cost normalisation factor, so warm starts did not start from the point
  supplied.

## 0.1.0

First release.
