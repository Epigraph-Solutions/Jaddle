# Changelog

## Unreleased

### Added

- `jaddle.jaddle_mip` (**highly experimental**): a branch-and-bound solver
  for mixed-integer linear programs, with `jaddle_linear.solve` at every
  node. Its interface, defaults and behaviour may change without notice;
  check its results against an established MIP solver.
  - `MIP` holds a Jaddle LP and a mask of integer columns, plus the
    objective constant and sense. `load` and `from_highs` read one from any
    model HiGHS reads (MPS/LP files, `highspy.Highs`, `highspy.HighsLp`),
    keeping its integrality, optionally after HiGHS's presolve; `to_highs`
    goes the other way.
  - `solve` runs the search: activity-based bound propagation at the root
    and at every child, warm-started node LPs, pseudocost or
    most-fractional branching, a dive to the first incumbent and then
    best-bound search, a rounding heuristic (with an LP repair of the
    continuous variables), and HiGHS presolve and postsolve with
    `presolve=True`.
  - Bounds from duals: each child is bounded by its parent's duals over
    its own propagated box (`lagrangian_terms`, `box_bound`) and closed
    without an LP solve when that reaches the cutoff; reduced-cost fixing
    (`reduced_cost_tightening`) at every node and on the root box whenever
    the incumbent improves.
  - Opt-in branch and cut (`tree_cuts=True`, `tree_cut_every`): global
    cover and MIR cuts separated at node LP points join a fixed-capacity
    cut pool, so the LP solver is never recompiled. On lseu (400 nodes)
    the bound reaches 1,080 instead of 1,064.
  - `device="auto"` (default): MIPs with at most 10,000 nonzeros run on
    the CPU, where their latency-bound node LPs are several times faster
    (flugpl: 3s, against 18s on a GPU).
  - Cuts are separated from the original rows as well as the
    coefficient-tightened ones.
  - Big-M coefficient tightening at the root (`tighten_coefficients`,
    `coefficient_tightening=True`): a binary's coefficient in an
    inequality row is cut to the row's excess over the propagated root
    box. Same integer-feasible points, a tighter relaxation and better
    conditioned rows; on fixnet6 the median within-row spread drops from
    500 to 5.
  - Cheaper incumbent repair: the LP over a rounded point's continuous
    variables gets its own budget (`repair_max_epochs`, default 20), and is
    skipped for integers that already failed, when no violated row has a
    continuous variable, or when propagation shows the fixed box
    infeasible (`lp_repairs`, `lp_repairs_skipped` in the result). On
    flugpl these LPs took two thirds of the time.
  - Integral objectives (`objective_step`): when every objective value is a
    multiple of a step, node bounds are rounded up to it and the cutoff is
    at most half a step below the incumbent.
  - Root cuts (cut-and-branch): knapsack cover cuts (up to three covers
    per row, lifted exactly by a dynamic program, `cover_cuts`) and c-MIR cuts on single and aggregated rows (`cover_cut`,
    `mir_cut`, `CutSeparator`), in up to `cut_rounds` rounds. The cut block
    is padded to power-of-two sizes so the rounds and the tree share one
    compiled LP solver. On a 6-site, 12-customer facility-location MIP
    they close 73% of the root gap.
  - A stronger MIR generator (`cmir_cut`): every integer coefficient as
    the scaling, efficacy in the original variables, and complementation
    flips (Marchand and Wolsey). Aggregation through integer variables
    (`aggregate_integers="auto"`, up to five rows) with MIR and cover
    cuts from every aggregate; cut rounds now stop on the gain over the
    last three rounds. lseu's root bound goes from 1,022 to 1,091 (HiGHS's
    cuts alone: 1,092) and it solves in 473 nodes instead of 1,929.
    `gomory_cuts(..., tableau_mir=True)` also runs c-MIR on tableau rows
    (off: it added little).
  - Gomory mixed-integer cuts (`gomory_cuts`, `gomory_cuts()`): each root
    round crosses over to an optimal vertex with `jaddle_linear.crossover`
    and builds GMI cuts from its basis. `"auto"` (default) turns them on for
    MIPs with at most 10,000 nonzeros, with up to 20 cut rounds
    (`cut_rounds=None`, `cut_min_progress=None`); lseu's root bound goes
    from 1,022 to 1,045, p0282's from 245,918 to 256,076.
  - After the cut rounds the cut block is rebuilt without idle cuts or
    growth padding, so node LPs stay small.
  - Feasibility jump (`feasibility_jump`): an LP-free weighted local search
    for feasible points, run on the device in compiled chunks, at the root
    before the feasibility pump.
  - Plunging in the default `"hybrid"` node selection (`plunge_depth`).
  - Batched node solves (`batch_size`): open nodes solved together in one
    `vmap`-ed `jaddle_linear.make_solver` call.
  - Opt-in HiGHS help for hard instances: `highs_incumbent_seconds` (a first
    incumbent from HiGHS).
  - Cut rounds re-solve the root LP for at most `cut_round_epochs` (20)
    epochs, with one converged solve after the last round. Root time on
    lseu 17s → 7s, mod008 25s → 7s; full solves: lseu 33s → 15s, p0282
    16s → 6s (141 → 19 nodes).
  - Incumbent heuristics: RENS at the root (`rens_seconds`, on for small
    MIPs), LP-based dives (`dive_every`, opt-in; fractional, guided and
    pseudocost rules), objective-aware propagation in fix-and-propagate
    and dives, and RINS triggered by every new incumbent. A shared
    `sub_search` runs RINS and RENS sub-MIPs on the compiled LP solver.
    On p0282, propagation and RENS cut the search from 301 to 141 nodes.
  - Fixed: a search could pass `jl.solve` a non-positive time limit when
    its deadline fell between the time check and the LP solve.
  - RINS (`rins_seconds`, opt-in), at the root and every `rins_every` nodes,
    with the sub-MIP searched by Jaddle's own branch and bound
    (`rins_solver="jaddle"`; `rins_nodes`) or by HiGHS (`"highs"`).
    `"auto"` (default) uses Jaddle for MIPs with at most 10,000 nonzeros and
    HiGHS above that: on ns1830653 (300s) the Jaddle sub-searches took a
    third of the run and left a 71% gap, against 52% with HiGHS. The Jaddle
    sub-search reuses the compiled LP solver, warm-starts its root, and
    prunes at the parent's cutoff.
  - Primal heuristics: `fix_and_propagate` (chunked rounding with bound
    propagation, at the root and in the tree) and an objective feasibility
    pump at the root, whose LPs change only the cost and so reuse the
    root's compiled solver.
  - Nodes are pruned on the Lagrangian bound of their duals
    (`lagrangian_bound`), which is valid for approximate first-order
    solutions and for LPs that did not converge.
  - `examples/mip/facility_location.py` and `examples/mip/miplib.py`.
- `highs_helpers.highs_integrality` reads a HiGHS model's integer columns,
  and `highs_helpers.jaddle_lp_to_highs` takes an optional `integrality`.
- `examples/cp/monitor_network_design.py`: optimal experimental design on
  open data. It chooses which of 470 candidate EPA PM2.5 sites to monitor so
  that the daily PM2.5 at 235 held-out target sites is predicted best. A
  factor model is fitted to 2023 data. The relaxed I-optimal design (one budget row, box bounds) is
  solved with `jc.solve`, rounded, and improved by Woodbury-updated
  exchanges, and its optimum certifies the 50-monitor network to within
  1.3%. The chosen networks are checked on 2024 against random,
  evenly spread, greedy and D-optimal ones; the designed networks have 8%
  less error than random ones. Other parts of the example:
  - one `vmap`-ed `jc.make_solver` call sweeps 16 budgets, and the dual of
    the budget row gives the marginal value of a monitor;
  - one `jax.grad` of `jc.make_optimal_value` differentiates the optimal
    design's error with respect to the budget and all 705 noise variances.

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
