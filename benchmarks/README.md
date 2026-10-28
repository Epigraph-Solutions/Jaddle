# Jaddle Benchmarks

Scripts for measuring Jaddle's solvers. They come in four kinds:

| Kind | Scripts | Purpose |
|---|---|---|
| **LP harnesses** | [`benchmark.py`](benchmark.py), [`benchmark_mpax.py`](benchmark_mpax.py), [`benchmark_pipeline.py`](benchmark_pipeline.py) | Solve the LP relaxations of a directory of MIPLIB instances with `jaddle_linear.solve`, then write a CSV and a README-ready markdown table. `benchmark_mpax.py` also runs [MPAX](https://github.com/MIT-Lu-Lab/MPAX) on the same presolved LPs for a head-to-head comparison. `benchmark_pipeline.py` runs the whole `jl.solve_with_presolve` pipeline with polishing and crossover. |
| **Convex A/B suites** | [`benchmark_convex.py`](benchmark_convex.py), [`benchmark_convex_heavy.py`](benchmark_convex_heavy.py) | Compare `jaddle_convex.solve` configurations (update mode, adaptive step, restarts) on synthetic convex problems. |
| **MIP comparison** | [`benchmark_small.py`](benchmark_small.py) | Run `jaddle_mip.solve` and HiGHS on 28 small MIPLIB 3.0 instances, downloaded into `data/small/` on first use, with a time limit each (`--seconds`, default 120). Writes `results/small_mips.csv` and prints a summary table. |
| **Primal heuristics** | [`benchmark_primal.py`](benchmark_primal.py) | Run only Jaddle's root heuristics (no HiGHS RENS, RINS or incumbent; HiGHS presolve) on the smallest `--count` MIPLIB 2017 files in `data/` with a known objective (default 30, 60 s each). It records whether a feasible point was found, how fast and by which heuristic, and the primal gap to MIPLIB's best known value (`data/miplib2017.solu`, downloaded on first use). `--set key=value` passes any `jm.solve` setting. Writes `results/primal.csv`. |
| **Instance diagnostics** | [`scan_bigm.py`](scan_bigm.py) | Flag MPS instances whose big-M structure is known to stall saddle-point solvers. The LP harnesses use it to skip those instances. |

Run every script from the repo root as `python benchmarks/<script>.py`. The
scripts import from each other (`benchmark_mpax` imports its presolve and
instance discovery from `benchmark`, and both LP harnesses import the detectors
from `scan_bigm`).
Python resolves these imports because it puts the script's own directory on
`sys.path`.

## Getting the data

The MPS files are **not** shipped with Jaddle (`data/` is gitignored). Download
instances from the [MIPLIB website](https://miplib.zib.de/) and put the
`.mps` files in `data/` at the repo root. The LP harnesses and `scan_bigm.py`
run on every `*.mps` file in that directory. Use `--data-dir` to point them
somewhere else.

The PDLP papers' reference set is the 383-instance list from FirstOrderLp.jl
(`mip_relaxations_instance_list`, 100k–10M nonzeros).

## LP harnesses

Both harnesses run Jaddle with the same settings (`jaddle_solve_kwargs` in
`benchmark.py`): `iterations_per_epoch=640`, `epochs_per_restart=10`, and the
primal, dual and gap tolerances all set to `--tol`.

`benchmark.py` hands each instance file to `jl.solve_with_presolve`: HiGHS
presolve, the defined-variable elimination, Jaddle on the reduced LP, then
postsolve of both the primal and the dual back to the original problem. The
reported objective is the original one. `benchmark_mpax.py` presolves each
instance once (HiGHS plus the defined-variable elimination, or nothing with
`--no-presolve`), hands the identical reduced LP to both solvers in separate
worker processes with the whole GPU, and re-checks every returned primal-dual
pair with `jl.evaluate_lp_certificate` in the original units. Its header
comment lists the remaining fairness choices.

| Script | Presolve | Reference optimum | Default CSV |
|---|---|---|---|
| `benchmark.py` | HiGHS | Yes. A HiGHS solve (`simplex` by default) is used as a ground-truth objective oracle, and the relative gap to it is reported. | `benchmark_results.csv` |
| `benchmark_mpax.py` | HiGHS, or none | Off by default: both solvers are judged by the shared certificate re-check. `--highs-solver` adds a HiGHS solve and reports each solver's relative gap to it. | `benchmark_mpax_results.csv` |

CSVs are written to the repo root. `benchmark.py` rewrites its CSV after every
instance, so a crash partway through a sweep keeps the results so far. It also
clears JAX's compile caches between instances to keep host memory flat over
long runs.

### Usage

```bash
# Every instance in data/ (100 MB cap by default)
python benchmarks/benchmark.py

# A named subset
python benchmarks/benchmark.py --only stp3d boeing

# A tighter tolerance and a per-instance time budget
python benchmarks/benchmark.py --tol 1e-4 --max-seconds 600

# Jaddle only: skip the HiGHS reference solve (much faster on large LPs)
python benchmarks/benchmark.py --highs-solver none

# Jaddle vs MPAX (needs `pip install mpax`)
python benchmarks/benchmark_mpax.py --tol 1e-4 --max-seconds 600
python benchmarks/benchmark_mpax.py --no-presolve
```

### Options shared by both harnesses

| Option | Default | Meaning |
|---|---|---|
| `--data-dir DIR` | `data/` | Directory to glob for `*.mps` files. |
| `--only NAME ...` | all | Restrict the run to these instance names (without `.mps`). |
| `--max-mb X` / `--min-mb X` | none / `0` | Skip files above or below a size in MB. By default there is no upper limit. |
| `--tol X` | `1e-3` | Relative tolerance for primal feasibility, dual feasibility and duality gap. |
| `--max-epochs N` | none | Cap on Jaddle epochs. |
| `--max-seconds S` | none | Per-instance Jaddle wall-clock budget, including setup and XLA compile. It is checked at epoch boundaries. |
| `--update-mode` | `alternating` | Passed to `jl.solve`: `alternating`, `pdhg` or `halpern`. |
| `--cost-col-floor X` | `0` | Passed to `jl.solve` as `cost_col_floor`. A costed column whose largest scaled matrix entry is below `X` is rescaled so that entry becomes 1. This fixes epigraph objectives such as fhnw-binschedule0. `0` disables it, as in `jl.solve`'s default; try `1e-2` to enable it. |
| `--norm {l2,inf}` | `l2` | Norm for Jaddle's primal and dual feasibility residuals, in both the termination test and the restart merit (`jl.solve`'s `termination_norm` and `restart_norm`). In `benchmark_mpax.py` it also sets MPAX's norm and the norm of the shared re-check. |
| `--jax-profile` | `float64` | Passed to `configure_jax`: `float64`, `float32` or `float16`. |
| `--skip-bigm` | off | Skip instances with cost big-M (see [`scan_bigm.py`](#scan_bigmpy)). |
| `--skip-bigm-matrix` | off | Skip instances with matrix (row) big-M. |
| `--skip-bigm-column` | off | Skip instances with column big-M. |
| `--presolve` | on | Presolve with HiGHS before Jaddle (and MPAX). Use `--no-presolve` to solve the file's LP relaxation as given, as in the cuPDLP and MPAX papers. That also skips the defined-variable elimination and turns presolve off in the HiGHS reference solve. Jaddle's certificate is then on the original LP. |
| `--eliminate-defined-vars` | on | After presolve, substitute out variables defined by a dense equality row `z = Σ aⱼxⱼ` (see `jaddle.presolve.eliminate_defined_variables`). Use `--no-eliminate-defined-vars` to turn it off. It is ignored with `--no-presolve`. |
| `--jaddle-verbose` | on | Run Jaddle with `verbose=True`, printing its per-epoch log and restart messages. Use `--no-jaddle-verbose` to silence it; with no `--max-epochs` or `--max-seconds` budget either, each solve then runs as a single device call. |
| `--csv PATH` | see table above | Where to write the results. |

### Options for a single harness

| Script | Option | Meaning |
|---|---|---|
| `benchmark.py` | `--highs-solver {simplex,ipm,pdlp,none}` | Solver for the reference optimum. Use `none` to skip it, in which case `rel_obj_gap` is NaN. The reference solve often takes longer than Jaddle's, so skip it for A/B runs if you already know the optima. |
| `benchmark.py` | `--highs-kkt-tolerance X` | KKT tolerance for the reference solve (defaults to `--tol`). |
| `benchmark.py` | `--highs-verbose` | Show HiGHS's own log. |
| `benchmark_mpax.py` | `--solvers {jaddle,mpax} ...` | Run only some of the solvers (default: both). |
| `benchmark_mpax.py` | `--highs-solver {simplex,ipm,pdlp,none}` | Also solve each instance with HiGHS and report both solvers' objective gap to its optimum (default `none`: no HiGHS solve). HiGHS runs on the original file with no time limit. |
| `benchmark_mpax.py` | `--mpax-algorithm {r2hpdhg,rapdhg}` | MPAX solver (default `r2hpdhg`). |
| `benchmark_mpax.py` | `--mpax-norm {2,inf}` | Norm for MPAX alone (default: the one `--norm` selects). If it differs from `--norm`, MPAX stops on a different norm than it is re-checked in. |
| `benchmark_mpax.py` | `--mpax-iteration-limit N` | Optional MPAX iteration cap. |
| `benchmark_mpax.py` | `--startup-timeout S` | Seconds a worker may take to start and load the LP (default `600`). |
| `benchmark_mpax.py` | `--verbose` | Stream both solvers' progress logs. `--jaddle-verbose` follows it unless set explicitly. |

### Reading the output

Three timings are reported for Jaddle:

- **`jaddle_solve_seconds`**: time spent in the iteration loop, including the
  first-epoch XLA compile but excluding scaling and setup. This is the headline
  figure.
- **`jaddle_corrected_seconds`**: the same, with the one-off compile amortised
  out: `n · (solve − first_epoch) / (n − 1)`. It estimates the runtime if
  every epoch had run at the warm rate.
- **`jaddle_wall_seconds`**: the whole call. In `benchmark.py` that includes
  presolve (`jaddle_presolve_seconds`) and postsolve.

`jaddle_converged` can be true for more than one reason, so check
`jaddle_stop_reason` alongside it:

| `jaddle_stop_reason` | Meaning |
|---|---|
| `certificate` | Full LP optimality certificate met (primal feasibility, dual feasibility and gap). |
| `primal_stall` | Primal-stop heuristic fired: the point is feasible but not certified optimal. |
| `max_epochs` / `time_limit` | Budget exhausted. |
| `presolve_infeasible` / `presolve_solved` | HiGHS presolve settled the problem itself (`benchmark.py` only). |

`jaddle_converged` means certified on the **presolved** LP, as in earlier
sweeps. `benchmark.py` also judges the postsolved point on the LP as given:
`jaddle_original_certified`, with that certificate's residuals in
`jaddle_orig_pfr`, `jaddle_orig_dfr` and `jaddle_orig_gap`. A point that
misses the original certificate is reported as such; no further solve is
attempted. The two verdicts can disagree when presolve rescales the problem, because the
tolerances are relative. leo1's presolve folds a row with 1e7 coefficients into
the cost (‖c‖ 1 → 3e9): the objective is right, but the original's dual test
fails. A certificate on a drastically rescaled reduced LP can also be false.
On proteindesign122trx11p8 (‖c‖ 1 → 6e6 after elimination), one run
"certified" the reduced LP after 13 epochs with an objective 7% above the
optimum, and `jaddle_original_certified` was false.

`benchmark.py` also reports `rel_obj_gap = |jaddle − opt| / (1 + |opt|)`
(PDLP's normalisation) against the HiGHS optimum, and both the original and the
presolved sizes (`n_vars`/`n_cons`, `n_vars_presolved`/`n_cons_presolved`,
counted as HiGHS columns and rows).

## Full-pipeline harness

`benchmark_pipeline.py` runs every step of `jl.solve_with_presolve` on each
instance:

1. HiGHS presolve.
2. Defined-variable elimination.
3. Jaddle with feasibility polishing (`polish=True`).
4. Crossover to an optimal vertex of the reduced LP (`run_crossover=True`).
5. HiGHS postsolve of that vertex and its basis.

The answer is a vertex, so the script compares it with an exact HiGHS solve
(`--highs-solver`, `simplex` by default). It compares the objectives and also
the end-to-end times: HiGHS's whole `run()` against Jaddle's whole call. It
shares `benchmark.py`'s Jaddle settings and its instance options (`--only`,
`--max-mb`, `--skip-bigm*`, `--tol`, `--norm`, ...). It adds these options:

| Option | Default | Meaning |
|---|---|---|
| `--polish` / `--no-polish` | on | Solve the reduced LP with feasibility polishing. |
| `--first-polish-epoch N`, `--polish-fraction X`, `--polish-trigger {doubling,auto}` | `16`, `0.125`, `doubling` | Passed to `solve_with_polishing`. |
| `--gap-tol X` | `--tol` | Gap tolerance alone. Polishing fires only once the gap is within it. |
| `--crossover` / `--no-crossover` | on | Cross over to a vertex before postsolve. |
| `--crossover-time-limit S` | none | HiGHS `time_limit` passed as `highs_options`. It applies to the presolving HiGHS instance too. |

The CSV (`benchmark_pipeline_results.csv` by default) splits Jaddle's wall
time into `jaddle_presolve_seconds`, `jaddle_solve_seconds` (with polishing,
the whole polished solve), `crossover_seconds` and `jaddle_other_seconds`.
The last covers elimination, postsolve with its simplex repair, and the
certificate check. It also records the crossover's statistics:
`crossover_status`, `crossover_converged`, `crossover_pushes`,
`crossover_simplex_iterations` and `postsolve_simplex_iterations`. Then
`jaddle_original_certified` reports whether the returned vertex is certified
on the LP as given.

```bash
python benchmarks/benchmark_pipeline.py --only stp3d mzzv11
python benchmarks/benchmark_pipeline.py --tol 1e-4 --max-seconds 600
python benchmarks/benchmark_pipeline.py --no-crossover   # polish only
```

## Convex A/B suites

Positional arguments select which configs or modes to run. The only flag is
`--jaddle-verbose` (off by default), which shows Jaddle's own per-epoch log
above each result line.

### `benchmark_convex.py`

Runs every (problem, config) pair on a small suite of synthetic problems.
Each run has a budget of 300 epochs × 200 iterations and uses tolerance `1e-6`.

| Problem | Structure |
|---|---|
| `isotonic` | 2000-variable least squares with ordering inequalities and box bounds |
| `ridge_ball` | RFF regression with an active `‖w‖² ≤ r` constraint |
| `simplex_qp` | Badly scaled QP on the probability simplex (equality constraint plus bounds) |
| `small_lp` | Random feasible bounded LP, posed as a convex program |
| `logistic_l1` | Sparse logistic regression with an L1-ball constraint written as many inequalities |

Every config except `default` starts from `OLD_DEFAULTS`, the solver's
defaults before 2026-10-20 (`k_scale=10`, no restarts, no diagonal metric,
no gradient scaling), and adds the settings below. `iterations_per_epoch` is
200 for every config.

| Config | Settings |
|---|---|
| `default` | `jc.solve`'s current defaults |
| `old_default` | `OLD_DEFAULTS` alone |
| `alt_gd` | `alternating` with plain gradient descent (`jo.gd(0.05)`) |
| `alt_ad` | `alternating` with adaptive step |
| `alt_ad_rs` | `alt_ad` with restarts every 5 epochs |
| `alt_ad_rs_dg` | `alt_ad_rs` with the diagonal metric (`diag_adapt`) |
| `eg_ad` | `extragradient` with adaptive step |
| `eg_ad_rs` | `eg_ad` with restarts every 5 epochs |
| `eg_ad_rs_dg` | `eg_ad_rs` with the diagonal metric |
| `eg_ad_rs_gs` | `eg_ad_rs` with gradient scaling |
| `pdhg_ad` | `pdhg` with adaptive step |
| `pdhg_ad_rs` | `pdhg_ad` with restarts every 5 epochs |
| `pdhg_ad_rs_dg` | `pdhg_ad_rs` with the diagonal metric |

```bash
python benchmarks/benchmark_convex.py              # all configs
python benchmarks/benchmark_convex.py eg_ad pdhg_ad # only these configs
python benchmarks/benchmark_convex.py eg_ad --jaddle-verbose  # with Jaddle's log
```

Each line of output gives convergence, epochs, solve time (excluding compile)
and final objective.

### `benchmark_convex_heavy.py`

A single gradient-heavy problem: dense L1-constrained logistic regression,
20000 × 1000. Each gradient costs two dense matvecs, so per-iteration cost is
dominated by gradient evaluations. The small suite above is dominated by
overhead instead. Use this script to weigh cost per epoch against epochs to
converge across update modes.

```bash
python benchmarks/benchmark_convex_heavy.py       # extragradient and pdhg
python benchmarks/benchmark_convex_heavy.py pdhg  # one mode
```

Each line of output gives convergence, epochs, total time, time per epoch and
objective.

## `scan_bigm.py`

Scans every `.mps` file in a directory and flags three independent kinds of
big-M / penalty structure. Each one stalls Jaddle in a different way, and Ruiz
scaling cannot remove any of them:

| Flag | Detector | Rule (defaults) | Typical instances | Failure mode |
|---|---|---|---|---|
| **Cost** | `is_bigm_cost` | `max\|c\| / median\|c\| > ratio`, and either `max\|c\| > abs` or `median\|c\| < med-floor` | `glass4`, `binkar10_1`, `mas74`, `mas76` | Stalls at a feasible point with a frozen duality gap, well short of the optimum |
| **Matrix (row)** | `is_bigm_matrix` | median per-row spread `max\|A_row\|/min\|A_row\| > 1e3`, and at least 50% of rows wide (spread > `1e2`) | `sp150x300d`, `neos-3754480-nidda` | Plateau in primal feasibility |
| **Column** | `is_bigm_column` | at least one dense big-M row: `max\|A_row\| > 1e3` and touching at least 30% of the columns | `germanrr`, `leo1`, `trento1` | Primal never approaches feasibility |

```bash
python benchmarks/scan_bigm.py
python benchmarks/scan_bigm.py --data-dir data --ratio 1e3 --abs 1e4
python benchmarks/scan_bigm.py --row-spread 1e3 --row-frac 0.5 --wide-row 1e2
python benchmarks/scan_bigm.py --col-row-abs 1e3 --col-row-density 0.3
```

Each MPS file is read once and all three checks run on it. The output is one
table per flag, sorted by severity, followed by a list of the flagged names.
The LP harnesses' `--skip-bigm*` options call the same detector functions with
the same default thresholds, so the scanner's output is exactly what those
options will skip.

## Tips

- **A/B comparisons:** pass `--highs-solver none` to `benchmark.py` so the
  reference solve doesn't dominate wall time.
- **GPU nondeterminism:** identical GPU runs can differ by several epochs,
  because scatter-add order varies. Repeat both arms before trusting a small
  difference in epoch counts.
- **Large sweeps:** use `--max-seconds` to bound how long any one instance
  can take, and `--max-mb`/`--min-mb` to choose a size band.
