# Changelog

## Unreleased

### Added

- Cut separation counts implied integers as integers
  (`jm.solve(implied_integer_cuts=True)`, the default), as HiGHS's cuts
  do. This covers the Gomory, cover, c-MIR, flow cover, clique (the
  conflict graph's binaries) and zero-half cuts, at the root and in the
  tree (`CutSeparator(integrality=...)`, `ConflictGraph(integrality=...)`).
  - Presolve has shown that every choice of the integers has a best
    completion with them integral. No such cut removes it, so node bounds
    stay valid, and node points with fractional implied columns still
    count as integral.
  - On ns1830653 the root bound (100 rounds, no restarts) rose from 11,819
    to 12,583 (HiGHS: 11,622), with 92 zero-half and 20 clique cuts in the
    LP against 8 and 3.
- Cut pool propagation in `jm.solve` (`cut_pool_propagation`, off by
  default). Every cut the root and tree rounds select is kept (those with
  at most `max(32, n / 8)` nonzeros, the newest 10,000), and the search's
  bound propagation runs them beside the rows at every node, also after
  they have left the LP. HiGHS propagates its cut pool the same way: on
  ns1830653 it keeps 6,000 to 10,000 cuts with 100 to 170 in its LP, and
  about 38% of its nodes are leaves. Jaddle's open nodes there grew almost
  one for one with the nodes solved. New result: `cut_pool_rows`.
- Three ways for `jm.solve` to raise the bound with the incumbent, all
  off by default until compared on large MIPs:
  - `cutoff_probing` (the most candidates per incumbent; with
    `node_lp_solver="highs"`, within `cutoff_probing_share`, default 0.1,
    of the elapsed time): exact LPs for the integers that root
    reduced-cost fixing only just fails to fix. Each is moved one unit off
    the bound its reduced cost favours, and HiGHS's dual simplex solves
    that child from the root basis. If the child is infeasible or its
    bound reaches the cutoff, the integer is fixed in the root box (the
    fixings count towards restarts), and the child's dual proof joins the
    conflict pool.
  - `root_redcost_propagation`: the root's reduced-cost row joins the
    conflict pool (`ConflictPool.set_global`) and propagates at every node
    with the current cutoff. A node that has moved variables off their
    bounds has used up part of the room, which can fix others before its
    LP.
  - `global_redcost_fixing`: reduced-cost fixing of the root box with every
    node LP's and probing child's duals, not only the root's.

  New results: `cutoff_probing_lps`, `cutoff_probing_fixed`,
  `cutoff_probing_seconds`, `global_redcost_fixed`.
- Settings for the growth of the root cut pool in `jm.solve`. The defaults
  keep the current behaviour.
  - `simplex_cut_age=10`: the number of rounds in a row a cut may stay
    slack in the simplex rounds (until now fixed at 10).
  - `max_cut_rows=None`: a cap on the cut rows in the root LP. A round
    that would pass it drops the kept cuts that have been slack longest
    (never a binding one), then takes fewer new cuts.
  - `gomory_candidates=None`: the number of tableau rows a round tries
    for Gomory cuts (until now always `2 × max_cuts_per_round`).

  On bab5 (20 rounds, `max_cuts_per_round=500`), slack cuts kept for 10
  rounds grew the LP to about 3,300 cut rows, 2,400 of them zero-half
  cuts, and a round slowed from about 5 s to 35 s.
- Restarts in `jm.solve` (`restarts=5`, the most; `0` turns them off).
  Once reduced-cost fixing of the root box has fixed 2.5% of the integers
  free at the start of a round (at the root, after its heuristics) or 10%
  (in the tree), the search starts again from that box, as HiGHS does:
  coefficient tightening, probing, the root LP, the cut rounds and the
  root's heuristics are redone over the smaller box; the incumbent,
  pseudocosts, conflict pool and symmetry rows carry over, the cuts that
  were tight or had a dual at their last solve join the searched MIP's
  rows (so the new root starts from about their bound), and the bound
  reached so far stays as a floor. `result["restarts"]` and
  `result["restart_log"]` report them. HiGHS restarts four times on bab5
  (20,508 → 11,243 columns); Jaddle's root reduced-cost fixing there fixed
  up to 14% of the integers.
- Symmetry breaking in `jm.solve` (`symmetry="auto"`: on when igraph is
  installed, `pip install jaddle[symmetry]`). `jm.symmetry_generators`
  finds the MIP's column symmetries over the root box as automorphisms of
  a coloured graph with bliss (through igraph), each checked exactly;
  `jm.orbitopes` assembles the ones that swap blocks of columns into
  orbitopes; `jm.symmetry_rows` breaks each at the root, keeping an optimal
  solution: packing orbitopes by `x[i, j] = 0` for `j > i` and
  `x[i, j] <= Σ_{k<i} x[k, j−1]`, others by ordering the columns' sums.
  The rows join the searched MIP (as coefficient tightening's do);
  incumbents are checked against the MIP as given. `result["symmetry"]`
  reports it. On bab5 bliss takes 0.02 s and finds 5 two-column orbitopes
  of 440 or 446 binaries (4,412 columns, none packing): 5 rows. A 600 s
  run found 104 strong-branching fixings (22 without), but its 20-round
  root bound was −108,874 (−107,344 in the run without), so it ended at a
  2.66% gap.
- Exact strong branching with HiGHS node LPs. With
  `node_lp_solver="highs"`, strong branching's child LPs are HiGHS's dual
  simplex from the node's basis, each for at most twice the node LPs' mean
  simplex iterations; a child HiGHS proves infeasible fixes its variable
  the other way. `jm.solve`'s `branching` default is now `"auto"`:
  reliability branching with HiGHS node LPs (strong branching in the tree
  pauses while it has taken more than half the elapsed time), pseudocost
  branching with Jaddle's (`node_lp_solver="jaddle"`). With HiGHS node LPs the root sweep takes
  `4 · strong_candidates` candidates. On bab5 plain pseudocost branching
  chose a variable with no observation in some direction 93% of the time,
  ranking a median 17th of about 60 by exact strong branching; the old
  root sweep (8 candidates, 50 PDHG epochs) recorded no gain for 7 of its
  8 variables, whose exact gains were 5–20. In 600 s runs the new
  branching found 22 fixings but solved 292 nodes instead of about 1,080,
  and the bound still levelled off near −107,330 to −107,480, as before.
- `jm.solve(node_lp_solver="highs")`, the default: node LPs solved exactly
  by HiGHS's dual simplex, one node at a time, each from the basis of the
  node before (the parent's while plunging). A node's bound is its LP
  value, rigorous from the vertex's exact duals; its point is a vertex;
  pseudocosts learn from every node. Dives use it too. `batch_size` is 1
  with it, so tree cuts are available. `node_lp_solver="jaddle"` keeps Jaddle's
  batched PDHG node LPs. On bab5 (benchmark settings, GPU) the PDHG node
  LPs rarely converged: their bounds sat a median 239 below the exact LP
  values at the same nodes, every child kept its parent's bound, about 470
  nodes finished in 600 s, and the global bound never left the root's.
  With HiGHS, about 1,080 nodes finished (0.11 s a node LP) and the bound
  rose from −107,565 to −107,484. New results: `node_lp_solver`,
  `node_lp_solves`, `node_simplex_iterations`, `node_lp_seconds`,
  `node_lp_fallbacks`.
- `jm.solve(max_tree_cuts_per_round=10)`: the most cuts a tree cut round
  (`tree_cuts`) adds, until now fixed at 10. The pool reserves room for at
  least two rounds' worth beyond the root's cuts. On a generalised
  assignment test model, 40 a round adds 40, 25 and 32 in its first three
  rounds.
- `jm.solve(tree_cut_rows=None, tree_cut_nnz=None)`: the size of the tree
  cut pool (cuts, the root's included, and their nonzeros), until now
  fixed at twice the root's. With only `tree_cut_rows`, the nonzeros
  scale with it. `result["tree_cuts_dropped"]` counts violated tree cuts
  left out for want of room, and the log's cut lines show the pool's
  cuts and nonzeros. On a generalised assignment test model with 40 cuts
  a round, the default pool fills and drops 156 cuts; with
  `tree_cut_rows=1000` it drops none.
- `jm.highs_rens`: RENS with HiGHS, with a dive when the RENS box is
  infeasible. Degenerate root LPs have vertices that are integral on many
  integers by accident, and fixing them all can leave no solution. When
  HiGHS proves the box empty, the time left goes to a dive in the manner
  of HiGHS's RENS: it fixes a tenth of the integers at a time towards the
  LP point and re-solves the LP with HiGHS's dual simplex until half as
  many are fixed as in the failed box, then searches that sub-MIP with the
  unfixed integers free. A box proven empty backs the dive up to half its
  depth and halves the target (down to 10%). `jm.solve`'s root RENS uses
  it. On neos-4306827-ravan, 39,000 of 41,000 integers are integral at the
  root vertex but only 331 have a nonzero reduced cost; the RENS box is
  infeasible in 0.2 s, and Jaddle found no incumbent in 300 s. The dive
  finds 11,557 offline (its first point after 5 s), and `jm.solve` now
  ends the 300 s run at 11,004.5 (HiGHS: 10,320.6), with a better bound
  (4,782 against 4,724).
- `jm.solve` simplex cut rounds (`simplex_cut_rounds="auto"`: on with
  Gomory cuts). The root LP stays in HiGHS through the cut rounds: the
  first crossover's basis is kept, each round deletes the cuts slack for
  10 rounds in a row, adds the new ones and re-solves with the dual
  simplex from that basis. Each round gets an exact vertex, exact duals
  and the Gomory basis without a first-order re-solve or a crossover, and
  the stall check sees exact bounds (over 10 rounds). The root is solved
  once more with the first-order method afterwards, warm from the exact
  point. With clique and zero-half cuts, bab5's root bound goes from
  −112,232 (71 s, first-order rounds) to −107,349 (171 s, 13 s of it in
  27 re-solves) and ns1830653's from 11,962 (32 s) to 12,622 (30 s), past
  HiGHS's root bounds of −107,454 and 11,622. `result["simplex_rounds"]`
  counts the re-solves.
- `jm.solve` zero-half cuts (`zero_half_cuts=True`, family `"zerohalf"`).
  `jm.zero_half_cuts` separates {0, ½}-Chvátal–Gomory cuts: half the sum of
  a set of rows with integer coefficients on integer variables, rounded
  down, with each variable measured from its nearer bound. The sets come
  from Gaussian elimination over GF(2) on the parities of the tight rows
  (packed 64-bit words), as in HiGHS's mod-k separator with k = 2. Root and
  tree rounds separate them from the LP's rows and the cuts in it. On
  bab5's root, with exact LPs and 20 rounds, they take the bound with the
  other families from −111,209 to −107,410, past HiGHS's root bound
  (−107,454); alone they reach −111,545 in 4 s.
- `jaddle.jaddle_stochastic`: stochastic saddle-point optimisation for
  convex problems whose objective and constraints are means over data.
  `JaddleCPStochastic` takes each term as a per-example function
  `fn(x, example)` with its data (a pytree with a leading example axis),
  plus optional deterministic parts (`*_det`). `js.solve` runs
  `jc.solve`'s algorithms on minibatch gradients of the Lagrangian
  (`batch_size`; every gradient within one iteration shares its minibatch),
  with SVRG variance reduction by default (`variance_reduction="svrg"`: an
  exact pass per epoch anchors the estimates). `variance_reduction=None`
  gives plain stochastic gradients, with a `step_decay` that shrinks the
  updates but not their exploration. The convergence test, restart merits,
  primal-weight initialisation and gradient scales use the full problem
  (`JaddleCPStochastic.full_cp`, `eval_chunk_size` for data that does not
  fit in one pass), so `"certificate"` certifies the full problem. The
  default `update_mode` is `"extragradient"`: `"alternating"` and `"pdhg"`
  are not contractive and can diverge when constraints are sampled. On
  fairness-constrained logistic regression with 200,000 examples, it
  certifies at 1e-5 in 1.3 s on CPU, where `jc.solve` takes 2.7 s.
- `jc.GradientOracle`: the hook through which `jc.solve`'s epoch loop takes
  stochastic gradients.
- `jm.solve` clique cuts (`clique_cuts=True`, family `"clique"`).
  `jm.ConflictGraph` builds a conflict graph on the binaries once from the
  rows over the root box (two literals conflict when a row cannot hold with
  both at 1), storing each row's clique and its other conflicts as edges.
  Root and tree rounds grow cliques greedily from the LP point's positive
  literals and lift them with literals at zero. On bab5 the graph takes
  0.2 s; on exact LPs the clique cuts alone take the root bound from
  −115,370 to −113,411 in 56 cuts, and they add about 1,000 to it alongside
  the other families.

### Fixed

- `MIP.violation` and `MIP.is_feasible`, and with them every incumbent
  check in `jm.solve`, now measure each row's violation against its own
  `1 + |b_i|`. Before, every row's was divided by `1 + ‖b‖∞`, so one large
  right-hand side excused large violations of every other row. On map06
  (one right-hand side of 5.1e7) the root rounding was taken as an
  incumbent at -402.49 against a known optimum of -289, with 7,174 rows
  violated by up to 0.32, and the search stopped as "optimal". The jump
  heuristic's row tolerances (`_jump_rows`) and the repair's violated rows
  (`_violated_rows`) use the same per-row measure.
- With HiGHS node LPs, each node's LP now starts from its parent's basis.
  The basis is stored with the node (`_Node.basis`, for the rows it
  belongs to) and set again when HiGHS holds another one. Before, every
  node LP started from whatever basis the last node left behind. Within a
  plunge that is the parent's, but after a jump to the best-bound node
  it's an unrelated node's. On ns1830653 node LPs averaged 215 to 353
  pivots, against about 70 for HiGHS's own search. A plunge child solved
  just after its parent skips the reset (no refactorisation). New result:
  `node_basis_restores`.
- Strong branching in the tree (reliability branching with HiGHS node LPs)
  is capped at 25% of the time spent in the tree, after a 2 s grace
  period. It used to be capped at half of all the time since the start,
  root included. So once the tree started it could run nonstop for as long
  as the root had taken. On ns1830653, with a 52 s root (100 cut rounds),
  a profile put 49 of the tree's 56 s in strong-branching child LPs: about
  16 a node at 28 ms each, against 20 ms for the node LP itself. HiGHS
  spends 27% of its simplex iterations on strong branching there.
- HiGHS's implicit integers now load as continuous columns. Presolve marks
  continuous columns that some optimal solution has integral
  (`kImplicitInteger`). They were loaded as integers, so a node LP point
  counted as integral only if they were integral too, and the search
  branched on them.
  - On ns1830653 they are 134 of the 592 columns after presolve. The model
    has 171 continuous columns and 1,458 binaries, and presolve leaves
    "458 binary, 134 implied int., 0 continuous".
  - No node LP ever gave an incumbent, and they took 11 of 32 branchings
    in a traced plunge, most at x = 0.5. HiGHS improves its incumbent there
    mostly from integral node LPs.
  - `MIP.implied` now marks them, carried through `with_rows`, coefficient
    tightening, sub-searches and restarts. They count as integral only for
    the objective step (`objective_step`): ns1830653's are all costed, and
    its objective still has step 1, as HiGHS reports.
- Plunges and depth-first dives now follow the LP point's rounding
  direction. `branch` compared each child's `down` flag with `prefer_up`,
  which put the wrong child last, so the plunge child (and, in a dive, the
  child popped next) was the one against the LP value. On ns1830653
  plunges went up at x = 0.08 and their bounds climbed by 500–1,000 a
  step, from 12,598 to 31,039. In 891 nodes no plunge reached an integral
  leaf, and the root incumbent (35,622) never improved. HiGHS improves
  its incumbent mostly from integral node LPs at the bottom of its
  plunges, and reaches the optimum (20,622) in 236 s.
- The plunge child, and in a dive the child taken next, is now the one
  with the higher `(1 + inferences) / (1 + cost / unit)`:
  - `inferences` is the number of bounds the child's propagation tightened
    beyond the split (now kept as `_Node.inferences`);
  - `cost` is its estimated bound gain: its bound (from the parent's
    duals or strong branching), plus its pseudocost gain once both
    directions of the variable have observations;
  - `unit` is the mean pseudocost gain observed so far.

  Equal scores go to the lower estimated bound, then to the rounding
  direction (within 1e-9, relative). `jm.solve(plunge_child=...)` picks
  the rule: `"score"` (the default), `"estimate"` (the lower estimated
  bound, then rounding) or `"rounding"` (the LP point's direction only).
  It was chosen while implicit integers still loaded as integers, so it
  has not been compared on ns1830653. A child that fixes many variables
  reaches an integral point in fewer steps. On ns1830653, plunges that
  kept to the cheaper child still had about 100 fractional integers at
  depth 27. Most of its branchings are at x = 0.5, where rounding says
  nothing.

### Changed

- A node whose LP point is integral but does not settle it (the point
  fails the feasibility check, or its LP did not converge) is split on a
  free integer instead of being closed on its LP bound. Preferred is the
  integer in a broken row of the MIP as given with the largest
  `|coefficient|`, else the one with the largest `|reduced cost|`. The
  split is half a unit from the point's value, and no pseudocost is
  learned from it. Closing it put its bound into the global bound for
  good: in a bab5 run, a node closed at −106,162 against an incumbent of
  −106,018 held the gap at 0.14% at least, and its other integer points
  were never searched. Only a node with every integer fixed is still
  closed (`"unresolved_nodes"`). New result: `"integral_splits"`.
- With batched node LPs, `"nodes"` (and the log's node count, and
  `max_nodes`) counts nodes finished. A node LP sent back unconverged
  for more epochs (`batch_requeues`) used to count once per slice, so
  with `batch_epochs=10` and `node_max_epochs=200` one node counted up
  to 20 times. On bab5 with `batch_size=64` the log showed 115 nodes
  with one open at depth 1: a few shallow nodes, each solved slice by
  slice.
- Faster root cut rounds, with the same cuts. On bab5 (20 rounds,
  `max_cuts_per_round=500`), a profile put the rounds at about 370 s:
  Gomory cuts 105 s, HiGHS's re-solves 89 s, `select_cuts` 75 s, the
  separators 74 s (cover cuts 60 s), and rebuilding the searched MIP's
  device LP every round 44 s.
  - `select_cuts` takes a block's cosines from one Gram matrix
    (`_gram`), computed with dense BLAS products over column chunks when
    the cuts are dense (bab5's Gomory cuts use about half its 20,508
    columns). Before, it ran one sparse product per chosen cut, each over
    all of the block's nonzeros. On the 20 rounds' recorded candidates it
    took 7.4 s instead of 40.8 s, and chose the same cuts.
  - `cover_cuts` returns at once when the binaries are integral at the LP
    point and fit the knapsack: no valid cover inequality can cut such a
    point off.
  - While HiGHS holds the rounds' LP (`simplex_cut_rounds`), each round
    keeps its cut rows on the host only (`MIP.rows_with`) for the bound
    and the zero-half cuts. The searched MIP and its device LP are built
    once, when the rounds end or the first-order solver takes over.
  - In the simplex rounds, Gomory cuts read their tableau rows from
    HiGHS's own factors of the basis (`gomory_cuts(highs=...)`,
    `getBasisInverseRow`) instead of a new sparse LU and dense solves.
    The cuts match the LU's to 3e-10. On mzzv11, a round's Gomory cuts
    took 0.9 s instead of 2.9 s once the LP held 1.8 million nonzeros,
    and on ns1830653 about half as long.
- With `node_lp_solver="highs"` (the default), `jm.solve` repairs a rounded
  point that breaks rows (integers fixed, the LP in the continuous
  variables) exactly with HiGHS's dual simplex, in the node LPs' instance,
  within `round_repair_seconds`. It was a PDHG solve of
  `repair_max_epochs` (20) epochs, far from converged on large LPs, after
  which a rounding an exact LP would have repaired was marked failed for
  good; `node_lp_solver="jaddle"` keeps it. Strong branching now starts
  its children from the node's own basis, whatever dives and repairs left
  in the instance.
- `jm.solve` solves the root LP to convergence with one
  `jaddle_linear.solve` call (within `root_max_epochs`; `None`, the
  default: no limit) instead of `detect_infeasibility`, whose chunks of
  doubling length with a check between them cost a feasible root about
  2.5×: on bab5's root 0.8 s and 24 epochs against 0.3 s and 17, on
  neos-4306827-ravan's 0.7 s and 20 against 0.3 s and 16 (compiled). The
  infeasibility check now runs only on a root LP that stops unconverged
  with time left, so without an epoch limit an infeasible or unbounded
  root LP runs until `max_seconds`.
- `jm.solve`'s HiGHS incumbent search (`highs_incumbent_seconds`) with
  `parallel_heuristics` starts as soon as the root LP is solved, beside
  the cut rounds, instead of after the root heuristics fail; while there
  is no incumbent it runs again with twice the budget each time it finds
  nothing (10 s, 20 s, 40 s, ...; `result["highs_incumbent_calls"]`).
  On neos-4306827-ravan HiGHS needs 16 s for its first point, so the one
  10 s call found nothing. When its point arrives first, the root
  heuristics that run only without an incumbent are skipped: on
  ns1830653 it gives 35,622 at 16.8 s, during the cut rounds.
- Faster root cut rounds; the cuts are unchanged. bab5's cut rounds take
  84 s instead of 141 s:
  - batched c-MIR and flow cover searches size each call to at most
    256 × 1,024 padded entries (`_SEPARATION_ENTRIES`), so one wide base
    no longer pads a whole 256-row chunk. bab5 had one 5,538-term base in
    a 256 × 16,384 call, nine times all the others' entries; c-MIR time
    went from 42 s to 5 s.
  - `jm.gomory_cuts` computes its tableau rows in blocks of 64 (one solve
    with the factors and one product with Aᵀ a block, the rest
    vectorised), from 35 s to 15 s.
  - `jm.select_cuts` takes its cosines a block of candidates at a time.
- `jm.solve`'s `max_cuts_per_round` default is 100 (was 10). With 10 a
  round, the strongest cover cuts took every slot on bab5 and no Gomory cut
  got in. With clique cuts, 100 takes bab5's root bound from −113,254 to
  −111,358 (357 s instead of 47 s) and ns1830653's from 10,621 to 11,622,
  HiGHS's root bound (22 s instead of 12 s). The tests already pinned 100.

### Fixed

- `jc.solve(verbose=True)` no longer fails with an index error when a
  restart is logged with `diag_adapt=True` but the adaptive step off.
- `js.solve`, and `jc.solve` on `JaddleCPStochastic.full_cp()`, pass the
  data to their compiled programs as an argument. Before, every program
  embedded it as a constant, so a 1.6 GB dataset (200,000 × 1,000) built a
  3.2 GB program, overran the compilation cache's 2 GiB limit and the GPU's
  memory, and took 79 s to solve instead of under 7 s. Terms given the same
  array now also share one device copy of it, rather than one each.

### Examples

- `examples/stochastic/fair_logistic_regression.py` now has 1,000 features
  (990 skills, 10 weak proxies) for 200,000 applicants, and solves the
  unconstrained model as well as the fair one. On a laptop GPU each
  stochastic solve takes 2–7 s (compilation included) and `jc.solve` 22 s.
- `examples/stochastic/risk_constrained_kelly.py`: risk-constrained Kelly
  betting over 2,000,000 scenarios with rare crashes, from minibatches. The
  drawdown limit holds out of sample, where fits to 1,000 scenarios break
  it. On a CPU the stochastic solves take 2.5 s and 6.5 s and `jc.solve`
  40 s.

## 0.3.0

### Added

- Round and repair in `jm.solve` (`round_repair=True`, `round_repair_every`,
  `round_repair_seconds`), marked `U`. An LP point's integers are rounded
  up, to nearest and down (`jm.integer_roundings`, snapping near-integral
  PDHG values by `100·lp_tolerance` first), and HiGHS solves the rest with
  them fixed (`jm.highs_round_repair`: presolve leaves an exact LP over the
  continuous variables), keeping only improving points. It runs on the
  root LP (its exact vertex first, when the root is crossed over), on the
  LP points of the root's strong children of its branching variable, and
  every `round_repair_every` nodes (default `heuristic_every`). In the tree
  it runs on the background HiGHS lane with `parallel_heuristics`, else
  within `heuristic_effort`. On buildingenergy's presolved model, rounding
  up and repairing gives 33,325 from the root vertex in 0.6 s, and 33,285.6
  from the sizing variable's up child (optimum 33,283.85). RENS gave 33,747
  after 190 s. Rounding to nearest, as `try_incumbent` does, gives 33,688
  there, and its PDHG repair (`repair_max_epochs=20`) is far from converged
  on a 145,000-variable LP.
- Mixed-integer LPs through CVXPY: `JaddleSolver` is now `MIP_CAPABLE`, and
  a problem with `integer=True` or `boolean=True` variables is solved by
  `jm.solve` (options of `problem.solve` pass through; `tol` sets
  `lp_tolerance`). Boolean columns get their 0/1 bounds, budget stops give
  `USER_LIMIT` with the incumbent, and `extra_stats` holds the bound and
  gap. No duals. Mixed-integer QPs, and mixed-integer problems on
  `jcv.solve(qp_route=False)`, raise `SolverError`. New example
  `examples/mip/cvxpy_facility_location.py`.
- Root strong-branching sweep in `jm.solve` (`root_strong_candidates`,
  `"auto"` = `strong_candidates`; `root_strong_epochs`, default
  `strong_epochs`). Before the root branches, the fractional integers with
  the largest `jm.branching_impact(mip) · f(1 − f)` (column length times
  `1 + |cⱼ|/c̄`) get both child LPs in one vmapped call. That seeds their
  pseudocosts, so `branching="pseudocost"` no longer starts most-fractional
  among variables that cannot move the bound. Reliability branching also
  ranks its unreliable candidates by score times impact. On buildingenergy,
  branching on the one fractional sizing variable (35,000 rows) takes the
  LP bound from 33,246 to the optimum, 33,283.8; the 8,000 fractional
  hourly integers (3 rows) each gain about 0. The sweep (8 candidates, 50
  epochs) ranks it first by a wide margin, in 28 s on a laptop GPU. Child
  bounds that are not rigorous only score; they neither raise nor close.
- `root_crossover` in `jm.solve` (`"auto"`: when the PDHG root bound is not
  rigorous). The root LP is crossed over to an exact vertex (reusing RENS's
  crossover), and its duals bound the root and drive root reduced-cost
  fixing. Root fixing now also runs once before the root branches, with
  the root heuristics' incumbent. Reduced costs at the vertex that point at
  an infinite bound and are round-off (at most `1e-9·(1 + max|c|)`) are
  zeroed. On buildingenergy the crossover takes 2 s and gives a rigorous
  33,246.2 (HiGHS's LP optimum), where the PDHG bound dropped 8,209
  reduced costs and gave 33,247.9, above the LP optimum. Reduced-cost
  fixing had been skipped for want of a rigorous bound; with exact duals,
  the RENS incumbent fixes the two dearest sizing variables to 0. Node
  crossovers (`node_crossover`) clean their vertices the same way, and
  their rigorous bound now wins over a PDHG bound that is not.

- `parallel_heuristics=True` in `jm.solve` runs the CPU heuristics on
  background threads while the main thread keeps the device busy with
  node LPs. HiGHS's RENS, RINS and incumbent search share one thread, one
  job at a time; the numba improving jump has another and runs in 5 s
  slices, each from the incumbent of its time. The search checks finished
  jobs' points between nodes. Background RINS takes no time from the
  tree, so `heuristic_effort` no longer limits it. Runs then depend on
  timing, so it is off by default. On academictimetablesmall with the
  example's settings (120 s, `batch_size=1`), the incumbent reached 1,071
  against 1,683–2,502 serially, at the same node rate. The result reports
  the jobs under `"parallel_heuristics"`. `highs_submip` and
  `highs_incumbent` take a prebuilt HiGHS `model` and a `stop` event, and
  `feasibility_jump` takes `stop` (numba).
- Incremental feasibility jump (`jaddle._jump_numba`, `pip install
  jaddle[fast]` for numba). It moves one variable a step and updates only
  that variable's rows: their activities, the in-place scores of their
  two-valued variables, and the jump values of the rest. HiGHS does the
  same. A step costs about the nonzeros of those rows, against a pass and
  a sort over every nonzero for the device jump. On academictimetablesmall
  (230,000 nonzeros) a move takes 1.5–3 µs, and 6 of 8 seeds find a
  feasible point in 0.1–0.8 s from zero. The device jump took 12 s to over
  90 s on a laptop GPU, and 85 ms a step on the CPU. `jm.feasibility_jump`
  and `jm.solve` take `backend`/`jump_backend` (`"auto"`, `"numba"`,
  `"jax"`); `"auto"` uses numba when it is installed. Under numba `steps`
  counts single moves, so the pump's and the tree's short jumps
  (`pump_jump_iterations`, `tree_jump_iterations`, now `None` by default)
  get `5·(n + m)` moves, at least 10,000. With JAX they keep 3000 steps.
- `jm.feasibility_jump(improve=True)` (numba): after each point it finds,
  the objective row `c·x <= cutoff` drops below that point by
  `improve_step` (the objective step, else 1e-4 relative), and the search
  goes on with its weights. It returns the best point. `jm.solve` gives the
  root heuristic time left after an incumbent to such a jump from it, and
  the tree heuristics' jumps improve the same way. On
  academictimetablesmall, with the example's settings (10 s of
  heuristics), the first incumbent went from 5,611 (from HiGHS, at 28 s)
  to 3,595 from Jaddle's jump at 11 s, then 1,509 at 14.5 s. Before, the
  run was still at 5,153 after 180 s; HiGHS alone has 4,562 after 120 s.

- `node_precision` (`jm.solve`, or `node_lp_options={"precision": ...}`):
  node LPs run at their own precision profile, apart from `precision`.
  Serial and batched node solves and strong branching's child LPs use it.
  The root LP, cut rounds, heuristic LPs and infeasibility checks stay at
  `precision`. Warm starts are cast between the two. For example,
  `precision="float64", node_precision="float32"` gives an accurate root
  and cheaper nodes.
- Node crossover (`jm.solve(node_crossover=True)`, opt-in). Every
  `node_crossover_every` (1) nodes the LP bound does not close, HiGHS
  crosses the node LP's PDHG point over to an optimal vertex
  (`jl.crossover`, at most `node_crossover_seconds` (2) a call). The vertex
  is used for the integrality test, so an integral vertex becomes a `T`
  incumbent as in HiGHS's "evaluate node", and for branching and the
  heuristics. Its duals' bound replaces the node's when higher. Results
  report `node_crossovers`, `node_crossover_vertices`,
  `node_crossover_integral` and `node_crossover_seconds`.
- GPU tree heuristics (`jm.solve(tree_heuristics=True)`, opt-in). Once there
  is an incumbent, every `tree_heuristic_every` (20) nodes, the node's LP
  point is bulk rounded and, if that finds nothing better, the feasibility
  jump runs from it (`tree_jump_iterations`, 3000 steps), within
  `tree_heuristic_share` (5%) of the elapsed time; each call stops once it
  has used the share left (at least 1 s). Before, the GPU
  heuristics ran only at the root, until the first incumbent. Results
  report `tree_heuristic_calls`, `tree_heuristic_finds` and
  `tree_heuristic_seconds`.
- `jm.feasibility_jump(cutoff=...)`: adds the row `c·x <= cutoff`, so the
  jump looks only for points better than an incumbent. The objective row is
  left out of the jump's one-move-per-row rule, so moves still batch.
  `data=` reuses the device arrays between calls.
- `node_lp_options` (`jm.solve`): `jl.solve` options for node LPs only, for
  example `{"iterations_per_epoch": 64}`. They sit over `lp_options`, which
  still applies to every LP, and over `node_lp_tolerance`. They reach serial
  and batched node LPs and strong branching's child LPs. A key that batched
  LPs (`jl.make_solver`) cannot take raises instead of being dropped. Fix:
  batched search (`batch_size > 1`) ignored `node_lp_tolerance`, in node LPs
  and strong branching alike; it now uses it.
- Faster node LPs in batched search.
  - **Continuous batching** (`batch_epochs=10`). A batched node LP runs at
    most this many epochs per call. An unfinished one goes back to the
    queue, warm-started, until `node_max_epochs` in all, so one slow member
    no longer holds its batch: the share of member-epochs lost to lockstep
    fell from 52% to about 30% on ns1830653.
  - **Root-box cutoff.** No feasible point has `c·x` above the objective's
    maximum over the root box, so the cutoff starts just above it (when
    finite). Infeasible node LPs then stop on their bound before there is
    an incumbent, where one used to run all 200 epochs and hold its batch
    for 1.8 s. On ns1830653, 1,047 nodes stopped this way before the first
    incumbent, and batches took 112 ms on average.
  - Device time per node LP after an incumbent, `batch_size=16` before →
    `batch_size=32` now: ~12 → ~6 ms. A batch of 32 costs little more than
    one of 16 on a laptop GPU; 64 is no better per node.
    `examples/mip/miplib.py` now batches 32. `batch_retries` is gone,
    replaced by the epoch budget.
- Early stopping for batched node LPs. `jl.make_solver(bound_stop=True)`'s
  solve function takes a `bound_target`: a member stops (`stop_code` 4, not
  converged) once its Lagrangian bound reaches the target, as
  `solve(bound_target=...)` does. Under `vmap` the target can be one for the
  batch or one per member. Batched tree search (`batch_size > 1`) passes
  the node target, so `node_bound_stop` now works there too, and so does
  strong branching's batch. Batched search also no longer builds a node LP
  for every member (only for the rare `detect`), and puts unconverged
  members back for another batch before a serial infeasibility solve. ns1830653, `batch_size=16`, GPU, 150 s: 8.1 → 27.9
  nodes/s, with the device time per node down from ~92 ms to ~12 ms; the
  host is now the larger share.
- Device-side children for batched search (`children_backend="auto"`: on
  above 10,000 nonzeros; `jm._ChildKernels`). The members of a batch are
  split together. One jitted call builds both children of every member,
  propagates them (`_propagate_jax`) and bounds them by the member's duals.
  `branch()` then does only the conflict pool, pruning and the queue.
  ns1830653, as above: 27.9 → 32.9 nodes/s, branching 8.4 → 2.7 ms a node,
  bound after 150 s 13,619 → 14,463. Batching the tree's roundings into one
  bulk rounding (`batch_roundings=True`) is opt-in: it took ~1 s a batch,
  against ~55 ms a call for the classic fix-and-propagate.
  `bulk_fix_and_propagate` takes `boxes=` (one per start), `pad_to=` and
  `propagator=`.
- Primal heuristics after cuOpt's GPU primal heuristics (Çördük et al.,
  2025).
  - **Root probing** (`probing=True`, `probing_seconds`,
    `probing_max_variables`, `probing_backend`; `jm.probe`,
    `jm.ProbingCache`). Each integer variable's two sides are propagated.
    An empty side fixes the variable, and otherwise all bounds tighten to
    the hull of the two sides. Conflict-prone variables go first. Above
    10,000 nonzeros, chunks of variables run in one vmapped JAX call.
    On misc07 it fixes 21 binaries, on p0201 6, and on fixnet6 it tightens
    483 bounds, each in under a second; it never changed the optimum on the
    19 models checked. It stops after `probing_max_useless=1000`
    variables in a row that change nothing. On fiball, probing all
    32,399 variables took 11 s for one tightened bound; it now stops
    after 1,088 variables, in 0.9 s, with the same result.
  - **Bulk fix-and-propagate** (`bulk_rounding=True`, `bulk_starts=4`;
    `jm.bulk_fix_and_propagate`). It runs after the classic one at the root
    and in the pump, marked `"B"`. Bulks are √(free) variables in
    implied-slack order. Two random roundings per bulk are screened by the
    probing cache and propagated together. Several starts run side by side,
    alternating two orders, because each suits different models: it finds
    points on lseu where the classic one finds none, and the classic one
    finds points on some models where it finds none.
  - **Fused pump** (`pump_jump_iterations=3000`, `pump_jump_share=0.2`).
    Each rounding the pump rejects gets a short feasibility jump, and a
    cycling pump continues from the jump's near miss. New result keys:
    `pump_jumps`, `pump_jump_finds`, `probing`.
  - **`repair_share=0.2`.** `repair` of a near-feasible jump now gets at
    most this share of the heuristic time left, instead of all of it. On
    ns1830653 it used to take the whole budget, so the jump from the bounds
    nearest zero never ran. Now that jump finds an incumbent (136,622) at
    29 s of a 90 s heuristic budget, where before none was found.
  - `benchmarks/benchmark_primal.py` measures root-heuristic coverage and
    primal gap on MIPLIB 2017. It has not been run on the full set yet.
- Reliability branching, opt-in (`branching="reliability"`,
  `reliability_threshold=4`, `strong_candidates=8`, `strong_epochs=50`,
  `strong_screen=32`, `strong_max_depth`, `strong_backend="jax"`).
  Variables whose pseudocosts have fewer than `reliability_threshold`
  observations per direction are strong-branched. Strong branching runs as
  one device pipeline (fixed shapes, so it compiles once), dispatched
  before the node's heuristics and read back after them:
  1. Both children of up to `strong_screen` candidates are propagated in
     JAX (`_propagate_jax`, matching `BoundPropagator`) and bounded by the
     node's duals.
  2. The best unreliable candidates get child LPs in one vmapped call.
  3. The children's Lagrangian bounds are computed on the device.

  Closed children fix their variable to the other side, and the node's
  bound rises to the best candidate's lesser child bound. Gains (the
  strong-branching scores and the pseudocost updates) are measured on
  unrounded bounds. `strong_backend="numpy"` runs on the host, and tree
  cuts use it. New result keys: `strong_branch_nodes`, `strong_branch_lps`,
  `strong_branch_seconds`, `strong_fixings`. `solve_batch` now reads a
  batch back in one transfer.

  60 s, `gomory_cuts=True`, CPU, pseudocost → reliability:

  | Instance | Nodes | Time / result |
  |---|---|---|
  | p0033 | 42 → 8 | 2.5 → 2.7 s |
  | flugpl | 197 → 114 | 3.0 → 2.9 s |
  | stein27 | 4354 → 4141 | 29.9 → 31.0 s |
  | bell5 | both at the time limit | gap 0.088% → 0.078% |
  | lseu | 397 → 668 | 13.0 → 46.9 s |
  | p0201 | 521 → 66 | optimal in 31.5 s → 2% gap at the time limit; strong branching took 45 s for 732 child LPs |

  Not the default: on instances with many root cuts, strong branching costs
  too much per node.
- `jm.solve` conflict analysis (`conflicts=True`, `conflict_pool_size`,
  `conflict_max_nnz`; `jm.ConflictPool`). Nodes and children closed by
  their Lagrangian bound leave dual proofs (`constant + r·x <= cutoff`,
  valid at any duals), infeasible node LPs leave Farkas proofs from
  `detect`'s dual ray, and children that propagation proves infeasible leave
  no-goods over their binary fixings, minimised by a deletion filter.
  Proofs are sparsified against the root box, and the pool is propagated at
  every node before its LP; a node it shows empty is closed without its LP
  (`conflict_pruned`). At Jaddle's node rates the effect is small so far:
  ns1830653 (3 min) 111 nodes against 97, bound 11,660 against 11,644.
- HiGHS-style primal heuristics by default. `rins_solver="highs"` (was
  `"auto"`) now covers RENS too: `jm.highs_submip` has HiGHS presolve and
  search the neighbourhood, with the cutoff as its `objective_bound` (no
  dense cutoff row) and the incumbent as its start. RENS runs at the root
  for every MIP size (`rens_seconds=None`: 10% of `max_seconds`, up to
  30s) from an exact vertex (the cut loop's crossover); RINS is on
  (`rins_seconds=None`: 5s a call), with an adaptive node limit and a
  `heuristic_effort=0.1` time budget. Why: a HiGHS ablation showed its root
  RENS is its most useful heuristic (misc07 20s → 45s without it; ns1830653
  25,808 → 30,808 after 60s). Jaddle (60s, `gomory_cuts=True`): vpm2 reaches
  its optimum 13.75 at 12s (14.5 before), pp08a is within 1% at 4.6s (30s
  before) and optimal at 5.1s; misc07 is unchanged (its RENS neighbourhood
  is infeasible). New result keys `submip_calls`, `submip_seconds`.
- Early stopping of node LPs (`node_bound_stop=True`; `jl.solve(bound_target=...)`).
  `jl.solve` can stop (`stop_reason="bound_target"`) once the Lagrangian
  bound of the reported point, `−bᵀy + Σ min(rⱼlⱼ, rⱼuⱼ)`, reaches a target;
  it is checked on the device every epoch and returned as
  `"lagrangian_bound"`. Branch and bound passes the cutoff (rounded as its
  pruning test rounds), so a node stops as soon as it would be pruned
  anyway. An infeasible LP's duals diverge along a Farkas ray, so its bound
  passes any finite target fast: 1 epoch against 500 unconverged on a test
  LP. ns1830653 (3 minutes): 140 → 385 nodes, 21.9 → 10.2 LP epochs a
  node, bound 11,678 → 11,976. No effect while the incumbent is far from the
  bound (misc07: no node reached the cutoff).
- Batched cut separation in JAX (`separation_backend="jax"`, the default;
  `jm.batched_cmir_cuts`, `jm.batched_flow_cover_cuts`). `CutSeparator`
  first builds every base inequality of a round (tight rows and their
  aggregation paths), then runs the c-MIR search (plain and over variable
  bounds) and the flow cover search on all of them in jitted float64
  kernels, in chunks of 256 rows padded to widths 16, 64, 256, … Same cuts
  as the per-row numpy searches (`"numpy"`) up to rounding in near-ties;
  rounds with fewer than 32 base inequalities use numpy. Per round, warm:
  fixnet6 277 → 113 ms, vpm2 4× faster over a root (5.8 → 1.35 s).
  `select_cuts` is vectorised too (one sparse product per chosen cut), with
  the same choices.
- Flow cover cuts (`jm.flow_cover_cut`, family `"flowcover"`, `flow_covers=True`):
  generalised flow cover inequalities from the single-node flow set of each
  row and aggregate with a continuous variable, over its variable upper
  bounds (`y <= u·z`). fixnet6 (`gomory_cuts=True`, 120 s): solved in 71 s
  against a 4.3% gap at the time limit without them.

- Per-call `device` and `precision`. `jl.solve`, `jc.solve` and
  `jc.quadratic_program` take `device=` (`"cpu"`, `"gpu"`, `"tpu"` or a
  `jax.Device`; `None` keeps JAX's default). They also take `precision=`
  (`"float64"`, `"float32"` or `"float16"`; `None` keeps the global
  profile), and so does `jm.solve`. The problem and warm start are converted
  to the requested precision. `jaddle_optimisers.local_settings(precision,
  device)` applies both to a whole block of code. `jm.solve`'s `device` also
  accepts a `jax.Device`, and `lp_options` can no longer set `device` or
  `precision`.
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

- `jm.solve`'s defaults are now the settings of `examples/mip/miplib.py`,
  tuned for large MIPs on a GPU: `precision="float64"`,
  `node_precision="float32"`, `lp_tolerance=1e-4`,
  `node_lp_options={"iterations_per_epoch": 64}` (when `None`, unless
  `lp_options` sets it), `batch_size=32`, `gomory_cuts=True`,
  `root_crossover=False`, `cut_round_epochs=10`, `max_cuts_per_round=10`,
  `heuristic_every=5`, `heuristic_seconds=10`, `fix_and_propagate=False`,
  `pump_jump_iterations=3000`, `highs_incumbent_seconds=10`, `rins_every=5`,
  `heuristic_effort=0.3`, `parallel_heuristics=True`, `dive_every=50`,
  `tree_heuristics=True` and `tree_heuristic_every=10`. Runs are no longer
  repeatable by default (pass `parallel_heuristics=False`), batching turns
  tree cuts off (pass `batch_size=1` with `tree_cuts=True`), and the first
  solve compiles the batched float32 node solver. `node_lp_tolerance` stays
  `None` (`lp_tolerance`): at 1e-3 with float32 node LPs, small MIPs with
  general integers ended `"unresolved"`. `miplib.py` and
  `benchmarks/benchmark_small.py` now rely on the defaults; the tests pin
  the previous ones (`tests/test_mip.py`'s `PINNED`).
- `jm.solve` root Gomory cuts. `gomory_cuts` takes `max_nnz` (`"auto"`, the
  old cap of `max(n // 2, min(n, 100))` nonzeros, or `None` for no cap), and
  the root cut loop uses `None`: the cap rejected 197 of 198 GMI cuts at the
  root of ns1830653. With Gomory cuts on, each round now drops the cuts
  whose rows are basic (slack) in its crossover basis, instead of keeping
  them for two rounds, and the default is 40 rounds whatever the MIP's size.
  On ns1830653 (`gomory_cuts=True`) the root bound goes from 7,630 to
  11,622 (HiGHS's root bound) in 13 s, with 10 cuts left in the tree LP.
- c-MIR cuts use variable bounds. `jm.variable_bounds(mip)` reads the
  two-term rows that bound a continuous variable by an integer one (such as
  `y <= u·z`), and `cmir_cut(..., variable_bounds=...)` may shift a
  continuous variable to such a bound when it is nearer the LP point than
  its simple bounds, keeping the better of that cut and the plain one.
  `CutSeparator` passes them in, so root and tree cuts both use them. On
  vpm1 (`gomory_cuts=True`) the root bound goes from 18.9 to 20, the
  optimum, and the search from 173 nodes to 31. Root bounds: vpm2 12.50 →
  13.00, pp08a 7,014 → 7,182, fixnet6 3,594 → 3,712.
- `jm.highs_incumbent` (`highs_incumbent_seconds`) returns HiGHS's best
  solution within its time limit instead of its first. On ns1830653 the
  first was 133,808, against 36,622 a few seconds later.
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

- A warm start's step state (`initial_opt_state`) from a solve in another
  update mode, or from an LP with other rows, crashed `solve`. That happened
  in the MIP search with `update_mode="halpern"` in `node_lp_options` only
  ("not enough values to unpack"), or in `lp_options` with cut rounds
  (shape mismatch). Halpern now anchors at the warm start when the state
  has no usable anchor, and the other modes ignore the anchor.
- With `node_precision` below float64 and batched nodes, repairing a node's
  rounding by an LP could fail with "lax.add requires arguments to have the
  same dtypes, got int32, int64". JAX reused a 32-bit copy of the LP's sort
  permutation from the node LPs in a float64 context. The permutations are
  now int32 (when the LP has under 2^31 nonzeros), which reads the same at
  every precision.
- Root bulk rounding (`bulk_rounding`) ran only after the classic
  fix-and-propagate, so `fix_and_propagate=False` silently turned it off,
  at the root and in the feasibility pump. It now runs on its own.
- One slow root heuristic could take the whole heuristic budget. On
  academictimetablesmall, the device jump from the LP point (and, once it
  ran again, bulk rounding) did, though a jump from zero finds a point in
  under a second. Bulk rounding, the two jump starts and the pump now each
  get an equal share of the time left when they start.

- `jm.solve`'s progress log no longer shows the bound falling. Nodes held
  outside the queue were left out of the global bound, so it read too high:
  the root while the root heuristics ran (an incumbent found there showed
  bound = incumbent and a 0% gap), and a batch's members not yet evaluated
  (`batch_size > 1`). Only the log and the incumbent history were affected;
  the gap check runs between batches.
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
