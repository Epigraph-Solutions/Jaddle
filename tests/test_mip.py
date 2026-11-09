"""Tests for the branch-and-bound MIP solver, jaddle.jaddle_mip."""

import time
import types

import highspy
import jax.numpy as jnp
import re

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

import jaddle.jaddle_linear as jl
import jaddle.jaddle_mip as jm
from jaddle import highs_helpers as hh

# solve()'s defaults are tuned for large MIPs on a GPU (benchmarks/benchmark_mip.py):
# loose LP tolerances, float32 node LPs, batched nodes and heuristics on
# background threads, so runs depend on timing. The tests run with the
# settings those replaced: exact, serial and repeatable, so they can check
# objectives tightly and which heuristic finds what. A test's own options
# override them; test_default_settings_solve covers the defaults.
PINNED = dict(
    gomory_cuts="auto",
    precision=None,
    node_precision=None,
    root_crossover="auto",
    cut_round_epochs=20,
    lp_tolerance=1e-6,
    node_lp_options={},
    max_cuts_per_round=100,
    heuristic_every=10,
    heuristic_seconds=None,
    fix_and_propagate=True,
    pump_jump_iterations=None,
    # Small test models are often symmetric on purpose (faces, cliques);
    # the symmetry tests turn it on.
    symmetry=False,
    # On small models a few fixings pass the restart shares, and each
    # restart redoes the root; the restart tests turn them on.
    restarts=0,
    rins_every=50,
    heuristic_effort=0.1,
    parallel_heuristics=False,
    dive_every=0,
    tree_heuristics=False,
    tree_heuristic_every=20,
    batch_size=1,
    # solve()'s defaults before they became the map06 setup (cuts, probing
    # and most heuristics off, exact nodes on): the tests were written
    # against these, and each feature's tests turn it on or off themselves.
    cuts=True,
    cut_rounds=None,
    cut_age=10,
    max_cut_rows=None,
    gomory_candidates=None,
    max_aggregation=None,
    max_tree_cuts_per_round=10,
    tree_cut_rows=None,
    inout_separation=None,
    root_redcost_propagation=False,
    global_redcost_fixing=False,
    cut_pool_propagation=False,
    objective_propagation=False,
    branch_implied=False,
    probing=True,
    plunge_depth=8,
    plunge_child="score",
    batch_epochs=2,
    node_lp_tolerance=None,
    root_strong_candidates="auto",
    root_strong_epochs=None,
    round_repair=True,
    feasibility_pump=True,
    bulk_rounding=True,
    one_opt=True,
    rins_seconds=None,
    rens_seconds=None,
    near_integral_tolerance=1e-2,
    parallel_rounding=True,
    exact_nodes=0,
    exact_overflow_share=0,
    submip_solver="jaddle",
    crossover_seconds=0,
    # The trivial heuristic would find many test models' first incumbent
    # before the heuristics their tests check; its own tests turn it on.
    trivial_heuristic=False,
)


def solve(model, **options):
    """``jm.solve`` with ``PINNED`` under ``options``."""
    return jm.solve(model, **{**PINNED, **options})


def no_rows(n):
    return sp.csc_matrix((0, n)), np.zeros(0)


def knapsack(seed=0, n=15):
    """A pure binary knapsack: maximise value within a third of the weight."""
    rng = np.random.default_rng(seed)
    w = rng.integers(5, 30, n).astype(float)
    v = w + rng.integers(0, 10, n)
    A_eq, b_eq = no_rows(n)
    lp = jl.LP(
        c=-v,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ineq=sp.csc_matrix(w[None]),
        b_ineq=np.array([w.sum() / 3]),
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
    )
    return jm.MIP(lp, np.ones(n, bool))


def general_integer(seed=1, n=8, m_eq=2, m_ineq=6):
    """General integers in [0, 8] with equality and inequality rows."""
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 5, n).astype(float)
    A_eq = rng.integers(-3, 4, (m_eq, n)).astype(float)
    A_ineq = rng.integers(-3, 4, (m_ineq, n)).astype(float)
    lp = jl.LP(
        c=rng.standard_normal(n),
        A_eq=sp.csc_matrix(A_eq),
        b_eq=A_eq @ x0,
        A_ineq=sp.csc_matrix(A_ineq),
        b_ineq=A_ineq @ x0 + rng.integers(0, 4, m_ineq),
        lower_bounds=np.zeros(n),
        upper_bounds=np.full(n, 8.0),
    )
    return jm.MIP(lp, np.ones(n, bool))


def facility_location(seed=2, n_fac=4, n_cust=8):
    """Capacitated facility location: binary openings, continuous flows with
    no upper bound."""
    rng = np.random.default_rng(seed)
    fixed = rng.uniform(20, 40, n_fac)
    cap = rng.uniform(20, 40, n_fac)
    demand = rng.uniform(3, 10, n_cust)
    cost = rng.uniform(1, 10, (n_fac, n_cust))
    n = n_fac + n_fac * n_cust
    flow = lambda f, k: n_fac + f * n_cust + k  # noqa: E731
    A_eq = sp.lil_matrix((n_cust, n))
    A_ineq = sp.lil_matrix((n_fac, n))
    for f in range(n_fac):
        A_ineq[f, f] = -cap[f]
        for k in range(n_cust):
            A_eq[k, flow(f, k)] = 1.0
            A_ineq[f, flow(f, k)] = 1.0
    lp = jl.LP(
        c=np.concatenate([fixed, cost.ravel()]),
        A_eq=A_eq.tocsc(),
        b_eq=demand,
        A_ineq=A_ineq.tocsc(),
        b_ineq=np.zeros(n_fac),
        lower_bounds=np.zeros(n),
        upper_bounds=np.concatenate([np.ones(n_fac), np.full(n - n_fac, np.inf)]),
    )
    return jm.MIP(lp, np.arange(n) < n_fac)


def reference(mip):
    """The optimal value of ``mip`` in its own sense, from SciPy's HiGHS."""
    result = milp(
        mip.c,
        constraints=LinearConstraint(mip.A.toarray(), mip.row_lower, mip.row_upper),
        integrality=mip.integrality.astype(int),
        bounds=Bounds(mip.lower, mip.upper),
    )
    assert result.success
    return mip.sign * (result.fun + mip.offset)


def assert_solved(result, mip, ref, rel=1e-4, rigorous=True):
    assert result["status"] == "optimal" and result["converged"]
    assert abs(result["objective"] - ref) <= rel * (1 + abs(ref))
    assert mip.is_feasible(result["x"])
    # The bound is on the right side of the optimum (up to LP accuracy).
    slack = 1e-6 * (1 + abs(ref))
    if mip.maximise:
        assert result["bound"] >= ref - slack
    else:
        assert result["bound"] <= ref + slack
    if rigorous:
        assert result["bound_rigorous"]


def write_mps(mip, path):
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.passModel(jm.to_highs(mip))
    highs.writeModel(str(path))


@pytest.mark.parametrize("make", [knapsack, general_integer, facility_location])
def test_solve_matches_reference(make):
    mip = make()
    assert_solved(solve(mip), mip, reference(mip))


@pytest.mark.parametrize(
    "branching, node_selection",
    [
        ("most_fractional", "best_first"),
        ("pseudocost", "depth_first"),
        ("reliability", "best_first"),
        ("reliability", "hybrid"),
    ],
)
def test_search_options(branching, node_selection):
    mip = general_integer()
    result = solve(mip, branching=branching, node_selection=node_selection)
    assert_solved(result, mip, reference(mip))


def test_maximisation_via_mps(tmp_path):
    # maximise -(c.x + 3): the same problem as the facility location.
    base = facility_location()
    mip = jm.MIP(base.lp, base.integrality, offset=3.0, maximise=True)
    path = tmp_path / "facility.mps"
    write_mps(mip, path)

    loaded = jm.load(path)
    assert loaded.maximise and loaded.offset == pytest.approx(3.0)
    np.testing.assert_array_equal(loaded.integrality, base.integrality)
    np.testing.assert_allclose(loaded.c, base.c)

    ref = reference(mip)
    assert ref == pytest.approx(-(reference(base) + 3.0))
    for model in (path, str(path), loaded):
        assert_solved(solve(model), loaded, ref)


def reducible_highs(mip):
    """``mip`` in a ``highspy.Highs`` plus a fixed integer column and a
    singleton row, which HiGHS's presolve removes."""
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.passModel(jm.to_highs(mip))
    n = mip.num_variables
    highs.addVar(2.0, 2.0)
    highs.changeColCost(n, 1.5)
    highs.changeColIntegrality(n, highspy.HighsVarType.kInteger)
    highs.addRow(-highspy.kHighsInf, 1.0, 1, np.array([0], np.int32), np.array([1.0]))
    return highs


@pytest.mark.parametrize("make", [knapsack, facility_location])
def test_presolve(make):
    highs = reducible_highs(make())
    original = jm.from_highs(highs)
    reduced = jm.from_highs(highs, presolve=True)
    assert reduced.num_variables < original.num_variables
    ref = reference(original)

    result = solve(reducible_highs(make()), presolve=True)
    assert result["presolve"] == "kReduced"
    assert result["postsolve_feasible"]
    assert result["x"].shape == (original.num_variables,)
    assert_solved(result, original, ref)


def test_presolve_maximisation(tmp_path):
    base = knapsack()
    mip = jm.MIP(base.lp, base.integrality, offset=1.5, maximise=True)
    path = tmp_path / "knapsack.mps"
    write_mps(mip, path)
    result = solve(path, presolve=True)
    assert_solved(result, jm.load(path), reference(mip))


def test_presolve_detects_infeasibility():
    # 2x = 1 with x integer: presolve sees it at once.
    lp = jl.LP(
        c=np.ones(1),
        A_eq=sp.csc_matrix([[2.0]]),
        b_eq=np.array([1.0]),
        A_ineq=sp.csc_matrix((0, 1)),
        b_ineq=np.zeros(0),
        lower_bounds=np.zeros(1),
        upper_bounds=np.full(1, 3.0),
    )
    mip = jm.MIP(lp, [True])
    assert solve(mip, presolve=True)["status"] == "infeasible"
    with pytest.raises(ValueError, match="infeasible"):
        jm.from_highs(jm.to_highs(mip), presolve=True)


def test_infeasible_with_feasible_relaxation():
    # 2 (x1 + ... + x6) = 3 over binaries: the LP relaxation is feasible,
    # every integer point is not, and propagation alone cannot tell at the root.
    n = 6
    A_eq = sp.csc_matrix(2.0 * np.ones((1, n)))
    lp = jl.LP(
        c=np.arange(n, dtype=float),
        A_eq=A_eq,
        b_eq=np.array([3.0]),
        A_ineq=sp.csc_matrix((0, n)),
        b_ineq=np.zeros(0),
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
    )
    result = solve(jm.MIP(lp, np.ones(n, bool)), probing=False)
    assert result["status"] == "infeasible"
    assert result["x"] is None and result["objective"] is None
    assert result["nodes"] >= 1
    # Probing tells at the root: every side of x1 leaves an odd right-hand
    # side, and propagation then empties both.
    probed = solve(jm.MIP(lp, np.ones(n, bool)))
    assert probed["status"] == "infeasible" and probed["nodes"] == 0


def test_unbounded_relaxation():
    # minimise -x0 s.t. x0 - x1 <= 0.5, x >= 0: unbounded.
    lp = jl.LP(
        c=np.array([-1.0, 0.0]),
        A_eq=sp.csc_matrix((0, 2)),
        b_eq=np.zeros(0),
        A_ineq=sp.csc_matrix([[1.0, -1.0]]),
        b_ineq=np.array([0.5]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.full(2, np.inf),
    )
    # The root LP runs to convergence without an epoch limit, so an
    # unbounded one is only classified once a limit stops it unconverged.
    result = solve(jm.MIP(lp, [True, False]), max_seconds=60, root_max_epochs=200)
    assert result["status"] == "unbounded_or_infeasible"


def test_limits_report_valid_bounds():
    mip = general_integer(n=12, m_ineq=8)
    ref = reference(mip)
    # Without root cuts: with them (simplex cut rounds) the root solves it.
    # Pseudocost branching: exact strong branching solves it in 5 nodes.
    result = solve(mip, max_nodes=5, cuts=False, branching="pseudocost")
    assert result["status"] == "node_limit" and not result["converged"]
    assert result["nodes"] <= 5
    assert result["bound"] <= ref + 1e-6 * (1 + abs(ref))
    if result["x"] is not None:
        assert mip.is_feasible(result["x"])
        assert result["objective"] >= ref - 1e-6 * (1 + abs(ref))

    # A limit spent before the root LP finishes (0.5 s was enough to solve
    # this MIP once compiled).
    result = solve(mip, max_seconds=1e-3)
    assert result["status"] == "time_limit"
    assert result["bound"] <= ref + 1e-6 * (1 + abs(ref))


@pytest.mark.parametrize("make", [knapsack, general_integer, facility_location])
def test_default_settings_solve(make):
    # jm.solve, not the pinned solve: batched float32 nodes at loose
    # tolerances, with the heuristics on background threads.
    mip = make()
    assert_solved(jm.solve(mip), mip, reference(mip), rigorous=False)


def test_verbose_log(capsys):
    """The progress table and the end of the log follow HiGHS's layout."""
    solve(knapsack(), verbose=True)
    out = capsys.readouterr().out
    assert "Root LP:" not in out
    assert "Src  Proc. InQueue |  Leaves   Expl. | BestBound" in out
    assert "Solving report" in out
    assert re.search(r"^  Status +Optimal$", out, re.M)
    # A table line: source, counts, explored share, bounds, gap, cut and
    # conflict counts, epochs and time.
    assert re.search(r"^ . +\d+ +\d+ +\d+ +[\d.]+% ", out, re.M)
    for noise in ("Root heuristics:", "Root dives:", "Tree cuts: room"):
        assert noise not in out


def test_verbose_legend_lists_enabled_heuristics(capsys):
    """The source legend names only the heuristics the solve has on."""
    solve(knapsack(), verbose=True, feasibility_pump=False, bulk_rounding=False)
    legend = capsys.readouterr().out.split("\n\n")[0]
    assert "T => Evaluate node" in legend
    assert "J => Feasibility jump" in legend
    for off in ("F => Feasibility pump", "B => Bulk rounding", "S => User start"):
        assert off not in legend
    solve(knapsack(), verbose=True, feasibility_pump=True, bulk_rounding=True)
    out = capsys.readouterr().out
    assert "F => Feasibility pump" in out and "B => Bulk rounding" in out


def test_rejects_bad_options():
    with pytest.raises(ValueError, match="branching"):
        solve(knapsack(), branching="strong")
    with pytest.raises(ValueError, match="lp_options"):
        solve(knapsack(), lp_options={"max_epochs": 3})


# %% Reading models
def test_load_reads_integrality_and_rounds_bounds(tmp_path):
    n = 3
    A_eq, b_eq = no_rows(n)
    lp = jl.LP(
        c=np.array([1.0, 2.0, 3.0]),
        A_eq=A_eq,
        b_eq=b_eq,
        A_ineq=sp.csc_matrix([[1.0, 1.0, 1.0]]),
        b_ineq=np.array([10.0]),
        lower_bounds=np.array([0.5, -1.5, 0.25]),
        upper_bounds=np.array([4.7, 2.0, 1.75]),
    )
    integrality = np.array([True, True, False])
    path = tmp_path / "small.mps"
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    model = hh.jaddle_lp_to_highs(lp, integrality)
    model.offset_ = 2.0
    highs.passModel(model)
    highs.writeModel(str(path))

    mip = jm.load(path)
    assert isinstance(mip.lp, jl.JaddleLP)
    np.testing.assert_array_equal(mip.integrality, integrality)
    np.testing.assert_allclose(mip.lower, [1.0, -1.0, 0.25])
    np.testing.assert_allclose(mip.upper, [4.0, 2.0, 1.75])
    np.testing.assert_allclose(np.asarray(mip.lp.lower_bounds), mip.lower)
    assert mip.offset == pytest.approx(2.0) and not mip.maximise
    assert mip.objective(np.array([1.0, -1.0, 1.0])) == pytest.approx(1.0 - 2.0 + 3.0 + 2.0)

    # The same model through each entry point.
    for model in (highs, highs.getLp(), str(path)):
        again = jm.from_highs(model)
        np.testing.assert_array_equal(again.integrality, integrality)
        np.testing.assert_allclose(again.c, mip.c)

    # The LP reader still relaxes integrality.
    assert jl.load(path).c.shape == (n,)


def test_continuous_model_has_no_integers():
    mip = jm.from_highs(hh.jaddle_lp_to_highs(knapsack().lp))
    assert mip.num_integers == 0


def test_rejects_semicontinuous():
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.passModel(jm.to_highs(knapsack()))
    highs.changeColIntegrality(0, highspy.HighsVarType.kSemiContinuous)
    with pytest.raises(NotImplementedError, match="semi"):
        jm.from_highs(highs)


def test_load_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError, match="could not read"):
        jm.load(tmp_path / "missing.mps")


def test_to_highs_round_trip():
    mip = facility_location()
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.passModel(jm.to_highs(mip))
    highs.run()
    assert highs.getInfo().objective_function_value == pytest.approx(reference(mip))


# %% Building blocks
def test_propagate_bounds_tightens_and_detects_infeasibility():
    # x0 + x1 <= 3, x0 - x1 >= 1 (as -x0 + x1 <= -1), x integer in [0, 10].
    # Row by row: x0 >= 1 from row 2, then x0 <= 3 and x1 <= 2 from row 1.
    A = sp.csr_matrix([[1.0, 1.0], [-1.0, 1.0]])
    row_lower = np.full(2, -np.inf)
    row_upper = np.array([3.0, -1.0])
    lower, upper, infeasible = jm.propagate_bounds(
        A, row_lower, row_upper, np.zeros(2), np.full(2, 10.0), np.array([True, True])
    )
    assert not infeasible
    np.testing.assert_allclose(lower, [1.0, 0.0])
    np.testing.assert_allclose(upper, [3.0, 2.0])

    # Now also require x0 + x1 >= 4: no point left.
    A = sp.csr_matrix([[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]])
    _, _, infeasible = jm.propagate_bounds(
        A,
        np.full(3, -np.inf),
        np.array([3.0, -1.0, -4.0]),
        np.zeros(2),
        np.full(2, 10.0),
        np.array([True, True]),
    )
    assert infeasible


def test_propagate_bounds_checks_the_final_box():
    # Binaries with x0 = 1, x0 <= x1, x0 <= x3 and x1 + x3 <= 1. One pass
    # raises x1 and x3 to 1 together, which breaks the third row; a pass
    # limit of 1 ends right after that tightening, so only a check of the
    # final box finds it.
    A = sp.csr_matrix(
        np.array([[1.0, -1.0, 0.0], [1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
    )
    row_upper = np.array([0.0, 0.0, 1.0])
    lower, upper = np.array([1.0, 0.0, 0.0]), np.ones(3)
    for passes in (1, 5):
        _, _, infeasible = jm.propagate_bounds(
            A, np.full(3, -np.inf), row_upper, lower, upper, np.ones(3, bool),
            max_passes=passes,
        )
        assert infeasible


def test_propagate_bounds_handles_infinite_bounds():
    # x0 + x1 = 4 with x0 free and x1 in [0, 1]: x0 in [3, 4].
    A = sp.csr_matrix([[1.0, 1.0]])
    lower, upper, infeasible = jm.propagate_bounds(
        A,
        np.array([4.0]),
        np.array([4.0]),
        np.array([-np.inf, 0.0]),
        np.array([np.inf, 1.0]),
        np.array([False, False]),
    )
    assert not infeasible
    np.testing.assert_allclose(lower, [3.0, 0.0])
    np.testing.assert_allclose(upper, [4.0, 1.0])


def test_lagrangian_bound():
    mip = facility_location()
    lp = mip.lp.to_scipy()
    bounds = [(lo, None if np.isinf(hi) else hi) for lo, hi in zip(mip.lower, mip.upper)]
    from scipy.optimize import linprog

    ref = linprog(
        mip.c,
        A_ub=lp.A_ineq,
        b_ub=lp.b_ineq,
        A_eq=lp.A_eq,
        b_eq=lp.b_eq,
        bounds=bounds,
        method="highs",
    )
    # At the optimal duals the bound is the LP optimum (SciPy's marginals are
    # the negated Jaddle duals).
    y = -np.concatenate([ref.eqlin.marginals, ref.ineqlin.marginals])
    upper = mip.upper.copy()
    upper[~np.isfinite(upper)] = 1e3  # finite bounds the flows never reach
    bound, rigorous = jm.lagrangian_bound(
        mip.c, mip.A, mip.b, mip.n_eq, mip.lower, upper, y
    )
    assert rigorous
    assert bound == pytest.approx(ref.fun, rel=1e-9)

    # Any other duals give a valid, weaker bound.
    rng = np.random.default_rng(0)
    for _ in range(20):
        y_random = y + rng.standard_normal(y.shape)
        bound, _ = jm.lagrangian_bound(
            mip.c, mip.A, mip.b, mip.n_eq, mip.lower, upper, y_random
        )
        assert bound <= ref.fun + 1e-9

    # A reduced cost pointing at an infinite bound gives -inf, unless it is
    # below the tolerance (then it is dropped and the bound is not rigorous).
    y_bad = y.copy()
    y_bad[0] -= 100.0
    bound, rigorous = jm.lagrangian_bound(
        mip.c, mip.A, mip.b, mip.n_eq, mip.lower, mip.upper, y_bad
    )
    assert bound == -np.inf and rigorous


# %% Cutting planes
def test_mir_cut_textbook():
    # x0 + x1 <= 1.5 with x integer in [0, 5], at (0.75, 0.75): the MIR with
    # divisor 1 is x0 + x1 <= 1.
    idx = np.array([0, 1])
    cut = jm.mir_cut(
        idx,
        np.array([1.0, 1.0]),
        1.5,
        np.array([0.75, 0.75]),
        np.zeros(2),
        np.full(2, 5.0),
        np.array([True, True]),
    )
    assert cut is not None
    cut_idx, coef, rhs = cut
    np.testing.assert_allclose(coef, [1.0, 1.0])
    assert rhs == pytest.approx(1.0)


def test_mir_cut_mixed_row():
    # x0 - y <= 0.5 with x0 integer, y >= 0 continuous, at (0.5, 0): the MIR
    # is x0 - 2y <= 0.
    cut = jm.mir_cut(
        np.array([0, 1]),
        np.array([1.0, -1.0]),
        0.5,
        np.array([0.5, 0.0]),
        np.zeros(2),
        np.array([5.0, np.inf]),
        np.array([True, False]),
    )
    _, coef, rhs = cut
    np.testing.assert_allclose(coef, [1.0, -2.0])
    assert rhs == pytest.approx(0.0, abs=1e-8)


def test_cover_cut_textbook():
    # 5 x0 + 5 x1 + 5 x2 + 9 x3 <= 12 over binaries at (0.8, 0.8, 0.8, 0):
    # {0, 1, 2} is a cover. Lifting x3: with x3 = 1 only 3 is left, so no
    # cover item fits and x3 gets 2: x0 + x1 + x2 + 2 x3 <= 2.
    cut = jm.cover_cut(
        np.arange(4),
        np.array([5.0, 5.0, 5.0, 9.0]),
        12.0,
        np.array([0.8, 0.8, 0.8, 0.0]),
        np.zeros(4),
        np.ones(4),
        np.ones(4, bool),
    )
    cut_idx, coef, rhs = cut
    np.testing.assert_array_equal(cut_idx, np.arange(4))
    np.testing.assert_allclose(coef, [1.0, 1.0, 1.0, 2.0])
    assert rhs == pytest.approx(2.0)


def test_cover_cut_complements_negative_coefficients():
    # 4 x0 - 4 x1 <= 0 (x0 <= x1) is 4 x0 + 4 z1 <= 4 with z1 = 1 - x1; the
    # cover {x0, z1} gives x0 + z1 <= 1, i.e. x0 - x1 <= 0. At (0.5, 0) it is
    # violated.
    cut = jm.cover_cut(
        np.arange(2),
        np.array([4.0, -4.0]),
        0.0,
        np.array([0.5, 0.25]),
        np.zeros(2),
        np.ones(2),
        np.ones(2, bool),
    )
    _, coef, rhs = cut
    np.testing.assert_allclose(coef, [1.0, -1.0])
    assert rhs == pytest.approx(0.0, abs=1e-8)


@pytest.mark.parametrize("make", [knapsack, facility_location, general_integer])
def test_separated_cuts_are_valid(make):
    # Each cut's largest left-hand side over the MIP's feasible points is at
    # most its right-hand side, and it cuts off the LP point.
    from scipy.optimize import linprog

    mip = make()
    lower, upper, _ = jm.propagate_bounds(
        mip.A, mip.row_lower, mip.row_upper, mip.lower, mip.upper, mip.integrality
    )
    A = mip.A.toarray()
    bounds = [(lo, None if np.isinf(hi) else hi) for lo, hi in zip(mip.lower, mip.upper)]
    lp = linprog(
        mip.c,
        A_ub=A[mip.n_eq :],
        b_ub=mip.b[mip.n_eq :],
        A_eq=A[: mip.n_eq] if mip.n_eq else None,
        b_eq=mip.b[: mip.n_eq] if mip.n_eq else None,
        bounds=bounds,
    )
    cuts = jm.CutSeparator(mip, lower, upper)(lp.x)
    if make is not general_integer:
        assert cuts
    for _, idx, coef, rhs in cuts:
        assert coef @ lp.x[idx] > rhs
        objective = np.zeros(mip.num_variables)
        objective[idx] = -coef
        best = milp(
            objective,
            constraints=LinearConstraint(A, mip.row_lower, mip.row_upper),
            integrality=mip.integrality.astype(int),
            bounds=Bounds(mip.lower, mip.upper),
        )
        assert -best.fun <= rhs + 1e-7 * (1 + abs(rhs))


def test_cut_block_padding():
    cuts = [
        dict(idx=np.array([0, 2]), coef=np.array([1.0, 2.0]), rhs=3.0),
        dict(idx=np.array([1]), coef=np.array([-1.0]), rhs=0.5),
    ]
    A, b = jm._cut_block(cuts, n=4, min_rows=8, min_nnz=16)
    assert A.shape == (8, 4) and A.nnz == 16
    np.testing.assert_allclose(A[:2].toarray(), [[1, 0, 2, 0], [0, -1, 0, 0]])
    assert np.count_nonzero(A.toarray()[2:]) == 0
    np.testing.assert_allclose(b, [3.0, 0.5, 0, 0, 0, 0, 0, 0])


def test_cuts_strengthen_the_root(monkeypatch):
    mip = facility_location()
    ref = reference(mip)
    after_cuts = []  # the root bound the cut loop leaves
    real_loop = jm._BranchAndBound.cut_loop

    def cut_loop(self, lower, upper, result):
        result = real_loop(self, lower, upper, result)
        bound, _ = self.lp_bound(result["solution"], lower, upper)
        after_cuts.append(self.model_value(bound))
        return result

    monkeypatch.setattr(jm._BranchAndBound, "cut_loop", cut_loop)
    with_cuts = solve(mip)
    without = solve(mip, cuts=False)
    for result in (with_cuts, without):
        assert_solved(result, mip, ref)
    # MIR or flow cover cuts (on these fixed-charge rows flow covers often
    # win).
    assert with_cuts["cuts"]["mir"] + with_cuts["cuts"]["flowcover"] > 0
    assert without["cuts"] == {
        "gomory": 0, "cover": 0, "mir": 0, "flowcover": 0, "clique": 0, "zerohalf": 0,
        "impliedbound": 0,
    }
    assert without["root_bound_before_cuts"] is None
    # The cuts close part of the root gap and the tree is no larger.
    root = with_cuts["root_bound_before_cuts"]
    assert after_cuts[0] > root + 1e-3 * abs(ref - root)
    assert with_cuts["nodes"] <= without["nodes"]


# %% Primal heuristics
def generalised_assignment(seed=0, agents=5, jobs=20, tightness=0.8):
    """Each job goes to exactly one agent within the agents' capacities.
    Rounding its LP point almost never gives a feasible assignment."""
    rng = np.random.default_rng(seed)
    weight = rng.integers(5, 25, (agents, jobs)).astype(float)
    cost = rng.integers(10, 50, (agents, jobs)).astype(float)
    capacity = tightness * weight.sum(1) / agents
    n = agents * jobs
    A_eq = sp.lil_matrix((jobs, n))
    A_ineq = sp.lil_matrix((agents, n))
    for i in range(agents):
        for j in range(jobs):
            A_eq[j, i * jobs + j] = 1.0
            A_ineq[i, i * jobs + j] = weight[i, j]
    lp = jl.LP(
        c=cost.ravel(),
        A_eq=A_eq.tocsc(),
        b_eq=np.ones(jobs),
        A_ineq=A_ineq.tocsc(),
        b_ineq=capacity,
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
    )
    return jm.MIP(lp, np.ones(n, bool))


def lp_point(mip):
    result = jl.solve(
        mip.lp,
        primal_feasibility_tolerance=1e-7,
        dual_feasibility_tolerance=1e-7,
        dual_gap_tolerance=1e-7,
    )
    return np.asarray(result["solution"].primal, np.float64)


def test_row_violation_is_relative_to_each_rows_rhs():
    # One large right-hand side must not excuse a violation of another row
    # (map06: b up to 5.1e7 let rows with b = 0 be off by 0.3).
    A_eq, b_eq = no_rows(2)
    lp = jl.LP(
        c=np.array([-1.0, 0.0]),
        A_eq=A_eq,
        b_eq=b_eq,
        A_ineq=sp.csc_matrix(np.array([[1.0, 1.0], [1.0, -1.0]])),
        b_ineq=np.array([1e7, 0.0]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.full(2, 10.0),
    )
    mip = jm.MIP(lp, np.array([False, True]))
    x = np.array([1.3, 1.0])  # x0 - x1 <= 0 violated by 0.3
    assert mip.violation(x)["rows"] == pytest.approx(0.3)
    assert not mip.is_feasible(x)
    assert list(jm._violated_rows(mip, x, 1e-6)) == [1]
    assert mip.is_feasible(np.array([1.0, 1.0]))


def test_fix_and_propagate_finds_an_assignment():
    mip = generalised_assignment()
    x = lp_point(mip)
    propagator = jm.BoundPropagator(
        mip.A, mip.row_lower, mip.row_upper, mip.integrality
    )
    candidate = jm.fix_and_propagate(x, mip.lower, mip.upper, mip.integrality, propagator)
    assert candidate is not None
    assert mip.is_feasible(candidate)
    # Plain rounding of the same point is not feasible.
    assert not mip.is_feasible(np.round(x))


def test_fix_and_propagate_gives_up_on_an_infeasible_box():
    mip = generalised_assignment()
    propagator = jm.BoundPropagator(
        mip.A, mip.row_lower, mip.row_upper, mip.integrality
    )
    # No agent may take job 0: its assignment row cannot be met.
    upper = mip.upper.copy()
    upper[np.arange(5) * 20] = 0.0
    x = np.full(mip.num_variables, 0.2)
    assert jm.fix_and_propagate(x, mip.lower, upper, mip.integrality, propagator) is None


@pytest.mark.parametrize("seed", [0, 1, 5])
def test_root_heuristics_find_incumbents(seed):
    mip = generalised_assignment(seed)
    ref = reference(mip)
    # RENS off too: on seed 0 its box is infeasible but its dive finds a point.
    off = solve(
        mip,
        max_nodes=1,
        cuts=False,
        fix_and_propagate=False,
        feasibility_jump=False,
        feasibility_pump=False,
        rens_seconds=0,
    )
    assert off["x"] is None
    result = solve(mip, max_nodes=1, cuts=False)
    assert result["x"] is not None and mip.is_feasible(result["x"])
    assert result["objective"] >= ref - 1e-6
    assert result["history"][0]["source"] in ("P", "J", "F")
    assert result["heuristic_seconds"] >= 0.0


def test_feasibility_pump_alone():
    # Seed 5: the pump succeeds without any fix-and-propagate (on some other
    # seeds it needs fix-and-propagate as its rounding step).
    mip = generalised_assignment(5)
    result = solve(
        mip,
        max_nodes=1,
        cuts=False,
        fix_and_propagate=False,
        feasibility_jump=False,
    )
    assert result["x"] is not None and mip.is_feasible(result["x"])
    assert result["history"][0]["source"] == "F"


def test_heuristics_keep_the_optimum():
    mip = generalised_assignment(3)
    assert_solved(solve(mip), mip, reference(mip))


# %% Gomory cuts
def vertex_and_basis(mip):
    result = jl.solve(
        mip.lp,
        primal_feasibility_tolerance=1e-8,
        dual_feasibility_tolerance=1e-8,
        dual_gap_tolerance=1e-8,
    )
    crossed = jl.crossover(mip.lp, result["solution"])
    assert crossed["converged"]
    return np.asarray(crossed["solution"].primal, np.float64), crossed["basis"]


def test_gomory_cut_textbook():
    # max x1 + x2 s.t. 2 x1 + 2 x2 <= 3, x integer in [0, 5]: the LP vertex
    # has x1 + x2 = 1.5, and the Gomory cut from it is x1 + x2 <= 1.
    lp = jl.LP(
        c=np.array([-1.0, -1.0]),
        A_eq=sp.csc_matrix((0, 2)),
        b_eq=np.zeros(0),
        A_ineq=sp.csc_matrix([[2.0, 2.0]]),
        b_ineq=np.array([3.0]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.full(2, 5.0),
    )
    mip = jm.MIP(lp, [True, True])
    x, basis = vertex_and_basis(mip)
    cuts = jm.gomory_cuts(
        mip.A, mip.b, mip.n_eq, mip.lower, mip.upper, mip.integrality, basis, x
    )
    assert cuts
    family, idx, coef, rhs = cuts[0]
    assert family == "gomory"
    assert coef @ x[idx] > rhs + 0.1  # cuts off the vertex
    # Every integer point of the box that meets the row meets the cut.
    for x1 in range(6):
        for x2 in range(6):
            point = np.array([x1, x2], float)
            if 2 * x1 + 2 * x2 <= 3:
                assert coef @ point[idx] <= rhs + 1e-9


@pytest.mark.parametrize(
    "make", [knapsack, facility_location, general_integer, generalised_assignment]
)
def test_gomory_cuts_are_valid(make):
    mip = make()
    x, basis = vertex_and_basis(mip)
    cuts = jm.gomory_cuts(
        mip.A, mip.b, mip.n_eq, mip.lower, mip.upper, mip.integrality, basis, x
    )
    assert cuts
    A = mip.A.toarray()
    for _, idx, coef, rhs in cuts:
        assert coef @ x[idx] > rhs
        objective = np.zeros(mip.num_variables)
        objective[idx] = -coef
        best = milp(
            objective,
            constraints=LinearConstraint(A, mip.row_lower, mip.row_upper),
            integrality=mip.integrality.astype(int),
            bounds=Bounds(mip.lower, mip.upper),
        )
        assert -best.fun <= rhs + 1e-7 * (1 + abs(rhs))


def test_gomory_cuts_in_the_search():
    mip = general_integer()
    ref = reference(mip)
    result = solve(mip, gomory_cuts=True)
    assert_solved(result, mip, ref)
    assert result["cuts"]["gomory"] > 0
    off = solve(mip, gomory_cuts=False)
    assert_solved(off, mip, ref)
    assert off["cuts"]["gomory"] == 0


def _count_cut_rounds(monkeypatch):
    """Record each root cut round's re-solve (one ``age_cuts`` call each,
    before compaction) and ``round_heuristics`` call."""
    seen = {"rounds": 0, "round_heuristics": [], "incumbent_at_cuts": []}
    real_age = jm._BranchAndBound.age_cuts
    real_round = jm._BranchAndBound.round_heuristics
    real_loop = jm._BranchAndBound.cut_loop

    def age_cuts(self, solution):
        seen["rounds"] += 1
        return real_age(self, solution)

    def round_heuristics(self, x, *args):
        seen["round_heuristics"].append(x.copy())
        return real_round(self, x, *args)

    def cut_loop(self, *args):
        seen["incumbent_at_cuts"].append(self.incumbent)
        return real_loop(self, *args)

    monkeypatch.setattr(jm._BranchAndBound, "age_cuts", age_cuts)
    monkeypatch.setattr(jm._BranchAndBound, "round_heuristics", round_heuristics)
    monkeypatch.setattr(jm._BranchAndBound, "cut_loop", cut_loop)
    return seen


def test_cut_rounds_judge_progress_only_after_three(monkeypatch):
    """However high ``cut_min_progress`` is, the root cut loop runs three
    rounds before it judges their progress (on rmine13 one round's gain was
    judged against three rounds' worth and the loop stopped after one)."""
    seen = _count_cut_rounds(monkeypatch)
    mip = general_integer()
    result = solve(
        mip, gomory_cuts=True, cut_rounds=10, cut_min_progress=1e9, max_nodes=1
    )
    assert sum(result["cuts"].values()) > 0
    assert seen["rounds"] == 3


def test_root_heuristics_run_before_the_cut_rounds(monkeypatch):
    """The root heuristics run on the first root LP, so the cut rounds
    start with an incumbent, and each round from the second rounds its
    point (``round_heuristics``)."""
    seen = _count_cut_rounds(monkeypatch)
    mip = generalised_assignment()
    ref = reference(mip)
    result = solve(mip, gomory_cuts=True, cut_rounds=5, cut_min_progress=1e9)
    assert_solved(result, mip, ref)
    assert seen["incumbent_at_cuts"][0] is not None
    assert seen["rounds"] >= 2
    assert len(seen["round_heuristics"]) == seen["rounds"] - 1
    assert all(x.shape == (mip.num_variables,) for x in seen["round_heuristics"])


def test_root_cut_heuristics_run_in_the_background(monkeypatch):
    """With ``parallel_heuristics`` the cut rounds do not wait for the
    heuristics: the first root LP's fix-and-propagate and jumps, and the
    later rounds' fix-and-propagate, go to background lanes before the
    rounds, and the main thread runs no root heuristics until they end."""
    submitted = []  # (lane, kind, inside the cut loop)
    main_thread = []  # root_heuristics calls, and whether inside the loop
    in_loop = []
    real_submit = jm._HeuristicWorkers.submit
    real_root = jm._BranchAndBound.root_heuristics
    real_loop = jm._BranchAndBound.cut_loop

    def submit(self, lane, kind, *args, **kwargs):
        submitted.append((lane, kind, bool(in_loop)))
        return real_submit(self, lane, kind, *args, **kwargs)

    def root_heuristics(self, *args, **kwargs):
        main_thread.append(bool(in_loop))
        return real_root(self, *args, **kwargs)

    def cut_loop(self, *args):
        in_loop.append(True)
        try:
            return real_loop(self, *args)
        finally:
            in_loop.clear()
            main_thread.append("after")

    monkeypatch.setattr(jm._HeuristicWorkers, "submit", submit)
    monkeypatch.setattr(jm._BranchAndBound, "root_heuristics", root_heuristics)
    monkeypatch.setattr(jm._BranchAndBound, "cut_loop", cut_loop)
    mip = generalised_assignment()
    ref = reference(mip)
    result = solve(
        mip,
        parallel_heuristics=True,
        jump_backend="numba",
        gomory_cuts=True,
        cut_rounds=5,
        cut_min_progress=1e9,
    )
    assert_solved(result, mip, ref, rigorous=False)
    # Before the rounds: fix-and-propagate on the rounding lane (and jumps
    # on the jump lane while there was no incumbent).
    assert ("round", "fixprop", False) in submitted
    # During them: each later round's point.
    assert any(lane == "round" and inside for lane, _, inside in submitted)
    # The main thread's root heuristics come only after the cut loop.
    assert main_thread and main_thread[0] == "after"


@pytest.mark.parametrize("age", [None, 1, 3])
@pytest.mark.parametrize("make", [general_integer, generalised_assignment])
def test_cut_rounds_keep_cuts_until_idle_for_cut_age_rounds(make, age, monkeypatch):
    """A root cut stays in the LP until it has been idle (slack, with a zero
    dual) at ``cut_age`` re-solves in a row (default ``_CUT_AGE``), then
    leaves it; every other cut stays."""
    age_setting = {} if age is None else {"cut_age": age}
    age = jm._CUT_AGE if age is None else age
    ages = []  # each round's cuts and their ages after its re-solve
    blocks = []  # the cut list each round's LP was built from
    real_age = jm._BranchAndBound.age_cuts
    real_cut_block = jm._cut_block
    real_compact = jm._BranchAndBound.compact_cuts
    compacting = []

    def age_cuts(self, solution):
        real_age(self, solution)
        ages.append([(id(cut), cut["age"]) for cut in self.cut_list])

    def cut_block(cuts, *args, **kwargs):
        if not compacting:
            blocks.append({id(cut) for cut in cuts})
        return real_cut_block(cuts, *args, **kwargs)

    def compact_cuts(self, *args):
        compacting.append(True)  # the rounds are over
        return real_compact(self, *args)

    monkeypatch.setattr(jm._BranchAndBound, "age_cuts", age_cuts)
    monkeypatch.setattr(jm, "_cut_block", cut_block)
    monkeypatch.setattr(jm._BranchAndBound, "compact_cuts", compact_cuts)
    mip = make()
    result = solve(mip, gomory_cuts=True, max_nodes=1, **age_setting)
    assert result["cuts"]["gomory"] + sum(result["cuts"].values()) > 0
    checked = dropped = 0
    # Round k's ages decide what round k + 1's block keeps.
    for aged, block in zip(ages, blocks[1:]):
        for key, cut_age in aged:
            assert (key in block) == (cut_age < age)
            checked += 1
            dropped += cut_age >= age
    assert checked > 0
    if age == 1:
        assert dropped > 0  # idle cuts do leave


def test_cap_cut_rows_drops_the_longest_idle_cuts_first():
    search = types.SimpleNamespace(
        max_cut_rows=4,
        cut_list=[dict(age=0), dict(age=3), dict(age=1), dict(age=5)],
    )
    cap = jm._BranchAndBound.cap_cut_rows
    # 4 kept + 3 new, room for 5: the two idle longest (5 and 3) go.
    search.max_cut_rows = 5
    binding = [True] * 4
    assert cap(search, binding, ["a", "b", "c"]) == ["a", "b", "c"]
    assert binding == [True, False, True, False]
    # Room for 4: all three idle ones go, and the new cuts fit.
    search.max_cut_rows = 4
    binding = [True] * 4
    assert cap(search, binding, ["a", "b", "c"]) == ["a", "b", "c"]
    assert binding == [True, False, False, False]
    # Only active cuts: they stay, and the new cuts fill what is left.
    search.cut_list = [dict(age=0)] * 3
    binding = [True] * 3
    assert cap(search, binding, ["a", "b", "c"]) == ["a"]
    assert binding == [True] * 3
    search.max_cut_rows = 2
    assert cap(search, [True] * 3, ["a"]) == []


def test_max_cut_rows_caps_the_root_lp(monkeypatch):
    sizes = []
    real_cut_block = jm._cut_block

    def cut_block(cuts, *args, **kwargs):
        sizes.append(len(cuts))
        return real_cut_block(cuts, *args, **kwargs)

    monkeypatch.setattr(jm, "_cut_block", cut_block)
    mip = generalised_assignment()
    settings = dict(gomory_cuts=True, max_nodes=1, max_cuts_per_round=4)
    solve(mip, **settings)
    assert max(sizes) > 6  # without the cap the pool grows past it
    sizes.clear()
    solve(mip, max_cut_rows=6, **settings)
    assert sizes and max(sizes) <= 6  # every round's block and the tree's
    assert_solved(solve(mip, max_cut_rows=6), mip, reference(mip))


def test_gomory_candidates_limits_the_tableau_rows(monkeypatch):
    asked = []
    real = jm.gomory_cuts

    def gomory_cuts(*args, **kwargs):
        asked.append(kwargs["max_cuts"])
        return real(*args, **kwargs)

    monkeypatch.setattr(jm, "gomory_cuts", gomory_cuts)
    mip = generalised_assignment()
    solve(mip, gomory_cuts=True, max_nodes=1, gomory_candidates=3, max_cuts_per_round=50)
    assert asked and set(asked) == {3}
    asked.clear()
    solve(mip, gomory_cuts=True, max_nodes=1, max_cuts_per_round=50)
    assert asked and set(asked) == {100}


@pytest.mark.parametrize(
    "setting", ["cut_age", "max_cut_rows", "gomory_candidates"]
)
def test_rejects_bad_cut_pool_settings(setting):
    with pytest.raises(ValueError, match=setting):
        solve(knapsack(), **{setting: 0})
    with pytest.raises(ValueError, match=setting):
        solve(knapsack(), **{setting: 2.5})

# %% Conflicts
def cover_mip(seed, n=14, m=9):
    """A small set cover with integer costs: its root LP leaves room below
    the root heuristics' incumbent with nonzero reduced costs."""
    rng = np.random.default_rng(seed)
    A = np.zeros((m, n))
    for i in range(m):
        A[i, rng.choice(n, int(rng.integers(3, 6)), replace=False)] = -1.0
    lp = jl.LP(
        c=rng.integers(1, 30, n).astype(float),
        A_eq=sp.csc_matrix((0, n)),
        b_eq=np.zeros(0),
        A_ineq=sp.csc_matrix(A),
        b_ineq=-np.ones(m),
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
    )
    return jm.MIP(lp, np.ones(n, bool))


def _best_outside(mip, j, lo, up, step=1.0):
    """The least (min-form) objective of a feasible point of ``mip`` with
    ``x_j`` outside ``[lo, up]`` by at least ``step``; ``inf`` when there is
    none."""
    best = np.inf
    for below in (True, False):
        lower, upper = mip.lower.copy(), mip.upper.copy()
        if below:
            if lo <= lower[j]:
                continue
            upper[j] = lo - step
        else:
            if up >= upper[j]:
                continue
            lower[j] = up + step
        result = milp(
            mip.c,
            constraints=LinearConstraint(mip.A.toarray(), mip.row_lower, mip.row_upper),
            integrality=mip.integrality.astype(int),
            bounds=Bounds(lower, upper),
        )
        if result.success:
            best = min(best, result.fun)
    return best


def _check_root_boxes(mip, records, continuous_step=None):
    """Each recorded root box ``(lower, upper, cutoff)`` keeps every
    feasible point better than its cutoff. Returns the bounds checked.
    ``continuous_step``: check continuous bounds this close (relative to
    ``1 + |bound|``) instead of a whole unit out."""
    checked = 0
    for lower, upper, cutoff in records:
        for j in np.flatnonzero((lower > mip.lower) | (upper < mip.upper)):
            step = 1.0
            if continuous_step is not None and not mip.integrality[j]:
                width = max(abs(lower[j]), abs(upper[j]) if np.isfinite(upper[j]) else 0.0)
                step = continuous_step * (1.0 + width)
            best = _best_outside(mip, j, lower[j], upper[j], step)
            assert best >= cutoff - 1e-6 * (1 + abs(cutoff))
            checked += 1
    return checked


@pytest.mark.parametrize("seed", range(10))
def test_global_redcost_row_holds_for_every_better_point(seed):
    rng = np.random.default_rng(seed)
    n = 8
    r = rng.normal(size=n) * rng.integers(0, 2, n)  # some zero
    constant, cutoff = float(rng.normal()), float(rng.uniform(0.5, 3.0))
    pool = jm.ConflictPool(n, np.ones(n, bool))
    pool.set_cutoff(cutoff + constant + np.minimum(r, 0).sum())  # room ``cutoff``
    if not pool.set_global(constant, r, np.zeros(n), np.ones(n), keep=4):
        return
    idx, coef, offset, objective = pool.global_row
    assert objective
    rhs = offset + pool.cutoff_value()
    points = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(float)
    better = points[constant + points @ r <= pool.cutoff_value()]
    assert np.all(better[:, idx] @ coef <= rhs + 1e-9)


def test_global_redcost_row_propagates_the_room():
    # 3 x0 + 3 x1 + 3 x2 + x3 <= 5 over binaries (constant 0): x0 = 1 leaves
    # room 2, so x1 = x2 = 0; x3 (|r| 1, under room / keep) is dropped.
    pool = jm.ConflictPool(4, np.ones(4, bool))
    pool.set_cutoff(5.0)
    assert pool.set_global(0.0, np.array([3.0, 3.0, 3.0, 1.0]), np.zeros(4), np.ones(4), keep=4)
    assert pool.global_row[0].tolist() == [0, 1, 2] and len(pool) == 1
    lower, upper, infeasible = pool.propagate(np.array([1.0, 0.0, 0.0, 0.0]), np.ones(4))
    assert not infeasible and upper[1] == 0.0 and upper[2] == 0.0 and upper[3] == 1.0
    # The right-hand side follows the cutoff.
    pool.set_cutoff(2.0)
    _, _, infeasible = pool.propagate(np.array([1.0, 0.0, 0.0, 0.0]), np.ones(4))
    assert infeasible
    # Replaced, not added to.
    assert pool.set_global(0.0, np.array([3.0, 0.0, 0.0, 0.0]), np.zeros(4), np.ones(4))
    assert pool.global_row[0].tolist() == [0] and len(pool) == 1


@pytest.mark.parametrize("make", [general_integer, generalised_assignment, knapsack])
def test_root_redcost_propagation_keeps_the_optimum(make):
    mip = make()
    result = solve(mip, batch_size=1, root_redcost_propagation=True)
    assert_solved(result, mip, reference(mip))


def test_global_redcost_fixing_holds_for_every_better_solution(monkeypatch):
    records = []
    real = jm._BranchAndBound.global_fix

    def global_fix(self, terms):
        real(self, terms)
        records.append((self.root_lower.copy(), self.root_upper.copy(), self.cutoff()))

    monkeypatch.setattr(jm._BranchAndBound, "global_fix", global_fix)
    mip = general_integer()
    result = solve(mip, batch_size=1, global_redcost_fixing=True)
    assert_solved(result, mip, reference(mip))
    assert result["global_redcost_fixed"] > 0
    assert _check_root_boxes(mip, records[-5:]) > 0


def strong_facility_location(seed=2, n_fac=4, n_cust=8):
    """``facility_location`` with the links ``y_fk <= d_k x_f`` as rows of
    their own: variable upper bounds on every flow."""
    mip = facility_location(seed, n_fac, n_cust)
    demand = mip.b[:n_cust]  # the equality rows: each customer's demand
    n = mip.num_variables
    links = sp.lil_matrix((n_fac * n_cust, n))
    for f in range(n_fac):
        for k in range(n_cust):
            links[f * n_cust + k, n_fac + f * n_cust + k] = 1.0
            links[f * n_cust + k, f] = -demand[k]
    return mip.with_rows(links.tocsr(), np.zeros(n_fac * n_cust))


def _lp_min(mip, objective, lower, upper, cutoff):
    """The least ``objective·x`` over ``mip``'s LP relaxation in the box
    with ``c·x <= cutoff``; ``inf`` when there is no such point."""
    from scipy.optimize import linprog

    A = sp.vstack([sp.csr_matrix(mip.A), sp.csr_matrix(mip.c[None])]).toarray()
    result = linprog(
        objective,
        A_ub=np.vstack([A[mip.n_eq :]]),
        b_ub=np.append(mip.b[mip.n_eq :], cutoff),
        A_eq=A[: mip.n_eq] if mip.n_eq else None,
        b_eq=mip.b[: mip.n_eq] if mip.n_eq else None,
        bounds=list(zip(lower, upper)),
        method="highs",
    )
    return result.fun if result.status == 0 else np.inf


def _optimum(mip):
    result = milp(
        mip.c,
        constraints=LinearConstraint(mip.A.toarray(), mip.row_lower, mip.row_upper),
        integrality=mip.integrality.astype(int),
        bounds=Bounds(mip.lower, mip.upper),
    )
    assert result.success
    return result.x


@pytest.mark.parametrize("seed", [0, 1])
def test_inout_cuts_cut_the_lp_point_and_keep_the_optimum(seed, monkeypatch):
    mip = generalised_assignment(seed, agents=5, jobs=20)
    optimum = _optimum(mip)
    rounds = []
    real_candidates = jm._BranchAndBound.inout_candidates
    real_result = jm._BranchAndBound.inout_result

    def inout_candidates(self, separators, x, *args, **kwargs):
        out = real_candidates(self, separators, x, *args, **kwargs)
        if out is not None:
            rounds.append([np.asarray(x, np.float64).copy(), out, None])
        return out

    def inout_result(self, found, inout):
        if inout is not None:
            rounds[-1][2] = [cut for cut in found if any(cut is c for c in inout)]
        return real_result(self, found, inout)

    monkeypatch.setattr(jm._BranchAndBound, "inout_candidates", inout_candidates)
    monkeypatch.setattr(jm._BranchAndBound, "inout_result", inout_result)
    result = solve(
        mip, batch_size=1, mip_start=optimum, inout_separation=0.5
    )
    assert_solved(result, mip, reference(mip))
    assert result["inout_rounds"] == len(rounds) > 0
    kept = [(x, cut) for x, _, found in rounds for cut in found or []]
    assert result["inout_cuts"] == len(kept) > 0
    for x, (_, idx, coef, rhs) in kept:
        assert coef @ x[idx] > rhs  # violated at the LP point
        assert coef @ optimum[idx] <= rhs + 1e-6 * (1 + abs(rhs))  # valid


def test_inout_separation_gives_up_without_cuts():
    # A knapsack row's covers at the in-out point: once a round keeps none,
    # λ moves towards the LP point, and past _INOUT_MAX_LAMBDA it stops.
    mip = general_integer(5, n=20, m_eq=3, m_ineq=15)
    result = solve(
        mip,
        batch_size=1,
        mip_start=_optimum(mip),
        inout_separation=0.5,
        cut_rounds=20,
        max_nodes=1,
    )
    assert result["inout_rounds"] <= 20
    if result["inout_cuts"] == 0:
        assert result["inout_rounds"] <= 3  # 0.5, 0.75, 0.875, then off


def test_inout_separation_is_off_by_default():
    mip = generalised_assignment(0, agents=5, jobs=20)
    result = solve(mip, batch_size=1, mip_start=_optimum(mip))
    assert result["inout_rounds"] == 0 and result["inout_cuts"] == 0


def _lp_value_and_terms(held, mip):
    """The LP value HiGHS holds and the Lagrangian terms of its duals over
    ``mip``'s rows, without and with the node cuts' own."""
    solution = held.solution()
    y = np.concatenate(
        [np.asarray(solution.dual_eq)[: mip.n_eq], np.asarray(solution.dual_ineq)]
    )[: mip.A.shape[0]]
    terms = jm.lagrangian_terms(mip.c, mip.A, mip.b, mip.n_eq, y)
    return (
        held.highs.getInfo().objective_function_value,
        terms,
        jm._with_cut_part(terms, held.cut_part()),
    )


def test_heuristic_clock_counts_tree_cut_time_as_nodes():
    bb = types.SimpleNamespace(
        use_tree_cuts=True,
        nodes=100,
        tree_cut_seconds=30.0,
        tree_clock=None,
        tree_seconds=40.0,  # 10 s of plain nodes: 0.1 s a node
        heuristic_fired={},
    )
    bb.tree_elapsed = types.MethodType(jm._BranchAndBound.tree_elapsed, bb)
    bb.heuristic_clock = types.MethodType(jm._BranchAndBound.heuristic_clock, bb)
    clock = bb.heuristic_clock()
    assert clock == pytest.approx(100 + 300)  # 30 s at 0.1 s a node
    due = types.MethodType(jm._BranchAndBound.heuristic_due, bb)
    assert due("rins", 5)  # first time past a multiple
    assert not due("rins", 5)  # not again at the same clock
    assert due("dive", 50)  # each schedule on its own
    bb.use_tree_cuts = False
    assert bb.heuristic_clock() == 100
    assert due("rins", 5) and not due("rins", 7)  # nodes % every, as before


def _scorer(n):
    """The branching-history parts of a ``_BranchAndBound`` over ``n``
    columns, for ``branching_scores``."""
    bb = types.SimpleNamespace(
        branch_children=np.zeros((2, n)),
        branch_cutoffs=np.zeros((2, n)),
        inference_sum=np.zeros((2, n)),
        inference_count=np.zeros((2, n)),
        conflict_score=np.zeros(n),
        conflict_bump=1.0,
        root_lower=np.zeros(n),
        root_upper=np.ones(n),
    )
    for name in ("branching_scores", "bump_conflict", "record_cutoff"):
        setattr(bb, name, types.MethodType(getattr(jm._BranchAndBound, name), bb))
    return bb


def test_branching_scores_put_gains_first_and_break_ties_by_history():
    bb = _scorer(4)
    candidates = np.arange(4)
    # Gains decide: no history outweighs a clearly larger product.
    bb.conflict_score[:] = [100.0, 0.0, 0.0, 0.0]
    bb.branch_children[:] = 1.0
    bb.branch_cutoffs[:, 0] = 1.0
    scores = bb.branching_scores(candidates, np.array([1.0, 2.0, 1.0, -np.inf]))
    assert np.argmax(scores) == 1 and scores[3] == -np.inf
    # Equal gains: conflicts, then cutoffs, then inferences break the tie.
    equal = np.array([1.0, 1.0, 1.0, 1.0])
    assert np.argmax(bb.branching_scores(candidates, equal)) == 0
    bb.conflict_score[:] = 0.0
    bb.branch_cutoffs[:] = 0.0
    bb.branch_cutoffs[1, 2] = 1.0
    assert np.argmax(bb.branching_scores(candidates, equal)) == 2
    bb.branch_cutoffs[:] = 0.0
    bb.inference_sum[:, 3] = 5.0
    bb.inference_count[:] = 1.0
    assert np.argmax(bb.branching_scores(candidates, equal)) == 3


def test_conflict_scores_decay_older_conflicts():
    bb = _scorer(3)
    lower, upper = np.zeros(3), np.ones(3)
    lower[0] = 1.0
    bb.bump_conflict(lower, upper)  # x0 tightened in a closed box
    bb.bump_conflict(lower, upper, cols=np.array([1]))
    assert bb.conflict_score[1] > bb.conflict_score[0] > 0.0 == bb.conflict_score[2]
    bb.record_cutoff((2, True, 0.5, 1.0))
    bb.record_cutoff(None)
    assert bb.branch_cutoffs[0, 2] == 1.0 and bb.branch_cutoffs.sum() == 1.0


def test_branching_history_is_kept(monkeypatch):
    mip = generalised_assignment(3, agents=6, jobs=30)
    searches = []
    real = jm._BranchAndBound.__init__

    def init(self, *args, **kwargs):
        real(self, *args, **kwargs)
        searches.append(self)

    monkeypatch.setattr(jm._BranchAndBound, "__init__", init)
    result = solve(mip, batch_size=1, branching="pseudocost")
    assert_solved(result, mip, reference(mip))
    search = searches[0]
    assert search.branch_children.sum() > 0 and search.inference_count.sum() > 0
    assert np.all(search.branch_cutoffs <= search.branch_children)
    assert search.branch_cutoffs.sum() > 0 and search.conflict_score.sum() > 0


def test_node_lps_stop_at_the_bound_target(monkeypatch):
    mip = generalised_assignment(3, agents=6, jobs=30)
    stopped = []
    real = jm._BranchAndBound._evaluate

    def evaluate(self, node, lower, upper, lp, result):
        # The search's own nodes, not RENS's or RINS's sub-searches'.
        if result.get("stop_reason") == "bound_target" and self.max_nodes is None:
            raw, rigorous = self.bound_over(self.result_terms(result, lower, upper), lower, upper)
            stopped.append((self.round_bound(raw) >= self.cutoff(), rigorous))
        return real(self, node, lower, upper, lp, result)

    monkeypatch.setattr(jm._BranchAndBound, "_evaluate", evaluate)
    # Tree cuts off: a cut refresh between a node's LP and its evaluation
    # zeroes the duals of replaced rows, which can drop a stopped node's
    # bound below the cutoff. Refreshes are paced by wall-clock time, so
    # whether one lands there depends on the machine.
    result = solve(mip, batch_size=1, tree_cuts=False)
    assert_solved(result, mip, reference(mip))
    # Every node LP stopped early closes its node, with a rigorous bound.
    assert stopped and all(closes and rigorous for closes, rigorous in stopped)
    assert result["bound_stops"] >= len(stopped)
    off = solve(mip, batch_size=1, node_bound_stop=False, tree_cuts=False)
    assert_solved(off, mip, reference(mip))
    assert off["bound_stops"] == 0


def test_presolved_mip_start_does_not_dive(monkeypatch):
    # With presolve, a start sets the reduced search's cutoff but is not in
    # its columns. Hybrid node selection used to dive for an incumbent the
    # whole search, which kept the bound near the root's.
    mip = general_integer(5, n=20, m_eq=3, m_ineq=15)  # one presolve reduces
    searches = []
    real = jm._BranchAndBound.__init__

    def init(self, *args, **kwargs):
        real(self, *args, **kwargs)
        searches.append(self)

    monkeypatch.setattr(jm._BranchAndBound, "__init__", init)
    result = solve(mip, presolve=True, mip_start=_optimum(mip), )
    assert result["status"] == "optimal"
    assert abs(result["objective"] - reference(mip)) <= 1e-4 * (1 + abs(reference(mip)))
    top = searches[0]
    assert top.cutoff_from_start and not top.diving()
    # Its sub-MIPs still dive for their own points.
    assert all(not search.cutoff_from_start for search in searches[1:])


def test_conflict_pool_sparsifies_and_propagates():
    n = 4
    is_int = np.ones(n, bool)
    lower, upper = np.zeros(n), np.ones(n)
    # The node fixed x0 = 1 and x1 = 0; x2 and x3 kept their global bounds.
    node_lower, node_upper = lower.copy(), upper.copy()
    node_lower[0], node_upper[1] = 1.0, 0.0
    r = np.array([5.0, -3.0, 1.0, -1.0])
    pool = jm.ConflictPool(n, is_int)
    pool.set_cutoff(3.5)
    # 5x0 − 3x1 + x2 − x3 <= 3.5 becomes 5x0 − 3x1 <= 4.5 (x3 at 1).
    assert pool.add(0.0, r, node_lower, node_upper, lower, upper, "objective")
    idx, coef, offset, objective = pool.rows[0]
    assert list(idx) == [0, 1] and list(coef) == [5.0, -3.0] and objective
    assert offset == pytest.approx(1.0)
    assert pool.propagate(node_lower, node_upper)[2]  # excludes the node
    fixed_up = lower.copy()
    fixed_up[0] = 1.0
    tightened, _, infeasible = pool.propagate(fixed_up, upper)
    assert not infeasible and tightened[1] == 1.0  # x0 = 1 forces x1 = 1
    # A looser cutoff loosens the row: no tightening.
    pool.set_cutoff(10.0)
    assert pool.propagate(fixed_up, upper)[0][1] == 0.0
    # Rejected: a proof that does not exclude the node, or one too long.
    assert not pool.add(0.0, r, node_lower, node_upper, lower, upper, "objective")
    short = jm.ConflictPool(n, is_int, max_nnz=1)
    short.set_cutoff(3.5)
    assert not short.add(0.0, r, node_lower, node_upper, lower, upper, "objective")


def record_conflicts(monkeypatch):
    """Every row a conflict pool keeps, as ``(idx, coef, rhs, cutoff, kind,
    box)``: ``coef · x[idx] <= rhs`` must hold at points in ``box`` with
    ``c·x <= cutoff``. ``box`` is the bounds of the MIP the pool's search
    searched: a RINS or RENS sub-search's pool holds for its sub-MIP only
    (its no-goods follow from fixings outside the sub-MIP's box)."""
    rows = []
    boxes = {}  # id(pool) -> the bounds its search searched
    real = jm.ConflictPool._add_row
    real_init = jm._BranchAndBound.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        if self.conflicts is not None:
            boxes[id(self.conflicts)] = (
                self.original.lower.copy(),
                self.original.upper.copy(),
            )

    def add_row(self, idx, coef, offset, objective, kind, *args):
        kept = real(self, idx, coef, offset, objective, kind, *args)
        if kept:
            cutoff = self.cutoff_value()
            rhs = offset + (cutoff if objective else 0.0)
            box = boxes.get(id(self))
            rows.append((idx.copy(), coef.copy(), rhs, cutoff, kind, box))
        return kept

    monkeypatch.setattr(jm._BranchAndBound, "__init__", init)
    monkeypatch.setattr(jm.ConflictPool, "_add_row", add_row)
    return rows


@pytest.mark.parametrize("make", [general_integer, generalised_assignment])
def test_conflicts_are_valid(make, monkeypatch):
    mip = make()
    rows = record_conflicts(monkeypatch)
    result = solve(mip, conflicts=True)
    assert_solved(result, mip, reference(mip))
    for idx, coef, rhs, cutoff, _, box in rows:
        bound = None if not np.isfinite(cutoff) else cutoff
        most = max_over_mip(mip, idx, coef, objective_at_most=bound, box=box)
        assert most <= rhs + 1e-6 * (1 + abs(rhs))
    if make is general_integer:
        assert rows and result["conflict_pruned"] > 0


def test_nogood_is_minimal(monkeypatch):
    # x0 + x1 <= 1 and x2 + x3 <= 1: fixing x0 = x1 = x2 = 1, x3 = 0 is
    # infeasible because of x0 and x1 alone.
    A_eq, b_eq = no_rows(4)
    lp = jl.LP(
        c=-np.ones(4),
        A_eq=A_eq,
        b_eq=b_eq,
        A_ineq=sp.csc_matrix(np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]])),
        b_ineq=np.ones(2),
        lower_bounds=np.zeros(4),
        upper_bounds=np.ones(4),
    )
    mip = jm.MIP(lp, np.ones(4, bool))
    search = search_for(mip, monkeypatch, conflicts=True)
    search.root_lower, search.root_upper = mip.lower.copy(), mip.upper.copy()
    box = np.array([1.0, 1.0, 1.0, 0.0])
    search.learn_nogood(box, box.copy())
    assert len(search.conflicts) == 1
    idx, coef, offset, objective = search.conflicts.rows[0]
    assert list(idx) == [0, 1] and list(coef) == [1.0, 1.0]
    assert offset == 1.0 and not objective


def test_conflicts_off():
    mip = general_integer()
    result = solve(mip, conflicts=False)
    assert_solved(result, mip, reference(mip))
    assert result["conflicts"] == {"objective": 0, "farkas": 0, "nogood": 0}
    assert result["conflict_pruned"] == 0


# %% Feasibility jump
NEEDS_NUMBA = pytest.mark.skipif(jm._numba_jump() is None, reason="needs numba")
JUMP_BACKENDS = ["jax", pytest.param("numba", marks=NEEDS_NUMBA)]


@pytest.mark.parametrize("backend", JUMP_BACKENDS)
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_feasibility_jump_assignment(seed, backend):
    mip = generalised_assignment(seed)
    for start in (np.zeros(mip.num_variables), lp_point(mip)):
        x, feasible, steps = jm.feasibility_jump(mip, start, backend=backend)
        assert feasible and steps > 0
        assert mip.is_feasible(x)
        np.testing.assert_array_equal(x, np.round(x))


@pytest.mark.parametrize("backend", JUMP_BACKENDS)
def test_feasibility_jump_knapsack_and_mixed(backend):
    mip = knapsack()
    x, feasible, _ = jm.feasibility_jump(mip, np.ones(mip.num_variables), backend=backend)
    assert feasible and mip.is_feasible(x)
    # Facility location has continuous flows: the jump finds openings and
    # flows that meet every row.
    mip = facility_location()
    x, feasible, _ = jm.feasibility_jump(mip, np.zeros(mip.num_variables), backend=backend)
    assert feasible and mip.is_feasible(x)


@NEEDS_NUMBA
@pytest.mark.parametrize(
    "make", [generalised_assignment, general_integer, facility_location, knapsack]
)
def test_numba_jump_scores_match_a_fresh_computation(make):
    # Two-valued variables' scores are updated in place as rows change and
    # weights grow; the rest are recomputed. Either way they must match a
    # computation from scratch at the same point and weights.
    from jaddle import _jump_numba

    mip = make()
    A, row_lower, row_upper, row_tol, _ = jm._jump_rows(mip, 1e-6)
    C = A.tocsc()
    rows = (A.indptr, A.indices, A.data, C.indptr, C.indices, C.data)
    box = (row_lower, row_upper, row_tol, mip.lower, mip.upper, mip.integrality)
    no_improve = (-1, 0.0, 0.0, np.empty(0))
    x = np.clip(np.zeros(mip.num_variables), mip.lower, mip.upper)
    w = np.ones(A.shape[0])
    for moves in (1, 7, 50, 400):
        done, feasible, target, score = _jump_numba.run(x, w, *rows, *box, moves, 3, *no_improve)
        _, _, fresh_target, fresh_score = _jump_numba.run(
            x.copy(), w.copy(), *rows, *box, 0, 3, *no_improve
        )
        # Equal up to rounding: continuous jump values come from activities
        # updated in place, a few ulps from the fresh ones.
        np.testing.assert_allclose(target, fresh_target, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(score, fresh_score, rtol=1e-9, atol=1e-9)
        if feasible:
            break


@NEEDS_NUMBA
@pytest.mark.parametrize("rows", [1, 5, 31, 32, 33, 60])
def test_numba_jump_value_minimises_the_weighted_violation(rows):
    # A continuous column in ``rows`` rows: its jump value must reach the
    # least weighted violation over its bounds, which a convex piecewise
    # linear function takes at one of the rows' interval ends. Up to 32 rows
    # (64 ends) they are sorted in place, past that by np.argsort.
    from jaddle import _jump_numba

    rng = np.random.default_rng(rows)
    a = rng.uniform(0.5, 2.0, rows) * rng.choice([-1.0, 1.0], rows)
    rest = rng.uniform(-3.0, 3.0, rows)  # the other variables' activity
    width = rng.uniform(0.0, 2.0, rows)
    row_lower = rng.uniform(-4.0, 4.0, rows)
    row_upper = row_lower + width
    w = rng.uniform(1.0, 5.0, rows)
    xj = 0.3
    act = rest + a * xj
    col_ptr = np.array([0, rows])
    col_row = np.arange(rows)
    lower, upper = np.array([-10.0]), np.array([10.0])
    buf = np.empty(2 * rows)
    target, change = _jump_numba._jump(
        0, np.array([xj]), act, w, col_ptr, col_row, a, row_lower, row_upper,
        lower, upper, np.zeros(1, bool), buf, buf.copy(),
    )

    def violation(t):
        new = rest + a * t
        return float(w @ (np.maximum(new - row_upper, 0) + np.maximum(row_lower - new, 0)))

    ends = np.clip(np.concatenate([(row_lower - rest) / a, (row_upper - rest) / a]), -10, 10)
    best = min(violation(t) for t in ends)
    assert violation(target) == pytest.approx(best, abs=1e-9)
    assert change == pytest.approx(violation(target) - violation(xj), abs=1e-9)


def test_root_jumps_start_before_the_root_lp_on_every_lane(monkeypatch):
    pytest.importorskip("numba")
    mip = knapsack()
    search = search_for(
        mip, monkeypatch, parallel_heuristics=True, jump_backend="numba", jump_threads=3
    )
    submitted = []
    monkeypatch.setattr(
        search.workers, "submit",
        lambda lane, kind, fn, *a, **k: submitted.append((lane, kind, fn, k["seed"])),
    )
    assert search.launch_root_jumps(mip.lower, mip.upper)
    assert [s[:3] for s in submitted] == [
        (lane, "jump", jm._early_jump_job) for lane in ("jump", "jump1", "jump2")
    ]
    assert len({s[3] for s in submitted}) == 3  # a seed each
    # The first incumbent stops them.
    assert not search.early_jump.is_set()
    x = np.zeros(mip.num_variables)
    search.try_incumbent(x, "J", None)
    assert search.early_jump.is_set()
    # None with an incumbent, or with root_jump_seconds=0.
    submitted.clear()
    assert not search.launch_root_jumps(mip.lower, mip.upper) and not submitted
    off = search_for(
        mip, monkeypatch, parallel_heuristics=True, jump_backend="numba",
        root_jump_seconds=0,
    )
    assert not off.launch_root_jumps(mip.lower, mip.upper)


@NEEDS_NUMBA
def test_early_jump_job_stops_when_done():
    # An infeasible row: the jump would run to its deadline. With ``done``
    # set it stops after its first chunk.
    import threading

    n = 6
    lp = jl.LP(
        c=np.arange(n, dtype=float),
        A_eq=sp.csc_matrix(2.0 * np.ones((1, n))),
        b_eq=np.array([3.0]),
        A_ineq=sp.csc_matrix((0, n)),
        b_ineq=np.zeros(0),
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
    )
    mip = jm.MIP(lp, np.ones(n, bool))
    done = threading.Event()
    done.set()
    started = time.time()
    _, feasible, steps = jm._early_jump_job(
        lambda: False, done, mip, np.zeros(n), deadline=time.time() + 30, backend="numba"
    )
    assert not feasible and steps <= 1000 and time.time() - started < 10
    # Skipped outright once there is an incumbent.
    assert jm._early_jump_job(lambda: True, done, mip, np.zeros(n)) == (None, False, None)


@NEEDS_NUMBA
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_improving_jump_finds_better_points(seed):
    mip = generalised_assignment(seed)
    first, feasible, _ = jm.feasibility_jump(mip, np.zeros(mip.num_variables), seed=seed)
    assert feasible
    x, feasible, _ = jm.feasibility_jump(
        mip, np.zeros(mip.num_variables), seed=seed, improve=True, max_iterations=20_000
    )
    assert feasible and mip.is_feasible(x)
    _, best = reference_point(mip)
    assert best - 1e-6 <= mip.c @ x < mip.c @ first - 1e-6
    # Below a cutoff: every point it returns meets it.
    cutoff = 0.5 * (float(mip.c @ first) + best)
    y, feasible, _ = jm.feasibility_jump(
        mip, first, seed=seed, cutoff=cutoff, improve=True, max_iterations=20_000
    )
    assert feasible and mip.is_feasible(y)
    assert mip.c @ y <= cutoff + 1e-6 * (1 + abs(cutoff))


def test_improving_jump_needs_numba():
    with pytest.raises(ValueError, match="improve"):
        jm.feasibility_jump(knapsack(), np.zeros(15), improve=True, backend="jax")


@NEEDS_NUMBA
def test_root_improving_jump_in_the_search(monkeypatch):
    calls = []
    real = jm._BranchAndBound.improve_jump

    def recorded(self, *args, **kwargs):
        calls.append(self.incumbent[0])
        return real(self, *args, **kwargs)

    monkeypatch.setattr(jm._BranchAndBound, "improve_jump", recorded)
    mip = generalised_assignment(2)
    result = solve(
        mip, max_nodes=1, cuts=False, fix_and_propagate=False, bulk_rounding=False,
        feasibility_pump=False, heuristic_seconds=2.0,
    )
    assert calls  # the jump's point, then the time left to improve on it
    assert mip.is_feasible(result["x"])
    assert result["objective"] <= mip.sign * (calls[0] + mip.offset) + 1e-9


def test_jump_backend_is_checked():
    with pytest.raises(ValueError, match="jump backend"):
        jm.feasibility_jump(knapsack(), np.zeros(15), backend="cuda")
    with pytest.raises(ValueError, match="jump backend"):
        solve(knapsack(), jump_backend="cuda")


@NEEDS_NUMBA
def test_numba_jump_takes_no_device_data():
    mip = knapsack()
    data = jm._jump_data(mip, mip.lower, mip.upper, 1e-6)
    with pytest.raises(ValueError, match="device"):
        jm.feasibility_jump(mip, np.zeros(15), data=data, backend="numba")


@pytest.mark.parametrize("backend", JUMP_BACKENDS)
def test_feasibility_jump_gives_up_on_infeasible(backend):
    n = 6
    lp = jl.LP(
        c=np.arange(n, dtype=float),
        A_eq=sp.csc_matrix(2.0 * np.ones((1, n))),
        b_eq=np.array([3.0]),
        A_ineq=sp.csc_matrix((0, n)),
        b_ineq=np.zeros(0),
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
    )
    _, feasible, steps = jm.feasibility_jump(
        jm.MIP(lp, np.ones(n, bool)), np.zeros(n), max_iterations=2000, backend=backend
    )
    assert not feasible and steps == 2000


@pytest.mark.parametrize("backend", JUMP_BACKENDS)
def test_feasibility_jump_in_the_search(backend):
    mip = generalised_assignment(2)
    result = solve(
        mip, max_nodes=1, cuts=False, fix_and_propagate=False, bulk_rounding=False,
        feasibility_pump=False, jump_backend=backend,
    )
    assert result["x"] is not None and mip.is_feasible(result["x"])
    assert result["history"][0]["source"] == "J"


@pytest.mark.parametrize("backend", JUMP_BACKENDS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_feasibility_jump_with_a_cutoff_finds_only_better_points(seed, backend):
    mip = generalised_assignment(seed)
    first, feasible, _ = jm.feasibility_jump(
        mip, np.zeros(mip.num_variables), backend=backend
    )
    assert feasible
    _, best = reference_point(mip)
    worst = float(mip.c @ first)
    assert worst > best + 1e-6  # the plain jump's point is not optimal
    cutoff = 0.5 * (worst + best)
    x, feasible, _ = jm.feasibility_jump(mip, lp_point(mip), cutoff=cutoff, backend=backend)
    assert feasible and mip.is_feasible(x)
    assert mip.c @ x <= cutoff + 1e-6 * (1 + abs(cutoff))


@pytest.mark.parametrize("backend", JUMP_BACKENDS)
def test_feasibility_jump_gives_up_below_the_optimum(backend):
    mip = generalised_assignment(0)
    _, best = reference_point(mip)
    _, feasible, steps = jm.feasibility_jump(
        mip, lp_point(mip), cutoff=best - 1e-2 * (1 + abs(best)), max_iterations=2000,
        backend=backend,
    )
    assert not feasible and steps == 2000


def test_root_bulk_rounding_runs_without_fix_and_propagate(monkeypatch):
    # Bulk rounding used to run at the root only after the classic
    # fix-and-propagate, so fix_and_propagate=False turned it off too.
    calls = []
    real = jm._BranchAndBound.bulk_round

    def recorded(self, *args, **kwargs):
        calls.append(1)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(jm._BranchAndBound, "bulk_round", recorded)
    mip = generalised_assignment(2)
    solve(
        mip, max_nodes=1, cuts=False, fix_and_propagate=False, feasibility_jump=False,
        feasibility_pump=False,
    )
    assert calls


def test_root_jump_starts_share_the_budget(monkeypatch):
    # Each root jump start gets an equal share of the time left (one more
    # share is kept for the pump), so the first cannot take all of it.
    budgets = []

    def recorded(self, start, lower, upper, warm, deadline, seed):
        budgets.append(deadline - time.time())

    monkeypatch.setattr(jm._BranchAndBound, "jump", recorded)
    solve(
        generalised_assignment(2), max_nodes=1, cuts=False, fix_and_propagate=False,
        bulk_rounding=False, feasibility_pump=True, pump_iterations=0,
        heuristic_seconds=30.0,
    )
    assert len(budgets) == 2
    assert 5.0 < budgets[0] < 10.5  # a third of 30 s
    assert 14.0 < budgets[1] < 15.5  # half of the time left (the stub took none)


def test_jump_data_with_a_moved_cutoff_matches_a_fresh_build():
    mip = generalised_assignment(1)
    start = lp_point(mip)
    _, best = reference_point(mip)
    old = jm._jump_data(mip, mip.lower, mip.upper, 1e-6, cutoff=best + 100.0)
    moved = jm._jump_with(old, mip.lower, mip.upper, cutoff=best + 1.0)
    reused = jm.feasibility_jump(mip, start, cutoff=best + 1.0, data=moved, seed=3)
    fresh = jm.feasibility_jump(mip, start, cutoff=best + 1.0, seed=3, backend="jax")
    np.testing.assert_array_equal(reused[0], fresh[0])
    assert reused[1:] == fresh[1:]


# %% GPU tree heuristics
def tree_heuristic_options(**options):
    """Tree heuristics at every node, with the other tree heuristics off."""
    return dict(
        dict(
            tree_heuristics=True,
            tree_heuristic_every=1,
            tree_heuristic_share=1.0,
            rins_seconds=0,
            rens_seconds=0,
            heuristic_every=0,
        ),
        **options,
    )


def test_tree_heuristics_in_the_search():
    mip = generalised_assignment(2)
    result = solve(mip, **tree_heuristic_options())
    assert_solved(result, mip, reference(mip))
    assert result["tree_heuristic_calls"] > 0
    assert result["tree_heuristic_seconds"] > 0
    assert result["tree_heuristic_finds"] <= result["tree_heuristic_calls"]
    off = solve(mip, **tree_heuristic_options(tree_heuristics=False))
    assert off["tree_heuristic_calls"] == 0 and off["tree_heuristic_finds"] == 0


def test_tree_heuristics_need_an_incumbent_and_stay_out_of_sub_searches(monkeypatch):
    calls = []
    real = jm._BranchAndBound.tree_heuristics

    def recorded(self, *args):
        calls.append((self.incumbent is not None, np.isfinite(self.initial_cutoff)))
        return real(self, *args)

    monkeypatch.setattr(jm._BranchAndBound, "tree_heuristics", recorded)
    mip = generalised_assignment(2)
    result = solve(
        mip,
        **tree_heuristic_options(rins_seconds=5, rins_every=5),
    )
    assert_solved(result, mip, reference(mip))
    assert calls
    assert all(has_incumbent for has_incumbent, _ in calls)
    assert not any(in_sub_search for _, in_sub_search in calls)


def test_tree_heuristics_skip_a_rounding_already_tried(monkeypatch):
    mip = generalised_assignment(2)
    search = search_for(mip, monkeypatch, **tree_heuristic_options())
    x = lp_point(mip)
    search.root_lower, search.root_upper = mip.lower.copy(), mip.upper.copy()
    search.incumbent = (1e9, np.zeros(mip.num_variables))
    search.use_jump = False
    rounds = []
    monkeypatch.setattr(search, "bulk_round", lambda *a, **k: rounds.append(1) or [])
    search.tree_heuristics(x, mip.lower, mip.upper, None)
    search.tree_heuristics(x, mip.lower, mip.upper, None)
    assert len(rounds) == 1 and search.tree_heuristic_calls == 1


def test_tree_heuristics_stop_at_their_share_not_the_solve_deadline(monkeypatch):
    # One bulk rounding of a large model (fiball, 32k integers) ran on to
    # the solve's deadline at the first node.
    mip = generalised_assignment(2)
    search = search_for(mip, monkeypatch, **tree_heuristic_options(max_seconds=3600))
    x = lp_point(mip)
    search.root_lower, search.root_upper = mip.lower.copy(), mip.upper.copy()
    search.incumbent = (1e9, np.zeros(mip.num_variables))
    search.use_jump = False
    search.tree_heuristic_share = 0.05
    search.start = time.time() - 100.0  # 100 s in: a 5 s share
    search.tree_heuristic_seconds = 2.0  # 3 s of it left
    deadlines = []
    monkeypatch.setattr(
        search, "bulk_round", lambda starts, lo, up, deadline: deadlines.append(deadline) or []
    )
    before = time.time()
    search.tree_heuristics(x, mip.lower, mip.upper, None)
    assert before + 2.9 <= deadlines[0] <= time.time() + 3.1
    # With the share used up, a call still gets the minimum.
    search.tree_heuristic_seconds = 10.0
    search.tree_heuristics(x + 0.5, mip.lower, mip.upper, None)
    assert deadlines[1] <= time.time() + jm._TREE_HEURISTIC_MIN_SECONDS + 0.1


@pytest.mark.parametrize(
    "options", [dict(tree_heuristic_every=0), dict(tree_heuristic_share=-0.1)]
)
def test_rejects_bad_tree_heuristic_options(options):
    with pytest.raises(ValueError):
        solve(knapsack(), **options)


# %% Node crossover
def pick_one(n=6):
    """Maximise the sum of n binaries, at most one of them 1: PDHG lands in
    the middle of the optimal face (1/n each), every vertex is integral."""
    lp = jl.LP(
        c=-np.ones(n),
        A_eq=sp.csc_matrix((0, n)),
        b_eq=np.zeros(0),
        A_ineq=sp.csc_matrix(np.ones((1, n))),
        b_ineq=np.array([1.0]),
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
    )
    return jm.MIP(lp, np.ones(n, bool))


# Every heuristic off: only node LP points give incumbents.
NO_HEURISTICS = dict(
    heuristic_every=0,
    fix_and_propagate=False,
    feasibility_jump=False,
    feasibility_pump=False,
    bulk_rounding=False,
    rens_seconds=0,
    rins_seconds=0,
    probing=False,
    cuts=False,
    round_repair=False,
)


def test_node_crossover_turns_a_face_point_into_an_integral_vertex():
    mip = pick_one()
    result = solve(mip, node_crossover=True, **NO_HEURISTICS)
    assert_solved(result, mip, reference(mip))
    assert result["nodes"] == 1  # the root vertex is integral: no branching
    assert [h["source"] for h in result["history"]] == ["T"]
    assert result["node_crossovers"] == result["node_crossover_vertices"] == 1
    assert result["node_crossover_integral"] == 1
    off = solve(mip, **NO_HEURISTICS)
    assert off["nodes"] > 1 and off["node_crossovers"] == 0


@pytest.mark.parametrize("batch_size", [1, 4])
def test_node_crossover_in_the_search(batch_size):
    mip = general_integer()
    result = solve(mip, node_crossover=True, batch_size=batch_size)
    assert_solved(result, mip, reference(mip))
    assert 0 < result["node_crossover_vertices"] <= result["node_crossovers"]
    assert result["node_crossover_seconds"] > 0


def test_node_crossover_every_and_failed_crossovers(monkeypatch):
    mip = general_integer()
    # Without root cuts: with them the tree is too small to cross over in.
    sparse = solve(mip, node_crossover=True, node_crossover_every=1000, cuts=False)
    assert_solved(sparse, mip, reference(mip))
    assert sparse["node_crossovers"] <= 1  # only the node count 0 qualifies
    # A crossover HiGHS cannot finish leaves the PDHG point in place.
    monkeypatch.setattr(
        jm.jl, "crossover", lambda lp, solution, highs_options=None: {"converged": False}
    )
    result = solve(mip, node_crossover=True, cuts=False)
    assert_solved(result, mip, reference(mip))
    assert result["node_crossovers"] > 0 and result["node_crossover_vertices"] == 0


@pytest.mark.parametrize(
    "options", [dict(node_crossover_every=0), dict(node_crossover_seconds=0)]
)
def test_rejects_bad_node_crossover_options(options):
    with pytest.raises(ValueError):
        solve(knapsack(), **options)


# %% Repair
def broken_assignment(seed=0):
    """A generalised assignment MIP and an integer point that breaks some
    of its rows: an optimal assignment with three jobs moved to agent 0."""
    mip = generalised_assignment(seed)
    best = milp(
        mip.c,
        constraints=LinearConstraint(mip.A.toarray(), mip.row_lower, mip.row_upper),
        integrality=mip.integrality.astype(int),
        bounds=Bounds(mip.lower, mip.upper),
    )
    x = np.round(best.x)
    jobs = 20
    x = x.reshape(5, jobs)
    for job in range(3):  # three jobs piled onto agent 0
        x[:, job] = 0.0
        x[0, job] = 1.0
    return mip, x.ravel()


def test_repair_fixes_a_near_miss():
    mip, x = broken_assignment()
    assert not mip.is_feasible(x)
    repaired = jm.repair(mip, x, max_seconds=60)
    assert repaired is not None and mip.is_feasible(repaired)
    # Only variables near the broken rows moved.
    assert np.count_nonzero(repaired != x) < mip.num_variables


def test_repair_returns_a_feasible_point_unchanged():
    mip, x = broken_assignment()
    good = jm.repair(mip, x, max_seconds=60)
    np.testing.assert_array_equal(jm.repair(mip, good, max_seconds=60), good)


def test_repair_gives_up_when_the_neighbourhood_is_infeasible():
    # 2 (x1 + ... + x6) = 3 has no integer point at all.
    n = 6
    lp = jl.LP(
        c=np.arange(n, dtype=float),
        A_eq=sp.csc_matrix(2.0 * np.ones((1, n))),
        b_eq=np.array([3.0]),
        A_ineq=sp.csc_matrix((0, n)),
        b_ineq=np.zeros(0),
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
    )
    assert jm.repair(jm.MIP(lp, np.ones(n, bool)), np.zeros(n), max_seconds=30) is None


def test_heuristic_effort_zero_turns_rins_off():
    mip = general_integer()
    result = solve(mip, heuristic_effort=0.0)
    assert_solved(result, mip, reference(mip))
    assert result["rins_calls"] == 0
    assert solve(mip)["rins_calls"] > 0  # on by default


def test_rins_node_limit_adapts(monkeypatch):
    mip = general_integer()
    search = search_for(mip, monkeypatch)
    search.incumbent = (float(mip.c @ np.zeros(mip.num_variables)), np.zeros(mip.num_variables))
    calls = iter([None, None])
    monkeypatch.setattr(search, "jaddle_rins", lambda x, seconds, warm=None: next(calls))
    nodes = search.rins_nodes
    search.rins(np.zeros(mip.num_variables), None)
    assert search.rins_nodes == max(50, nodes // 2)  # a failed call halves it
    assert search.submip_calls == 1


@pytest.mark.parametrize("plunge_depth", [0, 8])
def test_plunging_keeps_the_optimum(plunge_depth):
    mip = generalised_assignment(2)
    assert_solved(solve(mip, plunge_depth=plunge_depth), mip, reference(mip))


def test_rins_in_the_search():
    mip = generalised_assignment(2)
    result = solve(mip, rins_seconds=5, rins_every=5)
    assert_solved(result, mip, reference(mip))


# %% Batched node solves
@pytest.mark.parametrize(
    "make", [knapsack, general_integer, facility_location, generalised_assignment]
)
def test_batched_search_matches_reference(make):
    mip = make()
    # Without heuristics: with an incumbent from them, root reduced-cost
    # fixing can close the search before it branches.
    result = solve(mip, batch_size=4, gomory_cuts=False, **no_heuristics())
    assert_solved(result, mip, reference(mip))
    assert result["batches"] > 0


def test_batch_larger_than_the_tree():
    # Far more slots than open nodes: every batch is padded.
    mip = facility_location()
    # Without heuristics, so root reduced-cost fixing cannot end it at the root.
    result = solve(mip, batch_size=64, gomory_cuts=False, **no_heuristics())
    assert_solved(result, mip, reference(mip))
    assert 0 < result["batches"] <= result["nodes"]


def test_batched_search_limits_and_infeasibility():
    mip = general_integer(n=12, m_ineq=8)
    ref = reference(mip)
    # The root and one batch cut short by the limit. At 6 nodes a quick
    # incumbent could close the gap (10 nodes prove it), so a fast runner
    # reported "optimal".
    # Without root cuts, which solve this MIP at the root.
    result = solve(mip, batch_size=4, max_nodes=3, cuts=False)
    assert result["status"] == "node_limit" and result["nodes"] <= 3
    assert result["bound"] <= ref + 1e-6 * (1 + abs(ref))

    n = 6
    lp = jl.LP(
        c=np.arange(n, dtype=float),
        A_eq=sp.csc_matrix(2.0 * np.ones((1, n))),
        b_eq=np.array([3.0]),
        A_ineq=sp.csc_matrix((0, n)),
        b_ineq=np.zeros(0),
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
    )
    infeasible = jm.MIP(lp, np.ones(n, bool))
    assert solve(infeasible, batch_size=4)["status"] == "infeasible"


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("make", [knapsack, general_integer, generalised_assignment])
def test_logged_bound_never_falls(monkeypatch, make, batch_size):
    # Nodes in no list fell out of the global bound, so the logged bound read
    # too high, then fell back: the root during the root heuristics (an
    # incumbent showed a 0% gap) and a batch's members not yet evaluated.
    logged = {}
    real = jm._BranchAndBound.log

    def recorded(self, source, bound, force=False):
        # The search's own log: RENS and RINS run sub-searches (with a node
        # limit), which log nothing.
        if self.max_nodes is None:
            logged.setdefault(id(self), []).append(
                self.global_bound() if bound is None else bound
            )
        return real(self, source, bound, force)

    monkeypatch.setattr(jm._BranchAndBound, "log", recorded)
    mip = make()
    result = solve(mip, batch_size=batch_size, gomory_cuts=False)
    assert_solved(result, mip, reference(mip))
    assert logged
    for bounds in logged.values():
        finite = [b for b in bounds if np.isfinite(b)]
        assert all(b <= a + 1e-9 * (1 + abs(a)) for a, b in zip(finite[1:], finite))


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="batch_size"):
        solve(knapsack(), batch_size=0)


# %% Bounds from duals: child bounds, reduced-cost fixing, integral objective
def scaled_objective(mip, factor):
    """``mip`` with its costs multiplied by ``factor``."""
    A_eq, A_ineq = mip.A[: mip.n_eq], mip.A[mip.n_eq :]
    lp = jl.LP(
        c=factor * mip.c,
        A_eq=A_eq.tocsc(),
        b_eq=mip.b[: mip.n_eq],
        A_ineq=A_ineq.tocsc(),
        b_ineq=mip.b[mip.n_eq :],
        lower_bounds=mip.lower,
        upper_bounds=mip.upper,
    )
    return jm.MIP(lp, mip.integrality)


def optimum(mip):
    """An optimal point of ``mip``, from SciPy's HiGHS."""
    result = milp(
        mip.c,
        constraints=LinearConstraint(mip.A.toarray(), mip.row_lower, mip.row_upper),
        integrality=mip.integrality.astype(int),
        bounds=Bounds(mip.lower, mip.upper),
    )
    assert result.success
    return result.x


def lp_duals(mip):
    """Optimal duals of ``mip``'s LP relaxation in Jaddle's convention."""
    from scipy.optimize import linprog

    lp = mip.lp.to_scipy()
    bounds = [(lo, None if np.isinf(hi) else hi) for lo, hi in zip(mip.lower, mip.upper)]
    ref = linprog(
        mip.c,
        A_ub=lp.A_ineq,
        b_ub=lp.b_ineq,
        A_eq=lp.A_eq if lp.A_eq.shape[0] else None,
        b_eq=lp.b_eq if lp.A_eq.shape[0] else None,
        bounds=bounds,
        method="highs",
    )
    eq = ref.eqlin.marginals if lp.A_eq.shape[0] else np.zeros(0)
    return -np.concatenate([eq, ref.ineqlin.marginals])


def test_objective_step():
    ints = np.array([True, True, True, False])
    assert jm.objective_step(np.array([2.0, -4.0, 6.0, 0.0]), ints) == 2.0
    assert jm.objective_step(np.array([0.5, 1.5, 0.0, 0.0]), ints) == 0.5
    assert jm.objective_step(np.array([0.25, 1.0, 0.0, 0.0]), ints) == 0.25
    # A cost on a continuous variable, an irrational ratio, no costs at all.
    assert jm.objective_step(np.array([1.0, 2.0, 0.0, 0.1]), ints) is None
    assert jm.objective_step(np.array([1.0, np.sqrt(2.0), 0.0, 0.0]), ints) is None
    assert jm.objective_step(np.zeros(4), ints) is None


def test_box_bound_matches_lagrangian_bound():
    mip = general_integer()
    y = np.random.default_rng(0).standard_normal(mip.A.shape[0])
    constant, r = jm.lagrangian_terms(mip.c, mip.A, mip.b, mip.n_eq, y)
    assert jm.box_bound(constant, r, mip.lower, mip.upper) == jm.lagrangian_bound(
        mip.c, mip.A, mip.b, mip.n_eq, mip.lower, mip.upper, y
    )
    # A smaller box gives a bound at least as high.
    upper = mip.upper.copy()
    upper[0] = 2.0
    assert (
        jm.box_bound(constant, r, mip.lower, upper)[0]
        >= jm.box_bound(constant, r, mip.lower, mip.upper)[0]
    )


def test_reduced_cost_tightening_textbook():
    r = np.array([3.0, -2.0, 1.0, 0.0])
    ints = np.array([True, True, False, True])
    lower, upper = np.zeros(4), np.full(4, 10.0)
    # cᵀx >= 10 + 3·x0 and cᵀx >= 10 + 2·(10 − x1): below 16 needs x0 <= 2 and
    # x1 >= 7. The continuous x2 and the zero reduced cost of x3 stay.
    lo, up, changed = jm.reduced_cost_tightening(r, 10.0, 16.0, lower, upper, ints)
    assert changed
    np.testing.assert_array_equal(lo, [0.0, 7.0, 0.0, 0.0])
    np.testing.assert_array_equal(up, [2.0, 10.0, 10.0, 10.0])
    assert lower[1] == 0.0 and upper[0] == 10.0  # the inputs are not changed
    # No incumbent (an infinite cutoff): nothing to do.
    assert not jm.reduced_cost_tightening(r, 10.0, np.inf, lower, upper, ints)[2]


@pytest.mark.parametrize("make", [knapsack, general_integer, generalised_assignment])
def test_reduced_cost_tightening_keeps_better_solutions(make):
    mip = make()
    x_star = optimum(mip)
    cutoff = mip.c @ x_star + 1e-9 * (1 + abs(mip.c @ x_star))
    y = lp_duals(mip)
    rng = np.random.default_rng(0)
    tightened = 0
    for noise in [0.0] + [0.01, 0.1, 1.0] * 5:
        y_trial = y + noise * rng.standard_normal(y.shape)
        constant, r = jm.lagrangian_terms(mip.c, mip.A, mip.b, mip.n_eq, y_trial)
        bound, rigorous = jm.box_bound(constant, r, mip.lower, mip.upper)
        assert rigorous
        lo, up, changed = jm.reduced_cost_tightening(
            r, bound, cutoff, mip.lower, mip.upper, mip.integrality
        )
        # The optimum (below the cutoff) stays in the box.
        assert np.all(lo <= x_star + 1e-9) and np.all(x_star <= up + 1e-9)
        tightened += changed
    assert tightened > 0


@pytest.mark.parametrize("batch_size", [1, 4])
def test_bounds_from_duals_in_the_search(batch_size):
    mip = generalised_assignment(1)
    # Without the root's strongest cuts there is a tree to fix bounds in.
    result = solve(
        mip, batch_size=batch_size, gomory_cuts=False, aggregate_integers=False
    )
    assert_solved(result, mip, reference(mip))
    assert result["objective_step"] == 1.0
    assert result["reduced_cost_tightenings"] > 0
    # The bound is rounded to the step.
    assert result["bound"] == round(result["bound"])


@pytest.mark.parametrize("factor", [0.5, -0.25])
def test_fractional_objective_step(factor):
    mip = scaled_objective(knapsack(n=25), factor)
    result = solve(mip)
    assert result["objective_step"] == abs(factor)
    assert_solved(result, mip, reference(mip))


def test_integral_objective_with_offset_and_maximisation():
    # Values are integers in the model's own sense, with a fractional offset.
    n = 12
    rng = np.random.default_rng(3)
    w = rng.integers(5, 30, n).astype(float)
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = n, 1
    lp.col_cost_ = list(w + rng.integers(0, 10, n))
    lp.col_lower_, lp.col_upper_ = [0.0] * n, [1.0] * n
    lp.row_lower_, lp.row_upper_ = [-highspy.kHighsInf], [w.sum() / 3]
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = list(range(n + 1))
    lp.a_matrix_.index_ = [0] * n
    lp.a_matrix_.value_ = list(w)
    lp.integrality_ = [highspy.HighsVarType.kInteger] * n
    lp.sense_ = highspy.ObjSense.kMaximize
    lp.offset_ = 0.3
    highs.passModel(lp)
    mip = jm.from_highs(highs)
    assert mip.maximise
    result = solve(mip)
    assert result["objective_step"] == 1.0
    assert_solved(result, mip, reference(mip))


# %% Incumbent repair
def search_for(mip, monkeypatch, **options):
    """The ``_BranchAndBound`` that ``solve`` builds, before it runs."""
    monkeypatch.setattr(jm._BranchAndBound, "run", lambda self: self)
    return solve(mip, **options)


def test_node_points_are_rounded_in_the_background(monkeypatch):
    """With ``parallel_heuristics``, node LP points go to the background
    ``"round"`` lane (fix-and-propagate), even with the main thread's
    ``fix_and_propagate`` off; ``parallel_rounding=False`` stops them."""
    in_tree = []
    real_submit = jm._HeuristicWorkers.submit
    real_evaluate = jm._BranchAndBound._evaluate

    def submit(self, lane, kind, *args, **kwargs):
        if kind == "fixprop" and lane == "round":
            in_tree.append(evaluating[-1] if evaluating else False)
        return real_submit(self, lane, kind, *args, **kwargs)

    evaluating = []

    def _evaluate(self, node, *args, **kwargs):
        evaluating.append(node.depth > 0)
        try:
            return real_evaluate(self, node, *args, **kwargs)
        finally:
            evaluating.pop()

    monkeypatch.setattr(jm._HeuristicWorkers, "submit", submit)
    monkeypatch.setattr(jm._BranchAndBound, "_evaluate", _evaluate)
    mip = generalised_assignment(2)
    ref = reference(mip)
    options = dict(parallel_heuristics=True, jump_backend="numba", max_nodes=60)
    result = solve(mip, fix_and_propagate=False, **options)
    assert result["status"] in ("optimal", "node_limit")
    assert result["background_roundings"] == len(in_tree) > 0
    assert any(in_tree)  # at nodes below the root
    # Node jobs carry a jump for when their rounding fails.
    assert result["background_jumps"] > 0
    if result["status"] == "optimal":
        assert_solved(result, mip, ref, rigorous=False)
    in_tree.clear()
    off = solve(mip, parallel_rounding=False, **options)
    assert off["background_roundings"] == 0 and not in_tree


def test_node_heuristic_job_jumps_when_rounding_fails():
    """The background node job: fix-and-propagate, and when that fails, an
    improving jump from the point for the given seconds (marked ``"J"``)."""
    pytest.importorskip("numba")
    mip = knapsack()
    x = np.full(mip.num_variables, 0.5)
    rows = jm.BoundPropagator(mip.A, mip.row_lower, mip.row_upper, mip.integrality)
    point, found, source = jm._node_heuristic_job(
        x, mip.lower, mip.upper, mip.integrality, rows
    )
    assert source == "P" and (found == (point is not None))

    def rejects(lower, upper, max_passes=5):  # every rounding fails
        return lower, upper, True

    point, found, source = jm._node_heuristic_job(
        x, mip.lower, mip.upper, mip.integrality, rejects
    )
    assert (point, found, source) == (None, False, "P")  # no jump asked for
    jump = dict(mip=mip, seconds=1.0, tol=1e-6, seed=0, cutoff=None)
    point, found, source = jm._node_heuristic_job(
        x, mip.lower, mip.upper, mip.integrality, rejects, jump=jump
    )
    assert source == "J" and found
    assert mip.is_feasible(point, 1e-6)


def test_background_jumps_fill_every_lane_with_restarts(monkeypatch):
    """With ``jump_threads`` lanes, a new incumbent gets one improving
    slice and every other idle lane a restart (``jump_restarts``) from a
    perturbed incumbent or a node point; none sits idle."""
    pytest.importorskip("numba")
    mip = knapsack()
    search = search_for(
        mip, monkeypatch, parallel_heuristics=True, jump_backend="numba", jump_threads=3
    )
    assert search.workers.LANES == ("jump", "jump1", "jump2", "round")
    submitted = []
    monkeypatch.setattr(
        search.workers, "submit", lambda lane, kind, *a, **k: submitted.append((lane, kind))
    )
    x = np.zeros(mip.num_variables)  # the empty knapsack: feasible
    search.incumbent = (float(mip.c @ x), x)
    search.schedule_improve_jump()
    assert sorted(kind for _, kind in submitted) == ["jump", "jumprestart", "jumprestart"]
    assert {lane for lane, _ in submitted} == {"jump", "jump1", "jump2"}
    # The same incumbent again: restarts only.
    submitted.clear()
    search.schedule_improve_jump()
    assert [kind for _, kind in submitted] == ["jumprestart"] * 3
    off = search_for(
        mip, monkeypatch, parallel_heuristics=True, jump_backend="numba",
        jump_threads=2, jump_restarts=False,
    )
    submitted.clear()
    monkeypatch.setattr(
        off.workers, "submit", lambda lane, kind, *a, **k: submitted.append((lane, kind))
    )
    off.incumbent = (float(mip.c @ x), x)
    off.schedule_improve_jump()
    off.schedule_improve_jump()
    assert [kind for _, kind in submitted] == ["jump"]
    with pytest.raises(ValueError, match="jump_threads"):
        solve(mip, jump_threads=0)


def test_jump_restart_starts_move_part_of_the_incumbent(monkeypatch):
    mip = general_integer()
    search = search_for(mip, monkeypatch)
    x = np.clip(np.round(_optimum(mip)), mip.lower, mip.upper)
    ints = np.flatnonzero(mip.integrality)
    for k in range(4):
        search.jump_slices, search.jump_restarts = k, 0  # even: perturbed
        start = search.jump_restart_start(x, mip.lower, mip.upper)
        assert np.all(start >= mip.lower) and np.all(start <= mip.upper)
        moved = np.flatnonzero(start != x)
        assert 1 <= moved.size <= max(1, int(0.2 * ints.size))
        assert np.isin(moved, ints).all()
    search.recent_points.append(np.full(mip.num_variables, 0.5))
    search.jump_restarts = 1  # odd: a node point
    assert np.all(search.jump_restart_start(x, mip.lower, mip.upper) == 0.5)


def test_near_integral_points_are_tried_once(monkeypatch):
    """A node point whose integers are all within
    ``near_integral_tolerance`` of integers (though not within
    ``integrality_tolerance``) is rounded and tried as an incumbent
    (``"T"``), once per rounding; ``0`` turns it off."""
    mip = knapsack()
    best = _optimum(mip)
    near = np.where(best > 0.5, best - 0.004, best + 0.004)
    tried = []
    real = jm._BranchAndBound.try_incumbent

    def try_incumbent(self, x, source, warm):
        tried.append(source)
        return real(self, x, source, warm)

    monkeypatch.setattr(jm._BranchAndBound, "try_incumbent", try_incumbent)
    search = search_for(mip, monkeypatch)
    xi = near[search.is_int]
    search.try_near_integral(near, xi, None)
    assert tried == ["T"] and search.incumbent is not None
    assert search.incumbent[0] == pytest.approx(mip.c @ best)
    assert search.near_integral_finds == 1
    search.try_near_integral(near, xi, None)  # the same rounding: skipped
    assert tried == ["T"]
    far = np.where(best > 0.5, best - 0.3, best + 0.3)
    search.try_near_integral(far, far[search.is_int], None)  # not near enough
    assert tried == ["T"]
    off = search_for(mip, monkeypatch, near_integral_tolerance=0)
    off.try_near_integral(near, xi, None)
    assert tried == ["T"] and off.incumbent is None
    with pytest.raises(ValueError, match="near_integral_tolerance"):
        solve(mip, near_integral_tolerance=-1)


def count_lp_solves(monkeypatch, fake=None):
    """Count ``jl.solve`` calls (made by ``fake`` instead, if given)."""
    calls = []
    real = jl.solve

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return (fake or real)(*args, **kwargs)

    monkeypatch.setattr(jl, "solve", counted)
    return calls


def test_repair_skipped_when_propagation_proves_the_fixed_box_infeasible(monkeypatch):
    # Every facility closed: the demand rows can't be met by any flows.
    mip = facility_location()
    search = search_for(mip, monkeypatch)
    calls = count_lp_solves(monkeypatch)
    assert not search.try_incumbent(np.zeros(mip.num_variables), "H", None)
    assert calls == []
    assert search.lp_repairs == 0 and search.lp_repairs_skipped == 1


def test_failed_repair_is_not_repeated(monkeypatch):
    # Every facility open, no flows: worth a repair, which (faked) fails.
    mip = facility_location()
    search = search_for(mip, monkeypatch)
    x = np.where(mip.integrality, 1.0, 0.0)

    def failing_solve(lp, **kwargs):
        solution = type("Solution", (), {"primal": np.zeros(mip.num_variables)})()
        return {"solution": solution, "epochs": 1, "stop_reason": "max_epochs"}

    calls = count_lp_solves(monkeypatch, failing_solve)
    assert not search.try_incumbent(x, "H", None)
    assert not search.try_incumbent(x, "H", None)
    assert len(calls) == 1
    assert calls[0]["max_epochs"] == search.repair_max_epochs == 20
    assert search.lp_repairs == 1 and search.lp_repairs_skipped == 1


def test_repair_finds_the_continuous_variables(monkeypatch):
    # Every facility open, no flows: the repair LP routes the demand.
    mip = facility_location()
    search = search_for(mip, monkeypatch)
    x = np.where(mip.integrality, 1.0, 0.0)
    assert search.try_incumbent(x, "H", None)
    assert search.lp_repairs == 1 and mip.is_feasible(search.incumbent[1])


# %% Coefficient tightening
def two_row_mip(A_ineq, b_ineq, upper, integrality):
    n = A_ineq.shape[1]
    lp = jl.LP(
        c=-np.ones(n),
        A_eq=sp.csc_matrix((0, n)),
        b_eq=np.zeros(0),
        A_ineq=sp.csc_matrix(A_ineq),
        b_ineq=np.asarray(b_ineq, float),
        lower_bounds=np.zeros(n),
        upper_bounds=np.asarray(upper, float),
    )
    return jm.MIP(lp, np.asarray(integrality, bool))


def test_tighten_coefficients_textbook():
    # Row 0: y − 100 z <= 0 with y <= 7 cuts −100 to −7.
    # Row 1: 5 z + w <= 6 with w <= 3 (excess 2) becomes 2 z + w <= 3.
    mip = two_row_mip(
        np.array([[1.0, -100.0, 0.0], [0.0, 5.0, 1.0]]),
        [0.0, 6.0],
        [7.0, 1.0, 3.0],
        [False, True, False],
    )
    tightened, count = jm.tighten_coefficients(mip, mip.lower, mip.upper)
    assert count == 2
    A = tightened.A.toarray()
    np.testing.assert_allclose(A, [[1.0, -7.0, 0.0], [0.0, 2.0, 1.0]], rtol=1e-6)
    np.testing.assert_allclose(tightened.b, [0.0, 3.0], atol=1e-6)
    # Nothing to tighten: the same MIP back.
    again, count = jm.tighten_coefficients(tightened, mip.lower, mip.upper)
    assert count == 0 and again is tightened


def test_tighten_coefficients_needs_finite_bounds():
    # y unbounded: y − 100 z <= 0 stays.
    mip = two_row_mip(np.array([[1.0, -100.0]]), [0.0], [np.inf, 1.0], [False, True])
    assert jm.tighten_coefficients(mip, mip.lower, mip.upper)[1] == 0


@pytest.mark.parametrize(
    "make", [knapsack, general_integer, facility_location, generalised_assignment]
)
def test_tighten_coefficients_keeps_the_optimum(make):
    mip = make()
    lower, upper, _ = jm.propagate_bounds(
        mip.A, mip.row_lower, mip.row_upper, mip.lower, mip.upper,
        mip.integrality, max_passes=20,
    )
    tightened, _ = jm.tighten_coefficients(mip, lower, upper)
    assert reference(tightened) == pytest.approx(reference(mip), rel=1e-9, abs=1e-9)


def test_coefficient_tightening_in_the_search():
    # Capacities well above total demand: the capacity rows' big-M shrinks.
    mip = facility_location(seed=4, n_fac=4, n_cust=4)
    on = solve(mip)
    assert on["coefficients_tightened"] > 0
    assert_solved(on, mip, reference(mip))
    off = solve(mip, coefficient_tightening=False)
    assert off["coefficients_tightened"] == 0
    assert_solved(off, mip, reference(mip))


# %% Lifted cover cuts
@pytest.mark.parametrize("seed", range(20))
def test_lifted_cover_cuts_are_valid(seed):
    # Every cut holds at every 0/1 point of the row, with some coefficients
    # negative (complemented) and a continuous variable moved to its bound.
    rng = np.random.default_rng(seed)
    n = 10
    a = rng.integers(1, 20, n).astype(float) * rng.choice([1.0, 1.0, -1.0], n)
    a = np.append(a, 3.0)  # a continuous variable in [0, 2]
    b = 0.4 * np.abs(a[:n]).sum() - np.abs(a[:n][a[:n] < 0]).sum()
    lower, upper = np.zeros(n + 1), np.append(np.ones(n), 2.0)
    is_binary = np.append(np.ones(n, bool), False)
    x = np.append(rng.uniform(0, 1, n), 0.0)
    points = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(float)
    feasible = points[points @ a[:n] <= b]
    cuts = jm.cover_cuts(np.arange(n + 1), a, b, x, lower, upper, is_binary)
    for idx, coef, rhs in cuts:
        assert np.all(feasible[:, idx[idx < n]] @ coef[idx < n] <= rhs + 1e-9)
        assert coef @ x[idx] > rhs  # each cut cuts off the LP point


def test_cover_cuts_skip_integral_points_only_inside_the_knapsack():
    # 5 x0 + 5 x1 + 5 x2 + 9 x3 <= 12: (1, 1, 0, 0) fits, so nothing cuts it
    # off; (1, 1, 1, 0) does not (it breaks the row), and the cover
    # {0, 1, 2} still cuts it off.
    a = np.array([5.0, 5.0, 5.0, 9.0])
    args = (np.zeros(4), np.ones(4), np.ones(4, bool))
    assert jm.cover_cuts(np.arange(4), a, 12.0, np.array([1.0, 1.0, 0.0, 0.0]), *args) == []
    cuts = jm.cover_cuts(np.arange(4), a, 12.0, np.array([1.0, 1.0, 1.0, 0.0]), *args)
    assert cuts and all(coef @ np.array([1.0, 1.0, 1.0, 0.0])[idx] > rhs for idx, coef, rhs in cuts)


@pytest.mark.parametrize("seed", range(10))
def test_cover_cuts_hold_when_the_capacity_is_rounded_down(seed):
    # A 0/1 point fills the row exactly, and the stored right-hand side is
    # an ulp of float32 below it (as a cut's is after rounding to nearest).
    # Every cover must still hold at every 0/1 point of the true row: a
    # cover is a jump of a whole unit, so 1e-7 must not make one (this cut
    # off a generalised-assignment optimum with precision="float32").
    rng = np.random.default_rng(seed)
    n = 8
    a = rng.uniform(0.2, 2.0, n)
    z = (rng.random(n) < 0.6).astype(float)
    z[:2] = 1.0
    b_true = float(a @ z)
    # The largest float32 below the true right-hand side.
    b_stored = np.float32(b_true)
    if float(b_stored) >= b_true:
        b_stored = np.nextafter(b_stored, np.float32(-np.inf))
    b_stored = float(b_stored)
    assert b_true - 1e-6 < b_stored < b_true
    points = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(float)
    feasible = points[points @ a <= b_true]
    args = (np.zeros(n), np.ones(n), np.ones(n, bool))
    for x in (z, np.clip(z + rng.uniform(-0.2, 0.2, n), 0, 1)):
        for idx, coef, rhs in jm.cover_cuts(np.arange(n), a, b_stored, x, *args):
            assert np.all(feasible[:, idx] @ coef <= rhs + 1e-9)


def test_round_up_only_loosens_rows():
    values = np.array([5.480769230769231, 3.0, -1.1, 49.84, 1e-9])
    up = jm._round_up(values, np.float32)
    assert np.all(up >= values)
    assert np.array_equal(up.astype(np.float32).astype(np.float64), up)
    # The nearest such number: one ulp lower is below the value.
    below = np.nextafter(up.astype(np.float32), np.float32(-np.inf)).astype(np.float64)
    assert np.all(below < values)
    assert np.array_equal(jm._round_up(values, np.float64), values)


def test_with_rows_never_tightens_a_cut():
    given = generalised_assignment()
    mip = jm.MIP(
        jl._cast_lp(given.lp, np.float32), given.integrality, given.offset,
        given.maximise, given.implied,
    )
    assert mip.lp.c.dtype == np.float32
    n = mip.num_variables
    row = sp.csr_matrix(np.ones((1, n)))
    rhs = 5.480769230769231  # rounds down at float32 to nearest
    assert float(np.float32(rhs)) < rhs
    assert mip.with_rows(row, [rhs]).b[-1] >= rhs
    assert mip.rows_with(row, [rhs])[1][-1] >= rhs


def test_lifting_beats_the_extended_cover():
    # 8 x0 + 8 x1 + 8 x2 + 3 x3 + 3 x4 <= 17 at (0.9, 0.9, 0.9, 0, 0): the
    # extended cover of {0, 1, 2} adds nothing (no item weighs 8). Lifting
    # x3: with x3 = 1 the room is 14, which fits one cover item, so x3 gets
    # 2 − 1 = 1. Then x4 gets 0 (one cover item plus x3 fit in 14, value 2).
    cuts = jm.cover_cuts(
        np.arange(5),
        np.array([8.0, 8.0, 8.0, 3.0, 3.0]),
        17.0,
        np.array([0.9, 0.9, 0.9, 0.0, 0.0]),
        np.zeros(5),
        np.ones(5),
        np.ones(5, bool),
    )
    assert cuts
    idx, coef, rhs = cuts[0]
    np.testing.assert_array_equal(idx, [0, 1, 2, 3])
    np.testing.assert_allclose(coef, 1.0)
    assert rhs == pytest.approx(2.0)


# %% Device
def test_device_cpu_matches_reference():
    mip = facility_location()
    result = solve(mip, device="cpu")
    assert_solved(result, mip, reference(mip))


def test_device_auto_picks_the_cpu_for_small_mips(monkeypatch):
    search = search_for(knapsack(), monkeypatch, device="auto")
    assert search.device.platform == "cpu"
    assert search_for(knapsack(), monkeypatch, device=None).device is None


def test_device_must_be_known():
    with pytest.raises(ValueError, match="device"):
        solve(knapsack(), device="abacus")


# %% Tree cuts
@pytest.mark.parametrize("make", [knapsack, generalised_assignment, facility_location])
def test_tree_cuts_keep_the_optimum(make, monkeypatch):
    mip = make()
    search = search_for(mip, monkeypatch, tree_cuts=True, tree_cut_every=1)
    monkeypatch.undo()  # sub-searches (RENS) need the real run()
    result = search._run()
    assert_solved(result, mip, reference(mip))
    # Every cut in the pool (root and tree) holds at the optimum.
    x_star = optimum(mip)
    for cut in search.cut_list:
        assert cut["coef"] @ x_star[cut["idx"]] <= cut["rhs"] + 1e-6


def test_tree_cuts_are_added_without_recompiling(monkeypatch):
    mip = generalised_assignment(1)
    traces = []
    real = jl._SolvePrograms._device

    def counted(self, problem):
        traces.append(1)
        return real(self, problem)

    monkeypatch.setattr(jl._SolvePrograms, "_device", counted)
    # Without Gomory cuts the root leaves a tree to cut in.
    options = dict(tree_cuts=True, tree_cut_every=1, max_nodes=40, gomory_cuts=False)
    solve(mip, **options)  # compile
    traces.clear()
    result = solve(mip, **options)
    assert result["tree_cuts_added"] > 0
    assert traces == []  # the same shapes throughout


@pytest.mark.parametrize("per_round", [1, 3, 40])
def test_tree_cuts_per_round_follow_the_option(per_round, monkeypatch):
    mip = generalised_assignment(1)
    search = search_for(
        mip, monkeypatch, tree_cuts=True, tree_cut_every=1, max_nodes=40,
        gomory_cuts=False, max_tree_cuts_per_round=per_round,
    )
    monkeypatch.undo()
    added = []
    real = search.refresh_cuts

    def counted(solutions, best_first, **kwargs):
        before = search.tree_cuts_added
        out = real(solutions, best_first, **kwargs)
        added.append(search.tree_cuts_added - before)
        return out

    search.refresh_cuts = counted
    result = search._run()
    assert result["tree_cuts_added"] > 0
    assert max(added) <= per_round
    # The pool has room for two rounds beyond the root's cuts.
    assert search.tree_rows_cap - search.tree_padding_rows >= 2 * per_round


def test_tree_cuts_per_round_must_be_positive():
    with pytest.raises(ValueError, match="max_tree_cuts_per_round"):
        solve(knapsack(), tree_cuts=True, max_tree_cuts_per_round=0)
    for name in ("tree_cut_rows", "tree_cut_nnz"):
        with pytest.raises(ValueError, match=name):
            solve(knapsack(), tree_cuts=True, **{name: 0})


def test_tree_cut_pool_size_follows_the_options(monkeypatch):
    mip = generalised_assignment(2)
    options = dict(
        tree_cuts=True, tree_cut_every=1, max_nodes=200, gomory_cuts=False,
        max_tree_cuts_per_round=40,
    )
    small = solve(mip, **options)
    assert small["tree_cuts_dropped"] > 0  # the default pool fills
    search = search_for(mip, monkeypatch, tree_cut_rows=1000, **options)
    monkeypatch.undo()
    large = search._run()
    assert_solved(large, mip, reference(mip))
    assert search.tree_rows_cap - search.tree_padding_rows >= 1000
    assert search.tree_nnz_cap >= 1000 * 16  # scaled with the rows
    assert large["tree_cuts_dropped"] == 0
    search = search_for(mip, monkeypatch, tree_cut_rows=1000, tree_cut_nnz=5000, **options)
    monkeypatch.undo()
    search._run()
    assert search.tree_nnz_cap == 8192  # as given, to a power of two


def test_violated_cuts_keeps_every_violated_candidate():
    x = np.array([1.0, 1.0, 0.0])
    cuts = [
        ("a", np.array([0, 1]), np.array([1.0, 1.0]), 1.5),  # violated
        ("b", np.array([0, 1]), np.array([1.0, 1.0]), 1.4),  # parallel to a
        ("c", np.array([2]), np.array([1.0]), 0.0),  # tight, not violated
        ("d", np.array([0]), np.array([1.0]), 0.99999),  # under 1e-4
    ]
    assert [cut[0] for cut in jm.violated_cuts(cuts, x)] == ["a", "b"]
    # select_cuts keeps one of a pair of parallel cuts.
    assert [cut[0] for cut in jm.select_cuts(cuts, x, 3)] == ["b"]
    assert jm.violated_cuts([], x) == []


def test_the_pool_keeps_every_violated_root_candidate(monkeypatch):
    """The root rounds pool every candidate their point violates, not only
    the ``max_cuts_per_round`` they put into the LP."""
    chosen, pooled = [], []
    real_select, real_pool = jm.select_cuts, jm._BranchAndBound.pool_cuts

    def select_cuts(*args, **kwargs):
        out = real_select(*args, **kwargs)
        chosen.extend(jm._cut_key(cut) for cut in out)
        return out

    def pool_cuts(self, cuts):
        pooled.extend(jm._cut_key(cut) for cut in cuts)
        return real_pool(self, cuts)

    monkeypatch.setattr(jm, "select_cuts", select_cuts)
    monkeypatch.setattr(jm._BranchAndBound, "pool_cuts", pool_cuts)
    mip = general_integer()
    result = solve(mip, max_cuts_per_round=2, max_nodes=1, tree_cuts=False)
    assert set(chosen) <= set(pooled)
    assert len(set(pooled)) > len(set(chosen)) > 0
    assert result["cut_pool"] > len(set(chosen)) // 2


def test_solve_caps_the_cut_pool_at_cut_pool_size():
    """``solve``'s ``cut_pool_size`` reaches the global cut pool."""
    mip = general_integer()
    full = solve(mip, max_cuts_per_round=2, max_nodes=1, tree_cuts=False)
    assert full["cut_pool"] > 2
    capped = solve(
        mip, max_cuts_per_round=2, max_nodes=1, tree_cuts=False, cut_pool_size=2
    )
    assert capped["cut_pool"] <= 2
    with pytest.raises(ValueError, match="cut_pool_size"):
        solve(knapsack(), cut_pool_size=-1)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_tree_cuts_refresh_from_the_pool_and_requeue(batch_size):
    # The cut block is refreshed between node solves, serially and in
    # batches: pooled cuts come back, and members a new cut cuts off are
    # solved again before they branch.
    mip = general_integer()
    result = solve(mip, batch_size=batch_size, tree_cut_every=1)
    assert_solved(result, mip, reference(mip))
    assert result["tree_cut_refreshes"] > 0 and result["tree_cuts_added"] > 0
    assert result["tree_cut_pool_hits"] > 0
    assert result["tree_cut_requeued"] > 0
    assert sum(result["tree_cut_families"].values()) == result["tree_cuts_added"]
    off = solve(mip, batch_size=batch_size, tree_cuts=False)
    assert_solved(off, mip, reference(mip))
    assert off["tree_cut_refreshes"] == 0 and off["tree_cuts_added"] == 0


@pytest.mark.parametrize("batch_size", [1, 4])
def test_tree_cut_bounds_hold_for_every_better_solution(batch_size, monkeypatch):
    # Every node's bound, from duals over the refreshed block (replaced
    # slots' duals zeroed), holds for every solution in its box better than
    # the cutoff.
    mip = general_integer()
    records = []
    real = jm._BranchAndBound._evaluate

    def evaluate(self, node, lower, upper, lp, result):
        raw, _ = self.bound_over(self.result_terms(result, lower, upper), lower, upper)
        records.append((lower.copy(), upper.copy(), raw, self.cutoff()))
        return real(self, node, lower, upper, lp, result)

    monkeypatch.setattr(jm._BranchAndBound, "_evaluate", evaluate)
    result = solve(mip, batch_size=batch_size, tree_cut_every=1)
    assert_solved(result, mip, reference(mip))
    assert result["tree_cut_refreshes"] > 0
    checked = 0
    for lower, upper, raw, cutoff in records[:: max(1, len(records) // 25)]:
        best = milp(
            mip.c,
            constraints=LinearConstraint(mip.A.toarray(), mip.row_lower, mip.row_upper),
            integrality=mip.integrality.astype(int),
            bounds=Bounds(lower, upper),
        )
        value = best.fun if best.success else np.inf
        assert value >= min(raw, cutoff) - 1e-6 * (1 + abs(value if np.isfinite(value) else 0))
        checked += 1
    assert checked > 0


def test_fit_warm_zeroes_the_duals_of_replaced_slots(monkeypatch):
    mip = general_integer()
    search = search_for(mip, monkeypatch)
    base_m = search.base.A.shape[0] - search.base.n_eq
    k = 4
    search.slot_version = [0, 2, 0, 3]
    search.cut_version = 3
    solution = jl.SaddleState(
        primal=jnp.zeros(mip.num_variables),
        dual_ineq=jnp.arange(1.0, base_m + k + 1.0),
        dual_eq=jnp.zeros(max(mip.n_eq, 1)),
    )
    node = jm._Node(0.0, 1, (), (solution, None), None)
    node.cut_version = 1  # slots 1 and 3 changed since
    fitted = np.asarray(search.fit_warm(node)[0].dual_ineq)
    expected = np.arange(1.0, base_m + k + 1.0)
    expected[[base_m + 1, base_m + 3]] = 0.0
    np.testing.assert_array_equal(fitted, expected)
    node.cut_version = 3  # up to date: unchanged
    assert search.fit_warm(node)[0] is solution


def test_tree_cut_refreshes_keep_the_compiled_node_solver(monkeypatch):
    # The block keeps its capacity, so the batched node solver and the
    # device kernels take each refreshed block as data.
    traces, at_refresh = [], []
    real = jl._SolvePrograms._device

    def counted(self, problem):
        traces.append(1)
        return real(self, problem)

    real_refresh = jm._BranchAndBound.refresh_cuts

    def refresh(self, solutions, best_first, **kwargs):
        out = real_refresh(self, solutions, best_first, **kwargs)
        at_refresh.append(len(traces))
        return out

    monkeypatch.setattr(jl._SolvePrograms, "_device", counted)
    monkeypatch.setattr(jm._BranchAndBound, "refresh_cuts", refresh)
    mip = general_integer()
    result = solve(mip, batch_size=4, tree_cut_every=1)
    assert_solved(result, mip, reference(mip))
    assert len(at_refresh) >= 3
    # After the first refreshes nothing new compiles.
    assert at_refresh[-1] == at_refresh[len(at_refresh) // 2]


def test_pool_refreshes_run_between_full_refreshes(monkeypatch):
    """Every ``tree_pool_every`` nodes the cut block takes back violated
    pool cuts without separating (``separate=False``); a full refresh,
    with separation, comes every ``tree_cut_every`` nodes."""
    calls = []
    real_refresh = jm._BranchAndBound.refresh_cuts
    real_separated = jm._BranchAndBound.node_separated

    def refresh(self, solutions, best_first, separate=True):
        calls.append([separate, 0])
        return real_refresh(self, solutions, best_first, separate=separate)

    def node_separated(self, x):
        calls[-1][1] += 1
        return real_separated(self, x)

    monkeypatch.setattr(jm._BranchAndBound, "refresh_cuts", refresh)
    monkeypatch.setattr(jm._BranchAndBound, "node_separated", node_separated)
    mip = generalised_assignment(2)
    result = solve(
        mip, tree_cuts=True, tree_cut_every=10, tree_pool_every=1, max_nodes=60,
        gomory_cuts=False,
    )
    assert result["tree_pool_refreshes"] > result["tree_cut_refreshes"] > 0
    assert any(separate for separate, _ in calls)
    assert all(count == 0 for separate, count in calls if not separate)
    calls.clear()
    off = solve(
        mip, tree_cuts=True, tree_cut_every=10, tree_pool_every=0, max_nodes=60,
        gomory_cuts=False,
    )
    assert off["tree_pool_refreshes"] == 0
    assert all(separate for separate, _ in calls)
    with pytest.raises(ValueError, match="tree_pool_every"):
        solve(mip, tree_pool_every=-1)


@pytest.mark.parametrize(
    "setting, value",
    [
        ("tree_cut_points", 0),
        ("tree_cut_rounds", -1),
        ("tree_cut_share", 1.5),
        ("tree_cut_max_nnz", 0),
    ],
)
def test_rejects_bad_tree_cut_settings(setting, value):
    with pytest.raises(ValueError, match=setting):
        solve(knapsack(), **{setting: value})


# %% Gomory defaults and Jaddle RINS
def test_gomory_auto_follows_the_size(monkeypatch):
    small = search_for(knapsack(), monkeypatch)
    assert small.gomory and small.cut_rounds == 40 and small.cut_min_progress == 1e-3
    large = search_for(generalised_assignment(agents=50, jobs=120), monkeypatch)
    assert large.A_T.nnz > 10_000
    assert not large.gomory and large.cut_rounds == 5 and large.cut_min_progress == 1e-4
    forced = search_for(knapsack(), monkeypatch, gomory_cuts=False, cut_rounds=3)
    assert not forced.gomory and forced.cut_rounds == 3


def test_rejects_bad_gomory_options():
    with pytest.raises(ValueError, match="gomory_cuts"):
        solve(knapsack(), gomory_cuts="sometimes")


def test_jaddle_rins_returns_only_better_points(monkeypatch):
    mip = generalised_assignment(1)
    search = search_for(mip, monkeypatch, max_nodes=15)
    monkeypatch.undo()  # sub-searches need the real run()
    search._run()
    assert search.incumbent is not None
    # An LP point that agrees with the incumbent on 60% of the integers.
    x = search.incumbent[1].copy()
    free = np.random.default_rng(0).random(x.size) < 0.4
    x[free] = 0.5
    cutoff = search.cutoff()
    traces = []
    real = jl._SolvePrograms._device

    def counted(self, problem):
        traces.append(1)
        return real(self, problem)

    monkeypatch.setattr(jl._SolvePrograms, "_device", counted)
    found = search.jaddle_rins(x, seconds=10.0)
    assert traces == []  # the sub-MIP reuses the compiled solver
    if found is not None:
        assert mip.is_feasible(found)
        assert mip.c @ found < cutoff + 1e-9
    # The same neighbourhood again is skipped.
    assert search.jaddle_rins(x, seconds=10.0) is None
    assert search.rins_skipped == 1


def test_rins_in_the_search_on_a_harder_model():
    mip = generalised_assignment(2)
    result = solve(mip, rins_seconds=2, rins_every=10)
    assert_solved(result, mip, reference(mip))
    assert result["rins_calls"] > 0


# %% Stronger cut generator
def max_over_mip(mip, idx, coef, objective_at_most=None, box=None):
    """The largest ``coef · x[idx]`` over the MIP's feasible points (with
    ``c·x <= objective_at_most``, if given, and within ``box``, a ``(lower,
    upper)`` pair, if given; ``-inf`` when there are none)."""
    objective = np.zeros(mip.num_variables)
    objective[idx] = -coef
    A, row_lower, row_upper = mip.A.toarray(), mip.row_lower, mip.row_upper
    if objective_at_most is not None:
        A = np.vstack([A, mip.c])
        row_lower = np.append(row_lower, -np.inf)
        row_upper = np.append(row_upper, objective_at_most)
    lower, upper = (mip.lower, mip.upper) if box is None else box
    best = milp(
        objective,
        constraints=LinearConstraint(A, row_lower, row_upper),
        integrality=mip.integrality.astype(int),
        bounds=Bounds(lower, upper),
    )
    if (objective_at_most is not None or box is not None) and best.status == 2:
        return -np.inf  # infeasible
    return -best.fun


@pytest.mark.parametrize("seed", range(30))
def test_cmir_cut_is_valid_by_brute_force(seed):
    # Four integers in 0..3 and a continuous y in [0, 5]: the cut holds at
    # every feasible point (for each integer point, at both ends of y's
    # feasible interval, since the cut is linear in y).
    rng = np.random.default_rng(seed)
    a = np.append(rng.integers(-6, 9, 4).astype(float), rng.choice([-2.5, 1.5]))
    b = float(rng.integers(3, 15)) + 0.5
    lower, upper = np.zeros(5), np.array([3.0, 3.0, 3.0, 3.0, 5.0])
    is_int = np.array([True, True, True, True, False])
    x = np.append(rng.uniform(0, 3, 4), rng.uniform(0, 5))
    cut = jm.cmir_cut(np.arange(5), a, b, x, lower, upper, is_int)
    if cut is None:
        return
    idx, coef, rhs = cut
    full = np.zeros(5)
    full[idx] = coef
    assert full @ x > rhs  # it cuts off the point
    grid = np.array(np.meshgrid(*[np.arange(4)] * 4)).reshape(4, -1).T.astype(float)
    for point in grid:
        room = b - a[:4] @ point  # a_y · y <= room
        if a[4] > 0:
            ends = (0.0, min(5.0, room / a[4]))
        else:
            ends = (max(0.0, room / a[4]), 5.0)
        for y in ends:
            if ends[0] <= ends[1] + 1e-12:
                assert full[:4] @ point + full[4] * y <= rhs + 1e-9


@pytest.mark.parametrize(
    "make", [knapsack, general_integer, facility_location, generalised_assignment]
)
def test_tableau_mir_cuts_are_valid(make):
    mip = make()
    x, basis = vertex_and_basis(mip)
    cuts = jm.gomory_cuts(
        mip.A, mip.b, mip.n_eq, mip.lower, mip.upper, mip.integrality, basis, x,
        tableau_mir=True,
    )
    for _, idx, coef, rhs in cuts:
        assert coef @ x[idx] > rhs
        assert max_over_mip(mip, idx, coef) <= rhs + 1e-7 * (1 + abs(rhs))


@pytest.mark.parametrize("make", [knapsack, general_integer, generalised_assignment])
def test_integer_aggregation_cuts_are_valid(make):
    mip = make()
    x, _ = vertex_and_basis(mip)
    lower, upper, _ = jm.propagate_bounds(
        mip.A, mip.row_lower, mip.row_upper, mip.lower, mip.upper, mip.integrality
    )
    separator = jm.CutSeparator(mip, lower, upper, aggregate_integers=True, max_aggregation=4)
    for _, idx, coef, rhs in separator(x):
        assert max_over_mip(mip, idx, coef) <= rhs + 1e-7 * (1 + abs(rhs))


def fixed_charge(seed=0, n=6):
    """Open facilities (binary ``z``) to meet a demand with flows ``y``: a
    flow needs its facility open (``y_j <= u_j·z_j``) and an open one ships
    at least ``l_j`` (``y_j >= l_j·z_j``). Columns: ``z`` then ``y``; rows:
    the upper bounds, the lower bounds, then the demand."""
    rng = np.random.default_rng(seed)
    u = rng.integers(10, 30, n).astype(float)
    low = np.floor(0.2 * u)
    demand = 0.55 * u.sum() + 0.5
    A_ineq = sp.lil_matrix((2 * n + 1, 2 * n))
    for j in range(n):
        A_ineq[j, n + j], A_ineq[j, j] = 1.0, -u[j]  # y_j − u_j z_j <= 0
        A_ineq[n + j, n + j], A_ineq[n + j, j] = -1.0, low[j]  # l_j z_j − y_j <= 0
        A_ineq[2 * n, n + j] = -1.0  # −Σ y >= −demand
    b_ineq = np.zeros(2 * n + 1)
    b_ineq[-1] = -demand
    A_eq, b_eq = no_rows(2 * n)
    lp = jl.LP(
        c=np.concatenate([rng.uniform(20, 60, n), rng.uniform(1, 4, n)]),
        A_eq=A_eq,
        b_eq=b_eq,
        A_ineq=sp.csc_matrix(A_ineq),
        b_ineq=b_ineq,
        lower_bounds=np.zeros(2 * n),
        upper_bounds=np.concatenate([np.ones(n), np.full(n, u.sum())]),
    )
    return jm.MIP(lp, np.arange(2 * n) < n), u, low


def test_variable_bounds_found():
    mip, u, low = fixed_charge()
    n = u.size
    bounds = jm.variable_bounds(mip)
    assert sorted(bounds) == list(range(n, 2 * n))
    for j in range(n):
        assert bounds[n + j]["ub"] == [(j, pytest.approx(u[j]), pytest.approx(0.0))]
        assert bounds[n + j]["lb"] == [(j, pytest.approx(low[j]), pytest.approx(0.0))]
    # A pure integer MIP has none.
    assert jm.variable_bounds(knapsack()) == {}


@pytest.mark.parametrize("seed", range(4))
def test_variable_bound_cmir_cuts_are_valid(seed):
    mip, u, _ = fixed_charge(seed)
    n = u.size
    bounds = jm.variable_bounds(mip)
    demand = 2 * n
    idx, a = mip.A[demand].indices, mip.A[demand].data
    rng = np.random.default_rng(seed)
    x_lp, _ = vertex_and_basis(mip)
    # The LP vertex, then points with fractional z and flows between their
    # variable bounds, so either kind of bound can be the nearest.
    points = [x_lp] + [
        np.concatenate([z, rng.uniform(0.0, 1.0, n) * u * z])
        for z in rng.uniform(0.05, 0.95, (5, n))
    ]
    found = 0
    for x in points:
        cut = jm.cmir_cut(
            idx, a, mip.b[demand], x, mip.lower, mip.upper, mip.integrality,
            variable_bounds=bounds,
        )
        if cut is None:
            continue
        c_idx, coef, rhs = cut
        assert coef @ x[c_idx] > rhs
        assert max_over_mip(mip, c_idx, coef) <= rhs + 1e-7 * (1 + abs(rhs))
        found += 1
    assert found > 0
    separator = jm.CutSeparator(mip, mip.lower, mip.upper)
    assert separator.variable_bounds
    for _, c_idx, coef, rhs in separator(x_lp):
        assert max_over_mip(mip, c_idx, coef) <= rhs + 1e-7 * (1 + abs(rhs))


def test_variable_bound_cmir_is_stronger():
    # At the LP vertex the flows sit on their upper bounds u_j·z_j, which
    # the plain cut cannot see.
    mip, u, _ = fixed_charge()
    demand = 2 * u.size
    idx, a = mip.A[demand].indices, mip.A[demand].data
    x, _ = vertex_and_basis(mip)
    args = (idx, a, mip.b[demand], x, mip.lower, mip.upper, mip.integrality)
    plain = jm.cmir_cut(*args)
    substituted = jm.cmir_cut(*args, variable_bounds=jm.variable_bounds(mip))
    assert substituted is not None

    def efficacy(cut):
        c_idx, coef, rhs = cut
        return (coef @ x[c_idx] - rhs) / np.linalg.norm(coef)

    assert plain is None or efficacy(substituted) > efficacy(plain) + 1e-6


# %% Node LPs stop at the cutoff
@pytest.mark.parametrize("make", [general_integer, generalised_assignment, facility_location])
def test_node_bound_stop_keeps_the_optimum(make):
    mip = make()
    ref = reference(mip)
    on = solve(mip, node_bound_stop=True)
    off = solve(mip, node_bound_stop=False)
    assert_solved(on, mip, ref)
    assert_solved(off, mip, ref)
    assert off["bound_stops"] == 0
    if make is general_integer:
        # Without sub-MIP heuristics, so the two trees differ only by the
        # stop, and without zero-half cuts, which leave too small a tree for
        # it to save LP work (without any cuts the two trees differ).
        plain = dict(rins_seconds=0.0, rens_seconds=0.0, zero_half_cuts=False)
        on = solve(mip, node_bound_stop=True, **plain)
        off = solve(mip, node_bound_stop=False, **plain)
        assert on["bound_stops"] > 0
        assert on["lp_epochs"] < off["lp_epochs"]


def test_node_lp_tolerance_applies_to_node_solves_only(monkeypatch):
    mip = general_integer()
    calls = []
    real = jm._BranchAndBound.solve_lp

    def solve_lp(self, lp, warm, max_epochs, **overrides):
        calls.append((max_epochs, overrides.get("dual_gap_tolerance")))
        return real(self, lp, warm, max_epochs, **overrides)

    monkeypatch.setattr(jm._BranchAndBound, "solve_lp", solve_lp)
    # No feasibility pump: it also solves LPs with the node budget.
    result = solve(mip, node_lp_tolerance=3e-4, feasibility_pump=False)
    assert_solved(result, mip, reference(mip))
    nodes = [tol for epochs, tol in calls if epochs == 200]  # node_max_epochs
    assert nodes and all(tol == 3e-4 for tol in nodes)


def test_node_bound_target_matches_the_pruning_test(monkeypatch):
    mip = knapsack()  # integer costs: objective step 1
    search = search_for(mip, monkeypatch)
    assert search.objective_step == 1.0
    assert search.node_bound_target() is None  # no cutoff yet
    search.incumbent = (-120.0, np.zeros(mip.num_variables))
    cutoff = search.cutoff()
    target = search.node_bound_target()
    # A bound at the target closes a node; one a step lower does not.
    assert search.round_bound(target) >= cutoff
    assert search.round_bound(target - 1.0) < cutoff
    search.node_bound_stop = False
    assert search.node_bound_target() is None


# %% Flow cover cuts
def random_flow_row(seed):
    """A row over flows ``y_j <= c_j·z_j`` (either sign), lone binaries and
    plain continuous variables, with its MIP (the row, then the variable
    bounds) for exact checks. Seeds from 30 give demand rows (outflows
    only, ``−Σ y <= −D``)."""
    rng = np.random.default_rng(seed)
    k, m, p = rng.integers(2, 6), rng.integers(0, 3), rng.integers(0, 3)
    n = 2 * k + m + p  # columns: z, y, lone binaries, plain continuous
    is_int = np.zeros(n, bool)
    is_int[:k] = is_int[2 * k : 2 * k + m] = True
    lower, upper = np.zeros(n), np.ones(n)
    caps = rng.integers(2, 12, k).astype(float)
    upper[k : 2 * k] = caps + rng.integers(0, 5, k)
    upper[2 * k + m :] = rng.integers(1, 8, p).astype(float)
    lower[2 * k + m :] = -rng.integers(0, 3, p).astype(float)
    bounds = {k + j: {"ub": [(j, float(caps[j]), 0.0)], "lb": []} for j in range(k)}
    a = np.zeros(n)
    a[k : 2 * k] = rng.choice([1.0, -1.0], k, p=[0.7, 0.3]) * rng.integers(1, 4, k)
    a[2 * k : 2 * k + m] = rng.choice([-1.0, 1.0], m) * rng.integers(1, 9, m)
    a[2 * k + m :] = rng.choice([-1.0, 1.0], p) * rng.integers(1, 4, p)
    b = float(rng.integers(1, 15)) + rng.choice([0.0, 0.5])
    if seed >= 30:
        a[k : 2 * k] = -np.abs(a[k : 2 * k])
        b = -b
    rows = [a] + [np.eye(n)[k + j] - caps[j] * np.eye(n)[j] for j in range(k)]
    A, rhs = np.array(rows), np.concatenate([[b], np.zeros(k)])
    points = []
    for _ in range(4):
        x = rng.uniform(lower, upper)
        x[k : 2 * k] = np.minimum(x[k : 2 * k], caps * x[:k])
        points.append(x)
    return dict(
        idx=np.flatnonzero(a), a=a, b=b, lower=lower, upper=upper, is_int=is_int,
        bounds=bounds, A=A, rhs=rhs, points=points,
    )


def max_over_row_mip(row, idx, coef):
    """The largest ``coef · x[idx]`` over a ``random_flow_row``'s MIP, or
    ``None`` when HiGHS does not solve it."""
    objective = np.zeros(row["a"].size)
    objective[idx] = -coef
    best = milp(
        objective,
        constraints=LinearConstraint(row["A"], -np.inf, row["rhs"]),
        integrality=row["is_int"].astype(int),
        bounds=Bounds(row["lower"], row["upper"]),
    )
    return -best.fun if best.success else None


def cut_efficacy(cut, x):
    idx, coef, rhs = cut
    return (coef @ x[idx] - rhs) / np.linalg.norm(coef)


@pytest.mark.parametrize("seed", range(40))
def test_flow_cover_cut_is_valid_by_brute_force(seed):
    # Each cut, from the numpy search and the batched JAX one, is checked
    # with an exact MILP; the two find equally efficacious cuts.
    row = random_flow_row(seed)
    idx, a, b = row["idx"], row["a"][row["idx"]], row["b"]
    args = (row["lower"], row["upper"], row["is_int"], row["bounds"])
    batched = jm.batched_flow_cover_cuts([(idx, a, b)] * 4, row["points"][0], *args)
    for x in row["points"]:
        cut = jm.flow_cover_cut(idx, a, b, x, *args)
        other = jm.batched_flow_cover_cuts([(idx, a, b)], x, *args)[0]
        assert (cut is None) == (other is None)
        if cut is None:
            continue
        assert cut_efficacy(other, x) == pytest.approx(cut_efficacy(cut, x), rel=1e-7, abs=1e-9)
        for c_idx, coef, c_rhs in (cut, other):
            assert coef @ x[c_idx] > c_rhs  # it cuts off the point
            most = max_over_row_mip(row, c_idx, coef)
            if most is not None:
                assert most <= c_rhs + 1e-7 * (1 + abs(c_rhs))
    assert len(batched) == 4


@pytest.mark.parametrize("seed", range(0, 40, 3))
def test_batched_cmir_matches_the_reference(seed):
    # Rows with variable bounds, so both the plain and the substituted
    # c-MIR searches run; the batched cuts are as efficacious and valid.
    row = random_flow_row(seed)
    idx, a, b = row["idx"], row["a"][row["idx"]], row["b"]
    for x in row["points"]:
        for bounds in (None, row["bounds"]):
            args = (x, row["lower"], row["upper"], row["is_int"])
            cut = jm.cmir_cut(idx, a, b, *args, variable_bounds=bounds)
            other = jm.batched_cmir_cuts([(idx, a, b)], *args, variable_bounds=bounds)[0]
            assert (cut is None) == (other is None)
            if cut is None:
                continue
            assert cut_efficacy(other, x) == pytest.approx(cut_efficacy(cut, x), rel=1e-7, abs=1e-9)
            most = max_over_row_mip(row, other[0], other[1])
            if most is not None:
                assert most <= other[2] + 1e-7 * (1 + abs(other[2]))


@pytest.mark.parametrize(
    "make",
    [knapsack, general_integer, facility_location, generalised_assignment,
     lambda: fixed_charge()[0]],
)
def test_batched_separation_matches_numpy(make):
    mip = make()
    x, _ = vertex_and_basis(mip)
    rng = np.random.default_rng(0)
    points = [x, np.clip(x + rng.uniform(-0.3, 0.3, x.size), mip.lower, np.minimum(mip.upper, 1e3))]
    for point in points:
        for aggregate in (False, True):
            kwargs = dict(aggregate_integers=aggregate, max_aggregation=4 if aggregate else 2)
            ref = jm.CutSeparator(mip, mip.lower, mip.upper, backend="numpy", **kwargs)(point)
            got = jm.CutSeparator(
                mip, mip.lower, mip.upper, backend="jax", min_batch=0, **kwargs
            )(point)
            assert [cut[0] for cut in got] == [cut[0] for cut in ref]
            for (_, *mine), (_, *theirs) in zip(got, ref):
                assert cut_efficacy(mine, point) == pytest.approx(
                    cut_efficacy(theirs, point), rel=1e-7, abs=1e-9
                )


def _select_cuts_reference(cuts, x, n, min_efficacy=1e-4, max_cuts=100):
    """``select_cuts`` before it was vectorised: one stack per candidate."""
    scored = []
    for cut in cuts:
        _, idx, coef, rhs = cut
        norm = np.linalg.norm(coef)
        with np.errstate(divide="ignore", invalid="ignore"):
            efficacy = (coef @ x[idx] - rhs) / norm
        if efficacy >= min_efficacy:
            scored.append((efficacy, cut, norm))
    scored.sort(key=lambda item: -item[0])
    chosen, directions = [], []
    for _, cut, norm in scored:
        _, idx, coef, _ = cut
        with np.errstate(divide="ignore", invalid="ignore"):
            direction = sp.csr_matrix((coef / norm, idx, [0, idx.size]), shape=(1, n))
        if directions and np.abs(sp.vstack(directions) @ direction.T).max() > 0.99:
            continue
        chosen.append(cut)
        directions.append(direction)
        if len(chosen) >= max_cuts:
            break
    return chosen


@pytest.mark.parametrize("seed", range(20))
def test_select_cuts_matches_the_reference(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 60))
    x = rng.uniform(0, 1, n)
    cuts = []
    for _ in range(int(rng.integers(0, 300))):
        size = int(rng.integers(1, min(n, 12) + 1))
        idx = np.sort(rng.choice(n, size, replace=False))
        coef = rng.integers(-3, 4, size).astype(float)  # near-duplicates too
        cuts.append(("mir", idx, coef, float(coef @ x[idx] - rng.uniform(-0.2, 0.5))))
    expected = _select_cuts_reference(cuts, x, n, max_cuts=30)
    assert [id(c) for c in jm.select_cuts(cuts, x, n, max_cuts=30)] == [id(c) for c in expected]


@pytest.mark.parametrize("speedup", [0.0, np.inf])  # always sparse, always dense
@pytest.mark.parametrize("seed", range(10))
def test_select_cuts_gram_paths_match_the_reference(seed, speedup, monkeypatch):
    monkeypatch.setattr(jm, "_GRAM_DENSE_SPEEDUP", speedup)
    monkeypatch.setattr(jm, "_GRAM_CHUNK_ENTRIES", 1000)  # several column chunks
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(20, 200))
    x = rng.uniform(0, 1, n)
    cuts = []
    for _ in range(int(rng.integers(50, 400))):
        size = int(rng.integers(1, n + 1))  # dense rows too, as Gomory cuts are
        idx = np.sort(rng.choice(n, size, replace=False))
        coef = rng.integers(-2, 3, size).astype(float)
        cuts.append(("gomory", idx, coef, float(coef @ x[idx] - rng.uniform(-0.2, 0.5))))
    expected = _select_cuts_reference(cuts, x, n, max_cuts=40)
    got = jm.select_cuts(cuts, x, n, max_cuts=40)
    assert [id(c) for c in got] == [id(c) for c in expected]


def test_gram_paths_agree():
    D = sp.random(60, 300, density=0.3, random_state=1, format="csr")
    np.testing.assert_allclose(jm._gram(D), (D @ D.T).toarray(), rtol=1e-12, atol=1e-12)
    dense = jm._GRAM_DENSE_SPEEDUP
    try:
        jm._GRAM_DENSE_SPEEDUP = np.inf
        np.testing.assert_allclose(jm._gram(D), (D @ D.T).toarray(), rtol=1e-12, atol=1e-12)
    finally:
        jm._GRAM_DENSE_SPEEDUP = dense


def test_rows_with_matches_with_rows():
    mip, _, _ = fixed_charge()
    rng = np.random.default_rng(0)
    n = mip.num_variables
    cuts = [
        dict(idx=np.sort(rng.choice(n, 3, replace=False)), coef=rng.normal(size=3), rhs=1.0)
        for _ in range(5)
    ]
    A_cuts, b_cuts = jm._cut_block(cuts, n, min_rows=16, min_nnz=64)  # padded
    A, b = mip.rows_with(A_cuts, b_cuts)
    full = mip.with_rows(A_cuts, b_cuts)
    assert A.shape == full.A.shape
    assert (A != full.A).nnz == 0
    np.testing.assert_array_equal(b, full.b)


def test_batched_cuts_ignore_the_precision_profile():
    import jaddle.jaddle_optimisers as jo

    mip, _, _ = fixed_charge()
    x, _ = vertex_and_basis(mip)
    separator = jm.CutSeparator(mip, mip.lower, mip.upper, backend="jax", min_batch=0)
    cuts = separator(x)
    with jo.local_settings(precision="float32"):
        again = separator(x)
    assert len(again) == len(cuts) > 0
    for (fa, ia, ca, ra), (fb, ib, cb, rb) in zip(cuts, again):
        assert fa == fb and np.array_equal(ia, ib)
        assert np.allclose(ca, cb, rtol=1e-12, atol=0) and ra == pytest.approx(rb, rel=1e-12)


def test_rejects_a_bad_separation_backend():
    with pytest.raises(ValueError, match="separation_backend"):
        solve(knapsack(), separation_backend="gpu")
    with pytest.raises(ValueError, match="backend"):
        jm.CutSeparator(knapsack(), np.zeros(15), np.ones(15), backend="torch")


def test_flow_cover_cuts_are_valid_and_cut_off_the_vertex():
    mip, u, _ = fixed_charge()
    x, _ = vertex_and_basis(mip)
    separator = jm.CutSeparator(mip, mip.lower, mip.upper)
    flows = [cut for cut in separator(x) if cut[0] == "flowcover"]
    assert flows
    for _, idx, coef, rhs in flows:
        assert coef @ x[idx] > rhs
        assert max_over_mip(mip, idx, coef) <= rhs + 1e-7 * (1 + abs(rhs))


def test_flow_covers_in_the_search():
    mip, _, _ = fixed_charge(1)
    result = solve(mip, gomory_cuts=False)
    assert_solved(result, mip, reference(mip))
    assert result["cuts"]["flowcover"] >= 0
    # Pure integer rows get none.
    assert solve(knapsack(), max_nodes=1)["cuts"]["flowcover"] == 0


def random_binary_rows(seed, n=10, m_eq=1, m_ineq=6):
    """A pure binary MIP with ±1..3 rows that a random point satisfies."""
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 2, n).astype(float)
    A_eq = rng.integers(-2, 3, (m_eq, n)).astype(float)
    A_ineq = rng.integers(-3, 4, (m_ineq, n)).astype(float)
    lp = jl.LP(
        c=rng.standard_normal(n),
        A_eq=sp.csc_matrix(A_eq),
        b_eq=A_eq @ x0,
        A_ineq=sp.csc_matrix(A_ineq),
        b_ineq=A_ineq @ x0 + rng.integers(0, 3, m_ineq),
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
    )
    return jm.MIP(lp, np.ones(n, bool))


def binary_points(mip):
    """Every feasible point of a small pure binary MIP."""
    n = mip.num_variables
    grid = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(float)
    return np.array([x for x in grid if mip.is_feasible(x)])


@pytest.mark.parametrize("seed", range(10))
def test_conflict_graph_conflicts_hold_at_every_feasible_point(seed):
    mip = random_binary_rows(seed)
    graph = jm.ConflictGraph([mip], mip.lower, mip.upper)
    points = binary_points(mip)
    assert len(points)
    literals = np.stack([points, 1 - points], axis=2).reshape(len(points), -1)
    # Literal 2j is x_j and 2j + 1 is 1 - x_j.
    assert np.array_equal(literals[:, 0::2], points)
    conflicts = 0
    for u in range(2 * mip.num_variables):
        for v in range(u + 1, 2 * mip.num_variables):
            if graph.adjacent(u, v):
                conflicts += 1
                assert graph.adjacent(v, u) and v in graph.neighbours(u)
                assert not np.any((literals[:, u] == 1) & (literals[:, v] == 1))
    # ``num_edges`` counts a pair once per row that gives it.
    assert conflicts <= graph.num_edges
    # And it holds every conflict of a single row: raising both literals
    # from the row's least activity passes its right-hand side.
    A, b = mip.A.toarray(), mip.b
    rows = [(A[r], b[r]) for r in range(len(b))] + [(-A[r], -b[r]) for r in range(mip.n_eq)]
    for a, rhs in rows:
        least = np.minimum(a, 0).sum()
        cost = np.zeros(2 * mip.num_variables)
        cost[0::2] = np.maximum(a, 0)
        cost[1::2] = np.maximum(-a, 0)
        for u in range(2 * mip.num_variables):
            for v in range(u + 1, 2 * mip.num_variables):
                if u // 2 != v // 2 and least + cost[u] + cost[v] > rhs + 1e-6:
                    assert graph.adjacent(u, v)


@pytest.mark.parametrize("seed", range(6))
def test_clique_cuts_are_valid(seed):
    mip = random_binary_rows(seed)
    graph = jm.ConflictGraph([mip], mip.lower, mip.upper)
    points = binary_points(mip)
    rng = np.random.default_rng(seed)
    for _ in range(5):
        x = rng.uniform(0, 1, mip.num_variables)
        for family, idx, coef, rhs in graph.cuts(x):
            assert family == "clique"
            assert coef @ x[idx] > rhs
            assert np.all(points[:, idx] @ coef <= rhs + 1e-12)


def odd_cliques(k=5):
    """max Σx over the edges of a complete graph on k nodes (x_i + x_j <= 1):
    the LP optimum is k/2 at x = 1/2, the MIP's 1."""
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
    A = np.zeros((len(pairs), k))
    for r, (i, j) in enumerate(pairs):
        A[r, [i, j]] = 1.0
    A_eq, b_eq = no_rows(k)
    lp = jl.LP(
        c=-np.ones(k), A_eq=A_eq, b_eq=b_eq, A_ineq=sp.csc_matrix(A),
        b_ineq=np.ones(len(pairs)), lower_bounds=np.zeros(k), upper_bounds=np.ones(k),
    )
    return jm.MIP(lp, np.ones(k, bool))


def test_clique_cut_from_edge_rows_and_in_the_search():
    mip = odd_cliques()
    graph = jm.ConflictGraph([mip], mip.lower, mip.upper)
    cuts = graph.cuts(np.full(5, 0.5))
    # The five edge rows give one clique on all five nodes: Σx <= 1.
    assert len(cuts) == 1
    _, idx, coef, rhs = cuts[0]
    assert list(idx) == list(range(5)) and np.all(coef == 1.0) and rhs == 1.0
    # x0 + x1 + y <= 1.5 with continuous y: y >= 0.6 makes x0 and x1
    # conflict; with y unbounded below the row gives no conflicts.
    for y_lower, conflict in ((0.6, True), (-0.6, False), (-np.inf, False)):
        lp = jl.LP(
            c=np.zeros(3), A_eq=sp.csc_matrix((0, 3)), b_eq=np.zeros(0),
            A_ineq=sp.csc_matrix([[1.0, 1.0, 1.0]]), b_ineq=np.array([1.5]),
            lower_bounds=np.array([0.0, 0.0, y_lower]), upper_bounds=np.array([1.0, 1.0, 2.0]),
        )
        row = jm.MIP(lp, np.array([True, True, False]))
        graph = jm.ConflictGraph([row], row.lower, row.upper)
        assert graph.adjacent(0, 2) == conflict
    result = solve(mip)
    assert_solved(result, mip, reference(mip))
    assert result["cuts"]["clique"] >= 1
    assert solve(mip, clique_cuts=False)["cuts"]["clique"] == 0


def odd_cycle(k=5):
    """max Σx with x_i + x_{i+1} <= 1 around a cycle of odd length k: the
    LP optimum is k/2 at x = 1/2, the MIP's (k − 1)/2. Its cliques are the
    edges, so only an odd-cycle (zero-half) cut closes the gap."""
    A = np.zeros((k, k))
    for i in range(k):
        A[i, [i, (i + 1) % k]] = 1.0
    A_eq, b_eq = no_rows(k)
    lp = jl.LP(
        c=-np.ones(k), A_eq=A_eq, b_eq=b_eq, A_ineq=sp.csc_matrix(A),
        b_ineq=np.ones(k), lower_bounds=np.zeros(k), upper_bounds=np.ones(k),
    )
    return jm.MIP(lp, np.ones(k, bool))


def zero_half(mip, x):
    return jm.zero_half_cuts(
        mip.A, mip.b, mip.n_eq, x, mip.lower, mip.upper, mip.integrality
    )


def test_zero_half_cut_on_an_odd_cycle():
    mip = odd_cycle()
    cuts = zero_half(mip, np.full(5, 0.5))
    # Half the five edge rows: Σx <= 5/2, rounded down to 2.
    assert len(cuts) == 1
    family, idx, coef, rhs = cuts[0]
    assert family == "zerohalf"
    assert list(idx) == list(range(5)) and np.all(coef == 1.0)
    assert rhs == pytest.approx(2.0)
    # At an integral point nothing is violated.
    assert zero_half(mip, np.array([1.0, 0, 1, 0, 0])) == []


def small_general_integer(seed, n=5, m_eq=1, m_ineq=4):
    """General integers in [0, 3] (1,024 points) with rows a random point
    satisfies."""
    rng = np.random.default_rng(seed)
    x0 = rng.integers(0, 4, n).astype(float)
    A_eq = rng.integers(-2, 3, (m_eq, n)).astype(float)
    A_ineq = rng.integers(-3, 4, (m_ineq, n)).astype(float)
    lp = jl.LP(
        c=rng.standard_normal(n), A_eq=sp.csc_matrix(A_eq), b_eq=A_eq @ x0,
        A_ineq=sp.csc_matrix(A_ineq), b_ineq=A_ineq @ x0 + rng.integers(0, 3, m_ineq),
        lower_bounds=np.zeros(n), upper_bounds=np.full(n, 3.0),
    )
    return jm.MIP(lp, np.ones(n, bool))


@pytest.mark.parametrize("seed", range(8))
def test_zero_half_cuts_hold_at_every_feasible_point(seed):
    rng = np.random.default_rng(seed)
    for mip, values in ((random_binary_rows(seed), 2), (small_general_integer(seed), 4)):
        n = mip.num_variables
        grid = np.array(np.meshgrid(*[np.arange(values)] * n)).reshape(n, -1).T.astype(float)
        points = np.array([x for x in grid if mip.is_feasible(x)])
        found = 0
        for _ in range(20):
            x = rng.uniform(mip.lower, mip.upper)
            for family, idx, coef, rhs in zero_half(mip, x):
                found += 1
                assert family == "zerohalf" and coef @ x[idx] > rhs
                assert np.all(points[:, idx] @ coef <= rhs + 1e-9)
        if seed == 0:
            assert found


@pytest.mark.parametrize("make", [general_integer, fixed_charge, generalised_assignment])
def test_zero_half_cuts_are_valid_on_mixed_rows(make):
    # Rows with continuous variables or fractional coefficients sit out; the
    # cuts from the rest hold over the MIP (checked with HiGHS).
    made = make()
    mip = made[0] if isinstance(made, tuple) else made
    x, _ = vertex_and_basis(mip)
    for _, idx, coef, rhs in zero_half(mip, x):
        assert coef @ x[idx] > rhs
        assert max_over_mip(mip, idx, coef) <= rhs + 1e-7 * (1 + abs(rhs))


def test_zero_half_cuts_in_the_search():
    mip = odd_cycle(7)
    result = solve(mip, clique_cuts=False, gomory_cuts=False)
    assert_solved(result, mip, reference(mip))
    assert result["cuts"]["zerohalf"] >= 1
    assert result["root_bound_before_cuts"] == pytest.approx(-3.5, abs=1e-4)
    assert solve(mip, zero_half_cuts=False)["cuts"]["zerohalf"] == 0


# %% Incumbent heuristics: RENS, dives, objective-aware propagation
def test_rens_returns_only_better_points(monkeypatch):
    mip = generalised_assignment(1)
    search = search_for(mip, monkeypatch, max_nodes=1, rens_seconds=0, rins_seconds=0, dive_every=0)
    search._run()
    monkeypatch.undo()
    x = search.incumbent[1] if search.incumbent is not None else np.zeros(mip.num_variables)
    # A point that is integral on most integers and fractional on a few.
    x = x.copy()
    x[:5] = 0.5
    cutoff = search.cutoff()
    search.rens_seconds = 5.0
    before = search.incumbent
    search.rens(x, search.root_lower, search.root_upper, None)
    if search.incumbent is not before:
        assert mip.is_feasible(search.incumbent[1])
        assert search.incumbent[0] < cutoff


def rens_trap(n_binaries=20):
    """A MIP whose RENS box is empty: ``y = 3w`` with ``y >= 1/2`` has the
    LP point ``y = 1/2, w = 1/6``, but every integer point has ``y >= 3``;
    ``n_binaries`` binaries with cost 1 are integral (zero) there. Returns
    the MIP and its LP point."""
    n = 2 + n_binaries
    A_eq = sp.lil_matrix((1, n))
    A_eq[0, 0], A_eq[0, 1] = 1.0, -3.0
    A_ineq = sp.lil_matrix((1, n))
    A_ineq[0, 0] = -1.0
    c = np.ones(n)
    c[1] = 0.0
    lp = jl.LP(
        c=c, A_eq=sp.csc_matrix(A_eq), b_eq=np.zeros(1),
        A_ineq=sp.csc_matrix(A_ineq), b_ineq=np.array([-0.5]),
        lower_bounds=np.zeros(n),
        upper_bounds=np.concatenate([[10.0, 3.0], np.ones(n_binaries)]),
    )
    x = np.zeros(n)
    x[0], x[1] = 0.5, 1.0 / 6.0
    return jm.MIP(lp, np.ones(n, bool)), x


def test_dive_stops_at_its_deadline(monkeypatch):
    # Dives re-solve the LP at every step; past their deadline they stop.
    mip = general_integer()
    search = search_for(mip, monkeypatch, **no_heuristics())
    lower, upper = mip.lower.copy(), mip.upper.copy()
    search.root_lower, search.root_upper = lower, upper
    result = search.solve_lp(search.node_lp(lower, upper), None, 2000)
    x = np.asarray(result["solution"].primal, np.float64)
    assert np.any(np.abs(x - np.round(x)) > 1e-3)  # a dive has work to do
    calls = count_lp_solves(monkeypatch)
    warm = (result["solution"], result["opt_state"])
    search.dive(x, lower, upper, warm, rule="fractional", deadline=time.time() - 1.0)
    assert calls == []
    search.dive(x, lower, upper, warm, rule="fractional")  # the default budget
    assert len(calls) > 0
    assert search.dive_seconds() > 0


@pytest.mark.parametrize("rule", ["fractional", "guided", "pseudocost"])
def test_dive_finds_feasible_points_or_nothing(rule, monkeypatch):
    mip = generalised_assignment(2)
    search = search_for(mip, monkeypatch, max_nodes=20, rens_seconds=0, rins_seconds=0)
    search._run()
    monkeypatch.undo()
    lp = search.node_lp(search.root_lower, search.root_upper)
    result = search.solve_lp(lp, None, 200)
    x = np.asarray(result["solution"].primal, np.float64)
    before = search.incumbent
    search.dive(x, search.root_lower, search.root_upper,
                (result["solution"], result["opt_state"]), rule=rule)
    if search.incumbent is not before:
        assert mip.is_feasible(search.incumbent[1])
        assert before is None or search.incumbent[0] < before[0]


def test_objective_propagation_rejects_non_improving_roundings(monkeypatch):
    # Knapsack (as a minimisation of −value): with the cutoff at the value of
    # the empty knapsack, fixing everything to 0 cannot improve.
    mip = knapsack()
    search = search_for(mip, monkeypatch)
    search.root_lower, search.root_upper = mip.lower.copy(), mip.upper.copy()
    search.initial_cutoff = -1.0  # better than 0 (the empty knapsack)
    propagator = search.heuristic_propagator()
    zeros = np.zeros(mip.num_variables)
    _, _, infeasible = propagator(zeros, zeros.copy())
    assert infeasible
    search.use_objective_propagation = False
    _, _, infeasible = search.heuristic_propagator()(zeros, zeros.copy())
    assert not infeasible


@pytest.mark.parametrize("make", [knapsack, general_integer])
def test_per_call_precision(make):
    import jax
    import jax.numpy as jnp

    mip = make()  # built in float64
    assert_solved(solve(mip, precision="float32"), mip, reference(mip))
    assert jax.config.jax_enable_x64 and mip.lp.c.dtype == jnp.float64


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"batch_size": 4},
        {"branching": "reliability"},
        {"branching": "reliability", "strong_backend": "numpy"},
        {"tree_cuts": True},
    ],
)
def test_node_precision(monkeypatch, options):
    seen = {"node": set(), "root": set()}
    solve_lp = jm._BranchAndBound.solve_lp

    def spy_solve_lp(self, lp, warm, max_epochs, **overrides):
        # Node LPs pass the node precision; the root and its cut rounds not.
        seen["node" if "precision" in overrides else "root"].add(str(lp.c.dtype))
        return solve_lp(self, lp, warm, max_epochs, **overrides)

    monkeypatch.setattr(jm._BranchAndBound, "solve_lp", spy_solve_lp)
    mip = general_integer()
    result = solve(mip, precision="float64", node_precision="float32", **options)
    assert_solved(result, mip, reference(mip))
    assert seen["root"] == {"float64"}
    if options.get("batch_size", 1) == 1:  # serial node LPs go through solve_lp
        assert seen["node"] == {"float32"}


def test_node_precision_above_the_search_precision():
    import jax.numpy as jnp

    mip = general_integer()
    result = solve(mip, precision="float32", node_precision="float64", batch_size=4)
    assert_solved(result, mip, reference(mip))
    assert mip.lp.c.dtype == jnp.float64


def test_node_precision_through_node_lp_options():
    mip = general_integer()
    result = solve(mip, node_lp_options={"precision": "float32"}, batch_size=4)
    assert_solved(result, mip, reference(mip))
    with pytest.raises(ValueError, match="disagree"):
        solve(mip, node_precision="float16", node_lp_options={"precision": "float32"})
    with pytest.raises(ValueError, match="float8"):
        solve(mip, node_precision="float8")


def test_lp_repair_after_lower_precision_batched_nodes():
    # Batched node LPs at float16 used the LP's sort permutations with 64-bit
    # off; repairing a node's rounding at float64 then failed in JAX's
    # indexing ("lax.add requires arguments to have the same dtypes").
    mip = facility_location()
    result = solve(
        mip,
        precision="float64",
        node_precision="float16",
        batch_size=32,
        fix_and_propagate=False,
        heuristic_every=5,
        cuts=False,
    )
    assert_solved(result, mip, reference(mip))


@pytest.mark.parametrize(
    "options",
    [
        {"node_lp_options": {"update_mode": "halpern"}},
        {"lp_options": {"update_mode": "halpern"}},
        {
            "lp_options": {"update_mode": "halpern"},
            "node_lp_options": {"update_mode": "pdhg"},
        },
    ],
)
def test_root_and_node_lps_in_different_update_modes(options):
    # Halpern's step state carries an anchor point that the other modes do
    # not, and cut rounds change the LP's rows under it. Node LPs warm-started
    # from the root (or cut rounds from an earlier round) used to fail.
    mip = general_integer()
    result = solve(mip, batch_size=1, tree_cuts=True, **options)
    assert_solved(result, mip, reference(mip))


def test_lp_options_cannot_set_device_or_precision():
    for key, value in (("device", "cpu"), ("precision", "float32")):
        with pytest.raises(ValueError, match=key):
            solve(knapsack(), lp_options={key: value})


def test_device_accepts_jax_device():
    import jax

    mip = knapsack()
    assert_solved(solve(mip, device=jax.devices("cpu")[0]), mip, reference(mip))


# %% Reliability branching
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize(
    "make", [knapsack, general_integer, facility_location, generalised_assignment]
)
def test_reliability_branching_matches_reference(make, batch_size):
    mip = make()
    result = solve(mip, branching="reliability", batch_size=batch_size)
    assert_solved(result, mip, reference(mip))
    if result["nodes"] > 1:
        assert result["strong_branch_lps"] > 0


def test_reliability_branching_strong_branches_and_keeps_the_optimum():
    # With PDHG's truncated child LPs (``strong_epochs``) reliability
    # branching does not always take fewer nodes than pseudocost branching
    # (exact HiGHS children did: 14 nodes against 21 here).
    mip = general_integer()
    strong = solve(mip, branching="reliability", zero_half_cuts=False)
    assert_solved(strong, mip, reference(mip))
    assert strong["strong_branch_nodes"] > 0


def no_heuristics():
    return dict(
        cuts=False,
        heuristic_every=0,
        fix_and_propagate=False,
        bulk_rounding=False,
        feasibility_jump=False,
        feasibility_pump=False,
        rens_seconds=0.0,
        rins_seconds=0.0,
        round_repair=False,
    )


def test_strong_branching_fixes_a_variable_with_an_infeasible_child():
    # min y with y + a >= 1, y + b >= 1, a + b <= 1, all binary. The LP
    # point is y = a = b = 1/2, and nothing is fixed at the root, but y = 0
    # forces a = b = 1: propagation empties y's down child, so y is fixed
    # to 1.
    lp = jl.LP(
        c=np.array([1.0, 0.0, 0.0]),
        A_eq=sp.csc_matrix((0, 3)),
        b_eq=np.zeros(0),
        A_ineq=sp.csc_matrix(
            np.array([[-1.0, -1.0, 0.0], [-1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
        ),
        b_ineq=np.array([-1.0, -1.0, 1.0]),
        lower_bounds=np.zeros(3),
        upper_bounds=np.ones(3),
    )
    mip = jm.MIP(lp, np.ones(3, bool))
    # Probing would fix y at the root (the same argument, before the search).
    result = solve(mip, branching="reliability", probing=False, **no_heuristics())
    assert result["status"] == "optimal"
    assert result["objective"] == pytest.approx(1.0, abs=1e-6)
    assert result["strong_fixings"] >= 1
    probed = solve(mip, branching="reliability", **no_heuristics())
    assert probed["probing"]["fixed"] >= 1
    assert probed["objective"] == pytest.approx(1.0, abs=1e-6)


def test_strong_branching_bounds_stay_valid():
    # The node bounds strong branching raises must stay below the optimum.
    mip = general_integer(n=12, m_ineq=8)
    ref = reference(mip)
    for max_nodes in (2, 4, 8):
        result = solve(
            mip, branching="reliability", max_nodes=max_nodes, **no_heuristics()
        )
        assert result["bound"] <= ref + 1e-6 * (1 + abs(ref))
        if result["objective"] is not None:
            assert result["objective"] >= ref - 1e-6 * (1 + abs(ref))


def test_strong_branching_with_tree_cuts():
    # Tree cuts change the LP's shape, so the children are solved one by one.
    mip = general_integer()
    result = solve(mip, branching="reliability", tree_cuts=True, tree_cut_every=1)
    assert_solved(result, mip, reference(mip))
    assert result["strong_branch_lps"] > 0


def test_strong_branching_depth_limit():
    mip = general_integer()
    result = solve(mip, branching="reliability", strong_max_depth=-1)
    assert_solved(result, mip, reference(mip))
    assert result["strong_branch_nodes"] == 0


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_strong_backends_solve_to_the_reference(backend):
    for make in (general_integer, facility_location):
        mip = make()
        result = solve(mip, branching="reliability", strong_backend=backend)
        assert_solved(result, mip, reference(mip))


def test_strong_backend_is_checked():
    with pytest.raises(ValueError, match="strong_backend"):
        solve(general_integer(), branching="reliability", strong_backend="cuda")


def test_reliability_branching_in_float32():
    mip = general_integer()
    result = solve(mip, branching="reliability", precision="float32")
    assert_solved(result, mip, reference(mip))
    assert result["strong_branch_lps"] > 0


def test_strong_branching_with_a_small_screen():
    # More candidates than screen slots: the unreliable ones go first.
    mip = general_integer(n=12, m_ineq=8)
    result = solve(
        mip, branching="reliability", strong_screen=2, strong_candidates=2
    )
    assert_solved(result, mip, reference(mip))


# %% Root sweep and root crossover
def sizing(periods=8, seed=3):
    """
    Buildingenergy in miniature: two technologies, each with an integer
    size ``N_k`` (units bought, the only integer cost: 40 and 400 a unit)
    and an integer count ``n_kt`` of units running each period, with
    ``n_kt <= N_k`` and output ``n_kt <= x_kt <= 5 n_kt``; demand
    ``Σ_k x_kt + s_t >= d_t`` is met by output (1 a unit) or bought in (10 a
    unit). No variable has an upper bound, so node bounds from PDHG duals
    are not rigorous (``assert_solved(rigorous=False)``). Branching on the
    cheap size moves the bound; the counts ``n_kt`` hardly do. Variables: ``N`` (2),
    then ``n``, ``x`` (2·periods each, technology-major), then ``s``.
    """
    rng = np.random.default_rng(seed)
    d = rng.uniform(3.0, 23.0, periods)
    T, K = periods, 2
    n_var = K + 2 * K * T + T

    def N(k):
        return k

    def n_(k, t):
        return K + k * T + t

    def x_(k, t):
        return K + K * T + k * T + t

    def s_(t):
        return K + 2 * K * T + t

    rows, b = [], []
    for k in range(K):
        for t in range(T):
            rows.append({n_(k, t): 1.0, N(k): -1.0}); b.append(0.0)
            rows.append({n_(k, t): 1.0, x_(k, t): -1.0}); b.append(0.0)
            rows.append({x_(k, t): 1.0, n_(k, t): -5.0}); b.append(0.0)
    for t in range(T):
        rows.append({x_(0, t): -1.0, x_(1, t): -1.0, s_(t): -1.0}); b.append(-d[t])
    A = sp.lil_matrix((len(rows), n_var))
    for i, row in enumerate(rows):
        for j, v in row.items():
            A[i, j] = v
    c = np.zeros(n_var)
    c[N(0)], c[N(1)] = 40.0, 400.0
    c[K + K * T : K + 2 * K * T] = 1.0
    c[K + 2 * K * T :] = 10.0
    A_eq, b_eq = no_rows(n_var)
    lp = jl.LP(
        c=c,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ineq=sp.csc_matrix(A),
        b_ineq=np.array(b),
        lower_bounds=np.zeros(n_var),
        upper_bounds=np.full(n_var, np.inf),
    )
    integrality = np.zeros(n_var, bool)
    integrality[: K + K * T] = True
    return jm.MIP(lp, integrality)


def test_branching_impact_ranks_sizes_first():
    mip = sizing()
    impact = jm.branching_impact(mip)
    counts = impact[2:18]
    assert impact[0] > counts.max() and impact[1] > counts.max()
    assert np.all(impact[~mip.integrality] == 0.0)


def root_branches(monkeypatch):
    """The variable each depth-0 split is on, recorded."""
    calls = []
    real = jm._BranchAndBound.branch

    def recorded(self, node, lower, upper, x, var, *args, **kwargs):
        if node.depth == 0:
            calls.append(int(var))
        return real(self, node, lower, upper, x, var, *args, **kwargs)

    monkeypatch.setattr(jm._BranchAndBound, "branch", recorded)
    return calls


@pytest.mark.parametrize("branching", ["pseudocost", "reliability"])
def test_root_sweep_branches_on_the_size(monkeypatch, branching):
    mip = sizing()
    calls = root_branches(monkeypatch)
    result = solve(mip, branching=branching, **no_heuristics())
    assert_solved(result, mip, reference(mip), rigorous=False)
    assert result["root_strong_lps"] > 0
    # The cheap technology's size (0), or its peak period's count (4), which
    # equals it at the root and has the same child bounds: a tie the root
    # LP point's last digits break.
    assert calls and calls[0] in (0, 4)


def test_root_sweep_off():
    mip = sizing()
    result = solve(mip, root_strong_candidates=0, **no_heuristics())
    assert_solved(result, mip, reference(mip), rigorous=False)
    assert result["root_strong_lps"] == 0
    assert solve(mip, branching="most_fractional")["root_strong_lps"] == 0


@pytest.mark.parametrize("batch_size", [1, 4])
def test_root_sweep_solves_to_the_reference(batch_size):
    for make in (knapsack, general_integer, facility_location):
        mip = make()
        result = solve(mip, batch_size=batch_size)
        assert_solved(result, mip, reference(mip))


@pytest.mark.parametrize("crossover", [True, False])
def test_root_crossover_makes_reduced_cost_fixing_rigorous(monkeypatch, crossover):
    # PDHG duals at a loose tolerance leave tiny wrong-sign reduced costs on
    # variables with no upper bound: the root bound drops them and is not
    # rigorous, so reduced-cost fixing cannot use it. The crossover's exact
    # duals can: with an optimal incumbent, the dear size goes to 0.
    mip = sizing()
    search = search_for(mip, monkeypatch, root_crossover=crossover, **no_heuristics())
    lower, upper = mip.lower.copy(), mip.upper.copy()
    search.root_lower, search.root_upper = lower, upper
    result = search.solve_lp(search.node_lp(lower, upper), None, 5000)
    pdhg = result["solution"]
    real = search.dual_terms

    def noisy(solution):
        constant, r = real(solution)
        if solution is pdhg:
            r = r.copy()
            r[-1] = -1e-9  # a bought-in s_t, with no upper bound
        return constant, r

    search.dual_terms = noisy
    search.root_exact_terms(lower, upper, result)
    assert search.root_terms_rigorous == crossover
    assert search.root_crossover_used == crossover
    x, value = reference_point(mip)
    search.incumbent = (value, x)
    assert search.tighten_root()
    if crossover:
        assert search.reduced_cost_tightenings > 0
        assert search.root_upper[1] == 0.0  # the dear size
        assert lower[1] <= x[1] <= search.root_upper[1]  # the optimum stays
    else:
        assert search.reduced_cost_tightenings == 0


def test_integer_roundings_snap():
    ints = np.array([True, True, False])
    lower, upper = np.zeros(3), np.full(3, 10.0)
    x = np.array([2.0004, 3.4, 0.7])
    roundings = jm.integer_roundings(x, ints, lower, upper, snap=1e-3)
    # Up, then nearest (the same as down here, so dropped).
    assert [r.tolist() for r in roundings] == [[2.0, 4.0], [2.0, 3.0]]
    # Without snapping the noise rounds 2.0004 up.
    assert jm.integer_roundings(x, ints, lower, upper, snap=0.0)[0].tolist() == [3.0, 4.0]
    # Roundings stay in the box.
    capped = jm.integer_roundings(x, ints, lower, np.array([10.0, 3.0, 10.0]), 1e-3)
    assert capped[0].tolist() == [2.0, 3.0]


def test_round_repair_in_the_search():
    mip = sizing()
    result = solve(mip, **dict(no_heuristics(), round_repair=True))
    assert_solved(result, mip, reference(mip), rigorous=False)
    assert "U" in [entry["source"] for entry in result["history"]]
    assert result["round_repair_calls"] > 0 and result["round_repair_finds"] > 0
    off = solve(mip, **no_heuristics())
    assert off["round_repair_calls"] == 0
    assert "U" not in [entry["source"] for entry in off["history"]]
    assert result["nodes"] < off["nodes"]


def test_round_repair_is_off_in_sub_searches(monkeypatch):
    seen = []
    real = jm._BranchAndBound.__init__

    def recording(self, mip, **settings):
        seen.append(settings.get("round_repair"))
        real(self, mip, **settings)

    monkeypatch.setattr(jm._BranchAndBound, "__init__", recording)
    mip = facility_location()
    solve(mip, rens_seconds=2.0)
    assert seen[0] is True and len(seen) > 1  # RENS ran a sub-search
    assert all(flag is False for flag in seen[1:])


@pytest.mark.parametrize(
    "make", [knapsack, general_integer, facility_location, generalised_assignment]
)
def test_jax_propagation_matches_numpy(make):
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(3)
    mip = make()
    propagator = jm.BoundPropagator(
        mip.A, mip.row_lower, mip.row_upper, mip.integrality
    )
    m = mip.A.shape[0]
    with jax.enable_x64(True):
        data = jm._propagation_data(mip.A, mip.row_lower, mip.row_upper, mip.integrality)
        propagate = jax.jit(jax.vmap(lambda lo, up: jm._propagate_jax(data, lo, up, m)))
    ints = np.flatnonzero(mip.integrality)
    boxes = []
    for _ in range(48):
        lo, up = mip.lower.copy(), mip.upper.copy()
        for j in rng.choice(ints, size=min(ints.size, 4), replace=False):
            value = rng.integers(lo[j], min(up[j], lo[j] + 8) + 1)
            if rng.random() < 0.5:
                up[j] = value
            else:
                lo[j] = value
        boxes.append((lo, up))
    with jax.enable_x64(True):
        lowers, uppers, infeasible = jax.device_get(
            propagate(
                jnp.asarray(np.stack([b[0] for b in boxes])),
                jnp.asarray(np.stack([b[1] for b in boxes])),
            )
        )
    for i, (lo, up) in enumerate(boxes):
        new_lo, new_up, bad = propagator(lo, up)
        assert bool(infeasible[i]) == bad
        if not bad:
            np.testing.assert_allclose(lowers[i], new_lo, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(uppers[i], new_up, rtol=1e-9, atol=1e-9)


def test_jax_box_bound_matches_numpy():
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(4)
    n = 30
    for _ in range(20):
        r = rng.standard_normal(n) * (rng.random(n) < 0.7)
        lower = np.where(rng.random(n) < 0.2, -np.inf, rng.uniform(-5, 0, n))
        upper = np.where(rng.random(n) < 0.2, np.inf, rng.uniform(0, 5, n))
        tol = 0.3 if rng.random() < 0.5 else 0.0
        expected = jm.box_bound(1.5, r, lower, upper, reduced_cost_tolerance=tol)
        with jax.enable_x64(True):
            got = jax.device_get(
                jm._box_bound_jax(
                    jnp.float64(1.5), jnp.asarray(r), jnp.asarray(lower),
                    jnp.asarray(upper), tol,
                )
            )
        assert bool(got[1]) == expected[1]
        if np.isfinite(expected[0]):
            assert float(got[0]) == pytest.approx(expected[0], rel=1e-12, abs=1e-12)
        else:
            assert float(got[0]) == expected[0]


# %% Fused feasibility pump
def test_fused_pump_finds_points():
    # No fix-and-propagate, bulk rounding or (almost) root jumps, so the pump,
    # with jumps from its roundings, finds the first point. Bulk rounding
    # runs to a time share, so whether it beat the pump ("B") was down to
    # the runner.
    mip = generalised_assignment(2)
    settings = dict(
        max_nodes=1,
        cuts=False,
        fix_and_propagate=False,
        bulk_rounding=False,
        jump_iterations=1,
        rens_seconds=0.0,
        rins_seconds=0.0,
        round_repair=False,
    )
    fused = solve(mip, **settings)
    assert fused["x"] is not None and mip.is_feasible(fused["x"])
    assert fused["history"][0]["source"] == "F"
    assert fused["pump_jumps"] >= fused["pump_jump_finds"]


def test_fused_pump_jumps_can_be_turned_off():
    mip = generalised_assignment(5)
    result = solve(
        mip,
        max_nodes=1,
        cuts=False,
        fix_and_propagate=False,
        jump_iterations=1,
        pump_jump_iterations=0,
    )
    assert result["pump_jumps"] == 0


# %% Probing and bulk fix-and-propagate
def reference_point(mip):
    result = milp(
        mip.c,
        constraints=LinearConstraint(mip.A.toarray(), mip.row_lower, mip.row_upper),
        integrality=mip.integrality.astype(int),
        bounds=Bounds(mip.lower, mip.upper),
    )
    assert result.success
    return result.x, result.fun


def restricted_optimum(mip, lower, upper):
    result = milp(
        mip.c,
        constraints=LinearConstraint(mip.A.toarray(), mip.row_lower, mip.row_upper),
        integrality=mip.integrality.astype(int),
        bounds=Bounds(lower, upper),
    )
    return result.fun if result.success else None


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize(
    "make", [knapsack, general_integer, facility_location, generalised_assignment]
)
def test_probing_keeps_the_optimum(make, backend):
    mip = make()
    _, best = reference_point(mip)
    probed = jm.probe(
        mip.A, mip.row_lower, mip.row_upper, mip.integrality, mip.lower, mip.upper,
        backend=backend,
    )
    assert not probed.infeasible
    assert np.all(probed.lower >= mip.lower) and np.all(probed.upper <= mip.upper)
    assert restricted_optimum(mip, probed.lower, probed.upper) == pytest.approx(
        best, rel=1e-9, abs=1e-9
    )
    assert probed.probed == len(probed.cache) > 0


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_probing_backends_agree(backend):
    mip = facility_location()
    numpy = jm.probe(
        mip.A, mip.row_lower, mip.row_upper, mip.integrality, mip.lower, mip.upper,
        backend="numpy",
    )
    other = jm.probe(
        mip.A, mip.row_lower, mip.row_upper, mip.integrality, mip.lower, mip.upper,
        backend=backend,
    )
    assert other.tightened == numpy.tightened and other.fixed == numpy.fixed


@pytest.mark.parametrize("make", [general_integer, facility_location, generalised_assignment])
def test_probing_cache_implications_hold_at_the_optimum(make):
    mip = make()
    x, _ = reference_point(mip)
    x = np.where(mip.integrality, np.round(x), x)
    probed = jm.probe(
        mip.A, mip.row_lower, mip.row_upper, mip.integrality, mip.lower, mip.upper
    )
    for var in probed.cache.split:
        lower, upper = mip.lower.copy(), mip.upper.copy()
        assert probed.cache.implied(var, x[var], lower, upper)
        assert np.all(x >= lower - 1e-6) and np.all(x <= upper + 1e-6)


def test_probing_proves_an_empty_box():
    # 2 (x1 + ... + x4) = 3 over binaries.
    n = 4
    mip = jm.MIP(
        jl.LP(
            c=np.ones(n),
            A_eq=sp.csc_matrix(2.0 * np.ones((1, n))),
            b_eq=np.array([3.0]),
            A_ineq=sp.csc_matrix((0, n)),
            b_ineq=np.zeros(0),
            lower_bounds=np.zeros(n),
            upper_bounds=np.ones(n),
        ),
        np.ones(n, bool),
    )
    for backend in ("numpy", "jax"):
        probed = jm.probe(
            mip.A, mip.row_lower, mip.row_upper, mip.integrality, mip.lower, mip.upper,
            backend=backend,
        )
        assert probed.infeasible


@pytest.mark.parametrize("backend", ["numpy", "jax"])
@pytest.mark.parametrize("make", [knapsack, general_integer, generalised_assignment])
def test_bulk_fix_and_propagate_returns_feasible_points(make, backend):
    # Pure integer models: a fully fixed box that propagation accepts is a
    # feasible point.
    found = 0
    for seed in range(4):
        mip = make(seed=seed)
        x = np.asarray(jl.solve(mip.lp)["solution"].primal, np.float64)
        probed = jm.probe(
            mip.A, mip.row_lower, mip.row_upper, mip.integrality, mip.lower, mip.upper
        )
        points = jm.bulk_fix_and_propagate(
            [x] * 4, probed.lower, probed.upper, mip.integrality,
            mip.A, mip.row_lower, mip.row_upper, cache=probed.cache, backend=backend,
        )
        assert len(points) == 4
        for point in points:
            if point is not None:
                assert mip.is_feasible(point)
                found += 1
    if make is knapsack:
        assert found > 0


def test_bulk_rounding_in_the_search():
    for seed in range(3):
        mip = generalised_assignment(seed)
        # RENS off: on seed 0 its dive finds a point after bulk rounding fails.
        result = solve(
            mip, max_nodes=1, cuts=False, feasibility_jump=False, feasibility_pump=False,
            rens_seconds=0,
        )
        if result["x"] is not None:
            assert mip.is_feasible(result["x"])
            assert result["history"][0]["source"] in ("P", "B")
    off = solve(
        generalised_assignment(0), max_nodes=1, cuts=False, fix_and_propagate=False,
        feasibility_jump=False, feasibility_pump=False, rens_seconds=0,
    )
    assert off["x"] is None


# %% Batched children and roundings on the device
def test_tree_cut_refreshes_keep_the_child_kernels(monkeypatch):
    """The device child kernels are padded to the tree cut block's
    capacity, so refreshes that change the block's nonzeros reuse them
    (each new ``_ChildKernels`` is a compile; ns1830653 built 14 a
    minute)."""
    built = []
    real = jm._ChildKernels.__init__

    def init(self, *args, **kwargs):
        built.append(kwargs.get("nnz_capacity"))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(jm._ChildKernels, "__init__", init)
    mip = generalised_assignment(2)
    result = solve(
        mip, batch_size=4, children_backend="jax", tree_cuts=True,
        tree_cut_every=1, max_nodes=60, gomory_cuts=False,
    )
    assert result["tree_cuts_added"] > 0 and result["tree_cut_refreshes"] > 1
    assert len(built) == 1 and built[0] is not None



@pytest.mark.parametrize("capacity", [None, 4096])
@pytest.mark.parametrize("make", [general_integer, facility_location, generalised_assignment])
def test_child_kernels_match_the_host(make, capacity):
    # ``capacity``: the entries padded (a free row of ones) to that many.
    mip = make()
    rng = np.random.default_rng(5)
    propagator = jm.BoundPropagator(mip.A, mip.row_lower, mip.row_upper, mip.integrality)
    size, n = 4, mip.num_variables
    kernels = jm._ChildKernels(mip, True, 1e-9, size, nnz_capacity=capacity)
    if capacity:
        assert kernels.data["vals"].shape == (capacity,)
    ints = np.flatnonzero(mip.integrality)
    lower = np.tile(mip.lower, (size, 1))
    upper = np.tile(mip.upper, (size, 1))
    var = rng.choice(ints, size).astype(np.int32)
    xv = np.array([rng.uniform(lower[i, j], min(upper[i, j], lower[i, j] + 3)) for i, j in enumerate(var)])
    r = rng.standard_normal((size, n))
    constant = rng.standard_normal(size)
    out = kernels(lower, upper, var, xv, r, constant)
    cont = ~mip.integrality
    for i in range(size):
        for side, down in enumerate((True, False)):
            lo, up = lower[i].copy(), upper[i].copy()
            if down:
                up[var[i]] = min(up[var[i]], np.floor(xv[i]))
            else:
                lo[var[i]] = max(lo[var[i]], np.ceil(xv[i]))
            crossed = lo[var[i]] > up[var[i]]
            assert bool(out["crossed"][i, side]) == crossed
            if crossed:
                continue
            new_lo, new_up, empty = propagator(lo, up)
            assert bool(out["empty"][i, side]) == empty
            if empty:
                continue
            np.testing.assert_allclose(out["lower"][i, side], new_lo, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(out["upper"][i, side], new_up, rtol=1e-9, atol=1e-9)
            wlo = np.where(cont, new_lo - 1e-7 * (1 + np.abs(new_lo)), new_lo)
            wup = np.where(cont, new_up + 1e-7 * (1 + np.abs(new_up)), new_up)
            bound, rigorous = jm.box_bound(constant[i], r[i], wlo, wup, reduced_cost_tolerance=1e-9)
            assert bool(out["rigorous"][i, side]) == rigorous
            if np.isfinite(bound):
                assert float(out["bound"][i, side]) == pytest.approx(bound, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize(
    "make", [knapsack, general_integer, facility_location, generalised_assignment]
)
def test_batched_search_with_device_children(make):
    mip = make()
    result = solve(mip, batch_size=4, gomory_cuts=False, children_backend="jax")
    assert_solved(result, mip, reference(mip))
    if make is general_integer:  # it always branches in batches
        assert result["batched_children"] > 0


def test_batched_tree_roundings():
    mip = generalised_assignment(3)
    # Root cuts and heuristics off: with them the root can close the model
    # (a fast runner) or leave a few nodes (a slow one), so a tree is forced.
    result = solve(
        mip, batch_size=4, cuts=False, feasibility_pump=False, bulk_rounding=False,
        children_backend="jax", heuristic_every=1, batch_roundings=True,
    )
    assert_solved(result, mip, reference(mip))
    assert result["batches"] > 0 and result["batched_roundings"] > 0


def test_children_backend_is_checked():
    with pytest.raises(ValueError, match="children_backend"):
        solve(general_integer(), batch_size=4, children_backend="cuda")


def test_bulk_fix_and_propagate_with_a_box_per_start():
    mip = generalised_assignment(1)
    x = np.asarray(jl.solve(mip.lp)["solution"].primal, np.float64)
    boxes = []
    for j in range(3):
        lo, up = mip.lower.copy(), mip.upper.copy()
        up[j] = 0.0  # each start forbids a different assignment
        boxes.append((lo, up))
    points = jm.bulk_fix_and_propagate(
        [x] * 3, None, None, mip.integrality, mip.A, mip.row_lower, mip.row_upper,
        boxes=boxes, pad_to=8,
    )
    assert len(points) == 3
    for j, point in enumerate(points):
        if point is not None:
            assert mip.is_feasible(point) and point[j] == 0.0


# %% Continuous batching and the root-box cutoff
@pytest.mark.parametrize("make", [general_integer, facility_location, generalised_assignment])
def test_continuous_batching_keeps_the_optimum(make):
    # One epoch a call, 200 in all: almost every node LP goes back to the
    # queue.
    mip = make()
    result = solve(
        mip, batch_size=4, batch_epochs=1, node_max_epochs=200, gomory_cuts=False
    )
    assert_solved(result, mip, reference(mip))
    if make is general_integer:
        assert result["batch_requeues"] > 0


def test_requeued_node_lps_are_not_counted_as_nodes(monkeypatch):
    # Every LP slice is evaluated once; only the finished ones are nodes.
    # Only the main search's are counted, not its sub-searches' (RINS).
    evaluated, searches = [], []
    real = jm._BranchAndBound.evaluate
    real_init = jm._BranchAndBound.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        searches.append(self)

    def evaluate(self, *args, **kwargs):
        if self is searches[0]:
            evaluated.append(self.nodes)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(jm._BranchAndBound, "__init__", init)
    monkeypatch.setattr(jm._BranchAndBound, "evaluate", evaluate)
    mip = general_integer()
    result = solve(
        mip, batch_size=4, batch_epochs=1, node_max_epochs=200, gomory_cuts=False
    )
    assert_solved(result, mip, reference(mip))
    assert result["batch_requeues"] > 0
    # Each evaluation that was not a requeue is a node; a node can also be
    # closed without one (an earlier member's incumbent prunes it).
    assert len(evaluated) - result["batch_requeues"] <= result["nodes"]
    assert result["nodes"] < len(evaluated)  # the requeued slices are not


@pytest.mark.parametrize("make", [general_integer, facility_location, generalised_assignment])
def test_truncated_node_lps_branch_after_one_call(make, monkeypatch):
    """By default (``node_max_epochs=None``) a batched node LP gets one call
    of ``batch_epochs`` and branches on its point: no requeue, except for a
    node with every integer fixed, whose LP gets the full budget so it can
    close proven. Pseudocosts learn from the truncated LPs' bounds."""
    gains = []
    real = jm._BranchAndBound.record_gain

    def record_gain(self, var, down, gain):
        gains.append(gain)
        return real(self, var, down, gain)

    monkeypatch.setattr(jm._BranchAndBound, "record_gain", record_gain)
    mip = make()
    result = solve(mip, batch_size=4, batch_epochs=1, gomory_cuts=False)
    assert_solved(result, mip, reference(mip))
    if result["nodes"] > 3:
        assert gains  # the pseudocosts learnt, though few LPs converged


@pytest.mark.parametrize("pipeline", [True, False])
@pytest.mark.parametrize("make", [general_integer, facility_location, generalised_assignment])
def test_pipelined_batches_keep_the_optimum(make, pipeline, monkeypatch):
    """The pipelined batch loop (``pipeline_batches``: one batch evaluated
    on the host while the next is on the device, children two batches
    later) and the plain one reach the optimum with valid bounds, tree cut
    refreshes between dispatch and evaluation included, and leave nothing
    in flight."""
    searches = []
    real_init = jm._BranchAndBound.__init__

    def init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        searches.append(self)

    monkeypatch.setattr(jm._BranchAndBound, "__init__", init)
    mip = make()
    result = solve(
        mip, batch_size=4, pipeline_batches=pipeline, tree_cuts=True,
        tree_cut_every=1, gomory_cuts=False,
    )
    assert_solved(result, mip, reference(mip))
    search = searches[0]
    assert search.pipeline_batches == pipeline
    assert search.in_flight is None and search.pending_children is None
    assert search._device_thread is None  # shut down with the search


def test_stale_cut_duals_are_zeroed_for_a_batch_solved_before_a_refresh(monkeypatch):
    """A batch dispatched before a tree cut refresh is evaluated with the
    duals of the slots that refresh replaced zeroed (``stale_free``)."""
    zeroed = []
    real = jm._BranchAndBound.stale_free

    def stale_free(self, solution, version):
        out = real(self, solution, version)
        if version < self.cut_version:
            base_m = self.base.A.shape[0] - self.base.n_eq
            stale = [i for i, v in enumerate(self.slot_version) if v > version]
            duals = np.asarray(out.dual_ineq)
            zeroed.append(all(duals[base_m + i] == 0.0 for i in stale))
        return out

    monkeypatch.setattr(jm._BranchAndBound, "stale_free", stale_free)
    mip = generalised_assignment(2)
    result = solve(
        mip, batch_size=4, tree_cuts=True, tree_cut_every=1, max_nodes=80,
        gomory_cuts=False,
    )
    assert result["tree_cut_refreshes"] > 0
    assert zeroed and all(zeroed)


def test_auto_batch_size_follows_the_device(monkeypatch):
    # The tests run on the CPU: 32 a batch; a GPU or TPU would take 128.
    search = search_for(general_integer(), monkeypatch, batch_size="auto")
    assert search.batch_size == jm._AUTO_BATCH_CPU == 32
    assert jm._AUTO_BATCH_ACCELERATOR == 128
    with pytest.raises(ValueError, match="batch_size"):
        solve(general_integer(), batch_size="big")


def test_truncated_defaults_do_not_requeue(monkeypatch):
    search = search_for(general_integer(), monkeypatch, batch_size=4)
    assert search.batch_epochs == 2 and search.batch_node_epochs == 2
    assert search.node_max_epochs == 200  # the serial search, dives, the pump
    given = search_for(general_integer(), monkeypatch, batch_size=4, node_max_epochs=50)
    assert given.batch_node_epochs == 50


def test_integral_node_that_fails_the_check_is_split(monkeypatch):
    # A node whose integral LP point is turned down is split on a free
    # integer rather than closed on its LP bound, which would leave the
    # search "unresolved" with its bound held below the optimum.
    # The first integral point at a node with a free integer is turned
    # down (at a node with every integer fixed there is nothing to split).
    rejected, box = [], {}
    real, real_evaluate = jm._BranchAndBound.try_incumbent, jm._BranchAndBound._evaluate

    def _evaluate(self, node, lower, upper, lp, result):
        box["free"] = bool(np.any(self.is_int & (lower < upper)))
        return real_evaluate(self, node, lower, upper, lp, result)

    def try_incumbent(self, x, source, warm):
        if source == "T" and not rejected and box.get("free"):
            rejected.append(x)
            return False
        return real(self, x, source, warm)

    monkeypatch.setattr(jm._BranchAndBound, "_evaluate", _evaluate)
    monkeypatch.setattr(jm._BranchAndBound, "try_incumbent", try_incumbent)
    mip = knapsack()
    result = solve(mip, batch_size=1, **no_heuristics())
    assert rejected
    assert result["integral_splits"] >= 1
    assert result["unresolved_nodes"] == 0
    assert_solved(result, mip, reference(mip))


def test_integral_split_prefers_a_variable_in_a_broken_row():
    # x0 + x1 <= 1 broken by (1, 1, 0): x0 and x1 are in the row; x2 is not,
    # though its reduced cost is the largest.
    lp = jl.LP(
        c=np.array([-1.0, -1.0, -5.0]),
        A_eq=sp.csc_matrix((0, 3)),
        b_eq=np.zeros(0),
        A_ineq=sp.csc_matrix([[1.0, 2.0, 0.0]]),
        b_ineq=np.array([1.0]),
        lower_bounds=np.zeros(3),
        upper_bounds=np.ones(3),
    )
    mip = jm.MIP(lp, np.ones(3, bool))
    search = types.SimpleNamespace(
        is_int=mip.integrality, original=mip, feas_tol=1e-6
    )
    choose = jm._BranchAndBound.integral_split_variable
    x = np.array([1.0, 1.0, 0.0])
    terms = (0.0, np.array([0.1, 0.1, 9.0]))
    assert choose(search, x, np.zeros(3), np.ones(3), terms) == 1  # |a| = 2
    # x1 fixed: x0 is the free one in the row.
    assert choose(search, x, np.array([0.0, 1.0, 0.0]), np.ones(3), terms) == 0
    # No broken row: the largest reduced cost.
    assert choose(search, np.zeros(3), np.zeros(3), np.ones(3), terms) == 2
    # Nothing free.
    assert choose(search, x, x, x, terms) is None


@pytest.mark.parametrize(
    "settings",
    [{}, {"branching": "most_fractional", "root_strong_candidates": 0}],  # ties too
)
@pytest.mark.parametrize("rule", ["score", "estimate", "rounding"])
@pytest.mark.parametrize("make", [knapsack, general_integer, generalised_assignment])
def test_plunges_and_dives_take_the_best_scored_child(make, settings, rule, monkeypatch):
    # The plunge child (and, in a dive, the child popped next), by
    # ``plunge_child``: "score", the higher (1 + inferences) / (1 + cost /
    # unit), equal scores to the lower estimated bound, then to the rounding
    # direction of the LP value; "estimate", from the second step;
    # "rounding", the last only.
    checked, ties = [], []
    real_branch, real_push = jm._BranchAndBound.branch, jm._BranchAndBound.push
    pushed = []

    def push(self, node):
        pushed.append(node)
        return real_push(self, node)

    def branch(self, node, lower, upper, x, var, bound, *args, **kwargs):
        pushed.clear()
        before = self.plunge_node
        counts, sums = self.pc_count.copy(), self.pc_sum.copy()
        out = real_branch(self, node, lower, upper, x, var, bound, *args, **kwargs)
        kids = [n for n in pushed if n.depth == node.depth + 1]
        if self.plunge_node is not None and self.plunge_node is not before:
            chosen = self.plunge_node
            kids.append(chosen)
        elif self.diving() and len(kids) == 2:
            chosen = kids[-1]  # depth-first: the last pushed goes next
        else:
            return out
        if len(kids) != 2:
            return out  # one child pruned or infeasible: nothing to choose
        f = x[var] - np.floor(x[var])
        known = counts > 0
        unit = sums[known].sum() / counts[known].sum() if known.any() else 1.0
        unit = unit if unit > 0 else 1.0

        def estimate(child):
            down = child.branch[1]
            gain = 0.0
            if np.all(counts[:, var] > 0):
                side = 0 if down else 1
                gain = (f if down else 1.0 - f) * sums[side, var] / counts[side, var]
            return child.bound + gain

        def score(child):
            return (1.0 + child.inferences) / (1.0 + max(estimate(child) - bound, 0.0) / unit)

        other = kids[0] if kids[1] is chosen else kids[1]
        if rule == "score":
            a, b = score(chosen), score(other)
            if abs(a - b) > 1e-9 * (1.0 + abs(a) + abs(b)):
                checked.append(a > b)
                return out
        ea, eb = estimate(chosen), estimate(other)
        if rule != "rounding" and abs(ea - eb) > 1e-9 * (1.0 + abs(ea) + abs(eb)):
            checked.append(ea < eb)
        elif abs(f - 0.5) > 1e-6:
            ties.append(chosen.branch[1] == (f < 0.5))
        return out

    monkeypatch.setattr(jm._BranchAndBound, "branch", branch)
    monkeypatch.setattr(jm._BranchAndBound, "push", push)
    mip = make()
    result = solve(
        mip, batch_size=1, cuts=False, plunge_depth=1000,
        plunge_child=rule, **settings,
    )
    assert_solved(result, mip, reference(mip))
    assert checked or ties
    assert all(checked) and all(ties)


def test_plunge_child_is_checked():
    with pytest.raises(ValueError, match="plunge_child"):
        solve(knapsack(), plunge_child="deepest")


def test_implicit_integers_from_highs_stay_continuous():
    # HiGHS's presolve marks continuous columns that some optimum has
    # integral as implicit integers. They load as continuous (a point may
    # have them fractional; they are never branched on), marked
    # ``MIP.implied``, and count as integral for the objective step.
    import highspy

    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = 2, 1
    lp.col_cost_ = np.array([3.0, 2.0])
    lp.col_lower_ = np.zeros(2)
    lp.col_upper_ = np.array([1.0, 5.0])
    lp.row_lower_ = np.array([1.0])
    lp.row_upper_ = np.array([np.inf])
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = np.array([0, 1, 2], np.int32)
    lp.a_matrix_.index_ = np.array([0, 0], np.int32)
    lp.a_matrix_.value_ = np.array([1.0, 1.0])
    lp.integrality_ = [highspy.HighsVarType.kInteger, highspy.HighsVarType.kImplicitInteger]
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.passModel(lp)
    held = highs.getLp()
    assert int(held.integrality_[1]) == int(highspy.HighsVarType.kImplicitInteger)
    mip = jm._mip_from_highs_lp(held)
    assert mip.integrality.tolist() == [True, False]
    assert mip.implied.tolist() == [False, True]
    assert jm.objective_step(mip.c, mip.integrality) is None  # a cost on a continuous
    assert jm.objective_step(mip.c, mip.integrality | mip.implied) == 1.0


def test_implied_mask_is_checked_and_kept_to_continuous_columns():
    lp = jl.LP(
        c=np.array([1.0, 1.0, 0.5]),
        A_eq=sp.csc_matrix([[1.0, 1.0, -1.0]]),
        b_eq=np.zeros(1),
        A_ineq=sp.csc_matrix([[-1.0, -1.0, 0.0]]),
        b_ineq=np.array([-1.0]),
        lower_bounds=np.zeros(3),
        upper_bounds=np.array([1.0, 1.0, 2.0]),
    )
    mip = jm.MIP(lp, np.array([True, True, False]))
    assert not mip.implied.any()
    with pytest.raises(ValueError, match="implied"):
        jm.MIP(lp, np.ones(3, bool), implied=np.ones(2, bool))
    marked = jm.MIP(lp, np.array([True, True, False]), implied=np.ones(3, bool))
    assert marked.implied.tolist() == [False, False, True]  # continuous only
    assert marked.with_rows(sp.csr_matrix((1, 3)), np.zeros(1)).implied.tolist() == [False, False, True]


def _knapsack_with_an_implied_sum(seed=0, n=15):
    """``knapsack`` plus a continuous ``y = x0 + x1 + x2`` with a cost: an
    implied integer (integral whenever the binaries are), marked so."""
    base = knapsack(seed=seed, n=n)
    lp = base.lp.to_scipy()
    A_ineq = sp.hstack([sp.csr_matrix(lp.A_ineq), sp.csr_matrix((lp.A_ineq.shape[0], 1))]).tocsc()
    row = np.zeros(n + 1)
    row[:3], row[n] = 1.0, -1.0
    sense = 1.0 if base.maximise else -1.0
    implied = np.zeros(n + 1, bool)
    implied[n] = True
    mip = jm.MIP(
        jl.LP(
            c=np.append(lp.c, 0.37 * sense),
            A_eq=sp.csc_matrix(row[None, :]),
            b_eq=np.zeros(1),
            A_ineq=A_ineq,
            b_ineq=lp.b_ineq,
            lower_bounds=np.append(lp.lower_bounds, 0.0),
            upper_bounds=np.append(lp.upper_bounds, 3.0),
        ),
        np.append(base.integrality, False),
        base.offset,
        base.maximise,
        implied,
    )
    return mip


@pytest.mark.parametrize("on", [True, False])
def test_implied_integer_cuts_keep_the_optimum(on, monkeypatch):
    mip = _knapsack_with_an_implied_sum()
    seen = []
    real = jm.CutSeparator.__init__

    def init(self, *args, **kwargs):
        real(self, *args, **kwargs)
        seen.append(bool(self.is_int[-1]))

    monkeypatch.setattr(jm.CutSeparator, "__init__", init)
    result = solve(mip, batch_size=1, implied_integer_cuts=on)
    assert_solved(result, mip, reference(mip))
    assert seen and all(flag == on for flag in seen)  # y counts as an integer for cuts


def test_implied_columns_are_never_branched_on(monkeypatch):
    # facility_location's flows marked implied: still continuous, so the
    # search branches only on the openings and finds the same optimum.
    base = facility_location()
    implied = ~base.integrality
    mip = jm.MIP(base.lp, base.integrality, base.offset, base.maximise, implied)
    split = []
    real = jm._BranchAndBound.branch

    def branch(self, node, lower, upper, x, var, *args, **kwargs):
        split.append(var)
        return real(self, node, lower, upper, x, var, *args, **kwargs)

    monkeypatch.setattr(jm._BranchAndBound, "branch", branch)
    result = solve(mip, batch_size=1, **no_heuristics())
    assert_solved(result, mip, reference(mip))
    assert all(base.integrality[v] for v in split)


@pytest.mark.parametrize(
    "make", [knapsack, general_integer, generalised_assignment, facility_location]
)
def test_cut_pool_propagation_keeps_valid_cuts(make, monkeypatch):
    # Every archived cut holds at the optimum (it holds at every feasible
    # point), the search still finds the optimum, and its propagator runs
    # the archived cuts beside the rows.
    seen = {}
    real = jm._BranchAndBound.result

    def result(self, status):
        seen["archive"] = list(self.cut_archive.values())
        seen["pooled"] = self.propagator is not self.row_propagator
        return real(self, status)

    monkeypatch.setattr(jm._BranchAndBound, "result", result)
    mip = make()
    out = solve(mip, batch_size=1, gomory_cuts=True, cut_pool_propagation=True)
    assert_solved(out, mip, reference(mip))
    assert out["cut_pool_rows"] == len(seen["archive"])
    limit = max(32, mip.num_variables // 8)
    x, _ = reference_point(mip)
    for idx, coef, rhs in seen["archive"]:
        assert 0 < idx.size <= limit
        assert coef @ x[idx] <= rhs + 1e-6 * (1 + abs(rhs))
    if seen["archive"]:
        assert seen["pooled"]


def test_no_cut_pool_without_cut_pool_propagation():
    mip = generalised_assignment()
    out = solve(mip, batch_size=1, gomory_cuts=True)
    assert out["cut_pool_rows"] == 0


def test_children_count_their_propagation_fixings(monkeypatch):
    # max x0 + x1 + x2 with 3 x0 + 3 x1 + 3 x2 <= 4 over binaries: the LP
    # point is 4/9 each. Up on one fixes the other two to 0 (2 inferences);
    # down fixes nothing.
    lp = jl.LP(
        c=-np.ones(3),
        A_eq=sp.csc_matrix((0, 3)),
        b_eq=np.zeros(0),
        A_ineq=sp.csc_matrix([[3.0, 3.0, 3.0]]),
        b_ineq=np.array([4.0]),
        lower_bounds=np.zeros(3),
        upper_bounds=np.ones(3),
    )
    mip = jm.MIP(lp, np.ones(3, bool))
    seen = []
    real_branch, real_push = jm._BranchAndBound.branch, jm._BranchAndBound.push
    pushed = []

    def push(self, node):
        pushed.append(node)
        return real_push(self, node)

    def branch(self, node, *args, **kwargs):
        pushed.clear()
        out = real_branch(self, node, *args, **kwargs)
        kids = pushed + ([self.plunge_node] if self.plunge_node is not None else [])
        seen.append({child.branch[1]: child.inferences for child in kids if child.depth == node.depth + 1})
        return out

    monkeypatch.setattr(jm._BranchAndBound, "branch", branch)
    monkeypatch.setattr(jm._BranchAndBound, "push", push)
    result = solve(
        mip, batch_size=1, branching="most_fractional",
        root_strong_candidates=0, coefficient_tightening=False, probing=False,
        **no_heuristics(),
    )
    assert_solved(result, mip, reference(mip))
    first = seen[0]
    assert first == {True: 0, False: 2}  # down: none; up: the other two


def test_batch_epochs_is_checked():
    with pytest.raises(ValueError, match="batch_epochs"):
        solve(general_integer(), batch_size=4, batch_epochs=0)


def test_box_cutoff_is_above_every_feasible_point():
    for make in (knapsack, general_integer, generalised_assignment):
        mip = make()
        search = jm._BranchAndBound(mip, **jm_settings())
        search.run()
        ref = reference(mip)
        assert np.isfinite(search.box_cutoff)
        assert search.box_cutoff > mip.sign * ref - mip.offset


def jm_settings():
    import inspect
    import time

    defaults = {
        key: value.default
        for key, value in inspect.signature(jm.solve).parameters.items()
        if value.default is not inspect.Parameter.empty
    }
    defaults.update(PINNED)
    for key in ("max_seconds", "presolve", "highs_options", "precision"):
        defaults.pop(key, None)
    defaults.update(deadline=None, start=time.time())
    return defaults


def test_infeasible_mip_is_still_infeasible_with_the_box_cutoff():
    # x1 + x2 = 1.5 over integers in [0, 3]: bounded, infeasible.
    mip = jm.MIP(
        jl.LP(
            c=np.array([1.0, 2.0]),
            A_eq=sp.csc_matrix(np.array([[1.0, 1.0]])),
            b_eq=np.array([1.5]),
            A_ineq=sp.csc_matrix((0, 2)),
            b_ineq=np.zeros(0),
            lower_bounds=np.zeros(2),
            upper_bounds=np.full(2, 3.0),
        ),
        np.ones(2, bool),
    )
    for batch_size in (1, 4):
        assert solve(mip, batch_size=batch_size)["status"] == "infeasible"


# %% Node LP options
def test_node_lp_options_reach_node_lps_only(monkeypatch):
    seen = {"node": [], "root": []}
    solve_lp = jm._BranchAndBound.solve_lp

    def spy_solve_lp(self, lp, warm, max_epochs, **overrides):
        # The first LP is the root's (no cut rounds); the rest are nodes'.
        side = "node" if self.nodes else "root"
        seen[side].append(dict(self.lp_kwargs, **overrides).get("iterations_per_epoch"))
        return solve_lp(self, lp, warm, max_epochs, **overrides)

    monkeypatch.setattr(jm._BranchAndBound, "solve_lp", spy_solve_lp)
    mip = general_integer()
    result = solve(
        mip, node_lp_options={"iterations_per_epoch": 64}, feasibility_pump=False,
        cuts=False,
    )
    assert_solved(result, mip, reference(mip))
    assert seen["node"] and all(value == 64 for value in seen["node"])
    assert seen["root"] and all(value is None for value in seen["root"])


def test_batched_node_lps_get_node_settings(monkeypatch):
    import functools

    calls = []
    solve_many = jl.solve_many

    @functools.wraps(solve_many)  # keeps the signature the check reads
    def spy(lp, lower_bounds, upper_bounds, **kwargs):
        calls.append(kwargs)
        return solve_many(lp, lower_bounds, upper_bounds, **kwargs)

    monkeypatch.setattr(jl, "solve_many", spy)
    mip = general_integer()
    result = solve(
        mip,
        batch_size=4,
        gomory_cuts=False,
        lp_tolerance=1e-6,
        node_lp_tolerance=3e-5,
        node_lp_options={"iterations_per_epoch": 128},
    )
    assert_solved(result, mip, reference(mip))
    assert calls
    for kwargs in calls:
        assert kwargs["iterations_per_epoch"] == 128
        assert kwargs["primal_feasibility_tolerance"] == 3e-5


def test_batched_node_lps_start_from_their_parents_step_sizes(monkeypatch):
    import functools

    calls = []
    solve_many = jl.solve_many

    @functools.wraps(solve_many)
    def spy(lp, lower_bounds, upper_bounds, **kwargs):
        calls.append(kwargs.get("initial_opt_states"))
        return solve_many(lp, lower_bounds, upper_bounds, **kwargs)

    monkeypatch.setattr(jl, "solve_many", spy)
    mip = general_integer()
    result = solve(mip, batch_size=4, gomory_cuts=False)
    assert_solved(result, mip, reference(mip))
    # Every node batch descends from the root, whose solve has step sizes.
    assert calls and all(steps is not None for steps in calls)
    for k, eta in calls:
        assert np.all(np.isfinite(np.asarray(k))) and np.all(np.asarray(eta) > 0)


def test_node_lp_options_are_checked():
    with pytest.raises(ValueError, match="node_lp_options cannot set max_epochs"):
        solve(general_integer(), node_lp_options={"max_epochs": 5})
    with pytest.raises(ValueError, match="not_an_option"):
        solve(
            general_integer(), batch_size=4,
            node_lp_options={"not_an_option": 1},
        )


@NEEDS_NUMBA
def test_numba_jump_is_deterministic():
    # Chunk sizes follow a fixed schedule, so a run depends only on its inputs.
    mip = generalised_assignment(1)
    start = np.zeros(mip.num_variables)
    runs = [
        jm.feasibility_jump(mip, start, seed=4, improve=True, max_iterations=50_000)
        for _ in range(2)
    ]
    np.testing.assert_array_equal(runs[0][0], runs[1][0])
    assert runs[0][1:] == runs[1][1:]


def test_jaddle_node_lps_still_batch():
    mip = generalised_assignment(1)
    result = solve(mip, batch_size=4, root_strong_candidates=0)
    assert_solved(result, mip, reference(mip))
    if result["nodes"] > 1:
        assert result["batches"] > 0


def test_rounded_point_is_repaired(monkeypatch):
    # The optimum's openings with its flows lost: rows break, and the repair
    # LP (openings fixed, flows free) has to find flows again.
    mip = facility_location()
    best = optimum(mip)
    x = best.copy()
    x[~mip.integrality] = 0.0
    assert not mip.is_feasible(x)
    search = search_for(mip, monkeypatch, repair_max_epochs=2000)
    assert search.try_incumbent(x, "T", None)
    assert search.lp_repairs == 1
    value, point = search.incumbent
    assert mip.is_feasible(point)
    assert value == pytest.approx(mip.c @ best, rel=1e-6)


# %% Symmetry
def identical_machines(items=7, machines=3, seed=0):
    """Items to identical machines: ``Σ_j x[i, j] = 1`` for each item and a
    capacity row for each machine; item ``i`` costs ``c_i`` wherever it
    goes. Variables are item-major (``x[i, j]`` is ``i·machines + j``)."""
    rng = np.random.default_rng(seed)
    w = rng.integers(3, 10, items).astype(float)
    c = np.repeat(rng.integers(1, 6, items).astype(float), machines)
    n = items * machines
    A_eq = sp.lil_matrix((items, n))
    A_ineq = sp.lil_matrix((machines, n))
    for i in range(items):
        for j in range(machines):
            A_eq[i, i * machines + j] = 1.0
            A_ineq[j, i * machines + j] = w[i]
    lp = jl.LP(
        c=c - 3.0, A_eq=sp.csc_matrix(A_eq), b_eq=np.ones(items),
        A_ineq=sp.csc_matrix(A_ineq), b_ineq=np.full(machines, 0.45 * w.sum()),
        lower_bounds=np.zeros(n), upper_bounds=np.ones(n),
    )
    return jm.MIP(lp, np.ones(n, bool))


def bin_packing(items=6, bins=3, seed=1):
    """Bin packing with identical bins: ``x[i, j]`` (item-major) and a
    used flag ``y_j`` per bin (last), ``Σ_i w_i x[i, j] <= C y_j``,
    minimising the bins used."""
    rng = np.random.default_rng(seed)
    w = rng.integers(2, 7, items).astype(float)
    n = items * bins + bins
    A_eq = sp.lil_matrix((items, n))
    A_ineq = sp.lil_matrix((bins, n))
    for i in range(items):
        for j in range(bins):
            A_eq[i, i * bins + j] = 1.0
            A_ineq[j, i * bins + j] = w[i]
    for j in range(bins):
        A_ineq[j, items * bins + j] = -10.0
    c = np.concatenate([np.zeros(items * bins), np.ones(bins)])
    lp = jl.LP(
        c=c, A_eq=sp.csc_matrix(A_eq), b_eq=np.ones(items),
        A_ineq=sp.csc_matrix(A_ineq), b_ineq=np.zeros(bins),
        lower_bounds=np.zeros(n), upper_bounds=np.ones(n),
    )
    return jm.MIP(lp, np.ones(n, bool))


@pytest.mark.parametrize("make", [identical_machines, bin_packing])
def test_symmetry_generators_map_the_mip_to_itself(make):
    pytest.importorskip("igraph")
    mip = make()
    generators = jm.symmetry_generators(mip)
    assert generators
    x = optimum(mip)
    for perm in generators:
        moved = np.empty_like(x)
        moved[perm] = x  # column j's value goes to column perm[j]
        assert mip.is_feasible(moved)
        assert mip.c @ moved == pytest.approx(mip.c @ x)


def test_orbitopes_of_identical_machines():
    pytest.importorskip("igraph")
    mip = identical_machines(items=7, machines=3)
    (matrix,) = jm.orbitopes(jm.symmetry_generators(mip), mip.num_variables)
    assert matrix.shape == (7, 3)
    # Each row is one item's three machines.
    assert all(len(set(row // 3)) == 1 for row in matrix)


@pytest.mark.parametrize("make, packing", [(identical_machines, True), (bin_packing, False)])
def test_symmetry_breaking_keeps_the_optimum(make, packing):
    pytest.importorskip("igraph")
    mip = make()
    result = solve(mip, symmetry=True)
    assert_solved(result, mip, reference(mip))
    stats = result["symmetry"]
    assert stats["generators"] > 0 and stats["orbitopes"]
    if packing:
        assert stats["fixed"] > 0  # x[i, j] = 0 for j > i
    else:
        assert stats["fixed"] == 0 and stats["rows"] == stats["orbitopes"][0][1] - 1
    off = solve(mip, symmetry=False)
    assert off["symmetry"] is None
    assert off["objective"] == pytest.approx(result["objective"])


def test_symmetry_must_be_known():
    with pytest.raises(ValueError, match="symmetry"):
        solve(knapsack(), symmetry="sometimes")


# %% Restarts
def force_restarts(monkeypatch, root=0.0, tree=0.0):
    """Restart whenever one is allowed (any share fixed counts)."""
    monkeypatch.setattr(jm, "_RESTART_ROOT_SHARE", root)
    monkeypatch.setattr(jm, "_RESTART_TREE_SHARE", tree)


@pytest.mark.parametrize(
    "make", [generalised_assignment, facility_location, lambda: fixed_charge(0)[0]]
)
def test_restarts_keep_the_optimum(make, monkeypatch):
    force_restarts(monkeypatch)
    mip = make()
    result = solve(mip, restarts=2, cuts=False)
    assert_solved(result, mip, reference(mip))
    assert result["restarts"] == len(result["restart_log"]) == 2
    nodes = [entry["nodes"] for entry in result["restart_log"]]
    assert nodes == sorted(nodes) and result["nodes"] >= nodes[-1]  # cumulative


def test_restart_bound_is_a_floor(monkeypatch):
    # One restart in the tree: the bound after it never falls below the
    # bound the first round reached.
    force_restarts(monkeypatch, root=2.0)
    mip = generalised_assignment(1)
    result = solve(mip, restarts=1, cuts=False, root_strong_candidates=0)
    assert_solved(result, mip, reference(mip))
    (entry,) = result["restart_log"]
    assert mip.sign * (result["bound"] - entry["bound"]) >= -1e-9 * (1 + abs(entry["bound"]))


def test_restarts_off(monkeypatch):
    force_restarts(monkeypatch)
    mip = generalised_assignment(1)
    result = solve(mip, restarts=0, cuts=False)
    assert_solved(result, mip, reference(mip))
    assert result["restarts"] == 0 and result["restart_log"] == []
    with pytest.raises(ValueError, match="restarts"):
        solve(mip, restarts=-1)


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": 4},
        {"tree_cuts": True, "tree_cut_every": 1, "gomory_cuts": False},
        {"batch_size": 1, "tree_cuts": True, "gomory_cuts": False},
    ],
)
def test_restarts_with_other_features(options, monkeypatch):
    force_restarts(monkeypatch)
    mip = generalised_assignment(1)
    result = solve(mip, restarts=2, **options)
    assert_solved(result, mip, reference(mip))
    assert result["restarts"] >= 1


def test_restarts_keep_the_symmetry_rows(monkeypatch):
    pytest.importorskip("igraph")
    force_restarts(monkeypatch)
    mip = identical_machines()
    result = solve(mip, symmetry=True, restarts=2, cuts=False)
    assert_solved(result, mip, reference(mip))
    assert result["restarts"] >= 1 and result["symmetry"]["fixed"] > 0


# %% Background heuristics (parallel_heuristics)
@pytest.mark.parametrize(
    "make", [knapsack, general_integer, facility_location, generalised_assignment]
)
def test_parallel_heuristics_match_reference(make):
    mip = make()
    result = solve(
        mip, parallel_heuristics=True, rins_every=2
    )
    assert_solved(result, mip, reference(mip))
    stats = result["parallel_heuristics"]
    assert set(stats) == {
        "jobs", "seconds", "improvements", "jump_slices", "jump_restarts",
        "jump_restart_finds",
    }


def test_background_workers_run_jobs_in_order_and_stop():
    import threading

    workers = jm._HeuristicWorkers()
    seen = []

    def job(value, stop):
        seen.append(value)
        return value * 2

    def waits(stop):
        return stop.wait(30.0)  # until shutdown sets stop

    try:
        assert workers.idle("jump")
        workers.submit("jump", "a", job, 1, info="first")
        workers.submit("jump", "b", job, 2)
        assert not workers.idle("jump")
        done = []
        deadline = time.time() + 10
        while len(done) < 2 and time.time() < deadline:
            done += workers.finished()
        assert [(kind, out, info) for kind, out, _, info in done] == [
            ("a", 2, "first"),
            ("b", 4, None),
        ]
        assert seen == [1, 2] and workers.idle("jump")
        workers.submit("jump", "wait", waits)
    finally:
        started = time.time()
        workers.shutdown()
    assert time.time() - started < 5.0  # the waiting job saw stop
    assert workers.idle("jump")
    assert isinstance(workers.stop, threading.Event) and workers.stop.is_set()


def test_background_job_errors_reach_the_search():
    workers = jm._HeuristicWorkers()

    def fails(stop):
        raise RuntimeError("boom")

    try:
        workers.submit("jump", "x", fails)
        with pytest.raises(RuntimeError, match="boom"):
            deadline = time.time() + 10
            while time.time() < deadline:
                workers.finished()
    finally:
        workers.shutdown()


@NEEDS_NUMBA
def test_numba_jump_stops_when_asked():
    import threading

    stop = threading.Event()
    stop.set()
    mip = generalised_assignment(1)
    _, _, steps = jm.feasibility_jump(
        mip, np.zeros(mip.num_variables), improve=True, stop=stop
    )
    assert steps == 0


# -- user starting solutions (mip_start) -------------------------------------


def test_mip_start_is_the_first_incumbent():
    mip = knapsack()
    best = solve(mip)
    result = solve(mip, mip_start=best["x"])
    assert_solved(result, mip, reference(mip))
    assert result["mip_start_accepted"] is True
    assert result["history"][0]["source"] == "S"
    assert solve(mip)["mip_start_accepted"] is None


def test_infeasible_mip_start_is_ignored():
    mip = knapsack()
    result = solve(mip, mip_start=np.ones(mip.num_variables))  # over capacity
    assert result["mip_start_accepted"] is False
    assert_solved(result, mip, reference(mip))
    assert all(entry["source"] != "S" for entry in result["history"])


def test_mip_start_fills_in_continuous_variables():
    mip = facility_location()
    best = solve(mip)
    start = np.where(mip.integrality, best["x"], np.nan)
    result = solve(mip, mip_start=start, one_opt=False)
    assert result["mip_start_accepted"] is True
    assert result["history"][0]["source"] == "S"
    assert result["history"][0]["objective"] == pytest.approx(best["objective"], rel=1e-6)
    assert_solved(result, mip, reference(mip))


@pytest.mark.parametrize("make", [knapsack, facility_location])
def test_mip_start_with_presolve(make):
    highs = reducible_highs(make())
    original = jm.from_highs(highs)
    ref = reference(original)
    best = solve(reducible_highs(make()), presolve=True)
    result = solve(reducible_highs(make()), presolve=True, mip_start=best["x"])
    assert result["mip_start_accepted"] is True
    assert result["postsolve_feasible"]
    assert_solved(result, original, ref)
    # Nothing beats an optimal start by more than the gap: it comes back.
    np.testing.assert_allclose(result["x"], best["x"], atol=1e-6)
    assert result["gap"] <= 1e-4 * (1 + 1e-9)


def test_bad_mip_starts_raise():
    mip = facility_location()
    with pytest.raises(ValueError, match="entries"):
        solve(mip, mip_start=np.zeros(3))
    start = np.zeros(mip.num_variables)
    start[0] = np.nan  # an integer
    with pytest.raises(ValueError, match="NaN"):
        solve(mip, mip_start=start)


# -- implied-bound cuts --------------------------------------------------------


def implied_bounds_of(mip):
    probed = jm.probe(
        mip.A, mip.row_lower, mip.row_upper, mip.integrality,
        mip.lower.copy(), mip.upper.copy(), backend="numpy",
    )
    return jm.ImpliedBounds(probed.cache, probed.lower, probed.upper, mip.integrality), probed


@pytest.mark.parametrize("seed", [None, 0, 1, 5, 7])
def test_implied_bound_cuts_are_valid(seed):
    mip = facility_location() if seed is None else random_binary_rows(seed)
    implied, _ = implied_bounds_of(mip)
    assert len(implied)
    A = mip.A.toarray()
    for k in range(len(implied)):
        # The most any feasible point violates candidate k by: at most 0.
        c = np.zeros(mip.num_variables)
        c[implied.j[k]] -= implied.sense[k]
        c[implied.z[k]] -= implied.coef_z[k]
        worst = milp(
            c,
            constraints=LinearConstraint(A, mip.row_lower, mip.row_upper),
            integrality=mip.integrality.astype(int),
            bounds=Bounds(mip.lower, mip.upper),
        )
        assert worst.success
        assert -worst.fun - implied.rhs[k] <= 1e-7 * (1 + abs(implied.rhs[k]))


def test_implied_bound_cuts_link_flows_to_openings():
    mip = facility_location()
    implied, probed = implied_bounds_of(mip)
    # Half-open facilities with full flows: each y_fk <= u·x_f is violated.
    x = probed.upper.copy()
    x[mip.integrality] = 0.5
    x[~mip.integrality] = np.minimum(x[~mip.integrality], 3.0)
    cuts = implied.cuts(x)
    assert cuts
    for family, idx, coef, rhs in cuts:
        assert family == "impliedbound"
        assert coef @ x[idx] > rhs
        assert mip.integrality[idx[1]] and not mip.integrality[idx[0]]
    result = solve(mip)
    assert_solved(result, mip, reference(mip))
    assert result["cuts"]["impliedbound"] > 0
    assert solve(mip, implied_bound_cuts=False)["cuts"]["impliedbound"] == 0


def coverage_chains(seed=0, sites=5, depth=8, targets=6):
    """map06 in miniature: binaries ``z`` (open a site) within a budget, each
    with a chain of continuous ``x_{s,0} <= z_s``, ``x_{s,i} <= x_{s,i−1}``
    ``depth`` rows long, and targets ``v_t <= x_{a,end} + x_{c,end}``
    (covered by either of two sites) that the objective rewards. All in
    [0, 1]. Columns: ``z``, the chains, then ``v``."""
    rng = np.random.default_rng(seed)
    n = sites + sites * depth + targets
    chain = lambda s, i: sites + s * depth + i
    entries, b = [], []
    w = rng.integers(3, 9, sites).astype(float)
    entries.append({s: w[s] for s in range(sites)})
    b.append(np.floor(0.45 * w.sum()))
    for s in range(sites):
        entries.append({chain(s, 0): 1.0, s: -1.0})
        b.append(0.0)
        for i in range(1, depth):
            entries.append({chain(s, i): 1.0, chain(s, i - 1): -1.0})
            b.append(0.0)
    for t in range(targets):
        a, c = rng.choice(sites, 2, replace=False)
        end_a, end_c = chain(a, depth - 1), chain(c, depth - 1)
        entries.append({sites * (depth + 1) + t: 1.0, end_a: -1.0, end_c: -1.0})
        b.append(0.0)
    A = sp.lil_matrix((len(entries), n))
    for r, row in enumerate(entries):
        for j, v in row.items():
            A[r, j] = v
    cost = np.zeros(n)
    cost[sites * (depth + 1) :] = -rng.integers(1, 4, targets)
    A_eq, b_eq = no_rows(n)
    lp = jl.LP(
        c=cost,
        A_eq=A_eq,
        b_eq=b_eq,
        A_ineq=sp.csc_matrix(A),
        b_ineq=np.array(b),
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
    )
    integrality = np.zeros(n, bool)
    integrality[:sites] = True
    return jm.MIP(lp, integrality)


def highs_vertex(mip):
    """The LP relaxation's vertex from HiGHS's simplex (PDHG grinds on the
    degenerate chain rows of ``coverage_chains``)."""
    model = jm.to_highs(mip)
    model.integrality_ = [highspy.HighsVarType.kContinuous] * model.num_col_
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.passModel(model)
    highs.run()
    return np.array(highs.getSolution().col_value)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_probing_follows_chains_past_five_passes(backend):
    # z_0 = 0 empties the chain under it only after ``depth`` passes.
    depth = 8
    mip = coverage_chains(depth=depth)
    end = 5 + depth - 1  # the last column of site 0's chain
    args = (mip.A, mip.row_lower, mip.row_upper, mip.integrality, mip.lower, mip.upper)
    short = jm.probe(*args, backend=backend, max_passes=5)
    full = jm.probe(*args, backend=backend)
    for probed, reaches in ((short, False), (full, True)):
        lower, upper = mip.lower.copy(), mip.upper.copy()
        assert probed.cache.implied(0, 0.0, lower, upper)
        assert (upper[end] == 0.0) == reaches
    implied = jm.ImpliedBounds(full.cache, full.lower, full.upper, mip.integrality)
    assert (0, 1.0, 0.0) in implied.variable_bounds()[end]["ub"]  # x_end <= z_0
    # The device backend keeps every implication unless capped.
    capped = jm.probe(*args, backend=backend, max_implications=2)
    kept = max(len(side[0]) for sides in capped.cache.sides.values() for side in sides if side)
    assert kept <= 2 if backend == "jax" else kept > 2


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_cuts_over_implied_variable_bounds_are_valid(seed):
    mip = coverage_chains(seed)
    probed = jm.probe(
        mip.A, mip.row_lower, mip.row_upper, mip.integrality, mip.lower, mip.upper
    )
    implied = jm.ImpliedBounds(probed.cache, probed.lower, probed.upper, mip.integrality)
    separator = jm.CutSeparator(
        mip, probed.lower, probed.upper, max_aggregation=6, implied_bounds=implied
    )
    # Every chain column now has a variable bound on its site (the rows
    # give one only to the first of each chain).
    sites, depth = 5, 8
    chains = np.arange(sites, sites * (depth + 1))
    assert all(int(j) in separator.variable_bounds for j in chains)
    x = highs_vertex(mip)
    rng = np.random.default_rng(seed)
    points = [x, np.clip(x + rng.uniform(-0.3, 0.3, x.size), 0.0, 1.0)]
    for point in points:
        for _, idx, coef, rhs in separator(point):
            assert max_over_mip(mip, idx, coef) <= rhs + 1e-7 * (1 + abs(rhs))


def test_warm_start_from_an_lp_with_other_rows():
    # A background heuristic's warm start comes from the root LP of when its
    # job was queued; cut rounds may have changed the rows since.
    mip = facility_location()
    result = jl.solve(mip.lp)
    warm = (result["solution"], result["opt_state"])
    assert jm._fit_warm(warm, mip.lp) is warm
    for more in (mip.with_rows(np.ones((3, mip.num_variables)), np.full(3, 1e3)),):
        solution, opt_state = jm._fit_warm(warm, more.lp)
        np.testing.assert_array_equal(solution.primal, result["solution"].primal)
        assert solution.dual_ineq.shape[0] == more.lp.A_ineq.shape[0]
        assert not np.any(solution.dual_ineq)
        out = jl.solve(more.lp, initial_solution=solution, initial_opt_state=opt_state)
        assert out["solution"].primal.shape == result["solution"].primal.shape


@pytest.mark.parametrize("make", [facility_location, lambda: coverage_chains(1)])
def test_exact_node_lp_duals_bound_the_lp(make):
    mip = make()
    highs = jm._highs_rows(mip)
    lower, upper = mip.lower.copy(), mip.upper.copy()
    ints = np.flatnonzero(mip.integrality)
    for fixed in (None, 0.0, 1.0):
        if fixed is not None:
            lower[ints[0]] = upper[ints[0]] = fixed
        status, x, y_eq, y_ineq = jm._exact_node_lp(highs, mip, lower, upper, 10.0)
        assert status == "optimal"
        value = mip.c @ x
        # The duals' Lagrangian bound is the LP's optimum, and rigorous.
        terms = jm.lagrangian_terms(
            mip.c, mip.A, mip.b, mip.n_eq, np.concatenate([y_eq, y_ineq])
        )
        bound, rigorous = jm.box_bound(*terms, lower, upper)
        assert rigorous
        assert bound <= value + 1e-7 * (1 + abs(value))
        assert bound == pytest.approx(value, rel=1e-6, abs=1e-6)
        assert restricted_optimum(mip, lower, upper) >= bound - 1e-6


def test_exact_node_lp_proves_an_empty_box_infeasible():
    mip = coverage_chains(1)
    highs = jm._highs_rows(mip)
    lower, upper = mip.lower.copy(), mip.upper.copy()
    lower[:5] = 1.0  # every site open: over the budget
    status, *_ = jm._exact_node_lp(highs, mip, lower, upper, 10.0)
    assert status == "infeasible"


@pytest.mark.parametrize("make", [knapsack, facility_location])
def test_default_settings_solve_to_the_optimum(make):
    # jm.solve itself, not the PINNED helper: float32 LPs, no cuts or
    # probing, the jump as the only heuristic, and the exact lane.
    mip = make()
    result = jm.solve(mip, max_seconds=120)
    assert_solved(result, mip, reference(mip))
    assert result["exact_nodes"]["solved"] > 0


@pytest.mark.parametrize("share", [0.5, 0.0])
def test_band_overflow_takes_the_top_of_the_band(monkeypatch, share):
    import heapq

    search = search_for(
        generalised_assignment(), monkeypatch, exact_nodes=2, batch_size=4,
        exact_overflow_share=share,
    )
    nodes = [jm._Node(float(b), 1, (), None, None) for b in range(20)]
    for node in nodes:
        heapq.heappush(search.exact_reserved, (node.bound, next(search.counter), node))
    taken = search.overflow_band(int(search.exact_overflow_share * search.batch_size))
    if share == 0.0:
        assert taken == [] and len(search.exact_reserved) == 20
        return
    # The lane keeps 4 a thread (8, the lowest); at most 2 of the 12 above
    # go to the device a batch, the highest first taken, lowest first given.
    assert [node.bound for node in taken] == [18.0, 19.0]
    assert all(node.converge and node.exact_tried for node in taken)
    assert sorted(entry[0] for entry in search.exact_reserved) == [float(b) for b in range(18)]
    while len(search.exact_reserved) > 8:
        search.overflow_band(2)
    assert search.overflow_band(2) == []
    assert sorted(entry[0] for entry in search.exact_reserved) == [float(b) for b in range(8)]
    assert search.overflow_nodes == 12
    # Still a heap: the lane pops the lowest.
    assert heapq.heappop(search.exact_reserved)[0] == 0.0


def test_band_overflow_in_the_search():
    # One lane thread and a wide band: the band outgrows the lane, and its
    # top goes to the device, solved to convergence, with the optimum kept.
    mip = generalised_assignment(agents=10, jobs=40)
    ref = reference(mip)
    options = dict(
        max_seconds=300, batch_size=4, exact_nodes=1, exact_band=0.5,
        rins_seconds=0, crossover_seconds=0,
    )
    result = jm.solve(mip, exact_overflow_share=0.5, **options)
    assert result["bound"] <= ref + 1e-6
    if result["status"] == "optimal":
        assert_solved(result, mip, ref)
    stats = result["exact_nodes"]
    assert stats["overflow_nodes"] > 0
    assert 0 <= stats["overflow_converged"] <= stats["overflow_nodes"]
    off = jm.solve(mip, exact_overflow_share=0.0, **dict(options, max_seconds=30))
    assert off["exact_nodes"]["overflow_nodes"] == 0
    assert off["bound"] <= ref + 1e-6


@pytest.mark.parametrize("make", [facility_location, generalised_assignment])
def test_highs_submip_finds_the_optimum_below_the_cutoff(make):
    mip = make()
    ref = reference(mip) - mip.offset
    is_int = np.asarray(mip.integrality, bool)
    highs = jm._highs_submip(mip, is_int)
    x = jm._submip_job(highs, mip, mip.lower, mip.upper, np.inf, 30.0)
    assert mip.is_feasible(x) and float(mip.c @ x) == pytest.approx(ref, abs=1e-6)
    # Pruned at a cutoff below the optimum: nothing better to find.
    assert jm._submip_job(highs, mip, mip.lower, mip.upper, ref - 1e-3, 30.0) is None
    # Half the integers fixed at the optimum's values: the point stays in
    # the box, and the optimum is still in it.
    fixed = is_int & (np.arange(mip.num_variables) % 2 == 0)
    lower = np.where(fixed, np.round(x), mip.lower)
    upper = np.where(fixed, np.round(x), mip.upper)
    y = jm._submip_job(highs, mip, lower, upper, np.inf, 30.0)
    assert np.all(y >= lower - 1e-9) and np.all(y <= upper + 1e-9)
    assert float(mip.c @ y) == pytest.approx(ref, abs=1e-6)


def test_submip_lanes_run_beside_the_exact_lane():
    # HiGHS instances on several threads at once: two sub-MIP lanes (MIP
    # solves) and the exact lane (LPs) all finish with their answers.
    mip = generalised_assignment()
    is_int = np.asarray(mip.integrality, bool)
    lanes = [jm._SubMipLane() for _ in range(2)]
    exact = jm._ExactLane(2, 30.0)
    try:
        for lane in lanes:
            lane.submit("rins", mip, is_int, mip.lower, mip.upper, np.inf, 30.0)
        box = (mip.lower, mip.upper)
        for _ in range(2):
            exact.submit(None, box, box, mip, 0)
        done, deadline = [], time.time() + 60
        while (len(done) < 2 or exact.jobs) and time.time() < deadline:
            for lane in lanes:
                out = lane.finished()
                if out is not None:
                    done.append(out)
            exact.wait(0.05)
        finished = exact.finished()
        assert len(done) == 2 and all(x is not None for _, x, _ in done)
        assert all(float(mip.c @ x) == pytest.approx(reference(mip)) for _, x, _ in done)
        assert not exact.jobs
    finally:
        for lane in lanes:
            lane.shutdown()
        exact.shutdown()


def test_solution_pool_keeps_distinct_points_best_first(monkeypatch):
    mip = generalised_assignment()
    search = search_for(mip, monkeypatch, submip_solver="highs", crossover_seconds=5)
    ints = search.is_int
    rng = np.random.default_rng(0)
    for k in range(15):
        x = np.zeros(mip.num_variables)
        x[ints] = rng.integers(0, 2, ints.sum())
        search.remember_solution(float(k % 7), x)
        search.remember_solution(float(k % 7) + 0.5, x)  # same integers: ignored
    pool = search.solution_pool
    assert len(pool) == jm._SOLUTION_POOL_SIZE
    values = [value for value, _, _ in pool]
    assert values == sorted(values) and all(v == int(v) for v in values)
    assert len({key for _, key, _ in pool}) == len(pool)


def test_crossover_fixes_where_the_points_agree(monkeypatch):
    mip = generalised_assignment()
    search = search_for(
        mip, monkeypatch, submip_solver="highs", crossover_seconds=5,
        crossover_solutions=3, crossover_min_fixed=0.5,
    )
    ints = search.is_int
    rng = np.random.default_rng(1)
    first = np.zeros(mip.num_variables)
    first[ints] = rng.integers(0, 2, ints.sum())
    # Two others that each differ from the first on a few integers.
    cols = np.flatnonzero(ints)
    others = []
    for flip in (cols[:3], cols[3:7]):
        x = first.copy()
        x[flip] = 1.0 - x[flip]
        others.append(x)
    search.incumbent = (float(mip.c @ first), first)
    for value, x in [(0.0, first), (1.0, others[0]), (2.0, others[1])]:
        search.remember_solution(value, x)
    lower, upper = search.crossover_box(np.random.default_rng(0))
    agree = ints.copy()
    agree[cols[:7]] = False
    fixed = lower == upper
    assert np.array_equal(fixed & ints, agree)
    assert np.array_equal(lower[agree], first[agree])
    # The same neighbourhood at the same cutoff is not searched twice.
    assert search.crossover_box(np.random.default_rng(0)) is None


@pytest.mark.parametrize(
    "make", [generalised_assignment, lambda: coverage_chains(1)]
)
def test_highs_rins_and_crossover_in_the_search(make):
    mip = make()
    result = jm.solve(
        mip, max_seconds=120, batch_size=4, rins_seconds=5, crossover_seconds=5
    )
    assert_solved(result, mip, reference(mip))
    assert result["rins_calls"] > 0
    assert result["rins_finds"] <= result["rins_calls"]
    assert result["crossover_finds"] <= result["crossover_calls"]
    off = jm.solve(mip, max_seconds=120, batch_size=4, rins_seconds=0, crossover_seconds=0)
    assert off["rins_calls"] == off["crossover_calls"] == off["submip_lane_seconds"] == 0


@pytest.mark.parametrize(
    "make", [facility_location, generalised_assignment, lambda: coverage_chains(1)]
)
def test_exact_lane_solves_to_the_optimum(make):
    # Without cuts and probing they branch, so the lane gets nodes. (The
    # root's exact strong branching would solve facility_location's whole
    # tree at the root, leaving the lane nothing to solve.)
    mip = make()
    ref = reference(mip)
    options = dict(batch_size=4, cuts=False, probing=False, exact_root_candidates=0)
    for band in (0.005, 0.5, 0.0, "auto"):  # the band kept for the lane; 0: none
        result = solve(mip, exact_nodes=2, exact_band=band, **options)
        assert_solved(result, mip, ref)
        stats = result["exact_nodes"]
        assert stats["solved"] > 0 and stats["bound_gain"] >= 0.0
        assert stats["band_min"] <= stats["band"] <= stats["band_max"]
    assert solve(mip, **options)["exact_nodes"]["solved"] == 0


def test_adaptive_exact_band():
    # The auto band widens while the lane's queue is short, narrows while it
    # is long, holds in between, stays within its limits, and waits for a
    # cutoff.
    lane = types.SimpleNamespace(threads=2)  # target: 8 queued nodes
    search = types.SimpleNamespace(
        exact_band_auto=True,
        exact_lane=lane,
        exact_band=jm._EXACT_BAND_START,
        exact_band_range=[jm._EXACT_BAND_START] * 2,
        exact_reserved=[],
        cutoff=lambda: 1.0,
    )
    adapt = lambda: jm._BranchAndBound.adapt_exact_band(search)
    adapt()
    assert search.exact_band == pytest.approx(jm._EXACT_BAND_START * jm._EXACT_BAND_STEP)
    for _ in range(100):
        adapt()
    assert search.exact_band == jm._EXACT_BAND_MAX
    search.exact_reserved = [None] * 8
    adapt()
    assert search.exact_band == jm._EXACT_BAND_MAX  # within the target: holds
    search.exact_reserved = [None] * 17
    for _ in range(100):
        adapt()
    assert search.exact_band == jm._EXACT_BAND_MIN
    assert search.exact_band_range == [jm._EXACT_BAND_MIN, jm._EXACT_BAND_MAX]
    search.exact_reserved, search.cutoff = [], lambda: np.inf
    adapt()
    assert search.exact_band == jm._EXACT_BAND_MIN  # no incumbent: no change
    with pytest.raises(ValueError):
        jm.solve(facility_location(), batch_size=4, exact_band="wide")


@pytest.mark.parametrize("make", [facility_location, generalised_assignment])
def test_exact_strong_on_named_candidates(make):
    # Named candidates, no pivot limit: both sides of each, solved to
    # optimal (or empty), with bounds that hold; a past deadline splits none.
    mip = make()
    n = mip.num_variables
    highs = jm._highs_rows(mip)
    lower, upper = mip.lower.copy(), mip.upper.copy()
    status, x, _, _ = jm._exact_node_lp(highs, mip, lower, upper, 10.0)
    assert status == "optimal"
    ints = np.flatnonzero(mip.integrality)
    frac = ints[np.abs(x[ints] - np.round(x[ints])) > 1e-6]
    assert frac.size > 0
    named = np.concatenate([frac[::-1], ints[:1]])  # an integral one too
    request = (mip.integrality, 1e-6, np.zeros((2, n)), np.zeros((2, n)), n, None, 1)
    children = jm._exact_strong(
        highs, mip, lower, upper, x, request, candidates=named
    )
    assert [c[0] for c in children[::2]] == [v for v in named if v in set(frac)]
    for var, down, side_status, cx, y_eq, y_ineq in children:
        lo, up = lower.copy(), upper.copy()
        if down:
            up[var] = np.floor(x[var])
        else:
            lo[var] = np.ceil(x[var])
        optimum = lp_optimum(mip, lo, up)
        if side_status == "infeasible":
            assert optimum is None
            continue
        assert side_status == "optimal"
        assert mip.c @ cx == pytest.approx(optimum, rel=1e-6, abs=1e-6)
    late = jm._exact_strong(
        highs, mip, lower, upper, x, request, candidates=named, deadline=0.0
    )
    assert late == []


def test_finished_lane_nodes_count_in_the_bound_until_evaluated(monkeypatch):
    # The lane hands back several finished nodes at once; while the first
    # is evaluated (and logged), the global bound and the explored share
    # still count the others. They were in no list: the logged bound rose
    # past them (map06: -362.3, then -376.4 once they were queued).
    search = search_for(generalised_assignment(), monkeypatch, exact_nodes=1, batch_size=4)
    try:
        search.tree_started = True
        box = (search.mip.lower, search.mip.upper)
        first = jm._Node(-10.0, 1, None, None, None)
        second = jm._Node(-50.0, 2, None, None, None)
        extra = dict(pivots=0, warm=False, mip=None, basis=None, sides={})
        outcome = ("optimal", np.zeros(search.mip.num_variables), None, None)
        finished = [
            (node, box, search.cut_version, outcome, [], 0.0, 0.0, extra)
            for node in (first, second)
        ]
        monkeypatch.setattr(search.exact_lane, "finished", lambda: finished)
        monkeypatch.setattr(search, "exact_solution", lambda *args: object())
        monkeypatch.setattr(search, "exact_strong_children", lambda *args: None)
        seen = []

        def evaluate_exact(node, box, solution, strong=None):
            search.active_bound = node.bound  # as ``_evaluate`` sets it
            seen.append((node, search.global_bound(), search.explored()))

        monkeypatch.setattr(search, "evaluate_exact", evaluate_exact)
        search.collect_exact()
        assert [node for node, _, _ in seen] == [first, second]
        assert seen[0][1] == -50.0  # the second, not yet evaluated
        assert seen[0][2] == pytest.approx(1.0 - 0.25)  # its weight (depth 2) open
        assert search.exact_unevaluated == []
    finally:
        search.exact_lane.shutdown()


def test_root_exact_strong_branching_in_the_search():
    mip = generalised_assignment()
    ref = reference(mip)
    options = dict(batch_size=4, cuts=False, probing=False, exact_nodes=2)
    on = solve(mip, **options)
    assert_solved(on, mip, ref)
    stats = on["root_exact_strong"]
    assert stats["candidates"] > 0 and stats["lps"] == 2 * stats["candidates"]
    # The exact root LP's bound is the LP optimum, below the MIP's.
    assert stats["bound"] == pytest.approx(
        lp_optimum(mip, mip.lower, mip.upper) + mip.offset, rel=1e-6
    )
    off = solve(mip, exact_root_candidates=0, **options)
    assert_solved(off, mip, ref)
    assert off["root_exact_strong"]["candidates"] == 0
    # Without the lane there is nothing to run it on.
    assert solve(mip, batch_size=4, cuts=False, probing=False)[
        "root_exact_strong"
    ]["candidates"] == 0


def test_root_exact_strong_seeds_the_pseudocosts(monkeypatch):
    mip = generalised_assignment()
    real = jm._BranchAndBound.root_exact_strong
    seen = {}

    def spy(self, root, lower, upper, result):
        out = real(self, root, lower, upper, result)
        seen["exact"] = self.exact_pc_count.sum()
        seen["device"] = self.pc_count.sum()
        seen["out"] = out
        return out

    monkeypatch.setattr(jm._BranchAndBound, "root_exact_strong", spy)
    solve(mip, batch_size=4, cuts=False, probing=False, exact_nodes=2, max_nodes=5)
    assert seen["out"] is not None and seen["out"]["stop_reason"] == "exact"
    assert seen["exact"] > 0 and seen["device"] >= seen["exact"]


@pytest.mark.parametrize("make", [facility_location, generalised_assignment])
def test_exact_node_lp_from_the_parents_basis(make):
    # A child's LP from its parent's optimal basis has the same optimum as
    # one solved cold, in fewer pivots than from the slack basis.
    mip = make()
    highs = jm._highs_rows(mip)
    lower, upper = mip.lower.copy(), mip.upper.copy()
    status, x, _, _ = jm._exact_node_lp(highs, mip, lower, upper, 10.0)
    assert status == "optimal"
    parent = highs.getBasis()
    ints = np.flatnonzero(mip.integrality)
    frac = ints[np.abs(x[ints] - np.round(x[ints])) > 1e-6]
    assert frac.size > 0
    var = frac[0]
    for down in (True, False):
        lo, up = lower.copy(), upper.copy()
        if down:
            up[var] = np.floor(x[var])
        else:
            lo[var] = np.ceil(x[var])
        cold = jm._highs_rows(mip)
        cold_out = jm._exact_node_lp(cold, mip, lo, up, 10.0)
        cold_pivots = cold.getInfo().simplex_iteration_count
        warm = jm._highs_rows(mip)
        warm_out = jm._exact_node_lp(warm, mip, lo, up, 10.0, basis=parent)
        warm_pivots = warm.getInfo().simplex_iteration_count
        assert warm_out[0] == cold_out[0]
        if cold_out[0] == "optimal":
            assert mip.c @ warm_out[1] == pytest.approx(mip.c @ cold_out[1], abs=1e-6)
        assert warm_pivots <= cold_pivots


@pytest.mark.parametrize("tree_cuts", [True, False])
def test_exact_lane_children_start_from_their_parents_basis(tree_cuts):
    # The lane's nodes branch here, so their children reach the lane with
    # the basis kept for them, across tree cut refreshes too.
    mip = generalised_assignment()
    ref = reference(mip)
    options = dict(
        batch_size=4, cuts=False, probing=False, exact_nodes=2, exact_band=0.5,
        tree_cuts=tree_cuts,
    )
    on = solve(mip, **options)
    assert_solved(on, mip, ref)
    assert on["exact_nodes"]["pivots"] > 0
    assert on["exact_nodes"]["warm_bases"] > 0
    off = solve(mip, exact_basis_memory=0, **options)
    assert_solved(off, mip, ref)
    assert off["exact_nodes"]["warm_bases"] == 0


def test_basis_store_drops_bases_no_node_holds(monkeypatch):
    search = search_for(facility_location(), monkeypatch, exact_nodes=1, batch_size=4)
    try:
        highs = jm._highs_rows(search.mip)
        jm._exact_node_lp(highs, search.mip, search.mip.lower, search.mip.upper, 10.0)
        basis = highs.getBasis()
        key = search.store_basis(basis, search.mip)
        node = jm._Node(0.0, 1, None, None, None)
        node.exact_basis = key
        children = [jm._Node(0.0, 2, None, None, None) for _ in range(2)]
        for child in children:
            child.exact_basis = key
            search.hold_basis(key)
        search.release_bases(node)  # the parent lets go once it has branched
        assert search.basis_store[key][2] == 2
        assert search.take_basis(children[0].exact_basis) is basis
        assert key in search.basis_store
        assert search.take_basis(children[1].exact_basis) is basis
        assert key not in search.basis_store
        # A basis of another shape (a restart) is not handed out.
        other = search.store_basis(basis, facility_location(n_fac=5))
        assert search.take_basis(other) is None
        # No room: nothing kept.
        search.exact_basis_memory = 0
        assert search.store_basis(basis, search.mip) is None
    finally:
        if search.exact_lane is not None:
            search.exact_lane.shutdown()


def lp_optimum(mip, lower, upper):
    """The LP relaxation's optimum over a box (``None`` when infeasible)."""
    model = jm.to_highs(mip)
    model.integrality_ = [highspy.HighsVarType.kContinuous] * model.num_col_
    model.col_lower_, model.col_upper_ = np.asarray(lower, float), np.asarray(upper, float)
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.passModel(model)
    highs.run()
    if highs.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return None
    return highs.getInfo().objective_function_value - mip.offset


@pytest.mark.parametrize(
    "make", [facility_location, generalised_assignment, lambda: coverage_chains(1)]
)
@pytest.mark.parametrize("iterations", [100_000, 2])
def test_exact_strong_children_have_valid_bounds(make, iterations):
    # Converged sides and sides stopped after 2 pivots: their duals'
    # Lagrangian bound never passes the side's LP optimum.
    mip = make()
    n = mip.num_variables
    highs = jm._highs_rows(mip)
    lower, upper = mip.lower.copy(), mip.upper.copy()
    status, x, _, _ = jm._exact_node_lp(highs, mip, lower, upper, 10.0)
    assert status == "optimal"
    request = (mip.integrality, 1e-6, np.zeros((2, n)), np.zeros((2, n)), 4, iterations, 4)
    children = jm._exact_strong(highs, mip, lower, upper, x, request)
    assert children  # the root vertices are fractional
    for var, down, side_status, cx, y_eq, y_ineq in children:
        lo, up = lower.copy(), upper.copy()
        if down:
            up[var] = np.floor(x[var])
        else:
            lo[var] = np.ceil(x[var])
        optimum = lp_optimum(mip, lo, up)
        if side_status == "infeasible":
            assert optimum is None
            continue
        assert side_status in ("optimal", "partial")
        if iterations > 1000:
            assert side_status == "optimal"
        constant, r = jm.lagrangian_terms(
            mip.c, mip.A, mip.b, mip.n_eq, np.concatenate([y_eq, y_ineq])
        )
        # Round-off in c + Aᵀy at infinite bounds, zeroed as the search does
        # (``vertex_terms``): facility_location's down side of x0 has one.
        roundoff = (((r > 0) & np.isinf(lo)) | ((r < 0) & np.isinf(up))) & (
            np.abs(r) <= 1e-9 * (1 + np.abs(mip.c).max())
        )
        bound, rigorous = jm.box_bound(constant, np.where(roundoff, 0.0, r), lo, up)
        assert rigorous
        assert bound <= optimum + 1e-6 * (1 + abs(optimum))
        if side_status == "optimal":
            assert bound == pytest.approx(optimum, rel=1e-6, abs=1e-6)
    # The node's bounds and basis are back: it re-solves to the same vertex.
    again = jm._exact_node_lp(highs, mip, lower, upper, 10.0)
    assert mip.c @ again[1] == pytest.approx(mip.c @ x, rel=1e-9, abs=1e-9)


def test_exact_strong_branching_in_the_search():
    totals = dict(strong_lps=0, children_reused=0)
    # The lane's own strong branching (the root's, ``exact_root_candidates``,
    # also reuses children).
    options = dict(
        batch_size=4, cuts=False, probing=False, exact_nodes=2,
        exact_root_candidates=0,
    )
    for make in (facility_location, generalised_assignment, lambda: coverage_chains(1)):
        mip = make()
        ref = reference(mip)
        result = solve(mip, **options)
        assert_solved(result, mip, ref)
        for key in totals:
            totals[key] += result["exact_nodes"][key]
        off = solve(mip, exact_strong_candidates=0, **options)
        assert_solved(off, mip, ref)
        assert off["exact_nodes"]["strong_lps"] == 0
        assert off["exact_nodes"]["children_reused"] == 0
    assert totals["strong_lps"] > 0 and totals["children_reused"] > 0


def test_solve_with_implied_variable_bounds_and_long_aggregation():
    mip = coverage_chains(1)
    ref = reference(mip)
    for options in (
        dict(max_aggregation=6),
        dict(implied_variable_bounds=False, max_aggregation=0),
    ):
        assert_solved(solve(mip, **options), mip, ref)


# -- 1-opt ---------------------------------------------------------------------


def test_one_opt_improves_within_the_rows():
    mip = knapsack()
    x = np.zeros(mip.num_variables)
    better = jm.one_opt(mip, x)
    assert better is not None
    assert mip.is_feasible(better)
    assert mip.c @ better < mip.c @ x
    # At its result no single variable can move further.
    assert jm.one_opt(mip, better) is None


def test_one_opt_leaves_equality_rows_alone():
    # min −x0 − x1 with x0 + x1 = 1 (equality) and x2 free to improve.
    lp = jl.LP(
        c=np.array([-1.0, -1.0, -1.0]),
        A_eq=sp.csc_matrix([[1.0, 1.0, 0.0]]),
        b_eq=np.array([1.0]),
        A_ineq=sp.csc_matrix([[0.0, 0.0, 1.0]]),
        b_ineq=np.array([3.0]),
        lower_bounds=np.zeros(3),
        upper_bounds=np.full(3, 5.0),
    )
    mip = jm.MIP(lp, np.ones(3, bool))
    better = jm.one_opt(mip, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(better, [1.0, 0.0, 3.0])


@pytest.mark.parametrize("seed", range(5))
def test_one_opt_points_stay_feasible(seed):
    mip = general_integer(seed)
    found = solve(mip, one_opt=False, max_nodes=1)
    if found["x"] is None:
        pytest.skip("no point to start from")
    better = jm.one_opt(mip, found["x"])
    if better is not None:
        assert mip.is_feasible(better)
        assert mip.c @ better < mip.c @ found["x"]


# -- objective propagation -------------------------------------------------------


def test_cutoff_propagator_follows_the_cutoff():
    # min −x0 − x1 − x2 over x in {0..5}^3 with x0 + x1 + x2 <= 10.
    A = sp.csr_matrix([[1.0, 1.0, 1.0]])
    c = -np.ones(3)
    cutoff = [np.inf]
    prop = jm.CutoffPropagator(
        A, np.array([-np.inf]), np.array([10.0]), c, np.ones(3, bool), lambda: cutoff[0]
    )
    lower, upper = np.zeros(3), np.full(3, 5.0)
    lo, up, infeasible = prop(lower, upper)
    assert not infeasible and np.array_equal(lo, lower)
    # Better than −13 needs Σx >= 13, over the row's 10: no room.
    cutoff[0] = -13.0
    assert prop(lower, upper)[2]
    # Better than −9 (Σx >= 9) with x1, x2 <= 3 forces x0 >= 3.
    cutoff[0] = -9.0
    lo, up, infeasible = prop(lower, np.array([5.0, 3.0, 3.0]))
    assert not infeasible and lo[0] == 3.0
    # The plain rows alone do not.
    plain = jm.BoundPropagator(A, np.array([-np.inf]), np.array([10.0]), np.ones(3, bool))
    assert plain(lower, np.array([5.0, 3.0, 3.0]))[0][0] == 0.0


@pytest.mark.parametrize("make", [knapsack, facility_location, general_integer])
def test_objective_propagation_keeps_the_optimum(make):
    mip = make()
    result = solve(mip, objective_propagation=True)
    assert_solved(result, mip, reference(mip))


def test_objective_propagation_with_the_cut_pool():
    mip = facility_location()
    result = solve(mip, objective_propagation=True, cut_pool_propagation=True)
    assert_solved(result, mip, reference(mip))


# -- gap-aware plunging ------------------------------------------------------------


def test_plunge_gap_stops_plunges_and_keeps_the_optimum():
    mip = generalised_assignment(1)
    ref = reference(mip)
    # Without cuts the tree is deep enough for plunges (~100 nodes).
    deep = dict(plunge_depth=1000, cuts=False)
    stopped = solve(mip, plunge_gap=0.0, **deep)
    assert_solved(stopped, mip, ref)
    assert stopped["plunges_stopped"] > 0
    free = solve(mip, plunge_gap=1.0, **deep)
    assert_solved(free, mip, ref)
    assert free["plunges_stopped"] == 0


def test_plunge_reaches_measures_from_the_best_open_bound():
    node = lambda bound: jm._Node(bound, 1, None, None, None)  # noqa: E731
    # The method on a stand-in search: best open bound 0, cutoff 10.
    search = types.SimpleNamespace(
        plunge_gap=0.25, cutoff=lambda: 10.0, open=[((0.0,), 0, node(0.0))]
    )
    reaches = lambda child, bound: jm._BranchAndBound.plunge_reaches(  # noqa: E731
        search, node(child), bound
    )
    assert reaches(2.4, bound=5.0)
    assert not reaches(2.6, bound=5.0)
    # Measured from the node being split when it is the best: −10 + 0.25·20.
    assert reaches(-5.1, bound=-10.0) and not reaches(-4.9, bound=-10.0)
    # No incumbent (an infinite cutoff): no limit.
    search.cutoff = lambda: np.inf
    assert reaches(1e9, bound=5.0)
    search.cutoff, search.plunge_gap = (lambda: 10.0), 1.0
    assert reaches(9.9, bound=5.0)


def test_rejects_bad_plunge_gap():
    with pytest.raises(ValueError, match="plunge_gap"):
        solve(knapsack(), plunge_gap=1.5)


@pytest.mark.parametrize(
    "make", [generalised_assignment, facility_location, lambda: fixed_charge(0)[0]]
)
def test_restarts_with_objective_propagation_keep_the_optimum(make, monkeypatch):
    # After a restart the root box holds only solutions better than the
    # incumbent; when the objective row shows it empty, the incumbent is
    # optimal (it was reported "infeasible").
    force_restarts(monkeypatch)
    mip = make()
    result = solve(mip, restarts=2, cuts=False, objective_propagation=True)
    assert_solved(result, mip, reference(mip))


# -- branching on implied integers (branch_implied) ---------------------------------


def penalty_cycle(k=7):
    """An odd cycle of binaries y (neighbours not both 1) with a penalty
    p_i in [0, 1] costing 1,000 when y_i is 0 (y_i + p_i >= 1), and y
    costing 1. Every choice of y has its best p integral, so the p are
    implied integers that carry the objective, as on ns1830653; the LP
    puts y = p = 1/2 everywhere. The p come first, so most-fractional
    branching takes one on the tie."""
    n = 2 * k
    A = np.zeros((2 * k, n))
    for i in range(k):
        A[i, [k + i, k + (i + 1) % k]] = 1.0  # y_i + y_{i+1} <= 1
        A[k + i, [i, k + i]] = -1.0  # −p_i − y_i <= −1
    lp = jl.LP(
        c=np.concatenate([np.full(k, 1000.0), np.ones(k)]),
        A_eq=sp.csc_matrix((0, n)), b_eq=np.zeros(0),
        A_ineq=sp.csc_matrix(A), b_ineq=np.concatenate([np.ones(k), -np.ones(k)]),
        lower_bounds=np.zeros(n), upper_bounds=np.ones(n),
    )
    ints = np.arange(n) >= k
    return jm.MIP(lp, ints, implied=~ints)


# A tree to branch in: no root heuristics or root sweep to close it first.
TREE = dict(no_heuristics(), root_strong_candidates=0)


def test_branch_implied_branches_on_them_and_keeps_the_optimum(monkeypatch):
    # Branch on an implied integer whenever one is a candidate (the LP's
    # values tie at 1/2 up to rounding), to check the children, their LP
    # bounds and the closing of nodes end to end.
    real = jm._BranchAndBound.choose_variable

    def choose_variable(self, x, fractional):
        implied = fractional & self.mip.implied
        return real(self, x, implied if implied.any() else fractional)

    monkeypatch.setattr(jm._BranchAndBound, "choose_variable", choose_variable)
    mip = penalty_cycle()
    ref = reference(mip)
    assert ref == pytest.approx(4 * 1000 + 3)  # (k + 1) / 2 penalties, (k − 1) / 2 ones
    settings = dict(TREE, branching="most_fractional")
    plain = solve(mip, **settings)
    assert_solved(plain, mip, ref)
    assert plain["implied_branchings"] == 0
    implied = solve(mip, branch_implied=True, **settings)
    assert_solved(implied, mip, ref)
    assert implied["implied_branchings"] > 0


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"branching": "pseudocost"},
        {"strong_backend": "numpy"},
        {"strong_backend": "jax", "branching": "reliability"},
    ],
)
def test_branch_implied_candidates_reach_every_branching_rule(settings, monkeypatch):
    seen = []
    real = jm._BranchAndBound.choose_branch

    def choose_branch(self, node, x, fractional, *args, **kwargs):
        seen.append(bool((fractional & self.mip.implied).any()))
        return real(self, node, x, fractional, *args, **kwargs)

    monkeypatch.setattr(jm._BranchAndBound, "choose_branch", choose_branch)
    mip = penalty_cycle()
    result = solve(mip, branch_implied=True, **TREE, **settings)
    assert_solved(result, mip, reference(mip))
    assert any(seen)


def test_lp_bounds_take_branched_implied_bounds_rounded_outward():
    is_int = np.array([True, False, False])
    search = types.SimpleNamespace(
        is_int=is_int,
        mip=types.SimpleNamespace(lower=np.zeros(3), upper=np.full(3, 5.0)),
        branch_implied_cols=np.array([1]),  # column 2 is plain continuous
    )
    lower, upper = np.array([1.0, 2.3, 0.7]), np.array([4.0, 2.9999999, 4.2])
    lo = jm._BranchAndBound.lp_lower(search, lower)
    up = jm._BranchAndBound.lp_upper(search, upper)
    np.testing.assert_array_equal(lo, [1.0, 2.0, 0.0])
    np.testing.assert_array_equal(up, [4.0, 3.0, 5.0])
    search.branch_implied_cols = None
    np.testing.assert_array_equal(jm._BranchAndBound.lp_lower(search, lower), [1.0, 0.0, 0.0])


# -- numba bound propagation ----------------------------------------------------------


@pytest.mark.skipif(jm._numba_propagate() is None, reason="needs numba")
@pytest.mark.parametrize(
    "make",
    [knapsack, general_integer, facility_location, generalised_assignment,
     lambda: random_binary_rows(4), lambda: penalty_cycle(9)],
)
def test_numba_propagation_matches_numpy_bit_for_bit(make):
    mip = make()
    rng = np.random.default_rng(0)
    args = (mip.A, mip.row_lower, mip.row_upper, mip.integrality)
    fast = jm.BoundPropagator(*args, backend="numba")
    slow = jm.BoundPropagator(*args, backend="numpy")
    outcomes = set()
    for trial in range(200):
        lower, upper = mip.lower.copy(), mip.upper.copy()
        # Fix or narrow a random share of the variables.
        for j in np.flatnonzero(rng.random(mip.num_variables) < rng.uniform(0, 0.5)):
            lo = lower[j] if np.isfinite(lower[j]) else -5.0
            up = upper[j] if np.isfinite(upper[j]) else lo + 10.0
            v = rng.uniform(lo, up)
            if mip.integrality[j]:
                v = np.round(v)
            if rng.random() < 0.5:
                lower[j] = upper[j] = v
            elif rng.random() < 0.5:
                lower[j] = v
            else:
                upper[j] = v
        passes = int(rng.integers(1, 21))
        a = fast(lower, upper, max_passes=passes)
        b = slow(lower, upper, max_passes=passes)
        assert a[2] == b[2]
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        outcomes.add(a[2])
    assert outcomes == {True, False}  # both kinds of box were tried


@pytest.mark.skipif(jm._numba_propagate() is None, reason="needs numba")
def test_cutoff_propagator_uses_the_numba_kernel_with_the_cutoff():
    mip = knapsack()
    cutoff = [np.inf]
    args = (mip.A, mip.row_lower, mip.row_upper, mip.c, mip.integrality, lambda: cutoff[0])
    fast = jm.CutoffPropagator(*args, backend="numba")
    slow = jm.CutoffPropagator(*args, backend="numpy")
    for value in (np.inf, -60.0, -100.0, -1e9):
        cutoff[0] = value
        a, b = fast(mip.lower, mip.upper), slow(mip.lower, mip.upper)
        assert a[2] == b[2]
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
    assert a[2]  # no point is that good


def test_propagation_backend_is_checked():
    mip = knapsack()
    with pytest.raises(ValueError, match="backend"):
        jm.BoundPropagator(mip.A, mip.row_lower, mip.row_upper, mip.integrality, backend="gpu")




def disjoint_paths(size=6, nets=6, shift=2):
    """Vertex-disjoint paths on a ``size`` x ``size`` torus grid: net ``k``
    runs from node ``(k, 0)`` to node ``((k + shift) % size, size // 2)``
    over binary arc variables, one flow block per net. Each node's
    capacity row (at most one net enters it, none enters a source) has an
    entry for every net and in-arc, so the nets are blocks joined only by
    those rows (``linking_blocks``)."""
    node = lambda r, c: (r % size) * size + (c % size)  # noqa: E731
    arcs = []
    for r in range(size):
        for c in range(size):
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                arcs.append((node(r, c), node(r + dr, c + dc)))
    n_nodes, n_arcs = size * size, len(arcs)
    n = nets * n_arcs
    sources = [node(k, 0) for k in range(nets)]
    sinks = [node(k + shift, size // 2) for k in range(nets)]
    eq_rows, eq_cols, eq_vals, b_eq = [], [], [], []
    for k in range(nets):
        for v in range(n_nodes):
            row = len(b_eq)
            for a, (tail, head) in enumerate(arcs):
                if tail == v:
                    eq_rows.append(row), eq_cols.append(k * n_arcs + a), eq_vals.append(1.0)
                if head == v:
                    eq_rows.append(row), eq_cols.append(k * n_arcs + a), eq_vals.append(-1.0)
            b_eq.append(1.0 if v == sources[k] else -1.0 if v == sinks[k] else 0.0)
    in_rows, in_cols = [], []
    for k in range(nets):
        for a, (_, head) in enumerate(arcs):
            in_rows.append(head), in_cols.append(k * n_arcs + a)
    capacity = np.ones(n_nodes)
    capacity[sources] = 0.0
    lp = jl.LP(
        c=np.ones(n),
        A_eq=sp.csc_matrix((eq_vals, (eq_rows, eq_cols)), shape=(len(b_eq), n)),
        b_eq=np.array(b_eq),
        A_ineq=sp.csc_matrix(
            (np.ones(len(in_rows)), (in_rows, in_cols)), shape=(n_nodes, n)
        ),
        b_ineq=capacity,
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
    )
    return jm.MIP(lp, np.ones(n, bool))


def test_linking_blocks_finds_the_nets():
    mip = disjoint_paths()
    blocks = jm.linking_blocks(mip)
    assert blocks is not None
    assert len(blocks.blocks) == 6
    assert blocks.linking.size == 36  # one capacity row a node
    assert np.all(blocks.linking >= mip.n_eq)
    columns = np.sort(np.concatenate(blocks.blocks))
    assert np.array_equal(columns, np.arange(mip.num_variables))
    # Every other row lies inside one block.
    for cols, rows in zip(blocks.blocks, blocks.block_rows):
        inside = np.zeros(mip.num_variables, bool)
        inside[cols] = True
        assert np.all(inside[mip.A[rows].indices])
    # No structure: nothing found, so the heuristic costs nothing.
    for make in (knapsack, general_integer, facility_location):
        assert jm.linking_blocks(make()) is None


def test_negotiate_blocks_finds_a_feasible_point():
    mip = disjoint_paths()
    lines = []
    x, feasible, mark = jm.negotiate_blocks(
        mip, deadline=time.time() + 60, log=lines.append
    )
    assert feasible and mark == "K"
    assert mip.is_feasible(x)
    assert lines and lines[0].startswith("sweep 1:")
    # The warm LP relaxations first: the torus's flow blocks have integral
    # vertices, so no block needs its MIP.
    holder = jm._HighsHolder()
    x, feasible, _ = jm.negotiate_blocks(
        mip, lp_first=True, holder=holder, deadline=time.time() + 60
    )
    assert feasible and mip.is_feasible(x)
    assert holder.stats["mip_solves"] == 0


def test_negotiate_blocks_window_settles_a_stall():
    # With no prices the blocks never move off their shortest paths, so
    # the overuse stalls; a window sub-MIP around the conflicts (every
    # net's variables there freed together) settles it.
    mip = disjoint_paths()
    lines = []
    x, feasible, _ = jm.negotiate_blocks(
        mip,
        present_start=0.0,
        history_weight=0.0,
        stall_sweeps=1,
        joint_seconds=10.0,
        max_window_share=1.0,  # a small model: any window is a large share
        deadline=time.time() + 60,
        log=lines.append,
    )
    assert feasible and mip.is_feasible(x)
    assert any(line.startswith("window of") for line in lines)
    adjacent = jm._linking_adjacency(mip, jm.linking_blocks(mip))
    # On the torus every node has its 4 grid neighbours (and the arcs of a
    # net's flow rows reach no further).
    assert np.all(adjacent.sum(axis=1) >= 4)


def test_negotiate_blocks_stops_when_cancelled():
    holder = jm._HighsHolder()
    holder.cancel()
    x, feasible, _ = jm.negotiate_blocks(disjoint_paths(), holder=holder)
    assert x is None and not feasible


def test_block_decomposition_before_the_root(capsys):
    # Block decomposition runs before the root LP, in the foreground (no
    # background workers), and its point is the first incumbent; the
    # search then goes on from it. It is off by default.
    mip = disjoint_paths()
    options = dict(feasibility_pump=False, bulk_rounding=False, cuts=False, max_nodes=1)
    result = solve(mip, block_decomposition=True, **options)
    assert result["history"][0]["source"] == "K"
    assert mip.is_feasible(result["x"])
    stats = result["block_decomposition"]
    assert stats["blocks"] == 6 and stats["linking_rows"] == 36
    assert stats["sweeps"] >= 1 and stats["overused_rows"] == 0
    off = solve(mip, **options)
    assert off["block_decomposition"] is None
    assert "K" not in [entry["source"] for entry in off["history"]]
    # One line before the progress table says it ran.
    with_log = solve(mip, block_decomposition=True, verbose=True, **options)
    assert with_log["history"][0]["source"] == "K"
    out = capsys.readouterr().out
    assert "Block decomposition: 6 blocks joined by 36 linking rows" in out
    assert out.index("Block decomposition:") < out.index("Src  Proc. InQueue")
    # Nothing to decompose: it does not run.
    assert solve(knapsack(), block_decomposition=True, **options)["block_decomposition"] is None
    with pytest.raises(ValueError):
        solve(mip, block_decomposition="auto", **options)


def test_trivial_points():
    # A packing model (the knapsack) takes all zeros; a covering model
    # (each row needs one of its columns) takes all ones, its lock point.
    mip = knapsack()
    points = jm.trivial_points(mip)
    assert any(mip.is_feasible(p) and not p.any() for p in points)
    n = 6
    cover = np.array([[1, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]], float)
    A_eq, b_eq = no_rows(n)
    lp = jl.LP(
        c=np.arange(1.0, n + 1),
        A_eq=A_eq,
        b_eq=b_eq,
        A_ineq=sp.csc_matrix(-cover),
        b_ineq=-np.ones(3),
        lower_bounds=np.zeros(n),
        upper_bounds=np.ones(n),
    )
    covering = jm.MIP(lp, np.ones(n, bool))
    points = jm.trivial_points(covering)
    assert not covering.is_feasible(points[0])  # all zeros
    assert any(covering.is_feasible(p) and p.all() for p in points)


def test_trivial_heuristic_in_the_search():
    mip = knapsack()
    options = dict(feasibility_pump=False, bulk_rounding=False, max_nodes=1)
    result = solve(mip, trivial_heuristic=True, **options)
    assert result["history"][0]["source"] == "Z"
    off = solve(mip, **options)
    assert "Z" not in [entry["source"] for entry in off["history"]]


# %% Ctrl-C
def _sigint_at(monkeypatch, method, presses=1, call=1):
    """Send ``presses`` SIGINTs to this process at the ``call``-th call of
    the search's ``method``, as Ctrl-C would."""
    import os
    import signal

    original = getattr(jm._BranchAndBound, method)
    calls = [0]

    def patched(self, *args, **kwargs):
        calls[0] += 1
        if calls[0] == call:
            for _ in range(presses):
                os.kill(os.getpid(), signal.SIGINT)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(jm._BranchAndBound, method, patched)
    return calls


def test_ctrl_c_in_the_tree_returns_the_best_point(monkeypatch):
    # The first Ctrl-C winds the search down along its time-limit path:
    # status "interrupted", the incumbent kept, the bound still valid, and
    # the caller's SIGINT handler back afterwards.
    import signal

    mip = general_integer()
    ref = reference(mip)
    before = signal.getsignal(signal.SIGINT)
    calls = _sigint_at(monkeypatch, "solve_node_lp", call=2)
    result = solve(mip, cuts=False, probing=False, branching="pseudocost")
    assert calls[0] >= 2
    assert result["status"] == "interrupted" and not result["converged"]
    assert result["bound"] <= ref + 1e-6 * (1 + abs(ref))
    if result["x"] is not None:
        assert mip.is_feasible(result["x"])
        assert result["objective"] >= ref - 1e-6 * (1 + abs(ref))
    assert signal.getsignal(signal.SIGINT) is before


def test_ctrl_c_before_the_root_lp(monkeypatch):
    # Ctrl-C before the root LP (here in the trivial heuristic, which runs
    # first) used to escape as a traceback; now the search returns.
    import signal

    mip = knapsack()
    before = signal.getsignal(signal.SIGINT)
    _sigint_at(monkeypatch, "try_trivial")
    result = solve(mip)
    assert result["status"] == "interrupted"
    assert result["bound"] <= reference(mip) + 1e-6
    assert signal.getsignal(signal.SIGINT) is before


def test_ctrl_c_twice_aborts(monkeypatch):
    # A second press raises KeyboardInterrupt, as without the handler.
    import signal

    before = signal.getsignal(signal.SIGINT)
    _sigint_at(monkeypatch, "try_trivial", presses=2)
    with pytest.raises(KeyboardInterrupt):
        solve(knapsack())
    assert signal.getsignal(signal.SIGINT) is before
