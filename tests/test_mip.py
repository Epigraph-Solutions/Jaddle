"""Tests for the branch-and-bound MIP solver, jaddle.jaddle_mip."""

import highspy
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

import jaddle.jaddle_linear as jl
import jaddle.jaddle_mip as jm
from jaddle import highs_helpers as hh


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


def assert_solved(result, mip, ref, rel=1e-4):
    assert result["status"] == "optimal" and result["converged"]
    assert abs(result["objective"] - ref) <= rel * (1 + abs(ref))
    assert mip.is_feasible(result["x"])
    # The bound is on the right side of the optimum (up to LP accuracy).
    slack = 1e-6 * (1 + abs(ref))
    if mip.maximise:
        assert result["bound"] >= ref - slack
    else:
        assert result["bound"] <= ref + slack
    assert result["bound_rigorous"]


def write_mps(mip, path):
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.passModel(jm.to_highs(mip))
    highs.writeModel(str(path))


@pytest.mark.parametrize("make", [knapsack, general_integer, facility_location])
def test_solve_matches_reference(make):
    mip = make()
    assert_solved(jm.solve(mip), mip, reference(mip))


@pytest.mark.parametrize(
    "branching, node_selection",
    [("most_fractional", "best_first"), ("pseudocost", "depth_first")],
)
def test_search_options(branching, node_selection):
    mip = general_integer()
    result = jm.solve(mip, branching=branching, node_selection=node_selection)
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
        assert_solved(jm.solve(model), loaded, ref)


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

    result = jm.solve(reducible_highs(make()), presolve=True)
    assert result["presolve"] == "kReduced"
    assert result["postsolve_feasible"]
    assert result["x"].shape == (original.num_variables,)
    assert_solved(result, original, ref)


def test_presolve_maximisation(tmp_path):
    base = knapsack()
    mip = jm.MIP(base.lp, base.integrality, offset=1.5, maximise=True)
    path = tmp_path / "knapsack.mps"
    write_mps(mip, path)
    result = jm.solve(path, presolve=True)
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
    assert jm.solve(mip, presolve=True)["status"] == "infeasible"
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
    result = jm.solve(jm.MIP(lp, np.ones(n, bool)))
    assert result["status"] == "infeasible"
    assert result["x"] is None and result["objective"] is None
    assert result["nodes"] >= 1


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
    result = jm.solve(jm.MIP(lp, [True, False]), max_seconds=60)
    assert result["status"] == "unbounded_or_infeasible"


def test_limits_report_valid_bounds():
    mip = general_integer(n=12, m_ineq=8)
    ref = reference(mip)
    result = jm.solve(mip, max_nodes=5)
    assert result["status"] == "node_limit" and not result["converged"]
    assert result["nodes"] <= 5
    assert result["bound"] <= ref + 1e-6 * (1 + abs(ref))
    if result["x"] is not None:
        assert mip.is_feasible(result["x"])
        assert result["objective"] >= ref - 1e-6 * (1 + abs(ref))

    # A limit spent before the root LP finishes (0.5 s was enough to solve
    # this MIP once compiled).
    result = jm.solve(mip, max_seconds=1e-3)
    assert result["status"] == "time_limit"
    assert result["bound"] <= ref + 1e-6 * (1 + abs(ref))


def test_verbose_log(capsys):
    jm.solve(knapsack(), verbose=True)
    out = capsys.readouterr().out
    assert "Root LP" in out and "Status: optimal" in out


def test_rejects_bad_options():
    with pytest.raises(ValueError, match="branching"):
        jm.solve(knapsack(), branching="strong")
    with pytest.raises(ValueError, match="lp_options"):
        jm.solve(knapsack(), lp_options={"max_epochs": 3})


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


def test_cuts_strengthen_the_root():
    mip = facility_location()
    ref = reference(mip)
    with_cuts = jm.solve(mip)
    without = jm.solve(mip, cuts=False)
    for result in (with_cuts, without):
        assert_solved(result, mip, ref)
    assert with_cuts["cuts"]["mir"] > 0
    assert without["cuts"] == {"gomory": 0, "cover": 0, "mir": 0}
    assert without["root_bound_before_cuts"] is None
    # The cuts close part of the root gap and the tree is no larger.
    root = with_cuts["root_bound_before_cuts"]
    history_bound = with_cuts["history"][0]["bound"]
    assert history_bound > root + 1e-3 * abs(ref - root)
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
    off = jm.solve(
        mip,
        max_nodes=1,
        cuts=False,
        fix_and_propagate=False,
        feasibility_jump=False,
        feasibility_pump=False,
    )
    assert off["x"] is None
    result = jm.solve(mip, max_nodes=1, cuts=False)
    assert result["x"] is not None and mip.is_feasible(result["x"])
    assert result["objective"] >= ref - 1e-6
    assert result["history"][0]["source"] in ("P", "J", "F")
    assert result["heuristic_seconds"] >= 0.0


def test_feasibility_pump_alone():
    # Seed 5: the pump succeeds without any fix-and-propagate (on some other
    # seeds it needs fix-and-propagate as its rounding step).
    mip = generalised_assignment(5)
    result = jm.solve(
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
    assert_solved(jm.solve(mip), mip, reference(mip))


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
    result = jm.solve(mip, gomory_cuts=True)
    assert_solved(result, mip, ref)
    assert result["cuts"]["gomory"] > 0
    off = jm.solve(mip, gomory_cuts=False)
    assert_solved(off, mip, ref)
    assert off["cuts"]["gomory"] == 0


# %% Feasibility jump
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_feasibility_jump_assignment(seed):
    mip = generalised_assignment(seed)
    for start in (np.zeros(mip.num_variables), lp_point(mip)):
        x, feasible, steps = jm.feasibility_jump(mip, start)
        assert feasible and steps > 0
        assert mip.is_feasible(x)
        np.testing.assert_array_equal(x, np.round(x))


def test_feasibility_jump_knapsack_and_mixed():
    mip = knapsack()
    x, feasible, _ = jm.feasibility_jump(mip, np.ones(mip.num_variables))
    assert feasible and mip.is_feasible(x)
    # Facility location has continuous flows: the jump finds openings and
    # flows that meet every row.
    mip = facility_location()
    x, feasible, _ = jm.feasibility_jump(mip, np.zeros(mip.num_variables))
    assert feasible and mip.is_feasible(x)


def test_feasibility_jump_gives_up_on_infeasible():
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
        jm.MIP(lp, np.ones(n, bool)), np.zeros(n), max_iterations=2000
    )
    assert not feasible and steps == 2000


def test_feasibility_jump_in_the_search():
    mip = generalised_assignment(2)
    result = jm.solve(
        mip, max_nodes=1, cuts=False, fix_and_propagate=False, feasibility_pump=False
    )
    assert result["x"] is not None and mip.is_feasible(result["x"])
    assert result["history"][0]["source"] == "J"


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


def test_highs_incumbent_fallback():
    mip = generalised_assignment(0)
    off = dict(
        max_nodes=1,
        cuts=False,
        fix_and_propagate=False,
        feasibility_jump=False,
        feasibility_pump=False,
    )
    assert jm.solve(mip, **off)["x"] is None  # the fallback is off by default
    result = jm.solve(mip, highs_incumbent_seconds=10, **off)
    assert result["x"] is not None and mip.is_feasible(result["x"])
    assert result["history"][0]["source"] == "X"
    x = jm.highs_incumbent(mip, max_seconds=10)
    assert x is not None and mip.is_feasible(x)


# %% Improving the incumbent
def test_highs_rins_improves_an_incumbent():
    mip, _ = broken_assignment()
    best = milp(
        mip.c,
        constraints=LinearConstraint(mip.A.toarray(), mip.row_lower, mip.row_upper),
        integrality=mip.integrality.astype(int),
        bounds=Bounds(mip.lower, mip.upper),
    )
    optimum = np.round(best.x)
    # HiGHS's first solution, usually worse than the optimum.
    worse = jm.highs_incumbent(mip)
    assert worse is not None
    value = mip.c @ worse
    found = jm.highs_rins(mip, worse, optimum, cutoff=value - 1e-6, max_seconds=10)
    if np.allclose(worse, optimum):
        assert found is None  # nothing better exists
    else:
        assert found is not None and mip.is_feasible(found)
        assert mip.c @ found < value


def test_highs_rins_skips_small_agreement():
    mip, x = broken_assignment()
    # An LP point that agrees nowhere with the incumbent.
    assert jm.highs_rins(mip, x, 1.0 - x, cutoff=np.inf) is None


@pytest.mark.parametrize("plunge_depth", [0, 8])
def test_plunging_keeps_the_optimum(plunge_depth):
    mip = generalised_assignment(2)
    assert_solved(jm.solve(mip, plunge_depth=plunge_depth), mip, reference(mip))


def test_rins_in_the_search():
    mip = generalised_assignment(2)
    result = jm.solve(mip, rins_seconds=5, rins_every=5)
    assert_solved(result, mip, reference(mip))


# %% Batched node solves
@pytest.mark.parametrize(
    "make", [knapsack, general_integer, facility_location, generalised_assignment]
)
def test_batched_search_matches_reference(make):
    mip = make()
    result = jm.solve(mip, batch_size=4, gomory_cuts=False)
    assert_solved(result, mip, reference(mip))
    assert result["batches"] > 0


def test_batch_larger_than_the_tree():
    # Far more slots than open nodes: every batch is padded.
    mip = facility_location()
    result = jm.solve(mip, batch_size=64, gomory_cuts=False)
    assert_solved(result, mip, reference(mip))
    assert 0 < result["batches"] <= result["nodes"]


def test_batched_search_limits_and_infeasibility():
    mip = general_integer(n=12, m_ineq=8)
    ref = reference(mip)
    result = jm.solve(mip, batch_size=4, max_nodes=6)
    assert result["status"] == "node_limit" and result["nodes"] <= 6
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
    assert jm.solve(jm.MIP(lp, np.ones(n, bool)), batch_size=4)["status"] == "infeasible"


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="batch_size"):
        jm.solve(knapsack(), batch_size=0)


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
    result = jm.solve(
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
    result = jm.solve(mip)
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
    result = jm.solve(mip)
    assert result["objective_step"] == 1.0
    assert_solved(result, mip, reference(mip))


# %% Incumbent repair
def search_for(mip, monkeypatch, **options):
    """The ``_BranchAndBound`` that ``solve`` builds, before it runs."""
    monkeypatch.setattr(jm._BranchAndBound, "run", lambda self: self)
    return jm.solve(mip, **options)


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
        return {"solution": solution, "epochs": 1}

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
    on = jm.solve(mip)
    assert on["coefficients_tightened"] > 0
    assert_solved(on, mip, reference(mip))
    off = jm.solve(mip, coefficient_tightening=False)
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
    result = jm.solve(mip, device="cpu")
    assert_solved(result, mip, reference(mip))


def test_device_auto_picks_the_cpu_for_small_mips(monkeypatch):
    search = search_for(knapsack(), monkeypatch, device="auto")
    assert search.device.platform == "cpu"
    assert search_for(knapsack(), monkeypatch, device=None).device is None


def test_device_must_be_known():
    with pytest.raises(ValueError, match="device"):
        jm.solve(knapsack(), device="abacus")


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
    jm.solve(mip, **options)  # compile
    traces.clear()
    result = jm.solve(mip, **options)
    assert result["tree_cuts_added"] > 0
    assert traces == []  # the same shapes throughout


def test_tree_cuts_are_off_when_batched():
    result = jm.solve(knapsack(), tree_cuts=True, batch_size=4)
    assert result["tree_cuts_added"] == 0


# %% Gomory defaults and Jaddle RINS
def test_gomory_auto_follows_the_size(monkeypatch):
    small = search_for(knapsack(), monkeypatch)
    assert small.gomory and small.cut_rounds == 40 and small.cut_min_progress == 1e-3
    large = search_for(generalised_assignment(agents=50, jobs=120), monkeypatch)
    assert large.A_T.nnz > 10_000
    assert not large.gomory and large.cut_rounds == 5 and large.cut_min_progress == 1e-4
    forced = search_for(knapsack(), monkeypatch, gomory_cuts=False, cut_rounds=3)
    assert not forced.gomory and forced.cut_rounds == 3


def test_rins_solver_auto_follows_the_size(monkeypatch):
    assert search_for(knapsack(), monkeypatch).rins_solver == "jaddle"
    large = search_for(generalised_assignment(agents=50, jobs=120), monkeypatch)
    assert large.rins_solver == "highs"
    forced = search_for(knapsack(), monkeypatch, rins_solver="highs")
    assert forced.rins_solver == "highs"


def test_rejects_bad_gomory_and_rins_options():
    with pytest.raises(ValueError, match="gomory_cuts"):
        jm.solve(knapsack(), gomory_cuts="sometimes")
    with pytest.raises(ValueError, match="rins_solver"):
        jm.solve(knapsack(), rins_solver="cplex")


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


@pytest.mark.parametrize("solver", ["jaddle", "highs"])
def test_rins_in_the_search_with_either_solver(solver):
    mip = generalised_assignment(2)
    result = jm.solve(mip, rins_seconds=2, rins_every=10, rins_solver=solver)
    assert_solved(result, mip, reference(mip))
    assert result["rins_calls"] > 0


# %% Stronger cut generator
def max_over_mip(mip, idx, coef):
    """The largest ``coef · x[idx]`` over the MIP's feasible points."""
    objective = np.zeros(mip.num_variables)
    objective[idx] = -coef
    best = milp(
        objective,
        constraints=LinearConstraint(mip.A.toarray(), mip.row_lower, mip.row_upper),
        integrality=mip.integrality.astype(int),
        bounds=Bounds(mip.lower, mip.upper),
    )
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
