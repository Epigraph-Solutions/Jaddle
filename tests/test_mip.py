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

    result = jm.solve(mip, max_seconds=0.5)
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
