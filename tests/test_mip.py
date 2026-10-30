"""Tests for the branch-and-bound MIP solver, jaddle.jaddle_mip."""

import time

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
    assert_solved(jm.solve(mip), mip, reference(mip))


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
    result = jm.solve(jm.MIP(lp, np.ones(n, bool)), probing=False)
    assert result["status"] == "infeasible"
    assert result["x"] is None and result["objective"] is None
    assert result["nodes"] >= 1
    # Probing tells at the root: every side of x1 leaves an odd right-hand
    # side, and propagation then empties both.
    probed = jm.solve(jm.MIP(lp, np.ones(n, bool)))
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
    # MIR or flow cover cuts (on these fixed-charge rows flow covers often win).
    assert with_cuts["cuts"]["mir"] + with_cuts["cuts"]["flowcover"] > 0
    assert without["cuts"] == {"gomory": 0, "cover": 0, "mir": 0, "flowcover": 0}
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


@pytest.mark.parametrize("make", [general_integer, generalised_assignment])
def test_cut_rounds_drop_slack_cuts(make, monkeypatch):
    """With Gomory cuts, a cut whose row is basic (slack) in a round's
    crossover basis is gone from the next round's LP; binding ones stay."""
    rounds = []  # (cuts before the round, basis) per crossover
    after = []  # the cut list each round's LP was built from
    real_vertex = jm._BranchAndBound.root_vertex
    real_set_mip = jm._BranchAndBound.set_mip
    real_compact = jm._BranchAndBound.compact_cuts
    compacting = []

    def root_vertex(self, lower, upper, solution):
        vertex, basis = real_vertex(self, lower, upper, solution)
        rounds.append((list(self.cut_list), basis))
        return vertex, basis

    def set_mip(self, mip):
        if not compacting and len(after) < len(rounds):
            after.append(list(getattr(self, "cut_list", [])))
        real_set_mip(self, mip)

    def compact_cuts(self, *args):
        compacting.append(True)  # the rounds are over
        return real_compact(self, *args)

    monkeypatch.setattr(jm._BranchAndBound, "root_vertex", root_vertex)
    monkeypatch.setattr(jm._BranchAndBound, "set_mip", set_mip)
    monkeypatch.setattr(jm._BranchAndBound, "compact_cuts", compact_cuts)
    mip = make()
    result = jm.solve(mip, gomory_cuts=True, max_nodes=1)
    assert result["cuts"]["gomory"] > 0
    m = mip.A.shape[0]
    checked = 0
    for (before, basis), cuts in zip(rounds, after):
        if basis is None:
            continue
        kept = {id(cut) for cut in cuts}
        for i, cut in enumerate(before):
            basic = basis.row_status[m + i] == highspy.HighsBasisStatus.kBasic
            assert (id(cut) in kept) == (not basic)
            checked += 1
    assert checked > 0


# %% Conflicts
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
    """Every row the conflict pool keeps, as ``(idx, coef, rhs, cutoff)``:
    ``coef · x[idx] <= rhs`` must hold at points with ``c·x <= cutoff``."""
    rows = []
    real = jm.ConflictPool._add_row

    def add_row(self, idx, coef, offset, objective, kind, *args):
        kept = real(self, idx, coef, offset, objective, kind, *args)
        if kept:
            cutoff = self.cutoff_value()
            rhs = offset + (cutoff if objective else 0.0)
            rows.append((idx.copy(), coef.copy(), rhs, cutoff, kind))
        return kept

    monkeypatch.setattr(jm.ConflictPool, "_add_row", add_row)
    return rows


@pytest.mark.parametrize("make", [general_integer, generalised_assignment])
def test_conflicts_are_valid(make, monkeypatch):
    mip = make()
    rows = record_conflicts(monkeypatch)
    result = jm.solve(mip, conflicts=True)
    assert_solved(result, mip, reference(mip))
    for idx, coef, rhs, cutoff, _ in rows:
        bound = None if not np.isfinite(cutoff) else cutoff
        most = max_over_mip(mip, idx, coef, objective_at_most=bound)
        assert most <= rhs + 1e-6 * (1 + abs(rhs))
    if make is general_integer:
        assert rows and result["conflict_pruned"] > 0


def test_farkas_proof_excludes_an_infeasible_box(monkeypatch):
    mip = knapsack()
    search = search_for(mip, monkeypatch, conflicts=True)
    search.root_lower, search.root_upper = mip.lower.copy(), mip.upper.copy()
    # Fix the heaviest items at 1 until they overflow the knapsack.
    w = np.asarray(mip.A.toarray()[0])
    lower, upper = mip.lower.copy(), mip.upper.copy()
    for j in np.argsort(-w):
        lower[j] = 1.0
        if w @ lower > mip.b[0]:
            break
    detection = search.detect(search.node_lp(lower, upper), None, 2000)
    assert detection["status"] == "primal_infeasible"
    search.learn_farkas(detection, lower, upper)
    assert len(search.conflicts) == 1
    assert search.conflicts.propagate(lower, upper)[2]
    assert not search.conflicts.propagate(mip.lower, mip.upper)[2]


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
    result = jm.solve(mip, conflicts=False)
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
    result = jm.solve(
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
        jm.solve(knapsack(), jump_backend="cuda")


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
    result = jm.solve(
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
    jm.solve(
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
    jm.solve(
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
    result = jm.solve(mip, **tree_heuristic_options())
    assert_solved(result, mip, reference(mip))
    assert result["tree_heuristic_calls"] > 0
    assert result["tree_heuristic_seconds"] > 0
    assert result["tree_heuristic_finds"] <= result["tree_heuristic_calls"]
    off = jm.solve(mip, **tree_heuristic_options(tree_heuristics=False))
    assert off["tree_heuristic_calls"] == 0 and off["tree_heuristic_finds"] == 0


def test_tree_heuristics_need_an_incumbent_and_stay_out_of_sub_searches(monkeypatch):
    calls = []
    real = jm._BranchAndBound.tree_heuristics

    def recorded(self, *args):
        calls.append((self.incumbent is not None, np.isfinite(self.initial_cutoff)))
        return real(self, *args)

    monkeypatch.setattr(jm._BranchAndBound, "tree_heuristics", recorded)
    mip = generalised_assignment(2)
    result = jm.solve(
        mip,
        **tree_heuristic_options(rins_seconds=5, rins_every=5, rins_solver="jaddle"),
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
        jm.solve(knapsack(), **options)


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
)


def test_node_crossover_turns_a_face_point_into_an_integral_vertex():
    mip = pick_one()
    result = jm.solve(mip, node_crossover=True, **NO_HEURISTICS)
    assert_solved(result, mip, reference(mip))
    assert result["nodes"] == 1  # the root vertex is integral: no branching
    assert [h["source"] for h in result["history"]] == ["T"]
    assert result["node_crossovers"] == result["node_crossover_vertices"] == 1
    assert result["node_crossover_integral"] == 1
    off = jm.solve(mip, **NO_HEURISTICS)
    assert off["nodes"] > 1 and off["node_crossovers"] == 0


@pytest.mark.parametrize("batch_size", [1, 4])
def test_node_crossover_in_the_search(batch_size):
    mip = general_integer()
    result = jm.solve(mip, node_crossover=True, batch_size=batch_size)
    assert_solved(result, mip, reference(mip))
    assert 0 < result["node_crossover_vertices"] <= result["node_crossovers"]
    assert result["node_crossover_seconds"] > 0


def test_node_crossover_every_and_failed_crossovers(monkeypatch):
    mip = general_integer()
    sparse = jm.solve(mip, node_crossover=True, node_crossover_every=1000)
    assert_solved(sparse, mip, reference(mip))
    assert sparse["node_crossovers"] <= 1  # only the node count 0 qualifies
    # A crossover HiGHS cannot finish leaves the PDHG point in place.
    monkeypatch.setattr(
        jm.jl, "crossover", lambda lp, solution, highs_options=None: {"converged": False}
    )
    result = jm.solve(mip, node_crossover=True)
    assert_solved(result, mip, reference(mip))
    assert result["node_crossovers"] > 0 and result["node_crossover_vertices"] == 0


@pytest.mark.parametrize(
    "options", [dict(node_crossover_every=0), dict(node_crossover_seconds=0)]
)
def test_rejects_bad_node_crossover_options(options):
    with pytest.raises(ValueError):
        jm.solve(knapsack(), **options)


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
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("mip_max_improving_sols", 1)
    highs.passModel(jm.to_highs(mip))
    highs.run()
    worse = np.array(highs.getSolution().col_value)
    assert mip.is_feasible(worse)
    value = mip.c @ worse
    found = jm.highs_rins(mip, worse, optimum, cutoff=value - 1e-6, max_seconds=10)
    if value <= best.fun + 1e-6:
        # Nothing better exists.
        assert found is None or mip.c @ found >= best.fun - 1e-6
    else:
        assert found is not None and mip.is_feasible(found)
        assert mip.c @ found < value


def test_highs_submip_respects_the_box_and_cutoff():
    mip = generalised_assignment(0)
    best = milp(
        mip.c,
        constraints=LinearConstraint(mip.A.toarray(), mip.row_lower, mip.row_upper),
        integrality=mip.integrality.astype(int),
        bounds=Bounds(mip.lower, mip.upper),
    )
    optimum = np.round(best.x)
    # Fix half the variables at the optimum; the rest stay free.
    fixed = np.arange(mip.num_variables) % 2 == 0
    lower = np.where(fixed, optimum, mip.lower)
    upper = np.where(fixed, optimum, mip.upper)
    found = jm.highs_submip(mip, lower, upper, cutoff=best.fun + 1.0, max_seconds=10)
    assert found is not None and mip.is_feasible(found)
    assert np.all(found >= lower - 1e-9) and np.all(found <= upper + 1e-9)
    assert mip.c @ found <= best.fun + 1.0
    # Nothing in the box beats a cutoff below the optimum.
    assert jm.highs_submip(mip, lower, upper, cutoff=best.fun - 1.0, max_seconds=10) is None


@pytest.mark.parametrize(
    "make", [general_integer, facility_location, lambda: fixed_charge(0)[0]]
)
def test_root_rens_with_highs_finds_the_first_incumbent(make):
    mip = make()
    result = jm.solve(
        mip, gomory_cuts=True, feasibility_jump=False, fix_and_propagate=False,
        feasibility_pump=False, bulk_rounding=False,
    )
    assert_solved(result, mip, reference(mip))
    assert result["history"][0]["source"] == "E"  # RENS
    assert result["submip_calls"] >= 1


def test_heuristic_effort_zero_turns_rins_off():
    mip = general_integer()
    result = jm.solve(mip, heuristic_effort=0.0)
    assert_solved(result, mip, reference(mip))
    assert result["rins_calls"] == 0
    assert jm.solve(mip)["rins_calls"] > 0  # on by default


def test_rins_node_limit_adapts(monkeypatch):
    mip = general_integer()
    search = search_for(mip, monkeypatch)
    search.incumbent = (float(mip.c @ np.zeros(mip.num_variables)), np.zeros(mip.num_variables))
    calls = iter([None, None])
    monkeypatch.setattr(search, "highs_rins_search", lambda x, seconds: next(calls))
    nodes = search.rins_nodes
    search.rins(np.zeros(mip.num_variables), None)
    assert search.rins_nodes == max(50, nodes // 2)  # a failed call halves it
    assert search.submip_calls == 1


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
    # Without heuristics: with an incumbent from them, root reduced-cost
    # fixing can close the search before it branches.
    result = jm.solve(mip, batch_size=4, gomory_cuts=False, **no_heuristics())
    assert_solved(result, mip, reference(mip))
    assert result["batches"] > 0


def test_batch_larger_than_the_tree():
    # Far more slots than open nodes: every batch is padded.
    mip = facility_location()
    # Without heuristics, so root reduced-cost fixing cannot end it at the root.
    result = jm.solve(mip, batch_size=64, gomory_cuts=False, **no_heuristics())
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


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("make", [knapsack, general_integer, generalised_assignment])
def test_logged_bound_never_falls(monkeypatch, make, batch_size):
    # Nodes in no list fell out of the global bound, so the logged bound read
    # too high, then fell back: the root during the root heuristics (an
    # incumbent showed a 0% gap) and a batch's members not yet evaluated.
    logged = []
    real = jm._BranchAndBound.log

    def recorded(self, source, bound, depth, force=False):
        logged.append(self.global_bound() if bound is None else bound)
        return real(self, source, bound, depth, force)

    monkeypatch.setattr(jm._BranchAndBound, "log", recorded)
    mip = make()
    result = jm.solve(mip, batch_size=batch_size, gomory_cuts=False)
    assert_solved(result, mip, reference(mip))
    assert logged
    finite = [b for b in logged if np.isfinite(b)]
    assert all(b <= a + 1e-9 * (1 + abs(a)) for a, b in zip(finite[1:], finite))


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
    assert search_for(knapsack(), monkeypatch).rins_solver == "highs"  # the default
    small = search_for(knapsack(), monkeypatch, rins_solver="auto")
    assert small.rins_solver == "jaddle"
    large = search_for(
        generalised_assignment(agents=50, jobs=120), monkeypatch, rins_solver="auto"
    )
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
def max_over_mip(mip, idx, coef, objective_at_most=None):
    """The largest ``coef · x[idx]`` over the MIP's feasible points (with
    ``c·x <= objective_at_most``, if given; ``-inf`` when there are none)."""
    objective = np.zeros(mip.num_variables)
    objective[idx] = -coef
    A, row_lower, row_upper = mip.A.toarray(), mip.row_lower, mip.row_upper
    if objective_at_most is not None:
        A = np.vstack([A, mip.c])
        row_lower = np.append(row_lower, -np.inf)
        row_upper = np.append(row_upper, objective_at_most)
    best = milp(
        objective,
        constraints=LinearConstraint(A, row_lower, row_upper),
        integrality=mip.integrality.astype(int),
        bounds=Bounds(mip.lower, mip.upper),
    )
    if objective_at_most is not None and best.status == 2:  # infeasible
        return -np.inf
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
    on = jm.solve(mip, node_bound_stop=True)
    off = jm.solve(mip, node_bound_stop=False)
    assert_solved(on, mip, ref)
    assert_solved(off, mip, ref)
    assert off["bound_stops"] == 0
    if make is general_integer:
        # Without sub-MIP heuristics, so the two trees differ only by the stop.
        plain = dict(rins_seconds=0.0, rens_seconds=0.0)
        on = jm.solve(mip, node_bound_stop=True, **plain)
        off = jm.solve(mip, node_bound_stop=False, **plain)
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
    result = jm.solve(mip, node_lp_tolerance=3e-4, feasibility_pump=False)
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
        jm.solve(knapsack(), separation_backend="gpu")
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
    result = jm.solve(mip, gomory_cuts=False)
    assert_solved(result, mip, reference(mip))
    assert result["cuts"]["flowcover"] >= 0
    # Pure integer rows get none.
    assert jm.solve(knapsack(), max_nodes=1)["cuts"]["flowcover"] == 0


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


@pytest.mark.parametrize("make", [knapsack, general_integer])
def test_per_call_precision(make):
    import jax
    import jax.numpy as jnp

    mip = make()  # built in float64
    assert_solved(jm.solve(mip, precision="float32"), mip, reference(mip))
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
    detect = jm._BranchAndBound.detect

    def spy_solve_lp(self, lp, warm, max_epochs, **overrides):
        if "precision" in overrides:
            seen["node"].add(str(lp.c.dtype))
        return solve_lp(self, lp, warm, max_epochs, **overrides)

    def spy_detect(self, lp, result, max_epochs):
        seen["root"].add(str(lp.c.dtype))
        return detect(self, lp, result, max_epochs)

    monkeypatch.setattr(jm._BranchAndBound, "solve_lp", spy_solve_lp)
    monkeypatch.setattr(jm._BranchAndBound, "detect", spy_detect)
    mip = general_integer()
    result = jm.solve(mip, precision="float64", node_precision="float32", **options)
    assert_solved(result, mip, reference(mip))
    assert seen["root"] == {"float64"}
    if options.get("batch_size", 1) == 1:  # serial node LPs go through solve_lp
        assert seen["node"] == {"float32"}


def test_node_precision_above_the_search_precision():
    import jax.numpy as jnp

    mip = general_integer()
    result = jm.solve(mip, precision="float32", node_precision="float64", batch_size=4)
    assert_solved(result, mip, reference(mip))
    assert mip.lp.c.dtype == jnp.float64


def test_node_precision_through_node_lp_options():
    mip = general_integer()
    result = jm.solve(mip, node_lp_options={"precision": "float32"}, batch_size=4)
    assert_solved(result, mip, reference(mip))
    with pytest.raises(ValueError, match="disagree"):
        jm.solve(mip, node_precision="float16", node_lp_options={"precision": "float32"})
    with pytest.raises(ValueError, match="float8"):
        jm.solve(mip, node_precision="float8")


def test_lp_repair_after_lower_precision_batched_nodes():
    # Batched node LPs at float16 used the LP's sort permutations with 64-bit
    # off; repairing a node's rounding at float64 then failed in JAX's
    # indexing ("lax.add requires arguments to have the same dtypes").
    mip = facility_location()
    result = jm.solve(
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
    result = jm.solve(mip, batch_size=1, tree_cuts=True, **options)
    assert_solved(result, mip, reference(mip))


def test_lp_options_cannot_set_device_or_precision():
    for key, value in (("device", "cpu"), ("precision", "float32")):
        with pytest.raises(ValueError, match=key):
            jm.solve(knapsack(), lp_options={key: value})


def test_device_accepts_jax_device():
    import jax

    mip = knapsack()
    assert_solved(jm.solve(mip, device=jax.devices("cpu")[0]), mip, reference(mip))


# %% Reliability branching
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize(
    "make", [knapsack, general_integer, facility_location, generalised_assignment]
)
def test_reliability_branching_matches_reference(make, batch_size):
    mip = make()
    result = jm.solve(mip, branching="reliability", batch_size=batch_size)
    assert_solved(result, mip, reference(mip))
    if result["nodes"] > 1:
        assert result["strong_branch_lps"] > 0


def test_reliability_branching_needs_fewer_nodes():
    mip = general_integer()
    plain = jm.solve(mip, branching="pseudocost")
    strong = jm.solve(mip, branching="reliability")
    assert_solved(strong, mip, reference(mip))
    assert strong["strong_branch_nodes"] > 0
    assert strong["nodes"] < plain["nodes"]


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
    result = jm.solve(mip, branching="reliability", probing=False, **no_heuristics())
    assert result["status"] == "optimal"
    assert result["objective"] == pytest.approx(1.0, abs=1e-6)
    assert result["strong_fixings"] >= 1
    probed = jm.solve(mip, branching="reliability", **no_heuristics())
    assert probed["probing"]["fixed"] >= 1
    assert probed["objective"] == pytest.approx(1.0, abs=1e-6)


def test_strong_branching_bounds_stay_valid():
    # The node bounds strong branching raises must stay below the optimum.
    mip = general_integer(n=12, m_ineq=8)
    ref = reference(mip)
    for max_nodes in (2, 4, 8):
        result = jm.solve(
            mip, branching="reliability", max_nodes=max_nodes, **no_heuristics()
        )
        assert result["bound"] <= ref + 1e-6 * (1 + abs(ref))
        if result["objective"] is not None:
            assert result["objective"] >= ref - 1e-6 * (1 + abs(ref))


def test_strong_branching_with_tree_cuts():
    # Tree cuts change the LP's shape, so the children are solved one by one.
    mip = general_integer()
    result = jm.solve(mip, branching="reliability", tree_cuts=True, tree_cut_every=1)
    assert_solved(result, mip, reference(mip))
    assert result["strong_branch_lps"] > 0


def test_strong_branching_depth_limit():
    mip = general_integer()
    result = jm.solve(mip, branching="reliability", strong_max_depth=-1)
    assert_solved(result, mip, reference(mip))
    assert result["strong_branch_nodes"] == 0


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_strong_backends_solve_to_the_reference(backend):
    for make in (general_integer, facility_location):
        mip = make()
        result = jm.solve(mip, branching="reliability", strong_backend=backend)
        assert_solved(result, mip, reference(mip))


def test_strong_backend_is_checked():
    with pytest.raises(ValueError, match="strong_backend"):
        jm.solve(general_integer(), branching="reliability", strong_backend="cuda")


def test_reliability_branching_in_float32():
    mip = general_integer()
    result = jm.solve(mip, branching="reliability", precision="float32")
    assert_solved(result, mip, reference(mip))
    assert result["strong_branch_lps"] > 0


def test_strong_branching_with_a_small_screen():
    # More candidates than screen slots: the unreliable ones go first.
    mip = general_integer(n=12, m_ineq=8)
    result = jm.solve(
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
    result = jm.solve(mip, branching=branching, **no_heuristics())
    assert_solved(result, mip, reference(mip), rigorous=False)
    assert result["root_strong_lps"] > 0
    assert calls and calls[0] == 0  # the cheap technology's size


def test_root_sweep_off():
    mip = sizing()
    result = jm.solve(mip, root_strong_candidates=0, **no_heuristics())
    assert_solved(result, mip, reference(mip), rigorous=False)
    assert result["root_strong_lps"] == 0
    assert jm.solve(mip, branching="most_fractional")["root_strong_lps"] == 0


@pytest.mark.parametrize("batch_size", [1, 4])
def test_root_sweep_solves_to_the_reference(batch_size):
    for make in (knapsack, general_integer, facility_location):
        mip = make()
        result = jm.solve(mip, batch_size=batch_size)
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
    # No fix-and-propagate and (almost) no root jumps, so the pump, with
    # jumps from its roundings, finds the first point.
    mip = generalised_assignment(2)
    settings = dict(
        max_nodes=1,
        cuts=False,
        fix_and_propagate=False,
        jump_iterations=1,
        rens_seconds=0.0,
        rins_seconds=0.0,
    )
    fused = jm.solve(mip, **settings)
    assert fused["x"] is not None and mip.is_feasible(fused["x"])
    assert fused["history"][0]["source"] == "F"
    assert fused["pump_jumps"] >= fused["pump_jump_finds"]


def test_fused_pump_jumps_can_be_turned_off():
    mip = generalised_assignment(5)
    result = jm.solve(
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
        result = jm.solve(
            mip, max_nodes=1, cuts=False, feasibility_jump=False, feasibility_pump=False
        )
        if result["x"] is not None:
            assert mip.is_feasible(result["x"])
            assert result["history"][0]["source"] in ("P", "B")
    off = jm.solve(
        generalised_assignment(0), max_nodes=1, cuts=False, fix_and_propagate=False,
        feasibility_jump=False, feasibility_pump=False,
    )
    assert off["x"] is None


# %% Batched children and roundings on the device
@pytest.mark.parametrize("make", [general_integer, facility_location, generalised_assignment])
def test_child_kernels_match_the_host(make):
    mip = make()
    rng = np.random.default_rng(5)
    propagator = jm.BoundPropagator(mip.A, mip.row_lower, mip.row_upper, mip.integrality)
    size, n = 4, mip.num_variables
    kernels = jm._ChildKernels(mip, True, 1e-9, size)
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
    result = jm.solve(mip, batch_size=4, gomory_cuts=False, children_backend="jax")
    assert_solved(result, mip, reference(mip))
    if make is general_integer:  # it always branches in batches
        assert result["batched_children"] > 0


def test_batched_tree_roundings():
    mip = generalised_assignment(3)
    result = jm.solve(
        mip, batch_size=4, gomory_cuts=False, children_backend="jax", heuristic_every=1,
        batch_roundings=True,
    )
    assert_solved(result, mip, reference(mip))
    if result["batches"]:
        assert result["batched_roundings"] > 0


def test_children_backend_is_checked():
    with pytest.raises(ValueError, match="children_backend"):
        jm.solve(general_integer(), batch_size=4, children_backend="cuda")


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
    # One epoch a call: almost every node LP goes back to the queue.
    mip = make()
    result = jm.solve(mip, batch_size=4, batch_epochs=1, gomory_cuts=False)
    assert_solved(result, mip, reference(mip))
    if make is general_integer:
        assert result["batch_requeues"] > 0


def test_batch_epochs_is_checked():
    with pytest.raises(ValueError, match="batch_epochs"):
        jm.solve(general_integer(), batch_size=4, batch_epochs=0)


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
        assert jm.solve(mip, batch_size=batch_size)["status"] == "infeasible"


# %% Node LP options
def test_node_lp_options_reach_node_lps_only(monkeypatch):
    seen = {"node": [], "root": []}
    solve_lp = jm._BranchAndBound.solve_lp
    detect = jm._BranchAndBound.detect

    def spy_solve_lp(self, lp, warm, max_epochs, **overrides):
        seen["node"].append(dict(self.lp_kwargs, **overrides).get("iterations_per_epoch"))
        return solve_lp(self, lp, warm, max_epochs, **overrides)

    def spy_detect(self, lp, result, max_epochs):
        seen["root"].append(self.lp_kwargs.get("iterations_per_epoch"))
        return detect(self, lp, result, max_epochs)

    monkeypatch.setattr(jm._BranchAndBound, "solve_lp", spy_solve_lp)
    monkeypatch.setattr(jm._BranchAndBound, "detect", spy_detect)
    mip = general_integer()
    result = jm.solve(
        mip, node_lp_options={"iterations_per_epoch": 64}, feasibility_pump=False,
        gomory_cuts=False,
    )
    assert_solved(result, mip, reference(mip))
    assert seen["node"] and all(value == 64 for value in seen["node"])
    assert seen["root"] and all(value is None for value in seen["root"])


def test_batched_node_lps_get_node_settings(monkeypatch):
    import functools

    calls = []
    make_solver = jl.make_solver

    @functools.wraps(make_solver)  # keeps the signature the check reads
    def spy(build_lp, **kwargs):
        calls.append(kwargs)
        return make_solver(build_lp, **kwargs)

    monkeypatch.setattr(jl, "make_solver", spy)
    mip = general_integer()
    result = jm.solve(
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


def test_node_lp_options_are_checked():
    with pytest.raises(ValueError, match="node_lp_options cannot set max_epochs"):
        jm.solve(general_integer(), node_lp_options={"max_epochs": 5})
    with pytest.raises(ValueError, match="not_an_option"):
        jm.solve(general_integer(), batch_size=4, node_lp_options={"not_an_option": 1})


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


# %% Background heuristics (parallel_heuristics)
@pytest.mark.parametrize(
    "make", [knapsack, general_integer, facility_location, generalised_assignment]
)
def test_parallel_heuristics_match_reference(make):
    mip = make()
    result = jm.solve(
        mip, parallel_heuristics=True, highs_incumbent_seconds=2.0, rins_every=2
    )
    assert_solved(result, mip, reference(mip))
    stats = result["parallel_heuristics"]
    assert set(stats) == {"jobs", "seconds", "improvements", "jump_slices"}


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
        assert workers.idle("highs")
        workers.submit("highs", "a", job, 1, info="first")
        workers.submit("highs", "b", job, 2)
        assert not workers.idle("highs")
        done = []
        deadline = time.time() + 10
        while len(done) < 2 and time.time() < deadline:
            done += workers.finished()
        assert [(kind, out, info) for kind, out, _, info in done] == [
            ("a", 2, "first"),
            ("b", 4, None),
        ]
        assert seen == [1, 2] and workers.idle("highs")
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
        workers.submit("highs", "x", fails)
        with pytest.raises(RuntimeError, match="boom"):
            deadline = time.time() + 10
            while time.time() < deadline:
                workers.finished()
    finally:
        workers.shutdown()


def test_highs_submip_with_a_prebuilt_model():
    from jaddle import highs_helpers as hh

    mip = general_integer()
    model = hh.jaddle_lp_to_highs(mip.lp, mip.integrality)
    lower, upper = mip.lower.copy(), mip.upper.copy()
    upper[0] = lower[0] = np.clip(0.0, mip.lower[0], mip.upper[0])
    plain = jm.highs_submip(mip, lower, upper, max_seconds=10.0)
    shared = jm.highs_submip(mip, lower, upper, max_seconds=10.0, model=model)
    assert plain is not None and shared is not None
    assert mip.c @ shared == pytest.approx(mip.c @ plain)
    assert shared[0] == lower[0]
    # The shared model keeps its own bounds.
    np.testing.assert_array_equal(np.asarray(model.col_upper_), mip.upper)


def test_highs_stops_when_asked():
    import threading

    stop = threading.Event()
    stop.set()
    mip = generalised_assignment(1)
    started = time.time()
    jm.highs_incumbent(mip, max_seconds=60.0, stop=stop)
    assert time.time() - started < 10.0


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
