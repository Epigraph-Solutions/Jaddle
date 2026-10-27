import numpy as np
import scipy.sparse as sp
import highspy as hspy
import jax.numpy as jnp
import jax.experimental.sparse as jsp
from jaddle.jaddle_basic_types import JaddleLP, LP


def read_model(highs: hspy.Highs, path):
    """``highs.readModel(path)``, retrying an MPS file HiGHS's default (free
    format) parser rejects with its fixed-format parser. Some MIPLIB files
    (rvb-sub) carry fixed-format ``$`` comments in field 3 of ROWS lines,
    which only the fixed parser skips. Raises ``ValueError`` if neither reads
    the file, rather than leaving ``highs`` holding an empty model."""
    path = str(path)
    status = highs.readModel(path)
    if status == hspy.HighsStatus.kError and path.lower().endswith(
        (".mps", ".mps.gz")
    ):
        highs.setOptionValue("mps_parser_type_free", False)
        try:
            status = highs.readModel(path)
        finally:
            highs.setOptionValue("mps_parser_type_free", True)
    if status == hspy.HighsStatus.kError:
        raise ValueError(f"HiGHS could not read {path!r}")
    return status


def highs_to_standard_form_sparse(lp: hspy.HighsLp):
    """
    Converts a HighsLp object to standard form matrices:
        min c^T x
        s.t. A_eq x = b_eq
             A_ineq x <= b_ineq
             x >= lower_bounds
             x <= upper_bounds
    Returns: LP (JAX-native; device-side sparse). Stages that need a scipy
    view (scaling, polish, crossover) call ``LP.to_scipy()``.
    """

    c = np.array(lp.col_cost_, dtype=np.float64)
    lower_bounds = np.array(lp.col_lower_, dtype=np.float64)
    upper_bounds = np.array(lp.col_upper_, dtype=np.float64)

    # Build A matrix from sparse representation
    num_row = lp.a_matrix_.num_row_
    num_col = lp.a_matrix_.num_col_
    A = sp.csc_matrix(
        (lp.a_matrix_.value_, lp.a_matrix_.index_, lp.a_matrix_.start_),
        shape=(num_row, num_col),
        dtype=np.float64,
    )

    row_lower = np.array(lp.row_lower_, dtype=np.float64)
    row_upper = np.array(lp.row_upper_, dtype=np.float64)
    return rows_to_standard_form(c, A, row_lower, row_upper, lower_bounds, upper_bounds)


def rows_to_standard_form(c, A, row_lower, row_upper, lower_bounds, upper_bounds):
    """
    Converts a row-bounded LP
        min c^T x  s.t.  row_lower <= A x <= row_upper,  lower <= x <= upper
    (infinite bounds as +-inf) to the standard form of
    ``highs_to_standard_form_sparse``: rows with equal finite bounds become
    A_eq x = b_eq, every other finite row bound becomes a row of
    A_ineq x <= b_ineq. Returns: LP.
    """
    num_col = A.shape[1]
    eq_rows, upper_rows, lower_rows = standard_form_row_map(row_lower, row_upper)

    A = sp.csr_matrix(A)
    A_eq = A[eq_rows].tocsc().astype(np.float64)
    b_eq = row_lower[eq_rows].astype(np.float64)

    # Inequality rows: a.x <= upper for each finite upper bound, then
    # -a.x <= -lower for each finite lower bound.
    matrices = []
    vectors = []

    if upper_rows.size:
        matrices.append(A[upper_rows].tocsc().astype(np.float64))
        vectors.append(row_upper[upper_rows].astype(np.float64))

    if lower_rows.size:
        matrices.append((-A[lower_rows]).tocsc().astype(np.float64))
        vectors.append((-row_lower[lower_rows]).astype(np.float64))

    if len(matrices) > 0:
        A_ineq = sp.vstack(matrices, format="csc")
        b_ineq = np.concatenate(vectors)
    else:
        # No inequality rows: return empty (0 x num_col) matrix and empty RHS
        A_ineq = sp.csc_matrix((0, num_col), dtype=np.float64)
        b_ineq = np.empty(0, dtype=np.float64)

    return LP(c, A_eq, b_eq, A_ineq, b_ineq, lower_bounds, upper_bounds)


def standard_form_row_map(row_lower, row_upper):
    """Which rows of a row-bounded LP become which rows of
    ``rows_to_standard_form``'s output: ``(eq_rows, upper_rows, lower_rows)``.
    Rows with equal finite bounds (within 1e-8) are the equalities, in order;
    the inequality rows are ``upper_rows`` (finite upper bound,
    ``a.x <= upper``) followed by ``lower_rows`` (finite lower bound,
    ``-a.x <= -lower``). A row free on both sides appears in none."""
    row_lower = np.asarray(row_lower, dtype=np.float64)
    row_upper = np.asarray(row_upper, dtype=np.float64)
    eq_mask = (
        np.isfinite(row_lower)
        & np.isfinite(row_upper)
        & (np.abs(row_lower - row_upper) <= 1e-8)
    )
    ineq = np.flatnonzero(~eq_mask)
    return (
        np.flatnonzero(eq_mask),
        ineq[np.isfinite(row_upper[ineq])],
        ineq[np.isfinite(row_lower[ineq])],
    )


def jaddle_duals_to_highs(dual_eq, dual_ineq, row_lower, row_upper):
    """Row duals of a row-bounded LP from a Jaddle solution of its
    ``rows_to_standard_form``. Jaddle's Lagrangian is ``cᵀx + yᵀ(Ax − b)``
    with ``y >= 0`` on ``<=`` rows, HiGHS's reduced cost is ``c − Aᵀy``, so an
    equality row's dual is ``−y_eq`` and an inequality row's is
    ``y_lower − y_upper``. The column duals then equal Jaddle's reduced cost
    ``c + Aᵀy``."""
    eq_rows, upper_rows, lower_rows = standard_form_row_map(row_lower, row_upper)
    dual_eq, dual_ineq = np.asarray(dual_eq), np.asarray(dual_ineq)
    row_dual = np.zeros(len(row_lower))
    row_dual[eq_rows] = -dual_eq[: eq_rows.size]
    row_dual[upper_rows] -= dual_ineq[: upper_rows.size]
    row_dual[lower_rows] += dual_ineq[
        upper_rows.size : upper_rows.size + lower_rows.size
    ]
    return row_dual


def highs_duals_to_jaddle(row_dual, row_lower, row_upper):
    """Inverse of ``jaddle_duals_to_highs``: ``(dual_eq, dual_ineq)`` for
    ``rows_to_standard_form``'s rows. A two-sided row's HiGHS dual goes to the
    side its sign says is active."""
    eq_rows, upper_rows, lower_rows = standard_form_row_map(row_lower, row_upper)
    row_dual = np.asarray(row_dual)
    dual_ineq = np.concatenate(
        [np.maximum(-row_dual[upper_rows], 0.0), np.maximum(row_dual[lower_rows], 0.0)]
    )
    return -row_dual[eq_rows], dual_ineq


def standard_form_basis_to_highs(basis, row_lower, row_upper):
    """The basis of a row-bounded LP from a ``highspy.HighsBasis`` of its
    ``rows_to_standard_form`` (the rows of ``jaddle_lp_to_highs``). A
    two-sided row is two ``<=`` rows there: it is nonbasic at the bound whose
    copy is nonbasic, and basic when both copies are. A row free on both
    sides is basic."""
    eq_rows, upper_rows, lower_rows = standard_form_row_map(row_lower, row_upper)
    status = hspy.HighsBasisStatus
    standard = np.array(list(basis.row_status), dtype=object)
    nonbasic = standard != status.kBasic
    row_status = np.full(len(row_lower), status.kBasic, dtype=object)
    row_status[eq_rows] = standard[: eq_rows.size]
    row_status[lower_rows[nonbasic[eq_rows.size + upper_rows.size :]]] = status.kLower
    row_status[upper_rows[nonbasic[eq_rows.size : eq_rows.size + upper_rows.size]]] = (
        status.kUpper
    )
    out = hspy.HighsBasis()
    out.col_status = list(basis.col_status)
    out.row_status = list(row_status)
    out.valid = True
    return out


def highs_integrality(lp: hspy.HighsLp):
    """The integer columns of a ``highspy.HighsLp`` as a boolean mask. A model
    with no integrality information is all continuous. Implicit integers
    (which HiGHS's presolve can produce) count as integers. Semi-continuous
    and semi-integer columns raise ``NotImplementedError``."""
    n = lp.num_col_
    kinds = list(lp.integrality_)
    if not kinds:
        return np.zeros(n, dtype=bool)
    VT = hspy.HighsVarType
    unsupported = {VT.kSemiContinuous, VT.kSemiInteger}
    if any(kind in unsupported for kind in kinds):
        raise NotImplementedError(
            "semi-continuous and semi-integer variables are not supported"
        )
    return np.array(
        [kind in (VT.kInteger, VT.kImplicitInteger) for kind in kinds], dtype=bool
    )


def jaddle_lp_to_highs(lp, integrality=None):
    """A ``highspy.HighsLp`` for a Jaddle ``LP`` / ``JaddleLP``: the equality
    rows (``b_eq <= a.x <= b_eq``) then the inequality rows
    (``a.x <= b_ineq``). ``rows_to_standard_form`` maps it back to the same rows in the
    same order. ``integrality`` (a boolean mask of the integer columns) makes
    it a MIP."""
    if isinstance(lp, JaddleLP):
        A_eq = sp.csr_matrix(
            (np.asarray(lp.A_eq.data), np.asarray(lp.A_eq.indices).T),
            shape=lp.A_eq.shape,
        )
        A_ineq = sp.csr_matrix(
            (np.asarray(lp.A_ineq.data), np.asarray(lp.A_ineq.indices).T),
            shape=lp.A_ineq.shape,
        )
    else:
        A_eq, A_ineq = sp.csr_matrix(lp.A_eq), sp.csr_matrix(lp.A_ineq)
    b_eq = np.asarray(lp.b_eq, dtype=np.float64)
    b_ineq = np.asarray(lp.b_ineq, dtype=np.float64)
    A = sp.vstack([A_eq, A_ineq]).tocsc().astype(np.float64)
    A.sort_indices()

    out = hspy.HighsLp()
    out.num_col_ = A.shape[1]
    out.num_row_ = A.shape[0]
    out.col_cost_ = np.asarray(lp.c, dtype=np.float64)
    out.col_lower_ = np.asarray(lp.lower_bounds, dtype=np.float64)
    out.col_upper_ = np.asarray(lp.upper_bounds, dtype=np.float64)
    out.row_lower_ = np.concatenate([b_eq, np.full(b_ineq.shape, -np.inf)])
    out.row_upper_ = np.concatenate([b_eq, b_ineq])
    out.a_matrix_.format_ = hspy.MatrixFormat.kColwise
    out.a_matrix_.start_ = A.indptr
    out.a_matrix_.index_ = A.indices
    out.a_matrix_.value_ = A.data
    out.a_matrix_.num_col_ = A.shape[1]
    out.a_matrix_.num_row_ = A.shape[0]
    if integrality is not None:
        VT = hspy.HighsVarType
        out.integrality_ = [
            VT.kInteger if flag else VT.kContinuous for flag in np.asarray(integrality)
        ]
    return out
