# %% [markdown]
# # Big-M scanner (cost and matrix)
#
# Scans every `.mps` file in a directory and flags instances with a big-M /
# penalty structure. Two distinct structures, two distinct failure modes:
#
# 1. **Cost big-M** (`is_bigm_cost`): the *objective* spans a wide dynamic range
#    between the largest and the median nonzero cost coefficient, plus a large
#    absolute coefficient. This defeats Jaddle's saddle-point LP solver (e.g.
#    `glass4`, `binkar10_1`): the optimum is dominated by a handful of ~1e6
#    penalty costs, so a constraint residual that is tiny relative to the RHS
#    norm still leaves an enormous *objective* gap. The solver parks at a
#    primal/dual-feasible point with a frozen relative duality gap, far above
#    the true optimum.
#
# 2. **Matrix big-M** (`is_bigm_matrix`): the *constraint matrix* has rows where
#    one coefficient dwarfs its rowmates (e.g. `sp150x300d`: 300 rows of the
#    form [1, 1, ..., 1, 3050]). This is the constraint-side mirror of cost
#    big-M and stalls the saddle solver on a primal-feasibility plateau: Ruiz
#    scaling equilibrates whole rows/columns but cannot flatten the *within-row*
#    anisotropy, so the step geometry stays wildly skewed and a residual that is
#    tiny relative to ||b|| can still leave a binary far from {0,1}.
#
# 3. **Column big-M** (`is_bigm_column`): the anisotropy lives WITHIN columns
#    rather than within rows, funneled through one (or a few) DENSE rows with a
#    large absolute coefficient shared across most columns (e.g. `germanrr`:
#    a single dense equality `sum a_j x_j = 0`, 10575 nonzeros, coefficients
#    spanning 1 -> 8.2e5; also `leo1`, `trento1`). Every column then carries its
#    ~O(1) entries in the unit rows PLUS one ~1e5 entry in the shared big row,
#    so the per-COLUMN spread is huge even though rows look fine. This is the
#    column-side mirror of the row big-M and stalls the saddle solver even
#    harder: Ruiz equilibrates whole rows and whole columns but cannot flatten
#    within-column anisotropy threaded through a single shared row -- it is
#    forced into a bad compromise (crush the big row and lose every other
#    constraint's contribution to that column, or leave it dominant), so the
#    primal never approaches feasibility (germanrr: PFR frozen ~1e7, objective
#    still drifting at 80 epochs). Detecting this via per-column spread alone is
#    noisy (tiny min-entries fabricate a large ratio over max|A| ~ O(1), e.g.
#    ran14x18-disj-8, supportcase42), so the flag keys on the ROOT CAUSE: at
#    least one row that is BOTH wide in absolute terms (max|A_row| large) AND
#    dense (touches a high fraction of columns). That "dense big-M row" cleanly
#    separates germanrr/leo1/trento1 from the benign small-coefficient cases.
#
# A raw `max|A|` threshold is a much noisier signal -- a single large entry is
# often just units/scaling that Ruiz handles. So the matrix flag keys on the
# *per-row* spread (max/min within a row) being large across a *high fraction*
# of rows: a pervasive, systematic pattern that scaling cannot absorb, not a
# lone outlier. The three flags are independent; an instance may trip any,
# all, or none, and `benchmark.py` exposes them as separate skip flags.
#
# Usage:
#     python benchmarks/scan_bigm.py
#     python benchmarks/scan_bigm.py --data-dir data/pdlp_set --ratio 1e3 --abs 1e4
#     python benchmarks/scan_bigm.py --row-spread 1e3 --row-frac 0.5
#     python benchmarks/scan_bigm.py --col-row-abs 1e3 --col-row-density 0.3

# %%
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import glob

import numpy as np
import highspy as hspy
import jaddle.highs_helpers as hh

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "pdlp_set")

# Default big-M COST thresholds: flag when the objective spans a wide dynamic
# range (max/median over nonzero costs) AND the penalty is significant -- either
# in absolute terms (max|c| large, e.g. glass4's 1e6) OR relative to a tiny bulk
# of costs (median|c| below `DEFAULT_MED_FLOOR`, e.g. mas74/mas76 where max|c|=1
# but median|c|=1e-5, so cost-1 columns dominate a sea of 1e-5 costs). The
# absolute gate alone misses the tiny-median family, which slips under any
# reasonable abs threshold yet has the same penalty structure that stalls the
# saddle solver. Shared with benchmark.py's --skip-bigm so both use one
# definition.
DEFAULT_RATIO = 1e3
DEFAULT_ABS = 1e3
DEFAULT_MED_FLOOR = 1e-3

# Default big-M MATRIX thresholds: flag when the per-row coefficient spread
# (max|A_row| / min|A_row| over nonzero entries) has a large MEDIAN across rows
# AND a large FRACTION of rows are themselves wide. Keying on the median (not
# the max) row spread demands the wide-coefficient pattern be PERVASIVE rather
# than a single outlier row -- a lone skewed row is usually benign units that
# Ruiz scaling absorbs, whereas sp150x300d's 300 identical [1,...,1,3050] rows
# are a systematic big-M coupling that survives equilibration. Calibrated on
# data/pdlp_set/: this pair flags sp150x300d (median spread 3050, 67% of rows wide) and
# neos-3754480-nidda (1280, 74%) along with ~19 other known-hard instances,
# without firing on benign mild-scaling models. Shared with benchmark.py's
# --skip-bigm-matrix so both use one definition.
DEFAULT_ROW_SPREAD = 1e3
DEFAULT_ROW_FRAC = 0.5
DEFAULT_WIDE_ROW = 1e2

# Default big-M COLUMN thresholds: flag when at least one "dense big-M row"
# exists -- a row whose largest ABSOLUTE coefficient exceeds DEFAULT_COL_ROW_ABS
# AND whose nonzeros touch at least DEFAULT_COL_ROW_DENSITY of all columns. Such
# a row threads a ~1e5 entry through most columns' support, so the per-column
# spread explodes even though rows look fine; Ruiz cannot flatten it (see
# is_bigm_column). Keying on absolute magnitude + density -- rather than the raw
# per-column spread ratio -- avoids the tiny-min-entry false positives
# (ran14x18-disj-8, supportcase42, where max|A| ~ O(1)). Calibrated on data/pdlp_set/:
# flags germanrr (1 dense row, max|A|=8.2e5), leo1 (9.0e7), trento1 (1.0e8)
# without firing on glass4 (351 wide rows but none dense -- that is cost/row
# big-M) or the small-coefficient cases. Shared with benchmark.py's
# --skip-bigm-column so both use one definition.
DEFAULT_COL_ROW_ABS = 1e3
DEFAULT_COL_ROW_DENSITY = 0.3


def parse_args():
    p = argparse.ArgumentParser(
        description="Flag MPS instances with a big-M / penalty cost structure."
    )
    p.add_argument("--data-dir", default=DATA_DIR, help="Directory of .mps files.")
    p.add_argument(
        "--ratio",
        type=float,
        default=DEFAULT_RATIO,
        help="Flag when max|c| / median|c| over nonzero costs exceeds this "
        f"(default {DEFAULT_RATIO:g}).",
    )
    p.add_argument(
        "--abs",
        type=float,
        default=DEFAULT_ABS,
        help="With a wide ratio, flag when max|c| exceeds this absolute value "
        f"(default {DEFAULT_ABS:g}). A wide ratio over uniformly tiny costs is "
        "not flagged by this path alone -- see --med-floor for the "
        "relative-penalty (tiny-median) path.",
    )
    p.add_argument(
        "--med-floor",
        type=float,
        default=DEFAULT_MED_FLOOR,
        help="With a wide ratio, also flag when median|c| is below this value "
        f"(default {DEFAULT_MED_FLOOR:g}), catching the relative-penalty family "
        "(e.g. mas74/mas76: max|c|=1, median|c|=1e-5) that slips under --abs.",
    )
    p.add_argument(
        "--row-spread",
        type=float,
        default=DEFAULT_ROW_SPREAD,
        help="Matrix big-M: flag when the MEDIAN per-row coefficient spread "
        "(max|A_row|/min|A_row|) over rows exceeds this "
        f"(default {DEFAULT_ROW_SPREAD:g}).",
    )
    p.add_argument(
        "--row-frac",
        type=float,
        default=DEFAULT_ROW_FRAC,
        help="Matrix big-M: with a wide median row spread, also require this "
        f"fraction of rows to be wide (default {DEFAULT_ROW_FRAC:g}), i.e. the "
        "skew is pervasive rather than a single outlier row.",
    )
    p.add_argument(
        "--wide-row",
        type=float,
        default=DEFAULT_WIDE_ROW,
        help="Matrix big-M: a row counts as 'wide' for --row-frac when its "
        f"spread exceeds this (default {DEFAULT_WIDE_ROW:g}).",
    )
    p.add_argument(
        "--col-row-abs",
        type=float,
        default=DEFAULT_COL_ROW_ABS,
        help="Column big-M: flag when a 'dense big-M row' exists -- a row whose "
        f"largest ABSOLUTE coefficient exceeds this (default {DEFAULT_COL_ROW_ABS:g}) "
        "and that touches >= --col-row-density of all columns (e.g. germanrr's "
        "dense value-balance equality).",
    )
    p.add_argument(
        "--col-row-density",
        type=float,
        default=DEFAULT_COL_ROW_DENSITY,
        help="Column big-M: a row counts as 'dense' when its nonzeros touch at "
        f"least this fraction of all columns (default {DEFAULT_COL_ROW_DENSITY:g}).",
    )
    return p.parse_args()


class InstanceArrays:
    """Raw coefficient arrays for one model, read from an MPS file exactly once.

    All three big-M checks derive from the same three ingredients -- the cost
    vector and the (column-wise / CSC) constraint-matrix values with their row
    indices -- so we read the file once and hand the arrays to the pure
    ``*_stats_from`` reducers. ``nz`` is pre-applied to the matrix arrays (only
    nonzero entries are retained) since every consumer wants that."""

    __slots__ = ("num_row", "num_col", "cost_abs", "val", "row")

    def __init__(self, path):
        highs = hspy.Highs()
        highs.setOptionValue("output_flag", "false")
        hh.read_model(highs, path)
        lp = highs.getLp()
        self.num_row = lp.num_row_
        self.num_col = lp.num_col_

        c = np.abs(np.asarray(lp.col_cost_))
        self.cost_abs = c[c != 0]

        am = lp.a_matrix_
        val = np.abs(np.asarray(am.value_))
        # a_matrix_ is column-wise (CSC): index_ holds the ROW of each nonzero.
        row = np.asarray(am.index_)
        nz = val > 0
        self.val = val[nz]
        self.row = row[nz]


def cost_stats_from(arr):
    """(max|c|, median|c|, ratio, max|A|) from an :class:`InstanceArrays`.
    Stats are over NONZERO coefficients."""
    c = arr.cost_abs
    cost_max = float(c.max()) if c.size else 0.0
    cost_med = float(np.median(c)) if c.size else 0.0
    ratio = (cost_max / cost_med) if cost_med > 0 else 0.0
    a_max = float(arr.val.max()) if arr.val.size else 0.0
    return cost_max, cost_med, ratio, a_max


def cost_stats(path):
    """Return (max|c|, median|c|, ratio, max|A|) for the original (un-relaxed)
    model at ``path``. Stats are over NONZERO coefficients."""
    return cost_stats_from(InstanceArrays(path))


def is_bigm_cost(
    path, ratio=DEFAULT_RATIO, abs_thresh=DEFAULT_ABS, med_floor=DEFAULT_MED_FLOOR
):
    """True if the instance at ``path`` has a big-M / penalty COST structure:
    a wide dynamic range (max|c|/median|c| > ``ratio``) together with a
    significant penalty -- either max|c| > ``abs_thresh`` (absolutely large,
    e.g. glass4) OR median|c| < ``med_floor`` (the penalty dwarfs a tiny bulk of
    costs, e.g. mas74/mas76)."""
    cost_max, cost_med, cost_ratio, _a_max = cost_stats(path)
    return cost_ratio > ratio and (cost_max > abs_thresh or cost_med < med_floor)


def matrix_stats_from(arr, wide_row=DEFAULT_WIDE_ROW):
    """(median_row_spread, max_row_spread, wide_row_frac) from an
    :class:`InstanceArrays`. Row spread is max|A_row| / min|A_row| over a row's
    NONZERO entries; only rows with at least two nonzeros contribute (a
    single-entry row has no spread). ``wide_row_frac`` is the fraction of
    contributing rows whose spread exceeds ``wide_row``."""
    nr = arr.num_row
    row, val = arr.row, arr.val

    # Reduce per-row max/min/count by scattering over the row indices, without
    # materialising the matrix or looping in Python.
    row_max = np.zeros(nr)
    row_min = np.full(nr, np.inf)
    row_cnt = np.zeros(nr, dtype=np.int64)
    np.maximum.at(row_max, row, val)
    np.minimum.at(row_min, row, val)
    np.add.at(row_cnt, row, 1)

    has_spread = (row_cnt >= 2) & (row_min > 0)
    spreads = row_max[has_spread] / row_min[has_spread]
    if spreads.size == 0:
        return 1.0, 1.0, 0.0
    return (
        float(np.median(spreads)),
        float(spreads.max()),
        float(np.mean(spreads > wide_row)),
    )


def matrix_stats(path, wide_row=DEFAULT_WIDE_ROW):
    """Return (median_row_spread, max_row_spread, wide_row_frac) for the model at
    ``path``. See :func:`matrix_stats_from` for the definitions."""
    return matrix_stats_from(InstanceArrays(path), wide_row=wide_row)


def is_bigm_matrix(
    path,
    row_spread=DEFAULT_ROW_SPREAD,
    row_frac=DEFAULT_ROW_FRAC,
    wide_row=DEFAULT_WIDE_ROW,
):
    """True if the instance at ``path`` has a big-M / penalty MATRIX structure:
    the median per-row coefficient spread exceeds ``row_spread`` AND at least
    ``row_frac`` of rows are individually wide (spread > ``wide_row``). The
    median + fraction gate demands a pervasive within-row skew (e.g.
    sp150x300d's [1,...,1,3050] rows) rather than a single outlier row that Ruiz
    scaling would absorb."""
    med_spread, _max_spread, wide_frac = matrix_stats(path, wide_row=wide_row)
    return med_spread > row_spread and wide_frac >= row_frac


def column_stats_from(arr, col_row_abs=DEFAULT_COL_ROW_ABS):
    """(n_dense_big_rows, max_row_abs, max_dense_big_density) from an
    :class:`InstanceArrays`. A "dense big-M row" is a row whose largest ABSOLUTE
    nonzero exceeds ``col_row_abs``; its "density" is the fraction of all columns
    its nonzeros touch. ``n_dense_big_rows`` counts such rows;
    ``max_dense_big_density`` is the largest density among big rows (0.0 if none).
    This is the root-cause signal for column big-M: a big-M coefficient shared
    across most columns' support (see :func:`is_bigm_column`)."""
    nr = arr.num_row
    nc = arr.num_col
    row, val = arr.row, arr.val

    row_max = np.zeros(nr)
    row_cnt = np.zeros(nr, dtype=np.int64)
    np.maximum.at(row_max, row, val)
    np.add.at(row_cnt, row, 1)

    max_row_abs = float(row_max.max()) if nr else 0.0
    row_density = row_cnt / nc if nc else np.zeros(nr)
    big = row_max > col_row_abs
    max_big_density = float(row_density[big].max()) if big.any() else 0.0
    return int(big.sum()), max_row_abs, max_big_density


def column_stats(path, col_row_abs=DEFAULT_COL_ROW_ABS):
    """Return (n_dense_big_rows, max_row_abs, max_dense_big_density) for the model
    at ``path``. See :func:`column_stats_from` for the definitions."""
    return column_stats_from(InstanceArrays(path), col_row_abs=col_row_abs)


def is_bigm_column(
    path,
    col_row_abs=DEFAULT_COL_ROW_ABS,
    col_row_density=DEFAULT_COL_ROW_DENSITY,
):
    """True if the instance at ``path`` has a big-M / penalty COLUMN structure:
    at least one "dense big-M row" -- a row whose largest absolute coefficient
    exceeds ``col_row_abs`` AND whose nonzeros touch at least ``col_row_density``
    of all columns. Such a row threads a huge coefficient through most columns'
    support, so the per-column spread explodes and Ruiz scaling cannot flatten it
    (e.g. germanrr's dense `sum a_j x_j = 0` value-balance equality). Keying on
    absolute magnitude + density, not the raw per-column spread ratio, avoids the
    tiny-min-entry false positives (ran14x18-disj-8, supportcase42)."""
    _n_big, _max_abs, max_big_density = column_stats(path, col_row_abs=col_row_abs)
    return max_big_density >= col_row_density


def main():
    args = parse_args()
    paths = sorted(glob.glob(os.path.join(args.data_dir, "*.mps")))
    if not paths:
        print(f"No .mps files in {args.data_dir}.")
        return

    print(
        f"Scanning {len(paths)} instance(s) in {args.data_dir}\n"
        f"  COST flag:   max|c|/median|c| > {args.ratio:g} AND "
        f"[max|c| > {args.abs:g} OR median|c| < {args.med_floor:g}]\n"
        f"  MATRIX flag: median row spread > {args.row_spread:g} AND "
        f">={args.row_frac:g} of rows wide (spread > {args.wide_row:g})\n"
        f"  COLUMN flag: a dense big-M row exists (max|A_row| > {args.col_row_abs:g} "
        f"AND touches >= {args.col_row_density:g} of columns)\n"
    )

    cost_flagged = []
    matrix_flagged = []
    column_flagged = []
    errors = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            arr = InstanceArrays(path)  # read the MPS once, run all three checks
            cost_max, cost_med, ratio, a_max = cost_stats_from(arr)
            med_spread, max_spread, wide_frac = matrix_stats_from(arr, args.wide_row)
            n_big, max_row_abs, max_big_density = column_stats_from(
                arr, args.col_row_abs
            )
        except Exception as exc:
            errors.append((name, repr(exc)))
            continue
        if ratio > args.ratio and (cost_max > args.abs or cost_med < args.med_floor):
            cost_flagged.append((name, cost_max, cost_med, ratio, a_max))
        if med_spread > args.row_spread and wide_frac >= args.row_frac:
            matrix_flagged.append((name, med_spread, max_spread, wide_frac))
        if max_big_density >= args.col_row_density:
            column_flagged.append((name, n_big, max_row_abs, max_big_density))

    cost_flagged.sort(key=lambda r: r[3], reverse=True)
    matrix_flagged.sort(key=lambda r: r[1], reverse=True)
    column_flagged.sort(key=lambda r: r[2], reverse=True)

    print(
        f"{'instance':28s} {'max|c|':>10s} {'med|c|':>10s} {'c-ratio':>10s} {'max|A|':>10s}"
    )
    print("-" * 72)
    for name, cmax, cmed, ratio, amax in cost_flagged:
        print(f"{name:28s} {cmax:10.2e} {cmed:10.2e} {ratio:10.2e} {amax:10.2e}")
    print(f"\n=== {len(cost_flagged)} COST big-M instance(s) ===")
    print("  " + "  ".join(name for name, *_ in cost_flagged))

    print(
        f"\n{'instance':28s} {'med-spread':>11s} {'max-spread':>11s} {'wide-frac':>10s}"
    )
    print("-" * 72)
    for name, med_spread, max_spread, wide_frac in matrix_flagged:
        print(f"{name:28s} {med_spread:11.2e} {max_spread:11.2e} {wide_frac:10.2f}")
    print(f"\n=== {len(matrix_flagged)} MATRIX big-M instance(s) ===")
    print("  " + "  ".join(name for name, *_ in matrix_flagged))

    print(
        f"\n{'instance':28s} {'n-big-rows':>11s} {'max|A_row|':>11s} {'big-density':>12s}"
    )
    print("-" * 72)
    for name, n_big, max_row_abs, max_big_density in column_flagged:
        print(f"{name:28s} {n_big:11d} {max_row_abs:11.2e} {max_big_density:12.2f}")
    print(f"\n=== {len(column_flagged)} COLUMN big-M instance(s) ===")
    print("  " + "  ".join(name for name, *_ in column_flagged))

    if errors:
        print(f"\n{len(errors)} file(s) failed to read:")
        for name, err in errors:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()

# %%
