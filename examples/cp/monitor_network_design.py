# %% [markdown]
# # Designing an Air-Quality Monitoring Network
# The US EPA runs about a thousand monitors that measure fine particulate
# matter (PM2.5) every day. Every five years each state must assess its
# network: whether it still meets its objectives, and whether new monitors are
# needed or existing ones could be discontinued (40 CFR 58.10(d)). Behind that
# is a question of optimal experimental design:
#
# > Given a budget of `k` monitors, which `k` sites best predict the daily
# > PM2.5 at the places that have no monitor?
#
# This example answers it with EPA's open daily data. It fits a statistical
# model to 2023, chooses networks with a convex relaxation solved by
# `jaddle.jaddle_convex`, and checks the chosen networks on 2024, a year the
# model has not seen. Along the way it uses the solver's other tools:
#
# - The relaxation's optimum is a **lower bound** on the error of *every*
#   possible network, so a rounded network comes with a certificate of how far
#   from the best it can be.
# - `jc.make_solver` under `vmap` solves the design problem for a whole range
#   of budgets at once.
# - The budget constraint's dual is the **marginal value of one more monitor**,
#   and `jc.make_optimal_value` differentiates the optimal design's error with
#   respect to every site's noise level in one call.
#
# The data, about 20 MB, is downloaded from EPA's AirData site on the first run
# and cached in `data/`.

# %%
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import csv
import datetime
import io
import time
import urllib.request
import zipfile
from collections import defaultdict
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import jaddle.jaddle_convex as jc
import jaddle.jaddle_optimisers as jo

jo.configure_jax("float64")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TRAIN_YEAR, TEST_YEAR = 2023, 2024
MIN_COVERAGE = 0.85  # fraction of days a site must report, in both years
BUDGET = 50  # monitors in the network studied in detail

# %% [markdown]
# ## The data
# EPA publishes one file per year of daily PM2.5 averages (parameter 88101).
# We keep the 24-hour values used for the air-quality standard, drop the
# rows that EPA marks as "Excluded" duplicates of exceptional events
# (wildfires, fireworks; the "Included" versions are kept), and average
# collocated instruments at the same site. Alaska, Hawaii and the
# territories are left out, and so are sites that report on fewer than 85%
# of days in either year. Many monitors sample only every third or sixth
# day.


# %%
def load_year(year):
    """Daily PM2.5 (µg/m³) for one year, as {site: (lat, lon, name)} and a
    days × sites array with NaN where a site did not report."""
    path = DATA_DIR / f"daily_88101_{year}.zip"
    if not path.exists():
        url = f"https://aqs.epa.gov/aqsweb/airdata/daily_88101_{year}.zip"
        print(f"Downloading {url}")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    archive = zipfile.ZipFile(path)
    rows = csv.reader(
        io.TextIOWrapper(archive.open(archive.namelist()[0]), encoding="utf-8")
    )
    column = {name: i for i, name in enumerate(next(rows))}
    outside_contiguous_us = {"02", "15", "66", "72", "78", "80", "CC"}

    readings, info = defaultdict(list), {}
    for row in rows:
        if (
            row[column["Pollutant Standard"]] != "PM25 24-hour 2012"
            or row[column["Event Type"]] == "Excluded"
            or row[column["State Code"]] in outside_contiguous_us
        ):
            continue
        site = "-".join(row[:3])
        readings[site, row[column["Date Local"]]].append(
            float(row[column["Arithmetic Mean"]])
        )
        info[site] = (
            float(row[column["Latitude"]]),
            float(row[column["Longitude"]]),
            f"{row[column['Local Site Name']] or row[column['City Name']]}, "
            f"{row[column['State Name']]}",
        )

    start = datetime.date(year, 1, 1)
    days = (datetime.date(year + 1, 1, 1) - start).days
    index = {site: i for i, site in enumerate(info)}
    values = np.full((days, len(info)), np.nan)
    for (site, date), v in readings.items():
        values[(datetime.date.fromisoformat(date) - start).days, index[site]] = np.mean(
            v
        )
    return info, list(index), values


info, train_sites, train_raw = load_year(TRAIN_YEAR)
test_info, test_sites, test_raw = load_year(TEST_YEAR)


def well_covered(sites, values):
    coverage = np.mean(~np.isnan(values), axis=0)
    return {s for s, c in zip(sites, coverage) if c >= MIN_COVERAGE}


sites = sorted(
    well_covered(train_sites, train_raw) & well_covered(test_sites, test_raw)
)
train_pm = train_raw[:, [train_sites.index(s) for s in sites]]
test_pm = test_raw[:, [test_sites.index(s) for s in sites]]
lat, lon, names = (np.array(x) for x in zip(*(info[s] for s in sites)))
m = len(sites)
print(
    f"{len(train_sites)} sites in {TRAIN_YEAR}, {len(test_sites)} in {TEST_YEAR}; "
    f"{m} report on at least {MIN_COVERAGE:.0%} of days in both"
)

# %% [markdown]
# ## Candidates and targets
# To have something to score networks against, a random third of the sites
# become **targets**: places where we want to know the PM2.5 but will not
# put a monitor. Their readings are used only to measure prediction error.
# The other two-thirds are **candidate** monitor sites. Every network is
# scored on the same targets, so networks of different sizes and designs
# can be compared fairly.

# %%
rng = np.random.default_rng(0)
target = np.zeros(m, bool)
target[rng.choice(m, m // 3, replace=False)] = True
candidate = ~target
m_candidates = int(candidate.sum())
print(f"{m_candidates} candidate sites, {int(target.sum())} targets")

# %% [markdown]
# ## A statistical model of PM2.5
# PM2.5 is skewed: a quiet day is a few µg/m³, and a wildfire smoke day can
# be over 100 (June 2023 brought Canadian smoke across the Northeast). We
# model `y = log(1 + PM2.5)`, with the few small negative readings, which are
# instrument noise, set to zero.
#
# On day `t`, site `i` reads
#
# ```
# y[t, i] = μᵢ + wᵢᵀ f[t] + εᵢ[t],      f[t] ~ N(0, I_p),   εᵢ[t] ~ N(0, dᵢ)
# ```
#
# `f[t]` holds `p` regional factors (weather systems, smoke plumes, the
# seasons), and `wᵢ` says how site `i` responds to them. `εᵢ` is local
# variation the factors miss, and measurement error. This is probabilistic
# PCA: `W` comes from the top `p` eigenvectors of the 2023 covariance
# (missing days filled with the site mean), and `dᵢ` is the variance left
# over at each site.
#
# Given readings from a set of sites on one day, the posterior mean of `f`
# predicts every other site. We pick `p` on 2023 alone, so 2024 stays a
# clean test. Fit on odd weeks, then predict the targets in the even weeks
# from 50 random candidate sites.


# %%
def to_log(pm):
    return np.log1p(np.maximum(pm, 0.0))


train_y, test_y = to_log(train_pm), to_log(test_pm)


def fit_factor_model(y, p):
    mu = np.nanmean(y, axis=0)
    anomalies = np.nan_to_num(y - mu)
    covariance = anomalies.T @ anomalies / len(y)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    top = np.argsort(eigenvalues)[::-1][:p]
    W = eigenvectors[:, top] * np.sqrt(eigenvalues[top])
    d = np.maximum(
        np.diag(covariance) - np.sum(W**2, axis=1), 0.05 * np.diag(covariance)
    )
    return mu, W, d


@jax.jit
def predict(selected, mu, W, d, y):
    """Posterior-mean prediction of every site on every day of `y`, from the
    selected sites that reported that day."""
    p = W.shape[1]
    G = W / jnp.sqrt(d)[:, None]
    observed = selected & ~jnp.isnan(y)
    residual = jnp.where(observed, (y - mu) / jnp.sqrt(d), 0.0)
    # Each day's precision I + Σ gᵢgᵢᵀ over its observed sites, as one matmul.
    outer = (G[:, :, None] * G[:, None, :]).reshape(len(d), p * p)
    P = jnp.eye(p) + (observed @ outer).reshape(-1, p, p)
    f = jnp.linalg.solve(P, (residual @ G)[..., None])[..., 0]
    return mu + f @ W.T


def errors(selected, model, y):
    """RMS error at the targets, in log units and in µg/m³."""
    prediction = np.asarray(predict(selected, *model, y))
    scored = target & ~np.isnan(y)
    log_error = (prediction - y)[scored]
    ug_error = (np.expm1(prediction) - np.expm1(y))[scored]
    return np.sqrt(np.mean(log_error**2)), np.sqrt(np.mean(ug_error**2))


def random_network(k):
    return np.isin(
        np.arange(m), rng.choice(np.flatnonzero(candidate), k, replace=False)
    )


week = np.arange(len(train_y)) // 7
random_networks = [random_network(50) for _ in range(5)]
print("Factors  log RMSE on held-out 2023 weeks")
for p_try in (5, 10, 20, 30, 40, 60):
    model = fit_factor_model(train_y[week % 2 == 1], p_try)
    score = np.mean(
        [errors(s, model, train_y[week % 2 == 0])[0] for s in random_networks]
    )
    print(f"{p_try:7d}  {score:.4f}")

# %% [markdown]
# The error is lowest at about 20 factors, so we use `p = 20` and refit on
# the whole of 2023. Everything below is fitted to 2023 only.

# %%
p = 20
mu, W, d = fit_factor_model(train_y, p)
model = (mu, W, d)
C = W[target].T @ W[target]
print(
    f"{p} factors explain {np.sum(W**2) / (np.sum(W**2) + d.sum()):.0%} of the variance"
)

# %% [markdown]
# ## The design problem
# Let `zᵢ = 1` if candidate site `i` gets a monitor and `0` if not. From the
# monitored sites, the posterior over the factors has precision
#
# ```
# P(z) = I + Σᵢ zᵢ wᵢ wᵢᵀ / dᵢ
# ```
#
# and the posterior variance of the factor part of the prediction at site
# `j` is `wⱼᵀ P(z)⁻¹ wⱼ`. Averaging that over the targets `T` gives the
# **I-optimal** criterion
#
# ```
# φ(z) = Σⱼ∈T wⱼᵀ P(z)⁻¹ wⱼ / Σⱼ∈T ‖wⱼ‖² = tr(C P(z)⁻¹) / tr(C),     C = W_Tᵀ W_T,
# ```
#
# the share of the regional variance at the targets that the network leaves
# unexplained: 1 with no monitors, falling towards 0 as monitors are added.
# (The targets' local variances `dⱼ` add a constant that no network can
# remove.)
#
# Choosing `k` sites is a combinatorial problem. Relaxing `zᵢ ∈ {0, 1}` to
# `0 ≤ zᵢ ≤ 1` makes it convex, since `P` is affine in `z` and `tr(C P⁻¹)` is
# convex on positive definite matrices:
#
# ```
# minimise    φ(z)
# subject to  Σᵢ zᵢ = k,   0 ≤ zᵢ ≤ 1 for candidates,   zⱼ = 0 for targets
# ```
#
# That is a `JaddleCP` with one equality row and box bounds; the targets
# simply get an upper bound of 0. `P(z) ⪰ I` on
# the whole box, so every iterate is in the objective's domain, and its
# gradient comes from `jax.grad`. Solver tolerances are absolute, so the
# objective is divided by its value at the uniform design (`k/m` at every
# candidate), which
# keeps it of order one whatever the budget. Every quantity a later section
# differentiates (`k`, that scale, the noise variances `d`) is a parameter of
# `build_cp`.

# %%
W_j, C_j = jnp.asarray(W), jnp.asarray(C)
candidate_j = jnp.asarray(candidate, dtype=float)


def precision(z, d):
    G = W_j / jnp.sqrt(d)[:, None]
    return jnp.eye(p) + G.T @ (z[:, None] * G)


def i_criterion(z, d):
    return jnp.trace(jnp.linalg.solve(precision(z, d), C_j)) / jnp.trace(C_j)


def uniform_value(k, d):
    return i_criterion(candidate_j * k / m_candidates, d)


def build_cp(params):
    k, scale, d = params
    return jc.JaddleCP(
        num_variables=m,
        objective=lambda z: i_criterion(z, d) / scale,
        constraints_eq=lambda z: jnp.array([jnp.sum(z) - k]),
        constraints_ineq=lambda z: jnp.zeros(0),
        lower_bounds=jnp.zeros(m),
        upper_bounds=candidate_j,
    )


tolerances = dict(
    primal_grad_norm_tolerance=1e-6,
    primal_feasibility_tolerance=1e-6,
    complementarity_slack_tolerance=1e-6,
)
d_j = jnp.asarray(d)
scale = uniform_value(BUDGET, d_j)
start = time.perf_counter()
result = jc.solve(build_cp((float(BUDGET), scale, d_j)), **tolerances)
z_relaxed = np.asarray(result["solution"].primal)
bound = float(i_criterion(result["solution"].primal, d_j))
print(
    f"Converged: {result['converged']} after {result['epochs']} epochs "
    f"({time.perf_counter() - start:.1f} s including compilation)"
)
print(
    f"Σz = {z_relaxed.sum():.6f}, z in [{z_relaxed.min():.2e}, {z_relaxed.max():.6f}]"
)
print(
    f"{np.sum(z_relaxed > 1 - 1e-4)} sites at 1, "
    f"{np.sum((z_relaxed > 1e-4) & (z_relaxed < 1 - 1e-4))} fractional, "
    f"the other candidates at 0"
)
print(
    f"Relaxed optimum: φ = {bound:.5f}, a lower bound for every {BUDGET}-monitor network"
)

# %% [markdown]
# The optimality conditions are easy to read off. With `ν` the dual of
# `Σz = k`, the gradient `∂φ/∂zᵢ` equals `−ν` wherever `zᵢ` is fractional,
# is at least `−ν` where `zᵢ = 0` (another monitor there would help less
# than the budget is worth) and at most `−ν` where `zᵢ = 1`.

# %%
gradient = np.asarray(jax.grad(i_criterion)(result["solution"].primal, d_j))
nu = float(result["solution"].dual_eq[0]) * float(scale)
fractional = (z_relaxed > 1e-4) & (z_relaxed < 1 - 1e-4)
print(f"ν = {nu:.3e}")
print(f"fractional: max |∂φ/∂z + ν| = {np.max(np.abs(gradient[fractional] + nu)):.1e}")
print(
    f"at 0:       min (∂φ/∂z + ν) = {np.min(gradient[candidate & (z_relaxed <= 1e-4)] + nu):.1e}"
)
print(
    f"at 1:       max (∂φ/∂z + ν) = {np.max(gradient[z_relaxed >= 1 - 1e-4] + nu):.1e}"
)

# %% [markdown]
# ## From the relaxation to a network
# The relaxed design spreads some weight over more than `k` sites. To get a
# network, we keep the `k` sites with the largest weights, then improve that
# choice by exchanges: swap a kept site for a dropped one while that lowers
# `φ` (Fedorov's exchange algorithm).
#
# Swapping site `i` for site `j` changes `P` by `gⱼgⱼᵀ − gᵢgᵢᵀ`, with
# `gᵢ = wᵢ/√dᵢ`. That is a rank-two update, so the Woodbury identity gives
# the new criterion for *every* pair `(i, j)` at once from `M = P⁻¹`. Only
# the m × m matrices `A = G M Gᵀ` and `B = G M C M Gᵀ` are needed:
#
# ```
# tr(C P'⁻¹) = tr(C M) − [(Aᵢᵢ − 1) Bⱼⱼ − 2 Aᵢⱼ Bᵢⱼ + (1 + Aⱼⱼ) Bᵢᵢ]
#                        / [(1 + Aⱼⱼ)(Aᵢᵢ − 1) − Aᵢⱼ²]
# ```
#
# Adding `j` alone is the `i`-free case, `tr(C M) − Bⱼⱼ / (1 + Aⱼⱼ)`, which
# also gives a **greedy** network: add the best site, one at a time. Greedy
# selection is a strong, standard heuristic, but it comes with no
# certificate of its own.
#
# The same formulas, with determinants in place of traces, handle the
# **D-optimal** criterion `−log det P(z)`. That criterion maximises the
# information about the factors rather than the accuracy of the predictions,
# and we compare it below.

# %%
G = W / np.sqrt(d)[:, None]
trace_C = np.trace(C)


@partial(jax.jit, static_argnames="criterion")
def exchange_values(selected, criterion="I"):
    """The criterion (φ for "I", −log det P for "D") after each single
    addition (a vector) and each swap (a matrix, rows the site removed,
    columns the site added). Entries that are not valid moves are inf."""
    M = jnp.linalg.inv(precision(selected.astype(float), d_j))
    A = G @ M @ G.T
    a = jnp.diag(A)
    det_K = (1 + a[None, :]) * (a[:, None] - 1) - A**2
    if criterion == "I":
        B = G @ (M @ C @ M) @ G.T
        b = jnp.diag(B)
        tr_CM = jnp.trace(C @ M)
        add = (tr_CM - b / (1 + a)) / trace_C
        numerator = (
            (a[:, None] - 1) * b[None, :] - 2 * A * B + (1 + a[None, :]) * b[:, None]
        )
        swap = (tr_CM - numerator / det_K) / trace_C
    else:
        logdet = -jnp.linalg.slogdet(M)[1]
        add = -(logdet + jnp.log1p(a))
        swap = -(logdet + jnp.log(-det_K))
    addable = jnp.asarray(candidate) & ~selected
    valid = selected[:, None] & addable[None, :]
    return jnp.where(addable, add, jnp.inf), jnp.where(valid, swap, jnp.inf)


@partial(jax.jit, static_argnames="criterion")
def best_swap(selected, criterion="I"):
    swap = exchange_values(selected, criterion)[1]
    flat = jnp.argmin(swap)
    return flat // m, flat % m, swap.ravel()[flat]


def d_criterion(z, d):
    return -jnp.linalg.slogdet(precision(z, d))[1]


def value(selected, criterion="I"):
    f = i_criterion if criterion == "I" else d_criterion
    return float(f(jnp.asarray(selected, dtype=float), d_j))


def exchange(selected, criterion="I"):
    """Swap sites in and out until no single swap improves the criterion."""
    selected = np.array(selected)
    current = value(selected, criterion)
    for swaps in range(10 * m):
        i, j, new = (np.asarray(x) for x in best_swap(jnp.asarray(selected), criterion))
        if new >= current - 1e-12 * abs(current):
            return selected, swaps
        selected[i], selected[j], current = False, True, float(new)
    return selected, swaps


def greedy_path(k_max):
    """The order in which greedy selection adds sites."""
    selected, order = np.zeros(m, bool), []
    for _ in range(k_max):
        j = int(jnp.argmin(exchange_values(jnp.asarray(selected))[0]))
        selected[j] = True
        order.append(j)
    return np.array(order)


def top_k(z, k):
    selected = np.zeros(m, bool)
    selected[np.argsort(-z)[:k]] = True
    return selected


rounded = top_k(z_relaxed, BUDGET)
designed, swaps = exchange(rounded)
print(f"Swap formula check: {value(designed):.10f} (direct) vs ", end="")
check = designed.copy()
i, j = np.flatnonzero(check)[0], np.flatnonzero(candidate & ~check)[0]
swap_matrix = np.asarray(exchange_values(jnp.asarray(check))[1])
check[i], check[j] = False, True
print(f"{swap_matrix[i, j]:.10f} vs {value(check):.10f} (one swap)")
print(f"Top-{BUDGET} rounding:  φ = {value(rounded):.5f}")
print(
    f"After {swaps} swaps:   φ = {value(designed):.5f}, within "
    f"{value(designed) / bound - 1:.1%} of the bound"
)

# %% [markdown]
# The bound is the useful part. No network of 50 of the 470 candidates, out
# of about 10⁶⁸ choices, can leave less than the relaxed optimum unexplained,
# so the exchanged network is provably within about 1% of the best
# possible, under the model. Neither greedy selection nor exchanges alone
# can say that.
#
# ## Checking against 2024
# Now the networks meet data the model has not seen. For each network, we
# predict every target on every day of 2024 from that day's readings at the
# monitored sites, using the 2023 model. The networks compared are:
#
# - **No monitors**: each target's 2023 average.
# - **Random**: 20 random networks of `k` sites, averaged.
# - **Spread out**: farthest-point sampling on the map, which spreads
#   monitors as evenly over the country as possible. This is the intuitive
#   approach.
# - **Greedy**: greedy selection on `φ`.
# - **D-optimal**: the relaxation of `−log det P`, rounded and exchanged
#   the same way.
# - **I-optimal**: the network above.


# %%
def spread_out(k):
    xy = np.c_[lon * np.cos(np.deg2rad(lat)), lat]
    chosen = [
        int(
            np.argmin(
                np.where(
                    candidate, np.linalg.norm(xy - xy.mean(axis=0), axis=1), np.inf
                )
            )
        )
    ]
    distance = np.where(candidate, np.linalg.norm(xy - xy[chosen[0]], axis=1), -np.inf)
    for _ in range(k - 1):
        chosen.append(int(np.argmax(distance)))
        distance = np.minimum(distance, np.linalg.norm(xy - xy[chosen[-1]], axis=1))
        distance[target] = -np.inf
    return np.isin(np.arange(m), chosen)


def build_d_cp(params):
    k, d = params
    return jc.JaddleCP(
        num_variables=m,
        objective=lambda z: d_criterion(z, d),
        constraints_eq=lambda z: jnp.array([jnp.sum(z) - k]),
        constraints_ineq=lambda z: jnp.zeros(0),
        lower_bounds=jnp.zeros(m),
        upper_bounds=candidate_j,
    )


d_result = jc.solve(build_d_cp((float(BUDGET), d_j)), **tolerances)
print("D-optimal relaxation converged:", d_result["converged"])
d_designed, _ = exchange(top_k(np.asarray(d_result["solution"].primal), BUDGET), "D")
greedy_order = greedy_path(150)

networks = {
    "No monitors": np.zeros(m, bool),
    "Spread out": spread_out(BUDGET),
    "Greedy": np.isin(np.arange(m), greedy_order[:BUDGET]),
    "D-optimal": d_designed,
    "I-optimal": designed,
}
random_networks = [random_network(BUDGET) for _ in range(20)]
random_phi = np.mean([value(s) for s in random_networks])
random_errors = np.mean([errors(s, model, test_y) for s in random_networks], axis=0)

print(f"\n{BUDGET} monitors        φ (2023 model)   2024 RMSE: log    µg/m³")
for name, selected in networks.items():
    log_rmse, ug_rmse = errors(selected, model, test_y)
    print(f"{name:22s} {value(selected):10.4f}  {log_rmse:15.4f} {ug_rmse:8.2f}")
    if name == "No monitors":
        print(
            f"{'Random (mean of 20)':22s} {random_phi:10.4f}  {random_errors[0]:15.4f} {random_errors[1]:8.2f}"
        )
print(f"{'Relaxation bound':22s} {bound:10.4f}")

# %% [markdown]
# Any 50 monitors help a lot: the error at the targets falls from 5.2 µg/m³
# (their 2023 averages) to about 4.0 with random networks. The designed
# networks halve the model's unexplained variance compared with random
# ones, and that carries over to the unseen year: about 8% less error, 3.7
# µg/m³. The gain is smaller than the model's because much of each target's
# day-to-day variation is local (`dⱼ`), and no choice of monitors can
# predict it.
#
# Spreading monitors evenly over the map beats random choice on the model's
# criterion and in log units, but not in µg/m³. The likely reason is that
# sites cluster where people live. An even spread puts monitors in sparse,
# mostly clean areas, while the largest errors in µg/m³ come from polluted
# days in the clusters.
#
# The three designed networks are close. I-optimal is best on its own
# criterion, and on 2024 all three are within 0.02 µg/m³ of each other,
# which is within the noise. The relaxation does not beat a good heuristic
# by much here. What it adds is the certificate that nothing can do much
# better.

# %%
fig, ax = plt.subplots(figsize=(11, 6))
ax.scatter(
    lon[target], lat[target], marker="x", s=14, color="grey", label="Target sites"
)
points = ax.scatter(
    lon[candidate],
    lat[candidate],
    c=z_relaxed[candidate],
    cmap="viridis",
    s=14,
    vmin=0,
    vmax=1,
)
ax.scatter(
    lon[designed],
    lat[designed],
    s=90,
    facecolors="none",
    edgecolors="C3",
    linewidths=1.5,
    label=f"Chosen network ({BUDGET} sites)",
)
fig.colorbar(points, ax=ax, label="Relaxed weight zᵢ")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect(1 / np.cos(np.deg2rad(38)))
ax.set_title(f"{m} PM2.5 sites and the I-optimal {BUDGET}-monitor network")
ax.legend(loc="lower left")
plt.show()

# %% [markdown]
# ## Every budget at once
# How many monitors are enough? `jc.make_solver` turns the design problem
# into a pure JAX function of its parameters, and `jax.vmap` solves it for
# 16 budgets from 5 to 150 in one call. Each budget's network is then
# rounded and exchanged as above. Beyond 150 the optimum becomes very flat,
# since many networks are almost equally good, and the solver needs many
# more epochs to certify it.
#
# The dual `ν(k)` of `Σz = k` is the slope of the optimal value: by how
# much one more monitor would reduce `φ`.

# %%
budgets = np.unique(np.round(np.geomspace(5, 150, 16)))
scales = jax.vmap(uniform_value, in_axes=(0, None))(jnp.asarray(budgets), d_j)
sweep = jax.jit(
    jax.vmap(
        jc.make_solver(build_cp, max_epochs=1000, **tolerances), in_axes=((0, 0, None),)
    )
)
start = time.perf_counter()
curve = sweep((jnp.asarray(budgets), scales, d_j))
jax.block_until_ready(curve)
print(
    f"{len(budgets)} budgets in {time.perf_counter() - start:.1f} s, "
    f"converged: {int(curve.converged.sum())}/{len(budgets)}"
)
curve_bound = np.asarray(
    jax.vmap(i_criterion, in_axes=(0, None))(curve.solution.primal, d_j)
)
marginal = np.asarray(curve.solution.dual_eq[:, 0] * scales)

exchanged = [
    exchange(top_k(z, int(k)))[0]
    for z, k in zip(np.asarray(curve.solution.primal), budgets)
]
greedy = [np.isin(np.arange(m), greedy_order[: int(k)]) for k in budgets]
randoms = [[random_network(int(k)) for _ in range(10)] for k in budgets]
phi = {
    "I-optimal (exchanged)": [value(s) for s in exchanged],
    "Greedy": [value(s) for s in greedy],
    "Random (mean of 10)": [np.mean([value(s) for s in r]) for r in randoms],
}
rmse = {
    "I-optimal (exchanged)": [errors(s, model, test_y)[1] for s in exchanged],
    "Greedy": [errors(s, model, test_y)[1] for s in greedy],
    "Random (mean of 10)": [
        np.mean([errors(s, model, test_y)[1] for s in r]) for r in randoms
    ],
}
gap = np.array(phi["I-optimal (exchanged)"]) / curve_bound - 1
print("Budget   bound φ   exchanged φ   gap    greedy φ   ν (per monitor)")
for row in zip(
    budgets, curve_bound, phi["I-optimal (exchanged)"], gap, phi["Greedy"], marginal
):
    print("{:6.0f} {:9.4f} {:12.4f} {:7.1%} {:10.4f} {:12.2e}".format(*row))

# %%
fig, (ax_phi, ax_rmse, ax_value) = plt.subplots(1, 3, figsize=(17, 5))
ax_phi.plot(budgets, curve_bound, "k--", label="Relaxation (lower bound)")
for (label, values), style in zip(phi.items(), ("o-", "s-", "^-")):
    ax_phi.plot(budgets, values, style, ms=4, label=label)
ax_phi.set_xscale("log")
ax_phi.set_yscale("log")
ax_phi.set_xlabel("Monitors k")
ax_phi.set_ylabel("Unexplained regional variance φ")
ax_phi.set_title("Design criterion (2023 model)")
ax_phi.legend()

ax_rmse.axhline(
    errors(np.zeros(m, bool), model, test_y)[1],
    color="grey",
    ls=":",
    label="No monitors",
)
for (label, values), style in zip(rmse.items(), ("o-", "s-", "^-")):
    ax_rmse.plot(budgets, values, style, ms=4, label=label)
ax_rmse.set_xscale("log")
ax_rmse.set_xlabel("Monitors k")
ax_rmse.set_ylabel("RMSE at the targets (µg/m³)")
ax_rmse.set_title(f"Prediction error in {TEST_YEAR}")
ax_rmse.legend()

ax_value.plot(budgets, 100 * marginal, "o-", color="C3")
ax_value.set_xscale("log")
ax_value.set_yscale("log")
ax_value.set_xlabel("Monitors k")
ax_value.set_ylabel("Points of φ per extra monitor (%)")
ax_value.set_title("Marginal value of a monitor, ν(k)")
plt.tight_layout()
plt.show()

# %% [markdown]
# From about 24 monitors up, the exchanged network is within 5% of the
# bound, and within 0.1% at 150. Below that the bound is loose, 45% at
# `k = 5`. With fewer monitors than the model has factors (`p = 20`), the
# relaxation can spread small weights over many sites and learn a little
# about every factor, which no real network can do. There the certificate
# says little, though the exchanged network still beats greedy at every
# budget.
#
# On 2024 the designed networks beat random ones at every budget: about 35
# designed monitors predict the targets as well as 150 random ones. Greedy
# and the exchanged networks are close, with the exchanged ones ahead at
# most budgets below 30.
#
# Returns diminish fast. The marginal value of a monitor falls roughly as
# `k^-1.7`, so the 150th monitor is worth about a fiftieth of the 16th. A
# network assessment can use this curve to see where cutting monitors
# begins to cost real accuracy.
#
# ## Derivatives of the optimal design
# `jc.make_optimal_value(build_cp)` returns the optimal value as a
# differentiable function of the parameters. Its gradient comes from the
# envelope theorem, with no differentiation through the solver's iterations.
# One `jax.grad` call gives the derivative with respect to all three
# parameters at once:
#
# - with respect to `k`, the marginal value of a monitor, which should match
#   the dual `ν` and a finite difference;
# - with respect to all 705 noise variances `dᵢ`, how much the best network
#   would gain if site `i`'s local variation, which includes its measurement
#   error, were smaller. Collocating a second instrument, for example,
#   halves the measurement-error part.

# %%
optimal_value = jc.make_optimal_value(build_cp, **tolerances)
start = time.perf_counter()
d_value_dk, _, d_value_dd = jax.grad(optimal_value)((float(BUDGET), scale, d_j))
print(f"Gradient in {time.perf_counter() - start:.1f} s")
finite_difference = (
    optimal_value((BUDGET + 1.0, scale, d_j))
    - optimal_value((BUDGET - 1.0, scale, d_j))
) / 2
print(
    f"dφ*/dk: gradient {float(d_value_dk * scale):.4e}, "
    f"dual {-nu:.4e}, finite difference {float(finite_difference * scale):.4e}"
)

# How much φ* would fall if site i's dᵢ were 10% smaller.
gain = 0.1 * np.asarray(d_value_dd * scale) * d
print(
    "\nSites whose local noise costs the network most (fall in φ* for 10% less noise):"
)
for i in np.argsort(-gain)[:5]:
    print(f"  {names[i]:50s} z = {z_relaxed[i]:.2f}   {gain[i]:.2e}")
print(
    f"Sites with no derivative (targets, and candidates the design leaves out): {np.sum(np.abs(gain) < 1e-12)}"
)

# %% [markdown]
# Only the sites in the relaxed design have a nonzero derivative: a quieter
# instrument at a site the design leaves out does not change the design's
# value at first order. Among the chosen sites, the ranking points to where
# a second instrument would do the most good.
#
# ## Summary
# A convex relaxation of a combinatorial design problem, solved by Jaddle's
# convex solver, chose a 50-monitor network from 470 candidate sites. The
# network is provably within about 1% of the best possible under a model
# fitted to 2023. On 2024 it predicted PM2.5 at 235 held-out sites with 8%
# less error than random networks of the same size, and as well as greedy
# selection. One batched solve traced the whole trade-off between budget and
# accuracy, and its duals and gradients price both one more monitor and a
# better one.
