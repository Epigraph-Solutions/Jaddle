# %% [markdown]
# # Planning a Drone Swarm Through a Doorway with Jaddle's MIP Solver
# Four drones must swap sides of a room. A wall splits the room in two, and
# the only way through is a 3 m doorway: wide enough for two drones side by
# side, not for four. Each drone flies to the opposite corner in 12 seconds.
# They must not hit the wall or each other, and they should use as little
# fuel (total thrust) as possible.
#
# This is the mixed-integer trajectory planning of Richards and How, "Aircraft
# trajectory planning with collision avoidance using mixed integer linear
# programming" (2002). The dynamics, the thrust limits and the fuel are all
# linear. What makes it hard is the "or" in every avoidance constraint: a
# drone is clear of a box if it is to its left *or* right *or* below *or*
# above it. Each "or" becomes a choice of binary variable, so the plan is a
# mixed-integer LP. Nobody tells the drones which order to go through the
# door or which side to pass on. The branch and bound search decides.
#
# We write the model in CVXPY and solve it with `JaddleSolver`, which hands
# mixed-integer LPs to `jaddle_mip.solve`. That runs the search, solving many
# node LPs at once with Jaddle's first-order LP solver. Here it runs under a
# time limit, and we fly the best plan it has found by then.

# %%
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cvxpy as cp
import numpy as np
from jaddle.jaddle_cvxpy import JaddleSolver

# %% [markdown]
# ## The room, the drones and the time grid
# The room is 10 m × 10 m. The wall runs across the middle at
# `4.5 <= y <= 5.5`, with the doorway at `3.5 <= x <= 6.5`. Each drone is a
# disc of radius 0.35 m. It may accelerate at up to 1 m/s² and fly at up to
# 2 m/s along each axis.
#
# The plan has `T = 8` steps of 1.5 s. Each drone holds a constant
# acceleration over a step, so its path within a step is a parabola, and the
# animation below follows that exact motion.

# %%
size, radius = 10.0, 0.35
u_max, v_max = 1.0, 2.0
horizon, T = 12.0, 8
dt = horizon / T

starts = np.array([[2.0, 1.0], [8.0, 1.0], [8.0, 9.0], [2.0, 9.0]])
goals = starts[[2, 3, 0, 1]]  # each drone flies to the opposite corner
walls = [(0.0, 4.5, 3.5, 5.5), (6.5, 4.5, 10.0, 5.5)]  # (x0, y0, x1, y1)
n_drones = len(starts)
pairs = [(i, j) for i in range(n_drones) for j in range(i + 1, n_drones)]

# %% [markdown]
# ## Staying clear between time steps
# The constraints act at the 9 time points. A drone that is clear of a wall
# at two consecutive points can still clip a corner in between. So we use
# one binary for both ends of a step: if a drone is left of a wall at the
# start of a step *and* at its end, the straight chord between them is left
# of it too, because a half-plane is convex.
#
# The real path is a parabola, not the chord. With a constant acceleration
# `a`, it strays from the chord by at most `a dt² / 8` along each axis. So we
# grow each wall by the drone's radius plus that margin. We also ask any two
# drones to stay `2 (radius + margin)` apart along at least one axis. Then
# the continuous flight is safe, not only the sampled points. We check this
# after solving.

# %%
margin = u_max * dt**2 / 8
pad = radius + margin  # how far each wall is grown
separation = 2 * pad  # drone-to-drone, along x or y

# %% [markdown]
# ## Where each drone can be
# A binary is only needed where the choice is open. A drone starts at rest,
# so after time `τ` it has moved at most `reach(τ)` along each axis. It must
# also end at rest at its goal, so it is within `reach(horizon − τ)` of the
# goal. These boxes become position bounds. They also let us drop the
# binaries for any step where a drone cannot reach a wall, or two drones
# cannot meet. And they give each big-M its smallest valid value, which
# keeps the LP relaxation as tight as this formulation allows.


# %%
def reach(tau):
    """How far a drone starting at rest can move along one axis in time tau."""
    tau = np.asarray(tau, dtype=float)
    ramp = v_max / u_max  # time to reach top speed
    return np.where(
        tau <= ramp, u_max * tau**2 / 2, v_max * tau - v_max**2 / (2 * u_max)
    )


times = np.arange(T + 1) * dt
box_lo = np.maximum.reduce(
    [
        np.full((n_drones, T + 1, 2), radius),
        starts[:, None] - reach(times)[None, :, None],
        goals[:, None] - reach(horizon - times)[None, :, None],
    ]
)
box_hi = np.minimum.reduce(
    [
        np.full((n_drones, T + 1, 2), size - radius),
        starts[:, None] + reach(times)[None, :, None],
        goals[:, None] + reach(horizon - times)[None, :, None],
    ]
)

# %% [markdown]
# ## The "or" constraints
# Each avoidance constraint asks a point `q` (a drone's position, or the
# offset between two drones) to lie in one of four half-planes,
# `sign · q[axis] <= rhs`. For each step we keep the half-planes the
# position boxes leave possible. If one of them holds everywhere in the
# boxes, the step is safe and needs no binaries. Otherwise each half-plane
# gets a binary `z`, with `Σ z = 1`, and at both ends of the step
#
# ```
# sign · q[axis] <= rhs + M (1 − z)
# ```
#
# where `M` is the most `q` can break that half-plane by within its boxes.


# %%
def outside(x0, y0, x1, y1):
    """The four half-planes (axis, sign, rhs) outside a box: left of it,
    right of it, below it and above it."""
    return [(0, 1, x0), (0, -1, -x1), (1, 1, y0), (1, -1, -y1)]


def either_side(q, q_lo, q_hi, t, sides):
    """Keep q out of a box over step t, given boxes q_lo <= q <= q_hi at each
    time point. Returns the constraints and the number of binaries."""
    live = []
    for axis, sign, rhs in sides:
        ends = [sorted(sign * b[s][axis] for b in (q_lo, q_hi)) for s in (t, t + 1)]
        best = [lo - rhs for lo, _ in ends]
        worst = [hi - rhs for _, hi in ends]
        if max(worst) <= 0:
            return [], 0  # this side always holds: nothing to choose
        if max(best) <= 0:
            live.append((axis, sign, rhs, worst))
    z = cp.Variable(len(live), boolean=True)
    rows = [cp.sum(z) == 1]
    for k, (axis, sign, rhs, big_m) in enumerate(live):
        for s, M in zip((t, t + 1), big_m):
            if M > 0:
                rows.append(sign * q(s)[axis] <= rhs + M * (1 - z[k]))
    return rows, len(live)


# %% [markdown]
# ## The model
# Each drone has positions `p`, velocities `v`, accelerations `u` and fuel
# `f >= |u|`. The dynamics are exact for a constant acceleration over each
# step:
#
# ```
# p[t+1] = p[t] + dt v[t] + dt²/2 u[t]
# v[t+1] = v[t] + dt u[t]
# ```
#
# and the objective is the total fuel `Σ f`, the L1 norm of all the
# accelerations. `build_plan` builds the plan for any group of drones, so
# that we can also plan pairs of drones on their own (see below).
# `floors[(i, j)]`, when given, adds the row `fuel_i + fuel_j >= floor`.


# %%
def build_plan(drones):
    """The fuel-optimal plan for `drones`, a list of drone indices. Returns
    the problem, the lists of p, v and u variables, and the binary count."""
    p = {i: cp.Variable((T + 1, 2)) for i in drones}
    v = {i: cp.Variable((T + 1, 2)) for i in drones}
    u = {i: cp.Variable((T, 2)) for i in drones}
    f = {i: cp.Variable((T, 2)) for i in drones}
    constraints = []
    for i in drones:
        constraints += [
            p[i][0] == starts[i],
            p[i][T] == goals[i],
            v[i][0] == 0,
            v[i][T] == 0,
            p[i][1:] == p[i][:-1] + dt * v[i][:-1] + dt**2 / 2 * u[i],
            v[i][1:] == v[i][:-1] + dt * u[i],
            cp.abs(u[i]) <= f[i],
            f[i] <= u_max,
            cp.abs(v[i]) <= v_max,
            p[i] >= box_lo[i],
            p[i] <= box_hi[i],
        ]
    fuel = {i: cp.sum(f[i]) for i in drones}

    n_binaries = 0
    for x0, y0, x1, y1 in walls:
        sides = outside(x0 - pad, y0 - pad, x1 + pad, y1 + pad)
        for i in drones:
            for t in range(T):
                rows, k = either_side(lambda s: p[i][s], box_lo[i], box_hi[i], t, sides)
                constraints += rows
                n_binaries += k
    sides = outside(-separation, -separation, separation, separation)
    for i, j in pairs:
        if i not in p or j not in p:
            continue
        for t in range(T):
            rows, k = either_side(
                lambda s: p[i][s] - p[j][s],
                box_lo[i] - box_hi[j],
                box_hi[i] - box_lo[j],
                t,
                sides,
            )
            constraints += rows
            n_binaries += k

    problem = cp.Problem(cp.Minimize(sum(fuel.values())), constraints)
    variables = [[x[i] for i in drones] for x in (p, v, u)]
    return problem, variables, n_binaries


problem, (p, v, u), n_binaries = build_plan(list(range(n_drones)))
print(f"Swarm: {n_binaries} binaries, {len(problem.constraints)} constraints")

# %% [markdown]
# ## Solving the swarm
# Keyword arguments of `problem.solve` go to `jaddle_mip.solve`, here a time
# limit. The log shows the node count, the best bound, the incumbent (the
# best plan so far) and the gap between them. The letter in front of a line
# says what found that incumbent, for example `J` feasibility jump, `N`
# RINS, `C` crossover or `T` a node whose LP point was already integral.
#
# If the time runs out, CVXPY reports `user_limit`, warns that the solution
# may be inaccurate, and keeps the best plan found. That plan is feasible but
# may not be optimal. The gap is a certificate: no plan uses less fuel than
# the bound.

# %%
problem.solve(
    solver=JaddleSolver(),
    verbose=True,
    max_seconds=180,
    presolve=True,
    device=None,  # JAX's default device: the GPU when there is one
    log_every_seconds=10,
    exact_band="auto",
    batch_size=1,
    exact_strong_lookahead=8,
    tree_cut_every=100,
    cut_pool_size=1e5,
    tree_cut_rows=2000,
    cuts=False,
)
stats = problem.solver_stats.extra_stats
if p[0].value is None:
    raise RuntimeError(f"no plan found ({problem.status})")
print(f"\nStatus: {problem.status}, {stats['nodes']} nodes")
print(
    f"Fuel: {problem.value:.3f} (no plan uses less than {stats['bound']:.3f},"
    f" gap {stats['gap']:.1%})"
)

P = np.stack([x.value for x in p])
V = np.stack([x.value for x in v])
U = np.stack([x.value for x in u])

# %% [markdown]
# ## Checking the flight
# We sample the exact motion 50 times per step and check it against the
# real geometry: each disc stays out of the (ungrown) walls and inside the
# room, and no two discs overlap.

# %%
substeps = 50
tau = np.arange(substeps) / substeps * dt


def flight(P, V, U, tau):
    """Positions at offsets `tau` into every step: shape (drones, frames, 2)."""
    within = (
        P[:, :-1, None]
        + V[:, :-1, None] * tau[:, None]
        + U[:, :, None] * tau[:, None] ** 2 / 2
    )
    path = within.reshape(len(P), -1, 2)
    return np.concatenate([path, P[:, -1:]], axis=1)


path = flight(P, V, U, tau)
tolerance = 1e-6
for x0, y0, x1, y1 in walls:
    nearest = np.clip(path, [x0, y0], [x1, y1])
    assert np.linalg.norm(path - nearest, axis=-1).min() >= radius - tolerance
assert path.min() >= radius - tolerance and path.max() <= size - radius + tolerance
closest = min(np.linalg.norm(path[i] - path[j], axis=-1).min() for i, j in pairs)
assert closest >= 2 * radius - tolerance
print(
    f"Closest approach: {closest:.2f} m between centres"
    f" (discs touch at {2 * radius:.2f} m)"
)

# %% [markdown]
# ## Watching the flight
# Each drone is a disc with a fading trail, its plan is the faint dotted
# line, and its goal is the cross of the same colour. The grey walls are the
# real walls, and the dashed outlines are the grown walls the plan keeps the
# drone centres out of.

# %%
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

frames_per_step = 9
frames = flight(P, V, U, np.arange(frames_per_step) / frames_per_step * dt)
hold = 12  # extra frames on the final positions
trail = 2 * frames_per_step

fig, ax = plt.subplots(figsize=(5, 5.5))
fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.06)
ax.set_xlim(0, size)
ax.set_ylim(0, size)
ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])
for x0, y0, x1, y1 in walls:
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="0.3"))
    ax.add_patch(
        Rectangle(
            (x0 - pad, y0 - pad),
            x1 - x0 + 2 * pad,
            y1 - y0 + 2 * pad,
            fill=False,
            linestyle="--",
            edgecolor="0.6",
            linewidth=0.8,
        )
    )
colours = [f"C{i}" for i in range(n_drones)]
discs, trails = [], []
for i, colour in enumerate(colours):
    ax.plot(*frames[i].T, ":", color=colour, alpha=0.4)
    ax.plot(*goals[i], "x", color=colour, markersize=10, markeredgewidth=2)
    trails.append(ax.plot([], [], color=colour, linewidth=2.5, alpha=0.6)[0])
    discs.append(ax.add_patch(Circle(frames[i, 0], radius, color=colour, zorder=3)))
clock = ax.set_title("")
fig.text(
    0.5,
    0.02,
    f"Jaddle MIP: {n_binaries} binaries, {stats['nodes']} nodes,"
    f" {stats['seconds']:.0f} s, gap {stats['gap']:.0%}",
    ha="center",
    fontsize=9,
    color="0.35",
)


def draw(k):
    k = min(k, frames.shape[1] - 1)
    for i in range(n_drones):
        discs[i].center = frames[i, k]
        trails[i].set_data(*frames[i, max(0, k - trail) : k + 1].T)
    clock.set_text(f"t = {k * dt / frames_per_step:4.1f} s")
    return []


animation = FuncAnimation(fig, draw, frames=frames.shape[1] + hold)
# Saved next to where the script runs; matplotlib writes GIFs with Pillow.
animation.save("drone_swarm.gif", writer=PillowWriter(fps=frames_per_step / dt), dpi=80)
plt.show()

# %%
