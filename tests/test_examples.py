"""Run the self-contained examples end to end.

Each example runs in its own process, because examples call configure_jax with
their own precision and that setting is global. The MIPLIB examples are not
run: they need instances downloaded into data/. The RAG example is not run
either: it downloads a dataset and embeds it with fastembed, which takes minutes.
Nor is the monitor network example, which downloads EPA's data and takes
minutes.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def has(module):
    return importlib.util.find_spec(module) is not None


@pytest.mark.parametrize(
    "script, requires",
    [
        ("lp/intro_example.py", []),
        ("lp/cvxpy_transport.py", ["cvxpy"]),
        ("cp/isotonic_regression.py", ["matplotlib"]),
        ("cp/linear_regression.py", ["matplotlib", "sklearn"]),
        ("cp/cvxpy_lasso.py", ["cvxpy"]),
        ("cp/cvxpy_channel_capacity.py", ["cvxpy"]),
        ("cp/cvxpy_robust_kalman.py", ["cvxpy", "matplotlib"]),
        ("conic/cvxpy_portfolio.py", ["cvxpy"]),
        ("conic/cvxpy_maxcut.py", ["cvxpy"]),
        ("conic/cvxpy_kelly.py", ["cvxpy"]),
    ],
)
def test_example_runs(script, requires):
    missing = [m for m in requires if not has(m)]
    if missing:
        pytest.skip(f"needs {', '.join(missing)}")
    env = dict(os.environ, MPLBACKEND="Agg")
    env.setdefault("JAX_PLATFORMS", "cpu")
    proc = subprocess.run(
        [sys.executable, str(EXAMPLES / script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
