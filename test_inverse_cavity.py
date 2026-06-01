"""
test_inverse_cavity.py
======================
Five test cases for the inverse cavity localization solver.

Each test:
  1. Picks a known cavity center.
  2. Generates a synthetic measurement via the forward model.
  3. Calls GroupSolutionCP1 (or the lower-level invert_cavity_location)
     to recover the center.
  4. Asserts the estimated center is within the required tolerance.

Run with:
    pytest test_inverse_cavity.py -v
or directly:
    python test_inverse_cavity.py
"""

from __future__ import annotations

import warnings
import math
import time
import pytest
import numpy as np

# Suppress noisy PorePy/PyGeon warnings during tests
warnings.filterwarnings("ignore")

from inverse_cavity_v2 import (
    GroupSolutionCP1,
    generate_synthetic_measurement,
    LX, LY, AXES, MARGIN_X, MARGIN_Y,
)

# ---------------------------------------------------------------------------
# Tolerance used for all spatial assertions  [m]
# ---------------------------------------------------------------------------
TOL = 0.12          # 12 cm – achievable with the two-level grid strategy


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run_case(
    true_center: tuple[float, float],
    label: str,
    tol: float = TOL,
    verbose: bool = True,
) -> dict:
    """
    Full test pipeline: forward → inverse → assert.

    Returns a result dict for reporting.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"TEST CASE: {label}")
        print(f"  True center : {true_center}")

    t0 = time.perf_counter()
    measurement = generate_synthetic_measurement(true_center)
    t_fwd = time.perf_counter() - t0

    if verbose:
        print(f"  Measurement : ux={measurement[0]:.6e}, uy={measurement[1]:.6e}")
        print(f"  Forward wall-time: {t_fwd:.1f}s")

    t1 = time.perf_counter()
    estimated = GroupSolutionCP1(measurement)
    t_inv = time.perf_counter() - t1

    error = math.dist(estimated, true_center)

    if verbose:
        print(f"  Estimated   : {estimated}")
        print(f"  Error       : {error:.4f} m  (tol={tol} m)")
        print(f"  Inverse wall-time: {t_inv:.1f}s")
        status = "✓ PASS" if error <= tol else "✗ FAIL"
        print(f"  Status: {status}")

    return {
        "label": label,
        "true_center": true_center,
        "measurement": measurement,
        "estimated": estimated,
        "error": error,
        "t_fwd": t_fwd,
        "t_inv": t_inv,
        "passed": error <= tol,
    }


# ---------------------------------------------------------------------------
# Test cases definitions
# ---------------------------------------------------------------------------

# Five representative positions spread across the domain interior
TEST_CASES = [
    # (label, true_center)
    ("TC1 – Centre of domain",       (1.00, 0.50)),
    ("TC2 – Left layer, lower half",  (0.50, 0.25)),
    ("TC3 – Right layer, upper half", (1.50, 0.75)),
    ("TC4 – Near fault line, mid-y",  (0.70, 0.55)),
    ("TC5 – Right, low",              (1.60, 0.25)),
]


# ---------------------------------------------------------------------------
# pytest test functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,center", TEST_CASES)
def test_case(label: str, center: tuple[float, float]):
    """Parametrised test: recover cavity center within tolerance."""
    result = _run_case(center, label, tol=TOL, verbose=False)
    assert result["passed"], (
        f"{label}: estimated {result['estimated']} is "
        f"{result['error']:.4f} m from true {center} (tol={TOL} m)"
    )


# ---------------------------------------------------------------------------
# Standalone runner (no pytest dependency)
# ---------------------------------------------------------------------------

def run_all_tests() -> None:
    """Execute all five test cases and print a summary table."""
    print("\n" + "=" * 60)
    print("INVERSE CAVITY – TEST SUITE")
    print("=" * 60)

    results = []
    for label, center in TEST_CASES:
        result = _run_case(center, label, tol=TOL, verbose=True)
        results.append(result)

    # ---------- Summary ----------
    print("\n" + "=" * 60)
    print(f"{'Case':<35} {'Error [m]':>10} {'Time inv [s]':>13} {'Status':>7}")
    print("-" * 60)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['label']:<35} {r['error']:>10.4f} {r['t_inv']:>13.1f} {status:>7}")
    print("=" * 60)

    n_pass = sum(r["passed"] for r in results)
    print(f"Results: {n_pass}/{len(results)} test cases passed (tol={TOL} m)")


import multiprocessing as mp

if __name__ == "__main__":
    # Forza Python a creare processi completamente puliti 
    # per evitare che Gmsh vada in tilt
    mp.set_start_method('spawn', force=True)
    run_all_tests()