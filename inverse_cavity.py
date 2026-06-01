"""
inverse_cavity.py
=================
Inverse problem solver for cavity localization in a layered elastic domain.

Problem
-------
Given a displacement measurement at point P = (0, 1) on the top boundary,
estimate the (x, y) center of a hidden elliptical cavity in the domain.

Strategy
--------
1. Build a coarse search grid over the domain (fast survey).
2. For each candidate center, run the forward elasticity problem using
   ``create_layered_grid_with_hole`` + ``solve_and_export`` (with a
   suppressed export to avoid disk I/O overhead in the inversion loop).
3. All forward evaluations are parallelised with ``concurrent.futures``.
4. Identify the best coarse solution and refine with a fine local search.

Public API
----------
GroupSolutionCP1(measured_displacement) -> tuple[float, float]
    Entry point required by the checkpoint specification.

Additional helpers are exposed for testing.
"""

from __future__ import annotations

import warnings
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Sequence

from create_grid import create_layered_grid_with_hole
from common_functions import solve_and_export

from scipy.interpolate import RBFInterpolator

# ---------------------------------------------------------------------------
# Domain constants (matching the provided scripts)
# ---------------------------------------------------------------------------
LX: float = 2.0
LY: float = 1.0
AXES: tuple[float, float] = (0.2, 0.1)   # semi-axes of the cavity
OBS_NODE_INDEX: int = 60                  # index of node closest to P=(0,1)
MESH_COARSE: float = 0.05
MESH_FINE: float = 0.025

# Safety margin so the cavity does not touch the domain boundary
MARGIN_X: float = AXES[0] + 0.05
MARGIN_Y: float = AXES[1] + 0.05


# ---------------------------------------------------------------------------
# Forward solver helpers
# ---------------------------------------------------------------------------

def _find_obs_node(sd) -> int:
    """Return the index of the node closest to the observation point (0, 1)."""
    obs = np.array([0.0, 1.0])
    nodes_xy = sd.nodes[:2, :]          # shape (2, n_nodes)
    dist = np.linalg.norm(nodes_xy.T - obs, axis=1)
    return int(np.argmin(dist))


def _run_forward(
    center: tuple[float, float],
    mesh_size: float,
) -> tuple[float, float]:
    """
    Run the forward problem for a given cavity center.

    Parameters
    ----------
    center : tuple[float, float]
        Candidate (x, y) center of the cavity.
    mesh_size : float
        Target mesh size.

    Returns
    -------
    tuple[float, float]
        Displacement (ux, uy) at the observation node P=(0,1).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            sd, cell_layer = create_layered_grid_with_hole(
                mesh_size=mesh_size,
                center=center,
                axes=AXES,
            )
        except Exception:
            # Degenerate mesh → return sentinel value
            return (np.nan, np.nan)

    # Use a temporary folder (no real export needed inside the loop)
    tmp_folder = Path("/tmp/inv_fwd")
    tmp_folder.mkdir(parents=True, exist_ok=True)

    try:
        u = solve_and_export(
            sd,
            tmp_folder,
            export_name=f"inv_{center[0]:.4f}_{center[1]:.4f}",
            cell_data=[("layer", cell_layer.astype(float))],
        )
    except Exception:
        return (np.nan, np.nan)

    idx = _find_obs_node(sd)
    return (float(u[0][idx]), float(u[1][idx]))


# ---------------------------------------------------------------------------
# Grid-search helpers (called inside worker processes)
# ---------------------------------------------------------------------------

import os
import tempfile

def _worker(args: tuple) -> tuple:
    """Worker function for parallel forward evaluations."""
    cx, cy, mesh_size = args
    
    # Saving original work directory
    original_cwd = os.getcwd()
    
    # Creating a temp !folder for THIS specific process
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        try:
            ux, uy = _run_forward((cx, cy), mesh_size)
        finally:
            os.chdir(original_cwd)
            
    return (cx, cy, ux, uy)

def _build_search_grid(
    nx: int,
    ny: int,
) -> list[tuple[float, float]]:
    """Return a uniform grid of candidate centers inside the safe domain."""
    xs = np.linspace(MARGIN_X, LX - MARGIN_X, nx)
    ys = np.linspace(MARGIN_Y, LY - MARGIN_Y, ny)
    return [(float(x), float(y)) for x in xs for y in ys]


def _residual(
    simulated: tuple[float, float],
    measured: tuple[float, float],
) -> float:
    """Euclidean distance between simulated and measured displacements."""
    s = np.array(simulated, dtype=float)
    m = np.array(measured, dtype=float)
    if np.any(np.isnan(s)):
        return np.inf
    return float(np.linalg.norm(s - m))


# ---------------------------------------------------------------------------
# Parallel grid search
# ---------------------------------------------------------------------------

def parallel_grid_search(
    measured_displacement: tuple[float, float],
    candidates: list[tuple[float, float]],
    mesh_size: float,
    max_workers: int | None = None,
) -> tuple[tuple[float, float], list[dict]]:
    """
    Evaluate all candidate centers in parallel and return the best one.

    Parameters
    ----------
    measured_displacement : tuple[float, float]
        Observed (ux, uy) at P.
    candidates : list[tuple[float, float]]
        List of (x, y) candidate centers.
    mesh_size : float
        Mesh size for forward simulations.
    max_workers : int or None
        Number of parallel workers (None → os.cpu_count()).

    Returns
    -------
    best_center : tuple[float, float]
        Candidate center with lowest residual.
    results : list[dict]
        All evaluation results sorted by residual.
    """
    args_list = [(cx, cy, mesh_size) for (cx, cy) in candidates]

    raw_results: list[tuple[float, float, float, float]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_map = {pool.submit(_worker, args): args for args in args_list}
        for future in as_completed(future_map):
            try:
                raw_results.append(future.result())
            except Exception:
                args = future_map[future]
                raw_results.append((args[0], args[1], np.nan, np.nan))

    results = []
    for cx, cy, ux, uy in raw_results:
        res = _residual((ux, uy), measured_displacement)
        results.append({"center": (cx, cy), "ux": ux, "uy": uy, "residual": res})

    results.sort(key=lambda d: d["residual"])
    best_center = results[0]["center"]
    return best_center, results


# ---------------------------------------------------------------------------
# Two-level inversion (coarse + fine)
# ---------------------------------------------------------------------------

def invert_cavity_location(
    measured_displacement: tuple[float, float],
    nx_coarse: int = 7,  
    ny_coarse: int = 7,
    refine_radius: float = 0.09,
    nx_fine: int = 5,
    ny_fine: int = 5,
    max_workers: int | None = None,
    verbose: bool = True,
) -> tuple[float, float]:
    """
    Two-level inverse solver: coarse survey followed by local refinement.

    Parameters
    ----------
    measured_displacement : tuple[float, float]
        Observed displacement (ux, uy) at P = (0, 1).
    nx_coarse : int
        Number of candidate x-positions on the coarse grid.
    ny_coarse : int
        Number of candidate y-positions on the coarse grid.
    refine_radius : float
        Half-side of the local search box around the coarse best guess.
    nx_fine : int
        Number of candidate x-positions on the fine grid.
    ny_fine : int
        Number of candidate y-positions on the fine grid.
    max_workers : int or None
        Parallel workers for each search level.
    verbose : bool
        Print progress information.

    Returns
    -------
    tuple[float, float]
        Estimated (x, y) center of the cavity.
    """

    # ---- Level 1: coarse global search ----
    coarse_candidates = _build_search_grid(nx_coarse, ny_coarse)
    if verbose:
        print(f"[Coarse] Evaluating {len(coarse_candidates)} candidates ...")

    best_coarse, coarse_results = parallel_grid_search(
        measured_displacement, coarse_candidates, MESH_COARSE, max_workers
    )

    if verbose:
        print(f"[Coarse] Best center: {best_coarse}, "
              f"residual: {coarse_results[0]['residual']:.6e}")
        
    if coarse_results[0]['residual'] <= 5e-7:
        return best_coarse
    
    # ---- Level 2: fine local refinement (TOP-k MULTI-START) ----
    # If x_coarse >= 1.65, we take the top-2 candidates, while if x_coarse < 1.65 it's okay just one (since the previous coarse grid is precise)
    if coarse_results[0]['center'][0] >= 1.65:
        top_n = 2
    else:
        top_n = 1

    fine_candidates = []
    
    for i in range(min(top_n, len(coarse_results))):
        cx, cy = coarse_results[i]['center']
        
        xs_fine = np.linspace(
            max(MARGIN_X, cx - refine_radius),
            min(LX - MARGIN_X, cx + refine_radius),
            nx_fine,
        )
        ys_fine = np.linspace(
            max(MARGIN_Y, cy - refine_radius),
            min(LY - MARGIN_Y, cy + refine_radius),
            ny_fine,
        )
        
        # Adding points to global list
        for x in xs_fine:
            for y in ys_fine:
                fine_candidates.append((float(x), float(y)))

    # Removing duplicates 
    fine_candidates = list(set(fine_candidates))

    if verbose:
        print(f"[Fine]   Evaluating {len(fine_candidates)} candidates "
              f"around top {top_n} coarse spots ...")

    best_fine, fine_results = parallel_grid_search(
        measured_displacement, fine_candidates, MESH_FINE, max_workers
    )

    if verbose:
        print(f"[Fine]   Best center: {best_fine}, "
              f"residual: {fine_results[0]['residual']:.6e}")

    return best_fine

# ---------------------------------------------------------------------------
# Checkpoint-required public function
# ---------------------------------------------------------------------------

def GroupSolutionCP1(
    measured_displacement: tuple[float, float],
) -> tuple[float, float]:
    """
    Estimate the cavity center from a displacement measurement at P = (0, 1).

    Parameters
    ----------
    measured_displacement : tuple[float, float]
        Observed displacement (ux, uy) at the measurement point.

    Returns
    -------
    tuple[float, float]
        Estimated (x, y) coordinates of the cavity center.
    """
    center_location = invert_cavity_location(
        measured_displacement,
        nx_coarse=7,
        ny_coarse=7,
        refine_radius=0.09,
        nx_fine=5,
        ny_fine=5,
        verbose=True,
    )
    return center_location


# ---------------------------------------------------------------------------
# Utility: generate a synthetic measurement for a known center
# ---------------------------------------------------------------------------

def generate_synthetic_measurement(
    true_center: tuple[float, float],
    mesh_size: float = MESH_FINE,
) -> tuple[float, float]:
    """
    Run the forward problem for a known center and return the measurement.

    Useful to build test cases without real experimental data.

    Parameters
    ----------
    true_center : tuple[float, float]
        The true (x, y) cavity center.
    mesh_size : float
        Mesh size for the forward simulation.

    Returns
    -------
    tuple[float, float]
        Synthetic displacement (ux, uy) at P = (0, 1).
    """
    ux, uy = _run_forward(true_center, mesh_size)
    return (ux, uy)

   

def _surrogate_minimum(
    coarse_results: list[dict],
    n_grid: int = 300,
) -> tuple[float, float]:
    """
    Fit a thin-plate-spline RBF on the coarse residuals and return
    the predicted minimum location (essentially free: evaluated on a
    dense grid in <<1 ms, no PDE solve needed).
    """
    valid = [r for r in coarse_results if np.isfinite(r["residual"])]
    if len(valid) < 4:
        return coarse_results[0]["center"]

    pts = np.array([r["center"] for r in valid])   # (N, 2)
    res = np.array([r["residual"] for r in valid])  # (N,)

    # Log-transform: stabilises the fit when residuals span orders of magnitude
    rbf = RBFInterpolator(
        pts, np.log(res + 1e-30),
        kernel="thin_plate_spline",
        smoothing=1e-6,
    )

    sx = np.linspace(MARGIN_X, LX - MARGIN_X, n_grid)
    sy = np.linspace(MARGIN_Y, LY - MARGIN_Y, n_grid)
    XX, YY = np.meshgrid(sx, sy, indexing="ij")
    grid = np.column_stack([XX.ravel(), YY.ravel()])   # (n_grid², 2)

    best = grid[np.argmin(rbf(grid))]
    return (float(best[0]), float(best[1]))