#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APPROACH 1 — Three improvements over the baseline:
  1. Fast two-pass Phase-1: parallel coarse global scan → parallel local zoom
  2. Component-wise balanced residuals in the cost function
  3. Phase-2 runs from the top-k candidates (true multi-start)

Speed strategy
--------------
The previous Phase-1 ran ~456 sequential FEM solves at mesh_size=0.1, taking
~400 s.  The new Phase-1 uses two passes:

  Pass A (global, very coarse mesh 0.15):
      6×4 = 24 points covering the full domain, all run in parallel.
      Purpose: cheaply identify the 3 best regions.

  Pass B (local zoom, coarse mesh 0.1):
      For each of the top-3 Pass-A results, evaluate a tight 5×4 = 20-point
      grid in a ±0.25 neighbourhood, again all run in parallel.
      Purpose: sharpen the starting points for Phase-2.

Total Phase-1 evaluations: 24 + 60 = 84  (vs 456 before).
Expected wall-time at ~0.9 s/eval with N_WORKERS parallel processes:
  Sequential: ~76 s
  4 workers:  ~19 s
  8 workers:  ~10 s
"""

import os
import tempfile
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from scipy.interpolate import griddata
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import time

from create_grid import create_layered_grid_with_hole
from common_functions import solve_and_export, _default_traction, _standard_boundary_masks
from elastic_pb import ElasticProblem
import pygeon as pg


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def solve_fem_in_memory(sd):
    """Assemble and solve the elasticity system without any disk I/O."""
    lambda_ = 1.0
    mu = 0.5
    param = {pg.LAME_LAMBDA: lambda_, pg.LAME_MU: mu}
    bottom, top = _standard_boundary_masks(sd)
    elastic_pb = ElasticProblem(sd)
    A, b = elastic_pb.assemble_problem(param, _default_traction, top)
    u, sigma = elastic_pb.solve_linear_system(A, b, bottom)
    return u


def extract_displacement_interpolated(sd, u, target_point=(0.0, 1.0)):
    """Interpolate displacement at an arbitrary point via griddata."""
    x_nodes = sd.nodes[0, :]
    y_nodes = sd.nodes[1, :]
    points = np.column_stack((x_nodes, y_nodes))

    ux = griddata(points, u[0, :], target_point, method='linear')
    uy = griddata(points, u[1, :], target_point, method='linear')

    if np.isnan(ux) or np.isnan(uy):
        ux = griddata(points, u[0, :], target_point, method='nearest')
        uy = griddata(points, u[1, :], target_point, method='nearest')

    return np.array([float(ux), float(uy)])


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE 2 — Component-wise balanced (scaled) residuals
# ─────────────────────────────────────────────────────────────────────────────
def balanced_residual(u_pred, u_obs, eps=1e-6):
    """Component-wise scaled L2 residual."""
    diff = u_pred - np.array(u_obs)
    scale = np.abs(np.array(u_obs)) + eps
    return float(np.sum((diff / scale) ** 2))


def evaluation_cost_function(center, mesh_size, measured_disp, target_point=(0.0, 1.0)):
    """
    Build mesh → solve FEM → extract displacement → balanced residual.

    Each call runs in its own temporary directory so that concurrent or
    sequential invocations never share the gmsh scratch file.
    """
    import tempfile, os
    cx, cy = center
    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            sd, _ = create_layered_grid_with_hole(mesh_size=mesh_size, center=(cx, cy))
            u = solve_fem_in_memory(sd)
            sim_disp = extract_displacement_interpolated(sd, u, target_point=target_point)
            error = balanced_residual(sim_disp, measured_disp)
        except Exception as e:
            print(f"⚠️ Failure at ({cx:.2f}, {cy:.2f}) | {type(e).__name__}: {e}")
            error = np.inf
        finally:
            os.chdir(old_cwd)
    return error


# ─────────────────────────────────────────────────────────────────────────────
# SPEED — top-level worker function for ProcessPoolExecutor
#
# ROOT CAUSE OF CRASH: gmsh writes a file called 'gmsh_frac_file.msh' into
# the current working directory.  When multiple workers run in parallel they
# all share the same cwd and clobber each other's mesh file, producing
# "Couldn't read file gmsh_frac_file.msh" errors and KeyError crashes.
#
# FIX: each worker is given a unique temporary directory and os.chdir() into
# it before calling the mesher.  The tmpdir is deleted on exit.
# ─────────────────────────────────────────────────────────────────────────────
def _parallel_eval(args):
    """
    Worker: call evaluation_cost_function (which manages its own tmpdir)
    and return a result dict.  Must be module-level for pickle serialisation.
    """
    cx, cy, mesh_size, measured_disp, target_point = args
    err = evaluation_cost_function((cx, cy), mesh_size, measured_disp, target_point)
    return {'center': (cx, cy), 'error': err}


def parallel_grid_scan(grid_points, mesh_size, measured_disp, target_point, n_workers):
    """
    Evaluate all grid_points in parallel using ProcessPoolExecutor.

    Parameters
    ----------
    grid_points : list of (cx, cy) tuples
    n_workers   : number of worker processes.
                  Set to 1 to fall back to sequential execution (useful for
                  debugging or when the mesher is not fork-safe).

    Returns
    -------
    list of {'center': (cx,cy), 'error': float}, unsorted.
    """
    args_list = [
        (cx, cy, mesh_size, measured_disp, target_point)
        for cx, cy in grid_points
    ]

    results = [None] * len(args_list)

    if n_workers == 1:
        # Sequential fallback — each call still uses its own tmpdir via
        # _parallel_eval, so behaviour is identical to the parallel path.
        for i, args in enumerate(args_list):
            results[i] = _parallel_eval(args)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"    [{i+1}/{len(args_list)}] {results[i]['center']} "
                      f"err={results[i]['error']:.4e}")
    else:
        future_to_idx = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for i, args in enumerate(args_list):
                fut = pool.submit(_parallel_eval, args)
                future_to_idx[fut] = i
            done = 0
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    # Worker crashed (e.g. mesher error): treat as inf cost
                    cx, cy = args_list[i][0], args_list[i][1]
                    print(f"  ⚠️ Worker failed at ({cx:.3f},{cy:.3f}): {e}")
                    results[i] = {'center': (cx, cy), 'error': np.inf}
                done += 1
                if done % 10 == 0 or done == 1:
                    print(f"    [{done}/{len(args_list)}] done")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Early-stop wrapper
#
# FIX 1: tol tightened 1e-3 → 1e-5.  A balanced residual of 1e-3 is
#         reachable at spatially wrong locations when displacements are small.
# ─────────────────────────────────────────────────────────────────────────────
class EarlyStopException(Exception):
    pass


def make_cost_with_early_stop(mesh_size, measured_disp, tol=1e-5):
    def cost(center):
        err = evaluation_cost_function(center, mesh_size, measured_disp)
        if err < tol:
            raise EarlyStopException(center, err)
        return err
    return cost


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INVERSION
# ─────────────────────────────────────────────────────────────────────────────
def GroupSolutionCP1(measured_displacement, test_id=1, target_point=(0.0, 1.0)):
    print("\nStarting inverse Multi-Grid with Multi-Start (Approach 1 — fast)...")
    export_folder = Path.cwd() / "opt_final_results"
    export_folder.mkdir(exist_ok=True)
    log_file_path = export_folder / f"optimization_log_test_{test_id}.txt"

    # SPEED: use all available cores minus 1 (leave one for the OS).
    # Set N_WORKERS = 1 to disable parallelism (e.g. if mesher is not fork-safe).
    N_WORKERS = max(1, os.cpu_count() - 1)

    # Mesh sizes
    global_mesh  = 0.15   # SPEED: was 0.1 — coarser = faster per eval in Pass A
    zoom_mesh    = 0.1    # Pass B uses original coarse mesh for accuracy
    refined_mesh = 0.025   # Phase 2 fine mesh: the same of the test set to do not have numerical noise 

    # Domain bounds
    X_MIN, X_MAX = 0.3, 1.7
    Y_MIN, Y_MAX = 0.3, 0.7

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 1 — Pass A: parallel global coarse scan
    #
    # SPEED: only 6×4 = 24 points (vs 456 before) at a coarser mesh.
    # Purpose is purely to rank regions, not to be precise.
    # All 24 evaluations run in parallel.
    # ─────────────────────────────────────────────────────────────────────
    print(f"--> PHASE 1A: Global coarse scan  "
          f"(6×4 grid, mesh={global_mesh}, {N_WORKERS} workers)...")
    t0 = time.time()

    x_global = np.linspace(X_MIN, X_MAX, 6)
    y_global = np.linspace(Y_MIN, Y_MAX, 4)
    global_pts = [(cx, cy) for cx in x_global for cy in y_global]

    global_results = parallel_grid_scan(
        global_pts, global_mesh, measured_displacement, target_point, N_WORKERS)
    global_results.sort(key=lambda r: r['error'])

    print(f"  Pass A done: {time.time()-t0:.1f}s  "
          f"| best so far: {global_results[0]['center']}  "
          f"err={global_results[0]['error']:.4e}")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 1 — Pass B: parallel local zoom around top-3 Pass-A results
    #
    # SPEED: 3 × (5×4) = 60 points at zoom_mesh=0.1, all in parallel.
    # The ±0.25 neighbourhood is chosen to be ~2 grid spacings of Pass A
    # (global spacing ≈ 0.28 m in x, 0.13 m in y).
    # ─────────────────────────────────────────────────────────────────────
    N_ZOOM_SEEDS  = 3   # how many Pass-A winners to zoom around
    ZOOM_HALF_X   = 0.25
    ZOOM_HALF_Y   = 0.15
    ZOOM_NX, ZOOM_NY = 5, 4

    print(f"\n--> PHASE 1B: Local zoom  "
          f"({N_ZOOM_SEEDS}×{ZOOM_NX}×{ZOOM_NY} grid, mesh={zoom_mesh}, {N_WORKERS} workers)...")
    t1 = time.time()

    zoom_pts = []
    for seed in global_results[:N_ZOOM_SEEDS]:
        cx0, cy0 = seed['center']
        xs = np.linspace(max(X_MIN, cx0 - ZOOM_HALF_X), min(X_MAX, cx0 + ZOOM_HALF_X), ZOOM_NX)
        ys = np.linspace(max(Y_MIN, cy0 - ZOOM_HALF_Y), min(Y_MAX, cy0 + ZOOM_HALF_Y), ZOOM_NY)
        zoom_pts.extend([(cx, cy) for cx in xs for cy in ys])

    # Deduplicate points that overlap between zoom windows
    zoom_pts = list({(round(cx, 6), round(cy, 6)) for cx, cy in zoom_pts})

    zoom_results = parallel_grid_scan(
        zoom_pts, zoom_mesh, measured_displacement, target_point, N_WORKERS)
    zoom_results.sort(key=lambda r: r['error'])

    print(f"  Pass B done: {time.time()-t1:.1f}s  "
          f"| best so far: {zoom_results[0]['center']}  "
          f"err={zoom_results[0]['error']:.4e}")

    # Merge both passes and re-sort for the heatmap and Phase-2 seeds
    coarse_results = global_results + zoom_results
    coarse_results.sort(key=lambda r: r['error'])

    print(f"\nPhase 1 total: {time.time()-t0:.1f}s  "
          f"({len(coarse_results)} evaluations, {N_WORKERS} workers)")

    with open(log_file_path, "w") as f:
        f.write("--- PHASE 1 (parallel two-pass) ---\n")
        f.write("Center_X, Center_Y, Error\n")
        for res in coarse_results:
            f.write(f"{res['center'][0]:.4f}, {res['center'][1]:.4f}, {res['error']:.6e}\n")

    # ─────────────────────────────────────────────────────────────────────
    # PHASE 2 — CHANGE 3: run Nelder-Mead from the top-k candidates
    #   
    #   If the best point is on the left hand side we go super fast cause the signal is very good
    #       - take only the top-2 points
    #       - tol: 1e-6
    #       - GOOD_ENOUGH_ERROR: 1e-5
    #   If the best point is on the right hand side we act more careful cause the signal is slower and can be not good -> local minima
    #   To provent collapse we use:
    #       - take the top-5 points
    #       - tol: 1e-8, so it can't fool us with a false minima
    #       - GOOD_ENOUGH_ERROR: 1e-7, same as above
    # ─────────────────────────────────────────────────────────────────────
    if coarse_results[0]['center'][0] < 0.9:
        TOP_K = 2   
        early_tol = 1e-6
        GOOD_ENOUGH_ERROR = 1e-5
    else:
        TOP_K = 5
        early_tol = 1e-8
        GOOD_ENOUGH_ERROR = 1e-7

    top_candidates = [res['center'] for res in coarse_results[:TOP_K]]

    # FIX 2: seed best_overall with Phase-1 winner so Phase-2 can only improve.
    # This guarantees we never return a Phase-2 result that is spatially worse
    # than what the coarse grid already found.
    best_overall_center = coarse_results[0]['center']
    best_overall_error  = coarse_results[0]['error']

    print(f"\n--> PHASE 2: Refined Multi-Start (top-{TOP_K} candidates)...")
    print(f"    Phase-1 best seed: {best_overall_center}  err={best_overall_error:.4e}")
    t1 = time.time()

    bounds = [(0.25, 1.75), (0.25, 0.75)]

    with open(log_file_path, "a") as f:
        f.write(f"\n--- PHASE 2 (Nelder-Mead, top-{TOP_K}) ---\n")

    for idx, start_center in enumerate(top_candidates):
        print(f"  Candidate {idx+1}/{TOP_K}: start={start_center}")
        try:
            res = minimize(
                fun=make_cost_with_early_stop(refined_mesh, measured_displacement, tol=early_tol),
                x0=start_center,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 20, 'fatol': early_tol}
            )
            candidate_center = tuple(res.x)
            candidate_error  = res.fun

        except EarlyStopException as e:
            # FIX 4: on EarlyStop, extract result but DO NOT break immediately.
            # Record as a normal candidate and let the comparison below decide
            # whether it is actually the global best.
            candidate_center = tuple(e.args[0])
            candidate_error  = e.args[1]
            print(f"    Early-stop triggered (err={candidate_error:.4e})")

        with open(log_file_path, "a") as f:
            f.write(f"Start {start_center} -> End {candidate_center} | Error: {candidate_error:.6e}\n")

        # FIX 5: always compare — update global best only if this candidate
        # is genuinely better (includes the Phase-1 seed as the floor).
        if candidate_error < best_overall_error:
            best_overall_error  = candidate_error
            best_overall_center = candidate_center
            print(f"    New best: {best_overall_center}  err={best_overall_error:.4e}")

        # Only break early once the *confirmed* best meets the strict threshold.
        if best_overall_error < GOOD_ENOUGH_ERROR:
            print(f"  ✅ Good-enough threshold reached at candidate {idx+1}, stopping early.")
            break

    print(f"Phase 2 done: {time.time()-t1:.1f}s")

    # ─────────────────────────────────────────────────────────────────────
    # Export best FEM solution to disk
    # ─────────────────────────────────────────────────────────────────────
    print(f"\n✅ Final Error: {best_overall_error:.4e}")
    print("--> Exporting optimal configuration...")

    sd_final, cell_layer_final = create_layered_grid_with_hole(
        mesh_size=refined_mesh,
        center=best_overall_center
    )
    solve_and_export(
        sd=sd_final,
        folder_export=export_folder,
        export_name=f"best_estimated_cavity_test_{test_id}",
        cell_data=[("layer", cell_layer_final.astype(float))] if cell_layer_final is not None else None
    )
    print(f"Files saved to: {export_folder}")
    return best_overall_center, coarse_results


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISATION  (unchanged logic, labels updated)
# ─────────────────────────────────────────────────────────────────────────────
def plot_cost_heatmap(coarse_results, true_center, estimated_center, test_id, export_folder):
    valid_res = [r for r in coarse_results if r['error'] != np.inf]
    if len(valid_res) < 3:
        print(f"⚠️ Test {test_id}: not enough valid points for heatmap.")
        return

    x = np.array([r['center'][0] for r in valid_res])
    y = np.array([r['center'][1] for r in valid_res])
    z = np.log10(np.array([r['error'] for r in valid_res]) + 1e-12)

    fig, ax = plt.subplots(figsize=(9, 5))
    tricontour = ax.tricontourf(x, y, z, levels=25, cmap='viridis_r', alpha=0.85)
    cbar = fig.colorbar(tricontour, ax=ax)
    cbar.set_label('Log10(Balanced Residual)')   # CHANGE 2: label updated

    ax.scatter(x, y, color='black', s=15, alpha=0.4, marker='.', label='Phase-1 nodes')
    ax.scatter(*true_center, color='lime', marker='*', s=250,
               edgecolors='black', linewidths=1.5, zorder=5, label='Ground Truth')
    ax.scatter(*estimated_center, color='red', marker='X', s=150,
               edgecolors='black', linewidths=1, zorder=5, label='Optimised Estimate')

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Test {test_id}: Cost Function Landscape")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc='upper left', framealpha=0.9, bbox_to_anchor=(1.18, 1))
    ax.grid(True, linestyle='--', alpha=0.3)

    plot_path = export_folder / f"heatmap_approach1_test_{test_id}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    # plt.show()   
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# ROBUSTNESS TEST LOOP  (unchanged structure)
# ─────────────────────────────────────────────────────────────────────────────
def run_multiple_inversion_tests(num_tests=5, seed=42):
    if seed is not None:
        np.random.seed(seed)

    print("=" * 57)
    print(f"🚀 APPROACH 1 — ROBUSTNESS EVAL ON {num_tests} RANDOM CAVITIES")
    print("=" * 57)

    export_folder = Path.cwd() / "test_multiple_results_approach1"
    export_folder.mkdir(exist_ok=True)

    target_point = (0.0, 1.0)
    true_mesh_size = 0.025
    results_summary = []

    for i in range(num_tests):
        test_id = i + 1
        print(f"\n{'─'*50}\n TEST {test_id} OF {num_tests}\n{'─'*50}")

        true_cx = np.random.uniform(0.3, 1.7)
        true_cy = np.random.uniform(0.3, 0.7)
        true_center = (true_cx, true_cy)
        print(f"1. Ground Truth cavity at ({true_cx:.4f}, {true_cy:.4f})")

        try:
            sd_true, cell_layer_true = create_layered_grid_with_hole(
                mesh_size=true_mesh_size, center=true_center)
            u_true = solve_and_export(
                sd_true, export_folder,
                export_name=f"sol_ground_truth_test_{test_id}",
                cell_data=[("layer", cell_layer_true.astype(float))] if cell_layer_true is not None else None
            )
            measured_displacement = extract_displacement_interpolated(sd_true, u_true, target_point)
        except Exception as e:
            print(f"⚠️ Mesher error at {true_center}: {e}. Skipping.")
            continue

        print("2. Starting inversion...")
        estimated_center, coarse_results_data = GroupSolutionCP1(
            measured_displacement=tuple(measured_displacement),
            test_id=test_id,
            target_point=target_point
        )
        
        error_x = abs(true_center[0] - estimated_center[0])
        error_y = abs(true_center[1] - estimated_center[1])
        dist = np.sqrt(error_x**2 + error_y**2)

        print(f"\n TEST {test_id} RESULTS:")
        print(f"   Real center:     ({true_cx:.4f}, {true_cy:.4f})")
        print(f"   Estimated center:({estimated_center[0]:.4f}, {estimated_center[1]:.4f})")
        print(f"   Spatial error:   {dist:.4f} m")

        plot_cost_heatmap(
            coarse_results=coarse_results_data,
            true_center=true_center,
            estimated_center=estimated_center,
            test_id=test_id,
            export_folder=export_folder
        )

        results_summary.append({
            "test_id": test_id,
            "true": true_center,
            "est": estimated_center,
            "error": dist
        })

    # ─── Final report ────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("3. FINAL EVALUATION REPORT — APPROACH 1")
    print("=" * 50)

    if not results_summary:
        print("No tests completed successfully.")
        return

    for res in results_summary:
        status = "✅ PASSED" if res['error'] < 0.05 else "⚠️ IMPRECISE"
        t_x, t_y = res['true']
        e_x, e_y = res['est']
        print(f"Test {res['test_id']:02d}: Real ({t_x:.3f}, {t_y:.3f}) | "
              f"Estimate ({e_x:.3f}, {e_y:.3f}) | Error: {res['error']:>6.4f}m | {status}")

    avg_error = np.mean([r['error'] for r in results_summary])
    max_error = np.max([r['error'] for r in results_summary])
    print("-" * 50)
    print(f"Mean Error:    {avg_error:.4f} m")
    print(f"Maximum Error: {max_error:.4f} m")
    print("=" * 50)


# ProcessPoolExecutor on macOS/Windows uses 'spawn': without this guard the
# module would be re-imported in each worker and launch recursive pools.
if __name__ == '__main__':
    run_multiple_inversion_tests(num_tests=5, seed=42)
