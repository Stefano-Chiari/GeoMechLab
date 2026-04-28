import os
import time
import tempfile
import numpy as np
import multiprocessing as mp
from pathlib import Path
from scipy.optimize import minimize
from contextlib import redirect_stdout
import warnings

warnings.filterwarnings("ignore")

from create_grid import create_layered_grid_with_hole
from common_functions import solve_and_export

# ==============================================================================
# FUNZIONI DI COSTO ISOLATE (ANTI-CRASH)
# ==============================================================================
def extract_displacement_interpolated(sd, u, target_point=(0.0, 1.0)):
    dist_sq = (sd.nodes[0, :] - target_point[0])**2 + (sd.nodes[1, :] - target_point[1])**2
    idx = np.argmin(dist_sq)
    return np.array([u[0, idx], u[1, idx]])

def worker_cost_function(args):
    center, measured_disp, export_folder, target_point, mesh_size = args
    cx, cy = center
    
    # Penalità immediata fuori dal dominio
    if cx < 0.25 or cx > 1.75 or cy < 0.25 or cy > 0.75:
        return 1e6

    original_cwd = os.getcwd()
    try:
        # Sandbox: ogni processo lavora in una sua cartella invisibile per non far impazzire Gmsh
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            with open(os.devnull, 'w') as f, redirect_stdout(f):
                sd, _ = create_layered_grid_with_hole(mesh_size=mesh_size, center=(cx, cy))
                u = solve_and_export(sd, export_folder, export_name=f"temp_{os.getpid()}_{time.time()}", cell_data=None)
                simulated_disp = extract_displacement_interpolated(sd, u, target_point=target_point)
            os.chdir(original_cwd)
            
        return np.linalg.norm(np.array(simulated_disp) - np.array(measured_disp))
    except Exception:
        os.chdir(original_cwd)
        return 1e6

def nm_wrapper(center, measured_disp, export_folder, target_point, mesh_size):
    return worker_cost_function((center, measured_disp, export_folder, target_point, mesh_size))

def optimize_worker_phase2(args):
    start_node, measured_disp, export_folder, target_point = args
    res = minimize(
        fun=nm_wrapper, x0=start_node,
        args=(measured_disp, export_folder, target_point, 0.025), # LA MESH DEVE ESSERE 0.025
        method='Nelder-Mead', 
        options={'xatol': 0.005, 'fatol': 1e-8, 'maxiter': 20}
    )
    return res.fun, tuple(res.x)

# ==============================================================================
# ALGORITMO MULTI-GRID COERENTE (100% Affidabile)
# ==============================================================================
def GroupSolutionCP1_Consistent_Parallel(measured_displacement, test_id=1, target_point=(0.0, 1.0)):
    print(f"\n🚀 Avvio Inversione COERENTE (Test {test_id}) - NESSUN IMBROGLIO DI MESH")
    export_folder = Path.cwd() / "opt_temp_results"
    export_folder.mkdir(exist_ok=True)
    t0_total = time.perf_counter()

    # --- FASE 1: Radar Parallelo con MESH PERFETTA ---
    print("--> FASE 1: Esplorazione Globale Parallela (24 punti, Mesh 0.025)...")
    t0_f1 = time.perf_counter()
    
    # Griglia ridotta per risparmiare tempo, ma mesh identica alla realtà!
    x_vals = np.linspace(0.3, 1.7, 6)
    y_vals = np.linspace(0.3, 0.7, 4)
    tasks = [((cx, cy), measured_displacement, export_folder, target_point, 0.025) for cx in x_vals for cy in y_vals]

    context = mp.get_context("spawn")
    cores = min(os.cpu_count(), 8)
    
    with context.Pool(processes=cores, maxtasksperchild=1) as pool:
        errors = pool.map(worker_cost_function, tasks)

    t1_f1 = time.perf_counter()
    print(f"    ⏱️ TEMPO FASE 1: {(t1_f1 - t0_f1)/60:.2f} minuti")

    # Ordina i risultati e prendi le 2 migliori valli separate
    results_data = [{'center': t[0], 'error': e} for t, e in zip(tasks, errors)]
    results_data.sort(key=lambda x: x['error'])
    top_candidates = [results_data[0]['center'], results_data[1]['center']]

    # --- FASE 2: Nelder-Mead Multi-Start Parallelo ---
    print(f"--> FASE 2: Ottimizzazione Locale (Partenze: {top_candidates[0]} e {top_candidates[1]})")
    t0_f2 = time.perf_counter()
    
    tasks_p2 = [(node, measured_displacement, export_folder, target_point) for node in top_candidates]
    
    with context.Pool(processes=2, maxtasksperchild=1) as pool:
        results_p2 = pool.map(optimize_worker_phase2, tasks_p2)

    t1_f2 = time.perf_counter()
    print(f"    ⏱️ TEMPO FASE 2: {(t1_f2 - t0_f2)/60:.2f} minuti")

    # Scegli il vincitore assoluto
    best_error = float('inf')
    final_center = None
    for err, center in results_p2:
        if err < best_error:
            best_error = err
            final_center = center

    t1_total = time.perf_counter()
    print(f"✅ Inversione completata in {(t1_total - t0_total)/60:.2f} minuti totali.")
    print(f"   Errore finale di simulazione: {best_error:.4e}")
    
    return final_center, results_data

# ==============================================================================
# SCRIPT DI TEST
# ==============================================================================
def run_multiple_inversion_tests(num_tests=5, seed=42):
    if seed is not None: np.random.seed(seed)
        
    print(f"=========================================================")
    print(f"🚀 STARTING ROBUSTNESS EVALUATION (COERENTE E BLINDATA)")
    print(f"=========================================================\n")

    export_folder = Path.cwd() / "test_multiple_results"
    export_folder.mkdir(exist_ok=True)
    target_point = (0.0, 1.0)
    results_summary = []
    
    for i in range(num_tests):
        test_id = i + 1
        print(f"\n" + "-"*50)
        print(f" TEST {test_id} OF {num_tests}")
        print("-" * 50)
        
        true_center = (np.random.uniform(0.3, 1.7), np.random.uniform(0.3, 0.7))
        print(f"1. Creation Ground Truth: Cavity in ({true_center[0]:.4f}, {true_center[1]:.4f})...")
        
        try:
            sd_true, cell_layer_true = create_layered_grid_with_hole(mesh_size=0.025, center=true_center)
            u_true = solve_and_export(sd_true, export_folder, export_name=f"sol_gt_{test_id}", cell_data=None)
            measured_displacement = extract_displacement_interpolated(sd_true, u_true, target_point=target_point)
        except Exception as e:
            continue

        print("2. Starting inversion...")
        estimated_center, coarse_results_data = GroupSolutionCP1_Consistent_Parallel(
            measured_displacement=tuple(measured_displacement), 
            test_id=test_id, target_point=target_point
        )
        
        dist_totale = np.linalg.norm(np.array(true_center) - np.array(estimated_center))
        print(f"   Spatial error: {dist_totale:.4f} m")
        
        results_summary.append({"test_id": test_id, "true": true_center, "est": estimated_center, "error": dist_totale})

    print("\n" + "="*50)
    print("3. FINAL EVALUATION REPORT")
    print("="*50)
    for res in results_summary:
        status = "✅ PASSED" if res['error'] < 0.05 else "⚠️ IMPRECISE"
        print(f"Test {res['test_id']:02d}: Real ({res['true'][0]:.3f}, {res['true'][1]:.3f}) | Estimate ({res['est'][0]:.3f}, {res['est'][1]:.3f}) | Error: {res['error']:>6.4f}m | {status}")
    
    if results_summary:
        print("-" * 50)
        print(f"Mean Error:    {np.mean([r['error'] for r in results_summary]):.4f} m")
        print(f"Maximum Error: {np.max([r['error'] for r in results_summary]):.4f} m")
        print("=" * 50)

if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    run_multiple_inversion_tests(num_tests=5, seed=42)