import os
import time
import numpy as np
import multiprocessing as mp
from pathlib import Path
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import warnings

warnings.filterwarnings("ignore")

from create_grid import create_layered_grid_with_hole
from common_functions import solve_and_export

# ==============================================================================
# FUNZIONI DI ESTRAZIONE E COSTO
# ==============================================================================
def extract_displacement_interpolated(sd, u, target_point=(0.0, 1.0)):
    """Trova il nodo più vicino al target e restituisce lo spostamento."""
    dist_sq = (sd.nodes[0, :] - target_point[0])**2 + (sd.nodes[1, :] - target_point[1])**2
    idx = np.argmin(dist_sq)
    return np.array([u[0, idx], u[1, idx]])

def worker_cost_function(args):
    """
    args = (center, measured_disp, export_folder, target_point, side)
    """
    center, measured_disp, export_folder, target_point, side = args
    cx, cy = center

    # 🚧 GABBIA INVALICABILE: L'ottimizzatore non può fuggire dal suo lato
    if side == 'left':
        if cx < 0.25 or cx >= 1.0 or cy < 0.25 or cy > 0.75:
            return 1e6
        mesh_size = 0.045  # Tecnica Chloe (Veloce)
    elif side == 'right':
        if cx < 1.0 or cx > 1.75 or cy < 0.25 or cy > 0.75:
            return 1e6
        mesh_size = 0.025  # Tecnica Edo/Ste (Precisa)
    else: # 'fair_judge'
        if cx < 0.25 or cx > 1.75 or cy < 0.25 or cy > 0.75:
            return 1e6
        mesh_size = 0.025  # Il Giudice usa sempre la mesh perfetta

    try:
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            sd, _ = create_layered_grid_with_hole(mesh_size=mesh_size, center=(cx, cy))
            u = solve_and_export(sd, export_folder, export_name=f"temp_inv_{os.getpid()}", cell_data=None)
            simulated_disp = extract_displacement_interpolated(sd, u, target_point=target_point)
            
        return np.linalg.norm(np.array(simulated_disp) - np.array(measured_disp))
    except Exception:
        return 1e6

def nm_cost_function(center, measured_disp, export_folder, target_point, side):
    """Wrapper di supporto per scipy minimize"""
    return worker_cost_function((center, measured_disp, export_folder, target_point, side))


# ==============================================================================
# ALGORITMO IBRIDO BLINDATO
# ==============================================================================
def GroupSolutionCP1_Hybrid_Parallel(measured_displacement, test_id=1, target_point=(0.0, 1.0)):
    print(f"\n🚀 Avvio Inversione IBRIDA CHLOE/EDO (Test {test_id})")
    export_folder = Path.cwd() / "opt_temp_results"
    export_folder.mkdir(exist_ok=True)
    
    t0 = time.perf_counter()

    # --- FASE 1: Ricerca Globale Separata ---
    print("--> FASE 1: Esplorazione Parallela...")
    # Creiamo punti separati per sinistra e destra
    x_vals_left = np.linspace(0.3, 0.9, 4)
    x_vals_right = np.linspace(1.1, 1.7, 4)
    y_vals = np.linspace(0.3, 0.7, 5)
    
    tasks = []
    for cx in x_vals_left:
        for cy in y_vals:
            tasks.append(((cx, cy), measured_displacement, export_folder, target_point, 'left'))
    for cx in x_vals_right:
        for cy in y_vals:
            tasks.append(((cx, cy), measured_displacement, export_folder, target_point, 'right'))

    # 🔥 PROTEZIONE ANTI-FREEZE: "spawn" resetta Gmsh ad ogni processo!
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(mp_context=context) as executor:
        errors = list(executor.map(worker_cost_function, tasks))

    # Dividiamo i risultati nei due domini
    res_left = [{'center': t[0], 'error': e} for t, e in zip(tasks, errors) if t[4] == 'left']
    res_right = [{'center': t[0], 'error': e} for t, e in zip(tasks, errors) if t[4] == 'right']
    
    res_left.sort(key=lambda x: x['error'])
    res_right.sort(key=lambda x: x['error'])
    
    best_left_start = res_left[0]['center']
    best_right_start = res_right[0]['center']

    # --- FASE 2: Ottimizzazione Locale Indipendente ---
    print(f"--> FASE 2: Nelder-Mead (Sinistra: {best_left_start} | Destra: {best_right_start})")
    
    nm_left = minimize(
        fun=nm_cost_function, x0=best_left_start,
        args=(measured_displacement, export_folder, target_point, 'left'),
        method='Nelder-Mead', options={'xatol': 0.005, 'fatol': 1e-6, 'maxiter': 20}
    )
    
    nm_right = minimize(
        fun=nm_cost_function, x0=best_right_start,
        args=(measured_displacement, export_folder, target_point, 'right'),
        method='Nelder-Mead', options={'xatol': 0.005, 'fatol': 1e-8, 'maxiter': 20}
    )

    # --- FASE 3: IL GIUDICE EQUO ---
    print("--> FASE 3: Valutazione Equa (Il Giudice usa la mesh 0.025 per entrambi)...")
    
    # Testiamo i due vincitori usando la STESSA mesh del Ground Truth
    err_left_fair = worker_cost_function((nm_left.x, measured_displacement, export_folder, target_point, 'fair_judge'))
    err_right_fair = worker_cost_function((nm_right.x, measured_displacement, export_folder, target_point, 'fair_judge'))
    
    print(f"    Errore VERO Sinistra (Chloe): {err_left_fair:.4e}")
    print(f"    Errore VERO Destra (Edo):     {err_right_fair:.4e}")
    
    if err_left_fair < err_right_fair:
        final_center = tuple(nm_left.x)
        print("    🏆 Vince l'approccio di Chloe (Sinistra)")
    else:
        final_center = tuple(nm_right.x)
        print("    🏆 Vince l'approccio di Edo (Destra)")

    t1 = time.perf_counter()
    print(f"✅ Inversione completata in {(t1-t0)/60:.2f} minuti.")
    
    # Ritorna i dati anche per la Heatmap
    coarse_results_data = [{'center': t[0], 'error': e} for t, e in zip(tasks, errors)]
    return final_center, coarse_results_data


# ==============================================================================
# SCRIPT DI TEST
# ==============================================================================
def run_multiple_inversion_tests(num_tests=5, seed=42):
    if seed is not None: np.random.seed(seed)
        
    print(f"=========================================================")
    print(f"🚀 STARTING ROBUSTNESS EVALUATION (IBRIDO BLINDATO)")
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
        estimated_center, coarse_results_data = GroupSolutionCP1_Hybrid_Parallel(
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
        print(f"Mean Error: {np.mean([r['error'] for r in results_summary]):.4f} m")

if __name__ == "__main__":
    run_multiple_inversion_tests(num_tests=5, seed=42)