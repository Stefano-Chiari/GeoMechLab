import os
import time
import numpy as np
import multiprocessing as mp
from pathlib import Path
from scipy.optimize import minimize
from contextlib import redirect_stdout
import warnings

# Spegne i fastidiosi warning rossi di PorePy
warnings.filterwarnings("ignore")

from create_grid import create_layered_grid_with_hole
from common_functions import solve_and_export

# ==============================================================================
# FUNZIONI DI COSTO E SUPPORTO
# ==============================================================================
def extract_displacement_interpolated(sd, u, target_point=(0.0, 1.0)):
    """Trova il nodo più vicino al target e restituisce lo spostamento."""
    dist_sq = (sd.nodes[0, :] - target_point[0])**2 + (sd.nodes[1, :] - target_point[1])**2
    idx = np.argmin(dist_sq)
    return np.array([u[0, idx], u[1, idx]])

def worker_cost_function(args):
    center, measured_disp, export_folder, target_point, phase, side = args
    cx, cy = center

    # 🚧 GABBIA: Impediamo all'algoritmo di scappare dal suo lato
    if side == 'left' and cx >= 1.0: return 1e6
    if side == 'right' and cx < 1.0: return 1e6
    if cy < 0.25 or cy > 0.75 or cx < 0.25 or cx > 1.75: return 1e6

    # ⚖️ LA REGOLA EQUA:
    # Fase 1: Chloe (0.045) a sx per volare, Edo (0.025) a dx.
    # Fase 2: TUTTI usano 0.025 per non far "barare" la matematica.
    if phase == 1:
        mesh_size = 0.045 if side == 'left' else 0.025
    else:
        mesh_size = 0.025

    try:
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            sd, _ = create_layered_grid_with_hole(mesh_size=mesh_size, center=(cx, cy))
            # Nome univoco per evitare che processi simultanei scrivano sullo stesso file
            u = solve_and_export(sd, export_folder, export_name=f"temp_{os.getpid()}_{time.time()}", cell_data=None)
            simulated_disp = extract_displacement_interpolated(sd, u, target_point=target_point)
            
        return np.linalg.norm(np.array(simulated_disp) - np.array(measured_disp))
    except Exception:
        return 1e6

def nm_cost_function(center, measured_disp, export_folder, target_point, phase, side):
    """Wrapper per scipy minimize"""
    return worker_cost_function((center, measured_disp, export_folder, target_point, phase, side))

def optimize_worker_phase2(args):
    """Worker per eseguire il Nelder-Mead in parallelo in Fase 2"""
    start_node, measured_disp, export_folder, target_point, side = args
    
    # Tolleranze ibride: esigenti a destra, rilassate a sinistra
    tol_f = 1e-6 if side == 'left' else 1e-8
    
    res = minimize(
        fun=nm_cost_function, x0=start_node,
        args=(measured_disp, export_folder, target_point, 2, side),
        method='Nelder-Mead', 
        options={'xatol': 0.005, 'fatol': tol_f, 'maxiter': 15}
    )
    # Restituiamo il lato, l'errore finale e il centro calcolato
    return side, res.fun, tuple(res.x)


# ==============================================================================
# ALGORITMO IBRIDO PRINCIPALE
# ==============================================================================
def GroupSolutionCP1_Hybrid_Parallel(measured_displacement, test_id=1, target_point=(0.0, 1.0)):
    print(f"\n🚀 Avvio Inversione IBRIDA CHLOE/EDO (Test {test_id})")
    export_folder = Path.cwd() / "opt_temp_results"
    export_folder.mkdir(exist_ok=True)
    t0_total = time.perf_counter()

    # --- FASE 1: Ricerca Globale Separata ---
    print("--> FASE 1: Esplorazione Parallela...")
    t0_f1 = time.perf_counter()
    
    x_vals_left = np.linspace(0.3, 0.9, 4)
    x_vals_right = np.linspace(1.1, 1.7, 4)
    y_vals = np.linspace(0.3, 0.7, 5)
    
    tasks = []
    for cx in x_vals_left:
        for cy in y_vals: tasks.append(((cx, cy), measured_displacement, export_folder, target_point, 1, 'left'))
    for cx in x_vals_right:
        for cy in y_vals: tasks.append(((cx, cy), measured_displacement, export_folder, target_point, 1, 'right'))

    # 🛡️ PROTEZIONE ANTI-FREEZE: "spawn" e "maxtasksperchild=1" resettano la memoria di Gmsh
    context = mp.get_context("spawn")
    # Usa al massimo 8 core per non fondere il Mac
    cores_to_use = min(os.cpu_count(), 8) 
    with context.Pool(processes=cores_to_use, maxtasksperchild=1) as pool:
        errors = pool.map(worker_cost_function, tasks)

    t1_f1 = time.perf_counter()
    print(f"    ⏱️ TEMPO FASE 1: {(t1_f1 - t0_f1)/60:.2f} minuti")

    res_left = [{'center': t[0], 'error': e} for t, e in zip(tasks, errors) if t[5] == 'left']
    res_right = [{'center': t[0], 'error': e} for t, e in zip(tasks, errors) if t[5] == 'right']
    res_left.sort(key=lambda x: x['error'])
    res_right.sort(key=lambda x: x['error'])
    
    best_left = res_left[0]['center']
    best_right = res_right[0]['center']

    # --- FASE 2 & 3: Ottimizzazione Locale e Giudizio (IN PARALLELO) ---
    print(f"--> FASE 2: Nelder-Mead PARALLELO (Sinistra: {best_left} | Destra: {best_right})")
    t0_f2 = time.perf_counter()
    
    tasks_p2 = [
        (best_left, measured_displacement, export_folder, target_point, 'left'),
        (best_right, measured_displacement, export_folder, target_point, 'right')
    ]
    
    # Usiamo 2 processi per fare sinistra e destra in contemporanea esatta!
    with context.Pool(processes=2, maxtasksperchild=1) as pool:
        results_p2 = pool.map(optimize_worker_phase2, tasks_p2)

    t1_f2 = time.perf_counter()
    print(f"    ⏱️ TEMPO FASE 2: {(t1_f2 - t0_f2)/60:.2f} minuti")
        
    # Estraiamo i risultati del Giudice
    err_left_fair = [r[1] for r in results_p2 if r[0] == 'left'][0]
    center_left = [r[2] for r in results_p2 if r[0] == 'left'][0]
    
    err_right_fair = [r[1] for r in results_p2 if r[0] == 'right'][0]
    center_right = [r[2] for r in results_p2 if r[0] == 'right'][0]

    print("--> FASE 3: Valutazione Finale Equa...")
    print(f"    Errore VERO Sinistra (Chloe): {err_left_fair:.4e}")
    print(f"    Errore VERO Destra (Edo):     {err_right_fair:.4e}")

    if err_left_fair < err_right_fair:
        final_center = center_left
        print("    🏆 Vince l'approccio di Chloe (Sinistra)")
    else:
        final_center = center_right
        print("    🏆 Vince l'approccio di Edo (Destra)")

    t1_total = time.perf_counter()
    print(f"✅ Inversione completata in {(t1_total - t0_total)/60:.2f} minuti totali.")
    
    coarse_results_data = [{'center': t[0], 'error': e} for t, e in zip(tasks, errors)]
    return final_center, coarse_results_data


# ==============================================================================
# SCRIPT DI TEST
# ==============================================================================
def run_multiple_inversion_tests(num_tests=5, seed=42):
    if seed is not None: np.random.seed(seed)
        
    print(f"=========================================================")
    print(f"🚀 STARTING ROBUSTNESS EVALUATION (IBRIDO BLINDATO PARALLELO)")
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
            print(f"Errore creazione GT: {e}")
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
        print("-" * 50)
        print(f"Mean Error:    {np.mean([r['error'] for r in results_summary]):.4f} m")
        print(f"Maximum Error: {np.max([r['error'] for r in results_summary]):.4f} m")
        print("=" * 50)

if __name__ == "__main__":
    run_multiple_inversion_tests(num_tests=5, seed=42)