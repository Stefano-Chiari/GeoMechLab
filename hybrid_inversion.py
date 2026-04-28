import os
import sys
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout

# Import dal vostro progetto
from create_grid import create_layered_grid_with_hole
from common_functions import solve_and_export

# ==============================================================================
# FUNZIONI DI SUPPORTO
# ==============================================================================
def extract_displacement(sd, u, target_point=(0.0, 1.0)):
    """Trova il nodo più vicino al target e restituisce lo spostamento [ux, uy]."""
    dist_sq = (sd.nodes[0, :] - target_point[0])**2 + (sd.nodes[1, :] - target_point[1])**2
    idx = np.argmin(dist_sq)
    return np.array([u[0, idx], u[1, idx]])

def hybrid_evaluation_cost_function(center, measured_disp, export_folder):
    """
    Funzione di costo IBRIDA:
    - Sinistra (x < 1.0): Mesh larga (Chloe) per velocità.
    - Destra (x >= 1.0): Mesh fitta (Edo/Ste) per precisione.
    """
    cx, cy = center
    
    # Penalità immediata se fuori dal dominio sicuro
    if cx < 0.25 or cx > 1.75 or cy < 0.25 or cy > 0.75:
        return 1e6

    # DECISIONE MESH IBRIDA
    mesh_size = 0.045 if cx < 1.0 else 0.025 

    try:
        # SILENZIAMENTO OUTPUT: Evita il caos nella console
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            sd, _ = create_layered_grid_with_hole(mesh_size=mesh_size, center=(cx, cy))
            
            # Usiamo il PID del processo per evitare sovrascritture dei file in parallelo
            u = solve_and_export(sd, export_folder, export_name=f"temp_inv_{os.getpid()}", cell_data=None)
            
            simulated_disp = extract_displacement(sd, u, target_point=(0.0, 1.0))
        
        # Errore (Distanza Euclidea)
        return np.linalg.norm(simulated_disp - np.array(measured_disp))
    except Exception:
        return 1e6

def parallel_worker(args):
    """Wrapper necessario per passare argomenti multipli al ProcessPoolExecutor."""
    return hybrid_evaluation_cost_function(*args)

# ==============================================================================
# ALGORITMO PRINCIPALE
# ==============================================================================
def GroupSolutionCP1_Hybrid_Parallel(measured_displacement):
    print("🚀 Avvio Inversione IBRIDA PARALLELA (Chloe + Edo/Ste)...")
    
    export_folder = Path.cwd() / "opt_temp_results"
    export_folder.mkdir(exist_ok=True)
    
    # --- FASE 1: Ricerca Globale (In Parallelo) ---
    print("--> FASE 1: Esplorazione Spaziale Parallela (8x5)...")
    x_vals = np.linspace(0.3, 1.7, 8)
    y_vals = np.linspace(0.3, 0.7, 5)
    search_points = [(cx, cy) for cx in x_vals for cy in y_vals]
    
    tasks = [(point, measured_displacement, export_folder) for point in search_points]
    
    # QUI IL PARALLELISMO REALE FUNZIONA BENE
    with ProcessPoolExecutor() as executor:
        errors = list(executor.map(parallel_worker, tasks))
    
    results = [{'center': p, 'error': e} for p, e in zip(search_points, errors)]
    results.sort(key=lambda x: x['error'])
    
    top_candidates = [res['center'] for res in results[:2]]
    print(f"    Trovate le 2 migliori partenze: {top_candidates}")

    # --- FASE 2: Ottimizzazione Locale Multi-Start ---
    print("--> FASE 2: Raffinamento Locale Multi-Start...")
    best_final_center = None
    best_final_error = float('inf')
    
    for idx, start_node in enumerate(top_candidates):
        print(f"    Esecuzione Ottimizzazione {idx+1}/2 da {start_node}...")
        
        tol_f = 1e-8 if start_node[0] >= 1.0 else 1e-6
        
        res = minimize(
            fun=hybrid_evaluation_cost_function,
            x0=start_node,
            args=(measured_displacement, export_folder),
            method='Nelder-Mead',
            options={
                'xatol': 0.005,
                'fatol': tol_f,
                'maxiter': 20
            }
        )
        
        if res.fun < best_final_error:
            best_final_error = res.fun
            best_final_center = tuple(res.x)
            
        print(f"      Errore spostamento parziale: {res.fun:.4e}")

    print(f"\n✅ Inversione conclusa! Posizione stimata: {best_final_center}")
    return best_final_center

# ==============================================================================
# SCRIPT DI TEST
# ==============================================================================
def run_hybrid_tests(num_tests=5):
    print("="*60)
    print(f"🔬 INIZIO TEST ROBUSTEZZA SU {num_tests} CAVITÀ (SCRIPT .PY)")
    print("="*60)
    
    export_folder = Path.cwd() / "test_multiple_results"
    export_folder.mkdir(exist_ok=True)
    target_point = (0.0, 1.0)
    results_summary = []
    
    for i in range(num_tests):
        print(f"\n" + "-"*50)
        print(f"🔍 TEST {i+1} DI {num_tests}")
        print("-" * 50)
        
        true_center = (np.random.uniform(0.3, 1.7), np.random.uniform(0.3, 0.7))
        print(f"1️⃣ Creazione Realtà Nascosta in {true_center}...")
        
        try:
            with open(os.devnull, 'w') as f, redirect_stdout(f):
                sd_true, cell_layer_true = create_layered_grid_with_hole(mesh_size=0.025, center=true_center)
                u_true = solve_and_export(sd_true, export_folder, export_name=f"true_{i+1}", cell_data=None)
                measured_displacement = extract_displacement(sd_true, u_true, target_point=target_point)
        except Exception as e:
            print(f"⚠️ Errore mesher (salto il test): {e}")
            continue

        print("2️⃣ Avvio Algoritmo di Inversione...")
        estimated_center = GroupSolutionCP1_Hybrid_Parallel(tuple(measured_displacement))
        
        dist_totale = np.linalg.norm(np.array(true_center) - np.array(estimated_center))
        
        print(f"\n📊 RISULTATI TEST {i+1}:")
        print(f"   Centro REALE:   ({true_center[0]:.4f}, {true_center[1]:.4f})")
        print(f"   Centro STIMATO: ({estimated_center[0]:.4f}, {estimated_center[1]:.4f})")
        print(f"   Errore Spaziale: {dist_totale:.4f} m")
        
        results_summary.append({"test_id": i+1, "true": true_center, "est": estimated_center, "error": dist_totale})

    print("\n" + "="*50)
    print("📈 REPORT FINALE VALUTAZIONE")
    print("="*50)
    for res in results_summary:
        status = "✅ SUPERATO" if res['error'] < 0.08 else "⚠️ IMPRECISO"
        print(f"Test {res['test_id']}: Reale ({res['true'][0]:.3f}, {res['true'][1]:.3f}) | Stimato ({res['est'][0]:.3f}, {res['est'][1]:.3f}) | Errore: {res['error']:>6.4f}m | {status}")
    
    if results_summary:
        print("-" * 50)
        print(f"Errore Medio:   {np.mean([res['error'] for res in results_summary]):.4f} m")
        print("=" * 50)

# ==============================================================================
# ENTRY POINT SICURO PER MULTIPROCESSING
# ==============================================================================
if __name__ == "__main__":
    run_hybrid_tests(num_tests=5)