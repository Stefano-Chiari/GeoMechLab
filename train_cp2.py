import os
import tempfile
import numpy as np
import multiprocessing as mp
import joblib
from pathlib import Path
from scipy.stats import qmc
from scipy.interpolate import griddata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
import warnings
import shutil

warnings.filterwarnings("ignore")

# Assicurati che queste importazioni corrispondano ai tuoi file
from create_grid import create_layered_grid_with_hole
from common_functions import solve_and_export_pressure

def worker_fem_cp2(args):
    cx, cy, layer_ID, pressure_value = args
    mesh_size = 0.025  
    
    original_cwd = os.getcwd()
    
    # 1. Creazione manuale della cartella temporanea
    tmpdir = tempfile.mkdtemp() 
    
    try:
        os.chdir(tmpdir)

        from contextlib import redirect_stdout
        
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            # --- Inizio Codice PorePy/Gmsh ---
            sd, cell_layer = create_layered_grid_with_hole(
                mesh_size=mesh_size, 
                center=(cx, cy),
                axes=(0.1, 0.05) # Il buco rimpicciolito
            )
            
            u = solve_and_export_pressure(
                sd, cell_layer, layer_ID, pressure_value,
                folder_export=Path(tmpdir), export_name="temp_cp2",
                cell_data=[("layer", cell_layer.astype(float))]
            )
            
            nodes = np.column_stack((sd.nodes[0, :], sd.nodes[1, :]))

            # Estrazione Superficie
            y_max = float(np.max(sd.nodes[1, :]))
            mask = np.abs(sd.nodes[1, :] - y_max) < 1e-6
            x_s = sd.nodes[0, mask]
            ux_s = u[0, mask]
            uy_s = u[1, mask]

            # Ordinamento 
            order = np.argsort(x_s)
            x_s = x_s[order]
            ux_s = ux_s[order]
            uy_s = uy_s[order]

            # Interpolazione
            x_eval = np.linspace(0.0, 2.0, 50)
            from scipy.interpolate import interp1d
            f_ux = interp1d(x_s, ux_s, kind='linear', fill_value="extrapolate")
            f_uy = interp1d(x_s, uy_s, kind='linear', fill_value="extrapolate")
            ux_surf = f_ux(x_eval)
            uy_surf = f_uy(x_eval)

            # Estrazione Angolo a (0, 1)
            d2 = sd.nodes[0, :]**2 + (sd.nodes[1, :] - 1.0)**2
            idx = int(np.argmin(d2))
            ux_corner = float(u[0, idx])
            uy_corner = float(u[1, idx])
            # --- Fine Codice PorePy/Gmsh ---

        os.chdir(original_cwd)
        
        # 2. Pulizia manuale sicura (ignora il blocco di Windows!)
        shutil.rmtree(tmpdir, ignore_errors=True)
        
        return (cx, cy, ux_surf, uy_surf, ux_corner, uy_corner)

    except Exception as e:
        # Se siamo qui, il crash è VERO (es. Gmsh non è riuscito a fare la mesh)
        print(f"⚠️ Gmsh/PorePy vero crash a (cx={cx:.2f}, cy={cy:.2f}) - Errore: {e}")
        os.chdir(original_cwd)
        
        # Pulizia sicura anche in caso di errore
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None
    
def train_and_save_global_emulator(layer_ID, N_samples=200):
    """
    Genera i dati ed addestra un Processo Gaussiano Globale per un dato layer.
    Salva il modello su disco.
    """
    print(f"\n🚀 Inizio generazione dati per Layer {layer_ID}...")

    # 1. Definizione dei limiti fisici GLOBALI [cx, cy, pressione]
    margin = 0.1 
    bounds_lower = [0.0 + margin, 0.0 + margin, -0.5] # Pressione Minima: -0.5
    bounds_upper = [2.0 - margin, 1.0 - margin,  0.5] # Pressione Massima: +0.5

    # 2. Design of Experiments (LHS) a 3 dimensioni
    sampler = qmc.LatinHypercube(d=3, seed=42 + layer_ID) # Seed dinamico per variare i campioni tra layer
    sample = sampler.random(n=N_samples)
    
    # Scaliamo i campioni nei limiti fisici reali
    design_points = qmc.scale(sample, bounds_lower, bounds_upper)

    # Creiamo i task per il worker: (cx, cy, layer_ID, pressure)
    tasks = [(pt[0], pt[1], layer_ID, pt[2]) for pt in design_points]

    # 3. Simulazione Parallela
    print(f"   -> Avvio {N_samples} simulazioni FEM...")
    context = mp.get_context("spawn")
    with context.Pool(processes=min(mp.cpu_count(), 8), maxtasksperchild=1) as pool:
        results = pool.map(worker_fem_cp2, tasks)

    # Filtriamo eventuali simulazioni fallite
    valid_results = [r for r in results if r is not None and not np.isnan(r[4])]
    print(f"   -> Simulazioni completate con successo: {len(valid_results)}/{N_samples}")

    if len(valid_results) < N_samples * 0.8:
        print("⚠️ Attenzione: troppe simulazioni fallite. Controlla il raggio della cavità e il margine ai bordi.")

 # 4. Preparazione Dati per l'Addestramento
    valid_X = []
    valid_Y_ux = []
    valid_Y_uy = []
    valid_Y_corner = []

    # Iteriamo simultaneamente sui task originali e sui risultati
    for task, res in zip(tasks, results):
        if res is not None and not np.isnan(res[4]):
            # task = (cx, cy, layer_ID, pressure)
            # res  = (cx, cy, ux_surf, uy_surf, ux_corner, uy_corner)
            valid_X.append([task[0], task[1], task[3]]) # cx, cy, pressure
            valid_Y_ux.append(res[2])
            valid_Y_uy.append(res[3])
            valid_Y_corner.append([res[4], res[5]])

    X_train = np.array(valid_X)
    Y_ux_surf = np.array(valid_Y_ux)
    Y_uy_surf = np.array(valid_Y_uy)
    Y_corner = np.array(valid_Y_corner)

    print(f"   -> Dimensioni X_train: {X_train.shape[0]}, Dimensioni Y: {Y_ux_surf.shape[0]}")

    # 5. Definizione e Addestramento dei Modelli
    print("   -> Addestramento Processi Gaussiani in corso...")
    
    # KERNEL MATERN: Ideale per emulazione fisica FEM
    kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[1.0, 1.0, 1.0], length_scale_bounds=(1e-2, 1e2), nu=2.5)
    
    # Riportiamo alpha a 1e-5 per non smussare i picchi e aumentiamo i restart a 10
    gp_ux_surf = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True, alpha=1e-5)
    gp_uy_surf = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True, alpha=1e-5)
    
    gp_corner_ux = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True, alpha=1e-3)
    gp_corner_uy = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True, alpha=1e-3)

    gp_ux_surf.fit(X_train, Y_ux_surf)
    gp_uy_surf.fit(X_train, Y_uy_surf)
    
    gp_corner_ux.fit(X_train, Y_corner[:, 0])
    gp_corner_uy.fit(X_train, Y_corner[:, 1])

    # 6. Salvataggio su Disco
    print("   -> Salvataggio modelli su disco...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(gp_ux_surf, f"models/gp_ux_surf_layer_{layer_ID}.pkl")
    joblib.dump(gp_uy_surf, f"models/gp_uy_surf_layer_{layer_ID}.pkl")
    joblib.dump(gp_corner_ux, f"models/gp_corner_ux_layer_{layer_ID}.pkl")
    joblib.dump(gp_corner_uy, f"models/gp_corner_uy_layer_{layer_ID}.pkl")

    print(f"✅ Addestramento e salvataggio Layer {layer_ID} completato!")

if __name__ == '__main__':
    # I layer geologici reali vanno da 0 a 3!
    for layer in range(0, 4):
        train_and_save_global_emulator(layer_ID=layer, N_samples=400)