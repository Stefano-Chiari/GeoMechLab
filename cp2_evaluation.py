import os
import tempfile
import numpy as np
import multiprocessing as mp
import sklearn
from pathlib import Path
from scipy.stats import qmc
from scipy.interpolate import griddata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings
from scipy.interpolate import interp1d

warnings.filterwarnings("ignore")

from create_grid import create_layered_grid_with_hole
from common_functions import solve_and_export_pressure

def worker_fem_cp2(args):
    """
    Worker parallelo: esegue una singola simulazione FEM per un dato centro (cx, cy).
    Lavora in una cartella sandbox temporanea per evitare crash di Gmsh.
    ----
    Parallel Worker: do a single FEM simulation for a given center (cx, cy)
    Works in a temporary sandobx folder
    """
    cx, cy, layer_ID, pressure_value = args
    mesh_size = 0.025  
    
    original_cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            from contextlib import redirect_stdout
            with open(os.devnull, 'w') as f, redirect_stdout(f):
                # 1. Creazione Mesh / Mesh generation
                sd, cell_layer = create_layered_grid_with_hole(mesh_size=mesh_size, center=(cx, cy))
                
                # 2. Risoluzione (Assumiamo che ritorni il vettore spostamento u) / Resolution 
                u = solve_and_export_pressure(
                    sd, cell_layer, layer_ID, pressure_value,
                    folder_export=Path(tmpdir), export_name="temp_cp2",
                    cell_data=[("layer", cell_layer.astype(float))]
                )
                
                # Se la funzione non ritorna u direttamente ma lo salva in sd, puoi estrarlo con:
                # u = sd.point_data["u"] (adattalo in base all'implementazione interna di PorePy)

                nodes = np.column_stack((sd.nodes[0, :], sd.nodes[1, :]))

                # 3. Estrazione Profilo Superficiale (y = 1.0)
                x_eval = np.linspace(0.0, 2.0, 50) # Valuta 50 punti lungo la superficie
                target_surface = np.column_stack((x_eval, np.ones_like(x_eval) * 1.0))
                
                ux_surf = griddata(nodes, u[0, :], target_surface, method='linear')
                uy_surf = griddata(nodes, u[1, :], target_surface, method='linear')

                # 4. Estrazione Angolo in alto a sinistra (0.0, 1.0)
                corner_point = np.array([[0.0, 1.0]])
                ux_corner = griddata(nodes, u[0, :], corner_point, method='linear')[0]
                uy_corner = griddata(nodes, u[1, :], corner_point, method='linear')[0]

            os.chdir(original_cwd)
        return (cx, cy, ux_surf, uy_surf, ux_corner, uy_corner)
    except Exception as e:
        os.chdir(original_cwd)
        return None

def GroupSolutionCP2(layer_ID, pressure_value, center_location_mean, center_location_std):
    """
    Costruisce l'emulatore e calcola gli output richiesti usando Gaussian Processes.
    Build up the emulator and compute outputs using Gaussian Processes
    """
    # --- VALIDAZIONE RIGOROSA DEGLI INPUT FISICI E STATISTICI ---
    if not (-0.5 <= pressure_value <= 0.5):
        raise ValueError(f"Errore Fisico: Il valore di pressione ({pressure_value}) "
                         f"deve essere tassativamente compreso in [-0.5, 0.5].")
                         
    std_x, std_y = center_location_std
    if not (0.01 <= std_x <= 0.045) or not (0.01 <= std_y <= 0.045):
        raise ValueError(f"Errore Statistico: Le deviazioni standard ({std_x}, {std_y}) "
                         f"devono essere comprese nell'intervallo [0.01, 0.045].")
    # ------------------------------------------------------------

    print("🚀 Inizio Costruzione Emulatore CP2...")
    mu_x, mu_y = center_location_mean
    std_x, std_y = center_location_std

    # =========================================================================
    # FASE 1: Design of Experiments (LHS)
    # =========================================================================
    print("   -> Generazione dataset di addestramento (LHS)...")
    
    N_samples = 50 # 50 simulazioni sono sufficienti per un ottimo surrogato 2D
    
    sampler = qmc.LatinHypercube(d=2, seed=42)
    sample = sampler.random(n=N_samples)

    # Scaliamo il campione in un range di +/- 3 deviazioni standard
    l_bounds = [max(0.3, mu_x - 3*std_x), max(0.3, mu_y - 3*std_y)]
    u_bounds = [min(1.7, mu_x + 3*std_x), min(0.7, mu_y + 3*std_y)]
    centers = qmc.scale(sample, l_bounds, u_bounds)

    tasks = [(cx, cy, layer_ID, pressure_value) for cx, cy in centers]

    # =========================================================================
    # FASE 2: Simulazione Parallela (Raccolta Dati Reali)
    # PHASE 2: Parallel Simulation
    # =========================================================================
    print(f"   -> Avvio {N_samples} simulazioni FEM in parallelo...")
    context = mp.get_context("spawn")
    with context.Pool(processes=min(mp.cpu_count(), 8), maxtasksperchild=1) as pool:
        results = pool.map(worker_fem_cp2, tasks)

    valid_results = [r for r in results if r is not None and not np.isnan(r[4])]
    print(f"   -> Simulazioni completate con successo: {len(valid_results)}/{N_samples}")

    X_train = np.array([[r[0], r[1]] for r in valid_results])
    Y_ux_surf = np.array([r[2] for r in valid_results]) # Shape: (N, 50)
    Y_uy_surf = np.array([r[3] for r in valid_results]) # Shape: (N, 50)
    Y_corner = np.array([[r[4], r[5]] for r in valid_results]) # Shape: (N, 2)

    # =========================================================================
    # FASE 3: Addestramento Emulatore (Gaussian Process) / Emulator training (Guassian Process)
    # =========================================================================
    print("   -> Addestramento Processi Gaussiani...")
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (0.2, 1e2))
    
    gp_ux_surf = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True, alpha=1e-5)
    gp_uy_surf = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True, alpha=1e-5)
    gp_corner = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True, alpha=1e-5)

    gp_ux_surf.fit(X_train, Y_ux_surf)
    gp_uy_surf.fit(X_train, Y_uy_surf)
    gp_corner.fit(X_train, Y_corner)

    # =========================================================================
    # FASE 4: Propagazione Incertezza (Monte Carlo su Emulatore) / Monte Carlo on Emulator
    # =========================================================================
    print("   -> Propagazione Incertezza (Monte Carlo su 10.000 campioni)...")
    M_mc = 10000
    MC_samples_x = np.random.normal(mu_x, std_x, M_mc)
    MC_samples_y = np.random.normal(mu_y, std_y, M_mc)
    X_mc = np.column_stack((MC_samples_x, MC_samples_y))

    # Filtriamo i punti MC che cadono fuori dal dominio fisico
    mask = (X_mc[:,0] > 0.2) & (X_mc[:,0] < 1.8) & (X_mc[:,1] > 0.2) & (X_mc[:,1] < 0.8)
    X_mc = X_mc[mask]

    # Interroghiamo l'emulatore (è istantaneo!)
    ux_surf_pred = gp_ux_surf.predict(X_mc)
    uy_surf_pred = gp_uy_surf.predict(X_mc)
    corner_pred = gp_corner.predict(X_mc)

    # =========================================================================
    # FASE 5: Calcolo Statistiche Finali / Computing final statistics
    # =========================================================================
    # 1. Profilo medio della superficie (vettori discreti)
    x_eval = np.linspace(0.0, 2.0, 50)
    mean_ux_surf = np.mean(ux_surf_pred, axis=0)
    mean_uy_surf = np.mean(uy_surf_pred, axis=0)

    # Creiamo le funzioni di interpolazione continue
    # bounds_error=False e fill_value="extrapolate" ci assicurano che se il prof 
    # dovesse chiedere un valore leggermente fuori dal dominio (es x=2.01), 
    # il codice non vada in errore.
    interp_ux = interp1d(x_eval, mean_ux_surf, kind='cubic', bounds_error=False, fill_value="extrapolate")
    interp_uy = interp1d(x_eval, mean_uy_surf, kind='cubic', bounds_error=False, fill_value="extrapolate")

    # Creiamo la funzione f(x) che restituisce [ux(x), uy(x)]
    def mean_surface_displacement_func(x):
        """
        Funzione continua che valuta lo spostamento medio della superficie.
        Input: x (float o array) - Coordinata orizzontale sulla superficie (m).
        Output: [Ux(x), Uy(x)] - Vettore degli spostamenti.
        """
        return [float(interp_ux(x)), float(interp_uy(x))]

    # Il valore da ritornare ora è la funzione stessa
    mean_surface_displacement = mean_surface_displacement_func

    # 2. Intervallo di confidenza 90% (percentili 5° e 95°) all'angolo
    ux_corner_5 = np.percentile(corner_pred[:, 0], 5)
    ux_corner_95 = np.percentile(corner_pred[:, 0], 95)
    
    uy_corner_5 = np.percentile(corner_pred[:, 1], 5)
    uy_corner_95 = np.percentile(corner_pred[:, 1], 95)

    CI_corner = [(ux_corner_5, ux_corner_95), (uy_corner_5, uy_corner_95)]

    print(f"✅ Calcolo completato. CI Corner Ux: ({ux_corner_5:.4e}, {ux_corner_95:.4e})")
    return mean_surface_displacement, CI_corner