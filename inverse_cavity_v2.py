from __future__ import annotations

import warnings
import numpy as np
import pandas as pd  # <-- Nuovo
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Sequence
from scipy.spatial import KDTree  # <-- Nuovo
from scipy.interpolate import RBFInterpolator  # <-- Nuovo

from create_grid import create_layered_grid_with_hole
from common_functions import solve_and_export

# ---------------------------------------------------------------------------
# Domain constants (invariati)
# ---------------------------------------------------------------------------
LX: float = 2.0
LY: float = 1.0
AXES: tuple[float, float] = (0.2, 0.1)
OBS_NODE_INDEX: int = 60
MESH_COARSE: float = 0.08
MESH_FINE: float = 0.025

MARGIN_X: float = AXES[0] + 0.05
MARGIN_Y: float = AXES[1] + 0.05

# ---------------------------------------------------------------------------
# Caricamento Globale del Database (Online Setup)
# ---------------------------------------------------------------------------
DB_PATH = Path("cavity_db.csv")

if DB_PATH.exists():
    # Carica i dati precalcolati offline
    _db_df = pd.read_csv(DB_PATH)
    # Matrice delle feature (input dell'inversione): forma (N, 2) -> [ux, uy]
    _db_measurements = _db_df[['ux', 'uy']].values
    # Matrice dei target (output dell'inversione): forma (N, 2) -> [center_x, center_y]
    _db_centers = _db_df[['center_x', 'center_y']].values
    # Costruzione del KDTree nello spazio degli spostamenti per ricerca O(log N)
    _kdtree = KDTree(_db_measurements)
else:
    _db_df = None
    _db_measurements = None
    _db_centers = None
    _kdtree = None

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


def invert_cavity_db(
    measured_displacement: tuple[float, float], 
    k_neighbors: int = 6
) -> tuple[float, float]:
    """
    Risolve il problema inverso stimando il centro della cavità tramite 
    ricerca per vicini (KDTree) e interpolazione locale (RBF) su database.

    Parameters
    ----------
    measured_displacement : tuple[float, float]
        Spostamento osservato (ux, uy) nel punto P=(0,1).
    k_neighbors : int
        Numero di nodi vicini da usare per l'interpolazione locale.

    Returns
    -------
    tuple[float, float]
        Coordinate (x, y) stimate del centro della cavità.
    """
    if _kdtree is None or _db_df is None:
        raise FileNotFoundError(
            "Il database 'cavity_db.csv' non è stato trovato nella directory corrente. "
            "Esegui lo script di generazione offline prima di avviare l'inversione."
        )

    # Conversione dell'input in array NumPy per l'interrogazione geometrica
    meas_arr = np.array(measured_displacement, dtype=float).reshape(1, -1)
    
    # 1. Ricerca O(log N) dei K vicini più prossimi nello spazio degli spostamenti
    distances, indices = _kdtree.query(meas_arr, k=k_neighbors)
    indices = indices[0]  # Estrazione degli indici monodimensionali
    
    # Se viene richiesto un solo vicino, restituisce direttamente il punto di griglia
    if k_neighbors == 1:
        best_center = _db_centers[indices]
        return float(best_center[0]), float(best_center[1])

    # 2. Estrazione del subset locale di dati per l'interpolazione
    local_ux_uy = _db_measurements[indices]  # Coordinate nello spazio delle feature
    local_cx_cy = _db_centers[indices]       # Coordinate nello spazio fisico (target)
    
    # 3. Costruzione degli interpolatori RBF locali per le componenti x e y del centro.
    # L'uso del logaritmo o dello smoothing non è strettamente necessario se il subset 
    # locale è limitato e non presenta singolarità.
    rbf_x = RBFInterpolator(local_ux_uy, local_cx_cy[:, 0], kernel='thin_plate_spline')
    rbf_y = RBFInterpolator(local_ux_uy, local_cx_cy[:, 1], kernel='thin_plate_spline')
    
    # 4. Valutazione dell'interpolatore nel punto esatto della misura reale
    est_x = rbf_x(meas_arr)[0]
    est_y = rbf_y(meas_arr)[0]
    
    # Vincolo di sicurezza: assicura che l'output non esca dai margini fisici del dominio
    est_x = np.clip(est_x, MARGIN_X, LX - MARGIN_X)
    est_y = np.clip(est_y, MARGIN_Y, LY - MARGIN_Y)
    
    return float(est_x), float(est_y)


def GroupSolutionCP1(
    measured_displacement: tuple[float, float],
) -> tuple[float, float]:
    """
    Entry point richiesto dalle specifiche. Risolve l'inversione in frazioni
    di millisecondo interrogando il database precalcolato.
    """
    # Un valore di k_neighbors compreso tra 5 e 8 è matematicamente ideale 
    # per catturare la curvatura locale senza introdurre oscillazioni (runaway)
    return invert_cavity_db(measured_displacement, k_neighbors=6)