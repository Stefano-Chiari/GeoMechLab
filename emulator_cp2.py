import numpy as np
import joblib
from scipy.interpolate import PchipInterpolator

def GroupSolutionCP2(layer_ID, pressure_value, center_location_mean, center_location_std):
    """
    Emulatore Geomeccanico con UQ (Uncertainty Quantification) robusta 
    per l'ingegneria del sottosuolo.
    """
    # --- 1. VALIDAZIONE INPUT ---
    std_x, std_y = center_location_std
    if not (0.01 <= std_x <= 0.045) or not (0.01 <= std_y <= 0.045):
        raise ValueError(f"Errore: Le deviazioni standard ({std_x}, {std_y}) "
                         f"devono essere rigorosamente nell'intervallo [0.01, 0.045].")
    
    # --- 2. CARICAMENTO DEI 4 MODELLI SDOPPIATI ---
    try:
        gp_ux_surf   = joblib.load(f"models/gp_ux_surf_layer_{layer_ID}.pkl")
        gp_uy_surf   = joblib.load(f"models/gp_uy_surf_layer_{layer_ID}.pkl")
        gp_corner_ux = joblib.load(f"models/gp_corner_ux_layer_{layer_ID}.pkl")
        gp_corner_uy = joblib.load(f"models/gp_corner_uy_layer_{layer_ID}.pkl")
    except FileNotFoundError:
        raise FileNotFoundError(f"Modelli mancanti per Layer {layer_ID}.")

    # --- 3. PREPARAZIONE DATI MONTE CARLO ---
    N_mc = 2000
    cx_mean, cy_mean = center_location_mean
    
    # Campionamento posizioni (Incertezza Aleatoria)
    cx_samples = np.random.normal(cx_mean, std_x, N_mc)
    cy_samples = np.random.normal(cy_mean, std_y, N_mc)
    cx_samples = np.clip(cx_samples, 0.1, 1.9)
    cy_samples = np.clip(cy_samples, 0.1, 0.9)
    
    p_samples = np.full(N_mc, pressure_value)
    X_pred = np.column_stack((cx_samples, cy_samples, p_samples))

    # --- 4. PREDIZIONE SUPERFICIE ESATTA ---
    X_exact = np.array([[cx_mean, cy_mean, pressure_value]])
    exact_ux_surf = gp_ux_surf.predict(X_exact)[0]
    exact_uy_surf = gp_uy_surf.predict(X_exact)[0]
    
    x_eval = np.linspace(0.0, 2.0, 50)
    interp_ux = PchipInterpolator(x_eval, exact_ux_surf)
    interp_uy = PchipInterpolator(x_eval, exact_uy_surf)
    
    def mean_surface_displacement(x):
        return [float(interp_ux(x)), float(interp_uy(x))]

    # --- 5. PREDIZIONE ANGOLO (UQ Reale con Surrogate Error) ---
    mean_cx, std_cx = gp_corner_ux.predict(X_pred, return_std=True)
    mean_cy, std_cy = gp_corner_uy.predict(X_pred, return_std=True)
    
    # ERRORE DEL SURROGATO (Model Discrepancy)
    # È qui che dichiariamo matematicamente che l'emulatore ha un gap rispetto al FEM.
    # Usiamo un approccio iper-robusto: 8% di errore relativo + 1 millimetro fisso (0.001)
    surrogate_error_x = 0.08 * np.abs(mean_cx) + 0.001
    surrogate_error_y = 0.08 * np.abs(mean_cy) + 0.001
    
    # Uniamo l'errore statistico puro con la discrepanza del modello
    std_cx_adj = np.sqrt(std_cx**2 + surrogate_error_x**2)
    std_cy_adj = np.sqrt(std_cy**2 + surrogate_error_y**2)
    
    ux_corner_simulated = np.random.normal(mean_cx, std_cx_adj)
    uy_corner_simulated = np.random.normal(mean_cy, std_cy_adj)
    
    # --- 6. CALCOLO INTERVALLI DI CONFIDENZA (90%) ---
    ux_5 = np.percentile(ux_corner_simulated, 5)
    ux_95 = np.percentile(ux_corner_simulated, 95)
    
    uy_5 = np.percentile(uy_corner_simulated, 5)
    uy_95 = np.percentile(uy_corner_simulated, 95)
    
    CI_corner = [(ux_5, ux_95), (uy_5, uy_95)]

    return mean_surface_displacement, CI_corner