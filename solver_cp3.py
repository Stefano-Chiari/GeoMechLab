import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
# Sostituisci "emulator_utils" con il nome esatto del file creato prima (es. load_emulator_cp3)
from load_emulator_cp3 import predict_displacement 

def compute_loss(params, layer_id, top_u_obs, emulator):
    """
    Funzione obiettivo (Mean Squared Error).
    Calcola la discrepanza tra gli spostamenti misurati e quelli predetti dall'emulatore.
    """
    pressure, cx, cy = params
    
    # Ottieni la predizione (il GP restituisce solo la media, come impostato in emulator_utils)
    u_pred = predict_displacement(emulator, layer_id, pressure, cx, cy)
    
    # Assicuriamoci che le dimensioni combacino. 
    # Se top_u_obs è (2, N) per spostamenti X e Y, flatten() allinea tutto in un vettore 1D.
    error = np.mean((top_u_obs.flatten() - u_pred.flatten())**2)
    
    return error

def generate_lhs_samples(n_samples, bounds):
    """
    Genera campioni iniziali usando Latin Hypercube Sampling per esplorare lo spazio.
    """
    n_params = len(bounds)
    sampler = qmc.LatinHypercube(d=n_params, seed=42)
    sample = sampler.random(n=n_samples)
    
    # Scala i campioni da [0, 1] ai bound reali
    lower_bounds = np.array([b[0] for b in bounds])
    upper_bounds = np.array([b[1] for b in bounds])
    scaled_samples = qmc.scale(sample, lower_bounds, upper_bounds)
    
    return scaled_samples

def optimize_single_layer(layer_id, top_u_obs, emulator, spatial_bounds, n_starts=10, top_k=3):
    """
    Risolve il problema inverso continuo per un layer specifico, 
    usando una strategia Multi-Start.
    """
    # 1. Definizione dei vincoli (Bounds). La pressione è fissa in [-0.5, 0.5]
    # spatial_bounds è una tupla nel formato (x_min, x_max, y_min, y_max).
    pressure_bounds = (-0.5, 0.5)
    cx_bounds = (spatial_bounds[0], spatial_bounds[1])
    cy_bounds = (spatial_bounds[2], spatial_bounds[3])
    bounds = [pressure_bounds, cx_bounds, cy_bounds]  # (pressure, center_x, center_y)

    # 2. Generazione dei punti di partenza con LHS
    initial_guesses = generate_lhs_samples(n_starts, bounds)
    
    best_loss = np.inf
    best_params = None
    
    # 3. Pre-valutazione dei punti iniziali per scremare i peggiori
    initial_losses = []
    for guess in initial_guesses:
        loss = compute_loss(guess, layer_id, top_u_obs, emulator)
        initial_losses.append(loss)
    
    # Seleziona gli indici dei 'top_k' punti di partenza con l'errore minore
    best_start_indices = np.argsort(initial_losses)[:top_k]
    
    # 4. Ottimizzazione vera e propria (Gradient-based con L-BFGS-B) a partire dai punti migliori
    for idx in best_start_indices:
        x0 = initial_guesses[idx]
        
        result = minimize(
            fun=compute_loss,
            x0=x0,
            args=(layer_id, top_u_obs, emulator),
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-9, 'maxiter': 100} 
        )
        
        # Aggiorna il minimo globale se l'ottimizzazione corrente ha trovato un valore migliore
        if result.success and result.fun < best_loss:
            best_loss = result.fun
            best_params = result.x
            
    # Fallback nel caso in cui scipy.optimize fallisca per tutti i tentativi (raro con L-BFGS-B)
    if best_params is None:
        best_idx = best_start_indices[0]
        best_params = initial_guesses[best_idx]
        best_loss = initial_losses[best_idx]
        print(f"Warning: Ottimizzazione fallita per il layer {layer_id}. Restituisco il miglior guess iniziale.")

    return best_loss, best_params